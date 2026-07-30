"""Deterministic State Update Engine."""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from sqlalchemy.orm import Session

from raymember.config import RaymemberConfig
from raymember.engine.confidence import ConfidenceEngine
from raymember.engine.entity_resolution import EntityResolver
from raymember.engine.movement_detection import MovementDetector
from raymember.schemas import ObservationInput
from raymember.storage.models import (
    CurrentStateModel,
    CurrentStateRepository,
    ObservationModel,
    ObservationRepository,
)


class StateUpdateEngine:
    """Orchestrates observation ingestion and current state updates."""

    def __init__(self, session: Session, config: Optional[RaymemberConfig] = None, namespace: str = "default"):
        self.session = session
        self.config = config or RaymemberConfig()
        self.namespace = namespace
        self.entity_resolver = EntityResolver(session, namespace=namespace)
        self.obs_repo = ObservationRepository(session, namespace=namespace)
        self.current_state_repo = CurrentStateRepository(session, namespace=namespace)
        self.confidence_engine = ConfidenceEngine(
            min_confidence_threshold=self.config.min_confidence_threshold,
            decay_rate=self.config.belief_decay_rate,
        )
        self.movement_detector = MovementDetector(
            movement_distance_threshold=self.config.movement_distance_threshold
        )

    def process_observation(self, obs_input: ObservationInput) -> Dict[str, Any]:
        """
        Ingest an observation, append to evidence store, resolve entity,
        evaluate movement/state update using rules, and update current state.
        """
        entity, is_new = self.entity_resolver.resolve_or_create(obs_input)

        obs_record = self.obs_repo.add(entity.entity_id, obs_input, provenance=obs_input.provenance, namespace=self.namespace)

        current_state = self.current_state_repo.get(entity.entity_id, namespace=self.namespace)

        obs_ts_str = obs_input.get_iso_timestamp()
        loc = obs_input.location
        new_loc_dict = loc.to_dict()

        # Generalized state & per-attribute belief processing
        new_state_dict = obs_input.state or obs_input.attributes or {}
        if obs_input.location and "room" in obs_input.location.to_dict():
            new_state_dict["room"] = obs_input.location.room
            new_state_dict["location"] = obs_input.location.to_dict()

        if current_state is None:
            curr_attrs: Dict[str, Any] = {}
            attr_beliefs: Dict[str, Dict[str, Any]] = {}

            for k, v in new_state_dict.items():
                curr_attrs[k] = v
                attr_beliefs[k] = {
                    "attribute_name": k,
                    "believed_value": v,
                    "confidence": obs_input.confidence,
                    "provenance": obs_record.provenance,
                    "updated_at": obs_ts_str,
                    "alternative_values": [],
                    "has_conflict": False,
                }

            updated_state = self.current_state_repo.upsert(
                entity_id=entity.entity_id,
                room=loc.room if loc else "unknown",
                x=loc.x if loc else None,
                y=loc.y if loc else None,
                z=loc.z if loc else None,
                confidence=obs_input.confidence,
                last_seen=obs_ts_str,
                status="OBSERVED",
                belief_data={
                    "location_beliefs": [{"location": new_loc_dict, "probability": obs_input.confidence}],
                    "most_likely_location": new_loc_dict,
                    "belief_confidence": obs_input.confidence,
                    "alternative_locations": [],
                },
                attributes_data=curr_attrs,
                attribute_beliefs_data=attr_beliefs,
                provenance=obs_record.provenance,
                namespace=self.namespace,
                last_observation_id=obs_record.observation_id,
            )

            # Record initial transitions for all attributes
            for k, v in new_state_dict.items():
                self.current_state_repo.record_transition(
                    entity_id=entity.entity_id,
                    old_location={},
                    new_location=new_loc_dict,
                    old_timestamp=None,
                    new_timestamp=obs_ts_str,
                    transition_type="INITIAL",
                    confidence=obs_input.confidence,
                    attribute_name=k,
                    old_value=None,
                    new_value=v,
                    provenance=obs_record.provenance,
                    namespace=self.namespace,
                    observation_id=obs_record.observation_id,
                )

            return {
                "entity": entity,
                "observation": obs_record,
                "current_state": updated_state,
                "status": "OBSERVED",
                "is_new_entity": is_new,
                "action": "INITIALIZE",
            }

        try:
            t_old = datetime.fromisoformat(current_state.last_seen.replace("Z", "+00:00"))
            t_new = datetime.fromisoformat(obs_ts_str.replace("Z", "+00:00"))
            if t_new < t_old:
                return {
                    "entity": entity,
                    "observation": obs_record,
                    "current_state": current_state,
                    "status": "OUT_OF_ORDER",
                    "is_new_entity": False,
                    "action": "PRESERVE",
                }
        except Exception:
            pass

        has_moved, _, m_reason = self.movement_detector.is_moved(current_state.location_dict, new_loc_dict)
        old_location_snapshot = dict(current_state.location_dict)
        old_timestamp_snapshot = current_state.last_seen

        # Per-attribute belief updates
        curr_attrs = dict(current_state.attributes)
        attr_beliefs = dict(current_state.attribute_beliefs)

        PROVENANCE_HIERARCHY = {"user": 3, "sensor": 2, "tracking_api": 2, "agent": 1, "unreliable_sensor": 0}
        obs_prov_rank = PROVENANCE_HIERARCHY.get(str(obs_record.provenance).lower(), 1)

        for k, v in new_state_dict.items():
            if k not in attr_beliefs:
                curr_attrs[k] = v
                attr_beliefs[k] = {
                    "attribute_name": k,
                    "believed_value": v,
                    "confidence": obs_input.confidence,
                    "provenance": obs_record.provenance,
                    "updated_at": obs_ts_str,
                    "alternative_values": [],
                    "has_conflict": False,
                }
                self.current_state_repo.record_transition(
                    entity_id=entity.entity_id,
                    old_location=old_location_snapshot,
                    new_location=new_loc_dict,
                    old_timestamp=old_timestamp_snapshot,
                    new_timestamp=obs_ts_str,
                    transition_type="INITIAL",
                    confidence=obs_input.confidence,
                    attribute_name=k,
                    old_value=None,
                    new_value=v,
                    provenance=obs_record.provenance,
                    namespace=self.namespace,
                    observation_id=obs_record.observation_id,
                )
            else:
                existing_belief = attr_beliefs[k]
                existing_val = existing_belief.get("believed_value")
                existing_conf = float(existing_belief.get("confidence", 0.5))
                existing_prov_rank = PROVENANCE_HIERARCHY.get(str(existing_belief.get("provenance", "sensor")).lower(), 1)

                if v != existing_val:
                    # Evaluate attribute update vs conflict
                    should_accept = (obs_prov_rank > existing_prov_rank) or (
                        obs_prov_rank == existing_prov_rank and obs_input.confidence >= existing_conf
                    )

                    if should_accept:
                        # Record accepted attribute transition
                        old_v = existing_val
                        curr_attrs[k] = v
                        attr_beliefs[k] = {
                            "attribute_name": k,
                            "believed_value": v,
                            "confidence": obs_input.confidence,
                            "provenance": obs_record.provenance,
                            "updated_at": obs_ts_str,
                            "alternative_values": existing_belief.get("alternative_values", []),
                            "has_conflict": False,
                        }
                        self.current_state_repo.record_transition(
                            entity_id=entity.entity_id,
                            old_location=old_location_snapshot,
                            new_location=new_loc_dict,
                            old_timestamp=old_timestamp_snapshot,
                            new_timestamp=obs_ts_str,
                            transition_type="TRANSITION",
                            confidence=obs_input.confidence,
                            attribute_name=k,
                            old_value=old_v,
                            new_value=v,
                            provenance=obs_record.provenance,
                            namespace=self.namespace,
                            observation_id=obs_record.observation_id,
                        )
                    else:
                        # Reject or record as conflicting attribute belief
                        existing_belief["has_conflict"] = True
                        alt = existing_belief.get("alternative_values", [])
                        alt.append({
                            "value": v,
                            "confidence": obs_input.confidence,
                            "provenance": obs_record.provenance,
                            "timestamp": obs_ts_str,
                            "observation_id": obs_record.observation_id,
                        })
                        existing_belief["alternative_values"] = alt
                        attr_beliefs[k] = existing_belief
                else:
                    # Reobservation of same attribute value
                    existing_belief["confidence"] = max(existing_conf, obs_input.confidence)
                    existing_belief["updated_at"] = obs_ts_str
                    attr_beliefs[k] = existing_belief

        # Preserve existing memory if conflicting room observation has low confidence
        if has_moved and obs_input.confidence < 0.35 and current_state.confidence > obs_input.confidence:
            updated_state = self.current_state_repo.upsert(
                entity_id=entity.entity_id,
                room=current_state.room,
                x=current_state.x,
                y=current_state.y,
                z=current_state.z,
                confidence=current_state.confidence,
                last_seen=current_state.last_seen,
                status="PRESERVED",
                belief_data=current_state.belief_data,
                attributes_data=curr_attrs,
                attribute_beliefs_data=attr_beliefs,
                provenance=current_state.provenance,
                namespace=self.namespace,
                last_observation_id=current_state.last_observation_id,
            )
            return {
                "entity": entity,
                "observation": obs_record,
                "current_state": updated_state,
                "status": "PRESERVED",
                "is_new_entity": False,
                "action": "PRESERVE",
            }

        decayed_conf = self.confidence_engine.calculate_decayed_confidence(
            base_confidence=current_state.confidence,
            last_seen_iso=current_state.last_seen,
            current_time_iso=obs_ts_str,
        )

        fused_conf = max(decayed_conf, obs_input.confidence) if not has_moved else obs_input.confidence
        b_data = {
            "location_beliefs": [{"location": new_loc_dict, "probability": fused_conf}],
            "most_likely_location": new_loc_dict,
            "belief_confidence": fused_conf,
            "alternative_locations": [],
        }

        room_belief = attr_beliefs.get("room", {})
        believed_room = room_belief.get("believed_value", current_state.room)
        accepted_room_change = has_moved and (believed_room != current_state.room)

        status_str = "MOVED" if accepted_room_change else "REOBSERVED"
        active_room = believed_room if believed_room != "unknown" else current_state.room
        active_prov = room_belief.get("provenance", obs_record.provenance)
        active_conf = float(room_belief.get("confidence", obs_input.confidence))

        updated_state = self.current_state_repo.upsert(
            entity_id=entity.entity_id,
            room=active_room,
            x=loc.x if (loc and accepted_room_change) else current_state.x,
            y=loc.y if (loc and accepted_room_change) else current_state.y,
            z=loc.z if (loc and accepted_room_change) else current_state.z,
            confidence=active_conf,
            last_seen=obs_ts_str,
            status=status_str,
            belief_data=b_data,
            attributes_data=curr_attrs,
            attribute_beliefs_data=attr_beliefs,
            provenance=active_prov,
            namespace=self.namespace,
            last_observation_id=obs_record.observation_id if accepted_room_change else current_state.last_observation_id,
        )

        if accepted_room_change:
            self.current_state_repo.record_transition(
                entity_id=entity.entity_id,
                old_location=old_location_snapshot,
                new_location=new_loc_dict,
                old_timestamp=old_timestamp_snapshot,
                new_timestamp=obs_ts_str,
                transition_type="MOVEMENT",
                confidence=obs_input.confidence,
                attribute_name="location",
                old_value=old_location_snapshot,
                new_value=new_loc_dict,
                provenance=obs_record.provenance,
                namespace=self.namespace,
                observation_id=obs_record.observation_id,
            )

        return {
            "entity": entity,
            "observation": obs_record,
            "current_state": updated_state,
            "status": status_str,
            "is_new_entity": False,
            "action": "UPDATE" if has_moved else "REOBSERVE",
        }
