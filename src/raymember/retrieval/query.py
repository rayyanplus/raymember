"""Deterministic Natural Language Query Parser and Retrieval Engine."""

import re
from typing import Any, Dict, List, Optional, Set
from sqlalchemy.orm import Session

from raymember.exceptions import QueryParseError
from raymember.retrieval.language import (
    ConflictAnswerGenerator,
    ObservationKind,
    classify_observation,
    location_phrase,
    entity_subject,
    continuation_pronoun,
)
from raymember.schemas import QueryResult
from raymember.storage.models import (
    CurrentStateRepository,
    EntityRepository,
    ObservationRepository,
)


class DeterministicQueryEngine:
    """Parses natural language queries and fetches structured world memory results."""

    def __init__(self, session: Session):
        self.session = session
        self.entity_repo = EntityRepository(session)
        self.obs_repo = ObservationRepository(session)
        self.current_state_repo = CurrentStateRepository(session)

    def execute_query(self, query_text: str) -> QueryResult:
        """
        Parses query text and dispatches to appropriate retrieval pattern.
        """
        q = query_text.strip().lower()

        # Pattern 1: Attribute explanation query ("Why does Raymember believe X is Y?")
        if "why" in q and ("believe" in q or "think" in q or "is" in q):
            entity_label = self._extract_entity_label(q)
            if entity_label:
                return self._query_attribute_explanation(entity_label, q)

        # Pattern 2: Room contents query ("What is in the bedroom?", "Which objects are in living_room?")
        room_match = re.search(r"(?:in|inside)\s+(?:the\s+)?([a-z0-9_\-\s]+)\??$", q, re.IGNORECASE)
        if ("what is in" in q or "which objects are in" in q) and room_match:
            room_name = room_match.group(1).strip().replace(" ", "_")
            return self._query_room_contents(room_name)

        # Pattern 3: Attribute status / owner / property query ("What is the status of delivery_4821?", "Who owns task_17?")
        if ("status" in q or "owner" in q or "who" in q or "driver" in q or "state" in q) and "where" not in q:
            entity_label = self._extract_entity_label(q)
            if entity_label:
                return self._query_attribute(entity_label, q)

        # Pattern 4: Historical / previous location query ("Where was the backpack before?", "Where was backpack previously?")
        if "before" in q or "previously" in q or ("where was" in q and "where is" not in q):
            entity_label = self._extract_entity_label(q)
            if entity_label:
                return self._query_historical_location(entity_label)

        # Pattern 5: Movement / change query ("Did the backpack move?", "What changed recently?")
        if "move" in q or "changed" in q:
            entity_label = self._extract_entity_label(q)
            if entity_label:
                return self._query_movement(entity_label)

        # Pattern 6: Default current location / state query ("Where is the backpack?", "Where are the car keys?")
        if "where is" in q or "where are" in q or "where" in q or "location" in q or "position" in q:
            entity_label = self._extract_entity_label(q)
            if entity_label:
                return self._query_current_location(entity_label)

        # Fallback entity match
        entity_label = self._extract_entity_label(q)
        if entity_label:
            return self._query_current_location(entity_label)

        raise QueryParseError(f"Unable to parse query or locate matching entity for: '{query_text}'")

    def _extract_entity_label(self, q: str) -> Optional[str]:
        all_entities = self.entity_repo.get_all()
        for ent in all_entities:
            name = ent.canonical_name.lower()
            if name in q:
                return ent.canonical_name

        clean_q = re.sub(r"^(where is|where are|where was|did|when did|what is|what|who currently owns|who owns|who|why does|why|which objects are|the|a|an)\s+", "", q, flags=re.IGNORECASE)
        clean_q = re.sub(r"\s+(last seen|before|previously|move|currently|located)\??$", "", clean_q, flags=re.IGNORECASE)
        clean_q = clean_q.replace("?", "").strip()

        if "'s" in clean_q:
            parts = clean_q.split("'s")
            clean_q = parts[-1].strip()

        words = [w for w in clean_q.split() if w not in ("is", "was", "are", "owns", "owns?", "believe", "think", "currently", "the", "a", "an", "delivered", "delivered?")]
        return words[-1] if words else (clean_q if clean_q else None)

    def _build_conflict_data(self, entity_id: str) -> Dict[str, Any]:
        """
        Build conflict classification data for an entity.
        Returns a dict with:
          - curr_room
          - conflicting_obs  (list)
          - accepted_transition_rooms (set)
          - confirmed_previous_room (str | None)
          - interpreted_history (list)
          - accepted_ids, rejected_ids
        """
        cs = self.current_state_repo.get(entity_id)
        if not cs:
            return {}

        all_obs = self.obs_repo.get_by_entity(entity_id)
        transitions = self.current_state_repo.get_transitions(entity_id=entity_id)

        accepted_transition_rooms: Set[str] = set()
        for t in transitions:
            new_r = t.new_location.get("room", "")
            old_r = t.old_location.get("room", "")
            if new_r:
                accepted_transition_rooms.add(new_r)
            if old_r:
                accepted_transition_rooms.add(old_r)
        accepted_transition_rooms.add(cs.room)  # current room always accepted
        confirmed_previous_room: Optional[str] = None
        if transitions:
            confirmed_previous_room = transitions[0].old_location.get("room")

        current_obs_id = cs.last_observation_id

        conflicting_obs: List[Dict[str, Any]] = []
        accepted_ids: List[str] = []
        rejected_ids: List[str] = []
        interpreted: List[Dict[str, Any]] = []

        for obs in all_obs:
            kind = classify_observation(
                obs_room=obs.room,
                obs_confidence=obs.confidence,
                current_room=cs.room,
                accepted_transition_rooms=accepted_transition_rooms,
            )
            if obs.observation_id == current_obs_id:
                kind = ObservationKind.ACCEPTED_CURRENT

            interpreted.append({
                "observation_id": obs.observation_id,
                "room": obs.room,
                "confidence": obs.confidence,
                "provenance": getattr(obs, "provenance", "sensor"),
                "timestamp": obs.timestamp,
                "source": obs.source,
                "kind": kind.value,
            })

            if kind == ObservationKind.CONFLICTING:
                rejected_ids.append(obs.observation_id)
                conflicting_obs.append({
                    "observation_id": obs.observation_id,
                    "room": obs.room,
                    "confidence": obs.confidence,
                    "provenance": getattr(obs, "provenance", "sensor"),
                    "timestamp": obs.timestamp,
                    "reason": (
                        f"Room '{obs.room}' differs from current accepted room "
                        f"'{cs.room}' and was not accepted as a state transition."
                    ),
                })
            elif kind in (ObservationKind.ACCEPTED_CURRENT, ObservationKind.ACCEPTED_TRANSITION):
                accepted_ids.append(obs.observation_id)

        return {
            "curr_room": cs.room,
            "curr_confidence": cs.confidence,
            "curr_provenance": getattr(cs, "provenance", "sensor"),
            "curr_status": cs.status,
            "conflicting_obs": conflicting_obs,
            "accepted_transition_rooms": accepted_transition_rooms,
            "confirmed_previous_room": confirmed_previous_room,
            "interpreted_history": interpreted,
            "accepted_ids": accepted_ids,
            "rejected_ids": rejected_ids,
            "all_obs": all_obs,
            "cs": cs,
        }

    def _query_current_location(self, entity_name: str) -> QueryResult:
        entities = self.entity_repo.find_by_canonical_name(entity_name)
        if not entities:
            entities = self.entity_repo.get_all()
            entities = [e for e in entities if entity_name.lower() in e.canonical_name.lower()]

        if not entities:
            return QueryResult(
                answer=f"No record of '{entity_name}' found in memory.",
                entity=entity_name,
                current_location=None,
                state="UNCERTAIN",
            )

        entity = entities[0]
        curr_state = self.current_state_repo.get(entity.entity_id)

        if not curr_state:
            return QueryResult(
                answer=f"No location information recorded for '{entity.canonical_name}'.",
                entity=entity.canonical_name,
                state="UNCERTAIN",
            )

        # Build conflict-aware data
        cdata = self._build_conflict_data(entity.entity_id)
        conflicting_obs = cdata.get("conflicting_obs", [])
        confirmed_previous_room = cdata.get("confirmed_previous_room")
        interpreted = cdata.get("interpreted_history", [])
        accepted_ids = cdata.get("accepted_ids", [])
        rejected_ids = cdata.get("rejected_ids", [])
        all_obs = cdata.get("all_obs", [])

        curr_loc = curr_state.location_dict
        has_conflict = bool(conflicting_obs)

        # Conflict-aware answer
        answer = ConflictAnswerGenerator.build(
            entity=entity.canonical_name,
            current_room=curr_state.room,
            current_confidence=curr_state.confidence,
            current_provenance=getattr(curr_state, "provenance", "sensor"),
            state_status=curr_state.status,
            confirmed_previous_room=confirmed_previous_room,
            conflicting_obs=conflicting_obs,
        )

        # Previous location for structured result (only from accepted transitions)
        transitions = self.current_state_repo.get_transitions(entity_id=entity.entity_id)
        prev_loc = transitions[0].old_location if transitions else None

        b_data = curr_state.belief_data or {}
        belief_conf = b_data.get("belief_confidence", curr_state.confidence)
        alt_locs = b_data.get("alternative_locations", [])

        explanation = f"Last observed at {curr_state.last_seen} with {int(curr_state.confidence * 100)}% confidence."

        # conflict_summary
        conflict_summary = ""
        if has_conflict:
            c_rooms = list({c["room"] for c in conflicting_obs})
            conflict_summary = (
                f"{len(conflicting_obs)} conflicting observation(s) reported "
                f"{', '.join(c_rooms)} but did not replace current belief of '{curr_state.room}'."
            )

        state_label = "CONFLICT" if has_conflict else curr_state.status

        return QueryResult(
            answer=answer,
            entity=entity.canonical_name,
            current_location=curr_loc,
            current_attributes=curr_state.attributes,
            attribute_beliefs=curr_state.attribute_beliefs,
            confidence=curr_state.confidence,
            belief_confidence=belief_conf,
            last_seen=curr_state.last_seen,
            previous_location=prev_loc,
            state=state_label,
            evidence=[o.observation_id for o in all_obs[:5]],
            history=[{
                "room": o.room,
                "timestamp": o.timestamp,
                "confidence": o.confidence,
                "provenance": getattr(o, "provenance", "sensor"),
            } for o in all_obs],
            alternative_locations=alt_locs,
            explanation=explanation,
            # conflict fields
            has_conflict=has_conflict,
            conflicting_observations=conflicting_obs,
            accepted_observation_ids=accepted_ids,
            rejected_observation_ids=rejected_ids,
            conflict_summary=conflict_summary,
            interpreted_history=interpreted,
        )

    def _query_attribute(self, entity_name: str, query_text: str) -> QueryResult:
        res = self._query_current_location(entity_name)
        entities = self.entity_repo.find_by_canonical_name(entity_name)
        if not entities:
            return res

        entity = entities[0]
        cs = self.current_state_repo.get(entity.entity_id)
        if not cs:
            return res

        attrs = cs.attributes
        attr_beliefs = cs.attribute_beliefs

        q = query_text.lower()
        target_key = None
        for key in attrs.keys():
            if key.lower() in q:
                target_key = key
                break
        if not target_key and "who" in q:
            for key in ("owner", "driver", "assigned_agent", "user"):
                if key in attrs:
                    target_key = key
                    break
        if not target_key:
            target_key = "status" if "status" in attrs else (list(attrs.keys())[0] if attrs else None)

        if target_key and target_key in attrs:
            val = attrs[target_key]
            belief_info = attr_beliefs.get(target_key, {})
            conf = float(belief_info.get("confidence", cs.confidence))
            prov = str(belief_info.get("provenance", cs.provenance))
            has_attr_conflict = belief_info.get("has_conflict", False)

            if has_attr_conflict:
                alts = belief_info.get("alternative_values", [])
                alt_str = ", ".join([f"'{a['value']}' ({int(a['confidence']*100)}% via {a['provenance']})" for a in alts])
                answer = (
                    f"The current believed '{target_key}' for {entity.canonical_name} is '{val}' "
                    f"({int(conf*100)}% confidence, source: {prov}). "
                    f"However, conflicting update(s) reported {alt_str}."
                )
            else:
                answer = (
                    f"The current '{target_key}' for {entity.canonical_name} is '{val}' "
                    f"(confidence: {int(conf*100)}%, source: {prov})."
                )
            res.answer = answer
            res.current_attributes = attrs
            res.attribute_beliefs = attr_beliefs
        return res

    def _query_attribute_explanation(self, entity_name: str, query_text: str) -> QueryResult:
        res = self._query_current_location(entity_name)
        entities = self.entity_repo.find_by_canonical_name(entity_name)
        if not entities:
            return res

        entity = entities[0]
        cs = self.current_state_repo.get(entity.entity_id)
        if not cs:
            return res

        attrs = cs.attributes
        attr_beliefs = cs.attribute_beliefs
        all_obs = self.obs_repo.get_by_entity(entity.entity_id)

        target_attr = "status"
        for k in attrs.keys():
            if k.lower() in query_text.lower():
                target_attr = k
                break

        val = attrs.get(target_attr, cs.room)
        b_info = attr_beliefs.get(target_attr, {})
        conf = float(b_info.get("confidence", cs.confidence))
        prov = str(b_info.get("provenance", cs.provenance))

        explanation_text = (
            f"Raymember believes {entity.canonical_name} has {target_attr}='{val}' because it was "
            f"supported by high-trust provenance ('{prov}') with {int(conf*100)}% confidence. "
            f"Evidence log contains {len(all_obs)} recorded observation(s)."
        )
        res.answer = explanation_text
        res.explanation = explanation_text
        res.current_attributes = attrs
        res.attribute_beliefs = attr_beliefs
        return res

    def _query_historical_location(self, entity_name: str) -> QueryResult:
        res = self._query_current_location(entity_name)
        if res.previous_location:
            prev_room = res.previous_location.get("room", "unknown").replace("_", " ")
            curr_room = res.current_location.get("room", "unknown") if res.current_location else "unknown"
            subj_past = entity_subject(res.entity, past=True)
            prev_phrase = location_phrase(prev_room)
            curr_phrase = location_phrase(curr_room)
            res.answer = f"{subj_past} previously {prev_phrase} before moving to {curr_phrase}."
        elif res.has_conflict:
            # Historical query with conflict — keep the conflict answer
            pass
        return res

    def _query_movement(self, entity_name: str) -> QueryResult:
        res = self._query_current_location(entity_name)
        curr_room = res.current_location.get("room", "unknown") if res.current_location else "unknown"
        curr_phrase = location_phrase(curr_room)

        if res.state in ("MOVED",) or res.previous_location:
            prev_room = res.previous_location.get("room", "another room") if res.previous_location else "another room"
            prev_phrase = location_phrase(prev_room)
            subj_past = entity_subject(res.entity, past=True)
            res.answer = f"Yes, {entity_subject(res.entity, past=False).lower()} confirmed to have moved. Current location is {curr_phrase}. {subj_past} previously {prev_phrase}."
        elif res.has_conflict:
            # Conflict but no confirmed move
            res.answer = (
                f"No confirmed movement recorded for the {res.entity}. "
                f"A conflicting observation reported a different location, "
                f"but the current belief remains {curr_phrase}."
            )
        else:
            subj_past = entity_subject(res.entity, past=True)
            res.answer = f"No recent movement recorded for the {res.entity}. {subj_past} last observed {curr_phrase}."
        return res

    def _query_room_contents(self, room_name: str) -> QueryResult:
        from raymember.resolution.locations import LocationResolver
        resolver = LocationResolver()
        res = resolver.resolve(room_name)
        canonical_room = res.canonical_location
        raw_room_clean = res.raw_location.replace("_", " ").strip() or room_name.replace("_", " ")

        current_states = self.current_state_repo.get_by_room(canonical_room)
        if not current_states and room_name != canonical_room:
            # Fallback to direct room query
            current_states = self.current_state_repo.get_by_room(room_name)

        if not current_states:
            return QueryResult(
                answer=f"No entities are currently estimated to be {location_phrase(raw_room_clean)}.",
                entity=room_name,
                current_location={"room": canonical_room},
                state="OBSERVED",
            )
        entity_names = []
        for cs in current_states:
            ent = self.entity_repo.get_by_id(cs.entity_id)
            if ent:
                entity_names.append(ent.canonical_name)

        names_str = ", ".join(entity_names)

        # Single entity answer formatting
        if len(entity_names) == 1:
            e_name = entity_names[0]
            subj = entity_subject(e_name, past=False)
            answer_text = f"{subj} currently believed to be {location_phrase(raw_room_clean)}."
        else:
            answer_text = f"Entities currently {location_phrase(raw_room_clean)}: {names_str}."

        return QueryResult(
            answer=answer_text,
            entity=room_name,
            current_location={"room": canonical_room},
            state="OBSERVED",
            evidence=[cs.last_observation_id for cs in current_states if cs.last_observation_id],
        )
