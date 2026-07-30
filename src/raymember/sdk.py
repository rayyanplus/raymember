"""Primary developer-facing Raymember SDK class."""

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Union
from raymember.config import RaymemberConfig
from raymember.engine.state_update import StateUpdateEngine
from raymember.migrations.runner import MigrationRunner
from raymember.policy.auto import AutoMemoryPolicy
from raymember.retrieval.query import DeterministicQueryEngine
from raymember.retrieval.ranking import ContextResultDiagnostics, RankedContextRetriever
from raymember.retrieval.language import ObservationKind, classify_observation
from raymember.schemas import (
    ContextResult,
    Location,
    ObservationInput,
    ObservationRecord,
    QueryResult,
)
from raymember.resolution import (
    EntityResolver,
    LocationResolutionResult,
    LocationResolver,
)
from raymember.storage.database import DatabaseManager
from raymember.storage.models import (
    CurrentStateModel,
    CurrentStateRepository,
    EntityRepository,
    LocationAliasRepository,
    ObservationRepository,
)
from raymember.validation.writes import MemoryWriteValidator


@dataclass
class EntityStateResult:
    """Rich structured response for entity memory lookups."""

    entity_id: str
    entity_label: str
    current_location: Dict[str, Any]
    confidence: float
    previous_location: Optional[Dict[str, Any]]
    last_seen: str
    state: str
    evidence: List[str] = field(default_factory=list)
    explanation: str = ""
    uncertainty_status: str = "CONFIRMED"
    provenance: str = "sensor"
    namespace: str = "default"
    alternative_locations: List[Dict[str, Any]] = field(default_factory=list)

    # --- Generalized state fields ---
    current_attributes: Dict[str, Any] = field(default_factory=dict)
    attribute_beliefs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accepted_transitions: List[Dict[str, Any]] = field(default_factory=list)

    # --- Conflict metadata (all default to empty / False for backward compat) ---
    has_conflict: bool = False
    conflicting_observations: List[Dict[str, Any]] = field(default_factory=list)
    """Observations that were stored but did NOT replace the current belief.
    Each entry: {room, confidence, provenance, timestamp, reason, observation_id}."""
    accepted_observation_ids: List[str] = field(default_factory=list)
    """Observation IDs that contributed to the current accepted state."""
    rejected_observation_ids: List[str] = field(default_factory=list)
    """Observation IDs classified as conflicting (not necessarily write-blocked)."""
    conflict_summary: str = ""
    """Human-readable single-sentence summary of any detected conflict."""
    interpreted_history: List[Dict[str, Any]] = field(default_factory=list)
    """Derived evidence classification layer. Each entry annotates a raw observation
    with its ObservationKind: ACCEPTED_CURRENT, ACCEPTED_TRANSITION,
    REOBSERVATION, CONFLICTING, or UNCERTAIN."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_label": self.entity_label,
            "current_location": self.current_location,
            "current_attributes": self.current_attributes,
            "attribute_beliefs": self.attribute_beliefs,
            "accepted_transitions": self.accepted_transitions,
            "confidence": self.confidence,
            "previous_location": self.previous_location,
            "last_seen": self.last_seen,
            "state": self.state,
            "evidence": self.evidence,
            "explanation": self.explanation,
            "uncertainty_status": self.uncertainty_status,
            "provenance": self.provenance,
            "namespace": self.namespace,
            "alternative_locations": self.alternative_locations,
            # conflict fields
            "has_conflict": self.has_conflict,
            "conflicting_observations": self.conflicting_observations,
            "accepted_observation_ids": self.accepted_observation_ids,
            "rejected_observation_ids": self.rejected_observation_ids,
            "conflict_summary": self.conflict_summary,
            "interpreted_history": self.interpreted_history,
        }


class Raymember:
    """
    Plug-and-play persistent world-memory layer for AI agents.
    Maintains append-only observation logs, probabilistic belief distributions,
    namespaces, write safety validation, and automatic update routing.
    """

    def __init__(
        self,
        database_path: str = "raymember.db",
        policy: str = "auto",
        namespace: str = "default",
        source_reliability: Optional[Dict[str, float]] = None,
    ):
        self.database_path = database_path
        self.policy_name = policy
        self.namespace = namespace or "default"
        self.source_reliability = source_reliability

        # Run database schema version check / migration
        migration_runner = MigrationRunner(database_path)
        migration_runner.check_and_migrate()

        self.config = RaymemberConfig(database_path=database_path, update_policy=policy)
        self.db = DatabaseManager(database_path=self.config.database_path)
        self.auto_policy = AutoMemoryPolicy(random_seed=self.config.random_seed)
        self.write_validator = MemoryWriteValidator(source_reliability=source_reliability)

        self.location_resolver = LocationResolver(
            enable_builtin_aliases=self.config.enable_builtin_aliases,
            custom_aliases=self.config.custom_location_aliases,
            fuzzy_accept_threshold=self.config.fuzzy_accept_threshold,
            fuzzy_confirm_threshold=self.config.fuzzy_confirm_threshold,
        )
        self.entity_resolver = EntityResolver()

        # Load persisted location aliases from database
        self._load_persisted_aliases()

    def _load_persisted_aliases(self) -> None:
        """Loads user-confirmed and rejected aliases from SQLite storage for active namespace."""
        with self.db.session_scope() as session:
            alias_repo = LocationAliasRepository(session, namespace=self.namespace)
            stored = alias_repo.get_aliases(namespace=self.namespace)
            for item in stored:
                if item.status == "CONFIRMED":
                    self.location_resolver.confirm_alias(item.raw_alias, item.canonical_location)
                elif item.status == "REJECTED":
                    self.location_resolver.reject_mapping(item.raw_alias, item.canonical_location)

    def list_namespaces(self) -> List[str]:
        """Returns distinct namespaces present in database."""
        with self.db.session_scope() as session:
            e_repo = EntityRepository(session, namespace=self.namespace)
            return e_repo.list_namespaces()

    def create_namespace(self, namespace_name: str) -> str:
        """Creates or switches active namespace."""
        self.namespace = namespace_name.strip()
        self.location_resolver = LocationResolver(
            enable_builtin_aliases=self.config.enable_builtin_aliases,
            custom_aliases=self.config.custom_location_aliases,
            fuzzy_accept_threshold=self.config.fuzzy_accept_threshold,
            fuzzy_confirm_threshold=self.config.fuzzy_confirm_threshold,
        )
        self._load_persisted_aliases()
        return self.namespace

    def use_namespace(self, namespace_name: str) -> str:
        """Switches active namespace."""
        self.namespace = namespace_name.strip()
        self.location_resolver = LocationResolver(
            enable_builtin_aliases=self.config.enable_builtin_aliases,
            custom_aliases=self.config.custom_location_aliases,
            fuzzy_accept_threshold=self.config.fuzzy_accept_threshold,
            fuzzy_confirm_threshold=self.config.fuzzy_confirm_threshold,
        )
        self._load_persisted_aliases()
        return self.namespace

    def observe(
        self,
        entity: str,
        location: Optional[Union[Location, Dict[str, Any]]] = None,
        state: Optional[Dict[str, Any]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        source: str = "user",
        provenance: str = "sensor",
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> ObservationRecord:
        """Submits a new observation into permanent evidence store with provenance and namespace isolation."""
        # 1. Resolve entity name
        ent_res = self.entity_resolver.resolve(entity)

        state_dict: Dict[str, Any] = {}
        if state and isinstance(state, dict):
            state_dict.update(state)

        # Handle positional or keyword location vs state dictionary
        loc_arg = location
        if loc_arg is not None:
            if isinstance(loc_arg, dict) and "room" not in loc_arg and not state_dict:
                state_dict.update(loc_arg)
                loc_arg = None

        if loc_arg is not None:
            if isinstance(loc_arg, Location):
                raw_room = loc_arg.room
                loc_obj = loc_arg
            elif isinstance(loc_arg, dict):
                raw_room = str(loc_arg.get("room", "unknown"))
                loc_obj = Location(**loc_arg) if "room" in loc_arg else Location(room="unknown")
            else:
                raw_room = str(loc_arg)
                loc_obj = Location(room=raw_room)
        else:
            if "room" in state_dict:
                raw_room = str(state_dict["room"])
                loc_obj = Location(room=raw_room)
            elif "location" in state_dict and isinstance(state_dict["location"], dict):
                loc_d = state_dict["location"]
                raw_room = str(loc_d.get("room", "unknown"))
                loc_obj = Location(**loc_d) if "room" in loc_d else Location(room="unknown")
            else:
                raw_room = "unknown"
                loc_obj = Location(room="unknown")

        # Get existing known rooms for candidate resolution
        known_rooms: List[str] = []
        with self.db.session_scope() as session:
            rows = session.query(CurrentStateModel.room).filter_by(namespace=self.namespace).distinct().all()
            known_rooms = [r[0] for r in rows if r[0]]

        loc_res = self.location_resolver.resolve(raw_room, known_locations=known_rooms)
        loc_obj.room = loc_res.canonical_location

        if attributes:
            for k, val in attributes.items():
                if k not in state_dict:
                    state_dict[k] = val

        obs_input = ObservationInput(
            entity=ent_res.canonical_name,
            attributes=attributes or {},
            state=state_dict,
            location=loc_obj,
            confidence=confidence,
            source=source,
            provenance=provenance,
            timestamp=timestamp,
            metadata=metadata or {},
            entity_id=entity_id,
            entity_type=entity_type,
        )

        with self.db.session_scope() as session:
            cs_repo = CurrentStateRepository(session, namespace=self.namespace)
            e_repo = EntityRepository(session, namespace=self.namespace)

            existing_ent = e_repo.find_by_canonical_name(ent_res.canonical_name, namespace=self.namespace)
            curr_state = None
            if existing_ent:
                cs_model = cs_repo.get(existing_ent[0].entity_id, namespace=self.namespace)
                if cs_model:
                    curr_state = {
                        "location": cs_model.location_dict,
                        "confidence": cs_model.confidence,
                        "provenance": cs_model.provenance,
                    }

            allowed, val_reason, eff_conf = self.write_validator.validate_write(curr_state, obs_input, provenance=provenance)

            engine = StateUpdateEngine(session, self.config, namespace=self.namespace)
            result_dict = engine.process_observation(obs_input)

            if self.policy_name == "auto":
                curr_model = result_dict.get("current_state")
                curr_dict = {
                    "location": curr_model.location_dict,
                    "last_seen": curr_model.last_seen,
                    "confidence": curr_model.confidence,
                    "belief_data": curr_model.belief_data,
                    "provenance": curr_model.provenance,
                } if curr_model else None

                act, _, exp = self.auto_policy.predict_action_with_explanation(curr_dict, obs_input)
                if curr_model:
                    curr_model.provenance = str(provenance).lower()
                    curr_model.namespace = self.namespace
                    raw_data = curr_model.belief_data or {}
                    raw_data["update_explanation"] = exp
                    curr_model.belief_json = json.dumps(raw_data)
                    session.flush()

            raw_obs = result_dict["observation"]
            raw_obs.provenance = str(provenance).lower()
            raw_obs.namespace = self.namespace
            raw_obs.raw_location = loc_res.raw_location
            raw_obs.normalized_location = loc_res.normalized_location
            raw_obs.canonical_location = loc_res.canonical_location
            raw_obs.resolution_method = loc_res.resolution_method
            raw_obs.resolution_confidence = loc_res.resolution_confidence
            raw_obs.resolution_confirmed = not loc_res.requires_confirmation
            self._db_version = getattr(self, "_db_version", 0) + 1

            return ObservationRecord(
                observation_id=raw_obs.observation_id,
                entity_id=raw_obs.entity_id,
                entity_label=raw_obs.entity_label,
                attributes=raw_obs.attributes,
                location=raw_obs.location,
                room=raw_obs.room,
                x=raw_obs.x,
                y=raw_obs.y,
                z=raw_obs.z,
                timestamp=raw_obs.timestamp,
                confidence=raw_obs.confidence,
                source=raw_obs.source,
                metadata_dict=raw_obs.metadata_dict,
                raw_location=raw_obs.raw_location,
                normalized_location=raw_obs.normalized_location,
                canonical_location=raw_obs.canonical_location,
                resolution_method=raw_obs.resolution_method,
                resolution_confidence=raw_obs.resolution_confidence,
                resolution_confirmed=raw_obs.resolution_confirmed,
            )

    def get(self, entity_id_or_name: str) -> Optional[EntityStateResult]:
        """Retrieves current memory belief, confidence, history, and evidence explanation."""
        with self.db.session_scope() as session:
            e_repo = EntityRepository(session, namespace=self.namespace)
            cs_repo = CurrentStateRepository(session, namespace=self.namespace)
            o_repo = ObservationRepository(session, namespace=self.namespace)

            ent = e_repo.get_by_id(entity_id_or_name, namespace=self.namespace)
            if not ent:
                found = e_repo.find_by_canonical_name(entity_id_or_name, namespace=self.namespace)
                ent = found[0] if found else None
            if not ent:
                return None

            cs = cs_repo.get(ent.entity_id, namespace=self.namespace)
            if not cs:
                return None

            # Full observation history (all stored, append-only — never modified)
            all_obs = o_repo.get_by_entity(ent.entity_id, namespace=self.namespace)
            recent_obs = all_obs[:5]
            evidence_ids = [o.observation_id for o in recent_obs]

            # Accepted state transitions (ground truth for genuine movements)
            transitions = cs_repo.get_transitions(ent.entity_id, namespace=self.namespace)
            accepted_transition_rooms: set = set()
            for t in transitions:
                new_r = t.new_location.get("room", "")
                old_r = t.old_location.get("room", "")
                if new_r:
                    accepted_transition_rooms.add(new_r)
                if old_r:
                    accepted_transition_rooms.add(old_r)
            accepted_transition_rooms.add(cs.room)  # current room is always accepted

            # Previous confirmed location (most recent transition's old location)
            prev_loc = transitions[0].old_location if transitions else None

            exp = (cs.belief_data or {}).get("update_explanation", f"Belief based on {len(evidence_ids)} observations.")

            unc_status = "CONFIRMED"
            if cs.confidence < 0.6:
                unc_status = "UNCERTAIN"
            elif cs.status == "MOVED":
                unc_status = "MOVED"

            b_data = cs.belief_data or {}
            alts = b_data.get("alternative_locations", [])

            # ── Classify every stored observation ──────────────────────────────
            conflicting_obs: List[Dict[str, Any]] = []
            accepted_ids: List[str] = []
            rejected_ids: List[str] = []
            interpreted: List[Dict[str, Any]] = []

            # The observation that established the current state is the accepted one
            current_obs_id = cs.last_observation_id

            for obs in all_obs:
                kind = classify_observation(
                    obs_room=obs.room,
                    obs_confidence=obs.confidence,
                    current_room=cs.room,
                    accepted_transition_rooms=accepted_transition_rooms,
                )

                # The obs matching last_observation_id is always ACCEPTED_CURRENT
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

            has_conflict = bool(conflicting_obs)
            conflict_summary = ""
            if has_conflict:
                c_rooms = list({c["room"] for c in conflicting_obs})
                conflict_summary = (
                    f"{len(conflicting_obs)} conflicting observation(s) reported "
                    f"{', '.join(c_rooms)} but did not replace current belief "
                    f"of '{cs.room}'."
                )

            acc_trans_list = [
                {
                    "attribute_name": getattr(t, "attribute_name", "location"),
                    "old_value": t.old_value if hasattr(t, "old_value") else t.old_location,
                    "new_value": t.new_value if hasattr(t, "new_value") else t.new_location,
                    "timestamp": t.new_timestamp,
                    "transition_type": t.transition_type,
                }
                for t in transitions
            ]

            return EntityStateResult(
                entity_id=ent.entity_id,
                entity_label=ent.canonical_name,
                current_location=cs.location_dict,
                current_attributes=cs.attributes,
                attribute_beliefs=cs.attribute_beliefs,
                accepted_transitions=acc_trans_list,
                confidence=cs.confidence,
                previous_location=prev_loc,
                last_seen=cs.last_seen,
                state=cs.status,
                evidence=evidence_ids,
                explanation=exp,
                uncertainty_status=unc_status,
                provenance=getattr(cs, "provenance", "sensor"),
                namespace=getattr(cs, "namespace", self.namespace),
                alternative_locations=alts,
                # conflict fields
                has_conflict=has_conflict,
                conflicting_observations=conflicting_obs,
                accepted_observation_ids=accepted_ids,
                rejected_observation_ids=rejected_ids,
                conflict_summary=conflict_summary,
                interpreted_history=interpreted,
            )

    def ask(self, query_text: str) -> QueryResult:
        """Answers natural language questions about entity locations and history."""
        with self.db.session_scope() as session:
            q_engine = DeterministicQueryEngine(session)
            return q_engine.execute_query(query_text)

    def history(self, entity_id_or_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves append-only observation trajectory."""
        with self.db.session_scope() as session:
            e_repo = EntityRepository(session, namespace=self.namespace)
            o_repo = ObservationRepository(session, namespace=self.namespace)

            ent = e_repo.get_by_id(entity_id_or_name, namespace=self.namespace)
            if not ent:
                found = e_repo.find_by_canonical_name(entity_id_or_name, namespace=self.namespace)
                ent = found[0] if found else None
            if not ent:
                return []

            records = o_repo.get_by_entity(ent.entity_id, limit=limit, namespace=self.namespace)
            return [
                {
                    "observation_id": r.observation_id,
                    "entity_id": r.entity_id,
                    "entity_label": r.entity_label,
                    "room": r.room,
                    "location": r.location,
                    "timestamp": r.timestamp,
                    "confidence": r.confidence,
                    "source": r.source,
                    "provenance": getattr(r, "provenance", "sensor"),
                }
                for r in records
            ]

    def changes(self, entity_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves historical state transitions and movement log."""
        with self.db.session_scope() as session:
            cs_repo = CurrentStateRepository(session, namespace=self.namespace)
            transitions = cs_repo.get_transitions(entity_id=entity_id, limit=limit, namespace=self.namespace)
            return [
                {
                    "transition_id": t.transition_id,
                    "entity_id": t.entity_id,
                    "old_location": t.old_location,
                    "new_location": t.new_location,
                    "old_timestamp": t.old_timestamp,
                    "new_timestamp": t.new_timestamp,
                    "transition_type": t.transition_type,
                    "confidence": t.confidence,
                    "provenance": getattr(t, "provenance", "sensor"),
                }
                for t in transitions
            ]

    def context_result(self, query: str, max_items: int = 10, max_characters: int = 4000, mode: str = "standard") -> ContextResultDiagnostics:
        """Returns relevance-ranked evidence retrieval diagnostics object."""
        if not hasattr(self, "_context_cache"):
            self._context_cache = {}
            self._db_version = 0

        cache_key = (query, max_items, max_characters, mode, self.namespace, self._db_version)
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        with self.db.session_scope() as session:
            e_repo = EntityRepository(session, namespace=self.namespace)
            cs_repo = CurrentStateRepository(session, namespace=self.namespace)
            o_repo = ObservationRepository(session, namespace=self.namespace)

            all_entities = e_repo.get_all(namespace=self.namespace)
            if not all_entities:
                return RankedContextRetriever.generate_ranked_context(
                    query=query, candidate_items=[], max_items=max_items, max_characters=max_characters, mode=mode
                )

            # Pre-filter entities if query mentions specific entity label(s)
            q_lower = query.lower()
            matching_ents = []
            for ent in all_entities:
                c_name = ent.canonical_name.lower()
                c_name_underscore = c_name.replace(" ", "_")
                if c_name in q_lower or c_name_underscore in q_lower:
                    matching_ents.append(ent)

            target_entities = matching_ents if matching_ents else all_entities
            ent_ids = [e.entity_id for e in target_entities]

            # Batch fetch current states & observations in 2 SQL queries (eliminates N+1 query loop)
            cs_map = cs_repo.get_batch(ent_ids, namespace=self.namespace)
            obs_list = o_repo.get_all_for_entities(ent_ids, namespace=self.namespace)

            candidates: List[Dict[str, Any]] = []

            for ent in target_entities:
                cs = cs_map.get(ent.entity_id)
                if cs:
                    candidates.append({
                        "entity_label": ent.canonical_name,
                        "location": cs.location_dict,
                        "room": cs.room,
                        "attributes": cs.attributes,
                        "attribute_beliefs": cs.attribute_beliefs,
                        "confidence": cs.confidence,
                        "source": "memory",
                        "provenance": getattr(cs, "provenance", "sensor"),
                        "timestamp": cs.last_seen,
                        "is_current_state": True,
                    })

            # Limit 5 observations per entity
            obs_per_entity: Dict[str, int] = {}
            for o in obs_list:
                cnt = obs_per_entity.get(o.entity_id, 0)
                if cnt < 5:
                    obs_per_entity[o.entity_id] = cnt + 1
                    candidates.append({
                        "entity_label": o.entity_label,
                        "location": o.location,
                        "room": o.room,
                        "attributes": o.state or o.attributes,
                        "confidence": o.confidence,
                        "source": o.source,
                        "provenance": getattr(o, "provenance", "sensor"),
                        "timestamp": o.timestamp,
                        "is_current_state": False,
                    })

            diag = RankedContextRetriever.generate_ranked_context(
                query=query,
                candidate_items=candidates,
                max_items=max_items,
                max_characters=max_characters,
                mode=mode,
            )
            self._context_cache[cache_key] = diag
            return diag

    def context(self, query: str, max_items: int = 10, max_characters: int = 4000, mode: str = "standard") -> str:
        """Returns evidence-aware ranked context summary formatted for external LLMs."""
        res = self.context_result(query=query, max_items=max_items, max_characters=max_characters, mode=mode)
        return res.formatted_context

    def context_json(self, query: str, max_items: int = 10, max_characters: int = 4000, mode: str = "standard") -> Dict[str, Any]:
        """Returns structured JSON ranked context payload."""
        res = self.context_result(query=query, max_items=max_items, max_characters=max_characters, mode=mode)
        return {
            "query": query,
            "mode": mode,
            "selected_items": res.selected_items,
            "relevance_scores": res.relevance_scores,
            "truncated": res.truncated,
            "formatted_context": res.formatted_context,
        }

    def forget(self, entity_id_or_name: str) -> bool:
        """Marks an entity and its state inactive (soft delete)."""
        with self.db.session_scope() as session:
            e_repo = EntityRepository(session, namespace=self.namespace)
            ent = e_repo.get_by_id(entity_id_or_name, namespace=self.namespace)
            if not ent:
                found = e_repo.find_by_canonical_name(entity_id_or_name, namespace=self.namespace)
                ent = found[0] if found else None
            if ent:
                session.delete(ent)
                return True
            return False

    def resolve_location(self, raw_location: str, namespace: Optional[str] = None) -> LocationResolutionResult:
        """
        Resolves a raw location string using active aliases and fuzzy matching against known locations.
        """
        ns = namespace or self.namespace
        with self.db.session_scope() as session:
            rows = session.query(CurrentStateModel.room).filter_by(namespace=ns).distinct().all()
            known_rooms = [r[0] for r in rows if r[0]]

        return self.location_resolver.resolve(raw_location, known_locations=known_rooms)

    def confirm_location_alias(self, alias: str, canonical: str, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Persists a user-confirmed alias mapping (alias -> canonical) into local SQLite and updates active resolver.
        """
        ns = namespace or self.namespace
        self.location_resolver.confirm_alias(alias, canonical)
        with self.db.session_scope() as session:
            alias_repo = LocationAliasRepository(session, namespace=ns)
            obj = alias_repo.save_alias(alias, canonical, status="CONFIRMED", provenance="user_confirmed", namespace=ns)
            return {
                "status": "CONFIRMED",
                "alias": obj.raw_alias,
                "canonical": obj.canonical_location,
                "provenance": obj.provenance,
                "namespace": obj.namespace,
            }

    def reject_location_resolution(self, raw_location: str, canonical: Optional[str] = None, namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Persists a user rejection for a fuzzy location mapping to prevent future auto-mapping.
        """
        ns = namespace or self.namespace
        self.location_resolver.reject_mapping(raw_location, canonical)
        with self.db.session_scope() as session:
            alias_repo = LocationAliasRepository(session, namespace=ns)
            obj = alias_repo.save_alias(raw_location, canonical or "rejected", status="REJECTED", provenance="user_rejected", namespace=ns)
            return {
                "status": "REJECTED",
                "raw_location": obj.raw_alias,
                "rejected_canonical": obj.canonical_location,
                "namespace": obj.namespace,
            }

    def get_active_location_aliases(self, namespace: Optional[str] = None) -> Dict[str, str]:
        """Returns all active location alias mappings (alias -> canonical)."""
        return self.location_resolver.get_active_aliases()

    def add_location_alias(self, alias: str, canonical: str, namespace: Optional[str] = None) -> None:
        """Adds a custom location alias."""
        self.location_resolver.add_alias(alias, canonical)

    def remove_location_alias(self, alias: str, namespace: Optional[str] = None) -> None:
        """Removes a location alias."""
        self.location_resolver.remove_alias(alias)

    def close(self) -> None:
        """Closes database connection."""
        self.db.close()

    def __enter__(self) -> "Raymember":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
