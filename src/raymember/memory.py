"""Primary public SDK entry point for WorldMemory."""

from typing import Any, Dict, List, Optional, Union
from sqlalchemy.orm import Session

from raymember.config import RaymemberConfig
from raymember.engine.state_update import StateUpdateEngine
from raymember.learning.dataset import DatasetGenerator
from raymember.learning.policy import LearnedUpdatePolicy
from raymember.retrieval.query import DeterministicQueryEngine
from raymember.schemas import (
    ContextResult,
    Location,
    ObservationInput,
    ObservationRecord,
    QueryResult,
)
from raymember.storage.database import DatabaseManager
from raymember.storage.models import (
    CurrentStateRepository,
    EntityRepository,
    EntityModel,
    ObservationRepository,
)


class WorldMemory:
    """
    Model-agnostic persistent world-memory system for AI agents.
    Maintains append-only observations, probabilistic belief distributions,
    and optional learned update policy.
    """

    def __init__(
        self,
        database_path: str = "raymember.db",
        update_policy: str = "learned",
        config: Optional[RaymemberConfig] = None,
    ):
        self.config = config or RaymemberConfig(database_path=database_path, update_policy=update_policy)
        self.db = DatabaseManager(database_path=self.config.database_path)

        self.learned_policy: Optional[LearnedUpdatePolicy] = None
        if self.config.update_policy == "learned":
            self.learned_policy = LearnedUpdatePolicy(random_seed=self.config.random_seed)
            if self.config.model_path:
                self.learned_policy.load(self.config.model_path)
            else:
                ds = DatasetGenerator(random_seed=self.config.random_seed)
                (X_train, y_train, _), _, _, _ = ds.generate_split_dataset(num_scenarios=20, steps_per_scenario=10)
                self.learned_policy.train(X_train, y_train)

    def observe(
        self,
        entity: str,
        location: Union[Location, Dict[str, Any]],
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        source: str = "user",
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> ObservationRecord:
        """
        Ingests a new observation into permanent evidence store and updates current world belief.
        """
        obs_input = ObservationInput(
            entity=entity,
            attributes=attributes or {},
            location=location,
            confidence=confidence,
            source=source,
            timestamp=timestamp,
            metadata=metadata or {},
            entity_id=entity_id,
            entity_type=entity_type,
        )

        with self.db.session_scope() as session:
            engine = StateUpdateEngine(session, self.config)
            result_dict = engine.process_observation(obs_input)

            if self.learned_policy and result_dict.get("action") not in ("INITIALIZE", "PRESERVE"):
                curr_state_model = result_dict.get("current_state")
                curr_dict = {
                    "location": curr_state_model.location_dict,
                    "last_seen": curr_state_model.last_seen,
                    "confidence": curr_state_model.confidence,
                    "belief_data": curr_state_model.belief_data,
                } if curr_state_model else None

                learned_action, _ = self.learned_policy.predict_action(curr_dict, obs_input)
                if learned_action == "PRESERVE" and curr_state_model:
                    curr_state_model.status = "PRESERVE"
                    session.flush()

            raw_obs = result_dict["observation"]
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
            )

    def query(self, query_text: str) -> QueryResult:
        """
        Queries current world memory using natural language keyword matching.
        """
        with self.db.session_scope() as session:
            q_engine = DeterministicQueryEngine(session)
            return q_engine.execute_query(query_text)

    def get_entity(self, entity_id_or_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves entity metadata by ID or canonical name."""
        with self.db.session_scope() as session:
            repo = EntityRepository(session)
            ent = repo.get_by_id(entity_id_or_name)
            if not ent:
                found = repo.find_by_canonical_name(entity_id_or_name)
                ent = found[0] if found else None
            if not ent:
                return None
            return {
                "entity_id": ent.entity_id,
                "entity_type": ent.entity_type,
                "canonical_name": ent.canonical_name,
                "attributes": ent.attributes,
                "created_at": ent.created_at,
                "updated_at": ent.updated_at,
            }

    def get_history(self, entity_id_or_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves append-only observation trajectory for an entity."""
        with self.db.session_scope() as session:
            e_repo = EntityRepository(session)
            o_repo = ObservationRepository(session)

            ent = e_repo.get_by_id(entity_id_or_name)
            if not ent:
                found = e_repo.find_by_canonical_name(entity_id_or_name)
                ent = found[0] if found else None

            if not ent:
                return []

            records = o_repo.get_by_entity(ent.entity_id, limit=limit)
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
                }
                for r in records
            ]

    def get_changes(self, entity_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieves historical state transitions and movement log."""
        with self.db.session_scope() as session:
            cs_repo = CurrentStateRepository(session)
            transitions = cs_repo.get_transitions(entity_id=entity_id, limit=limit)
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
                }
                for t in transitions
            ]

    def get_context(self, query: str, max_items: int = 5) -> ContextResult:
        """
        Returns evidence-aware context summary formatted for external LLMs (GPT, Claude, Gemini).
        """
        q_res = self.query(query)
        recent_obs = self.get_history(q_res.entity, limit=max_items)
        evidence_lines = [
            f"{o['room']} observation at {o['timestamp']}, confidence {int(o['confidence']*100)}%"
            for o in recent_obs
        ]
        alt_str = None
        if q_res.alternative_locations:
            alt_item = q_res.alternative_locations[0]
            alt_loc = alt_item.get("location", {})
            alt_room = alt_loc.get("room") if isinstance(alt_loc, dict) else str(alt_loc)
            alt_p = int(alt_item.get("probability", 0.0) * 100)
            alt_str = f"{alt_room}, {alt_p}%"

        curr_loc_str = q_res.current_location.get("room") if isinstance(q_res.current_location, dict) else str(q_res.current_location)

        return ContextResult(
            summary=f"Entity {q_res.entity} is currently estimated in {curr_loc_str}.",
            entity=q_res.entity,
            current_belief=curr_loc_str or "unknown",
            belief_confidence=q_res.belief_confidence,
            alternative=alt_str,
            recent_evidence=evidence_lines,
            state=q_res.state,
        )

    def close(self) -> None:
        """Closes database connection and disposes resources."""
        self.db.close()

    def __enter__(self) -> "WorldMemory":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
