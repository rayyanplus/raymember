"""Schema-validated JSON export and import engine for Raymember."""

import json
import os
from typing import Any, Dict, Optional
from raymember.storage.database import DatabaseManager
from raymember.storage.models import (
    CurrentStateModel,
    EntityModel,
    ObservationModel,
    StateTransitionModel,
)


class ExportImportEngine:
    """Manages full database state exports and validated imports."""

    SCHEMA_VERSION = 2

    @classmethod
    def export_to_json(cls, db_path: str, output_path: Optional[str] = None, namespace: Optional[str] = None) -> Dict[str, Any]:
        db = DatabaseManager(database_path=db_path)
        with db.session_scope() as session:
            q_ent = session.query(EntityModel)
            q_obs = session.query(ObservationModel)
            q_cs = session.query(CurrentStateModel)
            q_trans = session.query(StateTransitionModel)

            if namespace:
                q_ent = q_ent.filter_by(namespace=namespace)
                q_obs = q_obs.filter_by(namespace=namespace)
                q_cs = q_cs.filter_by(namespace=namespace)
                q_trans = q_trans.filter_by(namespace=namespace)

            entities = [
                {
                    "entity_id": e.entity_id,
                    "entity_type": e.entity_type,
                    "canonical_name": e.canonical_name,
                    "attributes": e.attributes,
                    "namespace": getattr(e, "namespace", "default"),
                    "created_at": e.created_at,
                    "updated_at": e.updated_at,
                }
                for e in q_ent.all()
            ]

            observations = [
                {
                    "observation_id": o.observation_id,
                    "entity_id": o.entity_id,
                    "entity_label": o.entity_label,
                    "attributes": o.attributes,
                    "location": o.location,
                    "room": o.room,
                    "x": o.x,
                    "y": o.y,
                    "z": o.z,
                    "timestamp": o.timestamp,
                    "confidence": o.confidence,
                    "source": o.source,
                    "provenance": getattr(o, "provenance", "sensor"),
                    "namespace": getattr(o, "namespace", "default"),
                    "metadata": o.metadata_dict,
                }
                for o in q_obs.all()
            ]

            current_states = [
                {
                    "entity_id": c.entity_id,
                    "room": c.room,
                    "x": c.x,
                    "y": c.y,
                    "z": c.z,
                    "confidence": c.confidence,
                    "belief_data": c.belief_data,
                    "last_seen": c.last_seen,
                    "status": c.status,
                    "provenance": getattr(c, "provenance", "sensor"),
                    "namespace": getattr(c, "namespace", "default"),
                    "last_observation_id": c.last_observation_id,
                    "updated_at": c.updated_at,
                }
                for c in q_cs.all()
            ]

            transitions = [
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
                    "namespace": getattr(t, "namespace", "default"),
                    "observation_id": t.observation_id,
                }
                for t in q_trans.all()
            ]

            export_data = {
                "schema_version": cls.SCHEMA_VERSION,
                "exported_namespace": namespace or "all",
                "entities": entities,
                "observations": observations,
                "current_states": current_states,
                "state_transitions": transitions,
            }

            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2)

            return export_data

    @classmethod
    def import_from_json(cls, db_path: str, input_path: str) -> int:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Import file '{input_path}' not found.")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate schema format
        if not isinstance(data, dict) or "entities" not in data or "observations" not in data:
            raise ValueError("Invalid import payload: Missing required entities or observations keys.")

        db = DatabaseManager(database_path=db_path)
        db.create_tables()
        imported_count = 0

        with db.session_scope() as session:
            # 1. Import Entities
            for e_dict in data.get("entities", []):
                if not session.query(EntityModel).filter_by(entity_id=e_dict["entity_id"]).first():
                    ent = EntityModel(
                        entity_id=e_dict["entity_id"],
                        entity_type=e_dict.get("entity_type", "object"),
                        canonical_name=e_dict["canonical_name"],
                        attributes_json=json.dumps(e_dict.get("attributes", {})),
                        namespace=e_dict.get("namespace", "default"),
                        created_at=e_dict.get("created_at", ""),
                        updated_at=e_dict.get("updated_at", ""),
                    )
                    session.add(ent)
                    imported_count += 1

            # 2. Import Observations
            for o_dict in data.get("observations", []):
                if not session.query(ObservationModel).filter_by(observation_id=o_dict["observation_id"]).first():
                    obs = ObservationModel(
                        observation_id=o_dict["observation_id"],
                        entity_id=o_dict["entity_id"],
                        entity_label=o_dict.get("entity_label", ""),
                        attributes_json=json.dumps(o_dict.get("attributes", {})),
                        location_json=json.dumps(o_dict.get("location", {})),
                        room=o_dict.get("room", "unknown"),
                        x=o_dict.get("x"),
                        y=o_dict.get("y"),
                        z=o_dict.get("z"),
                        timestamp=o_dict.get("timestamp", ""),
                        confidence=o_dict.get("confidence", 1.0),
                        source=o_dict.get("source", "imported"),
                        provenance=o_dict.get("provenance", "imported"),
                        namespace=o_dict.get("namespace", "default"),
                        metadata_json=json.dumps(o_dict.get("metadata", {})),
                    )
                    session.add(obs)

            # 3. Import Current States
            for c_dict in data.get("current_states", []):
                cs = session.query(CurrentStateModel).filter_by(entity_id=c_dict["entity_id"]).first()
                if not cs:
                    cs = CurrentStateModel(
                        entity_id=c_dict["entity_id"],
                        room=c_dict.get("room", "unknown"),
                        x=c_dict.get("x"),
                        y=c_dict.get("y"),
                        z=c_dict.get("z"),
                        confidence=c_dict.get("confidence", 1.0),
                        belief_json=json.dumps(c_dict.get("belief_data", {})),
                        last_seen=c_dict.get("last_seen", ""),
                        status=c_dict.get("status", "OBSERVED"),
                        provenance=c_dict.get("provenance", "imported"),
                        namespace=c_dict.get("namespace", "default"),
                        last_observation_id=c_dict.get("last_observation_id"),
                        updated_at=c_dict.get("updated_at", ""),
                    )
                    session.add(cs)

            session.flush()

        return imported_count
