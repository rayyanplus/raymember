"""SQLAlchemy ORM models and repositories for Entities, Observations, Current State, and Transitions."""

from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import Float, String, Text
from sqlalchemy.orm import Mapped, mapped_column, Session

from raymember.schemas import ObservationInput
from raymember.storage.database import Base


# ----------------------------------------------------------------------
# 1. Entity ORM Model & Repository
# ----------------------------------------------------------------------

class EntityModel(Base):
    """Stores entity identities and attributes."""

    __tablename__ = "entities"

    entity_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(64), index=True)
    canonical_name: Mapped[str] = mapped_column(String(128), index=True)
    attributes_json: Mapped[str] = mapped_column(Text, default="{}")
    namespace: Mapped[str] = mapped_column(String(64), default="default", index=True)
    created_at: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[str] = mapped_column(String(64))

    @property
    def attributes(self) -> Dict[str, Any]:
        try:
            return json.loads(self.attributes_json)
        except Exception:
            return {}


class EntityRepository:
    """Repository for querying and creating Entity records."""

    def __init__(self, session: Session, namespace: str = "default"):
        self.session = session
        self.namespace = namespace

    def create(
        self,
        canonical_name: str,
        entity_type: str = "object",
        attributes: Optional[Dict[str, Any]] = None,
        entity_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> EntityModel:
        now_iso = datetime.now(timezone.utc).isoformat()
        eid = entity_id or f"entity_{uuid.uuid4().hex[:10]}"
        ns = namespace or self.namespace
        entity = EntityModel(
            entity_id=eid,
            entity_type=entity_type or "object",
            canonical_name=canonical_name,
            attributes_json=json.dumps(attributes or {}),
            namespace=ns,
            created_at=now_iso,
            updated_at=now_iso,
        )
        self.session.add(entity)
        self.session.flush()
        return entity

    def get_by_id(self, entity_id: str, namespace: Optional[str] = None) -> Optional[EntityModel]:
        ns = namespace or self.namespace
        return self.session.query(EntityModel).filter_by(entity_id=entity_id, namespace=ns).first()

    def find_by_canonical_name(self, name: str, namespace: Optional[str] = None) -> List[EntityModel]:
        ns = namespace or self.namespace
        return self.session.query(EntityModel).filter(EntityModel.canonical_name.ilike(name), EntityModel.namespace == ns).all()

    def find_by_type(self, entity_type: str, namespace: Optional[str] = None) -> List[EntityModel]:
        ns = namespace or self.namespace
        return self.session.query(EntityModel).filter(EntityModel.entity_type.ilike(entity_type), EntityModel.namespace == ns).all()

    def get_all(self, namespace: Optional[str] = None) -> List[EntityModel]:
        ns = namespace or self.namespace
        return self.session.query(EntityModel).filter_by(namespace=ns).all()

    def list_namespaces(self) -> List[str]:
        rows = self.session.query(EntityModel.namespace).distinct().all()
        ns_set = {r[0] for r in rows if r[0]}
        ns_set.add("default")
        return sorted(list(ns_set))

    def update_attributes(self, entity_id: str, new_attrs: Dict[str, Any]) -> Optional[EntityModel]:
        entity = self.get_by_id(entity_id)
        if entity:
            current = entity.attributes
            current.update(new_attrs)
            entity.attributes_json = json.dumps(current)
            entity.updated_at = datetime.now(timezone.utc).isoformat()
            self.session.flush()
        return entity


# ----------------------------------------------------------------------
# 2. Append-Only Observation ORM Model & Repository
# ----------------------------------------------------------------------

class ObservationModel(Base):
    """Append-only observation store (Layer 1). Never overwritten or deleted."""

    __tablename__ = "observations"

    observation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_id: Mapped[str] = mapped_column(String(64), index=True)
    entity_label: Mapped[str] = mapped_column(String(128))
    attributes_json: Mapped[str] = mapped_column(Text, default="{}")
    location_json: Mapped[str] = mapped_column(Text, default="{}")
    room: Mapped[str] = mapped_column(String(64), index=True)
    x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source: Mapped[str] = mapped_column(String(64), default="user")
    provenance: Mapped[str] = mapped_column(String(32), default="sensor")
    namespace: Mapped[str] = mapped_column(String(64), default="default", index=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")

    # Resolution metadata
    raw_location: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    normalized_location: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    canonical_location: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    resolution_method: Mapped[Optional[str]] = mapped_column(String(32), default="EXACT")
    resolution_confidence: Mapped[Optional[float]] = mapped_column(Float, default=1.0)
    resolution_confirmed: Mapped[Optional[bool]] = mapped_column(Float, default=False)

    # v4 Generalized state
    state_json: Mapped[str] = mapped_column(Text, default="{}")

    @property
    def attributes(self) -> Dict[str, Any]:
        try:
            return json.loads(self.attributes_json)
        except Exception:
            return {}

    @property
    def state(self) -> Dict[str, Any]:
        try:
            d = json.loads(self.state_json) if self.state_json else {}
            if not d:
                d = self.attributes
            return d
        except Exception:
            return self.attributes

    @property
    def location(self) -> Dict[str, Any]:
        try:
            return json.loads(self.location_json)
        except Exception:
            return {"room": self.room, "x": self.x, "y": self.y, "z": self.z}

    @property
    def metadata_dict(self) -> Dict[str, Any]:
        try:
            return json.loads(self.metadata_json)
        except Exception:
            return {}


class ObservationRepository:
    """Repository for managing append-only observation logs."""

    def __init__(self, session: Session, namespace: str = "default"):
        self.session = session
        self.namespace = namespace

    def add(
        self,
        entity_id: str,
        obs_input: ObservationInput,
        provenance: str = "sensor",
        namespace: Optional[str] = None,
        raw_location: Optional[str] = None,
        normalized_location: Optional[str] = None,
        canonical_location: Optional[str] = None,
        resolution_method: str = "EXACT",
        resolution_confidence: float = 1.0,
        resolution_confirmed: bool = False,
    ) -> ObservationModel:
        obs_id = f"obs_{uuid.uuid4().hex[:12]}"
        ts = obs_input.timestamp or datetime.now(timezone.utc).isoformat()
        ns = namespace or self.namespace

        loc = obs_input.location
        if hasattr(loc, "to_dict"):
            loc_dict = loc.to_dict()
            room = loc.room
            x, y, z = loc.x, loc.y, loc.z
        elif isinstance(loc, dict):
            loc_dict = loc
            room = loc.get("room", "unknown")
            x, y, z = loc.get("x"), loc.get("y"), loc.get("z")
        else:
            loc_dict = {"room": str(loc)}
            room = str(loc)
            x, y, z = None, None, None

        raw_loc = raw_location or room
        canon_loc = canonical_location or room
        state_dict = obs_input.state or obs_input.attributes or {}

        record = ObservationModel(
            observation_id=obs_id,
            entity_id=entity_id,
            entity_label=obs_input.entity,
            attributes_json=json.dumps(obs_input.attributes or {}),
            state_json=json.dumps(state_dict),
            location_json=json.dumps(loc_dict),
            room=room,
            x=x,
            y=y,
            z=z,
            timestamp=ts,
            confidence=obs_input.confidence,
            source=obs_input.source,
            provenance=str(provenance).lower(),
            namespace=ns,
            metadata_json=json.dumps(obs_input.metadata or {}),
            raw_location=raw_loc,
            normalized_location=normalized_location or raw_loc.lower(),
            canonical_location=canon_loc,
            resolution_method=resolution_method,
            resolution_confidence=resolution_confidence,
            resolution_confirmed=resolution_confirmed,
        )
        self.session.add(record)
        self.session.flush()
        return record

    def get_by_id(self, obs_id: str) -> Optional[ObservationModel]:
        return self.session.query(ObservationModel).filter_by(observation_id=obs_id).first()

    def get_by_entity(self, entity_id: str, limit: Optional[int] = None, namespace: Optional[str] = None) -> List[ObservationModel]:
        ns = namespace or self.namespace
        query = (
            self.session.query(ObservationModel)
            .filter_by(entity_id=entity_id, namespace=ns)
            .order_by(ObservationModel.timestamp.desc())
        )
        if limit:
            query = query.limit(limit)
        return query.all()

    def get_all_for_entities(self, entity_ids: List[str], namespace: Optional[str] = None) -> List[ObservationModel]:
        ns = namespace or self.namespace
        if not entity_ids:
            return []
        return (
            self.session.query(ObservationModel)
            .filter(ObservationModel.entity_id.in_(entity_ids), ObservationModel.namespace == ns)
            .order_by(ObservationModel.timestamp.desc())
            .all()
        )

    def get_by_room(self, room: str, namespace: Optional[str] = None) -> List[ObservationModel]:
        ns = namespace or self.namespace
        return (
            self.session.query(ObservationModel)
            .filter(ObservationModel.room.ilike(room), ObservationModel.namespace == ns)
            .order_by(ObservationModel.timestamp.desc())
            .all()
        )

    def get_all(self, namespace: Optional[str] = None) -> List[ObservationModel]:
        ns = namespace or self.namespace
        return self.session.query(ObservationModel).filter_by(namespace=ns).order_by(ObservationModel.timestamp.asc()).all()


# ----------------------------------------------------------------------
# 3. Current State & State Transitions ORM Models & Repository
# ----------------------------------------------------------------------

class CurrentStateModel(Base):
    """Materialized best-estimate current belief state for each entity."""

    __tablename__ = "current_state"

    entity_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    room: Mapped[str] = mapped_column(String(64), index=True)
    x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    z: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    belief_json: Mapped[str] = mapped_column(Text, default="{}")
    attributes_json: Mapped[str] = mapped_column(Text, default="{}")
    attribute_beliefs_json: Mapped[str] = mapped_column(Text, default="{}")
    last_seen: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="OBSERVED")
    provenance: Mapped[str] = mapped_column(String(32), default="sensor")
    namespace: Mapped[str] = mapped_column(String(64), default="default", index=True)
    last_observation_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[str] = mapped_column(String(64))

    @property
    def location_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {"room": self.room}
        if self.x is not None:
            res["x"] = self.x
        if self.y is not None:
            res["y"] = self.y
        if self.z is not None:
            res["z"] = self.z
        return res

    @property
    def belief_data(self) -> Dict[str, Any]:
        try:
            return json.loads(self.belief_json)
        except Exception:
            return {}

    @property
    def attributes(self) -> Dict[str, Any]:
        try:
            return json.loads(self.attributes_json) if self.attributes_json else {}
        except Exception:
            return {}

    @property
    def attribute_beliefs(self) -> Dict[str, Dict[str, Any]]:
        try:
            return json.loads(self.attribute_beliefs_json) if self.attribute_beliefs_json else {}
        except Exception:
            return {}


class StateTransitionModel(Base):
    """Log of movements and state transitions."""

    __tablename__ = "state_transitions"

    transition_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_id: Mapped[str] = mapped_column(String(64), index=True)
    attribute_name: Mapped[str] = mapped_column(String(64), default="location")
    old_location_json: Mapped[str] = mapped_column(Text, default="{}")
    new_location_json: Mapped[str] = mapped_column(Text, default="{}")
    old_value_json: Mapped[str] = mapped_column(Text, default="{}")
    new_value_json: Mapped[str] = mapped_column(Text, default="{}")
    old_timestamp: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    new_timestamp: Mapped[str] = mapped_column(String(64))
    transition_type: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    provenance: Mapped[str] = mapped_column(String(32), default="sensor")
    namespace: Mapped[str] = mapped_column(String(64), default="default", index=True)
    observation_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    @property
    def old_location(self) -> Dict[str, Any]:
        try:
            return json.loads(self.old_location_json)
        except Exception:
            return {}

    @property
    def new_location(self) -> Dict[str, Any]:
        try:
            return json.loads(self.new_location_json)
        except Exception:
            return {}

    @property
    def old_value(self) -> Any:
        try:
            return json.loads(self.old_value_json) if self.old_value_json else None
        except Exception:
            return self.old_value_json

    @property
    def new_value(self) -> Any:
        try:
            return json.loads(self.new_value_json) if self.new_value_json else None
        except Exception:
            return self.new_value_json


class CurrentStateRepository:
    """Repository for current entity beliefs and transitions."""

    def __init__(self, session: Session, namespace: str = "default"):
        self.session = session
        self.namespace = namespace

    def get(self, entity_id: str, namespace: Optional[str] = None) -> Optional[CurrentStateModel]:
        ns = namespace or self.namespace
        return self.session.query(CurrentStateModel).filter_by(entity_id=entity_id, namespace=ns).first()

    def get_all(self, namespace: Optional[str] = None) -> List[CurrentStateModel]:
        ns = namespace or self.namespace
        return self.session.query(CurrentStateModel).filter_by(namespace=ns).all()

    def get_batch(self, entity_ids: List[str], namespace: Optional[str] = None) -> Dict[str, CurrentStateModel]:
        ns = namespace or self.namespace
        if not entity_ids:
            return {}
        records = self.session.query(CurrentStateModel).filter(CurrentStateModel.entity_id.in_(entity_ids), CurrentStateModel.namespace == ns).all()
        return {r.entity_id: r for r in records}

    def upsert(
        self,
        entity_id: str,
        room: str,
        confidence: float,
        last_seen: str,
        status: str,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        belief_data: Optional[Dict[str, Any]] = None,
        attributes_data: Optional[Dict[str, Any]] = None,
        attribute_beliefs_data: Optional[Dict[str, Dict[str, Any]]] = None,
        provenance: str = "sensor",
        namespace: Optional[str] = None,
        last_observation_id: Optional[str] = None,
    ) -> CurrentStateModel:
        now_iso = datetime.now(timezone.utc).isoformat()
        ns = namespace or self.namespace
        current = self.get(entity_id, namespace=ns)
        b_json = json.dumps(belief_data or {})
        attrs_json = json.dumps(attributes_data or {})
        attr_beliefs_json = json.dumps(attribute_beliefs_data or {})
        if current:
            current.room = room
            current.x = x
            current.y = y
            current.z = z
            current.confidence = confidence
            current.belief_json = b_json
            current.attributes_json = attrs_json
            current.attribute_beliefs_json = attr_beliefs_json
            current.last_seen = last_seen
            current.status = status
            current.provenance = str(provenance).lower()
            current.last_observation_id = last_observation_id
            current.updated_at = now_iso
        else:
            current = CurrentStateModel(
                entity_id=entity_id,
                room=room,
                x=x,
                y=y,
                z=z,
                confidence=confidence,
                belief_json=b_json,
                attributes_json=attrs_json,
                attribute_beliefs_json=attr_beliefs_json,
                last_seen=last_seen,
                status=status,
                provenance=str(provenance).lower(),
                namespace=ns,
                last_observation_id=last_observation_id,
                updated_at=now_iso,
            )
            self.session.add(current)
        self.session.flush()
        return current

    def record_transition(
        self,
        entity_id: str,
        old_location: Dict[str, Any],
        new_location: Dict[str, Any],
        old_timestamp: Optional[str],
        new_timestamp: str,
        transition_type: str,
        confidence: float,
        attribute_name: str = "location",
        old_value: Any = None,
        new_value: Any = None,
        provenance: str = "sensor",
        namespace: Optional[str] = None,
        observation_id: Optional[str] = None,
    ) -> StateTransitionModel:
        tid = f"trans_{uuid.uuid4().hex[:12]}"
        ns = namespace or self.namespace
        trans = StateTransitionModel(
            transition_id=tid,
            entity_id=entity_id,
            attribute_name=attribute_name,
            old_location_json=json.dumps(old_location or {}),
            new_location_json=json.dumps(new_location or {}),
            old_value_json=json.dumps(old_value) if old_value is not None else json.dumps(old_location or {}),
            new_value_json=json.dumps(new_value) if new_value is not None else json.dumps(new_location or {}),
            old_timestamp=old_timestamp,
            new_timestamp=new_timestamp,
            transition_type=transition_type,
            confidence=confidence,
            provenance=str(provenance).lower(),
            namespace=ns,
            observation_id=observation_id,
        )
        self.session.add(trans)
        self.session.flush()
        return trans

    def get_transitions(self, entity_id: Optional[str] = None, limit: Optional[int] = None, namespace: Optional[str] = None) -> List[StateTransitionModel]:
        ns = namespace or self.namespace
        query = self.session.query(StateTransitionModel).filter_by(namespace=ns)
        if entity_id:
            query = query.filter_by(entity_id=entity_id)
        query = query.order_by(StateTransitionModel.new_timestamp.desc())
        if limit:
            query = query.limit(limit)
        return query.all()

    def get_by_room(self, room: str, namespace: Optional[str] = None) -> List[CurrentStateModel]:
        ns = namespace or self.namespace
        return self.session.query(CurrentStateModel).filter(CurrentStateModel.room.ilike(room), CurrentStateModel.namespace == ns).all()


# ----------------------------------------------------------------------
# 5. Persistent Location Alias ORM Model & Repository
# ----------------------------------------------------------------------

class LocationAliasModel(Base):
    """Stores user-confirmed or rejected location alias mappings."""

    __tablename__ = "location_aliases"

    alias_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    raw_alias: Mapped[str] = mapped_column(String(128), index=True)
    canonical_location: Mapped[str] = mapped_column(String(128), index=True)
    namespace: Mapped[str] = mapped_column(String(64), default="default", index=True)
    status: Mapped[str] = mapped_column(String(32), default="CONFIRMED")  # CONFIRMED or REJECTED
    provenance: Mapped[str] = mapped_column(String(32), default="user_confirmed")
    created_at: Mapped[str] = mapped_column(String(64))


class LocationAliasRepository:
    """Repository for user-confirmed and rejected alias persistence."""

    def __init__(self, session: Session, namespace: str = "default"):
        self.session = session
        self.namespace = namespace

    def save_alias(
        self,
        raw_alias: str,
        canonical_location: str,
        status: str = "CONFIRMED",
        provenance: str = "user_confirmed",
        namespace: Optional[str] = None,
    ) -> LocationAliasModel:
        ns = namespace or self.namespace
        now_iso = datetime.now(timezone.utc).isoformat()

        # Check existing
        existing = (
            self.session.query(LocationAliasModel)
            .filter_by(raw_alias=raw_alias.strip().lower(), namespace=ns)
            .first()
        )
        if existing:
            existing.canonical_location = canonical_location.strip().lower()
            existing.status = status
            existing.provenance = provenance
            existing.created_at = now_iso
            self.session.flush()
            return existing

        aid = f"alias_{uuid.uuid4().hex[:10]}"
        alias_obj = LocationAliasModel(
            alias_id=aid,
            raw_alias=raw_alias.strip().lower(),
            canonical_location=canonical_location.strip().lower(),
            namespace=ns,
            status=status,
            provenance=provenance,
            created_at=now_iso,
        )
        self.session.add(alias_obj)
        self.session.flush()
        return alias_obj

    def get_aliases(self, namespace: Optional[str] = None) -> List[LocationAliasModel]:
        ns = namespace or self.namespace
        return self.session.query(LocationAliasModel).filter_by(namespace=ns).all()

