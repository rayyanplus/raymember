"""Pydantic schemas and data transfer objects for Raymember."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Location(BaseModel):
    """3D spatial location model."""

    room: str = Field(..., description="Room or spatial zone identifier")
    x: Optional[float] = Field(default=None, description="X coordinate")
    y: Optional[float] = Field(default=None, description="Y coordinate")
    z: Optional[float] = Field(default=None, description="Z coordinate")

    model_config = ConfigDict(extra="ignore")

    def distance_to(self, other: "Location") -> float:
        """Calculate Euclidean distance to another 3D location if coordinates present."""
        if (
            self.x is not None
            and self.y is not None
            and self.z is not None
            and other.x is not None
            and other.y is not None
            and other.z is not None
        ):
            return float(
                ((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2) ** 0.5
            )
        return 0.0 if self.room.lower() == other.room.lower() else 1.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"room": self.room}
        if self.x is not None:
            d["x"] = float(self.x)
        if self.y is not None:
            d["y"] = float(self.y)
        if self.z is not None:
            d["z"] = float(self.z)
        return d


class AttributeBelief(BaseModel):
    """Belief state for an individual entity attribute."""

    attribute_name: str
    believed_value: Any
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: str = "sensor"
    updated_at: str = ""
    alternative_values: List[Dict[str, Any]] = Field(default_factory=list)
    has_conflict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attribute_name": self.attribute_name,
            "believed_value": self.believed_value,
            "confidence": self.confidence,
            "provenance": self.provenance,
            "updated_at": self.updated_at,
            "alternative_values": self.alternative_values,
            "has_conflict": self.has_conflict,
        }


class ObservationInput(BaseModel):
    """Input payload for a new observation supporting location and generalized arbitrary state."""

    entity: str = Field(..., description="Entity canonical name or label")
    attributes: Dict[str, Any] = Field(default_factory=dict)
    state: Optional[Dict[str, Any]] = Field(default=None, description="Arbitrary state key-value mapping")
    location: Optional[Union[Location, Dict[str, Any]]] = Field(default=None, description="Optional spatial location model")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="user")
    provenance: str = Field(default="sensor")
    timestamp: Optional[Union[datetime, str]] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    entity_id: Optional[str] = Field(default=None, description="Explicit entity ID if known")
    entity_type: Optional[str] = Field(default=None, description="Entity type category if known")

    @field_validator("confidence")
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return float(v)

    @field_validator("location", mode="before")
    def validate_location(cls, v: Any) -> Optional[Location]:
        if v is None:
            return None
        if isinstance(v, Location):
            return v
        if isinstance(v, dict):
            if "room" not in v:
                v["room"] = "unknown"
            return Location(**v)
        return Location(room=str(v))

    def model_post_init(self, __context: Any) -> None:
        if self.location is None:
            if self.state and isinstance(self.state, dict):
                if "room" in self.state:
                    self.location = Location(room=str(self.state["room"]))
                elif "location" in self.state and isinstance(self.state["location"], dict):
                    loc_d = self.state["location"]
                    self.location = Location(**loc_d) if "room" in loc_d else Location(room="unknown")
                else:
                    self.location = Location(room="unknown")
            else:
                self.location = Location(room="unknown")

        if self.state is None:
            self.state = {}

        if self.attributes:
            for k, val in self.attributes.items():
                if k not in self.state:
                    self.state[k] = val

        if self.location:
            if "room" not in self.state or self.state["room"] == "unknown":
                self.state["room"] = self.location.room
            if "location" not in self.state:
                self.state["location"] = self.location.to_dict()

    def get_iso_timestamp(self) -> str:
        """Returns standard ISO string timestamp."""
        if self.timestamp is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(self.timestamp, datetime):
            return self.timestamp.isoformat()
        return str(self.timestamp)


class ObservationRecord(BaseModel):
    """Read schema for a stored observation record."""

    observation_id: str
    entity_id: str
    entity_label: str
    attributes: Dict[str, Any]
    state: Dict[str, Any] = Field(default_factory=dict)
    location: Dict[str, Any]
    room: str
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    timestamp: str
    confidence: float
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict, alias="metadata_dict")

    # --- Resolution metadata ---
    raw_location: Optional[str] = None
    normalized_location: Optional[str] = None
    canonical_location: Optional[str] = None
    resolution_method: str = "EXACT"
    resolution_confidence: float = 1.0
    resolution_confirmed: bool = False

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "entity_id": self.entity_id,
            "entity_label": self.entity_label,
            "attributes": self.attributes,
            "state": self.state or self.attributes,
            "location": self.location,
            "room": self.room,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
            "raw_location": self.raw_location or self.room,
            "normalized_location": self.normalized_location or self.room,
            "canonical_location": self.canonical_location or self.room,
            "resolution_method": self.resolution_method,
            "resolution_confidence": self.resolution_confidence,
            "resolution_confirmed": self.resolution_confirmed,
        }


class LocationBelief(BaseModel):
    """Probability associated with a location candidate."""

    location: Union[str, Dict[str, Any]]
    probability: float = Field(..., ge=0.0, le=1.0)


class BeliefDistribution(BaseModel):
    """Belief state distribution for an entity's current location."""

    entity_id: str
    location_beliefs: List[LocationBelief]
    most_likely_location: Union[str, Dict[str, Any]]
    belief_confidence: float = Field(..., ge=0.0, le=1.0)
    alternative_locations: List[Dict[str, Any]] = Field(default_factory=list)
    entropy: float = 0.0


class QueryResult(BaseModel):
    """Structured response returned by WorldMemory queries."""

    answer: str
    entity: str
    current_location: Optional[Union[Dict[str, Any], str]] = None
    confidence: float = 1.0
    belief_confidence: float = 1.0
    last_seen: str = ""
    previous_location: Optional[Union[Dict[str, Any], str]] = None
    state: str = "OBSERVED"  # OBSERVED, MOVED, REOBSERVED, CONFLICT, UNCERTAIN
    evidence: List[str] = Field(default_factory=list)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    alternative_locations: List[Dict[str, Any]] = Field(default_factory=list)
    explanation: str = ""

    # Generalized state & attribute fields
    current_attributes: Dict[str, Any] = Field(default_factory=dict)
    attribute_beliefs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    accepted_transitions: List[Dict[str, Any]] = Field(default_factory=list)

    # --- Conflict metadata (all optional for backward compatibility) ---
    has_conflict: bool = False
    conflicting_observations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Observations that were stored but did NOT replace the current belief. "
            "Each entry contains: room/attribute, confidence, provenance, timestamp, reason."
        ),
    )
    accepted_observation_ids: List[str] = Field(
        default_factory=list,
        description="Observation IDs that contributed to accepted state transitions.",
    )
    rejected_observation_ids: List[str] = Field(
        default_factory=list,
        description="Observation IDs classified as conflicting or blocked.",
    )
    conflict_summary: str = Field(
        default="",
        description="Human-readable single-sentence summary of any detected conflict.",
    )
    interpreted_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Derived evidence classification layer. Each entry annotates a raw "
            "observation with its ObservationKind: ACCEPTED_CURRENT, "
            "ACCEPTED_TRANSITION, REOBSERVATION, CONFLICTING, or UNCERTAIN."
        ),
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ContextResult(BaseModel):
    """Formatted context representation for external LLMs."""

    summary: str
    entity: str
    current_belief: Union[str, Dict[str, Any]]
    belief_confidence: float
    alternative: Optional[str] = None
    recent_evidence: List[str] = Field(default_factory=list)
    state: str = "OBSERVED"

    def to_formatted_prompt(self) -> str:
        """Converts context into clean text block suitable for LLM injection."""
        lines = [
            "WORLD MEMORY",
            f"Entity: {self.entity}",
            f"Current belief: {self.current_belief}",
            f"Belief confidence: {int(self.belief_confidence * 100)}%",
        ]
        if self.alternative:
            lines.append(f"Alternative: {self.alternative}")
        if self.recent_evidence:
            lines.append("Recent evidence:")
            for ev in self.recent_evidence:
                lines.append(f"- {ev}")
        lines.append(f"State: {self.state}")
        return "\n".join(lines)
