"""Belief state data representations and probability normalization."""

import json
from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field


class LocationBeliefItem(BaseModel):
    """Single location candidate probability pairing."""

    location: Union[str, Dict[str, Any]]
    probability: float = Field(..., ge=0.0, le=1.0)

    def get_room(self) -> str:
        if isinstance(self.location, str):
            return self.location
        if isinstance(self.location, dict):
            return str(self.location.get("room", "unknown"))
        return "unknown"


class BeliefState(BaseModel):
    """Complete belief probability distribution for an entity."""

    entity_id: str
    location_beliefs: List[LocationBeliefItem] = Field(default_factory=list)
    most_likely_location: Union[str, Dict[str, Any]] = "unknown"
    belief_confidence: float = 0.0
    entropy: float = 0.0

    def normalize(self) -> None:
        """Normalizes probabilities across candidates so they sum to 1.0."""
        if not self.location_beliefs:
            self.most_likely_location = "unknown"
            self.belief_confidence = 0.0
            self.entropy = 0.0
            return

        total_p = sum(item.probability for item in self.location_beliefs)
        if total_p > 0:
            for item in self.location_beliefs:
                item.probability = float(item.probability / total_p)

        # Sort by highest probability descending
        self.location_beliefs.sort(key=lambda x: x.probability, reverse=True)
        top = self.location_beliefs[0]
        self.most_likely_location = top.location
        self.belief_confidence = float(top.probability)

    def to_dict(self) -> Dict[str, Any]:
        """Converts belief state to JSON-serializable dictionary."""
        alt = [
            {"location": item.location, "probability": float(item.probability)}
            for item in self.location_beliefs[1:]
        ]
        return {
            "entity_id": self.entity_id,
            "location_beliefs": [
                {"location": item.location, "probability": float(item.probability)}
                for item in self.location_beliefs
            ],
            "most_likely_location": self.most_likely_location,
            "belief_confidence": self.belief_confidence,
            "alternative_locations": alt,
            "entropy": self.entropy,
        }
