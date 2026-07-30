"""Probabilistic belief engine with explicit Bayesian fusion and entropy calculations."""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from raymember.schemas import Location, ObservationInput


class LocationBeliefItem(BaseModel):
    """Single candidate location probability pairing."""

    location: Union[str, Dict[str, Any]]
    probability: float = Field(..., ge=0.0)

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

    def calculate_entropy(self) -> float:
        """
        Calculates normalized Shannon entropy across N candidate locations:
        H(p) = - sum(p_i * log2(p_i)) / log2(max(N, 2))
        """
        if not self.location_beliefs or len(self.location_beliefs) <= 1:
            self.entropy = 0.0
            return 0.0

        probs = [item.probability for item in self.location_beliefs if item.probability > 0.0]
        if not probs:
            self.entropy = 0.0
            return 0.0

        total = sum(probs)
        norm_probs = [p / total for p in probs]

        h = -sum(p * math.log2(p) for p in norm_probs)
        max_h = math.log2(max(len(norm_probs), 2))
        self.entropy = float(max(0.0, min(1.0, h / max_h)))
        return self.entropy

    def normalize(self) -> None:
        """Normalizes probabilities across candidate locations to sum to 1.0."""
        if not self.location_beliefs:
            self.most_likely_location = "unknown"
            self.belief_confidence = 0.0
            self.entropy = 0.0
            return

        total_p = sum(item.probability for item in self.location_beliefs)
        if total_p > 0:
            for item in self.location_beliefs:
                item.probability = float(item.probability / total_p)

        self.location_beliefs.sort(key=lambda x: x.probability, reverse=True)
        top = self.location_beliefs[0]
        self.most_likely_location = top.location
        self.belief_confidence = float(top.probability)
        self.calculate_entropy()

    def to_dict(self) -> Dict[str, Any]:
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


class BeliefEngine:
    """
    Probabilistic belief fusion engine.
    Computes location belief distributions using explicit, deterministic Bayesian equations.
    """

    SOURCE_RELIABILITY: Dict[str, float] = {
        "simulator": 1.0,
        "camera": 0.95,
        "lidar": 0.95,
        "user": 0.90,
        "sensor": 0.85,
        "unknown": 0.70,
    }

    def __init__(self, decay_rate: float = 0.05):
        self.decay_rate = decay_rate  # lambda in e^(-lambda * delta_t)

    def fuse_observation(
        self,
        current_belief: Optional[BeliefState],
        obs: ObservationInput,
        entity_id: str,
        time_delta_hours: float = 1.0,
    ) -> BeliefState:
        """
        Bayesian belief fusion:
        P_unnorm(L) = P_prev(L) * e^(-lambda * dt) + I(L == L_obs) * (confidence_obs * reliability_source)
        P_new(L) = P_unnorm(L) / sum(P_unnorm(L'))
        """
        loc_dict = obs.location.to_dict() if isinstance(obs.location, Location) else obs.location
        obs_room = loc_dict.get("room", "unknown")

        source_weight = self.SOURCE_RELIABILITY.get(obs.source.lower(), 0.7)
        obs_evidence_power = obs.confidence * source_weight

        # Persistence decay factor for prior location probabilities
        persist_weight = math.exp(-self.decay_rate * max(0.0, time_delta_hours))

        if current_belief is None or not current_belief.location_beliefs:
            item = LocationBeliefItem(location=loc_dict, probability=obs_evidence_power)
            state = BeliefState(
                entity_id=entity_id,
                location_beliefs=[item],
                most_likely_location=loc_dict,
                belief_confidence=obs_evidence_power,
                entropy=0.0,
            )
            state.normalize()
            return state

        unnorm_beliefs: List[LocationBeliefItem] = []
        found_match = False

        for prev in current_belief.location_beliefs:
            prev_room = prev.get_room()
            if prev_room.lower() == obs_room.lower():
                found_match = True
                # Match location: prior decayed + new evidence
                new_p = (prev.probability * persist_weight) + obs_evidence_power
                unnorm_beliefs.append(LocationBeliefItem(location=loc_dict, probability=new_p))
            else:
                # Other locations: prior decayed
                new_p = prev.probability * persist_weight
                unnorm_beliefs.append(LocationBeliefItem(location=prev.location, probability=new_p))

        if not found_match:
            # New candidate location
            unnorm_beliefs.append(LocationBeliefItem(location=loc_dict, probability=obs_evidence_power))

        updated_state = BeliefState(entity_id=entity_id, location_beliefs=unnorm_beliefs)
        updated_state.normalize()
        return updated_state
