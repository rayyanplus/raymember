"""Evidence fusion logic for combining observations into belief distributions."""

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional

from raymember.belief.state import BeliefState, LocationBeliefItem
from raymember.belief.uncertainty import calculate_belief_confidence, calculate_entropy
from raymember.schemas import ObservationInput, Location


class EvidenceFusion:
    """Probabilistic evidence fusion engine."""

    SOURCE_RELIABILITY: Dict[str, float] = {
        "simulator": 1.0,
        "camera": 0.95,
        "lidar": 0.95,
        "user": 0.90,
        "sensor": 0.85,
        "unknown": 0.70,
    }

    def __init__(self, decay_rate: float = 0.05):
        self.decay_rate = decay_rate

    def Fuse_observation(
        self,
        current_belief: Optional[BeliefState],
        obs: ObservationInput,
        entity_id: str,
    ) -> BeliefState:
        """Fuses a new observation into an updated belief distribution."""
        loc_dict = obs.location.to_dict() if isinstance(obs.location, Location) else obs.location
        room_name = loc_dict.get("room", "unknown")

        source_weight = self.SOURCE_RELIABILITY.get(obs.source.lower(), 0.8)
        new_evidence_power = obs.confidence * source_weight

        if current_belief is None or not current_belief.location_beliefs:
            item = LocationBeliefItem(location=loc_dict, probability=new_evidence_power)
            b_state = BeliefState(
                entity_id=entity_id,
                location_beliefs=[item],
                most_likely_location=loc_dict,
                belief_confidence=new_evidence_power,
                entropy=0.0,
            )
            b_state.normalize()
            return b_state

        # Apply time decay to prior beliefs
        decayed_beliefs: List[LocationBeliefItem] = []
        found_matching_location = False

        for prev_item in current_belief.location_beliefs:
            prev_loc = prev_item.location
            prev_room = prev_loc if isinstance(prev_loc, str) else prev_loc.get("room")

            # Check if this prior candidate matches new observation room
            if prev_room and prev_room.lower() == room_name.lower():
                found_matching_location = True
                # Updated belief using Bayesian-style update
                updated_p = prev_item.probability + new_evidence_power * (1.0 - prev_item.probability)
                decayed_beliefs.append(LocationBeliefItem(location=loc_dict, probability=updated_p))
            else:
                # Decay prior belief for non-observed location
                decayed_p = prev_item.probability * math.exp(-self.decay_rate * 0.5)
                decayed_beliefs.append(LocationBeliefItem(location=prev_loc, probability=decayed_p))

        if not found_matching_location:
            # Add new candidate location
            decayed_beliefs.append(LocationBeliefItem(location=loc_dict, probability=new_evidence_power))

        new_belief_state = BeliefState(entity_id=entity_id, location_beliefs=decayed_beliefs)
        new_belief_state.normalize()
        new_belief_state.entropy = calculate_entropy(new_belief_state.location_beliefs)
        new_belief_state.belief_confidence = calculate_belief_confidence(
            new_belief_state.belief_confidence, new_belief_state.entropy
        )

        return new_belief_state
