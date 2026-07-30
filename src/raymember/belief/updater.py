"""Belief updater orchestrator for Layer 2."""

from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from raymember.belief.fusion import EvidenceFusion
from raymember.belief.state import BeliefState, LocationBeliefItem
from raymember.schemas import ObservationInput
from raymember.storage.models import CurrentStateRepository


class BeliefUpdater:
    """Updates probabilistic entity location beliefs."""

    def __init__(self, session: Session, decay_rate: float = 0.05):
        self.session = session
        self.fusion = EvidenceFusion(decay_rate=decay_rate)
        self.current_state_repo = CurrentStateRepository(session)

    def get_belief(self, entity_id: str) -> Optional[BeliefState]:
        """Retrieves existing belief state for an entity."""
        curr = self.current_state_repo.get(entity_id)
        if not curr:
            return None
        b_data = curr.belief_data
        if not b_data or "location_beliefs" not in b_data:
            loc = curr.location_dict
            item = LocationBeliefItem(location=loc, probability=curr.confidence)
            b = BeliefState(
                entity_id=entity_id,
                location_beliefs=[item],
                most_likely_location=loc,
                belief_confidence=curr.confidence,
                entropy=0.0,
            )
            return b

        items = [LocationBeliefItem(**lb) for lb in b_data.get("location_beliefs", [])]
        b = BeliefState(
            entity_id=entity_id,
            location_beliefs=items,
            most_likely_location=b_data.get("most_likely_location", curr.location_dict),
            belief_confidence=b_data.get("belief_confidence", curr.confidence),
            entropy=b_data.get("entropy", 0.0),
        )
        return b

    def update_belief(self, entity_id: str, obs_input: ObservationInput) -> BeliefState:
        """Computes updated belief state after receiving observation."""
        current_b = self.get_belief(entity_id)
        updated_b = self.fusion.Fuse_observation(current_b, obs_input, entity_id)
        return updated_b
