"""Unit tests for Layer 2 Probabilistic Belief Engine equations and entropy."""

import math
import pytest
from raymember.belief.engine import BeliefEngine, BeliefState, LocationBeliefItem
from raymember.schemas import Location, ObservationInput


def test_belief_state_normalization_and_entropy():
    items = [
        LocationBeliefItem(location={"room": "bedroom"}, probability=0.8),
        LocationBeliefItem(location={"room": "living_room"}, probability=0.2),
    ]
    state = BeliefState(entity_id="test_ent", location_beliefs=items)
    state.normalize()

    assert abs(sum(i.probability for i in state.location_beliefs) - 1.0) < 1e-5
    assert state.most_likely_location == {"room": "bedroom"}
    assert abs(state.belief_confidence - 0.8) < 1e-5
    assert 0.0 <= state.entropy <= 1.0


def test_bayesian_fusion_equation():
    engine = BeliefEngine(decay_rate=0.05)
    obs1 = ObservationInput(entity="backpack", location=Location(room="bedroom"), confidence=0.9, source="simulator")
    b1 = engine.fuse_observation(current_belief=None, obs=obs1, entity_id="bp_1", time_delta_hours=0.0)

    assert b1.most_likely_location == {"room": "bedroom"}
    assert b1.belief_confidence == 1.0  # Single location normalized probability

    # Second observation in living room with confidence 0.4 (conflict)
    obs2 = ObservationInput(entity="backpack", location=Location(room="living_room"), confidence=0.4, source="simulator")
    b2 = engine.fuse_observation(current_belief=b1, obs=obs2, entity_id="bp_1", time_delta_hours=1.0)

    # Bedroom should remain most likely because 0.9 prior decayed exceeds 0.4 new weak evidence
    assert b2.most_likely_location == {"room": "bedroom"}
    assert len(b2.location_beliefs) == 2
    assert b2.entropy > 0.0  # Entropy increased due to candidate uncertainty
