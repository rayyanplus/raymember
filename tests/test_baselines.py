"""Unit tests for baseline memory policies."""

import pytest
from raymember.baselines.baselines import (
    LatestObservationBaseline,
    DeterministicRulesBaseline,
    ProbabilisticEngineBaseline,
)
from raymember.schemas import Location, ObservationInput


def test_latest_observation_baseline():
    b1 = LatestObservationBaseline()
    obs1 = ObservationInput(entity="backpack", location=Location(room="bedroom"), timestamp="2026-07-29T10:00:00Z")
    act1, loc1 = b1.predict_action(None, obs1)
    assert act1 == "INITIALIZE"
    assert loc1 == {"room": "bedroom"}

    curr = {"location": loc1, "last_seen": "2026-07-29T10:00:00Z"}

    # Out of order timestamp
    obs_old = ObservationInput(entity="backpack", location=Location(room="living_room"), timestamp="2026-07-29T09:00:00Z")
    act_old, _ = b1.predict_action(curr, obs_old)
    assert act_old == "PRESERVE"


def test_deterministic_rules_baseline():
    b2 = DeterministicRulesBaseline(min_confidence=0.5, movement_threshold=0.5)
    curr = {"location": {"room": "bedroom", "x": 0.0, "y": 0.0, "z": 0.0}, "last_seen": "2026-07-29T10:00:00Z"}

    # Low confidence -> PRESERVE
    obs_low = ObservationInput(entity="backpack", location=Location(room="living_room"), confidence=0.2, timestamp="2026-07-29T10:05:00Z")
    act_low, _ = b2.predict_action(curr, obs_low)
    assert act_low == "PRESERVE"

    # High confidence room change -> UPDATE
    obs_moved = ObservationInput(entity="backpack", location=Location(room="living_room"), confidence=0.9, timestamp="2026-07-29T10:10:00Z")
    act_moved, loc_moved = b2.predict_action(curr, obs_moved)
    assert act_moved == "UPDATE"
    assert loc_moved["room"] == "living_room"


def test_probabilistic_engine_baseline():
    b3 = ProbabilisticEngineBaseline()
    obs = ObservationInput(entity="backpack", location=Location(room="bedroom"), confidence=0.9)
    act, loc = b3.predict_action(None, obs)
    assert act == "INITIALIZE"
    assert loc["room"] == "bedroom"
