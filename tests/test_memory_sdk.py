"""Unit tests for the underlying Raymember WorldMemory engine."""

import pytest
from raymember import Location, WorldMemory


@pytest.fixture
def memory(tmp_path):
    db_file = str(tmp_path / "test_memory.db")
    mem = WorldMemory(database_path=db_file)
    yield mem
    mem.close()


def test_1_initialize_memory_with_new_entity(memory):
    obs = memory.observe(
        entity="laptop",
        location=Location(room="office", x=2.5, y=1.0, z=0.8),
        attributes={"color": "silver", "brand": "apple"},
        confidence=0.95,
        source="camera_1",
    )
    assert obs.entity_label == "laptop"
    assert obs.room == "office"

    res = memory.query("Where is the laptop?")
    assert res.entity == "laptop"
    assert res.current_location["room"] == "office"
    assert res.belief_confidence == 0.95
    assert res.state == "OBSERVED"


def test_2_query_nonexistent_entity(memory):
    res = memory.query("Where is the bicycle?")
    assert res.entity == "bicycle"
    assert res.current_location in ({}, None)
    assert res.belief_confidence in (0.0, 1.0)
    assert res.state == "UNCERTAIN"


def test_3_relocation_movement_detection(memory):
    memory.observe(entity="keys", location={"room": "kitchen"}, confidence=0.9)
    memory.observe(entity="keys", location={"room": "living_room"}, confidence=0.92)

    res = memory.query("Where are the keys?")
    assert res.current_location["room"] == "living_room"
    assert res.previous_location["room"] == "kitchen"
    assert res.state == "MOVED"


def test_4_append_only_observation_log(memory):
    memory.observe(entity="backpack", location={"room": "hallway"}, confidence=0.8)
    memory.observe(entity="backpack", location={"room": "bedroom"}, confidence=0.9)

    hist = memory.get_history("backpack")
    assert len(hist) == 2
    r1 = hist[0].room if hasattr(hist[0], "room") else hist[0]["room"]
    r2 = hist[1].room if hasattr(hist[1], "room") else hist[1]["room"]
    assert {r1, r2} == {"hallway", "bedroom"}


def test_5_reobservation_at_same_location(memory):
    memory.observe(entity="phone", location={"room": "office"}, confidence=0.9)
    memory.observe(entity="phone", location={"room": "office"}, confidence=0.95)

    res = memory.query("Where is the phone?")
    assert res.current_location["room"] == "office"
    assert res.state in ("REOBSERVED", "OBSERVED")


def test_11_lower_confidence_conflicting_observations(memory):
    memory.observe(entity="laptop", location={"room": "office"}, confidence=0.95)
    memory.observe(entity="laptop", location={"room": "kitchen"}, confidence=0.20)

    res = memory.query("Where is the laptop?")
    assert res.current_location["room"] == "office"
    assert res.state in ("CONFLICT", "UNCERTAIN", "PRESERVE", "PRESERVED", "OBSERVED")
