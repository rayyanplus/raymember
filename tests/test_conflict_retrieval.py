"""
Regression tests for conflict-aware natural-language retrieval.

Tests A–F as specified:
  A. High-trust user vs lower-trust agent conflict
  B. Confirmed movement
  C. Reobservation
  D. Grammar (plural/singular)
  E. Spatial wording (on/in)
  F. Existing SDK compatibility smoke test
"""

import pytest
from raymember.sdk import Raymember
from raymember.retrieval.language import (
    location_phrase,
    entity_subject,
    ObservationKind,
    classify_observation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test A: High-trust user observation vs lower-trust agent conflict
# ─────────────────────────────────────────────────────────────────────────────
class TestA_ConflictDetection:
    """
    A high-confidence user observation says car keys -> desk (conf 0.98, prov user).
    A lower-confidence agent observation says car keys -> kitchen (conf 0.60, prov agent).

    Expected:
    - current_location remains desk
    - has_conflict is True
    - conflicting_observations contains the kitchen entry
    - ask() answer mentions the conflict and does NOT say keys moved from kitchen to desk
    - ask() answer does NOT use "was" for "car keys"
    """

    def test_current_state_stays_on_desk(self, tmp_path):
        db = str(tmp_path / "a.db")
        mem = Raymember(database_path=db)
        mem.observe("car keys", {"room": "desk"}, confidence=0.98, source="user", provenance="user")
        mem.observe("car keys", {"room": "kitchen"}, confidence=0.60, source="agent", provenance="agent")

        state = mem.get("car keys")
        assert state is not None
        assert state.current_location.get("room") == "desk", (
            f"Expected desk but got {state.current_location.get('room')}"
        )
        mem.close()

    def test_conflict_is_exposed_in_entity_state(self, tmp_path):
        db = str(tmp_path / "a2.db")
        mem = Raymember(database_path=db)
        mem.observe("car keys", {"room": "desk"}, confidence=0.98, source="user", provenance="user")
        mem.observe("car keys", {"room": "kitchen"}, confidence=0.60, source="agent", provenance="agent")

        state = mem.get("car keys")
        assert state.has_conflict is True, "Expected has_conflict to be True"
        assert len(state.conflicting_observations) >= 1

        conflict_rooms = [c["room"] for c in state.conflicting_observations]
        assert "kitchen" in conflict_rooms, f"Expected kitchen in conflicting rooms, got {conflict_rooms}"
        mem.close()

    def test_conflict_summary_is_meaningful(self, tmp_path):
        db = str(tmp_path / "a3.db")
        mem = Raymember(database_path=db)
        mem.observe("car keys", {"room": "desk"}, confidence=0.98, source="user", provenance="user")
        mem.observe("car keys", {"room": "kitchen"}, confidence=0.60, source="agent", provenance="agent")

        state = mem.get("car keys")
        assert isinstance(state.conflict_summary, str)
        assert len(state.conflict_summary) > 0
        assert "kitchen" in state.conflict_summary.lower() or "conflict" in state.conflict_summary.lower()
        mem.close()

    def test_ask_mentions_conflict(self, tmp_path):
        db = str(tmp_path / "a4.db")
        mem = Raymember(database_path=db)
        mem.observe("car keys", {"room": "desk"}, confidence=0.98, source="user", provenance="user")
        mem.observe("car keys", {"room": "kitchen"}, confidence=0.60, source="agent", provenance="agent")

        result = mem.ask("Where are the car keys?")
        answer = result.answer.lower()

        # Must mention the conflict
        assert "kitchen" in answer, f"Expected 'kitchen' in answer: {result.answer}"
        assert "did not replace" in answer or "conflict" in answer or "lower" in answer, (
            f"Answer should mention the conflict was not accepted: {result.answer}"
        )
        mem.close()

    def test_ask_does_not_say_moved_from_kitchen(self, tmp_path):
        db = str(tmp_path / "a5.db")
        mem = Raymember(database_path=db)
        mem.observe("car keys", {"room": "desk"}, confidence=0.98, source="user", provenance="user")
        mem.observe("car keys", {"room": "kitchen"}, confidence=0.60, source="agent", provenance="agent")

        result = mem.ask("Where are the car keys?")
        answer = result.answer.lower()

        # Must NOT imply accepted movement from kitchen to desk
        bad_phrases = [
            "previously observed in the kitchen",
            "previously in the kitchen",
            "moved from the kitchen",
            "last seen in the kitchen",
        ]
        for phrase in bad_phrases:
            assert phrase not in answer, (
                f"Answer should not imply kitchen was a previous confirmed location. "
                f"Found '{phrase}' in: {result.answer}"
            )
        mem.close()

    def test_has_conflict_flag_in_query_result(self, tmp_path):
        db = str(tmp_path / "a6.db")
        mem = Raymember(database_path=db)
        mem.observe("car keys", {"room": "desk"}, confidence=0.98, source="user", provenance="user")
        mem.observe("car keys", {"room": "kitchen"}, confidence=0.60, source="agent", provenance="agent")

        result = mem.ask("Where are the car keys?")
        assert result.has_conflict is True
        assert len(result.conflicting_observations) >= 1
        assert result.conflict_summary != ""
        mem.close()

    def test_interpreted_history_classifies_conflicting(self, tmp_path):
        db = str(tmp_path / "a7.db")
        mem = Raymember(database_path=db)
        mem.observe("car keys", {"room": "desk"}, confidence=0.98, source="user", provenance="user")
        mem.observe("car keys", {"room": "kitchen"}, confidence=0.60, source="agent", provenance="agent")

        state = mem.get("car keys")
        kinds = [h["kind"] for h in state.interpreted_history]
        assert ObservationKind.CONFLICTING.value in kinds, (
            f"Expected CONFLICTING in interpreted kinds: {kinds}"
        )
        mem.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test B: Confirmed movement is correctly described
# ─────────────────────────────────────────────────────────────────────────────
class TestB_ConfirmedMovement:
    """
    bedroom -> living room is described as a genuine confirmed movement.
    ask() should say "previously observed in the bedroom" and use
    "living room" as the current location.
    """

    def test_movement_described_correctly(self, tmp_path):
        db = str(tmp_path / "b.db")
        mem = Raymember(database_path=db)
        mem.observe("backpack", {"room": "bedroom"}, confidence=1.0, source="camera")
        mem.observe("backpack", {"room": "living room"}, confidence=1.0, source="camera")

        result = mem.ask("Where is the backpack?")
        answer = result.answer.lower()

        assert "living room" in answer, f"Expected current location 'living room' in: {result.answer}"
        assert "bedroom" in answer, f"Expected previous location 'bedroom' in: {result.answer}"
        mem.close()

    def test_movement_does_not_show_conflict(self, tmp_path):
        db = str(tmp_path / "b2.db")
        mem = Raymember(database_path=db)
        mem.observe("backpack", {"room": "bedroom"}, confidence=1.0, source="camera")
        mem.observe("backpack", {"room": "living room"}, confidence=1.0, source="camera")

        state = mem.get("backpack")
        # Genuine confirmed movement should not be tagged as conflict
        # (bedroom was accepted as a prior location)
        assert state is not None
        assert state.current_location.get("room") == "living room"
        mem.close()

    def test_no_conflict_on_clean_movement(self, tmp_path):
        db = str(tmp_path / "b3.db")
        mem = Raymember(database_path=db)
        mem.observe("backpack", {"room": "bedroom"}, confidence=1.0, source="camera")
        mem.observe("backpack", {"room": "living room"}, confidence=1.0, source="camera")

        result = mem.ask("Where is the backpack?")
        # On clean high-confidence movement, conflict flag should be False
        # (bedroom is an accepted transition, not conflicting evidence)
        assert result.has_conflict is False, (
            f"Clean high-confidence movement should not be flagged as conflict. "
            f"Answer: {result.answer}"
        )
        mem.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test C: Reobservation is NOT described as movement
# ─────────────────────────────────────────────────────────────────────────────
class TestC_Reobservation:
    """
    Repeated observations of the same location should not be described as movement.
    """

    def test_repeated_same_room_not_movement(self, tmp_path):
        db = str(tmp_path / "c.db")
        mem = Raymember(database_path=db)
        mem.observe("phone", {"room": "office"}, confidence=0.9, source="sensor")
        mem.observe("phone", {"room": "office"}, confidence=0.95, source="sensor")
        mem.observe("phone", {"room": "office"}, confidence=0.88, source="sensor")

        result = mem.ask("Where is the phone?")
        answer = result.answer.lower()

        # Should not say "previously observed in the office" as if it moved
        assert "moved" not in answer or "no confirmed" in answer, (
            f"Should not describe reobservation as movement: {result.answer}"
        )
        assert "office" in answer
        mem.close()

    def test_reobservation_classified_correctly(self, tmp_path):
        db = str(tmp_path / "c2.db")
        mem = Raymember(database_path=db)
        mem.observe("phone", {"room": "office"}, confidence=0.9, source="sensor")
        mem.observe("phone", {"room": "office"}, confidence=0.95, source="sensor")

        state = mem.get("phone")
        kinds = {h["kind"] for h in state.interpreted_history}
        # All same-room obs should be ACCEPTED_CURRENT or REOBSERVATION
        assert ObservationKind.CONFLICTING.value not in kinds, (
            f"Same-room reobservations should not be CONFLICTING: {kinds}"
        )
        mem.close()

    def test_no_conflict_on_reobservation(self, tmp_path):
        db = str(tmp_path / "c3.db")
        mem = Raymember(database_path=db)
        mem.observe("phone", {"room": "office"}, confidence=0.9, source="sensor")
        mem.observe("phone", {"room": "office"}, confidence=0.95, source="sensor")

        state = mem.get("phone")
        assert state.has_conflict is False
        mem.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test D: Grammar — plural vs singular
# ─────────────────────────────────────────────────────────────────────────────
class TestD_Grammar:
    """
    Plural: "car keys were", "scissors were"
    Singular: "backpack was", "phone was"
    """

    def test_car_keys_plural(self):
        assert "were" in entity_subject("car keys"), (
            f"'car keys' should use 'were', got: {entity_subject('car keys')}"
        )

    def test_keys_plural(self):
        assert "were" in entity_subject("keys")

    def test_scissors_plural(self):
        assert "were" in entity_subject("scissors")

    def test_backpack_singular(self):
        assert "was" in entity_subject("backpack"), (
            f"'backpack' should use 'was', got: {entity_subject('backpack')}"
        )

    def test_phone_singular(self):
        assert "was" in entity_subject("phone")

    def test_laptop_singular(self):
        assert "was" in entity_subject("laptop")

    def test_car_keys_ask_uses_were(self, tmp_path):
        """The ask() answer for 'car keys' must use 'were', not 'was'."""
        db = str(tmp_path / "d.db")
        mem = Raymember(database_path=db)
        mem.observe("car keys", {"room": "desk"}, confidence=0.98, source="user", provenance="user")

        result = mem.ask("Where are the car keys?")
        answer_lower = result.answer.lower()

        assert " was " not in answer_lower or "it was" in answer_lower, (
            f"'car keys' answer should not use singular 'was': {result.answer}"
        )
        mem.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test E: Spatial wording (on vs in)
# ─────────────────────────────────────────────────────────────────────────────
class TestE_SpatialWording:
    """
    desk, table, shelf -> "on the ..."
    bedroom, kitchen, living room -> "in the ..."
    """

    def test_on_desk(self):
        assert location_phrase("desk") == "on the desk"

    def test_on_table(self):
        assert location_phrase("table") == "on the table"

    def test_on_shelf(self):
        assert location_phrase("shelf") == "on the shelf"

    def test_in_bedroom(self):
        assert location_phrase("bedroom") == "in the bedroom"

    def test_in_kitchen(self):
        assert location_phrase("kitchen") == "in the kitchen"

    def test_in_living_room(self):
        assert location_phrase("living room") == "in the living room"

    def test_in_living_room_underscored(self):
        assert location_phrase("living_room") == "in the living room"

    def test_on_counter(self):
        assert location_phrase("counter") == "on the counter"

    def test_in_office(self):
        assert location_phrase("office") == "in the office"

    def test_ask_desk_uses_on(self, tmp_path):
        """The ask() answer for an entity on the desk must say 'on the desk'."""
        db = str(tmp_path / "e.db")
        mem = Raymember(database_path=db)
        mem.observe("wallet", {"room": "desk"}, confidence=0.95, source="user")

        result = mem.ask("Where is the wallet?")
        answer_lower = result.answer.lower()

        assert "on the desk" in answer_lower, (
            f"Expected 'on the desk', got: {result.answer}"
        )
        assert "in the desk" not in answer_lower, (
            f"Should NOT say 'in the desk', got: {result.answer}"
        )
        mem.close()

    def test_ask_bedroom_uses_in(self, tmp_path):
        """The ask() answer for an entity in the bedroom must say 'in the bedroom'."""
        db = str(tmp_path / "e2.db")
        mem = Raymember(database_path=db)
        mem.observe("suitcase", {"room": "bedroom"}, confidence=0.95, source="user")

        result = mem.ask("Where is the suitcase?")
        answer_lower = result.answer.lower()

        assert "in the bedroom" in answer_lower, (
            f"Expected 'in the bedroom', got: {result.answer}"
        )
        mem.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test F: Existing SDK behavior compatibility smoke test
# ─────────────────────────────────────────────────────────────────────────────
class TestF_SDKCompatibility:
    """
    Verify that the existing public SDK API still works correctly after changes.
    """

    def test_observe_returns_observation_record(self, tmp_path):
        db = str(tmp_path / "f.db")
        mem = Raymember(database_path=db)
        rec = mem.observe("laptop", {"room": "office"}, confidence=0.9, source="camera")
        assert rec.entity_label == "laptop"
        mem.close()

    def test_get_returns_entity_state_result(self, tmp_path):
        db = str(tmp_path / "f2.db")
        mem = Raymember(database_path=db)
        mem.observe("laptop", {"room": "office"}, confidence=0.9, source="camera")
        state = mem.get("laptop")
        assert state is not None
        assert state.current_location.get("room") == "office"
        assert state.confidence > 0.0
        mem.close()

    def test_get_returns_none_for_unknown_entity(self, tmp_path):
        db = str(tmp_path / "f3.db")
        mem = Raymember(database_path=db)
        state = mem.get("nonexistent_entity_xyz")
        assert state is None
        mem.close()

    def test_history_is_still_append_only(self, tmp_path):
        db = str(tmp_path / "f4.db")
        mem = Raymember(database_path=db)
        mem.observe("badge", {"room": "lobby"}, confidence=0.9)
        mem.observe("badge", {"room": "hallway"}, confidence=0.9)
        mem.observe("badge", {"room": "lobby"}, confidence=0.9)

        hist = mem.history("badge")
        assert len(hist) == 3, f"Expected 3 observations in history, got {len(hist)}"
        # History must include all rooms (raw, unfiltered)
        rooms = [h["room"] for h in hist]
        assert "lobby" in rooms
        assert "hallway" in rooms
        mem.close()

    def test_context_still_works(self, tmp_path):
        db = str(tmp_path / "f5.db")
        mem = Raymember(database_path=db)
        mem.observe("robot", {"room": "lab"}, confidence=0.9)
        ctx = mem.context("where is robot")
        assert isinstance(ctx, str)
        assert len(ctx) > 0
        mem.close()

    def test_to_dict_includes_conflict_fields(self, tmp_path):
        db = str(tmp_path / "f6.db")
        mem = Raymember(database_path=db)
        mem.observe("badge", {"room": "lobby"}, confidence=0.95)
        state = mem.get("badge")
        d = state.to_dict()
        assert "has_conflict" in d
        assert "conflicting_observations" in d
        assert "accepted_observation_ids" in d
        assert "rejected_observation_ids" in d
        assert "conflict_summary" in d
        assert "interpreted_history" in d
        mem.close()
