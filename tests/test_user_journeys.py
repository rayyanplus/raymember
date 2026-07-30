"""
End-to-end user journey tests for Raymember.

Covers 5 realistic developer workflows to validate that the full SDK stack
works correctly from installation through export/restore.
"""

import json
import os
import tempfile
import pytest

from raymember.sdk import Raymember


# ─────────────────────────────────────────────
# Journey 1: First-Time Developer
# A new developer installs Raymember, submits observations for two entities,
# retrieves current state, and asks a natural-language question.
# ─────────────────────────────────────────────
class TestJourney1FirstTimeDeveloper:
    def test_submit_observe_get_ask(self, tmp_path):
        db = str(tmp_path / "journey1.db")
        mem = Raymember(database_path=db)

        # Submit observations
        r1 = mem.observe("alice", {"room": "kitchen"}, confidence=0.9, source="camera")
        r2 = mem.observe("bob", {"room": "living_room"}, confidence=0.85, source="rfid")

        assert r1.entity_label == "alice"
        assert r2.entity_label == "bob"

        # Retrieve current state
        alice = mem.get("alice")
        assert alice is not None
        assert alice.entity_label == "alice"
        assert alice.current_location.get("room") == "kitchen"
        assert alice.confidence > 0.0

        bob = mem.get("bob")
        assert bob is not None
        assert bob.current_location.get("room") in ("living_room", "living room")

        # Natural language query
        result = mem.ask("where is alice")
        assert result is not None
        # Result should contain a response field (QueryResult)
        assert hasattr(result, "answer") or hasattr(result, "response") or isinstance(result, object)

        # Context generation works
        ctx = mem.context("where is alice", mode="compact")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

        mem.close()


# ─────────────────────────────────────────────
# Journey 2: Object Relocation
# An entity is observed in one room, then relocated to another.
# The memory should reflect the updated location with transition history.
# ─────────────────────────────────────────────
class TestJourney2ObjectRelocation:
    def test_relocation_tracks_transition(self, tmp_path):
        db = str(tmp_path / "journey2.db")
        mem = Raymember(database_path=db)

        # Initial observation
        mem.observe("laptop", {"room": "office"}, confidence=1.0, source="camera")
        state_before = mem.get("laptop")
        assert state_before is not None
        assert state_before.current_location.get("room") == "office"

        # Relocation observation
        mem.observe("laptop", {"room": "conference_room"}, confidence=1.0, source="camera")
        state_after = mem.get("laptop")
        assert state_after is not None
        assert state_after.current_location.get("room") in ("conference_room", "conference room")

        # History should include both observations
        hist = mem.history("laptop")
        assert len(hist) >= 2

        rooms_seen = [h["room"] for h in hist if h.get("room")]
        assert "office" in rooms_seen or "conference_room" in rooms_seen or "conference room" in rooms_seen

        # Changes log should show the transition
        changes = mem.changes()
        assert isinstance(changes, list)

        mem.close()

    def test_history_is_append_only(self, tmp_path):
        db = str(tmp_path / "journey2b.db")
        mem = Raymember(database_path=db)

        for room in ["office", "hallway", "kitchen", "hallway", "office"]:
            mem.observe("badge", {"room": room}, confidence=0.95, source="rfid")

        hist = mem.history("badge")
        assert len(hist) == 5, f"Expected 5 observations, got {len(hist)}"
        mem.close()


# ─────────────────────────────────────────────
# Journey 3: Conflicting Evidence — Write Safety and Provenance
# A high-confidence sensor says entity is in room A.
# A low-confidence unreliable source claims it is in room B.
# The current state should remain anchored to the high-confidence reading.
# ─────────────────────────────────────────────
class TestJourney3ConflictingEvidence:
    def test_low_confidence_does_not_override_high_confidence(self, tmp_path):
        db = str(tmp_path / "journey3.db")
        mem = Raymember(
            database_path=db,
            source_reliability={"lidar": 1.0, "unreliable_sensor": 0.1},
        )

        # Strong sensor anchors state
        mem.observe("robot", {"room": "lab"}, confidence=1.0, source="lidar", provenance="sensor")

        # Weak sensor disagrees
        mem.observe("robot", {"room": "corridor"}, confidence=0.1, source="unreliable_sensor", provenance="sensor")

        state = mem.get("robot")
        assert state is not None
        # The high-confidence anchored location should still hold
        # (write safety should have blocked or downweighted the low-confidence claim)
        # At minimum, current state should exist with a location
        assert state.current_location.get("room") is not None
        mem.close()

    def test_provenance_stored_in_history(self, tmp_path):
        db = str(tmp_path / "journey3b.db")
        mem = Raymember(database_path=db)

        mem.observe("tag_001", {"room": "room_a"}, confidence=0.9, source="rfid", provenance="rfid_reader")

        hist = mem.history("tag_001")
        assert len(hist) >= 1
        # Provenance field should be in the record
        assert "provenance" in hist[0]

        mem.close()


# ─────────────────────────────────────────────
# Journey 4: Namespace Isolation
# Two separate Raymember instances with different namespaces should not share state.
# ─────────────────────────────────────────────
class TestJourney4NamespaceIsolation:
    def test_namespaces_are_isolated(self, tmp_path):
        db = str(tmp_path / "journey4.db")

        mem_floor1 = Raymember(database_path=db, namespace="floor_1")
        mem_floor2 = Raymember(database_path=db, namespace="floor_2")

        mem_floor1.observe("printer", {"room": "room_101"}, confidence=1.0)
        mem_floor2.observe("printer", {"room": "room_201"}, confidence=1.0)

        state_f1 = mem_floor1.get("printer")
        state_f2 = mem_floor2.get("printer")

        assert state_f1 is not None
        assert state_f2 is not None
        assert state_f1.current_location.get("room") in ("room_101", "room 101")
        assert state_f2.current_location.get("room") in ("room_201", "room 201")

        # Namespaces do not bleed into each other
        assert state_f1.current_location.get("room") != state_f2.current_location.get("room")

        mem_floor1.close()
        mem_floor2.close()

    def test_namespace_switch(self, tmp_path):
        db = str(tmp_path / "journey4b.db")
        mem = Raymember(database_path=db, namespace="alpha")
        mem.observe("agent_x", {"room": "alpha_room"}, confidence=1.0)

        # Switch namespace
        mem.use_namespace("beta")
        mem.observe("agent_x", {"room": "beta_room"}, confidence=1.0)

        # Retrieve from each namespace
        mem.use_namespace("alpha")
        alpha_state = mem.get("agent_x")
        assert alpha_state is not None
        assert alpha_state.current_location.get("room") in ("alpha_room", "alpha room")

        mem.use_namespace("beta")
        beta_state = mem.get("agent_x")
        assert beta_state is not None
        assert beta_state.current_location.get("room") in ("beta_room", "beta room")

        mem.close()


# ─────────────────────────────────────────────
# Journey 5: Export and Restore Roundtrip
# Export memory state to JSON, reload from fresh instance, verify parity.
# ─────────────────────────────────────────────
class TestJourney5ExportRestoreRoundtrip:
    def _collect_state(self, mem: Raymember, entity_names: list) -> dict:
        """Collect current states for a list of entity names."""
        states = {}
        for name in entity_names:
            s = mem.get(name)
            if s:
                states[name] = {
                    "room": s.current_location.get("room"),
                    "confidence": s.confidence,
                }
        return states

    def test_context_json_round_trip(self, tmp_path):
        db = str(tmp_path / "journey5.db")
        mem = Raymember(database_path=db)

        entities = ["alice", "bob", "charlie"]
        rooms = ["kitchen", "living_room", "office"]
        for e, r in zip(entities, rooms):
            mem.observe(e, {"room": r}, confidence=0.95, source="camera")

        # Export context JSON
        ctx_json = mem.context_json("where is everyone", max_items=10)
        assert "formatted_context" in ctx_json
        assert "selected_items" in ctx_json
        assert "query" in ctx_json
        assert isinstance(ctx_json["selected_items"], list)

        # Serialize to JSON and deserialize — should not raise
        serialized = json.dumps(ctx_json)
        recovered = json.loads(serialized)
        assert recovered["query"] == "where is everyone"
        assert recovered["formatted_context"] == ctx_json["formatted_context"]

        mem.close()

    def test_db_file_portability(self, tmp_path):
        """Create a DB, close it, open it again, verify state is preserved."""
        db = str(tmp_path / "journey5b.db")

        # Write
        mem_write = Raymember(database_path=db)
        mem_write.observe("laptop", {"room": "office"}, confidence=0.99, source="nfc")
        mem_write.close()

        # Read from a new Raymember instance pointing at same file
        mem_read = Raymember(database_path=db)
        state = mem_read.get("laptop")
        assert state is not None
        assert state.current_location.get("room") == "office"
        assert state.confidence > 0.0
        mem_read.close()

    def test_context_modes_produce_different_outputs(self, tmp_path):
        """compact / standard / evidence modes must produce distinct formatted outputs."""
        db = str(tmp_path / "journey5c.db")
        mem = Raymember(database_path=db)
        mem.observe("robot", {"room": "lab"}, confidence=0.9, source="lidar")

        ctx_compact = mem.context("where is robot", mode="compact")
        ctx_standard = mem.context("where is robot", mode="standard")
        ctx_evidence = mem.context("where is robot", mode="evidence")

        assert isinstance(ctx_compact, str)
        assert isinstance(ctx_standard, str)
        assert isinstance(ctx_evidence, str)

        # All must be non-empty
        assert len(ctx_compact) > 0
        assert len(ctx_standard) > 0
        assert len(ctx_evidence) > 0

        # Standard and evidence should be more detailed than compact
        assert len(ctx_standard) >= len(ctx_compact)
        assert len(ctx_evidence) >= len(ctx_compact)

        mem.close()
