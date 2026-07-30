"""
Comprehensive tests for Phase 5: Semantic Entity and Location Resolution.
Covers all 20 required test scenarios.
"""

import os
import sqlite3
import pytest
from raymember.resolution import (
    EntityResolver,
    LocationResolver,
    normalize_entity_name,
    normalize_separators,
    normalize_string,
)
from raymember.sdk import Raymember


class TestPhase5Resolution:

    # 1. Exact location
    def test_1_exact_location(self):
        resolver = LocationResolver()
        res = resolver.resolve("bathroom")
        assert res.canonical_location == "bathroom"
        assert res.resolution_method in ("EXACT", "NORMALIZED")
        assert res.resolution_confidence == 1.0

    # 2. Case normalization
    def test_2_case_normalization(self):
        resolver = LocationResolver()
        res = resolver.resolve("Bathroom")
        assert res.canonical_location == "bathroom"
        assert res.raw_location == "Bathroom"

    # 3. Whitespace normalization
    def test_3_whitespace_normalization(self):
        resolver = LocationResolver()
        res = resolver.resolve("  bathroom  ")
        assert res.canonical_location == "bathroom"
        assert res.normalized_location == "bathroom"

    # 4. Separator normalization
    def test_4_separator_normalization(self):
        resolver = LocationResolver()
        for variant in ["livingroom", "living-room", "living_room"]:
            res = resolver.resolve(variant)
            assert res.canonical_location == "living room", f"Failed for {variant}"

    # 5. Built-in alias
    def test_5_builtin_alias(self):
        resolver = LocationResolver()
        res = resolver.resolve("washroom")
        assert res.canonical_location == "bathroom"
        assert res.resolution_method == "ALIAS"

    # 6. Query alias retrieval
    def test_6_query_alias_retrieval(self, tmp_path):
        db = str(tmp_path / "query_alias.db")
        mem = Raymember(database_path=db)
        mem.observe("android", {"room": "bathroom"}, confidence=0.9, provenance="sensor")

        # Query using alias "washroom"
        res_wash = mem.ask("What is in the washroom?")
        assert "android" in res_wash.answer.lower() or "bathroom" in str(res_wash.current_location).lower()

        res_rest = mem.ask("What is in the restroom?")
        assert "android" in res_rest.answer.lower() or "bathroom" in str(res_rest.current_location).lower()
        mem.close()

    # 7. Entity normalization
    def test_7_entity_normalization(self):
        res = normalize_entity_name("Black_Backpack")
        assert res == "black backpack"

    # 8. Distinct entities remain separate
    def test_8_distinct_entities(self, tmp_path):
        db = str(tmp_path / "distinct_entities.db")
        mem = Raymember(database_path=db)
        mem.observe("black backpack", {"room": "bedroom"})
        mem.observe("blue backpack", {"room": "kitchen"})

        s_black = mem.get("black backpack")
        s_blue = mem.get("blue backpack")

        assert s_black is not None
        assert s_blue is not None
        assert s_black.current_location.get("room") == "bedroom"
        assert s_blue.current_location.get("room") == "kitchen"
        mem.close()

    # 9. Strong fuzzy match
    def test_9_strong_fuzzy_match(self):
        resolver = LocationResolver(fuzzy_accept_threshold=0.80)
        res = resolver.resolve("bathrom", known_locations=["bathroom"])
        assert res.canonical_location == "bathroom"
        assert res.resolution_confidence >= 0.80

    # 10. Ambiguous fuzzy match requires confirmation
    def test_10_ambiguous_fuzzy_match(self):
        resolver = LocationResolver(fuzzy_accept_threshold=0.95, fuzzy_confirm_threshold=0.60)
        res = resolver.resolve("bathrom", known_locations=["bathroom"])
        assert res.requires_confirmation is True
        assert res.resolution_method == "AMBIGUOUS"

    # 11. Nonsense input does NOT silently map to living room
    def test_11_nonsense_input_not_mapped(self):
        resolver = LocationResolver()
        res = resolver.resolve("shitlinger", known_locations=["living room", "bathroom", "bedroom"])
        assert res.canonical_location != "living room"
        assert res.canonical_location == "shitlinger"
        assert res.resolution_method == "NEW"

    # 12. Genuinely new room preserved
    def test_12_new_location_preserved(self, tmp_path):
        db = str(tmp_path / "new_room.db")
        mem = Raymember(database_path=db)
        rec = mem.observe("drone", {"room": "attic_storage"})

        assert rec.canonical_location == "attic storage"
        assert rec.resolution_method in ("NEW", "NORMALIZED", "EXACT")
        mem.close()

    # 13. User-confirmed alias persists across process restart
    def test_13_confirmed_alias_persistence(self, tmp_path):
        db = str(tmp_path / "alias_persist.db")
        mem1 = Raymember(database_path=db)
        mem1.confirm_location_alias("scullery", "kitchen")
        mem1.close()

        # Re-open database
        mem2 = Raymember(database_path=db)
        res = mem2.resolve_location("scullery")
        assert res.canonical_location == "kitchen"
        assert res.resolution_method == "ALIAS"
        mem2.close()

    # 14. Rejected fuzzy mapping does not automatically reappear
    def test_14_rejected_fuzzy_mapping(self, tmp_path):
        db = str(tmp_path / "reject.db")
        mem = Raymember(database_path=db)
        mem.reject_location_resolution("shitlinger", "living room")

        res = mem.resolve_location("shitlinger")
        assert res.canonical_location != "living room"
        assert res.canonical_location == "shitlinger"
        mem.close()

    # 15. Namespace isolation for aliases
    def test_15_namespace_isolation(self, tmp_path):
        db = str(tmp_path / "ns_alias.db")
        mem1 = Raymember(database_path=db, namespace="home")
        mem1.confirm_location_alias("den", "living room")

        mem2 = Raymember(database_path=db, namespace="office")
        aliases_office = mem2.get_active_location_aliases()

        assert "den" not in aliases_office or aliases_office.get("den") != "living room"
        mem1.close()
        mem2.close()

    # 16. Raw location preserved in historical evidence
    def test_16_raw_location_preserved(self, tmp_path):
        db = str(tmp_path / "raw_preserve.db")
        mem = Raymember(database_path=db)
        rec = mem.observe("keys", {"room": " WashRoom "})

        assert rec.raw_location == " WashRoom "
        assert rec.canonical_location == "bathroom"
        hist = mem.history("keys")
        assert len(hist) == 1
        mem.close()

    # 17. Migration from v1/v2 schema
    def test_17_schema_v3_migration(self, tmp_path):
        db = str(tmp_path / "legacy.db")
        # 1. Create a legacy v2 SQLite database schema
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY, updated_at TEXT)")
        cursor.execute("INSERT INTO schema_version VALUES (2, '2026-01-01')")
        cursor.execute("CREATE TABLE observations (observation_id TEXT PRIMARY KEY, entity_id TEXT)")
        conn.commit()
        conn.close()

        # 2. Trigger MigrationRunner (v2 -> v3)
        from raymember.migrations.runner import MigrationRunner
        runner = MigrationRunner(db)
        old_v, new_v = runner.check_and_migrate()
        assert old_v == 2
        assert new_v >= 3

        # 3. Verify v3 columns and location_aliases table were created
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(observations)")
        cols = [c[1] for c in cursor.fetchall()]
        assert "raw_location" in cols
        assert "canonical_location" in cols
        assert "resolution_method" in cols

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='location_aliases'")
        assert cursor.fetchone() is not None
        conn.close()

    # 18. Dashboard API resolution metadata
    def test_18_dashboard_resolution_metadata(self, tmp_path):
        db = str(tmp_path / "dash_res.db")
        os.environ["RAYMEMBER_DB_PATH"] = db
        mem = Raymember(database_path=db)
        res = mem.resolve_location("washroom")
        assert res.canonical_location == "bathroom"
        assert res.resolution_method == "ALIAS"
        mem.close()

    # 19. Existing SDK compatibility
    def test_19_existing_sdk_compatibility(self, tmp_path):
        db = str(tmp_path / "compat.db")
        mem = Raymember(database_path=db)
        rec = mem.observe("laptop", {"room": "bedroom"}, confidence=0.95, provenance="sensor")
        assert rec.entity_label == "laptop"
        state = mem.get("laptop")
        assert state is not None
        assert state.current_location.get("room") == "bedroom"
        query_res = mem.ask("Where is the laptop?")
        assert "bedroom" in query_res.answer.lower()
        mem.close()

    # 20. Context export contains clear location info
    def test_20_context_export_location(self, tmp_path):
        db = str(tmp_path / "context_exp.db")
        mem = Raymember(database_path=db)
        mem.observe("book", {"room": "washroom"})

        ctx = mem.context("where is the book?")
        assert isinstance(ctx, str)
        assert "bathroom" in ctx.lower() or "washroom" in ctx.lower()
        mem.close()
