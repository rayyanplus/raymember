"""Comprehensive Phase 4 Release Candidate Test Suite."""

import json
import os
import pytest

from raymember.cli.main import cli
from raymember.integrations.generic import MemoryAgent
from raymember.migrations.runner import MigrationRunner
from raymember.retrieval.ranking import RankedContextRetriever
from raymember.sdk import Raymember
from raymember.storage.export_import import ExportImportEngine
from raymember.validation.provenance import ProvenanceValidator
from raymember.validation.writes import MemoryWriteValidator


def mock_model_func(prompt: str) -> str:
    if "backpack" in prompt.lower():
        return "The backpack is in the living room."
    return "No information."


def test_agent_integration_layer(tmp_path):
    db_path = str(tmp_path / "agent_int.db")
    with Raymember(database_path=db_path) as mem:
        mem.observe("backpack", {"room": "living_room"}, confidence=0.95, provenance="user")
        agent = MemoryAgent(memory=mem, model=mock_model_func)
        resp = agent.run("Where is the backpack?")
        assert "living room" in resp


def test_provenance_and_write_safety(tmp_path):
    validator = MemoryWriteValidator()
    curr_user = {"provenance": "user", "confidence": 0.95}

    # Attempt low-confidence agent write over user memory
    low_agent_obs = type("Obs", (), {"confidence": 0.50, "location": {"room": "kitchen"}})()
    allowed, reason, _ = validator.validate_write(curr_user, low_agent_obs, provenance="agent")
    assert allowed is False
    assert "blocked" in reason.lower()

    # Valid user write
    valid_user_obs = type("Obs", (), {"confidence": 0.90, "location": {"room": "living_room"}})()
    allowed2, _, _ = validator.validate_write(curr_user, valid_user_obs, provenance="user")
    assert allowed2 is True


def test_namespace_isolation_and_switching(tmp_path):
    db_path = str(tmp_path / "ns_test.db")

    with Raymember(database_path=db_path, namespace="home") as mem:
        mem.observe("mug", {"room": "kitchen"}, confidence=0.90)

        # Switch to office namespace
        mem.use_namespace("office")
        mem.observe("desk_lamp", {"room": "office"}, confidence=0.95)

        # Query office namespace
        off_res = mem.get("desk_lamp")
        assert off_res is not None
        assert mem.get("mug") is None

        # Switch back to home
        mem.use_namespace("home")
        home_res = mem.get("mug")
        assert home_res is not None
        assert mem.get("desk_lamp") is None

        namespaces = mem.list_namespaces()
        assert "home" in namespaces
        assert "office" in namespaces


def test_import_export_round_trip(tmp_path):
    db1_path = str(tmp_path / "source.db")
    db2_path = str(tmp_path / "restored.db")
    export_json_path = str(tmp_path / "export.json")

    with Raymember(database_path=db1_path, namespace="home") as mem:
        mem.observe("laptop", {"room": "office"}, confidence=0.95, provenance="user")

    # Export to JSON
    data = ExportImportEngine.export_to_json(db1_path, output_path=export_json_path)
    assert os.path.exists(export_json_path)
    assert len(data["entities"]) == 1

    # Import into restored database
    count = ExportImportEngine.import_from_json(db2_path, input_path=export_json_path)
    assert count == 1

    with Raymember(database_path=db2_path, namespace="home") as mem2:
        res = mem2.get("laptop")
        assert res is not None
        assert res.current_location["room"] == "office"


def test_invalid_import_rejection(tmp_path):
    bad_json = str(tmp_path / "bad.json")
    with open(bad_json, "w") as f:
        json.dump({"invalid_key": "data"}, f)

    db_path = str(tmp_path / "invalid.db")
    with pytest.raises(ValueError):
        ExportImportEngine.import_from_json(db_path, bad_json)


def test_schema_migration_runner(tmp_path):
    db_path = str(tmp_path / "migrate.db")
    runner = MigrationRunner(db_path)
    old_v, new_v = runner.check_and_migrate()
    assert new_v == MigrationRunner.CURRENT_SCHEMA_VERSION


def test_ranked_context_retrieval(tmp_path):
    db_path = str(tmp_path / "ranking.db")
    with Raymember(database_path=db_path) as mem:
        mem.observe("backpack", {"room": "bedroom"}, confidence=0.90, provenance="sensor")
        mem.observe("backpack", {"room": "living_room"}, confidence=0.95, provenance="user")

        diag = mem.context_result("Where is the backpack?", max_items=5, max_characters=1000)
        assert len(diag.selected_items) > 0
        assert diag.relevance_scores[0] > 0
        assert "RAYMEMBER WORLD CONTEXT" in diag.formatted_context
