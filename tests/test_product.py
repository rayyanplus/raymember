"""Comprehensive Product MVP Unit and Integration Test Suite."""

import os
import pytest

from raymember.cli.main import cli
from raymember.policy.auto import AutoMemoryPolicy
from raymember.sdk import Raymember
from raymember.schemas import Location, ObservationInput


def test_automatic_policy_routing():
    policy = AutoMemoryPolicy(random_seed=42)
    obs1 = ObservationInput(entity="backpack", location=Location(room="bedroom"), confidence=0.95, source="camera")
    act1, loc1, exp1 = policy.predict_action_with_explanation(None, obs1)

    assert act1 == "INITIALIZE"
    assert loc1["room"] == "bedroom"
    assert "Initial observation" in exp1

    curr = {"location": {"room": "bedroom"}, "confidence": 0.90}
    obs2 = ObservationInput(entity="backpack", location=Location(room="living_room"), confidence=0.55, source="camera")
    act2, loc2, exp2 = policy.predict_action_with_explanation(curr, obs2)
    assert act2 in ("UPDATE", "REOBSERVE", "PRESERVE", "UNCERTAIN")
    assert len(exp2) > 0


def test_sdk_observe_get_and_persistence(tmp_path):
    db_path = str(tmp_path / "sdk_test.db")

    with Raymember(database_path=db_path) as mem:
        mem.observe("backpack", {"room": "bedroom", "x": 1.0, "y": 0.0, "z": 2.0}, confidence=0.92, source="sensor_1")
        res1 = mem.get("backpack")

        assert res1 is not None
        assert res1.current_location["room"] == "bedroom"
        assert res1.confidence == 0.92
        assert res1.uncertainty_status in ("CONFIRMED", "MOVED", "UNCERTAIN")

    # Restart session to verify SQLite persistence
    with Raymember(database_path=db_path) as mem2:
        res2 = mem2.get("backpack")
        assert res2 is not None
        assert res2.current_location["room"] == "bedroom"


def test_sdk_ask_history_and_context(tmp_path):
    db_path = str(tmp_path / "context_test.db")

    with Raymember(database_path=db_path) as mem:
        mem.observe("laptop", {"room": "office"}, confidence=0.95, source="camera")
        ans = mem.ask("Where is the laptop?")
        assert ans.entity == "laptop"
        assert "office" in ans.answer

        hist = mem.history("laptop")
        assert len(hist) == 1

        ctx_str = mem.context("Where is the laptop?")
        assert "RAYMEMBER WORLD CONTEXT" in ctx_str
        assert "laptop" in ctx_str

        ctx_json = mem.context_json("Where is the laptop?")
        assert ctx_json["query"] == "Where is the laptop?"
        assert len(ctx_json["selected_items"]) >= 1


def test_cli_execution(tmp_path, capsys):
    db_path = str(tmp_path / "cli_test.db")

    # 1. init
    cli(["init", "--db", db_path])
    assert os.path.exists(db_path)

    # 2. query
    with Raymember(database_path=db_path) as mem:
        mem.observe("keys", {"room": "kitchen"}, confidence=0.90)

    cli(["query", "Where are the keys?", "--db", db_path])
    captured = capsys.readouterr()
    assert "kitchen" in captured.out

    # 3. export
    cli(["export", "--db", db_path])
    captured_exp = capsys.readouterr()
    assert "keys" in captured_exp.out


def test_dashboard_fastapi_endpoints(tmp_path):
    try:
        from fastapi.testclient import TestClient
        from raymember.dashboard.app import app
    except (ImportError, RuntimeError):
        pytest.skip("fastapi/httpx optional dependency not installed")

    db_path = str(tmp_path / "dash_test.db")
    with Raymember(database_path=db_path) as mem:
        mem.observe("mug", {"room": "kitchen"}, confidence=0.88)

    os.environ["RAYMEMBER_DB_PATH"] = db_path
    client = TestClient(app)

    # Overview endpoint
    res_ov = client.get("/api/overview")
    assert res_ov.status_code == 200
    assert res_ov.json()["total_entities"] == 1

    # Entities endpoint
    res_ent = client.get("/api/entities")
    assert res_ent.status_code == 200
    assert len(res_ent.json()["entities"]) == 1

    # Query endpoint
    res_q = client.get("/api/query?q=Where+is+the+mug?")
    assert res_q.status_code == 200
    assert "kitchen" in res_q.json()["answer"]
