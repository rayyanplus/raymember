"""
Unit and integration tests for Phase 6: Real Agent Integration and Behavioral Validation.
"""

import pytest
from raymember import Raymember
from raymember.evaluation.agent_comparison import (
    AgentComparisonBenchmark,
    DeterministicEvaluatorModel,
    EvaluationScenario,
)
from examples.real_agent_demo import OfflineMockAgentModel


class TestPhase6AgentIntegration:
    """Test suite verifying agent context delivery, conflict formatting, and offline execution."""

    def test_agent_receives_raymember_context(self, tmp_path):
        """Confirms that the agent prompt receives structured Raymember context."""
        db = str(tmp_path / "test_context.db")
        mem = Raymember(database_path=db)
        mem.observe(
            entity="shipment_482",
            state={"status": "out_for_delivery", "estimated_arrival": "16:30"},
            confidence=0.95,
            provenance="tracking_api",
        )

        ctx_str = mem.context("When will shipment_482 arrive?")
        assert "RAYMEMBER WORLD CONTEXT" in ctx_str
        assert "shipment 482" in ctx_str
        assert "16:30" in ctx_str
        mem.close()

    def test_conflicting_values_presented_correctly(self, tmp_path):
        """Confirms that conflicting values are presented in Raymember context."""
        db = str(tmp_path / "test_conflict_ctx.db")
        mem = Raymember(database_path=db)
        mem.observe(
            entity="shipment_482",
            state={"estimated_arrival": "16:30"},
            confidence=0.95,
            provenance="tracking_api",
        )
        mem.observe(
            entity="shipment_482",
            state={"estimated_arrival": "17:15"},
            confidence=0.30,
            provenance="unreliable_sensor",
        )

        ctx_str = mem.context("What is the arrival time for shipment_482?")
        assert "CONFLICTS:" in ctx_str
        assert "17:15" in ctx_str
        assert "30%" in ctx_str or "unreliable_sensor" in ctx_str
        mem.close()

    def test_no_memory_and_raymember_receive_identical_questions(self, tmp_path):
        """Confirms that Strategy A and Strategy C receive identical user questions."""
        scenario = EvaluationScenario(
            scenario_id="test_sc_01",
            category="normal",
            title="Identical question test",
            observations=[{"entity": "box", "location": {"room": "attic"}}],
            question="Where is the box?",
            ground_truth_attribute={"key": "room", "value": "attic"},
        )
        bm = AgentComparisonBenchmark(scenarios=[scenario])
        model = DeterministicEvaluatorModel()

        res_a = bm._evaluate_scenario(scenario, strategy="A", model=model)
        res_c = bm._evaluate_scenario(scenario, strategy="C", model=model)

        assert scenario.question in res_a.prompt
        assert scenario.question in res_c.prompt
        assert res_a.prompt.split("Question:")[-1] == res_c.prompt.split("Question:")[-1]

    def test_offline_demo_runs_without_cloud_dependencies(self, tmp_path):
        """Confirms that OfflineMockAgentModel executes without API keys or network."""
        model = OfflineMockAgentModel()

        # Call with no context
        resp_a = model("Question: When will shipment_482 arrive?\nAnswer:")
        assert "do not have access" in resp_a.lower()

        # Call with Raymember context
        sample_context = (
            "RAYMEMBER WORLD CONTEXT\n"
            "- [CURRENT BELIEF] Entity: shipment_482 | State: {'estimated_arrival': '16:30'} | CONFLICTS: estimated_arrival had conflicting update '17:15'"
        )
        resp_c = model(f"System Context:\n{sample_context}\n\nQuestion: When will shipment_482 arrive?\nAnswer:")
        assert "16:30" in resp_c
        assert "17:15" in resp_c or "rejected" in resp_c.lower() or "conflicting" in resp_c.lower()

    def test_agent_comparison_benchmark_executes_scenarios(self, tmp_path):
        """Confirms that AgentComparisonBenchmark executes default scenarios deterministically."""
        bm = AgentComparisonBenchmark()
        assert len(bm.scenarios) == 32

        summary = bm.run_benchmark(model=None, model_name="TestOfflineModel", is_real_model=False)

        assert summary["evaluation_metadata"]["total_scenarios"] == 32
        assert summary["evaluation_metadata"]["is_real_model"] is False
        assert "Strategy A (No Memory)" in summary["strategies"]
        assert "Strategy B (Naive History)" in summary["strategies"]
        assert "Strategy C (Raymember)" in summary["strategies"]

        strat_c = summary["strategies"]["Strategy C (Raymember)"]
        assert strat_c["accepted_state_accuracy"]["mean"] > 0.90
        assert strat_c["conflict_interpretation_accuracy"]["mean"] == 1.0
