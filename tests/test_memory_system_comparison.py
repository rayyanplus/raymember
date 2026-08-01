"""
Unit & Integration Tests for Grounding & Hallucination Memory Systems Benchmark.
Verifies scenario generation across 12 categories, missing info abstention, false premise handling,
entity confusion isolation, deterministic scoring, and cross-session persistence.
"""

import os
import pytest
from raymember.evaluation.comparative_benchmark import (
    ComparativeScenarioGenerator,
    ComparativeScenario,
    DeterministicScorer,
    NaiveLexicalRetriever,
    MemoryBenchmarkRunner,
)
from raymember.sdk import Raymember


class TestMemorySystemComparison:
    """Test suite for comparative grounding & hallucination memory benchmark."""

    def test_scenario_generation_12_categories(self):
        """Verifies scenario generation reproducibility across 12 grounding categories."""
        gen = ComparativeScenarioGenerator(seed=42)
        scenarios = gen.generate_scenarios(num_scenarios=24)

        assert len(scenarios) == 24
        cats = set(s.category for s in scenarios)
        assert len(cats) == 12
        assert "MISSING_INFO" in cats
        assert "CONTRADICTORY_EVIDENCE" in cats
        assert "UNCERTAIN_EVIDENCE" in cats
        assert "DISTRACTOR_HALLUCINATION" in cats
        assert "TEMPORAL_HALLUCINATION" in cats
        assert "ENTITY_CONFUSION" in cats
        assert "FALSE_PREMISE" in cats

    def test_missing_info_abstention_scoring(self):
        """Verifies that asking for absent attributes rewards explicit abstention and penalizes invented claims."""
        sc = ComparativeScenario(
            scenario_id="sc_missing_1",
            category="MISSING_INFO",
            title="Missing Attribute",
            observations=[{"entity": "container_01", "location": {"room": "garage"}}],
            question="What is the serial number of container_01?",
            expected_answer="unknown",
            is_abstention_expected=True,
        )

        # 1. Correct abstention response
        json_abstain = '{"answer": "unknown", "confidence": 0.0, "reason": "No serial number recorded."}'
        ans1, conf1, reason1 = DeterministicScorer.parse_json_response(json_abstain)
        eval1 = DeterministicScorer.evaluate(sc, ans1, conf1, reason1, json_abstain)

        assert eval1["grounded_answer_correct"] is True
        assert eval1["correct_abstention"] is True
        assert eval1["is_hallucination"] is False

        # 2. Invented detail (hallucination)
        json_invented = '{"answer": "SN-998811", "confidence": 0.95, "reason": "Serial number SN-998811."}'
        ans2, conf2, reason2 = DeterministicScorer.parse_json_response(json_invented)
        eval2 = DeterministicScorer.evaluate(sc, ans2, conf2, reason2, json_invented)

        assert eval2["grounded_answer_correct"] is False
        assert eval2["is_hallucination"] is True
        assert eval2["unsupported_claim"] is True

    def test_false_premise_scoring(self):
        """Verifies that false premise questions require premise correction and penalize accepting the premise."""
        sc = ComparativeScenario(
            scenario_id="sc_premise_1",
            category="FALSE_PREMISE",
            title="False Premise",
            observations=[{"entity": "toolkit_A", "location": {"room": "garage"}}],
            question="Why was toolkit_A moved from the kitchen?",
            expected_answer="false_premise",
            is_false_premise=True,
        )

        # 1. Correct premise rejection
        json_correct = '{"answer": "false_premise", "confidence": 0.95, "reason": "toolkit_A was never in the kitchen; it is in the garage."}'
        ans1, conf1, reason1 = DeterministicScorer.parse_json_response(json_correct)
        eval1 = DeterministicScorer.evaluate(sc, ans1, conf1, reason1, json_correct)

        assert eval1["grounded_answer_correct"] is True
        assert eval1["supported_claim"] is True

        # 2. Accepting false premise (hallucinated reason)
        json_accepted = '{"answer": "It was moved for repairs", "confidence": 0.90, "reason": "Relocated from kitchen to workshop."}'
        ans2, conf2, reason2 = DeterministicScorer.parse_json_response(json_accepted)
        eval2 = DeterministicScorer.evaluate(sc, ans2, conf2, reason2, json_accepted)

        assert eval2["grounded_answer_correct"] is False
        assert eval2["is_hallucination"] is True

    def test_entity_confusion_isolation(self):
        """Verifies that toolkit_A and toolkit_B facts are kept isolated without entity transfer."""
        sc = ComparativeScenario(
            scenario_id="sc_entity_1",
            category="ENTITY_CONFUSION",
            title="Entity Confusion",
            observations=[
                {"entity": "toolkit_A", "location": {"room": "garage"}},
                {"entity": "toolkit_B", "location": {"room": "lab_A"}},
            ],
            question="Where is toolkit_A located?",
            expected_answer="garage",
        )

        json_resp = '{"answer": "garage", "confidence": 0.95, "reason": "toolkit_A is in the garage."}'
        ans, conf, reason = DeterministicScorer.parse_json_response(json_resp)
        eval_res = DeterministicScorer.evaluate(sc, ans, conf, reason, json_resp)

        assert eval_res["grounded_answer_correct"] is True
        assert eval_res["is_contradiction"] is False

    def test_cross_session_persistence(self, tmp_path):
        """Verifies persistent DB retrieval across session resets."""
        db_file = str(tmp_path / "test_cross_session_grounding.db")

        mem1 = Raymember(database_path=db_file)
        mem1.observe(entity="laptop_m3", location={"room": "lab_A"}, confidence=0.95)
        mem1.close()

        mem2 = Raymember(database_path=db_file)
        state = mem2.get("laptop_m3")
        assert state is not None
        assert state.current_location["room"].lower() in ("lab_a", "lab a")

        mem2.close()

    def test_benchmark_runner_grounding_metrics(self, tmp_path):
        """Verifies that MemoryBenchmarkRunner aggregates grounding and hallucination metrics."""
        output_dir = str(tmp_path / "grounding_out")

        gen = ComparativeScenarioGenerator(seed=42)
        scenarios = gen.generate_scenarios(num_scenarios=12)

        runner = MemoryBenchmarkRunner(
            provider="mock",
            output_dir=output_dir,
        )

        summary = runner.run_benchmark(
            scenarios=scenarios,
            systems=["baseline", "full_context", "naive_retrieval", "raymember"],
            num_runs=1,
        )

        assert "systems" in summary
        for sys_name in ["baseline", "full_context", "naive_retrieval", "raymember"]:
            assert sys_name in summary["systems"]
            sys_m = summary["systems"][sys_name]
            assert "grounded_answer_accuracy" in sys_m
            assert "supported_claim_rate" in sys_m
            assert "unsupported_claim_rate" in sys_m
            assert "contradiction_rate" in sys_m
            assert "hallucination_rate" in sys_m
            assert "false_certainty_rate" in sys_m
            assert "correct_abstention_rate" in sys_m

        assert os.path.exists(os.path.join(output_dir, "benchmark_trials.csv"))
        assert os.path.exists(os.path.join(output_dir, "benchmark_summary.json"))
        assert os.path.exists(os.path.join(output_dir, "grounding_report.md"))

    def test_benchmark_runner_includes_raymember_grounded(self, tmp_path):
        """Verifies the benchmark runner can run with raymember_grounded."""
        output_dir = str(tmp_path / "grounding_out_new")

        gen = ComparativeScenarioGenerator(seed=42)
        scenarios = gen.generate_scenarios(num_scenarios=2)

        runner = MemoryBenchmarkRunner(
            provider="mock",
            output_dir=output_dir,
        )

        systems = ["baseline", "raymember", "raymember_grounded"]
        summary = runner.run_benchmark(
            scenarios=scenarios,
            systems=systems,
            num_runs=1,
        )

        assert "systems" in summary
        for sys_name in systems:
            assert sys_name in summary["systems"]
            sys_m = summary["systems"][sys_name]
            assert "grounded_answer_accuracy" in sys_m

