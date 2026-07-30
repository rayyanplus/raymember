"""Phase 3 Unit and Integration Test Suite."""

import os
import pytest
import numpy as np

from raymember.evaluation.diagnostics import (
    run_calibration_analysis,
    run_error_taxonomy,
    run_hybrid_policy_audit,
    run_hybrid_sensitivity_test,
)
from raymember.evaluation.metrics import BenchmarkRunner
from raymember.learning.dataset import DatasetGenerator
from raymember.learning.policy import HybridPolicy, LearnedUpdatePolicy
from raymember.schemas import Location, ObservationInput
from raymember.simulation.world import SimulationWorld


def test_hybrid_policy_independence_and_audit(tmp_path):
    audit_file = str(tmp_path / "audit.json")
    audit = run_hybrid_policy_audit(random_seed=42, output_path=audit_file)

    assert os.path.exists(audit_file)
    assert audit["policy_independence_confirmed"] is True
    assert audit["hybrid_predict_trust_weight_called"] is True
    assert len(audit["execution_trace"]) > 0


def test_hybrid_sensitivity_monotonicity(tmp_path):
    sens_file = str(tmp_path / "sens.json")
    sens = run_hybrid_sensitivity_test(output_path=sens_file)

    assert os.path.exists(sens_file)
    test_rows = sens["sensitivity_test"]
    assert len(test_rows) == 5

    # Monotonicity check: higher trust weight -> higher effective evidence weight
    eff_weights = [r["effective_evidence_weight"] for r in test_rows]
    for i in range(len(eff_weights) - 1):
        assert eff_weights[i] <= eff_weights[i + 1]


def test_deterministic_new_entity_handling_5_classes():
    ds = DatasetGenerator(random_seed=42)
    assert len(ds.ACTION_MAP) == 5
    assert "NEW_ENTITY" not in ds.ACTION_MAP


def test_scenario_families_a_to_h():
    world = SimulationWorld(random_seed=42)
    families = [
        "repeated_movement",
        "long_gaps",
        "multiple_similar_entities",
        "dynamic_source_reliability",
        "stale_observation_bursts",
        "partial_observations",
        "ambiguous_identity",
        "adversarial_mixed_noise",
    ]

    for fam in families:
        sc = world.generate_scenario(f"sc_{fam}", num_steps=5, scenario_family=fam)
        assert len(sc.steps) == 5


def test_ood_scenario_generation():
    world = SimulationWorld(random_seed=42)
    ood_sc = world.generate_ood_scenario("ood_sc_01", num_steps=10)
    assert ood_sc.noise_condition == "ood_extreme"
    assert len(ood_sc.steps) == 10


def test_calibration_and_error_taxonomy(tmp_path):
    cal_file = str(tmp_path / "cal.json")
    err_file = str(tmp_path / "err.json")

    cal = run_calibration_analysis(random_seed=42, output_path=cal_file)
    err = run_error_taxonomy(random_seed=42, output_path=err_file)

    assert os.path.exists(cal_file)
    assert os.path.exists(err_file)
    assert "brier_score" in cal
    assert "expected_calibration_error_ece" in cal
    assert "error_counts" in err


def test_baseline_fairness_and_no_ground_truth_leakage():
    runner = BenchmarkRunner(random_seed=42, dataset_size="small")
    summary = runner.run_multi_seed_benchmark(seeds=[42])
    assert "mixed" in summary
    assert "ood_extreme" in summary
