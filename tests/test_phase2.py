"""Phase 2 Unit & Integration Test Suite."""

import os
import pytest
import numpy as np

from raymember.evaluation.diagnostics import run_dataset_diagnostics, run_model_diagnostics
from raymember.evaluation.metrics import BenchmarkRunner
from raymember.learning.dataset import DatasetGenerator
from raymember.learning.features import FeatureExtractor
from raymember.learning.policy import HybridPolicy, LearnedUpdatePolicy
from raymember.schemas import Location, ObservationInput


def test_counterfactual_label_generation():
    ds = DatasetGenerator(random_seed=42)
    (X_tr, y_act, y_tr), _, _, _ = ds.generate_split_dataset(num_scenarios=5, steps_per_scenario=5)

    assert len(X_tr) > 0
    assert len(y_act) == len(X_tr)
    assert len(y_tr) == len(X_tr)

    assert np.all((y_tr >= 0.0) & (y_tr <= 1.0))


def test_zero_ground_truth_leakage_during_inference():
    extractor = FeatureExtractor()
    obs = ObservationInput(entity="laptop", location=Location(room="office"), confidence=0.85, source="camera")
    curr_state = {"location": {"room": "bedroom"}, "last_seen": "2026-07-29T10:00:00Z", "confidence": 0.9}

    feat = extractor.extract(curr_state, obs)
    assert len(feat) == 9
    for fname in FeatureExtractor.FEATURE_NAMES:
        assert "ground_truth" not in fname.lower()


def test_scenario_split_disjointness_and_diagnostics(tmp_path):
    diag_file = str(tmp_path / "diag.json")
    diag = run_dataset_diagnostics(random_seed=42, output_path=diag_file)

    assert os.path.exists(diag_file)
    assert diag["split_counts"]["scenarios_disjoint"] is True
    assert "class_imbalance_warning" in diag


def test_confusion_matrix_and_model_diagnostics(tmp_path):
    diag_file = str(tmp_path / "model_diag.json")
    results = run_model_diagnostics(random_seed=42, output_path=diag_file)

    assert os.path.exists(diag_file)
    assert "RandomForestClassifier" in results
    assert "confusion_matrix" in results["RandomForestClassifier"]
    assert "balanced_accuracy" in results["RandomForestClassifier"]


def test_hybrid_policy_trust_weight_and_effective_weight():
    hybrid = HybridPolicy(random_seed=42)
    obs = ObservationInput(entity="backpack", location=Location(room="bedroom"), confidence=0.8, source="camera")

    w_trust = hybrid.predict_trust_weight(None, obs)
    assert 0.0 <= w_trust <= 1.0

    out = hybrid.evaluate_decision(None, obs)
    assert out.obs_confidence == 0.8
    assert out.source_reliability == 0.95
    assert 0.0 <= out.effective_evidence_weight <= 1.0


def test_hybrid_policy_joblib_persistence(tmp_path):
    model_file = str(tmp_path / "hybrid_model.joblib")
    ds = DatasetGenerator(random_seed=42)
    (X_tr, _, y_tr), _, _, _ = ds.generate_split_dataset(num_scenarios=5, steps_per_scenario=5)

    hybrid1 = HybridPolicy(random_seed=42)
    hybrid1.train(X_tr, y_tr)
    hybrid1.save(model_file)

    assert os.path.exists(model_file)

    hybrid2 = HybridPolicy(random_seed=42)
    hybrid2.load(model_file)
    assert hybrid2.is_trained is True


def test_feature_ablations_and_multi_seed_runner(tmp_path):
    summary_file = str(tmp_path / "summary.json")
    multi_file = str(tmp_path / "multi.json")
    ab_file = str(tmp_path / "ablations.json")

    runner = BenchmarkRunner(random_seed=42)
    summary = runner.run_multi_seed_benchmark(seeds=[42], multi_seed_output=multi_file, summary_output=summary_file)
    ablations = runner.run_feature_ablations(output_path=ab_file)

    assert os.path.exists(summary_file)
    assert os.path.exists(ab_file)
    assert "A_all_features" in ablations
    assert "clean" in summary
