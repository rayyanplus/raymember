"""Unit tests for benchmark evaluation harness and metrics calculation."""

import os
import pytest
from raymember.evaluation.metrics import BenchmarkRunner, SystemMetrics


def test_benchmark_runner_execution(tmp_path):
    out_json = str(tmp_path / "benchmark_test.json")
    sum_json = str(tmp_path / "summary_test.json")
    runner = BenchmarkRunner(random_seed=42)
    summary = runner.run_multi_seed_benchmark(seeds=[42], multi_seed_output=out_json, summary_output=sum_json)

    assert os.path.exists(out_json)
    assert os.path.exists(sum_json)
    assert "clean" in summary
    assert "mixed" in summary

    clean_metrics = summary["clean"]
    assert "1_latest_observation" in clean_metrics
    assert "4_raymember_learned_policy" in clean_metrics
    assert "5_raymember_hybrid_policy" in clean_metrics

    m = clean_metrics["4_raymember_learned_policy"]
    assert "mean_accuracy" in m
    assert "mean_macro_f1" in m
