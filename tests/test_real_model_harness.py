"""
Unit tests for Phase 7 Real Model Evaluation Harness and Developer Demo.
Executes 100% offline without network access or API keys.
"""

import pytest
from raymember.evaluation.harness import (
    ModelHarness,
    DeterministicEvaluatorModel,
    OpenAICompatibleAdapter,
    OllamaAdapter,
    AnthropicAdapter,
)
from raymember.evaluation.agent_comparison import AgentComparisonBenchmark, EvaluationScenario


class TestPhase7ModelHarness:
    """Test suite verifying ModelHarness factory and adapter configuration."""

    def test_harness_returns_mock_model_by_default(self):
        """Verifies that ModelHarness defaults to DeterministicEvaluatorModel when no provider is specified."""
        model_fn, model_name, is_real = ModelHarness.get_model()
        assert isinstance(model_fn, DeterministicEvaluatorModel)
        assert model_name == "DeterministicOfflineModel"
        assert is_real is False

    def test_harness_configures_openai_adapter(self, monkeypatch):
        """Verifies OpenAI adapter initialization via env vars without network calls."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

        model_fn, model_name, is_real = ModelHarness.get_model(provider="openai")
        assert isinstance(model_fn, OpenAICompatibleAdapter)
        assert "gpt-4o-mini" in model_name
        assert is_real is True

    def test_harness_configures_ollama_adapter(self, monkeypatch):
        """Verifies Ollama adapter initialization via env vars without network calls."""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("OLLAMA_MODEL", "llama3")

        model_fn, model_name, is_real = ModelHarness.get_model(provider="ollama")
        assert isinstance(model_fn, OllamaAdapter)
        assert "llama3" in model_name
        assert is_real is True

    def test_harness_configures_anthropic_adapter(self, monkeypatch):
        """Verifies Anthropic adapter initialization via env vars without network calls."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key")

        model_fn, model_name, is_real = ModelHarness.get_model(provider="anthropic")
        assert isinstance(model_fn, AnthropicAdapter)
        assert "Anthropic" in model_name
        assert is_real is True

    def test_benchmark_contains_32_scenarios(self):
        """Verifies that AgentComparisonBenchmark builds 32 default scenarios covering 12 categories."""
        bm = AgentComparisonBenchmark()
        assert len(bm.scenarios) == 32

        categories = set(s.category for s in bm.scenarios)
        assert len(categories) == 12
        assert "conflicting_values" in categories
        assert "stale_information" in categories
        assert "partial_updates" in categories
        assert "multi_attribute" in categories
        assert "multiple_similar_entities" in categories
        assert "delayed_observations" in categories
        assert "provenance_conflicts" in categories
        assert "changing_ownership" in categories
        assert "logistics_state" in categories
        assert "customer_support_state" in categories
        assert "task_agent_state" in categories
        assert "robotics_world_state" in categories

    def test_benchmark_executes_repeated_runs_with_stats(self):
        """Verifies statistical mean, std dev, and confidence interval metrics computation."""
        bm = AgentComparisonBenchmark()
        summary = bm.run_benchmark(num_runs=2)

        assert summary["evaluation_metadata"]["total_scenarios"] == 32
        assert summary["evaluation_metadata"]["num_runs"] == 2

        strat_c = summary["strategies"]["Strategy C (Raymember)"]
        assert "mean" in strat_c["accepted_state_accuracy"]
        assert "std_dev" in strat_c["accepted_state_accuracy"]
        assert "ci_95_margin" in strat_c["accepted_state_accuracy"]
        assert strat_c["conflict_interpretation_accuracy"]["mean"] == 1.0
