"""
Unit & Integration Tests for Raymember LLM Connection Layer.
"""

import pytest
from raymember import Raymember, connect_llm
from raymember.integrations.llm import RaymemberLLMAgent
from raymember.evaluation.harness import (
    AntigravityAdapter,
    GeminiAdapter,
    OpenAICompatibleAdapter,
    OllamaAdapter,
    AnthropicAdapter,
    DeterministicEvaluatorModel,
    ModelHarness,
)


class TestLLMIntegration:
    """Test suite verifying LLM connection factory, provider adapters, context injection, and state write-backs."""

    def test_connect_llm_antigravity_default(self, tmp_path):
        """Verifies connecting Antigravity AI assistant adapter to Raymember."""
        db = str(tmp_path / "test_antigravity.db")
        mem = Raymember(database_path=db)
        agent = connect_llm(memory=mem, provider="antigravity")

        assert isinstance(agent, RaymemberLLMAgent)
        assert "Antigravity" in agent.provider_name
        mem.close()

    def test_connect_llm_all_providers(self, tmp_path, monkeypatch):
        """Verifies factory creation across all supported providers without network calls."""
        db = str(tmp_path / "test_providers.db")
        mem = Raymember(database_path=db)

        # Gemini
        agent_gemini = connect_llm(memory=mem, provider="gemini", api_key="fake_key")
        assert isinstance(agent_gemini.model, GeminiAdapter)

        # OpenAI
        agent_openai = connect_llm(memory=mem, provider="openai", api_key="fake_key")
        assert isinstance(agent_openai.model, OpenAICompatibleAdapter)

        # Ollama
        agent_ollama = connect_llm(memory=mem, provider="ollama")
        assert isinstance(agent_ollama.model, OllamaAdapter)

        # Anthropic
        agent_anthropic = connect_llm(memory=mem, provider="anthropic", api_key="fake_key")
        assert isinstance(agent_anthropic.model, AnthropicAdapter)

        # Mock
        agent_mock = connect_llm(memory=mem, provider="mock")
        assert isinstance(agent_mock.model, DeterministicEvaluatorModel)

        mem.close()

    def test_llm_agent_ask_retrieves_context(self, tmp_path):
        """Verifies that agent.ask retrieves context and formats prompt properly."""
        db = str(tmp_path / "test_ask.db")
        mem = Raymember(database_path=db)
        mem.observe(entity="toolkit", location={"room": "garage"}, confidence=0.9)

        agent = connect_llm(memory=mem, provider="antigravity")
        resp = agent.ask("Where is the toolkit?")

        assert "garage" in resp.lower() or "toolkit" in resp.lower()
        mem.close()

    def test_llm_agent_run_and_remember_closed_loop(self, tmp_path):
        """Verifies closed-loop state extraction and automatic persistence to Raymember memory."""
        db = str(tmp_path / "test_closed_loop.db")
        mem = Raymember(database_path=db)

        # Initial state
        mem.observe(entity="toolkit", location={"room": "garage"}, confidence=0.9)

        agent = connect_llm(memory=mem, provider="antigravity")

        # Statement indicating toolkit was moved
        statement = "The toolkit was moved from the garage to the workshop."
        answer, obs_record = agent.run_and_remember(statement)

        assert obs_record is not None
        assert obs_record["entity"] == "toolkit"
        assert obs_record["room"] == "workshop"

        # Verify state updated in Raymember
        state = mem.get("toolkit")
        assert state is not None
        assert state.current_location["room"] == "workshop"

        mem.close()
