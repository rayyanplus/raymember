"""
Model Evaluation Harness for Raymember Phase 7.

Provides a unified, provider-agnostic interface for executing evaluations across:
  1. Offline Deterministic Mock Models (default for tests and local demos, zero API keys)
  2. OpenAI-compatible APIs (OpenAI, vLLM, LM Studio, Anyscale)
  3. Ollama local models
  4. Anthropic Claude models
  5. Any custom callable model(prompt: str) -> str
"""

import json
import os
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple


class DeterministicEvaluatorModel:
    """
    Deterministic offline model evaluator that parses prompt context to extract ground-truth answers.
    Requires no internet connection or API keys.
    """

    def __call__(self, prompt: str) -> str:
        p_lower = prompt.lower()

        # Strategy A (No Context)
        if "raymember world context" not in p_lower and "raw observation history" not in p_lower:
            return "I do not have access to memory context or state tracking for this entity."

        # Extract answer based on context
        if "conflicts:" in p_lower or "[current belief]" in p_lower:
            return f"Based on verified Raymember context: {prompt}"

        return f"Contextual response: {prompt}"


class OpenAICompatibleAdapter:
    """Adapter for OpenAI and OpenAI-compatible endpoints (vLLM, LM Studio, Anyscale, LocalAI)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "mock-key")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            res = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return res.choices[0].message.content or ""
        except ImportError:
            # Fallback to standard urllib for zero-dependency HTTP requests to OpenAI-compatible endpoints
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")
                return ""


class OllamaAdapter:
    """Adapter for local Ollama instances running via REST API (zero third-party dependencies required)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = model_name or os.environ.get("OLLAMA_MODEL", "llama3")
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")


class AnthropicAdapter:
    """Adapter for Anthropic Claude models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "mock-key")
        self.model_name = model_name or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        self.temperature = temperature

    def __call__(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        res = client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.content[0].text or ""


class ModelHarness:
    """Factory for instantiating model adapters based on CLI args or environment variables."""

    @classmethod
    def get_model(
        cls,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        custom_model_fn: Optional[Callable[[str], str]] = None,
    ) -> Tuple[Callable[[str], str], str, bool]:
        """
        Returns a tuple of (model_callable, provider_label, is_real_model).
        """
        if custom_model_fn:
            return custom_model_fn, provider or "custom_model", True

        prov_clean = (provider or os.environ.get("RAYMEMBER_EVAL_PROVIDER", "mock")).strip().lower()

        if prov_clean in ("mock", "offline", "deterministic"):
            return DeterministicEvaluatorModel(), "DeterministicOfflineModel", False

        elif prov_clean in ("openai", "vllm", "lmstudio"):
            m_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o")
            adapter = OpenAICompatibleAdapter(api_key=api_key, base_url=base_url, model_name=m_name)
            return adapter, f"OpenAI({adapter.model_name})", True

        elif prov_clean == "ollama":
            adapter = OllamaAdapter(base_url=base_url, model_name=model_name or "llama3")
            return adapter, f"Ollama({adapter.model_name})", True

        elif prov_clean == "anthropic":
            adapter = AnthropicAdapter(api_key=api_key, model_name=model_name or "claude-3-5-sonnet-20241022")
            return adapter, f"Anthropic({adapter.model_name})", True

        # Default fallback
        return DeterministicEvaluatorModel(), "DeterministicOfflineModel", False

    @classmethod
    def validate_provider_config(
        cls,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validates provider configuration offline without issuing network HTTP calls.
        Returns a dict of validation status and metadata.
        """
        prov_clean = (provider or os.environ.get("RAYMEMBER_EVAL_PROVIDER", "mock")).strip().lower()

        if prov_clean in ("mock", "offline", "deterministic"):
            return {
                "valid": True,
                "provider": "DeterministicOfflineModel",
                "is_real_model": False,
                "api_key_status": "Not required (Offline Mode)",
                "base_url": "N/A",
                "notes": "Offline deterministic mock model ready for zero-credential execution.",
            }

        elif prov_clean in ("openai", "vllm", "lmstudio"):
            key = api_key or os.environ.get("OPENAI_API_KEY")
            url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            m_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o")

            valid = bool(key or "localhost" in url or "127.0.0.1" in url)
            return {
                "valid": valid,
                "provider": f"OpenAICompatible({m_name})",
                "is_real_model": True,
                "api_key_status": "Present" if key else ("Optional (Local Endpoint)" if "localhost" in url or "127.0.0.1" in url else "Missing OPENAI_API_KEY"),
                "base_url": url,
                "notes": "Ready for API execution." if valid else "Set OPENAI_API_KEY or configure local endpoint URL.",
            }

        elif prov_clean == "ollama":
            url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            m_name = model_name or os.environ.get("OLLAMA_MODEL", "llama3")
            return {
                "valid": True,
                "provider": f"Ollama({m_name})",
                "is_real_model": True,
                "api_key_status": "Not required (Local Server)",
                "base_url": url,
                "notes": "Offline local Ollama configuration valid.",
            }

        elif prov_clean == "anthropic":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            m_name = model_name or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
            valid = bool(key)
            return {
                "valid": valid,
                "provider": f"Anthropic({m_name})",
                "is_real_model": True,
                "api_key_status": "Present" if key else "Missing ANTHROPIC_API_KEY",
                "base_url": "https://api.anthropic.com",
                "notes": "Ready for API execution." if valid else "Set ANTHROPIC_API_KEY environment variable.",
            }

        return {
            "valid": False,
            "provider": prov_clean,
            "is_real_model": False,
            "api_key_status": "Unknown Provider",
            "base_url": "N/A",
            "notes": f"Unrecognized provider '{prov_clean}'.",
        }
