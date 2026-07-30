"""Generic callable model agent wrapper."""

from typing import Any, Callable, Dict, Optional
from raymember.integrations.base import BaseAgentIntegration
from raymember.sdk import Raymember


class MemoryAgent(BaseAgentIntegration):
    """
    Model-agnostic agent integration wrapper.
    Accepts any callable model function `model(prompt: str) -> str`.
    Requires zero cloud dependencies or external API keys.
    """

    def __init__(
        self,
        memory: Raymember,
        model: Callable[[str], str],
        system_prompt: Optional[str] = None,
    ):
        super().__init__(memory)
        self.model = model
        self.system_prompt = system_prompt or (
            "You are a helpful physical world assistant equipped with persistent world memory. "
            "Use the provided RAYMEMBER WORLD CONTEXT to answer the user accurately."
        )

    def run(
        self,
        user_input: str,
        max_context_items: int = 10,
        max_context_chars: int = 4000,
    ) -> str:
        # 1. Retrieve ranked context from Raymember
        context_str = self.memory.context(query=user_input, max_items=max_context_items, max_characters=max_context_chars)

        # 2. Construct model prompt payload
        prompt = (
            f"{self.system_prompt}\n\n"
            f"=== WORLD MEMORY CONTEXT ===\n"
            f"{context_str}\n\n"
            f"=== USER QUERY ===\n"
            f"{user_input}\n"
        )

        # 3. Call model function
        response = self.model(prompt)
        return response
