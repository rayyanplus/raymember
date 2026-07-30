"""OpenAI-compatible chat completion agent wrapper."""

from typing import Any, Callable, Dict, List, Optional
from raymember.integrations.base import BaseAgentIntegration
from raymember.sdk import Raymember


class OpenAICompatibleAgent(BaseAgentIntegration):
    """Adapter for chat completion callables accepting a list of message dicts."""

    def __init__(
        self,
        memory: Raymember,
        model_chat_callable: Callable[[List[Dict[str, str]]], str],
        system_prompt: Optional[str] = None,
    ):
        super().__init__(memory)
        self.model_chat = model_chat_callable
        self.system_prompt = system_prompt or "You are an AI agent with persistent world memory."

    def run(
        self,
        user_input: str,
        max_context_items: int = 10,
        max_context_chars: int = 4000,
    ) -> str:
        context_str = self.memory.context(query=user_input, max_items=max_context_items, max_characters=max_context_chars)

        messages = [
            {"role": "system", "content": f"{self.system_prompt}\n\n{context_str}"},
            {"role": "user", "content": user_input},
        ]

        return self.model_chat(messages)
