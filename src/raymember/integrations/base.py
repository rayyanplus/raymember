"""Abstract base class for model-agnostic agent integrations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from raymember.sdk import Raymember


class BaseAgentIntegration(ABC):
    """Abstract agent wrapper retrieving Raymember context before model execution."""

    def __init__(self, memory: Raymember):
        self.memory = memory

    @abstractmethod
    def run(self, user_input: str, max_context_items: int = 10, max_context_chars: int = 4000) -> str:
        """Runs user input through memory retrieval, context injection, and model generation."""
        pass
