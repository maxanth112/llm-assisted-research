"""Base classes and interfaces for model providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Completion:
    """Represents a completion response from a model provider."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    model_version: str
    latency_ms: float
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelProvider(ABC):
    """Abstract base class for model providers."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        params: dict[str, Any]
    ) -> Completion:
        """
        Generate a completion for the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            params: Generation parameters (temperature, max_tokens, etc.)

        Returns:
            Completion object with response and metadata
        """
        pass

    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the name of this provider.

        Returns:
            Provider name string (e.g., 'openai', 'mock')
        """
        pass

    @abstractmethod
    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Estimate the cost in USD for a completion.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        pass
