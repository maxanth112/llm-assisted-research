"""Model provider interfaces and implementations."""

from harness.providers.base import ModelProvider, Completion
from harness.providers.mock_adapter import MockAdapter
from harness.providers.openai_compat import OpenAICompatAdapter

__all__ = [
    "ModelProvider",
    "Completion",
    "MockAdapter",
    "OpenAICompatAdapter",
]
