"""LLM client skeleton module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LLMRequest:
    """Data required to make an LLM call."""

    prompt: str
    parameters: Dict[str, Any] | None = None


class LLMClient:
    """Placeholder large language model client."""

    def complete(self, request: LLMRequest) -> str:
        """Return a dummy completion string."""
        _ = request
        # TODO: Integrate with the configured LLM provider.
        return "[llm response placeholder]"


def get_llm_client() -> LLMClient:
    """Factory for the LLM client."""
    return LLMClient()


__all__ = ["LLMClient", "LLMRequest", "get_llm_client"]
