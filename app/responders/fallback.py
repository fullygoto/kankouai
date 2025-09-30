"""Fallback responder skeleton."""
from __future__ import annotations

from typing import Any, Dict


class FallbackResponder:
    """Placeholder responder for unmatched intents."""

    def respond(self, message: str, *, context: Dict[str, Any] | None = None) -> str:
        """Return a dummy fallback response."""
        _ = (message, context)
        # TODO: Implement graceful fallback behavior.
        return "[fallback response placeholder]"


__all__ = ["FallbackResponder"]
