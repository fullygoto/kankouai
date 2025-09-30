"""Intent detection skeleton module."""
from __future__ import annotations

from typing import Any, Dict


class IntentDetector:
    """Placeholder intent detector.

    TODO: Replace with real intent classification logic when the responder
    refactor is implemented.
    """

    def detect(self, message: str, *, context: Dict[str, Any] | None = None) -> str:
        """Return a dummy intent label for the given message."""
        _ = (message, context)
        return "fallback"


def get_intent_detector() -> IntentDetector:
    """Factory for the intent detector."""
    return IntentDetector()


__all__ = ["IntentDetector", "get_intent_detector"]
