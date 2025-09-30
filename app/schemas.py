"""Shared schemas for the responder skeleton."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class ConversationContext:
    """Placeholder schema for conversation state."""

    user_id: str
    metadata: Dict[str, Any] | None = None


@dataclass(slots=True)
class ResponderOutput:
    """Placeholder schema for responder outputs."""

    message: str
    intent: str


__all__ = ["ConversationContext", "ResponderOutput"]
