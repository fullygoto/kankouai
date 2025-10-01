"""Helpers for query length thresholds shared across the app."""
from __future__ import annotations

from coreapp import config as cfg

from .normalize import normalize_text


def min_query_chars() -> int:
    """Return the effective minimum number of characters required for entries search."""

    base = getattr(cfg, "MIN_QUERY_CHARS", 2)
    if hasattr(cfg, "ENABLE_ENTRIES_2CHAR") and not getattr(cfg, "ENABLE_ENTRIES_2CHAR"):
        return max(base, 3)
    return base


def normalize_for_length(text: str | None) -> str:
    """Normalise text for length comparison."""

    return normalize_text(text or "")


def is_too_short(text: str | None) -> bool:
    """Return ``True`` when ``text`` is shorter than the allowed threshold."""

    return len(normalize_for_length(text)) < min_query_chars()


__all__ = ["is_too_short", "min_query_chars", "normalize_for_length"]
