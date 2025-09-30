"""Text normalization helpers for search modules."""
from __future__ import annotations

from typing import Iterable


def normalize_query(tokens: Iterable[str]) -> list[str]:
    """Return a normalized token list.

    TODO: Implement proper normalization once search is refactored.
    """

    return [token.lower() for token in tokens]


__all__ = ["normalize_query"]
