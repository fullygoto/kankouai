"""Pamphlet search index skeleton."""
from __future__ import annotations

from typing import Any, Iterable, List


class PamphletIndex:
    """Placeholder in-memory index for pamphlets."""

    def search(self, query: str) -> List[Any]:
        """Return dummy search results for pamphlets."""
        _ = query
        # TODO: Replace with actual search implementation.
        return []


def load_pamphlet_index(_: Iterable[Any] | None = None) -> PamphletIndex:
    """Factory for pamphlet index instances."""
    # TODO: Load real data once the refactor lands.
    return PamphletIndex()


__all__ = ["PamphletIndex", "load_pamphlet_index"]
