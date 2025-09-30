"""Entries search index skeleton."""
from __future__ import annotations

from typing import Any, Iterable, List


class EntriesIndex:
    """Placeholder in-memory index for entries."""

    def search(self, query: str) -> List[Any]:
        """Return dummy search results for entries."""
        _ = query
        # TODO: Replace with actual search implementation.
        return []


def load_entries_index(_: Iterable[Any] | None = None) -> EntriesIndex:
    """Factory for entries index instances."""
    # TODO: Load real data once the refactor lands.
    return EntriesIndex()


__all__ = ["EntriesIndex", "load_entries_index"]
