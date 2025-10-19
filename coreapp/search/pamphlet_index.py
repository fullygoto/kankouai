"""Lightweight pamphlet search index used by the responder layer."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Sequence


_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z一-龥ぁ-んァ-ンー]+")


def _tokenize(text: str) -> List[str]:
    """Return lowercase tokens suitable for a simple term match."""

    return [token.lower() for token in _TOKEN_PATTERN.findall(text or "")]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _split_fragments(text: str) -> List[str]:
    """Split pamphlet text into short fragments preserving sentence order."""

    normalized = _clean_text(text)
    if not normalized:
        return []

    # First try to split by Japanese sentence terminators and full stops.
    sentences = [seg.strip() for seg in re.split(r"(?<=[。．！？!?])", normalized) if seg.strip()]
    if not sentences:
        return [normalized]

    fragments: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        sent_len = len(sentence)
        if current and current_len + sent_len > 160:
            fragments.append("".join(current))
            current = [sentence]
            current_len = sent_len
            continue
        current.append(sentence)
        current_len += sent_len
    if current:
        fragments.append("".join(current))

    return fragments or [normalized]


@dataclass(frozen=True)
class PamphletFragment:
    """A short span of text used as retrieval evidence."""

    city: str
    source: str
    text: str
    score: float


class PamphletIndex:
    """In-memory index that scores fragments using simple token overlap."""

    def __init__(self, pamphlets: Iterable[Dict[str, str]] | None = None) -> None:
        self._fragments: Dict[str, List[Dict[str, object]]] = {}
        self._cache: Dict[tuple[str, str, int], List[PamphletFragment]] = {}
        if pamphlets:
            self.extend(pamphlets)

    def extend(self, pamphlets: Iterable[Dict[str, str]]) -> None:
        for item in pamphlets:
            if not isinstance(item, dict):
                continue
            city = _clean_text(str(item.get("city", "")))
            text = str(item.get("text", ""))
            source = _clean_text(str(item.get("source") or item.get("title") or "")) or "pamphlet"
            if not city or not text.strip():
                continue
            fragments = _split_fragments(text)
            bucket = self._fragments.setdefault(city, [])
            for fragment in fragments:
                tokens = _tokenize(fragment)
                if not tokens:
                    continue
                bucket.append(
                    {
                        "text": fragment,
                        "tokens": tokens,
                        "source": source,
                    }
                )

    def search(self, city: str, query: str, top_k: int = 3) -> List[PamphletFragment]:
        """Return the highest scoring fragments for *city* and *query*."""

        key = (city, query, top_k)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        items = self._fragments.get(city)
        if not items or top_k <= 0:
            self._cache[key] = []
            return []

        tokens = _tokenize(query)
        if not tokens:
            # Fall back to the first few fragments for empty queries.
            trimmed = self._deduplicate(items)[:top_k]
            result = [
                PamphletFragment(
                    city=city,
                    source=str(entry["source"]),
                    text=str(entry["text"]),
                    score=0.0,
                )
                for entry in trimmed
            ]
            self._cache[key] = result
            return result

        scored: List[PamphletFragment] = []
        for entry in items:
            entry_tokens: Sequence[str] = entry["tokens"]  # type: ignore[assignment]
            overlap = sum(1 for token in tokens if token in entry_tokens)
            if overlap <= 0:
                continue
            scored.append(
                PamphletFragment(
                    city=city,
                    source=str(entry["source"]),
                    text=str(entry["text"]),
                    score=float(overlap),
                )
            )

        if not scored:
            trimmed = self._deduplicate(items)[:top_k]
            result = [
                PamphletFragment(
                    city=city,
                    source=str(entry["source"]),
                    text=str(entry["text"]),
                    score=0.0,
                )
                for entry in trimmed
            ]
            self._cache[key] = result
            return result

        deduped: List[PamphletFragment] = []
        seen: set[str] = set()
        for fragment in sorted(scored, key=lambda item: (-item.score, -len(item.text))):
            signature = fragment.text
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append(fragment)
            if len(deduped) >= top_k:
                break

        self._cache[key] = deduped
        return deduped

    @staticmethod
    def _deduplicate(items: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        seen: set[str] = set()
        unique: List[Dict[str, object]] = []
        for entry in items:
            text = str(entry["text"])
            if text in seen:
                continue
            seen.add(text)
            unique.append(entry)
        return unique


def load_pamphlet_index(pamphlets: Iterable[Dict[str, str]] | None = None) -> PamphletIndex:
    """Factory that initialises :class:`PamphletIndex` with *pamphlets*."""

    return PamphletIndex(pamphlets)


__all__ = ["PamphletFragment", "PamphletIndex", "load_pamphlet_index"]
