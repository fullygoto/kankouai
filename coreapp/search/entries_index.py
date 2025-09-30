"""In-memory index for tourism entries."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import json
from collections import Counter

from .normalize import (
    hiragana,
    hiragana_without_diacritics,
    normalize_query,
    normalize_text,
    simplify_kana,
)


@dataclass(slots=True)
class EntryMatch:
    """A single match returned by the index."""

    entry: Dict[str, Any]
    score: float
    matched_tokens: Sequence[str]


def _load_synonyms() -> Dict[str, List[str]]:
    path = Path(__file__).resolve().parents[2] / "data" / "synonyms.json"
    try:
        raw = json.loads(path.read_text("utf-8")) if path.exists() else {}
    except Exception:
        raw = {}

    mapping: Dict[str, set[str]] = {}
    for key, values in raw.items():
        if not isinstance(key, str) or not isinstance(values, list):
            continue
        base_forms = {normalize_text(key)}
        for value in values:
            if isinstance(value, str):
                base_forms.add(normalize_text(value))
        base_forms.discard("")
        if not base_forms:
            continue
        ordered = sorted(base_forms)
        for form in ordered:
            bucket = mapping.setdefault(form, set())
            for other in ordered:
                if other != form:
                    bucket.add(other)
    return {key: sorted(value) for key, value in mapping.items()}


def _expand_tokens(tokens: Sequence[str], synonyms: Dict[str, List[str]]) -> List[List[str]]:
    groups: List[List[str]] = []
    for token in tokens:
        if not token:
            continue
        options: List[str] = [token]
        seen = {token}
        for alt in synonyms.get(token, []):
            if alt and alt not in seen:
                seen.add(alt)
                options.append(alt)
        groups.append(options)
    return groups


def _tokenize(query: str) -> List[str]:
    if not query:
        return []
    raw_tokens = []
    for chunk in str(query).replace("/", " ").replace("・", " ").split():
        raw_tokens.extend(chunk.split("-"))
    return normalize_query(raw_tokens)


class _Document:
    __slots__ = (
        "entry",
        "title_forms",
        "aux_forms",
        "tag_forms",
        "area_forms",
        "score_bias",
    )

    def __init__(self, entry: Dict[str, Any]):
        self.entry = entry
        self.title_forms = self._build_title_forms(entry)
        self.aux_forms = self._build_aux_forms(entry)
        self.tag_forms = self._build_tag_forms(entry)
        self.area_forms = self._build_area_forms(entry)
        title = normalize_text(entry.get("title", ""))
        self.score_bias = max(0.0, 5.0 - len(title) * 0.05)

    @staticmethod
    def _build_title_forms(entry: Dict[str, Any]) -> List[str]:
        forms: List[str] = []
        title = normalize_text(entry.get("title", ""))
        if title:
            forms.append(title)
        for key in ("kana", "yomi", "reading", "ruby", "furigana"):
            value = entry.get(key)
            if value:
                forms.append(normalize_text(value))
                forms.append(hiragana(value))
                forms.append(hiragana_without_diacritics(value))
                forms.append(simplify_kana(value))
        unique: List[str] = []
        seen: set[str] = set()
        for item in forms:
            if item and item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    @staticmethod
    def _build_aux_forms(entry: Dict[str, Any]) -> List[str]:
        aux: List[str] = []
        for key in ("desc", "catch", "address"):
            value = entry.get(key)
            if value:
                aux.append(normalize_text(value))
        for key in ("romaji", "roman", "romanized"):
            value = entry.get(key)
            if value:
                aux.append(normalize_text(value))
        return [item for item in aux if item]

    @staticmethod
    def _build_tag_forms(entry: Dict[str, Any]) -> List[str]:
        tags: List[str] = []
        for tag in entry.get("tags") or []:
            if isinstance(tag, str):
                tags.append(normalize_text(tag))
        return [tag for tag in tags if tag]

    @staticmethod
    def _build_area_forms(entry: Dict[str, Any]) -> List[str]:
        forms: List[str] = []
        for area in entry.get("areas") or []:
            if isinstance(area, str):
                forms.append(normalize_text(area))
        return [area for area in forms if area]

    def matches_filters(self, filters: Dict[str, str] | None) -> bool:
        if not filters:
            return True
        if area := filters.get("area"):
            key = normalize_text(area)
            if key and key not in self.area_forms:
                return False
        if tag := filters.get("tag"):
            key = normalize_text(tag)
            if key and key not in self.tag_forms:
                return False
        return True

    def score(self, token_groups: Sequence[Sequence[str]]) -> tuple[float, List[str]]:
        matched: List[str] = []
        total = 0.0

        for group in token_groups:
            best_score = 0.0
            best_token: str | None = None
            for token in group:
                if not token:
                    continue
                token_score = 0.0
                for title in self.title_forms:
                    pos = title.find(token)
                    if pos < 0:
                        continue
                    if title == token:
                        token_score = max(token_score, 6.0)
                    elif pos == 0:
                        token_score = max(token_score, 5.0)
                    elif pos + len(token) == len(title):
                        token_score = max(token_score, 4.5)
                    else:
                        token_score = max(token_score, 4.0)
                if token_score == 0.0:
                    for aux in self.aux_forms:
                        if token in aux:
                            token_score = max(token_score, 2.5)
                    for tag in self.tag_forms:
                        if token in tag:
                            token_score = max(token_score, 3.0)
                    for area in self.area_forms:
                        if token in area:
                            token_score = max(token_score, 2.0)

                if token_score > best_score:
                    best_score = token_score
                    best_token = token

            if best_score == 0.0:
                return 0.0, []

            matched.append(best_token or group[0])
            total += best_score

        total += self.score_bias
        return total, matched


class EntriesIndex:
    """In-memory index built from tourism entry dictionaries."""

    def __init__(self, entries: Iterable[Dict[str, Any]] | None = None):
        self._synonyms = _load_synonyms()
        self._documents: List[_Document] = []
        if entries:
            for entry in entries:
                if isinstance(entry, dict):
                    self._documents.append(_Document(entry))

    def search(
        self,
        query: str,
        *,
        filters: Dict[str, str] | None = None,
        limit: int | None = None,
    ) -> List[EntryMatch]:
        token_groups = _expand_tokens(_tokenize(query), self._synonyms)
        if not token_groups:
            return []

        matches: List[EntryMatch] = []
        for document in self._documents:
            if not document.matches_filters(filters):
                continue
            score, matched = document.score(token_groups)
            if score <= 0:
                continue
            matches.append(EntryMatch(document.entry, score, tuple(matched)))

        matches.sort(key=lambda item: (-item.score, normalize_text(item.entry.get("title", ""))))
        if limit is not None:
            return matches[:limit]
        return matches

    def top_filters(self, matches: Sequence[EntryMatch], *, limit: int = 5) -> List[Dict[str, str]]:
        area_counter: Counter[str] = Counter()
        tag_counter: Counter[str] = Counter()
        for match in matches:
            entry = match.entry
            for area in entry.get("areas") or []:
                if isinstance(area, str):
                    area_counter[area] += 1
            for tag in entry.get("tags") or []:
                if isinstance(tag, str):
                    tag_counter[tag] += 1

        options: List[Dict[str, str]] = []
        for area, _ in area_counter.most_common():
            options.append({"type": "area", "value": area})
        for tag, _ in tag_counter.most_common():
            options.append({"type": "tag", "value": tag})
        deduped: List[Dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for option in options:
            key = (option["type"], option["value"])
            if key not in seen:
                seen.add(key)
                deduped.append(option)
            if len(deduped) >= limit:
                break
        return deduped


def load_entries_index(entries: Iterable[Any] | None = None) -> EntriesIndex:
    return EntriesIndex(entries)


__all__ = ["EntriesIndex", "EntryMatch", "load_entries_index"]
