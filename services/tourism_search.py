"""Utilities for ranking tourism entries with partial matching."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import pamphlet_search


@dataclass(frozen=True)
class ScoredEntry:
    """A tourism entry with an associated relevance score."""

    entry: Dict[str, object]
    score: float
    matched_tokens: Tuple[str, ...]
    tie_breaker: Tuple[int, float]


MATCH_THRESHOLD = 8.0


def _normalize_text(text: str, *, keep_spaces: bool = False) -> str:
    """Normalise Japanese text for robust matching."""

    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    converted: List[str] = []
    for char in normalized:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F4:  # Katakana → Hiragana
            converted.append(chr(code - 0x60))
        elif code == 0x30F7:  # ヷ
            converted.append("わ")
        elif code == 0x30F8:  # ヸ
            converted.append("ゐ")
        elif code == 0x30F9:  # ヹ
            converted.append("ゑ")
        elif code == 0x30FA:  # ヺ
            converted.append("を")
        else:
            converted.append(char)
    lowered = "".join(converted).lower()
    lowered = lowered.replace("~", "〜")
    if keep_spaces:
        return " ".join(lowered.split())
    return "".join(lowered.split())


def _tokenize(query: str) -> List[str]:
    normalized = _normalize_text(query or "", keep_spaces=True)
    tokens: List[str] = []
    for token in normalized.replace("/", " ").replace("・", " ").split():
        token = token.strip()
        if token:
            tokens.append(token)
    return tokens


def _synonym_map() -> Dict[str, List[str]]:
    path = Path(__file__).resolve().parent.parent / "synonyms.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        return {}

    mapping: Dict[str, List[str]] = {}
    for key, values in (raw or {}).items():
        if not isinstance(key, str) or not isinstance(values, list):
            continue
        forms = {_normalize_text(key)}
        forms.update(_normalize_text(item) for item in values if isinstance(item, str))
        forms.discard("")
        if not forms:
            continue
        ordered = sorted(forms)
        for form in ordered:
            bucket = mapping.setdefault(form, [])
            for alt in ordered:
                if alt != form and alt not in bucket:
                    bucket.append(alt)
    return mapping


def _expand_tokens(tokens: Sequence[str]) -> List[str]:
    mapping = _synonym_map()
    expanded: List[str] = []
    seen = set()
    for token in tokens:
        norm = _normalize_text(token)
        if not norm or norm in seen:
            continue
        expanded.append(norm)
        seen.add(norm)
        for alt in mapping.get(norm, []):
            if alt and alt not in seen:
                expanded.append(alt)
                seen.add(alt)
    return expanded


def _city_token_set(city_key: str) -> List[str]:
    label = pamphlet_search.CITY_LABELS.get(city_key, city_key)
    base = {_normalize_text(label)}
    for alias, key in pamphlet_search.CITY_ALIASES.items():
        if key == city_key:
            base.add(_normalize_text(alias))
    mapping = _synonym_map()
    variants = set(base)
    for token in list(base):
        variants.update(mapping.get(token, []))
    variants.discard("")
    return sorted(variants)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on", "ok", "y"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _entry_matches_city(entry: Dict[str, object], city_key: Optional[str]) -> bool:
    if not city_key:
        return True
    if not _coerce_bool(entry.get("area_checked")):
        return False
    areas = entry.get("areas") or []
    if not isinstance(areas, (list, tuple)):
        areas = [areas]
    normalized_areas = {_normalize_text(str(area)) for area in areas if area}
    if not normalized_areas:
        return False
    city_tokens = set(_city_token_set(city_key))
    return bool(normalized_areas & city_tokens)


def _parse_timestamp(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value, fmt).timestamp()
            except Exception:
                continue
    return 0.0


def _entry_title_length(entry: Dict[str, object]) -> int:
    title = _normalize_text(str(entry.get("title", "")))
    return len(title)


def _score_entry(
    entry: Dict[str, object],
    tokens: Sequence[str],
    *,
    query_city: Optional[str],
) -> Tuple[float, List[str]]:
    if not tokens:
        return 0.0, []
    title = _normalize_text(entry.get("title", ""))
    desc = _normalize_text(entry.get("desc", ""))
    tags = [
        _normalize_text(tag)
        for tag in (entry.get("tags") or [])
        if isinstance(tag, str)
    ]

    matched: List[str] = []
    score = 0.0
    for token in tokens:
        if not token:
            continue

        token_score = 0.0
        if title:
            if token == title:
                token_score = max(token_score, 20.0)
            else:
                idx = title.find(token)
                if idx == 0:
                    token_score = max(token_score, 12.0)
                elif 0 < idx <= 3:
                    token_score = max(token_score, 12.0)
                    prefix = title[:idx]
                    if prefix in {"(有)", "（有）", "(株)", "（株）"}:
                        token_score = max(token_score, 13.0)
                elif idx >= 0:
                    token_score = max(token_score, 8.0)

        if desc and token in desc:
            token_score = max(token_score, 4.0, token_score)

        for tag in tags:
            if tag and token in tag:
                token_score = max(token_score, 3.0, token_score)
                break

        if token_score > 0:
            score += token_score
            matched.append(token)

    if score <= 0:
        return 0.0, []

    if query_city and _entry_matches_city(entry, query_city):
        score += 4.0

    return score, matched


def search(
    entries: Iterable[Dict[str, object]],
    query: str,
    *,
    city_key: Optional[str] = None,
    limit: int = 3,
) -> List[ScoredEntry]:
    """Return scored tourism entries for ``query`` limited to ``limit`` hits."""

    tokens = _tokenize(query)
    if not tokens:
        return []
    expanded_tokens = _expand_tokens(tokens)
    query_city = detect_city_from_text(query)
    results: List[ScoredEntry] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not _entry_matches_city(entry, city_key):
            continue
        score, matched = _score_entry(entry, expanded_tokens, query_city=query_city)
        if score <= 0:
            continue
        updated_ts = _parse_timestamp(
            entry.get("updated_at")
            or entry.get("updated")
            or entry.get("modified_at")
            or entry.get("modified")
            or entry.get("created_at")
        )
        tie_breaker = (_entry_title_length(entry), -updated_ts)
        results.append(
            ScoredEntry(
                entry=entry,
                score=score,
                matched_tokens=tuple(dict.fromkeys(matched)),
                tie_breaker=tie_breaker,
            )
        )

    results.sort(
        key=lambda item: (
            -item.score,
            -_parse_timestamp(
                item.entry.get("updated_at")
                or item.entry.get("updated")
                or item.entry.get("modified_at")
                or item.entry.get("modified")
                or item.entry.get("created_at")
            ),
            _entry_title_length(item.entry),
            _normalize_text(item.entry.get("title", "")),
        )
    )
    return results[:max(1, limit)]


def detect_city_from_text(text: str) -> str | None:
    """Detect a city key from free-form text using known aliases."""

    if not text:
        return None
    detected = pamphlet_search.detect_city_from_text(text)
    if detected:
        return detected
    normalized = _normalize_text(text)
    for key, _label in pamphlet_search.CITY_LABELS.items():
        tokens = _city_token_set(key)
        if any(token and token in normalized for token in tokens):
            return key
    return None


def city_prompt(*, asked: bool) -> str:
    choices = [choice["text"] for choice in pamphlet_search.city_choices()]
    if not asked:
        lines = ["どの市町の資料ですか？"]
        lines.extend(f"- {choice}" for choice in choices)
        return "\n".join(lines)
    return "市町を「五島市」「新上五島町」「宇久町」「小値賀町」から教えてください。"


__all__ = [
    "ScoredEntry",
    "search",
    "detect_city_from_text",
    "city_prompt",
    "MATCH_THRESHOLD",
]
