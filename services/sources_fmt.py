"""Utilities for normalizing and rendering pamphlet sources."""
from __future__ import annotations

import os
import re
from typing import Any, Iterable, List, Tuple

from . import pamphlet_search

_SRC_RE = re.compile(
    r"""
    ^(?P<city>[^/]+)/              # 市町 (五島市 など)
    (?P<file>[^/]+?)               # ファイル名 (拡張子なしを想定)
    (?:\.(?:txt|md))?             # txt/md 拡張子は捨てる
    (?:/L?\d+(?:-\d+)?)?$        # /L12-15 のような行番号は捨てる
    """,
    re.X | re.I,
)


def _clean_file_name(value: str | None) -> str | None:
    if not value:
        return None
    name = os.path.basename(str(value))
    name = re.sub(r"\.(txt|md)$", "", name, flags=re.I)
    return name or None


def normalize_sources(sources: Iterable[Any]) -> List[Tuple[str, str]]:
    """Normalize heterogeneous source payloads to ``(city, stem)`` tuples."""

    seen: set[Tuple[str, str]] = set()
    items: List[Tuple[str, str]] = []

    for raw in sources or []:
        city: str | None = None
        file_stem: str | None = None

        if isinstance(raw, str):
            match = _SRC_RE.match(raw.strip())
            if match:
                city = match.group("city")
                file_stem = match.group("file")
        elif isinstance(raw, dict):
            city_val = raw.get("city") or raw.get("City")
            if city_val:
                city = pamphlet_search.city_label(str(city_val))

            file_val = raw.get("file") or raw.get("filename") or raw.get("path")
            file_stem = _clean_file_name(file_val)

        if not city or not file_stem:
            continue

        key = (city, file_stem)
        if key in seen:
            continue
        seen.add(key)
        items.append(key)

    return items


def format_sources_md(sources: Iterable[Any], heading: str = "### 出典") -> str:
    """Render a markdown block from sources after normalization."""

    items = normalize_sources(sources)
    if not items:
        return ""
    body = "\n".join(f"- {city}/{file_stem}" for city, file_stem in items)
    return f"{heading}\n{body}"
