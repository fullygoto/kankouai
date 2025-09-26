"""Session-aware pamphlet fallback flow logic."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from . import pamphlet_rag, pamphlet_search
from .pamphlet_search import SearchResult, city_label, detect_city_from_text


@dataclass
class PamphletResponse:
    kind: str  # "answer" | "ask_city" | "error" | "noop"
    message: str
    city: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    quick_choices: List[Dict[str, str]] = field(default_factory=list)
    more_available: bool = False


def build_response(
    text: str,
    *,
    user_id: str,
    session_store: Dict[str, Dict[str, dict]],
    topk: int,
    ttl: int,
    searcher: Callable[[str, str, int], Iterable[SearchResult]],
    summarizer: Callable[..., str],
    detailed: bool = False,
) -> PamphletResponse:
    now = time.time()
    pam = session_store.setdefault(
        "pamphlet",
        {"city": {}, "pending": {}, "followup": {}},
    )
    city_cache: Dict[str, tuple[str, float]] = pam.setdefault("city", {})
    pending: Dict[str, dict] = pam.setdefault("pending", {})
    followup: Dict[str, dict] = pam.setdefault("followup", {})

    # cleanup stale entries
    for store in (city_cache, pending, followup):
        for key in list(store.keys()):
            ts = store[key][1] if isinstance(store[key], tuple) else store[key].get("ts", 0.0)
            if now - ts > ttl:
                store.pop(key, None)

    stripped = (text or "").strip()
    if not stripped:
        return PamphletResponse(kind="noop", message="")

    # follow-up request for more details
    if stripped.startswith("もっと詳しく"):
        info = followup.get(user_id)
        if info:
            query = info.get("query", "")
            city = info.get("city")
            if city:
                detailed = True
                stripped = query
            else:
                return PamphletResponse(kind="error", message="詳細を取得できませんでした。")
        else:
            return PamphletResponse(kind="error", message="詳細の対象が見つかりませんでした。")

    pending_query = pending.get(user_id)
    detected_city = detect_city_from_text(stripped)

    # treat a pure city label as selection if pending exists
    if stripped in {c["text"] for c in pamphlet_search.city_choices()} and pending_query:
        detected_city = pamphlet_search.CITY_ALIASES.get(stripped, detected_city)
        stripped = pending_query.get("query", stripped)
        pending.pop(user_id, None)

    # resolved city from cache if none detected
    city_key = detected_city
    if not city_key:
        cached = city_cache.get(user_id)
        if cached and now - cached[1] <= ttl:
            city_key = cached[0]

    if not city_key:
        record = pending.get(user_id)
        if record:
            record.update({"query": stripped, "ts": now})
        else:
            record = {"query": stripped, "ts": now, "asked": False}
            pending[user_id] = record
        show_quick = not record.get("asked", False)
        record["asked"] = True
        choices = pamphlet_search.city_choices() if show_quick else []
        message = "どの市町の資料からお探ししますか？"
        if not show_quick:
            message = "対象の市町を「五島市」「新上五島町」「小値賀町」「宇久町」から教えてください。"
        return PamphletResponse(
            kind="ask_city",
            message=message,
            quick_choices=choices,
        )

    city_cache[user_id] = (city_key, now)

    answer = pamphlet_rag.answer_from_pamphlets(stripped, city_key)
    message = answer.get("answer", "").strip()
    sources_info = answer.get("sources", []) or []
    normalized_sources = pamphlet_rag.normalize_sources(sources_info)
    formatted_sources = [f"{city}/{file_}" for city, file_ in normalized_sources]

    if not message:
        message = "資料に該当する記述が見当たりません。もう少し条件（市町/施設名/時期等）を教えてください。"

    if not normalized_sources:
        return PamphletResponse(
            kind="error",
            message=message,
            city=city_key,
            sources=[],
            more_available=False,
        )

    footer = pamphlet_rag.format_sources_md(sources_info)
    if footer:
        message = f"{message}\n\n{footer}" if message else footer

    followup[user_id] = {"query": stripped, "city": city_key, "ts": now}

    return PamphletResponse(
        kind="answer",
        message=message,
        city=city_key,
        sources=formatted_sources,
        more_available=False,
    )
