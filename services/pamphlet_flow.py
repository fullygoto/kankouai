"""Session-aware pamphlet fallback flow logic."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from . import pamphlet_rag, pamphlet_search, pamphlet_session
from .message_builder import build_pamphlet_message, parse_pamphlet_answer
from .sources_fmt import format_sources_md, normalize_sources
from .pamphlet_search import SearchResult, detect_city_from_text

_ENABLE_EVIDENCE_TOGGLE = os.getenv("ENABLE_EVIDENCE_TOGGLE", "true").lower() == "true"


@dataclass
class PamphletResponse:
    kind: str  # "answer" | "ask_city" | "error" | "noop"
    message: str
    city: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    sources_md: str = ""
    quick_choices: List[Dict[str, str]] = field(default_factory=list)
    more_available: bool = False
    citations: List[Dict[str, Any]] = field(default_factory=list)
    message_with_labels: str = ""
    show_evidence_toggle: bool = _ENABLE_EVIDENCE_TOGGLE


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
    session = pamphlet_session.PamphletSession(session_store, ttl, now=now)

    stripped = (text or "").strip()
    if not stripped:
        return PamphletResponse(kind="noop", message="")

    # follow-up request for more details
    city_key: Optional[str] = None

    if stripped.startswith("もっと詳しく"):
        info = session.get_followup(user_id)
        if info and info.get("city"):
            detailed = True
            city_key = info.get("city")
            stripped = info.get("query", stripped) or stripped
        else:
            return PamphletResponse(kind="error", message="詳細の対象が見つかりませんでした。")

    pending_query = session.get_pending(user_id)
    detected_city = detect_city_from_text(stripped)

    choice_texts = {c["text"] for c in pamphlet_search.city_choices()}
    if stripped in choice_texts and pending_query:
        detected = pamphlet_search.CITY_ALIASES.get(stripped, detected_city)
        if detected:
            city_key = detected
        stripped = pending_query.get("query", stripped)
        session.clear_pending(user_id)
    elif detected_city:
        city_key = detected_city

    if not city_key:
        cached = session.get_city(user_id)
        if cached:
            city_key = cached

    if not city_key:
        record = session.set_pending(user_id, stripped)
        show_quick = not record.get("asked", False)
        session.set_pending(user_id, stripped, asked=True)
        choices = pamphlet_search.city_choices() if show_quick else []
        message = "どの市町の資料からお探ししますか？"
        if not show_quick:
            message = "対象の市町を「五島市」「新上五島町」「小値賀町」「宇久町」から教えてください。"
        return PamphletResponse(
            kind="ask_city",
            message=message,
            quick_choices=choices,
        )

    session.set_city(user_id, city_key)
    session.clear_pending(user_id)

    answer = pamphlet_rag.answer_from_pamphlets(stripped, city_key)
    labelled_message = (answer.get("answer_with_labels") or "").strip()
    raw_message = labelled_message or (answer.get("answer") or "").strip()
    citations_info = answer.get("citations", []) or []
    sources_payload = citations_info or (answer.get("sources", []) or [])
    normalized_sources = normalize_sources(sources_payload)
    formatted_sources = [f"{city}/{file_}" for city, file_ in normalized_sources]

    parsed_answer = parse_pamphlet_answer(raw_message)
    built = build_pamphlet_message(parsed_answer, sources_payload)
    message = built.text
    sources_md = built.sources_md

    def _sentences_have_labels(text: str) -> bool:
        sentences = [seg.strip() for seg in re.split(r"(?<=[。！？!?])", text or "") if seg.strip()]
        if not sentences:
            return False
        for seg in sentences:
            if not seg.endswith("]]"):
                return False
        return True

    if not message:
        message = "資料に該当する記述が見当たりません。もう少し条件（市町/施設名/時期等）を教えてください。"

    if labelled_message and _sentences_have_labels(labelled_message) and not _sentences_have_labels(message):
        message = labelled_message

    if not normalized_sources:
        fallback_docs = list(searcher(city_key, stripped, topk)) if city_key else []
        fallback_text = summarizer(stripped, fallback_docs, detailed=detailed) if fallback_docs else ""
        if fallback_text:
            sections = fallback_text.strip()
            sources_md = ""
            if "### 出典" in sections:
                body, tail = sections.split("### 出典", 1)
                message_body = body.strip()
                sources_md = "### 出典\n" + tail.strip()
            else:
                message_body = sections
            used_labels = sorted({int(m.group(1)) for m in re.finditer(r"\[\[(\d+)\]\]", fallback_text)})
            mapped_sources: List[str] = []
            for idx in used_labels:
                if 1 <= idx <= len(fallback_docs):
                    chunk = fallback_docs[idx - 1].chunk
                    mapped_sources.append(f"{chunk.city}/{chunk.source_file}")
            session.set_followup(user_id, query=stripped, city=city_key)
            return PamphletResponse(
                kind="answer",
                message=message_body,
                city=city_key,
                sources=mapped_sources,
                sources_md=sources_md,
                more_available=False,
                citations=[],
                message_with_labels=message_body,
            )
        session.clear_followup(user_id)
        return PamphletResponse(
            kind="error",
            message=message,
            city=city_key,
            sources=[],
            sources_md="",
            more_available=False,
            citations=[],
            message_with_labels=labelled_message,
        )

    footer = sources_md if sources_md else format_sources_md(sources_payload)

    session.set_followup(user_id, query=stripped, city=city_key)

    return PamphletResponse(
        kind="answer",
        message=message,
        city=city_key,
        sources=formatted_sources,
        sources_md=footer,
        more_available=False,
        citations=citations_info,
        message_with_labels=message if message else labelled_message,
    )
