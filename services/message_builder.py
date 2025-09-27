"""Utilities to parse and format pamphlet RAG answers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence

from .sources_fmt import format_sources_md
from .summary_config import get_summary_style

SHOW_NOTES = False
_MAX_DETAILS = 2
_SENTENCE_END = "。.!?！？"

_HEADER_MAP = {
    "質問の意図": "intent",
    "要約": "summary",
    "詳細": "details",
    "補足": "notes",
    "出典": "sources",
}


def _normalize_header(line: str) -> str | None:
    stripped = (line or "").strip()
    if not stripped:
        return None
    stripped = stripped.lstrip("# ")
    stripped = stripped.rstrip("：:").strip()
    return _HEADER_MAP.get(stripped)


_BULLET_PREFIX = re.compile(r"^[\s\-・*●○◆◇▶▷■□◉◎]+")
_NUMBERED_PREFIX = re.compile(r"^[0-9]{1,2}[\.)、]\s*")


def _strip_bullet(line: str) -> str:
    text = (line or "").strip()
    if not text:
        return ""
    text = _BULLET_PREFIX.sub("", text)
    text = _NUMBERED_PREFIX.sub("", text)
    return text.strip()


def _ensure_sentence(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return ""
    if cleaned[-1] not in _SENTENCE_END:
        cleaned += "。"
    return cleaned


def _split_sentences(text: str) -> List[str]:
    merged = " ".join(part.strip() for part in (text or "").splitlines() if part.strip())
    if not merged:
        return []
    sentences: List[str] = []
    buf: List[str] = []
    for ch in merged:
        buf.append(ch)
        if ch in _SENTENCE_END:
            sentence = "".join(buf).strip()
            if sentence:
                sentences.append(_ensure_sentence(sentence))
            buf = []
    if buf:
        sentence = "".join(buf).strip()
        if sentence:
            sentences.append(_ensure_sentence(sentence))
    return sentences


@dataclass
class PamphletAnswer:
    summary: str = ""
    details: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class PamphletMessage:
    text: str
    summary: str
    details: List[str]
    sources_md: str


def parse_pamphlet_answer(raw: str) -> PamphletAnswer:
    raw_text = (raw or "").strip()
    if not raw_text:
        return PamphletAnswer()

    lines = raw_text.splitlines()
    if not any(_normalize_header(line) for line in lines):
        return PamphletAnswer(summary=raw_text, details=[], notes=[])

    sections: dict[str, List[str]] = {name: [] for name in _HEADER_MAP.values()}
    current = "summary"

    for line in lines:
        header = _normalize_header(line)
        if header:
            current = header
            continue
        if not line.strip():
            continue
        sections.setdefault(current, []).append(line.strip())

    summary = "\n".join(sections.get("summary", []))
    detail_lines = [_strip_bullet(ln) for ln in sections.get("details", [])]
    details = [ln for ln in detail_lines if ln]
    note_lines = [_strip_bullet(ln) for ln in sections.get("notes", [])]
    notes = [ln for ln in note_lines if ln]

    return PamphletAnswer(summary=summary, details=details, notes=notes)


def _expand_summary(base: str, details: Sequence[str]) -> tuple[str, int]:
    if get_summary_style() in {"polite_long", "adaptive"}:
        return (base or "").strip(), 0

    sentences = _split_sentences(base)
    detail_sentences = [_ensure_sentence(d) for d in details if d]

    used = 0
    idx = 0
    while len(sentences) < 2 and idx < len(detail_sentences):
        sentences.append(detail_sentences[idx])
        idx += 1
    used = max(used, idx)

    while idx < len(detail_sentences) and (len(sentences) < 4 or len("".join(sentences)) < 200):
        sentences.append(detail_sentences[idx])
        idx += 1
    used = max(used, idx)

    if not sentences and detail_sentences:
        sentences.append(detail_sentences[0])
        used = max(used, 1)

    if len(sentences) > 4:
        sentences = sentences[:4]

    while len("".join(sentences)) > 360 and len(sentences) > 2:
        sentences.pop()

    summary_text = "".join(sentences)
    return summary_text, min(used, len(detail_sentences))


def build_pamphlet_message(
    answer: PamphletAnswer,
    sources: Iterable[Any],
    *,
    show_notes: bool = SHOW_NOTES,
) -> PamphletMessage:
    cleaned_details = [d for d in answer.details if d and d != "資料に明記なし"]
    summary_text, used_detail_count = _expand_summary(answer.summary, cleaned_details)

    remaining_details = cleaned_details[used_detail_count:]
    remaining_details = [d for d in remaining_details if d and d != "資料に明記なし"]
    if len(remaining_details) > _MAX_DETAILS:
        remaining_details = remaining_details[:_MAX_DETAILS]

    style = get_summary_style()

    if style in {"polite_long", "adaptive"}:
        summary_candidate = summary_text.strip()
        if not summary_candidate:
            fallback = _ensure_sentence(answer.summary)
            if fallback:
                summary_candidate = fallback.strip()
            elif remaining_details:
                summary_candidate = _ensure_sentence(remaining_details[0]).strip()
                remaining_details = remaining_details[1:]

        summary_candidate = summary_candidate.strip()
        text_parts: List[str] = [summary_candidate] if summary_candidate else []
        sources_md = format_sources_md(sources)
        if sources_md:
            text_parts.append(sources_md)

        text = "\n\n".join(part for part in text_parts if part).strip()
        return PamphletMessage(
            text=text,
            summary=summary_candidate,
            details=[],
            sources_md=sources_md,
        )

    if not summary_text:
        fallback = _ensure_sentence(answer.summary)
        if fallback:
            summary_text = fallback
        elif remaining_details:
            summary_text = _ensure_sentence(remaining_details[0])
            remaining_details = remaining_details[1:]

    parts: List[str] = []
    if summary_text:
        parts.append(f"### 要約\n{summary_text}")

    detail_lines = [d for d in remaining_details if d]
    if detail_lines:
        parts.append("### 詳細\n- " + "\n- ".join(detail_lines))

    if show_notes and answer.notes:
        note_lines = [n for n in answer.notes if n and n != "資料に明記なし"]
        if note_lines:
            parts.append("### 補足\n- " + "\n- ".join(note_lines[:_MAX_DETAILS]))

    sources_md = format_sources_md(sources)
    if sources_md:
        parts.append(sources_md)

    text = "\n\n".join(parts).strip()

    return PamphletMessage(text=text, summary=summary_text, details=detail_lines, sources_md=sources_md)
