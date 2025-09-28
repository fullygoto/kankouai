"""Summarisation helpers for pamphlet fallback answers."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .pamphlet_search import CITY_LABELS, PamphletChunk, SearchResult

logger = logging.getLogger(__name__)

_CLIENT = None


def configure(openai_client) -> None:
    """Register an OpenAI client instance used for summarisation."""
    global _CLIENT
    _CLIENT = openai_client


@dataclass(frozen=True)
class _Sentence:
    text: str
    score: int
    order: tuple[int, int]
    ref_index: int


def _split_sentences(text: str) -> List[str]:
    stripped = text.replace("\r", " ").replace("\n", " ")
    pieces = re.split(r"(?<=[。！？!?])\s*", stripped)
    sentences: List[str] = []
    for piece in pieces:
        clean = piece.strip()
        if clean:
            sentences.append(clean)
    if not sentences and stripped.strip():
        sentences.append(stripped.strip())
    return sentences


def _extract_keywords(query: str) -> List[str]:
    base_tokens = re.findall(r"[A-Za-z0-9一-龥ぁ-んァ-ンー]+", query or "")
    keywords: List[str] = []
    for token in base_tokens:
        parts = re.split(r"[がはをにでとのやへもよりからまで、。,\.\s]+", token)
        for part in parts:
            candidate = part.strip()
            if len(candidate) >= 2:
                keywords.append(candidate)
    return keywords


def _score_sentence(sentence: str, keywords: Sequence[str]) -> int:
    if not keywords:
        return max(1, len(sentence) // 40)
    score = 0
    for kw in keywords:
        if kw and kw in sentence:
            score += max(len(kw), 2)
    return score


def _choose_sentences(
    sentences: List[_Sentence],
    *,
    summary_limit: int,
    detailed: bool,
) -> tuple[List[_Sentence], List[_Sentence]]:
    ordered = sorted(sentences, key=lambda s: s.order)
    summary: List[_Sentence] = []
    used_orders: set[tuple[int, int]] = set()

    for sent in ordered:
        if sent.score <= 0:
            continue
        if sent.order in used_orders:
            continue
        summary.append(sent)
        used_orders.add(sent.order)
        if len(summary) >= summary_limit:
            break

    if not summary:
        extras = sorted(sentences, key=lambda s: (s.score, -len(s.text)), reverse=True)
        for cand in extras:
            if cand.order in used_orders:
                continue
            summary.append(cand)
            used_orders.add(cand.order)
            if len(summary) >= max(1, summary_limit):
                break

    summary.sort(key=lambda s: s.order)

    detail_candidates = [s for s in ordered if s.order not in used_orders]
    if detailed:
        prioritized = [s for s in detail_candidates if s.score > 0]
        if not prioritized:
            prioritized = detail_candidates
        detail = prioritized[:5]
    else:
        detail = []
    return summary, detail


def _format_output(
    summary: List[_Sentence],
    detail: List[_Sentence],
    refs: List[tuple[int, PamphletChunk]],
) -> str:
    if not summary:
        return ""

    used_refs = {s.ref_index for s in summary}
    used_refs.update(s.ref_index for s in detail)

    def _with_citation(text: str, ref_idx: int) -> str:
        return f"{text} [[{ref_idx}]]"

    summary_lines = [_with_citation(s.text, s.ref_index) for s in summary]

    detail_lines = []
    for sent in detail:
        detail_lines.append(f"- {_with_citation(sent.text, sent.ref_index)}")

    source_lines = []
    for idx, chunk in refs:
        if idx not in used_refs:
            continue
        city_label = CITY_LABELS.get(chunk.city, chunk.city or "")
        label = f"{city_label} / {chunk.source_file}" if city_label else chunk.source_file
        source_lines.append(f"- [[{idx}]] {label}")

    sections: List[str] = ["### 要約", "\n".join(summary_lines)]
    if detail_lines:
        sections.append("\n### 詳細")
        sections.append("\n".join(detail_lines))
    sections.append("\n### 出典")
    sections.append("\n".join(source_lines))
    text = "\n".join(sections).strip()

    if len(text) > 800 and detail_lines:
        while detail_lines and len(text) > 800:
            detail_lines.pop()
            sections = ["### 要約", "\n".join(summary_lines)]
            if detail_lines:
                sections.append("\n### 詳細")
                sections.append("\n".join(detail_lines))
            sections.append("\n### 出典")
            sections.append("\n".join(source_lines))
            text = "\n".join(sections).strip()

    return text


def summarize_with_gpt_nano(query: str, top_docs: Iterable[SearchResult], *, detailed: bool = False) -> str:
    docs = [doc for doc in top_docs if doc and getattr(doc, "chunk", None)]
    if not docs:
        return ""

    primary_city = docs[0].chunk.city
    city_filtered = [doc for doc in docs if doc.chunk.city == primary_city]
    if city_filtered:
        docs = city_filtered

    keywords = _extract_keywords(query)

    contexts: List[tuple[int, PamphletChunk, List[str]]] = []
    for idx, doc in enumerate(docs, start=1):
        chunk = doc.chunk
        sentences = _split_sentences(chunk.text)
        if not sentences:
            continue
        contexts.append((idx, chunk, sentences))

    if not contexts:
        return ""

    sentence_entries: List[_Sentence] = []
    for ctx_index, (ref_idx, chunk, sentences) in enumerate(contexts):
        for sent_index, sentence in enumerate(sentences):
            clean = sentence.strip()
            if not clean:
                continue
            score = _score_sentence(clean, keywords)
            sentence_entries.append(
                _Sentence(text=clean, score=score, order=(ctx_index, sent_index), ref_index=ref_idx)
            )

    summary_limit = 3 if detailed else 2
    summary, detail = _choose_sentences(sentence_entries, summary_limit=summary_limit, detailed=detailed)

    output = _format_output(summary, detail, [(ref, chunk) for ref, chunk, _ in contexts])
    if output:
        return output

    fallback_sentences = [entry for entry in sentence_entries if entry.text]
    fallback_summary = fallback_sentences[: max(1, summary_limit)]
    return _format_output(fallback_summary, [], [(ref, chunk) for ref, chunk, _ in contexts])
