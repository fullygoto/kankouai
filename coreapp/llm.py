"""Utility LLM client with deterministic summarisation for tests."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List

from coreapp.config import (
    MODEL_DEFAULT,
    MODEL_HARD,
    THRESHOLD_PIECES_HARD,
    THRESHOLD_REPROMPTS_HARD,
    THRESHOLD_SCORE_HARD,
)


MODEL_NAME = MODEL_DEFAULT


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[:limit].rstrip("、。．. ,")
    return f"{clipped}…"


def _build_summary(query: str, city: str, fragments: Iterable[Dict[str, Any]]) -> str:
    pieces: List[str] = []
    refs: List[Dict[str, Any]] = []
    for idx, fragment in enumerate(fragments):
        if idx >= 4:
            break
        text = _clean(str(fragment.get("text", "")))
        if not text:
            continue
        refs.append(fragment)
        source = _clean(str(fragment.get("source") or f"資料{idx + 1}"))
        pieces.append(f"・{_trim(text, 72)}（出典: {source}）")

    header = f"{city}パンフレット要約：{query or '概要'}"
    footer = "※数値や日時はパンフレット記載どおりに引用しています。"

    summary_lines = [header]
    summary_lines.extend(pieces)
    summary_lines.append(footer)

    summary = "\n".join(summary_lines)

    # Ensure the summary stays within 200〜350 characters as requested.
    if len(summary) < 200:
        filler_src = refs[0]["text"] if refs else query
        filler = _clean(str(filler_src))
        if filler:
            need = min(350 - len(summary), max(0, 200 - len(summary)))
            extra = _trim(filler, need)
            if extra:
                summary = "\n".join(summary_lines + [f"補足: {extra}"])

    if len(summary) > 350:
        # Gradually shorten bullet snippets to fit the upper bound.
        available = list(pieces)
        idx = 0
        while len(summary) > 350 and available:
            line = available[idx % len(available)]
            body, _, tail = line.partition("（出典:")
            body = _trim(body.lstrip("・"), max(10, int(math.ceil(len(body) * 0.85))))
            available[idx % len(available)] = f"・{body}（出典:{tail}" if tail else f"・{body}"
            summary = "\n".join([header, *available, footer])
            idx += 1

    return summary


@dataclass
class LLMRequest:
    """Data required to make an LLM call."""

    prompt: str
    parameters: Dict[str, Any] | None = None


class LLMClient:
    """Deterministic stand-in for gpt-4o-mini used in tests."""

    model: str = MODEL_NAME

    def complete(self, request: LLMRequest) -> str:
        params = request.parameters or {}
        fragments = params.get("fragments") or []
        city = params.get("city_label") or params.get("city") or "五島市"
        query = params.get("query") or request.prompt
        return _build_summary(str(query), str(city), fragments)


def get_llm_client() -> LLMClient:
    """Factory for the LLM client."""

    return LLMClient()


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score):
        return None
    return score


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def pick_model(score: Any, pieces_cnt: Any, reprompts: Any) -> str:
    """Return the most suitable model based on retrieval metrics."""

    numeric_score = _coerce_score(score)
    pieces = max(0, _coerce_int(pieces_cnt))
    retries = max(0, _coerce_int(reprompts))

    upgrade = False
    if numeric_score is not None and numeric_score < THRESHOLD_SCORE_HARD:
        upgrade = True
    if pieces >= THRESHOLD_PIECES_HARD:
        upgrade = True
    if retries >= THRESHOLD_REPROMPTS_HARD:
        upgrade = True

    return MODEL_HARD if upgrade else MODEL_DEFAULT


__all__ = [
    "LLMClient",
    "LLMRequest",
    "MODEL_NAME",
    "get_llm_client",
    "pick_model",
]
