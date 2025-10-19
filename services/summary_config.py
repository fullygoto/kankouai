"""Helpers for shared summary generation configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_STYLE = "polite_long"
_DEFAULT_MODE = "adaptive"
_DEFAULT_MIN = 550
_DEFAULT_MAX = 800
_DEFAULT_FALLBACK = 300


@dataclass(frozen=True)
class SummaryBounds:
    min_chars: int
    max_chars: int
    is_short_context: bool


def get_summary_mode() -> str:
    """Return the current summary mode."""
    value = os.getenv("SUMMARY_MODE", _DEFAULT_MODE)
    value = (value or "").strip().lower()
    if value in {"adaptive", "terse", "long"}:
        return value
    return _DEFAULT_MODE


def get_summary_style() -> str:
    """Return the configured summary style."""
    override = os.getenv("SUMMARY_STYLE")
    if override:
        value = override.strip().lower()
        if value:
            return value
    mode = get_summary_mode()
    if mode == "terse":
        return "terse"
    if mode == "long":
        return "polite_long"
    return "adaptive"


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def get_summary_min_chars() -> int:
    return _read_int("SUMMARY_MIN_CHARS", _DEFAULT_MIN)


def get_summary_max_chars() -> int:
    return _read_int("SUMMARY_MAX_CHARS", _DEFAULT_MAX)


def get_summary_min_fallback() -> int:
    return _read_int("SUMMARY_MIN_FALLBACK", _DEFAULT_FALLBACK)


def get_summary_bounds(context_text: str | None) -> SummaryBounds:
    """Determine length targets based on context richness."""
    min_chars = get_summary_min_chars()
    max_chars = get_summary_max_chars()
    fallback = get_summary_min_fallback()

    text = (context_text or "").strip()
    context_len = len(text)
    line_count = text.count("\n") + 1 if text else 0

    short_threshold = max(400, fallback * 2)
    is_short = context_len <= short_threshold or line_count <= 2

    if is_short:
        short_min = fallback
        short_max = min(max_chars, max(fallback + 200, 500))
        if short_max < short_min:
            short_max = short_min
        return SummaryBounds(min_chars=short_min, max_chars=short_max, is_short_context=True)

    if max_chars < min_chars:
        max_chars = min_chars

    return SummaryBounds(min_chars=min_chars, max_chars=max_chars, is_short_context=False)
