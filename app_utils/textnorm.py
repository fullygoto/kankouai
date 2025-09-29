"""Lightweight Japanese text normalisation utilities."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize"]

_REMOVED = {
    "・",
    "、",
    "，",
    ",",
    "。",
    ".",
    "．",
    "（",
    "）",
    "(",
    ")",
    "[",
    "]",
    "［",
    "］",
    "『",
    "』",
    "「",
    "」",
    "~",
    "〜",
    "-",
    "−",
    "_",
    "／",
    "/",
    "\\",
    "・",
    "!",
    "！",
    "?",
    "？",
    "︖",
    "︕",
    "：",
    ":",
    "；",
    ";",
    "・",
    "·",
}

_SPACE_RE = re.compile(r"\s+")


def _katakana_to_hiragana(char: str) -> str:
    code = ord(char)
    if 0x30A1 <= code <= 0x30FA:
        return chr(code - 0x60)
    if code in {0x30F7, 0x30F8, 0x30F9, 0x30FA}:
        return chr(code - 0x60)
    return char


def normalize(text: str | None, *, keep_spaces: bool = False) -> str:
    """Normalise ``text`` for fuzzy matching."""

    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    converted = []
    for char in normalized:
        if char.isspace():
            if keep_spaces:
                converted.append(" ")
            continue
        char = _katakana_to_hiragana(char)
        lower = char.lower()
        if lower in _REMOVED:
            continue
        converted.append(lower)

    collapsed = "".join(converted)
    if keep_spaces:
        collapsed = _SPACE_RE.sub(" ", collapsed).strip()
    return collapsed
