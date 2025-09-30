"""Text normalisation helpers for search modules."""
from __future__ import annotations

from typing import Iterable, List

import re
import unicodedata


_SMALL_KANA = {
    "ぁ": "あ",
    "ぃ": "い",
    "ぅ": "う",
    "ぇ": "え",
    "ぉ": "お",
    "ゃ": "や",
    "ゅ": "ゆ",
    "ょ": "よ",
    "ゎ": "わ",
    "っ": "つ",
}


def _katakana_to_hiragana(text: str) -> str:
    chars: List[str] = []
    for char in text:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F4:
            chars.append(chr(code - 0x60))
        elif char in {"ヵ", "ヶ"}:
            chars.append("か" if char == "ヵ" else "け")
        else:
            chars.append(char)
    return "".join(chars)


def _strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    filtered = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", filtered)


def normalize_text(text: str, *, keep_spaces: bool = False) -> str:
    """Return a canonical form suitable for fuzzy matching."""

    if text is None:
        return ""

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("〜", "~")
    normalized = normalized.lower()
    normalized = _katakana_to_hiragana(normalized)

    if keep_spaces:
        collapsed = re.sub(r"\s+", " ", normalized).strip()
    else:
        collapsed = re.sub(r"\s+", "", normalized)

    return collapsed


def normalize_query(tokens: Iterable[str]) -> list[str]:
    """Return a list of distinct normalised tokens.

    The function performs character width harmonisation, converts katakana
    into hiragana, lowercases ASCII characters and strips diacritics so that
    variations such as voiced sounds or half-width forms can be matched.  The
    returned list preserves insertion order while removing duplicates.
    """

    seen: set[str] = set()
    results: list[str] = []

    for token in tokens:
        if token is None:
            continue

        base = normalize_text(token)
        if not base:
            continue

        # Preserve distinct variations (with/without diacritics, spaces kept).
        variants = {base}
        variants.add(_strip_diacritics(base))
        spaced = normalize_text(token, keep_spaces=True).replace(" ", "")
        if spaced:
            variants.add(spaced)

        for variant in variants:
            if variant and variant not in seen:
                seen.add(variant)
                results.append(variant)

    return results


def hiragana(text: str) -> str:
    """Return hiragana representation for Japanese tokens."""

    return _katakana_to_hiragana(normalize_text(text, keep_spaces=False))


def hiragana_without_diacritics(text: str) -> str:
    """Return hiragana with dakuten/handakuten stripped."""

    hira = hiragana(text)
    return _strip_diacritics(hira)


def simplify_kana(text: str) -> str:
    """Collapse small kana characters to their base forms."""

    hira = hiragana(text)
    replaced = "".join(_SMALL_KANA.get(ch, ch) for ch in hira)
    return replaced


__all__ = [
    "normalize_query",
    "normalize_text",
    "hiragana",
    "hiragana_without_diacritics",
    "simplify_kana",
]
