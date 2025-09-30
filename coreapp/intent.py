"""Intent detection helpers for the responder skeleton."""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Dict, Iterable, Optional


def _normalize(text: str | None) -> str:
    """Return a lower-cased NFKC-normalised string for matching."""

    if text is None:
        return ""
    return unicodedata.normalize("NFKC", text).strip().lower()


def normalize_for_matching(text: str | None) -> str:
    """Public helper exposing the normalisation used for intent checks."""

    return _normalize(text)


# --- Priority trigger dictionaries ---------------------------------------

WEATHER_KEYWORDS: tuple[str, ...] = ("天気", "天候", "予報", "weather")

TRANSPORT_STATE_KEYWORDS: tuple[str, ...] = (
    "運行",
    "運航",
    "運休",
    "欠航",
    "遅延",
    "見合わせ",
    "運転見合わせ",
    "状況",
    "情報",
    "status",
)

TRANSPORT_VEHICLE_KEYWORDS: tuple[str, ...] = (
    "船",
    "フェリー",
    "ジェットフォイル",
    "高速船",
    "太古",
    "飛行機",
    "空港",
    "福江空港",
    "五島つばき空港",
    "ana",
    "jal",
    "orc",
    "オリエンタルエアブリッジ",
    "九州商船",
    "産業汽船",
)

TRANSPORT_SHIP_KEYWORDS: tuple[str, ...] = (
    "船",
    "フェリー",
    "ジェットフォイル",
    "高速船",
    "太古",
    "九州商船",
    "産業汽船",
)

TRANSPORT_FLIGHT_KEYWORDS: tuple[str, ...] = (
    "飛行機",
    "空港",
    "フライト",
    "福江空港",
    "五島つばき空港",
    "ana",
    "jal",
    "orc",
    "オリエンタルエアブリッジ",
)

MAP_KEYWORDS: tuple[str, ...] = (
    "展望所マップ",
    "展望マップ",
    "展望所の地図",
    "展望台マップ",
    "展望台の地図",
)

MAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"展望\s*(?:所|台)?\s*(?:マップ|地図)"),
    re.compile(r"viewpoints?\s*map", re.IGNORECASE),
)

PRIORITY_ORDER: tuple[str, ...] = ("weather", "transport", "viewpoint_map")


def is_weather_query(text: str | None) -> bool:
    """Return ``True`` if the text should trigger the weather responder."""

    normalized = _normalize(text)
    return any(keyword in normalized for keyword in WEATHER_KEYWORDS)


def _contains_any(normalized: str, keywords: Iterable[str]) -> bool:
    return any(keyword in normalized for keyword in keywords)


def is_transport_query(text: str | None) -> bool:
    """Return ``True`` if the text should trigger the transport responder."""

    normalized = _normalize(text)
    if not normalized:
        return False
    state_hit = _contains_any(normalized, TRANSPORT_STATE_KEYWORDS)
    vehicle_hit = _contains_any(normalized, TRANSPORT_VEHICLE_KEYWORDS)
    return state_hit or vehicle_hit


def is_viewpoint_map_query(text: str | None) -> bool:
    """Return ``True`` if the text refers to the viewpoint map."""

    normalized = _normalize(text)
    if not normalized:
        return False
    if _contains_any(normalized, MAP_KEYWORDS):
        return True
    return any(pattern.search(normalized) for pattern in MAP_PATTERNS)


@dataclass
class IntentDetector:
    """Utility class providing simple keyword-based intent detection."""

    priority_order: tuple[str, ...] = PRIORITY_ORDER

    def detect(self, message: str, *, context: Dict[str, Any] | None = None) -> str:
        """Return a coarse intent label for the given message."""

        _ = context
        priority = self.classify_priority(message)
        return priority or "fallback"

    def classify_priority(self, message: str | None) -> Optional[str]:
        """Return the priority label if the message matches one."""

        checks = {
            "weather": is_weather_query,
            "transport": is_transport_query,
            "viewpoint_map": is_viewpoint_map_query,
        }
        for label in self.priority_order:
            checker = checks.get(label)
            if checker and checker(message):
                return label
        return None

    def is_priority(self, message: str | None) -> bool:
        """Return ``True`` if the message triggers any priority responder."""

        return self.classify_priority(message) is not None


def get_intent_detector() -> IntentDetector:
    """Factory for the intent detector."""

    return IntentDetector()


__all__ = [
    "IntentDetector",
    "MAP_KEYWORDS",
    "MAP_PATTERNS",
    "PRIORITY_ORDER",
    "TRANSPORT_FLIGHT_KEYWORDS",
    "TRANSPORT_SHIP_KEYWORDS",
    "TRANSPORT_STATE_KEYWORDS",
    "TRANSPORT_VEHICLE_KEYWORDS",
    "WEATHER_KEYWORDS",
    "get_intent_detector",
    "is_transport_query",
    "normalize_for_matching",
    "is_viewpoint_map_query",
    "is_weather_query",
]
