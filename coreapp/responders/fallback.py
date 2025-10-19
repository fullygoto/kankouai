"""Fallback responder returning a menu of quick actions (spec v1 Step5-④)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class FallbackResponderResult:
    """Structured result produced by :class:`FallbackResponder`."""

    message: str
    quick_replies: Sequence[Dict[str, str]]


class FallbackResponder:
    """Responder that guides the user when no intent matches."""

    class Action(StrEnum):
        WEATHER = "weather"
        STATUS = "transport_status"
        VIEW_MAP = "viewpoint_map"
        ENTRY_SEARCH = "entry_search"
        SPOT_SEARCH = "spot_search"

    _BASE_MESSAGE = "該当なし: 該当するものが見つかりませんでした。目的に近いメニューから選んでください。"
    _MENU_DEFAULT: List[Dict[str, str]] = [
        {"label": "今日の天気", "payload": Action.WEATHER.value},
        {"label": "運行状況", "payload": Action.STATUS.value},
        {"label": "展望所マップ", "payload": Action.VIEW_MAP.value},
        {"label": "店舗名で検索", "payload": Action.ENTRY_SEARCH.value},
        {"label": "観光地名で検索", "payload": Action.SPOT_SEARCH.value},
    ]

    _MENU_CONTEXTUAL: List[Dict[str, str]] = [
        {"label": "天気", "payload": "天気"},
        {"label": "運行状況", "payload": "運行状況"},
        {"label": "展望所マップ", "payload": "展望所マップ"},
        {"label": "店舗名で探す", "payload": "店舗名検索"},
        {"label": "観光地名で探す", "payload": "観光地名検索"},
    ]

    def respond(
        self,
        message: str,
        *,
        context: Dict[str, Any] | None = None,
    ) -> FallbackResponderResult:
        """Return fallback message with quick-reply suggestions."""

        _ = (message,)
        use_contextual = context is not None
        menu = self._MENU_CONTEXTUAL if use_contextual else self._MENU_DEFAULT
        return FallbackResponderResult(
            message=self._BASE_MESSAGE,
            quick_replies=[dict(item) for item in menu],
        )


__all__ = ["FallbackResponder", "FallbackResponderResult"]
