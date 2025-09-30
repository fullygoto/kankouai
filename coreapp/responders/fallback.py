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
        ENTRY_SEARCH = "entry_search"   # 店舗名で検索
        SPOT_SEARCH = "spot_search"     # 観光地名で検索

    _BASE_MESSAGE = "該当するものが見つかりませんでした。目的に近いメニューから選んでください。"
    _MENU: List[Dict[str, str]] = [
        {"label": "今日の天気", "payload": Action.WEATHER.value},
        {"label": "運行状況", "payload": Action.STATUS.value},
        {"label": "展望所マップ", "payload": Action.VIEW_MAP.value},
        {"label": "店舗名で検索", "payload": Action.ENTRY_SEARCH.value},
        {"label": "観光地名で検索", "payload": Action.SPOT_SEARCH.value},
    ]

    def respond(
        self,
        message: str,
        *,
        context: Dict[str, Any] | None = None,
    ) -> FallbackResponderResult:
        """Return fallback message with quick-reply suggestions."""

        _ = (message, context)
        return FallbackResponderResult(
            message=self._BASE_MESSAGE,
            quick_replies=list(self._MENU),
        )


__all__ = ["FallbackResponder", "FallbackResponderResult"]
