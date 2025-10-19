"""Priority responder implementation for immediate answers."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Optional
from urllib.parse import quote

from coreapp.intent import (
    TRANSPORT_FLIGHT_KEYWORDS,
    TRANSPORT_SHIP_KEYWORDS,
    is_transport_query,
    is_viewpoint_map_query,
    is_weather_query,
)
from coreapp.intent import normalize_for_matching


logger = logging.getLogger(__name__)


FALLBACK_MESSAGE = "すみません、即答リンクの取得に失敗しました。時間をおいてもう一度お試しください。"


WEATHER_MESSAGE = (
    "【五島列島の主な天気情報リンク】\n"
    "五島市: https://weathernews.jp/onebox/tenki/nagasaki/42211/\n"
    "新上五島町: https://weathernews.jp/onebox/tenki/nagasaki/42411/\n"
    "小値賀町: https://tenki.jp/forecast/9/45/8440/42383/\n"
    "宇久町: https://weathernews.jp/onebox/33.262381/129.131027/q=%E9%95%B7%E5%B4%8E%E7%9C%8C%E4%BD%90%E4%B8%96%E4%BF%9D%E5%B8%82%E5%AE%87%E4%B9%85%E7%94%BA&v=da56215a2617fc2203c6cae4306d5fd8c92e3e26c724245d91160a4b3597570a&lang=ja&type=week"
)

SHIP_SECTION = (
    "【長崎ー五島航路 運行状況】\n"
    "・野母商船「フェリー太古」運航情報  \n"
    "  http://www.norimono-info.com/frame_set.php?usri=&disp=group&type=ship\n"
    "・九州商船「フェリー・ジェットフォイル」運航情報  \n"
    "  https://kyusho.co.jp/status\n"
    "・五島産業汽船「フェリー」運航情報  \n"
    "  https://www.goto-sangyo.co.jp/\n"
    "その他の航路や詳細は各リンクをご覧ください。"
)

FLIGHT_SECTION = (
    "五島つばき空港の最新の運行状況は、公式Webサイトでご確認いただけます。\n"
    "▶ https://www.fukuekuko.jp/"
)


def weather_reply_text() -> str:
    """Return the fixed weather response text."""

    return WEATHER_MESSAGE


def transport_reply_text(message: str | None) -> str:
    """Return the transport response text based on the query contents."""

    normalized = normalize_for_matching(message)
    wants_ship = any(keyword in normalized for keyword in TRANSPORT_SHIP_KEYWORDS)
    wants_fly = any(keyword in normalized for keyword in TRANSPORT_FLIGHT_KEYWORDS)

    if not wants_ship and not wants_fly:
        # クエリが「運行状況」だけ等の場合は両方提示する
        wants_ship = True
        wants_fly = True

    parts = []
    if wants_ship:
        parts.append(SHIP_SECTION)
    if wants_fly:
        parts.append(FLIGHT_SECTION)

    return "\n\n".join(parts) if parts else FALLBACK_MESSAGE


def _viewpoints_map_url() -> str:
    direct = os.getenv("VIEWPOINTS_URL", "").strip()
    if direct:
        return direct

    mid = os.getenv("VIEWPOINTS_MID", "").strip()
    if not mid:
        return ""

    base = f"https://www.google.com/maps/d/viewer?mid={quote(mid)}&femb=1"

    ll = os.getenv("VIEWPOINTS_LL", "").strip()
    ll_part = ""
    if ll:
        try:
            lat, lng = [segment.strip() for segment in ll.split(",", 1)]
            ll_part = f"&ll={quote(lat)},{quote(lng)}"
        except Exception:
            logger.debug("Failed to parse VIEWPOINTS_LL: %s", ll)

    z = os.getenv("VIEWPOINTS_ZOOM", "").strip()
    zoom_part = ""
    if z:
        try:
            zoom_part = f"&z={int(z)}"
        except Exception:
            logger.debug("Invalid VIEWPOINTS_ZOOM: %s", z)

    return f"{base}{ll_part}{zoom_part}"


def viewpoint_map_reply_text() -> str:
    url = _viewpoints_map_url()
    if not url:
        return FALLBACK_MESSAGE
    return f"展望所マップはこちら：\n{url}"


@dataclass(frozen=True)
class PriorityAnswer:
    """Container for priority responder answers."""

    kind: str
    message: str


class PriorityResponder:
    """Responder that returns deterministic priority answers."""

    fallback_message: str = FALLBACK_MESSAGE

    def answer(self, message: str | None, *, label: Optional[str] = None) -> Optional[PriorityAnswer]:
        """Return a :class:`PriorityAnswer` for recognised priority intents."""

        effective_label = label or self._infer_label(message)
        if effective_label is None:
            return None

        try:
            if effective_label == "weather":
                if not is_weather_query(message):
                    return None
                return PriorityAnswer(kind="weather", message=weather_reply_text())

            if effective_label == "transport":
                if not is_transport_query(message):
                    return None
                return PriorityAnswer(kind="transport", message=transport_reply_text(message))

            if effective_label == "viewpoint_map":
                if not is_viewpoint_map_query(message):
                    return None
                return PriorityAnswer(kind="map", message=viewpoint_map_reply_text())

        except Exception:  # pragma: no cover - defensive guard
            logger.exception("priority responder failed: label=%s", effective_label)
            return PriorityAnswer(kind=effective_label, message=self.fallback_message)

        return PriorityAnswer(kind=effective_label, message=self.fallback_message)

    @staticmethod
    def _infer_label(message: str | None) -> Optional[str]:
        if is_weather_query(message):
            return "weather"
        if is_transport_query(message):
            return "transport"
        if is_viewpoint_map_query(message):
            return "viewpoint_map"
        return None


__all__ = [
    "FALLBACK_MESSAGE",
    "PriorityAnswer",
    "PriorityResponder",
    "transport_reply_text",
    "viewpoint_map_reply_text",
    "weather_reply_text",
]
