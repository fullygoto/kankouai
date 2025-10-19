"""Session aware responder that implements the pamphlet summary flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from coreapp.llm import LLMClient, LLMRequest, get_llm_client
from coreapp.search.pamphlet_index import (
    PamphletFragment,
    PamphletIndex,
    load_pamphlet_index,
)
from services.pamphlet_constants import CITY_PROMPT


CITY_CHOICES: List[Dict[str, str]] = [
    {"key": "goto", "label": "五島市"},
    {"key": "shinkamigoto", "label": "新上五島町"},
    {"key": "ojika", "label": "小値賀町"},
    {"key": "uku", "label": "宇久町"},
]

CITY_ALIASES = {
    "五島市": "goto",
    "五島": "goto",
    "新上五島町": "shinkamigoto",
    "上五島": "shinkamigoto",
    "小値賀町": "ojika",
    "小値賀": "ojika",
    "宇久町": "uku",
    "宇久": "uku",
}


@dataclass
class PamphletResponderResult:
    kind: str
    message: str
    city: Optional[str] = None
    fragments: List[PamphletFragment] = field(default_factory=list)
    quick_replies: Optional[List[Dict[str, str]]] = None
    web_buttons: Optional[List[Dict[str, str]]] = None


class PamphletResponder:
    """Return pamphlet summaries based on the specification flow."""

    def __init__(
        self,
        pamphlets: List[Dict[str, str]] | None = None,
        *,
        index: PamphletIndex | None = None,
        llm_client: LLMClient | None = None,
        top_k: int = 4,
    ) -> None:
        self._index = index or load_pamphlet_index(pamphlets or [])
        self._llm = llm_client or get_llm_client()
        self._top_k = max(3, min(top_k, 5))

    @staticmethod
    def _city_from_message(message: str) -> Optional[str]:
        text = (message or "").strip()
        for label in sorted(CITY_ALIASES, key=len, reverse=True):
            if label in text:
                return CITY_ALIASES[label]
        return None

    @staticmethod
    def _city_label(key: str | None) -> str:
        for item in CITY_CHOICES:
            if item["key"] == key:
                return item["label"]
        return key or ""

    def _choices_payload(self) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        quick = [{"type": "city", "label": item["label"], "text": item["label"]} for item in CITY_CHOICES]
        web = [{"label": item["label"], "value": item["label"]} for item in CITY_CHOICES]
        return quick, web

    @staticmethod
    def _normalise_key(city: str, query: str) -> str:
        return f"{city}:{query.strip()}".strip().lower()

    def respond(self, message: str, *, context: Dict[str, Any] | None = None) -> PamphletResponderResult:
        state = context.setdefault("pamphlet", {}) if context is not None else {}
        text = (message or "").strip()
        if not text:
            return PamphletResponderResult(kind="noop", message="")

        pending_query = state.get("pending_query") if isinstance(state.get("pending_query"), str) else None

        # Determine if the incoming text selects a city from the pending query.
        city_key = None
        if text in CITY_ALIASES:
            city_key = CITY_ALIASES[text]
        else:
            city_key = self._city_from_message(text)

        if pending_query and text in {choice["label"] for choice in CITY_CHOICES}:
            city_key = CITY_ALIASES.get(text, city_key)
            text = pending_query
            pending_query = None

        if city_key is None:
            # Ask user to pick a city if not provided.
            state["pending_query"] = text
            quick, web = self._choices_payload()
            return PamphletResponderResult(
                kind="ask_city",
                message=CITY_PROMPT,
                quick_replies=quick,
                web_buttons=web,
            )

        query = text.replace(self._city_label(city_key), "").strip() or text
        city_label = self._city_label(city_key)
        state.pop("pending_query", None)

        answered = state.setdefault("answered", [])
        key = self._normalise_key(city_key, query)
        if key in answered:
            return PamphletResponderResult(kind="noop", message="", city=city_key)

        fragments = self._index.search(city_label, query, self._top_k)
        if not fragments:
            return PamphletResponderResult(
                kind="no_hit",
                message="パンフレットから該当する記述が見つかりませんでした。",
                city=city_key,
            )

        payload = [
            {"text": fragment.text, "source": fragment.source}
            for fragment in fragments
        ]
        summary = self._llm.complete(
            LLMRequest(
                prompt=query,
                parameters={"fragments": payload, "city_label": city_label, "query": query},
            )
        )

        answered.append(key)
        if len(answered) > 10:
            del answered[:-10]

        return PamphletResponderResult(
            kind="summary",
            message=summary,
            city=city_key,
            fragments=fragments,
        )


__all__ = ["CITY_CHOICES", "PamphletResponder", "PamphletResponderResult"]
