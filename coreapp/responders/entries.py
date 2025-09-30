"""Responders for the tourism entry index."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from coreapp.search.entries_index import EntriesIndex, EntryMatch, load_entries_index


@dataclass
class EntriesResponderResult:
    """Container returned by :class:`EntriesResponder`."""

    kind: str
    message: str
    entry: Optional[Dict[str, Any]] = None
    quick_replies: Optional[List[Dict[str, str]]] = None
    image_url: Optional[str] = None


class EntriesResponder:
    """High level helper that prepares answers for tourism entries."""

    def __init__(
        self,
        entries: Iterable[Dict[str, Any]] | None = None,
        *,
        index: EntriesIndex | None = None,
    ) -> None:
        if index is not None:
            self._index = index
        else:
            self._index = load_entries_index(entries)

    @staticmethod
    def _parse_filter_command(message: str) -> Dict[str, str] | None:
        text = (message or "").strip()
        if not text.startswith("@"):
            return None
        body = text[1:]
        if ":" not in body:
            return None
        prefix, value = body.split(":", 1)
        prefix = prefix.strip().lower()
        value = value.strip()
        if prefix not in {"area", "tag"}:
            return None
        if not value:
            return None
        return {"type": prefix, "value": value}

    def respond(
        self,
        message: str,
        *,
        context: Dict[str, Any] | None = None,
    ) -> EntriesResponderResult:
        state = context if context is not None else {}

        command = self._parse_filter_command(message)
        if command and state.get("base_query"):
            filters = dict(state.get("filters") or {})
            filters[command["type"]] = command["value"]
            matches = self._index.search(state["base_query"], filters=filters)
            if not matches:
                state["filters"] = filters
                return EntriesResponderResult(
                    kind="no_hit",
                    message="条件に合うスポットが見つかりませんでした。",
                )
            if len(matches) == 1:
                state.clear()
                return self._build_detail(matches[0])
            state["filters"] = filters
            return self._build_choices(matches)

        # New query or state reset.
        state.clear()
        tokens = (message or "").strip()
        matches = self._index.search(tokens)
        if not matches:
            return EntriesResponderResult(
                kind="no_hit",
                message="観光データから該当する情報は見つかりませんでした。",
            )
        if len(matches) == 1:
            return self._build_detail(matches[0])

        state["base_query"] = tokens
        state["filters"] = {}
        return self._build_choices(matches)

    def _build_detail(self, match: EntryMatch) -> EntriesResponderResult:
        entry = match.entry
        lines: List[str] = []
        title = (entry.get("title") or "").strip()
        if title:
            lines.append(title)
        desc = (entry.get("desc") or "").strip()
        if desc:
            clean = " ".join(desc.split())
            lines.append(clean)

        def _field(label: str, value: Any) -> None:
            if isinstance(value, list):
                clean = " / ".join(str(item).strip() for item in value if str(item).strip())
            else:
                clean = str(value or "").strip()
            if clean:
                lines.append(f"{label}：{clean}")

        _field("住所", entry.get("address"))
        _field("電話", entry.get("tel"))
        _field("営業時間", entry.get("open_hours") or entry.get("hours"))
        _field("定休日", entry.get("holiday") or entry.get("closed"))
        _field("料金", entry.get("price"))
        _field("タグ", entry.get("tags") or [])
        _field("エリア", entry.get("areas") or [])

        image_url = self._first_image(entry)
        return EntriesResponderResult(
            kind="detail",
            message="\n".join(lines),
            entry=entry,
            image_url=image_url,
        )

    @staticmethod
    def _first_image(entry: Dict[str, Any]) -> str | None:
        for key in ("image", "image_url", "map", "url"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        images = entry.get("images")
        if isinstance(images, list):
            for image in images:
                if isinstance(image, str) and image.strip():
                    return image.strip()
                if isinstance(image, dict):
                    for key in ("url", "image", "src", "path"):
                        value = image.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
        return None

    def _build_choices(self, matches: List[EntryMatch]) -> EntriesResponderResult:
        quick_replies: List[Dict[str, str]] = []
        for option in self._index.top_filters(matches, limit=5):
            label_prefix = "エリア" if option["type"] == "area" else "タグ"
            value = option["value"]
            payload = f"@{option['type']}:{value}"
            quick_replies.append(
                {
                    "type": option["type"],
                    "value": value,
                    "label": f"{label_prefix}: {value}",
                    "payload": payload,
                }
            )

        summary_lines = [
            "候補が複数見つかりました。",
            "エリアやタグで絞り込んでください。",
        ]
        for match in matches[:3]:
            entry = match.entry
            title = (entry.get("title") or "").strip()
            areas = [str(area).strip() for area in (entry.get("areas") or []) if str(area).strip()]
            suffix = f"（{' / '.join(areas)}）" if areas else ""
            desc = " ".join(str(entry.get("desc") or "").split())
            snippet = desc[:60].rstrip("、。・ ") if desc else ""
            if snippet:
                summary_lines.append(f"- {title}{suffix}：{snippet}")
            else:
                summary_lines.append(f"- {title}{suffix}")

        return EntriesResponderResult(
            kind="choices",
            message="\n".join(summary_lines),
            quick_replies=quick_replies,
        )


__all__ = ["EntriesResponder", "EntriesResponderResult"]
