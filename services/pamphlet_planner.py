"""Intent and style planner for pamphlet answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


_QUESTION_PREFIX = re.compile(r"^(?:[A-Za-z]+|[\wぁ-んァ-ン一-龠々〆ゝゞ]+)[:：]\s*")
_WH_WORDS = {"いつ", "どこ", "何", "なに", "なぜ", "どう", "どんな", "どの", "だれ", "誰"}
_FACT_HINTS = {
    "いつ",
    "どこ",
    "時期",
    "期間",
    "開催日",
    "何年",
    "年代",
    "誰",
    "だれ",
    "料金",
    "値段",
    "費用",
    "アクセス",
    "行き方",
    "所要",
    "時間",
    "住所",
    "電話",
    "人数",
    "規模",
}
_DETAIL_HINTS = {
    "詳しく",
    "ポイント",
    "楽しみ方",
    "モデルコース",
    "おすすめ",
    "初めて",
    "注意点",
    "過ごし方",
    "体験",
    "計画",
    "見どころ",
    "魅力",
}
_LIST_HINTS = {"一覧", "リスト"}
_COMPARE_HINTS = {"違い", "比較", "比べ"}
_HOWTO_HINTS = {"どうやって", "方法", "行き方", "アクセス", "参加", "申し込み"}

_STOPWORDS = {
    "について",
    "こと",
    "もの",
    "教えて",
    "ください",
    "出来ます",
    "できます",
    "です",
    "ます",
    "よう",
    "したい",
    "下さい",
    "ください",
    "とは",
    "を",
    "が",
    "は",
    "に",
    "で",
    "と",
    "へ",
    "も",
    "の",
    "や",
    "より",
    "など",
    "って",
    "ってる",
}


@dataclass
class Plan:
    intent: str
    scope: Dict[str, List[str]]
    style: str
    sections: List[str]
    clarification: Optional[str] = None
    query: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "intent": self.intent,
            "scope": dict(self.scope),
            "style": self.style,
            "sections": list(self.sections),
            "clarification": self.clarification,
            "query": self.query,
        }


def _strip_sender_prefix(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return ""
    cleaned = _QUESTION_PREFIX.sub("", stripped, count=1)
    return cleaned.strip()


def _detect_intent(query: str) -> str:
    text = query or ""
    if any(token in text for token in _COMPARE_HINTS):
        return "compare"
    if any(token in text for token in _LIST_HINTS):
        return "list"
    if any(token in text for token in _HOWTO_HINTS):
        return "howto"
    if any(token in text for token in _FACT_HINTS):
        return "fact"
    if "概要" in text or "どんな" in text:
        return "overview"
    return "overview"


def _select_style(query: str) -> str:
    text = query or ""
    if any(token in text for token in _DETAIL_HINTS):
        return "long_explanatory"
    if any(token in text for token in _FACT_HINTS):
        return "short_direct"
    if len(text) <= 15 and any(token in text for token in _WH_WORDS):
        return "short_direct"
    if "概要" in text or "まとめ" in text:
        return "medium_structured"
    return "medium_structured"


def _collect_tokens(query: str) -> List[str]:
    cleaned = re.sub(r"[\s、。,.!?！？\-ー（）()\[\]【】『』\n\r]+", " ", query or "").strip()
    if not cleaned:
        return []
    matches = re.findall(r"[A-Za-z0-9一-龠々〆ヵヶぁ-んァ-ンゝゞー]{2,}", cleaned)
    tokens: List[str] = []
    for item in matches:
        if item in _STOPWORDS:
            continue
        tokens.append(item)
    return tokens


def _extract_scope(query: str, ctx_meta: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    scope: Dict[str, List[str]] = {}
    tokens = _collect_tokens(query)
    keywords: List[str] = []
    for token in tokens:
        if token in _WH_WORDS or token in _STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        keywords.append(token)
    if keywords:
        scope["keywords"] = keywords

    eras = re.findall(r"(奈良|平安|鎌倉|室町|江戸|明治|大正|昭和|平成|令和)", query)
    years = re.findall(r"[0-9]{3,4}年", query)
    time_scope: List[str] = []
    if eras:
        time_scope.extend(sorted(set(eras)))
    if years:
        time_scope.extend(sorted(set(years)))
    if time_scope:
        scope["time"] = time_scope

    if ctx_meta and ctx_meta.get("city"):
        scope.setdefault("city", [ctx_meta["city"]])

    return scope


def _choose_sections(style: str, intent: str) -> List[str]:
    if style == "short_direct":
        return ["結論", "補足"]
    if style == "medium_structured":
        return ["結論", "要点", "補足"]
    if style == "long_explanatory":
        if intent == "howto":
            return ["結論", "手順", "注意"]
        return ["結論", "要点", "注意"]
    return ["結論"]


def plan_answer(query: str, ctx_meta: Optional[Dict[str, str]] = None) -> Plan:
    stripped = _strip_sender_prefix(query)
    intent = _detect_intent(stripped)
    style = _select_style(stripped)
    scope = _extract_scope(stripped, ctx_meta)
    sections = _choose_sections(style, intent)
    clarification: Optional[str] = None
    if not (ctx_meta or {}).get("city"):
        clarification = "city"

    return Plan(
        intent=intent,
        scope=scope,
        style=style,
        sections=sections,
        clarification=clarification,
        query=stripped,
    )


__all__ = ["Plan", "plan_answer"]

