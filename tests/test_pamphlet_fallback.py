import os
from typing import Iterable

import pytest

from services import pamphlet_flow, pamphlet_search


@pytest.fixture
def pamphlet_base(tmp_path, monkeypatch):
    base = tmp_path / "pamphlets"
    for city in pamphlet_search.CITY_KEYS:
        (base / city).mkdir(parents=True, exist_ok=True)
    (base / "goto" / "history_guide_2025.txt").write_text(
        "五島市の歴史を紹介するパンフレットです。教会や海の文化について詳しく説明しています。"
        "おすすめの散策コースも掲載されています。",
        encoding="utf-8",
    )
    (base / "goto" / "festivals.txt").write_text(
        "五島市では夏祭りや花火大会が開催され、家族連れに人気です。海鮮グルメも楽しめます。",
        encoding="utf-8",
    )
    (base / "shinkamigoto" / "island_guide.txt").write_text(
        "新上五島町の教会群や釣り体験、特産品についてまとめた資料です。",
        encoding="utf-8",
    )

    pamphlet_search.configure(
        {
            "PAMPHLET_BASE_DIR": str(base),
            "PAMPHLET_CHUNK_SIZE": 400,
            "PAMPHLET_CHUNK_OVERLAP": 80,
        }
    )
    pamphlet_search.reindex_all()
    return base


def _search_wrapper(city: str, query: str, topk: int):
    return pamphlet_search.search(city, query, topk)


def test_city_name_query_returns_summary(pamphlet_base):
    calls = {}

    def fake_summarizer(query: str, docs: Iterable, detailed: bool = False) -> str:
        docs = list(docs)
        calls["docs"] = docs
        calls["detailed"] = detailed
        names = ", ".join(d.chunk.source_file for d in docs)
        return f"{query}のポイント: {names}"

    session = {}
    res = pamphlet_flow.build_response(
        "五島市の歴史を教えて",
        user_id="u1",
        session_store=session,
        topk=3,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=fake_summarizer,
    )

    assert res.kind == "answer"
    assert res.city == "goto"
    assert "出典（" in res.message
    assert "history_guide_2025.txt" in res.message
    assert res.more_available is True
    assert len(calls["docs"]) <= 3
    assert calls["detailed"] is False


def test_unknown_city_triggers_quick_reply_then_selection(pamphlet_base):
    session = {}

    def fake_summary(query: str, docs: Iterable, detailed: bool = False) -> str:
        return "サマリー"

    first = pamphlet_flow.build_response(
        "おすすめの観光情報は？",
        user_id="u2",
        session_store=session,
        topk=3,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=fake_summary,
    )

    assert first.kind == "ask_city"
    assert len(first.quick_choices) == 4

    # 同じ質問を再度送ると、案内だけでクイックリプライは抑制される
    second = pamphlet_flow.build_response(
        "おすすめの観光情報は？",
        user_id="u2",
        session_store=session,
        topk=3,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=fake_summary,
    )
    assert second.kind == "ask_city"
    assert second.quick_choices == []

    # 市町が選択されたら pending クエリで検索される
    outputs = []

    def detailed_summary(query: str, docs: Iterable, detailed: bool = False) -> str:
        outputs.append(detailed)
        return "詳細サマリー" if detailed else "通常サマリー"

    answer = pamphlet_flow.build_response(
        "五島市",
        user_id="u2",
        session_store=session,
        topk=2,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=detailed_summary,
    )
    assert answer.kind == "answer"
    assert answer.city == "goto"
    assert "出典（" in answer.message
    assert outputs[-1] is False

    more = pamphlet_flow.build_response(
        "もっと詳しく",
        user_id="u2",
        session_store=session,
        topk=2,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=detailed_summary,
    )
    assert more.kind == "answer"
    assert more.more_available is False
    assert outputs[-1] is True


def test_error_when_no_documents_returns_message(pamphlet_base):
    session = {}

    def fail_summary(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("summary should not be called when no documents")

    res = pamphlet_flow.build_response(
        "宇久町の見どころは？",
        user_id="u3",
        session_store=session,
        topk=3,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=fail_summary,
    )
    assert res.kind == "error"
    assert "宇久町" in res.message
