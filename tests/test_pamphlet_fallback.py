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
    session = {}
    res = pamphlet_flow.build_response(
        "五島市の歴史を教えて",
        user_id="u1",
        session_store=session,
        topk=3,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=lambda *args, **kwargs: "",
    )

    assert res.kind == "answer"
    assert res.city == "goto"
    assert "要約" in res.message
    assert "詳細" in res.message
    assert any("history_guide_2025.txt" in src for src in res.sources)
    assert res.more_available is False


def test_unknown_city_triggers_quick_reply_then_selection(pamphlet_base):
    session = {}

    first = pamphlet_flow.build_response(
        "おすすめの観光情報は？",
        user_id="u2",
        session_store=session,
        topk=3,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=lambda *args, **kwargs: "",
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
        summarizer=lambda *args, **kwargs: "",
    )
    assert second.kind == "ask_city"
    assert second.quick_choices == []

    # 市町が選択されたら pending クエリで検索される
    answer = pamphlet_flow.build_response(
        "五島市",
        user_id="u2",
        session_store=session,
        topk=2,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=lambda *args, **kwargs: "",
    )
    assert answer.kind == "answer"
    assert answer.city == "goto"
    assert any("history_guide_2025.txt" in src for src in answer.sources)

    more = pamphlet_flow.build_response(
        "もっと詳しく",
        user_id="u2",
        session_store=session,
        topk=2,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=lambda *args, **kwargs: "",
    )
    assert more.kind == "answer"
    assert more.more_available is False


def test_error_when_no_documents_returns_message(pamphlet_base):
    session = {}

    res = pamphlet_flow.build_response(
        "宇久町の見どころは？",
        user_id="u3",
        session_store=session,
        topk=3,
        ttl=1800,
        searcher=_search_wrapper,
        summarizer=lambda *args, **kwargs: "",
    )
    assert res.kind == "error"
    assert "資料に該当する記述が見当たりません" in res.message
