import re

import pytest

from services import pamphlet_flow, pamphlet_rag, pamphlet_search


@pytest.fixture
def summary_env(tmp_path, monkeypatch):
    base = tmp_path / "pamphlets"
    (base / "goto").mkdir(parents=True, exist_ok=True)
    long_text = "五島市の自然や文化、観光スポットを詳しく紹介する資料です。" * 60
    (base / "goto" / "guide.txt").write_text(long_text, encoding="utf-8")
    short_text = "行事予定とお問い合わせ先を簡潔にまとめた資料です。"
    (base / "goto" / "short.txt").write_text(short_text, encoding="utf-8")

    monkeypatch.setenv("SUMMARY_STYLE", "polite_long")
    monkeypatch.setenv("SUMMARY_MIN_CHARS", "550")
    monkeypatch.setenv("SUMMARY_MAX_CHARS", "800")
    monkeypatch.setenv("SUMMARY_MIN_FALLBACK", "300")

    pamphlet_search.configure(
        {
            "PAMPHLET_BASE_DIR": str(base),
            "PAMPHLET_CHUNK_SIZE": 400,
            "PAMPHLET_CHUNK_OVERLAP": 40,
        }
    )
    pamphlet_search.reindex_all()
    return base


def _summary_text(message: str) -> str:
    return message.split("### 出典", 1)[0].strip()


def _length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def test_standard_summary_respects_bounds(summary_env):
    session = {}
    res = pamphlet_flow.build_response(
        "五島市の観光情報を詳しく教えて",
        user_id="s1",
        session_store=session,
        topk=2,
        ttl=600,
        searcher=lambda city, query, limit: pamphlet_search.search(city, query, limit),
        summarizer=lambda *args, **kwargs: "",
    )

    assert res.kind == "answer"
    summary = _summary_text(res.message)
    length = _length(summary)
    assert 550 <= length <= 800


def test_short_context_uses_fallback_min(summary_env):
    session = {}
    res = pamphlet_flow.build_response(
        "五島市の行事予定について",
        user_id="s2",
        session_store=session,
        topk=1,
        ttl=600,
        searcher=lambda city, query, limit: pamphlet_search.search(city, query, limit),
        summarizer=lambda *args, **kwargs: "",
    )

    summary = _summary_text(res.message)
    assert _length(summary) >= 300


def test_overlong_generation_is_truncated(summary_env, monkeypatch):
    session = {}

    def fake_generate(prompt_cfg):
        sentence = "五島列島の魅力を紹介します。[[1]]"
        return sentence * 120

    monkeypatch.setattr(pamphlet_rag, "_generate_with_constraints", fake_generate)

    res = pamphlet_flow.build_response(
        "五島市で長い説明をお願い",
        user_id="s3",
        session_store=session,
        topk=1,
        ttl=600,
        searcher=lambda city, query, limit: pamphlet_search.search(city, query, limit),
        summarizer=lambda *args, **kwargs: "",
    )

    summary = _summary_text(res.message)
    length = _length(summary)
    assert length <= 800
    assert summary.endswith("]]")
