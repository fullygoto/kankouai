import re

import pytest

from services import pamphlet_flow, pamphlet_search


@pytest.fixture
def pamphlet_env(tmp_path, monkeypatch):
    base = tmp_path / "pamphlets"
    (base / "goto").mkdir(parents=True, exist_ok=True)
    (base / "goto" / "history.txt").write_text(
        "五島市の歴史や教会群について紹介する資料です。\n"
        "主要な観光スポットやアクセス、文化行事が詳しくまとめられています。\n",
        encoding="utf-8",
    )
    (base / "goto" / "food.txt").write_text(
        "五島うどんや海鮮料理、島内の飲食店情報を掲載した資料です。",
        encoding="utf-8",
    )

    monkeypatch.setenv("SUMMARY_STYLE", "polite_long")
    pamphlet_search.configure(
        {
            "PAMPHLET_BASE_DIR": str(base),
            "PAMPHLET_CHUNK_SIZE": 300,
            "PAMPHLET_CHUNK_OVERLAP": 20,
        }
    )
    pamphlet_search.reindex_all()
    return base


def _labels_in_message(text: str) -> set[int]:
    pattern = re.compile(r"\[\[(\d+)\]\]")
    return {int(m.group(1)) for m in pattern.finditer(text)}


def _sentences(text: str) -> list[str]:
    header_split = text.split("### 出典", 1)[0]
    parts = [seg for seg in re.split(r"(?<=[。！？!?])", header_split) if seg]
    sentences: list[str] = []
    buffer = ""
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        buffer = f"{buffer}{chunk}" if buffer else chunk
        if buffer.endswith("]]"):
            sentences.append(buffer)
            buffer = ""
    if buffer:
        sentences.append(buffer)
    return sentences


def test_each_sentence_has_label(pamphlet_env):
    session = {}
    res = pamphlet_flow.build_response(
        "五島市の歴史を詳しく教えて",
        user_id="u1",
        session_store=session,
        topk=2,
        ttl=600,
        searcher=lambda city, query, limit: pamphlet_search.search(city, query, limit),
        summarizer=lambda *args, **kwargs: "",
    )

    assert res.kind == "answer"
    sentences = _sentences(res.message)
    assert sentences, "summary should contain sentences"
    for sentence in sentences:
        assert sentence.endswith("]]")


def test_sources_match_used_labels(pamphlet_env):
    session = {}
    res = pamphlet_flow.build_response(
        "五島市のおすすめグルメ", user_id="u2", session_store=session, topk=2, ttl=600,
        searcher=lambda city, query, limit: pamphlet_search.search(city, query, limit),
        summarizer=lambda *args, **kwargs: "",
    )

    assert res.kind == "answer"
    labels = _labels_in_message(res.message)
    allowed_labels = {label for citation in res.citations for label in citation.get("labels", [])}
    assert labels.issubset(allowed_labels)
    expected_sources = {
        f"{citation.get('city')}/{citation.get('file') or citation.get('title')}"
        for citation in res.citations
    }
    assert set(res.sources) == expected_sources


def test_no_external_labels_when_not_available(pamphlet_env):
    session = {}
    res = pamphlet_flow.build_response(
        "五島市で宇宙旅行の方法を教えて",
        user_id="u3",
        session_store=session,
        topk=2,
        ttl=600,
        searcher=lambda city, query, limit: pamphlet_search.search(city, query, limit),
        summarizer=lambda *args, **kwargs: "",
    )

    assert res.kind in {"error", "answer"}
    if res.kind == "answer":
        labels = _labels_in_message(res.message)
        allowed_labels = {ref_label for citation in res.citations for ref_label in citation.get("labels", [])}
        assert labels.issubset(allowed_labels)
