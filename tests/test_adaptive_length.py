from services.pamphlet_search import PamphletChunk, SearchResult
from services.pamphlet_summarize import summarize_with_gpt_nano


def _make_result(city: str, source: str, text: str, score: float = 0.9) -> SearchResult:
    chunk = PamphletChunk(
        city=city,
        source_file=source,
        chunk_index=0,
        text=text,
        char_start=0,
        char_end=len(text),
        line_start=1,
        line_end=text.count("\n") + 1,
    )
    return SearchResult(chunk=chunk, score=score)


def _section(body: str, title: str) -> str:
    if title not in body:
        return ""
    after = body.split(title, 1)[1]
    for marker in ("\n### ", "\n# "):
        if marker in after:
            after = after.split(marker, 1)[0]
            break
    return after.strip()


def _sample_docs() -> list[SearchResult]:
    base_text = (
        "804年に遣唐使船が寄港した記録が残っており、当時の外交航海では福江の港が補給地となりました。"
        "五島市は寄港船の水や食料を提供し、航路の安全を支えました。"
    )
    extra = (
        "航海の準備には僧や技術者が同行し、寄港後は現地での交流行事が開かれたと伝わっています。"
        "福江城下ではその歴史を紹介する展示が整備されています。"
    )
    detailed = (
        "奈良時代末期の遣唐使派遣は国家的事業であり、804年の船団は藤原冬嗣らが参加しました。"
        "旅程には長崎県の島々が含まれ、五島で風待ちを行った記録が『続日本紀』に残ります。"
        "航行を支える船団は、補給・修繕・祈祷を目的とした一行を伴いました。"
    )
    return [
        _make_result("goto", "history.txt", base_text, 0.95),
        _make_result("goto", "culture.txt", extra, 0.88),
        _make_result("goto", "chronicle.txt", detailed, 0.86),
    ]


def test_fact_question_prefers_short_summary():
    output = summarize_with_gpt_nano("遣唐使の派遣年を教えて", _sample_docs(), detailed=False)
    summary = _section(output, "### 要約")
    detail = _section(output, "### 詳細")

    assert output.startswith("### 要約")
    assert len(summary.replace("\n", "")) <= 200
    assert not detail


def test_detailed_request_respects_length_limit():
    output = summarize_with_gpt_nano("遣唐使の背景を詳しく説明して", _sample_docs(), detailed=True)
    summary = _section(output, "### 要約")
    detail = _section(output, "### 詳細")

    assert output.startswith("### 要約")
    assert len(output) <= 800
    assert detail
    assert detail.count("\n- ") <= 5
    assert summary.count("[[") >= 1
    assert "[[1]]" in output
