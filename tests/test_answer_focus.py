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


def _extract_section(body: str, title: str) -> str:
    if title not in body:
        return ""
    after = body.split(title, 1)[1]
    for marker in ("\n### ", "\n# "):
        if marker in after:
            after = after.split(marker, 1)[0]
            break
    return after.strip()


def test_summary_focuses_on_target_city_and_topic():
    docs = [
        _make_result(
            "goto",
            "history.txt",
            "804年に遣唐使船が寄港した記録が残り、奈良時代末期の外交航海が五島を通じて行われました。"
            "福江島では寄港地を整備し、航路の安全を支えました。",
            0.95,
        ),
        _make_result(
            "goto",
            "culture.txt",
            "福江港周辺では遣唐使の往来を伝える史跡が残り、当時の交流を紹介する展示が行われています。",
            0.83,
        ),
        _make_result(
            "ojika",
            "timeline.txt",
            "小値賀町の教会年表と祭礼の歴史をまとめた資料です。",
            0.8,
        ),
    ]

    output = summarize_with_gpt_nano("遣唐使が派遣された時代は？", docs)
    assert output.startswith("### 要約")

    summary_section = _extract_section(output, "### 要約")
    summary_lines = [line for line in summary_section.splitlines() if line.strip()]
    assert summary_lines, output
    first_line = summary_lines[0]
    assert "遣唐使" in first_line
    assert "804" in first_line
    assert first_line.strip().endswith("]]"), first_line
    assert "小値賀" not in summary_section
    assert "[[3]]" not in output

    detail_section = _extract_section(output, "### 詳細")
    if detail_section:
        assert "小値賀" not in detail_section

    sources_section = _extract_section(output, "### 出典")
    assert "[[1]]" in sources_section
    assert "timeline" not in sources_section
