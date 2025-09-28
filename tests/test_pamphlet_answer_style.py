from services.pamphlet_search import PamphletChunk, SearchResult
from services.pamphlet_summarize import summarize_with_gpt_nano


def _result(city: str, source: str, text: str, score: float = 0.9) -> SearchResult:
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


def test_pamphlet_summary_structure():
    docs = [
        _result(
            "goto",
            "history.txt",
            "630年から約260年の間に遣唐使が20回ほど派遣され、五島は最終寄港地として整備された。",
            0.95,
        ),
        _result(
            "goto",
            "purpose.txt",
            "遣唐使は唐の政治制度や仏教を学ぶために派遣され、乗船者は命がけで往復した。",
            0.9,
        ),
    ]

    output = summarize_with_gpt_nano("遣唐使が派遣された時代について教えて", docs)
    assert output.startswith("### 要約"), output
    assert "### 出典" in output, output

    summary_section = _extract_section(output, "### 要約")
    assert "遣唐使" in summary_section
    assert "最終寄港地" in summary_section
    assert "教会" not in summary_section

    if "### 詳細" in output:
        detail_section = _extract_section(output, "### 詳細")
        assert detail_section, "詳細セクションが空です"

    assert output.index("### 要約") < output.index("### 出典"), output
    if "### 詳細" in output:
        assert output.index("### 要約") < output.index("### 詳細") < output.index("### 出典"), output
