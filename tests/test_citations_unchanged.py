from services import pamphlet_flow
from services.pamphlet_search import PamphletChunk, SearchResult


def _chunk(city: str, source: str, text: str) -> SearchResult:
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
    return SearchResult(chunk=chunk, score=0.9)


def test_citation_footer_unchanged():
    user_id = "tester"
    session_store = {}

    def fake_searcher(city, query, limit):
        return [_chunk(city, "history.txt", "遣唐使の寄港地について触れている。")]

    def fake_summarizer(question, docs, detailed=False):
        return (
            "### 要約\n"
            "遣唐使が寄港した記録が残っています。[[1]]\n\n"
            "### 出典\n"
            "- [[1]] 五島市 / history.txt"
        )

    response = pamphlet_flow.build_response(
        "五島市の遣唐使を知りたい",
        user_id=user_id,
        session_store=session_store,
        topk=3,
        ttl=1800,
        searcher=fake_searcher,
        summarizer=fake_summarizer,
    )

    assert response.sources_md.strip() == "### 出典\n- [[1]] 五島市 / history.txt"
