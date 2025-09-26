"""Minimal RAG pipeline demo used in CI smoke runs."""

from services import pamphlet_rag, pamphlet_search


def run_demo() -> dict:
    chunk = pamphlet_search.PamphletChunk(
        city="goto",
        source_file="demo.txt",
        chunk_index=0,
        text="五島市の観光スポットを紹介するパンフレットです。",
        char_start=0,
        char_end=30,
        line_start=1,
        line_end=3,
    )
    candidate = pamphlet_rag._Candidate(
        chunk=chunk,
        combined_score=0.4,
        bm25_details=[],
        embed_details=[],
        vector_index=None,
    )
    context, id_map = pamphlet_rag.build_context_with_labels([candidate])
    prompt = pamphlet_rag._build_prompt("観光スポットは？", "goto", ["観光"], context)
    mocked_answer = (
        "### 要約\n五島市のパンフレットでは主要な観光地が紹介されています。[[1]]\n\n"
        "### 出典\n- 五島市/demo[[1]]"
    )
    result = pamphlet_rag.postprocess_answer(
        mocked_answer,
        id_map,
        min_chars=1,
        min_score=0.0,
    )
    return {
        "prompt": prompt,
        "context": context,
        "answer_without_labels": result.answer_without_labels,
        "answer_with_labels": result.answer_with_labels,
        "citations": result.citations,
    }


if __name__ == "__main__":
    demo = run_demo()
    for key, value in demo.items():
        print(f"=== {key} ===")
        if isinstance(value, list):
            for item in value:
                print(item)
        else:
            print(value)
