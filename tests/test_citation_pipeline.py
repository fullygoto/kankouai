import textwrap

import pytest

from services import pamphlet_rag, pamphlet_search


@pytest.fixture
def sample_id_map():
    ref1 = pamphlet_rag.CitationRef(
        doc_id="goto/history.txt",
        chunk_id="0",
        title="history",
        city="五島市",
        start_offset=0,
        end_offset=50,
        score=0.32,
        text="長い説明が含まれるテキストです。",
    )
    ref2 = pamphlet_rag.CitationRef(
        doc_id="goto/festival.txt",
        chunk_id="1",
        title="festival",
        city="五島市",
        start_offset=50,
        end_offset=90,
        score=0.05,
        text="夏祭りの概要",
    )
    return {1: ref1, 2: ref2}


def test_build_context_with_labels(sample_id_map):
    chunk = pamphlet_search.PamphletChunk(
        city="goto",
        source_file="history.txt",
        chunk_index=0,
        text="長い説明が続きます。",
        char_start=0,
        char_end=10,
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

    assert "[[1]]" in context
    assert "五島市/history" in context
    assert id_map[1].doc_id == "goto/history.txt"
    assert id_map[1].title == "history"


def test_postprocess_applies_thresholds(sample_id_map):
    answer = textwrap.dedent(
        """
        ### 要約
        長い説明をまとめました。[[1]]

        ### 詳細
        - 補[[2]]

        ### 出典
        - 五島市/history
        """
    ).strip()

    processed = pamphlet_rag.postprocess_answer(
        answer,
        sample_id_map,
        min_chars=30,
        min_score=0.2,
    )

    assert processed.answer_with_labels.startswith("### 要約")
    assert processed.answer_without_labels.startswith("### 要約")
    assert len(processed.citations) == 1
    assert processed.citations[0]["doc_id"] == "goto/history.txt"
    assert processed.used_labels == [1]


def test_postprocess_missing_labels_adds_note(sample_id_map):
    processed = pamphlet_rag.postprocess_answer(
        "根拠のない文章です。[[9]]",
        sample_id_map,
        min_chars=5,
        min_score=0.1,
    )

    assert processed.answer_with_labels == ""
    assert "根拠ラベルが見つかりませんでした" in processed.answer_without_labels
    assert processed.citations == []
    assert processed.used_labels == []
    assert processed.invalid_labels == [9]
