from services.pamphlet_rag import CitationRef, postprocess_answer


def test_invalid_labels_removed_from_answer():
    ref = CitationRef(
        doc_id="goto/guide.txt",
        chunk_id="1",
        title="guide",
        city="五島市",
        start_offset=0,
        end_offset=100,
        score=1.0,
        text="五島市の歴史は奈良時代に遣唐使の寄港地となったことに始まります。",
    )
    id_map = {1: ref}
    result = postprocess_answer("奈良時代の寄港地でした[[1]] 新情報です[[2]]", id_map, min_chars=0, min_score=0)

    assert result.used_labels == [1]
    assert "[[2]]" not in result.answer_with_labels
    assert result.citations[0]["labels"] == [1]
