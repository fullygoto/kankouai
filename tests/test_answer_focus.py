import pytest

from services import pamphlet_rag, pamphlet_search


@pytest.fixture
def focus_env(tmp_path, monkeypatch):
    base = tmp_path / "pamphlets"
    (base / "goto").mkdir(parents=True, exist_ok=True)
    text = (
        "遣唐使が630年ごろから派遣され、五島市から唐へ向かった記録があります。"
        "航海の準備や寄港地としての役割も紹介されています。"
    )
    (base / "goto" / "history.txt").write_text(text, encoding="utf-8")

    monkeypatch.setenv("SUMMARY_MODE", "adaptive")
    monkeypatch.setenv("CITATION_MIN_CHARS", "0")
    monkeypatch.setenv("CITATION_MIN_SCORE", "0")
    monkeypatch.setattr(pamphlet_rag, "_CITATION_MIN_CHARS", 0)
    monkeypatch.setattr(pamphlet_rag, "_CITATION_MIN_SCORE", 0)

    pamphlet_search.configure(
        {
            "PAMPHLET_BASE_DIR": str(base),
            "PAMPHLET_CHUNK_SIZE": 400,
            "PAMPHLET_CHUNK_OVERLAP": 40,
        }
    )
    pamphlet_search.reindex_all()

    yield


def test_answer_focus_removes_irrelevant_lines(focus_env, monkeypatch):
    def noisy_generate(prompt_cfg):
        return (
            "### 要約\n"
            "遣唐使は630年ごろから唐へ派遣されました[[1]]\n"
            "教会年表: 1895年創建[[1]]\n\n"
            "### 詳細\n"
            "- 804年に遣唐使船が五島を出航しました[[1]]\n"
            "- 教会群の世界遺産登録が続きます[[1]]\n\n"
            "### 出典\n"
            "- 五島市/history.txt"
        )

    monkeypatch.setattr(pamphlet_rag, "_generate_with_constraints", noisy_generate)

    answer = pamphlet_rag.answer_from_pamphlets("遣唐使の時代は？", "goto")
    labelled = answer.get("answer_with_labels", "")

    assert "教会年表" not in labelled
    assert "世界遺産" not in labelled
    assert "遣唐使" in labelled


def test_optional_details_section_omitted(focus_env, monkeypatch):
    def minimal_generate(prompt_cfg):
        return (
            "### 要約\n"
            "遣唐使の最終寄港地として知られています[[1]]\n\n"
            "### 詳細\n\n"
            "### 出典\n"
            "- 五島市/history.txt"
        )

    monkeypatch.setattr(pamphlet_rag, "_generate_with_constraints", minimal_generate)

    answer = pamphlet_rag.answer_from_pamphlets("遣唐使について", "goto")
    labelled = answer.get("answer_with_labels", "")

    assert "### 詳細" not in labelled


def test_citations_used_only(focus_env, monkeypatch):
    def extra_reference_generate(prompt_cfg):
        return (
            "### 要約\n"
            "遣唐使の航路が紹介されています[[1]]\n\n"
            "### 出典\n"
            "- 五島市/history.txt\n"
            "- 五島市/extra.txt"
        )

    monkeypatch.setattr(pamphlet_rag, "_generate_with_constraints", extra_reference_generate)

    answer = pamphlet_rag.answer_from_pamphlets("遣唐使の航路は？", "goto")
    labelled = answer.get("answer_with_labels", "")

    assert "- 五島市/extra" not in labelled
    assert "五島市/history" in labelled
    assert len(answer.get("citations", [])) == 1
