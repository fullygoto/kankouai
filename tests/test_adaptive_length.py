import re

import pytest

from services import pamphlet_rag, pamphlet_search


@pytest.fixture
def adaptive_env(tmp_path, monkeypatch):
    base = tmp_path / "pamphlets"
    (base / "goto").mkdir(parents=True, exist_ok=True)
    text = (
        "五島市の歴史をまとめたパンフレットです。遣唐使が寄港した804年の記録や"
        "福江港の開港について説明しています。初めて訪れる方向けの見どころも紹介しています。"
    ) * 10
    (base / "goto" / "history.txt").write_text(text, encoding="utf-8")

    monkeypatch.setenv("SUMMARY_MODE", "adaptive")
    monkeypatch.setenv("CITATION_MIN_CHARS", "0")
    monkeypatch.setenv("CITATION_MIN_SCORE", "0")

    pamphlet_search.configure(
        {
            "PAMPHLET_BASE_DIR": str(base),
            "PAMPHLET_CHUNK_SIZE": 400,
            "PAMPHLET_CHUNK_OVERLAP": 40,
        }
    )
    pamphlet_search.reindex_all()

    yield


def _fake_generate(prompt_cfg):
    style = prompt_cfg.style
    if style == "short_direct":
        return "遣唐使の寄港地として知られています[[1]] 奈良時代の交流が残ります[[1]]"
    if style == "medium_structured":
        return "奈良時代の役割を紹介します[[1]]\n- 港の役目が記されています[[1]]"
    return (
        "遣唐使との交流背景を説明します[[1]]\n"
        "奈良時代からの歴史や見どころを2文でまとめます[[1]]"
    )


def test_short_direct_is_two_sentences(adaptive_env, monkeypatch):
    monkeypatch.setattr(pamphlet_rag, "_generate_with_constraints", _fake_generate)
    answer = pamphlet_rag.answer_from_pamphlets("開港はいつ？", "goto")
    labelled = answer.get("answer_with_labels", "")
    assert answer["debug"]["plan"]["style"] == "short_direct"
    assert 1 <= labelled.count("[[") <= 2
    assert len(labelled) <= 220


def test_medium_structured_contains_bullet(adaptive_env, monkeypatch):
    monkeypatch.setattr(pamphlet_rag, "_generate_with_constraints", _fake_generate)
    answer = pamphlet_rag.answer_from_pamphlets("歴史の概要を教えて", "goto")
    labelled = answer.get("answer_with_labels", "")
    assert answer["debug"]["plan"]["style"] == "medium_structured"
    assert "\n- " in labelled


def test_long_explanatory_two_paragraphs(adaptive_env, monkeypatch):
    monkeypatch.setattr(pamphlet_rag, "_generate_with_constraints", _fake_generate)
    answer = pamphlet_rag.answer_from_pamphlets("初めて行くので詳しく知りたい", "goto")
    labelled = answer.get("answer_with_labels", "")
    assert answer["debug"]["plan"]["style"] == "long_explanatory"
    paragraphs = [p for p in labelled.split("\n") if p.strip()]
    assert len(paragraphs) <= 4
    assert len(labelled) <= 800
