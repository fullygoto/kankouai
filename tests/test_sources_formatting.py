import pytest

from services import pamphlet_rag


def test_normalize_sources_from_strings():
    raw = [
        "五島市/長崎五島観光ガイド.txt/9-11",
        "五島市/長崎五島観光ガイド.txt/12-15",
        "新上五島町/しま山ガイド.md/5-5",
    ]

    normalized = pamphlet_rag.normalize_sources(raw)

    assert normalized == [
        ("五島市", "長崎五島観光ガイド.txt"),
        ("新上五島町", "しま山ガイド.md"),
    ]


def test_normalize_sources_from_dicts():
    raw = [
        {"city": "goto", "file": "五島市_観光ガイドブックひとたび五島.txt", "line_from": 1, "line_to": 5},
        {"City": "goto", "filename": "五島市_観光ガイドブックひとたび五島.txt", "line_from": 6, "line_to": 8},
        {"city": "shinkamigoto", "path": "nested/しま山ガイドブック.md", "line_from": 1, "line_to": 2},
    ]

    normalized = pamphlet_rag.normalize_sources(raw)

    assert normalized == [
        ("五島市", "五島市_観光ガイドブックひとたび五島.txt"),
        ("新上五島町", "しま山ガイドブック.md"),
    ]


def test_normalize_sources_mixed_and_invalid():
    raw = [
        "invalid",
        {"city": None, "file": None},
        "五島市/長崎五島観光ガイド.txt/9-11",
        {"city": "goto", "file": "長崎五島観光ガイド.txt"},
    ]

    normalized = pamphlet_rag.normalize_sources(raw)

    assert normalized == [("五島市", "長崎五島観光ガイド.txt")]


def test_format_sources_md_output():
    raw = [
        "五島市/長崎五島観光ガイド.txt/9-11",
        "五島市/長崎五島観光ガイド.txt/12-15",
        "五島市/五島市_観光ガイドブックひとたび五島.txt/20-21",
    ]

    md = pamphlet_rag.format_sources_md(raw)

    assert md == (
        "### 出典\n"
        "- [[1]] 五島市 / 長崎五島観光ガイド.txt\n"
        "- [[2]] 五島市 / 五島市_観光ガイドブックひとたび五島.txt"
    )


def test_format_sources_md_empty():
    assert pamphlet_rag.format_sources_md([]) == ""
    assert pamphlet_rag.format_sources_md(None) == ""
