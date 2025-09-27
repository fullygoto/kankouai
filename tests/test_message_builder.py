import importlib

import services.message_builder as message_builder


def test_build_message_removes_intent_and_notes(monkeypatch):
    monkeypatch.setenv("SUMMARY_STYLE", "terse_short")
    raw = """質問の意図
五島市の遣唐使について整理する。
要約
遣唐使は630年から派遣が始まり、五島は唐への最終寄港地でした。
詳細
- 804年には空海が五島から出帆した。
- 唐の制度や仏教を学ぶために派遣された。
- 航海は危険で多くの準備が必要だった。
補足
- 資料に明記なし
出典
- 五島市/長崎五島観光ガイド.txt/L10-12
- 五島市/五島市_観光ガイドブックひとたび五島.md/L20-22
"""

    importlib.reload(message_builder)
    parsed = message_builder.parse_pamphlet_answer(raw)
    built = message_builder.build_pamphlet_message(
        parsed,
        [
            "五島市/長崎五島観光ガイド.txt/L10-12",
            "五島市/長崎五島観光ガイド.txt/L13-14",
            {"city": "goto", "file": "五島市_観光ガイドブックひとたび五島.md", "line_from": 20, "line_to": 22},
        ],
    )

    assert built.text.startswith("### 要約")
    assert "質問の意図" not in built.text
    assert "補足" not in built.text
    assert built.text.count("### 出典") == 1
    assert len(built.details) <= 2
    for line in built.sources_md.splitlines()[1:]:
        assert line.startswith("- 五島市/")
        assert ".txt" not in line and ".md" not in line
        assert "L" not in line


def test_build_message_polite_long_style(monkeypatch):
    monkeypatch.setenv("SUMMARY_STYLE", "polite_long")
    raw = "海辺の散策は潮の香りと教会の景観を楽しめます。[[1]]\n\n家族で参加できる体験も紹介されています。[[2]]"

    importlib.reload(message_builder)
    parsed = message_builder.parse_pamphlet_answer(raw)
    built = message_builder.build_pamphlet_message(
        parsed,
        ["五島市/海と教会ガイド.txt", "五島市/体験プラン集.md"],
    )

    assert not built.text.startswith("###")
    assert built.text.count("[[1]]") == 1
    assert "### 出典" in built.text
    assert built.details == []


def test_parse_plain_answer_returns_summary(monkeypatch):
    monkeypatch.setenv("SUMMARY_STYLE", "polite_long")
    raw = "港町の散策路は季節の花が彩ります。[[3]]"

    importlib.reload(message_builder)
    parsed = message_builder.parse_pamphlet_answer(raw)
    assert parsed.summary == raw
    assert parsed.details == []
