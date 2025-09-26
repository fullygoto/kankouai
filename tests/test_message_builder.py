from services.message_builder import build_pamphlet_message, parse_pamphlet_answer


def test_build_message_removes_intent_and_notes():
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

    parsed = parse_pamphlet_answer(raw)
    built = build_pamphlet_message(
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
