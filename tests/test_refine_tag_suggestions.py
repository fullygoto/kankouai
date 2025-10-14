from tests.utils import load_test_app


class _DummyText:
    def __init__(self, text: str, *_, **__):
        self.text = text


def test_multiple_hits_append_tag_refinement(monkeypatch, tmp_path):
    entries = [
        {
            "title": "武家屋敷通り",
            "desc": "武家屋敷が立ち並ぶ通り。",
            "areas": ["五島市"],
            "tags": ["歴史", "散策"],
            "category": "観光",
        },
        {
            "title": "武家屋敷資料館",
            "desc": "歴史資料が展示されている施設。",
            "areas": ["五島市"],
            "tags": ["歴史", "資料館"],
            "category": "観光",
        },
        {
            "title": "武家屋敷茶房",
            "desc": "和の雰囲気のカフェ。",
            "areas": ["五島市"],
            "tags": ["カフェ", "和スイーツ"],
            "category": "飲食",
        },
    ]

    with load_test_app(monkeypatch, tmp_path, extra_env={"SECRET_KEY": "test"}) as module:
        monkeypatch.setattr(module, "load_entries", lambda: entries)
        monkeypatch.setattr(module, "TextSendMessage", _DummyText)
        monkeypatch.setattr(module, "_split_for_line", lambda text, limit: [text])

        messages, hit = module._answer_from_entries_rich("武家屋敷")

        assert hit is True
        combined = "\n".join(message.text for message in messages)
        assert "🔍 絞り込み候補" in combined
        assert "タグ例:" in combined
