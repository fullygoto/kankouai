from services import tourism_search


def _entry(title, **extra):
    payload = {
        "title": title,
        "desc": extra.get("desc", ""),
        "tags": extra.get("tags", []),
        "areas": extra.get("areas", ["五島市"]),
        "updated_at": extra.get("updated_at", "2024-01-01T00:00:00"),
    }
    payload.update(extra)
    return payload


def test_title_exact_match_wins():
    entries = [
        _entry("福江城跡", popularity=1),
        _entry("福江城", popularity=100),
    ]
    results = tourism_search.search(entries, "福江城")
    assert results[0].entry["title"] == "福江城"


def test_title_startswith_ranks_above_contains():
    entries = [
        _entry("鬼岳温泉", popularity=10),
        _entry("トレッキング鬼岳", popularity=10),
    ]
    results = tourism_search.search(entries, "鬼岳")
    assert results[0].entry["title"].startswith("鬼岳温泉")


def test_title_contains_finds_substring_even_with_nfkc():
    entries = [
        _entry("高浜ビーチ", desc="白い砂浜"),
        _entry("たかはま海水浴場", desc="青い海"),
    ]
    results = tourism_search.search(entries, "高浜ﾋﾞｰﾁ")
    titles = [item.entry["title"] for item in results]
    assert "高浜ビーチ" in titles


def test_multiple_hits_return_suggestions():
    entries = [
        _entry("鬼岳ハイキング", tags=["観光", "自然"], areas=["五島市"]),
        _entry("鬼岳キャンプ", tags=["観光", "体験"], areas=["五島市"]),
        _entry("鬼岳温泉", tags=["温泉"], areas=["新上五島町"]),
    ]
    results = tourism_search.search(entries, "鬼岳", limit=5)
    suggestions = tourism_search.build_narrowing_suggestions(results)
    assert "観光" in suggestions["tags"]
    assert "五島市" in suggestions["areas"]
