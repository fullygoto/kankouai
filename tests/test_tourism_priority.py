from services import tourism_search


def _entry(title, *, desc="", tags=None, areas=None, area_checked=True):
    return {
        "title": title,
        "desc": desc,
        "tags": tags or [],
        "areas": areas or ["五島市"],
        "area_checked": area_checked,
    }


def test_title_matches_rank_above_description_and_tags():
    entries = [
        _entry("絶景カフェ風の丘", desc="", tags=["カフェ"]),
        _entry("風の丘", desc="絶景カフェが楽しめるテラス", tags=["カフェ"]),
        _entry("夕陽カフェ", desc="夕日が美しい", tags=["絶景カフェ"]),
    ]

    results = tourism_search.search(entries, "絶景カフェ", city_key="goto")
    assert [item.entry["title"] for item in results] == [
        "絶景カフェ風の丘",
        "風の丘",
        "夕陽カフェ",
    ]
