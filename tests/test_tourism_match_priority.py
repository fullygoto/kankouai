from services import tourism_search


def _entry(title, *, desc="", tags=None, areas=None, area_checked=True, **extra):
    payload = {
        "title": title,
        "desc": desc,
        "tags": tags or [],
        "areas": areas or ["五島市"],
        "area_checked": area_checked,
    }
    payload.update(extra)
    return payload


def test_title_matches_rank_above_description_and_tags():
    entries = [
        _entry("絶景カフェ風の丘", desc="海を見渡せるカフェ"),
        _entry("風の丘", desc="絶景カフェが楽しめるテラス"),
        _entry("夕陽カフェ", tags=["絶景カフェ"]),
    ]

    results = tourism_search.search(entries, "絶景カフェ")
    assert [item.entry["title"] for item in results] == [
        "絶景カフェ風の丘",
        "風の丘",
        "夕陽カフェ",
    ]
