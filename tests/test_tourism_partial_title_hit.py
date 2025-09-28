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


def test_partial_title_match_returns_top_hit():
    entries = [
        _entry("高浜海水浴場", desc="白砂が続く人気のビーチ", tags=["ビーチ"]),
        _entry("高浜港マルシェ", desc="地元の産品が集まるマルシェ"),
        _entry("富江ビーチ", desc="家族連れに人気の海水浴場"),
    ]

    results = tourism_search.search(entries, "高浜")
    assert results, "検索結果が得られませんでした"
    assert results[0].entry["title"] == "高浜海水浴場"
    assert results[0].score >= tourism_search.MATCH_THRESHOLD
