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


def test_partial_business_name_match():
    entries = [
        _entry("(有)青山電機商会", desc="五島市内の電器専門店"),
        _entry("青山そば店", desc="青山にある人気そば店"),
        _entry("青浜食堂", desc="地元の食堂"),
    ]

    results = tourism_search.search(entries, "青山")
    assert results, "検索結果が得られませんでした"
    assert results[0].entry["title"].startswith("(有)青山電機商会")
    assert results[0].score >= tourism_search.MATCH_THRESHOLD
