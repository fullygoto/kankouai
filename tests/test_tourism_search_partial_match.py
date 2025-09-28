from services import tourism_search


def _entry(title, *, areas, desc="", tags=None, area_checked=True, extra=None):
    data = {
        "title": title,
        "desc": desc,
        "areas": areas,
        "tags": tags or [],
        "area_checked": area_checked,
    }
    if extra:
        data.update(extra)
    return data


def test_partial_match_hits_title(monkeypatch):
    entries = [
        _entry("(有)青山電機商会", areas=["五島市"], desc="五島市内の電器専門店"),
        _entry("高浜海水浴場", areas=["五島市"], desc="白砂のビーチが続く海水浴場", tags=["ビーチ", "海水浴場"]),
        _entry("青山そば店", areas=["新上五島町"], desc="新上五島町の人気そば店"),
    ]

    results = tourism_search.search(entries, "青山", city_key="goto")
    assert [item.entry["title"] for item in results] == ["(有)青山電機商会"]


def test_title_and_description_priority(monkeypatch):
    entries = [
        _entry("高浜海水浴場", areas=["五島市"], desc="高浜の白砂が続く遠浅の海水浴場", tags=["ビーチ"]),
        _entry("高浜ビーチの夕日", areas=["五島市"], desc="夕日が美しいスポット"),
        _entry("高浜そば", areas=["五島市"], desc="そばが名物", tags=["グルメ"]),
        _entry("高浜港マルシェ", areas=["新上五島町"], desc="新上五島町のマーケット"),
    ]

    results = tourism_search.search(entries, "高浜", city_key="goto")
    assert results, "期待した検索結果がありません"
    assert results[0].entry["title"] == "高浜海水浴場"
    assert all("新上五島町" not in (res.entry.get("areas") or []) for res in results)
