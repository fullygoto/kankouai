from coreapp.responders.entries import EntriesResponder
from coreapp.search.entries_index import load_entries_index


MULTI_ENTRIES = [
    {"id": "spot-1", "title": "教会資料館", "desc": "教会群の資料を展示"},
    {"id": "spot-2", "title": "堂崎教会", "kana": "どうざききょうかい"},
    {"id": "spot-3", "title": "青砂ヶ浦天主堂", "desc": "長崎の教会群の一つ"},
    {"id": "spot-4", "title": "旧五輪教会堂", "desc": "世界遺産の教会です"},
    {"id": "spot-5", "title": "福江観光案内所", "desc": "教会巡りの相談窓口"},
    {"id": "spot-6", "title": "巡礼ツアーセンター", "tags": ["教会", "ツアー"]},
    {"id": "spot-7", "title": "高浜ビーチカフェ", "desc": "海沿いのカフェ"},
]

SINGLE_ENTRY = [
    {
        "id": "single-1",
        "title": "江上天主堂",
        "desc": "世界遺産の教会です。",
        "areas": ["上五島"],
    }
]


def test_two_char_query_hits_and_limit():
    index = load_entries_index(MULTI_ENTRIES)
    results = index.search("教会")

    assert results, "2文字のクエリでヒットが返ること"
    assert len(results) <= 5, "2文字クエリでは最大5件に制限される"
    titles = [match.entry.get("title") for match in results]
    assert titles[0] == "教会資料館"
    assert titles[1] == "堂崎教会"


def test_one_char_query_is_ignored():
    index = load_entries_index(MULTI_ENTRIES)
    assert index.search("教") == []


def test_two_char_single_hit_returns_detail():
    responder = EntriesResponder(index=load_entries_index(SINGLE_ENTRY))
    context: dict = {}
    result = responder.respond("教会", context=context)

    assert result.kind == "detail"
    assert "教会" in result.message
