import pytest

from coreapp.responders.entries import EntriesResponder
from coreapp.search.entries_index import load_entries_index


ENTRIES_FIXTURE = [
    {
        "id": "spot-1",
        "title": "堂崎教会",
        "desc": "海沿いに建つ世界遺産の教会です。",
        "areas": ["五島市"],
        "tags": ["教会", "世界遺産"],
        "address": "長崎県五島市奥浦町",
        "tel": "0959-00-0000",
        "open_hours": "9:00-17:00",
        "holiday": "年末年始",
        "kana": "どうざききょうかい",
        "romaji": "dozaki kyokai",
    },
    {
        "id": "spot-2",
        "title": "高浜ビーチカフェ",
        "desc": "高浜ビーチを眺めながら海鮮ランチが楽しめるカフェ。",
        "areas": ["五島市"],
        "tags": ["海", "カフェ", "ランチ"],
        "address": "長崎県五島市高浜町",
        "open_hours": "11:00-16:00",
        "kana": "たかはまびーちかふぇ",
        "romaji": "takahama beach cafe",
    },
    {
        "id": "spot-3",
        "title": "奈留港シーサイド食堂",
        "desc": "奈留港の海が目の前。新鮮な魚でランチが人気です。",
        "areas": ["奈留島"],
        "tags": ["海", "ランチ"],
        "address": "長崎県五島市奈留町",
        "open_hours": "10:30-15:00",
        "kana": "なるこうしーさいどしょくどう",
        "romaji": "naruko seaside shokudo",
    },
]


def _make_responder():
    index = load_entries_index(ENTRIES_FIXTURE)
    return EntriesResponder(index=index)


@pytest.mark.parametrize("query", ["堂崎教会", "dozaki kyokai", "どうざききょうかい"])
def test_entries_responder_single_hit_detail(query):
    responder = _make_responder()
    result = responder.respond(query, context={})
    assert result.kind == "detail"
    assert "堂崎教会" in result.message
    assert "住所" in result.message
    assert result.quick_replies is None or result.quick_replies == []


def test_entries_responder_multiple_flow():
    responder = _make_responder()
    context: dict = {}

    initial = responder.respond("海 ランチ", context=context)
    assert initial.kind == "choices"
    assert initial.quick_replies
    assert len(initial.quick_replies) <= 5
    areas = [item for item in initial.quick_replies if item["type"] == "area"]
    assert areas, "area options should be suggested"

    area_option = areas[0]
    followup = responder.respond(area_option["payload"], context=context)
    assert followup.kind == "detail"
    assert "高浜ビーチカフェ" in followup.message
    assert followup.quick_replies is None or followup.quick_replies == []
