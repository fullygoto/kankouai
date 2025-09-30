import pytest

from coreapp.responders.pamphlet import PamphletResponder
from coreapp.search.pamphlet_index import PamphletIndex, load_pamphlet_index


@pytest.fixture
def pamphlet_dataset():
    return [
        {
            "city": "五島市",
            "source": "events.txt",
            "text": "五島市では2024年夏に花火大会が開催され、家族連れに人気です。夕方から屋台も並びます。",
        },
        {
            "city": "五島市",
            "source": "culture.txt",
            "text": "世界遺産に登録された五島市の教会群は、巡礼と観光の両面で高い評価を受けています。",
        },
        {
            "city": "新上五島町",
            "source": "guide.txt",
            "text": "新上五島町では教会と海上タクシーを組み合わせた巡礼ツアーが人気で、2023年から夜間ライトアップも実施されています。",
        },
    ]


def test_responder_requests_city_when_not_specified(pamphlet_dataset):
    responder = PamphletResponder(pamphlets=pamphlet_dataset)
    context: dict = {}

    result = responder.respond("パンフレットを見せて", context=context)

    assert result.kind == "ask_city"
    assert result.quick_replies is not None
    assert len(result.quick_replies) == 4
    assert {item["label"] for item in result.quick_replies} == {"五島市", "新上五島町", "小値賀町", "宇久町"}
    assert result.web_buttons is not None
    assert len(result.web_buttons) == 4


def test_summary_generation_after_city_selection(pamphlet_dataset):
    responder = PamphletResponder(pamphlets=pamphlet_dataset)
    context: dict = {}

    responder.respond("イベント情報", context=context)
    result = responder.respond("五島市", context=context)

    assert result.kind == "summary"
    assert 200 <= len(result.message) <= 350
    assert result.message.count("・") <= 4
    assert "2024" in result.message
    assert "events.txt" in result.message


def test_city_detected_from_query_text(pamphlet_dataset):
    responder = PamphletResponder(pamphlets=pamphlet_dataset)
    context: dict = {}

    result = responder.respond("新上五島町の教会の様子を教えて", context=context)

    assert result.kind == "summary"
    assert "新上五島町" in result.message
    assert "2023" in result.message


def test_duplicate_question_returns_noop(pamphlet_dataset):
    responder = PamphletResponder(pamphlets=pamphlet_dataset)
    context: dict = {}

    first = responder.respond("五島市のイベント", context=context)
    assert first.kind == "summary"

    second = responder.respond("五島市のイベント", context=context)
    assert second.kind == "noop"


def test_index_deduplicates_fragments():
    dataset = [
        {
            "city": "五島市",
            "source": "dup.txt",
            "text": "同じ文を繰り返す段落です。同じ文を繰り返す段落です。",
        },
        {
            "city": "五島市",
            "source": "dup2.txt",
            "text": "同じ文を繰り返す段落です。同じ文を繰り返す段落です。",
        },
    ]
    index = load_pamphlet_index(dataset)

    results = index.search("五島市", "同じ文", top_k=5)

    assert len(results) == 1
    assert results[0].source == "dup.txt"


def test_top_k_clamped_between_three_and_five(pamphlet_dataset):
    index = PamphletIndex(pamphlets=pamphlet_dataset)
    responder = PamphletResponder(index=index, top_k=8)
    context: dict = {}

    responder.respond("パンフレット", context=context)
    result = responder.respond("五島市", context=context)

    assert result.kind == "summary"
    assert len(result.fragments) <= 5
