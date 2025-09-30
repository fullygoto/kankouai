"""Tests for fallback quick replies menu."""


def test_fallback_quick_replies():
    from coreapp.responders.fallback import FallbackResponder

    result = FallbackResponder().respond("not found")
    assert "見つかりませんでした" in result.message
    labels = [item["label"] for item in result.quick_replies]
    assert labels == [
        "今日の天気",
        "運行状況",
        "展望所マップ",
        "店舗名で検索",
        "観光地名で検索",
    ]
    payloads = [item["payload"] for item in result.quick_replies]
    assert payloads == [
        "weather",
        "transport_status",
        "viewpoint_map",
        "entry_search",
        "spot_search",
    ]
