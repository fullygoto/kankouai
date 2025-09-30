import pytest

from coreapp.intent import get_intent_detector
from coreapp.responders.priority import (
    FALLBACK_MESSAGE,
    PriorityResponder,
    transport_reply_text,
    viewpoint_map_reply_text,
    weather_reply_text,
)


def test_intent_detector_priority_labels():
    detector = get_intent_detector()

    assert detector.classify_priority("今日の天気") == "weather"
    assert detector.classify_priority("運行状況") == "transport"
    assert detector.classify_priority("展望所マップ") == "viewpoint_map"


def test_priority_responder_weather():
    responder = PriorityResponder()
    answer = responder.answer("今日の天気", label="weather")

    assert answer is not None
    assert answer.kind == "weather"
    assert "五島列島の主な天気情報リンク" in answer.message
    assert answer.message == weather_reply_text()


def test_priority_responder_transport_variants():
    responder = PriorityResponder()
    ferry_answer = responder.answer("フェリーの運行状況を知りたい", label="transport")
    assert ferry_answer is not None
    assert "長崎ー五島航路" in ferry_answer.message
    assert "五島つばき空港" not in ferry_answer.message

    flight_answer = responder.answer("空港の運行状況", label="transport")
    assert flight_answer is not None
    assert "五島つばき空港" in flight_answer.message

    both_reply = transport_reply_text("運行状況を教えて")
    assert "長崎ー五島航路" in both_reply
    assert "五島つばき空港" in both_reply


def test_priority_responder_viewpoint_map(monkeypatch: pytest.MonkeyPatch):
    responder = PriorityResponder()

    monkeypatch.setenv("VIEWPOINTS_URL", "https://example.com/viewpoints")
    monkeypatch.delenv("VIEWPOINTS_MID", raising=False)
    monkeypatch.delenv("VIEWPOINTS_LL", raising=False)
    monkeypatch.delenv("VIEWPOINTS_ZOOM", raising=False)

    answer = responder.answer("展望所マップ", label="viewpoint_map")
    assert answer is not None
    assert answer.kind == "map"
    assert "https://example.com/viewpoints" in answer.message


def test_viewpoint_map_fallback(monkeypatch: pytest.MonkeyPatch):
    responder = PriorityResponder()

    for key in ("VIEWPOINTS_URL", "VIEWPOINTS_MID", "VIEWPOINTS_LL", "VIEWPOINTS_ZOOM"):
        monkeypatch.delenv(key, raising=False)

    answer = responder.answer("展望所マップ", label="viewpoint_map")
    assert answer is not None
    assert answer.message == FALLBACK_MESSAGE
    assert viewpoint_map_reply_text() == FALLBACK_MESSAGE
