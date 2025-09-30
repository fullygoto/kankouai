import datetime as dt
import hashlib
import json

from coreapp import llm
from coreapp.logging_utils import log_interaction
from coreapp.responders.fallback import FallbackResponder


def test_fallback_responder_returns_menu():
    responder = FallbackResponder()
    result = responder.respond("未知の質問", context={})

    assert "該当なし" in result.message
    assert len(result.quick_replies) == 5
    labels = [item["label"] for item in result.quick_replies]
    payloads = [item["payload"] for item in result.quick_replies]
    assert labels == ["天気", "運行状況", "展望所マップ", "店舗名で探す", "観光地名で探す"]
    assert payloads == labels[:3] + ["店舗名検索", "観光地名検索"]


def test_pick_model_threshold_switch(monkeypatch):
    monkeypatch.setattr(llm, "MODEL_DEFAULT", "gpt-4o-mini")
    monkeypatch.setattr(llm, "MODEL_HARD", "gpt-5-mini")
    monkeypatch.setattr(llm, "THRESHOLD_SCORE_HARD", 0.7)
    monkeypatch.setattr(llm, "THRESHOLD_PIECES_HARD", 5)
    monkeypatch.setattr(llm, "THRESHOLD_REPROMPTS_HARD", 2)

    assert llm.pick_model(0.9, 2, 0) == "gpt-4o-mini"
    assert llm.pick_model(0.5, 2, 0) == "gpt-5-mini"
    assert llm.pick_model(None, 6, 0) == "gpt-5-mini"
    assert llm.pick_model(0.9, 2, 3) == "gpt-5-mini"


def test_log_interaction_outputs_expected_json(tmp_path):
    log_path = tmp_path / "interactions.jsonl"
    entry = log_interaction(
        log_path,
        user_id="user-42",
        channel="line",
        intent="fallback",
        hit_source="none",
        query="おすすめは？",
        top_score=0.42,
        model_used="gpt-5-mini",
        tokens=128,
        latency_ms=345.6,
        errors=["timeout", ""],
        timestamp=dt.datetime(2024, 1, 2, 3, 4, 5),
    )

    expected_hash = hashlib.sha256("user-42".encode("utf-8")).hexdigest()
    assert entry.user_id == expected_hash
    assert entry.errors == ["timeout"]

    content = log_path.read_text(encoding="utf-8").strip()
    data = json.loads(content)
    assert data["user_id"] == expected_hash
    assert data["model_used"] == "gpt-5-mini"
    assert data["tokens"] == 128
    assert data["latency_ms"] == 345.6

