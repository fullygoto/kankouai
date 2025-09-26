import base64
import hashlib
import hmac
import json
import time

import pytest

import app
from antiflood import AntiFlood


class DummyLineApi:
    def __init__(self):
        self.reply_calls = []
        self.push_calls = []

    def reply_message(self, token, messages, *args, **kwargs):
        self.reply_calls.append((token, messages))

    def push_message(self, to, messages, *args, **kwargs):
        self.push_calls.append((to, messages))

    def get_profile(self, user_id):
        class _Profile:
            display_name = "tester"

        return _Profile()


def _signature(body: str) -> str:
    secret = (app.LINE_CHANNEL_SECRET or "dummy").encode("utf-8")
    return base64.b64encode(
        hmac.new(secret, body.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")


def _post_message(client, body_dict):
    body = json.dumps(body_dict, separators=(",", ":"))
    sig = _signature(body)
    return client.post(
        "/callback",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Line-Signature": sig,
        },
    )


@pytest.fixture
def patched_env(monkeypatch, tmp_path):
    current = [time.time()]
    antiflood = AntiFlood.from_env(base_dir=tmp_path, time_func=lambda: current[0])
    antiflood.clear()

    dummy_api = DummyLineApi()
    monkeypatch.setattr(app, "line_bot_api", dummy_api, raising=False)
    monkeypatch.setattr(app, "ANTIFLOOD", antiflood, raising=False)
    monkeypatch.setattr(app, "ANTIFLOOD_TTL_SEC", 120, raising=False)
    monkeypatch.setattr(app, "RECENT_TEXT_TTL_SEC", 60, raising=False)
    monkeypatch.setattr(app, "REPLAY_GUARD_SEC", 150, raising=False)
    monkeypatch.setattr(app, "ENABLE_PUSH", True, raising=False)
    monkeypatch.setattr(app, "_line_enabled", lambda: True)
    monkeypatch.setattr(app, "_line_mute_gate", lambda *args, **kwargs: False)
    monkeypatch.setattr(app, "get_weather_reply", lambda text: ("", False))
    monkeypatch.setattr(app, "get_mobility_reply", lambda text: ("", False))
    monkeypatch.setattr(app, "get_transport_reply", lambda text: ("", False))
    monkeypatch.setattr(app, "smalltalk_or_help_reply", lambda text: "テスト応答")
    monkeypatch.setattr(app, "save_qa_log", lambda *args, **kwargs: None)

    return antiflood, current, dummy_api


def test_webhook_deduplicates_events(patched_env):
    antiflood, current, dummy_api = patched_env
    client = app.app.test_client()

    def make_body(message_id: str, text: str, ts_ms: int):
        return {
            "destination": "dummy",
            "events": [
                {
                    "type": "message",
                    "timestamp": ts_ms,
                    "replyToken": "token",
                    "source": {"type": "user", "userId": "U1"},
                    "message": {"id": message_id, "type": "text", "text": text},
                }
            ],
        }

    now_sec = current[0]
    now_ts = int(now_sec * 1000)

    res = _post_message(client, make_body("msg-1", "テストです", now_ts))
    assert res.status_code == 200
    assert len(dummy_api.reply_calls) == 1

    res = _post_message(client, make_body("msg-1", "テストです", now_ts))
    assert res.status_code == 200
    assert len(dummy_api.reply_calls) == 1

    res = _post_message(client, make_body("msg-2", "テストです", now_ts + 1000))
    assert res.status_code == 200
    assert len(dummy_api.reply_calls) == 1

    current[0] += 200
    future_ts = int(current[0] * 1000)
    res = _post_message(client, make_body("msg-3", "テストです", future_ts))
    assert res.status_code == 200
    assert len(dummy_api.reply_calls) == 2

    old_ts = int((time.time() - app.REPLAY_GUARD_SEC - 10) * 1000)
    res = _post_message(client, make_body("msg-4", "別のメッセージ", old_ts))
    assert res.status_code == 200
    assert len(dummy_api.reply_calls) == 2


def test_safe_push_line_deduplicates(monkeypatch, tmp_path):
    current = [time.time()]
    antiflood = AntiFlood.from_env(base_dir=tmp_path, time_func=lambda: current[0])
    antiflood.clear()

    dummy_api = DummyLineApi()
    monkeypatch.setattr(app, "line_bot_api", dummy_api, raising=False)
    monkeypatch.setattr(app, "ANTIFLOOD", antiflood, raising=False)
    monkeypatch.setattr(app, "ANTIFLOOD_TTL_SEC", 180, raising=False)
    monkeypatch.setattr(app, "ENABLE_PUSH", True, raising=False)
    monkeypatch.setattr(app, "_line_enabled", lambda: True)

    assert app.safe_push_line("U1", ["push test"], label="unit") is True
    assert len(dummy_api.push_calls) == 1
    assert app.safe_push_line("U1", ["push test"], label="unit") is False
    assert len(dummy_api.push_calls) == 1

    current[0] += 200
    assert app.safe_push_line("U1", ["push test"], label="unit") is True
    assert len(dummy_api.push_calls) == 2
