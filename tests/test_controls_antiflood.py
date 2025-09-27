import importlib
import logging
from types import SimpleNamespace

import pytest


class DummyEvent:
    def __init__(self, user_id: str = "U123") -> None:
        self.reply_token = "dummy"
        self.source = SimpleNamespace(user_id=user_id, group_id=None, room_id=None)


@pytest.fixture
def control_env(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_STATE_DB_PATH", str(tmp_path / "user_state.sqlite"))
    monkeypatch.setenv("CONTROL_CMD_ENABLED", "true")
    monkeypatch.setenv("PAUSE_DEFAULT_TTL_SEC", "120")
    monkeypatch.setenv("ANTIFLOOD_TTL_SEC", "180")
    monkeypatch.setenv("REPLAY_GUARD_SEC", "150")

    state = importlib.import_module("services.state")
    dupe = importlib.import_module("services.dupe_guard")
    handlers = importlib.import_module("services.line_handlers")

    state = importlib.reload(state)
    dupe = importlib.reload(dupe)
    handlers = importlib.reload(handlers)

    state.reset_for_tests()
    dupe.reset()

    yield state, dupe, handlers

    dupe.reset()
    state.reset_for_tests()


def test_resume_allows_followup(control_env):
    state, dupe, handlers = control_env
    event = DummyEvent()
    logger = logging.getLogger("test.controls")

    allowed_initial = dupe.should_process_incoming(
        user_id=event.source.user_id,
        message_id="m1",
        text="こんにちは",
        event_ts=100.0,
        now=100.0,
    )
    assert allowed_initial is True

    handlers.process_control_command(
        "停止 2分",
        user_id=event.source.user_id,
        event=event,
        reply_func=lambda *_: None,
        logger=logger,
        now=100,
    )
    assert state.is_paused(event.source.user_id, 101)

    handlers.process_control_command(
        "解除",
        user_id=event.source.user_id,
        event=event,
        reply_func=lambda *_: None,
        logger=logger,
        now=102,
    )

    allowed_after = dupe.should_process_incoming(
        user_id=event.source.user_id,
        message_id="m2",
        text="観光情報",
        event_ts=103.0,
        now=103.0,
    )
    assert allowed_after is True


def test_duplicate_message_suppressed(control_env):
    _state, dupe, _handlers = control_env

    assert dupe.should_process_incoming(
        user_id="U", message_id="mid", text="質問", event_ts=200.0, now=200.0
    ) is True
    assert dupe.should_process_incoming(
        user_id="U", message_id="mid", text="質問", event_ts=201.0, now=201.0
    ) is False

    # 同じ本文が短時間で再送された場合も抑止される
    assert dupe.should_process_incoming(
        user_id="U", message_id="other", text="質問", event_ts=202.0, now=202.0
    ) is False


def test_old_events_rejected(control_env):
    _state, dupe, _handlers = control_env

    now = 500.0
    assert dupe.should_process_incoming(
        user_id="U", message_id="fresh", text="最新", event_ts=now, now=now
    ) is True
    assert dupe.should_process_incoming(
        user_id="U", message_id="old", text="遅延", event_ts=now - 200, now=now
    ) is False


def test_resume_command_not_blocked(control_env):
    _state, _dupe, handlers = control_env
    event = DummyEvent("U-resume")
    logger = logging.getLogger("test.controls")

    first = handlers.process_control_command(
        "解除",
        user_id=event.source.user_id,
        event=event,
        reply_func=lambda *_: None,
        logger=logger,
        now=10,
    )
    second = handlers.process_control_command(
        "解除",
        user_id=event.source.user_id,
        event=event,
        reply_func=lambda *_: None,
        logger=logger,
        now=11,
    )

    assert first["action"] == "resume"
    assert second["action"] == "resume"
