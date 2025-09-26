import importlib
import logging
from types import SimpleNamespace

import pytest


class DummyEvent:
    def __init__(self, user_id: str = "U123") -> None:
        self.reply_token = "dummy"
        self.source = SimpleNamespace(user_id=user_id, group_id=None, room_id=None)


@pytest.fixture
def control_modules(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_STATE_DB_PATH", str(tmp_path / "user_state.sqlite"))
    monkeypatch.setenv("CONTROL_CMD_ENABLED", "true")
    monkeypatch.setenv("PAUSE_DEFAULT_TTL_SEC", "86400")

    state = importlib.import_module("services.state")
    state = importlib.reload(state)

    handlers = importlib.import_module("services.line_handlers")
    handlers = importlib.reload(handlers)

    state.reset_for_tests()

    yield state, handlers

    state.reset_for_tests()


def test_parse_control_command_variations(monkeypatch):
    monkeypatch.setenv("CONTROL_CMD_ENABLED", "true")
    monkeypatch.setenv("PAUSE_DEFAULT_TTL_SEC", "3600")
    handlers = importlib.import_module("services.line_handlers")
    handlers = importlib.reload(handlers)

    assert handlers.parse_control_command("停止") == ("pause", 3600)
    assert handlers.parse_control_command("停止 2分") == ("pause", 120)
    assert handlers.parse_control_command("停止2h")[1] == 7200
    assert handlers.parse_control_command("停止 1日")[1] == 86400
    assert handlers.parse_control_command("解除") == ("resume", None)
    assert handlers.parse_control_command("  再開  ") == ("resume", None)
    assert handlers.parse_control_command("こんにちは") is None


def test_pause_resume_flow(control_modules):
    state, handlers = control_modules
    replies = []

    def fake_reply(event, message):
        replies.append(message)
        return message

    event = DummyEvent()
    logger = logging.getLogger("test")

    pause_result = handlers.process_control_command(
        "停止 2分",
        user_id=event.source.user_id,
        event=event,
        reply_func=fake_reply,
        logger=logger,
        now=100,
    )

    assert pause_result["action"] == "pause"
    assert pause_result["ttl_sec"] == 120
    assert "停止します" in replies[-1]
    assert handlers.control_is_paused(event.source.user_id, 150)

    replies.clear()
    resume_result = handlers.process_control_command(
        "解除",
        user_id=event.source.user_id,
        event=event,
        reply_func=fake_reply,
        logger=logger,
        now=200,
    )

    assert resume_result["action"] == "resume"
    assert resume_result["changed"] is True
    assert "再開しました" in replies[-1]
    assert not handlers.control_is_paused(event.source.user_id, 201)

    replies.clear()
    resume_again = handlers.process_control_command(
        "解除",
        user_id=event.source.user_id,
        event=event,
        reply_func=fake_reply,
        logger=logger,
        now=210,
    )

    assert resume_again["action"] == "resume"
    assert resume_again["changed"] is False
    assert "稼働中" in replies[-1]


def test_pause_expiry(control_modules):
    state, handlers = control_modules
    state.pause("U999", ttl_sec=60, now=1)
    assert handlers.control_is_paused("U999", 30)
    assert not handlers.control_is_paused("U999", 200)
    assert state.get_state("U999")["paused_until"] is None


def test_admin_pause_blocks_command(control_modules):
    _state, handlers = control_modules
    replies = []

    def fake_reply(event, message):
        replies.append(message)

    event = DummyEvent()
    logger = logging.getLogger("test")

    result = handlers.process_control_command(
        "解除",
        user_id=event.source.user_id,
        event=event,
        reply_func=fake_reply,
        logger=logger,
        global_pause=(True, "admin"),
    )

    assert result["action"] == "blocked"
    assert "管理者" in replies[-1]
