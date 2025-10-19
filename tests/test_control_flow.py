import importlib
import logging
import threading
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


def test_pause_resume_allows_immediate_followup(control_modules):
    state, handlers = control_modules
    event = DummyEvent()
    logger = logging.getLogger("test.control_flow")
    replies: list[str] = []

    def fake_reply(_event, message):
        replies.append(message)
        return message

    pause_result = handlers.process_control_command(
        "停止 5分",
        user_id=event.source.user_id,
        event=event,
        reply_func=fake_reply,
        logger=logger,
        now=100,
    )

    assert pause_result["action"] == "pause"
    assert handlers.control_is_paused(event.source.user_id, 101)

    resume_result = handlers.process_control_command(
        "解除",
        user_id=event.source.user_id,
        event=event,
        reply_func=fake_reply,
        logger=logger,
        now=110,
    )

    assert resume_result["action"] == "resume"
    assert resume_result["changed"] is True
    assert resume_result["state"]["paused_until"] is None
    assert not handlers.control_is_paused(event.source.user_id, 111)
    assert state.get_state(event.source.user_id)["paused_until"] is None


def test_resume_when_not_paused_keeps_responses_active(control_modules):
    _state, handlers = control_modules
    event = DummyEvent()
    logger = logging.getLogger("test.control_flow")

    replies: list[str] = []

    def fake_reply(_event, message):
        replies.append(message)
        return message

    resume_result = handlers.process_control_command(
        "解除",
        user_id=event.source.user_id,
        event=event,
        reply_func=fake_reply,
        logger=logger,
        now=200,
    )

    assert resume_result["action"] == "resume"
    assert resume_result["changed"] is False
    assert resume_result["state"]["paused_until"] is None
    assert not handlers.control_is_paused(event.source.user_id, 201)


def test_resume_visibility_across_threads(control_modules):
    state, handlers = control_modules
    user_id = "U-thread"
    state.pause(user_id, ttl_sec=300, now=100)
    assert handlers.control_is_paused(user_id, 150)

    event = DummyEvent(user_id)
    logger = logging.getLogger("test.control_flow")
    replies: list[str] = []

    def fake_reply(_event, message):
        replies.append(message)
        return message

    resume_done = threading.Event()
    paused_result: dict[str, bool] = {}

    def do_resume() -> None:
        handlers.process_control_command(
            "解除",
            user_id=user_id,
            event=event,
            reply_func=fake_reply,
            logger=logger,
            now=200,
        )
        resume_done.set()

    def check_state() -> None:
        resume_done.wait(timeout=2)
        paused_result["value"] = handlers.control_is_paused(user_id, 200)

    thread_resume = threading.Thread(target=do_resume)
    thread_check = threading.Thread(target=check_state)
    thread_resume.start()
    thread_check.start()
    thread_resume.join()
    thread_check.join()

    assert paused_result.get("value") is False
    assert state.get_state(user_id)["paused_until"] is None
