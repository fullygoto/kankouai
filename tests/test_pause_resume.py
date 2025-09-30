import importlib
import logging


def test_suspend_resume_persistence(tmp_path, monkeypatch):
    base_dir = tmp_path / "data"
    monkeypatch.setenv("DATA_BASE_DIR", str(base_dir))
    monkeypatch.setenv("USER_STATE_DB_PATH", str(tmp_path / "user_state.sqlite"))
    monkeypatch.setenv("CONTROL_CMD_ENABLED", "true")
    monkeypatch.setenv("PAUSE_DEFAULT_TTL_SEC", "60")

    state_module = importlib.import_module("coreapp.state")
    state = importlib.reload(state_module)
    state.reset_for_tests()

    handlers_module = importlib.import_module("services.line_handlers")
    handlers = importlib.reload(handlers_module)

    user_id = "U123"
    state.suspend_user(user_id)
    assert state.is_suspended(user_id) is True
    assert handlers.control_is_paused(user_id, 0) is True

    # Persistence across reload
    state = importlib.reload(state)
    assert state.is_suspended(user_id) is True

    state.resume_user(user_id)
    assert state.is_suspended(user_id) is False
    assert handlers.control_is_paused(user_id, 0) is False

    # Suspend again via the control command to ensure both stores are updated
    pause_replies: list[str] = []

    handlers.process_control_command(
        "停止",
        user_id=user_id,
        event=None,
        reply_func=lambda _event, message: pause_replies.append(message),
        logger=logging.getLogger("test"),
        now=240,
    )

    assert state.is_suspended(user_id) is True
    assert handlers.control_is_paused(user_id, 0) is True

    replies: list[str] = []

    def fake_reply(_event, message):
        replies.append(message)
        return message

    result = handlers.process_control_command(
        "解除",
        user_id=user_id,
        event=None,
        reply_func=fake_reply,
        logger=logging.getLogger("test"),
        now=260,
    )

    assert result["action"] == "resume"
    assert state.is_suspended(user_id) is False
    assert handlers.control_is_paused(user_id, 0) is False
    assert any("再開しました" in message for message in replies)
