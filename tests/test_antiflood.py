import time

from antiflood import AntiFlood, make_event_key, make_text_key, normalize_text


def test_antiflood_acquire_and_contains(tmp_path):
    current = [time.time()]

    antiflood = AntiFlood.from_env(base_dir=tmp_path, time_func=lambda: current[0])
    antiflood.clear()

    key = make_event_key("user", "msg", "hello")
    assert antiflood.acquire(key, 5) is True
    assert antiflood.contains(key) is True
    assert antiflood.acquire(key, 5) is False

    current[0] += 6
    assert antiflood.contains(key) is False
    assert antiflood.acquire(key, 5) is True

    text_key = make_text_key("user", normalize_text("テスト"))
    assert antiflood.acquire(text_key, 1) is True
    current[0] += 2
    assert antiflood.contains(text_key) is False
