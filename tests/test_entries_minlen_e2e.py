import json
from pathlib import Path

from coreapp import config as cfg

from tests.utils import load_test_app


ENTRIES_FIXTURE = [
    {
        "id": "spot-1",
        "title": "青砂ヶ浦天主堂",
        "desc": "新上五島町を代表する教会です。",
        "areas": ["新上五島町"],
        "tags": ["教会"],
    },
    {
        "id": "spot-2",
        "title": "教会資料センター",
        "desc": "教会に関する資料を展示しています。",
        "areas": ["五島市"],
        "tags": ["資料"],
    },
]


def _write_entries(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    payload = base_dir / "entries.json"
    payload.write_text(json.dumps(ENTRIES_FIXTURE, ensure_ascii=False), encoding="utf-8")


def test_entries_two_char_query_hits(monkeypatch, tmp_path):
    _write_entries(tmp_path)
    monkeypatch.setattr(cfg, "MIN_QUERY_CHARS", 2, raising=False)
    monkeypatch.setattr(cfg, "ENABLE_ENTRIES_2CHAR", True, raising=False)

    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={"SECRET_KEY": "test", "MIN_QUERY_CHARS": 2, "ENABLE_ENTRIES_2CHAR": 1},
    ) as module:
        module.handle_mobility = lambda *args, **kwargs: ("", False)
        client = module.app.test_client()
        assert module.load_entries(), "entries should be loaded for search"
        assert module.find_entry_info("教会"), "fixture must provide hits for 教会"

        response = client.post("/ask", json={"question": "教会"})
        assert response.status_code == 200
        payload = response.get_json()
        assert "青砂ヶ浦天主堂" in payload["answer"]
        assert payload["meta"].get("suggestions")


def test_entries_min_query_chars_toggle(monkeypatch, tmp_path):
    _write_entries(tmp_path)
    monkeypatch.setenv("MIN_QUERY_CHARS", "3")
    monkeypatch.setattr(cfg, "MIN_QUERY_CHARS", 3, raising=False)
    monkeypatch.setattr(cfg, "ENABLE_ENTRIES_2CHAR", False, raising=False)

    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "SECRET_KEY": "test",
            "MIN_QUERY_CHARS": 3,
            "ENABLE_ENTRIES_2CHAR": 0,
        },
    ) as module:
        module.handle_mobility = lambda *args, **kwargs: ("", False)
        client = module.app.test_client()
        assert module.load_entries(), "entries should be loaded for search"

        blocked = client.post("/ask", json={"question": "教会"})
        assert blocked.status_code == 202
        skipped = blocked.get_json()
        assert skipped["meta"].get("skipped") == "too_short"

        allowed = client.post("/ask", json={"question": "教会資料"})
        assert allowed.status_code == 200
        payload = allowed.get_json()
        assert "教会資料センター" in payload["answer"]
