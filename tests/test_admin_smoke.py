import io
import json
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from werkzeug.security import generate_password_hash

from services import pamphlet_store
from tests.utils import load_test_app


ADMIN_USER = {
    "user_id": "admin@example.com",
    "name": "管理者",
    "password_hash": generate_password_hash("secret"),
    "role": "admin",
    "active": True,
}


CITY_SLUGS = ["goto", "shinkamigoto", "ojika", "uku"]


def _bootstrap_base_dir(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "entries.json").write_text("[]", encoding="utf-8")
    (base_dir / "synonyms.json").write_text("{}", encoding="utf-8")
    (base_dir / "notices.json").write_text("[]", encoding="utf-8")
    (base_dir / "shop_infos.json").write_text("[]", encoding="utf-8")
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_log = {
        "timestamp": "2024-01-01T12:00:00",
        "question": "未ヒットの質問",
        "answer": "",
        "hit_db": False,
        "source": "line",
        "extra": {},
    }
    (log_dir / "questions_log.jsonl").write_text(
        "\n".join(json.dumps(sample_log, ensure_ascii=False) for _ in range(2)),
        encoding="utf-8",
    )
    (base_dir / "users.json").write_text(
        json.dumps([ADMIN_USER], ensure_ascii=False),
        encoding="utf-8",
    )
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "initial.txt").write_text("seed", encoding="utf-8")

    pamphlet_root = base_dir / "pamphlets"
    for slug in CITY_SLUGS:
        city_dir = pamphlet_root / slug
        city_dir.mkdir(parents=True, exist_ok=True)
        (city_dir / "readme.txt").write_text("パンフレットの下書き", encoding="utf-8")


@contextmanager
def _admin_app(monkeypatch, tmp_path: Path):
    _bootstrap_base_dir(tmp_path)
    pamphlet_store.BASE = tmp_path / "pamphlets"
    pamphlet_store.ensure_dirs()
    extra_env = {
        "SECRET_KEY": "test",
        "PAMPHLET_BASE_DIR": str(tmp_path / "pamphlets"),
        "ENABLE_PAMPHLET_SUMMARY": "true",
    }
    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as module:
        client = module.app.test_client()
        yield module, client


def _login(client):
    response = client.post(
        "/login",
        data={"username": ADMIN_USER["user_id"], "password": "secret"},
        follow_redirects=False,
    )
    assert response.status_code == 302
    return client


def test_admin_login_logout_flow(monkeypatch, tmp_path):
    with _admin_app(monkeypatch, tmp_path) as (_module, client):
        login_page = client.get("/login")
        assert login_page.status_code == 200

        gated = client.get("/admin/entry", follow_redirects=False)
        assert gated.status_code == 302
        assert gated.headers["Location"].endswith("/login")

        _login(client)
        dashboard = client.get("/admin/entry")
        assert dashboard.status_code == 200
        assert "観光データ管理" in dashboard.get_data(as_text=True)

        logout = client.get("/logout", follow_redirects=False)
        assert logout.status_code == 302
        assert logout.headers["Location"].endswith("/login")


def test_admin_entry_crud_and_search(monkeypatch, tmp_path):
    with _admin_app(monkeypatch, tmp_path) as (module, client):
        _login(client)

        listing = client.get("/admin/entry")
        assert listing.status_code == 200

        create_payload = {
            "category": "観光",
            "title": "テストスポット",
            "desc": "説明です",
            "address": "長崎県五島市",
            "areas": ["五島市"],
            "links": "https://example.com",
            "tags": "海,絶景",
            "source": "テスト出典",
        }
        created = client.post("/admin/entry", data=create_payload, follow_redirects=False)
        assert created.status_code == 302

        entries = json.loads((tmp_path / "entries.json").read_text(encoding="utf-8"))
        assert entries and entries[0]["title"] == "テストスポット"

        search_res = client.get("/admin/entry", query_string={"q": "テスト"})
        assert search_res.status_code == 200
        assert "テストスポット" in search_res.get_data(as_text=True)

        edit_payload = {
            "edit_id": "0",
            "category": "観光",
            "title": "更新スポット",
            "desc": "更新説明",
            "address": "長崎県五島市",
            "areas": ["五島市"],
            "links": "https://example.com",
            "tags": "海",
        }
        edited = client.post("/admin/entry", data=edit_payload, follow_redirects=False)
        assert edited.status_code == 302

        updated_entries = json.loads((tmp_path / "entries.json").read_text(encoding="utf-8"))
        assert updated_entries[0]["title"] == "更新スポット"

        deleted = client.post("/admin/entry/delete/0", follow_redirects=False)
        assert deleted.status_code == 302
        assert json.loads((tmp_path / "entries.json").read_text(encoding="utf-8")) == []


def test_admin_pamphlet_management(monkeypatch, tmp_path):
    with _admin_app(monkeypatch, tmp_path) as (module, client):
        _login(client)

        index = client.get("/admin/pamphlets", query_string={"city": "goto"})
        assert index.status_code == 200

        upload = client.post(
            "/admin/pamphlets/upload",
            data={
                "city": "goto",
                "file": (io.BytesIO("本文".encode("utf-8")), "guide.txt"),
            },
            content_type="multipart/form-data",
        )
        assert upload.status_code == 302

        assert pamphlet_store.BASE.resolve() == (tmp_path / "pamphlets").resolve()
        files = pamphlet_store.list_files("goto")
        assert any(f["name"] == "guide.txt" for f in files)
        target = tmp_path / "pamphlets" / "goto" / "guide.txt"
        if not target.exists():
            target = tmp_path / "pamphlets" / "goto" / files[0]["name"]
        assert target.exists()
        mtime = str(target.stat().st_mtime)

        save = client.post(
            "/admin/pamphlets/save",
            data={
                "city": "goto",
                "name": "guide.txt",
                "expected_mtime": mtime,
                "content": "更新された本文",
            },
        )
        assert save.status_code == 302
        assert target.read_text(encoding="utf-8") == "更新された本文"

        def fake_answer(question, city):
            return {
                "answer": "プレビュー本文",
                "sources": [{"doc_id": f"{city}/guide.txt"}],
                "debug": {"prompt": "PROMPT", "combined": []},
            }

        monkeypatch.setattr(module.pamphlet_rag, "answer_from_pamphlets", fake_answer)
        debug = client.get(
            "/admin/pamphlets/debug",
            query_string={"city": "goto", "q": "案内"},
        )
        assert debug.status_code == 200
        payload = debug.get_json()
        assert payload["answer"] == "プレビュー本文"
        assert payload["sources"][0]["doc_id"].endswith("guide.txt")


def test_admin_backup_restore_and_status(monkeypatch, tmp_path):
    with _admin_app(monkeypatch, tmp_path) as (module, client):
        _login(client)

        page = client.get("/admin/backup")
        assert page.status_code == 200

        dl = client.get("/admin/backup", query_string={"download": "1"})
        assert dl.status_code == 200
        assert dl.headers["Content-Type"].startswith("application/zip")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("entries.json", json.dumps([{ "title": "zip復元" }], ensure_ascii=False))
            zf.writestr("synonyms.json", "{}")
            zf.writestr("notices.json", "[]")
        buf.seek(0)

        restore = client.post(
            "/admin/restore",
            data={"backup_zip": (buf, "backup.zip")},
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        assert restore.status_code == 302
        restored = json.loads((tmp_path / "entries.json").read_text(encoding="utf-8"))
        assert restored[0]["title"] == "zip復元"

        url_buf = io.BytesIO()
        with zipfile.ZipFile(url_buf, "w") as zf:
            zf.writestr("entries.json", json.dumps([{ "title": "URL復元" }], ensure_ascii=False))
            zf.writestr("synonyms.json", "{}")
            zf.writestr("notices.json", "[]")
        url_buf.seek(0)

        class DummyResponse(io.BytesIO):
            def __init__(self, payload: bytes):
                super().__init__(payload)
                self.headers = {"Content-Length": str(len(payload))}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()

        monkeypatch.setattr("socket.gethostbyname", lambda host: "93.184.216.34")
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=60, context=None: DummyResponse(url_buf.getvalue()))
        monkeypatch.setattr("ssl.create_default_context", lambda: object())

        restore_url = client.post(
            "/admin/restore_from_url",
            data={"backup_url": "https://example.com/backup.zip"},
            follow_redirects=False,
        )
        assert restore_url.status_code == 302
        restored_url = json.loads((tmp_path / "entries.json").read_text(encoding="utf-8"))
        assert restored_url[0]["title"] == "URL復元"

        stats = client.get("/admin/storage_stats")
        assert stats.status_code == 200
        body = stats.get_json()
        assert body["base_dir"] == str(tmp_path)

        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.get_json()["status"] in {"ready", "degraded"}

        health = client.get("/healthz")
        assert health.status_code == 200


def test_admin_logs_and_unhit(monkeypatch, tmp_path):
    with _admin_app(monkeypatch, tmp_path) as (_module, client):
        _login(client)

        logs_page = client.get("/admin/logs")
        assert logs_page.status_code == 200
        assert "未ヒットの質問" in logs_page.get_data(as_text=True)

        unhit_page = client.get("/admin/unhit_questions")
        assert unhit_page.status_code == 200
        assert "未ヒット" in unhit_page.get_data(as_text=True)

        unhit_json = client.get("/admin/unhit_questions", query_string={"fmt": "json"})
        assert unhit_json.status_code == 200
        payload = unhit_json.get_json()
        assert payload["ok"] is True
        assert payload["stats"]["total_unhit"] >= 1
