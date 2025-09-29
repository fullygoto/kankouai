import io
import json
from pathlib import Path

from tests.utils import load_test_app


def _login_admin(client):
    with client.session_transaction() as sess:
        sess["role"] = "admin"


def test_data_dirs_created(monkeypatch, tmp_path):
    with load_test_app(monkeypatch, tmp_path) as module:
        app = module.app
        base = Path(app.config["DATA_BASE_DIR"])
        expected = {
            base / "pamphlets",
            base / "entries",
            base / "uploads",
            base / "images",
            base / "logs",
        }
        for path in expected:
            assert path.exists() and path.is_dir(), f"missing {path}"


def test_atomic_write_and_read_back(monkeypatch, tmp_path):
    with load_test_app(monkeypatch, tmp_path) as module:
        target = tmp_path / "entries.json"
        module._atomic_json_dump(target, {"value": 1})
        module._atomic_json_dump(target, {"value": 2})
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["value"] == 2


def test_upload_pamphlet_persists(monkeypatch, tmp_path):
    base_dir = tmp_path / "store"
    extra_env = {
        "DATA_BASE_DIR": str(base_dir),
        "SECRET_KEY": "test",
    }

    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as module:
        client = module.app.test_client()
        _login_admin(client)
        response = client.post(
            "/admin/pamphlets/upload",
            data={"city": "goto", "file": (io.BytesIO(b"hello"), "案内.txt")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 302

    with load_test_app(monkeypatch, tmp_path, extra_env=extra_env) as module:
        pamphlet_path = Path(module.app.config["PAMPHLET_BASE_DIR"]) / "goto" / "案内.txt"
        assert pamphlet_path.exists()
        assert pamphlet_path.read_text(encoding="utf-8") == "hello"
