from __future__ import annotations

import shutil
from pathlib import Path

from coreapp import config as cfg

from tests.utils import load_test_app


def _load_readyz_app(monkeypatch, tmp_path, base_dir_name: str):
    base_dir = tmp_path / base_dir_name
    pamphlet_dir = tmp_path / "pamphlets"
    extra_env = {
        "DATA_BASE_DIR": base_dir,
        "PAMPHLET_BASE_DIR": pamphlet_dir,
        "SECRET_KEY": "test",
    }
    return load_test_app(monkeypatch, tmp_path, extra_env=extra_env)


def _assert_data_base_dir_error(response_json, expected: str) -> None:
    errors = response_json.get("errors") or []
    assert expected in errors, errors


def test_readyz_reports_missing_data_base_dir(monkeypatch, tmp_path):
    with _load_readyz_app(monkeypatch, tmp_path, "missing") as module:
        app = module.app
        data_dir = Path(app.config["DATA_BASE_DIR"])
        assert data_dir.exists()
        shutil.rmtree(data_dir)

        client = app.test_client()
        response = client.get("/readyz")

        assert response.status_code == 503
        payload = response.get_json()
        assert payload["status"] == "error"
        _assert_data_base_dir_error(payload, "data_base_dir:not_found")
        assert payload["details"]["data_base_dir"] == str(data_dir)
        assert payload["details"]["pamphlet_base_dir"].endswith("pamphlets")
        flags = payload["details"].get("flags") or {}
        assert flags.get("MIN_QUERY_CHARS") == cfg.MIN_QUERY_CHARS


def test_readyz_reports_when_data_base_dir_is_not_directory(monkeypatch, tmp_path):
    with _load_readyz_app(monkeypatch, tmp_path, "file_path") as module:
        app = module.app
        data_dir = Path(app.config["DATA_BASE_DIR"])
        assert data_dir.exists()
        shutil.rmtree(data_dir)
        data_dir.write_text("not a directory", encoding="utf-8")

        client = app.test_client()
        response = client.get("/readyz")

        assert response.status_code == 503
        payload = response.get_json()
        assert payload["status"] == "error"
        _assert_data_base_dir_error(payload, "data_base_dir:not_directory")


def test_readyz_reports_not_writable(monkeypatch, tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    probe_dir = base / ".readyz_probe"
    probe_dir.mkdir()
    base.chmod(0o555)

    monkeypatch.setenv("DATA_BASE_DIR", str(base))
    monkeypatch.setenv("SECRET_KEY", "test")

    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={"DATA_BASE_DIR": base, "SECRET_KEY": "test"},
    ) as module:
        app = module.app
        assert Path(app.config["DATA_BASE_DIR"]) == base

        client = app.test_client()
        response = client.get("/readyz")
        assert response.status_code == 503
        payload = response.get_json()
        assert payload["status"] == "error"
        _assert_data_base_dir_error(payload, "data_base_dir:not_writable")

    base.chmod(0o755)
    probe_dir.rmdir()
