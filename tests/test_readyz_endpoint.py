from coreapp import config as cfg

import sys

from tests.utils import load_test_app


def test_readyz_returns_expected_schema(monkeypatch, tmp_path):
    base_dir = tmp_path / "pamphlets"
    base_dir.mkdir()

    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "PAMPHLET_BASE_DIR": base_dir,
            "SECRET_KEY": "test",
        },
    ) as module:
        app = module.app
        client = app.test_client()

        monkeypatch.setattr(
            module,
            "_describe_mount",
            lambda path: {
                "path": str(path),
                "resolved_path": str(path),
                "device": "tmpfs",
                "mount_point": str(path),
                "fs_type": "tmpfs",
                "options": ["rw"],
            },
        )

        response = client.get("/readyz")
        assert response.status_code == 200

        payload = response.get_json()
        assert payload["status"] == "ok"
        assert payload["errors"] == []
        assert "pamphlet_base_dir:empty" in payload["warnings"]

        details = payload["details"]
        assert details["data_base_dir"] == str(app.config["DATA_BASE_DIR"])
        assert details["pamphlet_base_dir"] == str(base_dir)
        assert details.get("pamphlet_count") == 0
        assert details["fs_mount"]["device"] == "tmpfs"
        assert details["fs_mount"]["mount_point"] == str(app.config["DATA_BASE_DIR"])
        flags = details.get("flags") or {}
        assert flags.get("MIN_QUERY_CHARS") == cfg.MIN_QUERY_CHARS
        assert flags.get("ENABLE_ENTRIES_2CHAR") == cfg.ENABLE_ENTRIES_2CHAR
        assert flags.get("EFFECTIVE_MIN_QUERY_CHARS") == cfg.MIN_QUERY_CHARS
        assert flags.get("DATA_BASE_DIR") == str(app.config["DATA_BASE_DIR"])
        assert flags.get("PAMPHLET_BASE_DIR") == str(base_dir)


def test_readyz_warns_when_rate_storage_ping_fails(monkeypatch, tmp_path):
    base_dir = tmp_path / "pamphlets"
    base_dir.mkdir()

    class DummyRedisClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def ping(self):
            raise TimeoutError("redis ping failed")

    class DummyRedisModule:
        def __init__(self):
            self.kwargs = {}

        def from_url(self, url, **kwargs):
            self.kwargs = kwargs
            return DummyRedisClient(**kwargs)

    dummy_redis = DummyRedisModule()
    monkeypatch.setitem(sys.modules, "redis", dummy_redis)

    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "PAMPHLET_BASE_DIR": base_dir,
            "SECRET_KEY": "test",
            "RATE_STORAGE_URI": "redis://localhost:6379/0",
        },
    ) as module:
        client = module.app.test_client()

        response = client.get("/readyz")
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["ok"] is True
        assert any(msg.startswith("redis:") for msg in payload["warnings"])
        assert dummy_redis.kwargs.get("socket_connect_timeout") <= 1
        assert dummy_redis.kwargs.get("socket_timeout") <= 1


def test_readyz_skips_http_rate_storage_ping(monkeypatch, tmp_path):
    base_dir = tmp_path / "pamphlets"
    base_dir.mkdir()

    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "PAMPHLET_BASE_DIR": base_dir,
            "SECRET_KEY": "test",
            "RATE_STORAGE_URI": "https://example.com/redis",
        },
    ) as module:
        client = module.app.test_client()

        response = client.get("/readyz")
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["ok"] is True
        assert "redis:http_scheme" in payload["warnings"]
