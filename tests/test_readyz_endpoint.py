import sys

from coreapp import config as cfg
from limits.storage import MemoryStorage

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


def test_readyz_skips_redis_check_for_memory_storage(monkeypatch, tmp_path):
    base_dir = tmp_path / "pamphlets"
    base_dir.mkdir()

    class _RedisStub:
        def __getattr__(self, name):  # pragma: no cover - defensive
            raise AssertionError("redis should not be accessed when using memory storage")

    monkeypatch.setitem(sys.modules, "redis", _RedisStub())

    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "PAMPHLET_BASE_DIR": base_dir,
            "SECRET_KEY": "test",
            "RATE_STORAGE_URL": "memory://",
        },
    ) as module:
        client = module.app.test_client()
        response = client.get("/readyz")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["errors"] == []
        assert not any(w.startswith("redis:") for w in payload["warnings"])


def test_readyz_warns_on_redis_failure(monkeypatch, tmp_path):
    base_dir = tmp_path / "pamphlets"
    base_dir.mkdir()

    class _RedisStub:
        __version__ = "4.0.0"

        def from_url(self, *args, **kwargs):
            return self

        def ping(self):
            raise RuntimeError("redis unavailable")

    monkeypatch.setitem(sys.modules, "redis", _RedisStub())
    monkeypatch.setattr(
        "flask_limiter._extension.storage_from_string",
        lambda uri, **opts: MemoryStorage(),
    )

    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "PAMPHLET_BASE_DIR": base_dir,
            "SECRET_KEY": "test",
            "APP_ENV": "production",
            "FLASK_SECRET_KEY": "prod-secret",
            "OPENAI_API_KEY": "x",
            "LINE_CHANNEL_SECRET": "x",
            "LINE_CHANNEL_ACCESS_TOKEN": "x",
            "UPSTASH_REDIS_URL": "redis://upstash/2",
        },
    ) as module:
        client = module.app.test_client()
        response = client.get("/readyz")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["errors"] == []
        assert any(w.startswith("redis:") for w in payload["warnings"])
