from tests.utils import load_test_app


def test_rate_storage_prefers_url(monkeypatch, tmp_path):
    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "SECRET_KEY": "test",
            "RATE_STORAGE_URL": "redis://url-primary/0",
            "RATE_STORAGE_URI": "redis://legacy/0",
        },
    ) as module:
        assert module.RATE_STORAGE_URL == "redis://url-primary/0"
        assert module.app.config["RATE_STORAGE_URL"] == "redis://url-primary/0"
        assert module.app.config["RATE_STORAGE_URI"] == "redis://url-primary/0"


def test_rate_storage_falls_back_to_uri(monkeypatch, tmp_path):
    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "SECRET_KEY": "test",
            "RATE_STORAGE_URL": None,
            "RATE_STORAGE_URI": "redis://legacy/1",
        },
    ) as module:
        assert module.RATE_STORAGE_URL == "redis://legacy/1"
        assert module.app.config["RATE_STORAGE_URL"] == "redis://legacy/1"
        assert module.app.config["RATE_STORAGE_URI"] == "redis://legacy/1"


def test_rate_storage_uses_redis_fallback_in_production(monkeypatch, tmp_path):
    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "SECRET_KEY": "test",
            "APP_ENV": "production",
            "FLASK_SECRET_KEY": "prod-secret",
            "RATE_STORAGE_URL": None,
            "RATE_STORAGE_URI": None,
            "REDIS_URL": "redis://auto/0",
        },
    ) as module:
        assert module.RATE_STORAGE_URL == "redis://auto/0"
        assert module.app.config["RATE_STORAGE_URL"] == "redis://auto/0"
        assert module.app.config["RATE_STORAGE_URI"] == "redis://auto/0"
