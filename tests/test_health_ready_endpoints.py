from tests.utils import load_test_app


def test_healthz_returns_json_ok(monkeypatch, tmp_path):
    with load_test_app(monkeypatch, tmp_path, extra_env={"SECRET_KEY": "test"}) as module:
        client = module.app.test_client()

        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.is_json
        assert response.get_json() == {"ok": True}
        assert response.headers["Cache-Control"] == "no-store"

        head = client.head("/healthz")
        assert head.status_code == 200
        assert head.headers["Content-Type"].startswith("application/json")


def test_readyz_includes_ok_flag(monkeypatch, tmp_path):
    pamphlet_dir = tmp_path / "pamphlets"
    pamphlet_dir.mkdir()

    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "PAMPHLET_BASE_DIR": pamphlet_dir,
            "SECRET_KEY": "test",
        },
    ) as module:
        client = module.app.test_client()

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
        assert payload["ok"] is True
        assert payload["errors"] == []
        assert payload["status"] == "ok"
        assert payload["details"]["data_base_dir"] == str(module.app.config["DATA_BASE_DIR"])
        assert payload["details"]["pamphlet_count"] == 0
        assert "pamphlet_count_by_city" in payload["details"]
