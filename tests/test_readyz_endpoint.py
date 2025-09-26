from tests.utils import load_test_app


def test_readyz_reports_build_and_pamphlet_status(monkeypatch, tmp_path):
    base_dir = tmp_path / "pamphlets"

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

        response = client.get("/readyz")
        assert response.status_code == 200

        payload = response.get_json()
        assert payload["status"] in {"ready", "degraded"}
        assert payload["pamphlet_index"]

        per_city = payload["pamphlet_index_status"]
        assert isinstance(per_city, dict)
        # 4 市町すべてが返ること
        assert {"goto", "shinkamigoto", "ojika", "uku"}.issubset(per_city.keys())

        build = payload.get("build")
        assert build is None or "commit" in build or "branch" in build
        if build is not None:
            assert build.get("env")
            assert "dirty" in build
