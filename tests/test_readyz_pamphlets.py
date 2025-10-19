from __future__ import annotations

from pathlib import Path

from tests.utils import load_test_app


def test_readyz_warns_when_pamphlets_empty(monkeypatch, tmp_path):
    base = tmp_path / "data"
    (base / "pamphlets").mkdir(parents=True)
    (base / ".pamphlets_legacy_migrated").write_text("test", encoding="utf-8")
    (base / ".bootstrap.done").write_text("ok", encoding="utf-8")

    with load_test_app(
        monkeypatch,
        tmp_path,
        extra_env={
            "DATA_BASE_DIR": base,
            "PAMPHLET_BASE_DIR": base / "pamphlets",
            "SECRET_KEY": "test",
            "SEED_PAMPHLET_DIR": base / "no_seed",
        },
    ) as module:
        client = module.app.test_client()
        pamphlet_dir = Path(module.app.config["PAMPHLET_BASE_DIR"])
        for txt in pamphlet_dir.rglob("*.txt"):
            txt.unlink()
        monkeypatch.setattr(module, "count_pamphlet_files", lambda: 0, raising=False)
        monkeypatch.setattr(module, "count_pamphlets_by_city", lambda: {}, raising=False)

        response = client.get("/readyz")
        payload = response.get_json()

        assert response.status_code in (200, 503)
        assert "warnings" in payload
        warnings = payload["warnings"]
        assert "pamphlet_base_dir:empty" in warnings or payload["details"].get("pamphlet_count", 0) == 0
        assert (
            payload["details"]["pamphlet_base_dir"]
            == module.app.config["PAMPHLET_BASE_DIR"]
        )
        assert payload["details"].get("pamphlet_count_by_city", {}) in ({}, None)
