from __future__ import annotations

from pathlib import Path

from tests.utils import load_test_app


def test_pamphlets_survive_restart(monkeypatch, tmp_path):
    base = tmp_path / "data"
    (base / "pamphlets" / "goto").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DATA_BASE_DIR", str(base))
    monkeypatch.setenv("PAMPHLET_BASE_DIR", str(base / "pamphlets"))

    env = {
        "DATA_BASE_DIR": base,
        "PAMPHLET_BASE_DIR": base / "pamphlets",
        "SECRET_KEY": "test",
        "SEED_PAMPHLET_DIR": base / "no_seed",
    }

    with load_test_app(monkeypatch, tmp_path, extra_env=env) as module:
        pamphlet_dir = Path(module.app.config["PAMPHLET_BASE_DIR"])
        pamphlet = pamphlet_dir / "goto" / "sample.txt"
        pamphlet.parent.mkdir(parents=True, exist_ok=True)
        pamphlet.write_text("五島の歴史…", encoding="utf-8")

    with load_test_app(monkeypatch, tmp_path, extra_env=env) as module:
        pamphlet_dir = Path(module.app.config["PAMPHLET_BASE_DIR"])
        pamphlet = pamphlet_dir / "goto" / "sample.txt"
        assert pamphlet.exists()
        assert "五島" in pamphlet.read_text(encoding="utf-8")
