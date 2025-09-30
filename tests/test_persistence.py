from __future__ import annotations

from services.paths import ensure_data_directories, get_data_base_dir


def test_get_data_base_dir_prefers_env(monkeypatch, tmp_path):
    override = tmp_path / "override"
    config_dir = tmp_path / "config"
    monkeypatch.setenv("DATA_BASE_DIR", str(override))

    resolved = get_data_base_dir({"DATA_BASE_DIR": str(config_dir)})

    assert resolved == override.resolve()


def test_get_data_base_dir_uses_config_when_env_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("DATA_BASE_DIR", raising=False)
    config_dir = tmp_path / "config"

    resolved = get_data_base_dir({"DATA_BASE_DIR": str(config_dir)})

    assert resolved == config_dir.resolve()


def test_ensure_data_directories_creates_expected_structure(tmp_path):
    base_dir = tmp_path / "base"
    pamphlet_dir = tmp_path / "pamphlets"
    extra_dir = tmp_path / "extra"

    ensure_data_directories(base_dir, pamphlet_dir=pamphlet_dir, extra_dirs=[extra_dir])

    expected = [
        base_dir,
        base_dir / "data",
        base_dir / "data" / "images",
        base_dir / "logs",
        base_dir / "system",
        pamphlet_dir,
        extra_dir,
    ]

    for path in expected:
        assert path.is_dir(), f"{path} should exist"
