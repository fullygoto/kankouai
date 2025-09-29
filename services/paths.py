"""Helpers for resolving application data directories."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any
import os


def _normalize_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    try:
        return path.expanduser().resolve()
    except Exception:
        return path.expanduser()


def default_data_base_dir(app_env: str | None) -> Path:
    """Return the default base directory for persisted application data."""

    # Render の Disk 永続化仕様に合わせ、既定で /var/data を用いる。
    # ローカル開発でも Render との挙動差分を避けるため同一パスを採用するが、
    # テストや個別環境では DATA_BASE_DIR 環境変数で上書き可能とする。
    _ = app_env  # 実質未使用だが既存シグネチャ互換のため残す
    return Path("/var/data")


def get_data_base_dir(config: Mapping[str, Any] | None = None) -> Path:
    env_override = os.environ.get("DATA_BASE_DIR")
    if env_override:
        return _normalize_path(env_override)

    if config is not None:
        config_value = config.get("DATA_BASE_DIR")
        if config_value:
            return _normalize_path(str(config_value))

    app_env = None
    if config is not None:
        app_env = config.get("APP_ENV")
        if isinstance(app_env, str):
            app_env = app_env
        else:
            app_env = None
    return default_data_base_dir(app_env)


def ensure_data_directories(base_dir: Path) -> Mapping[str, Path]:
    """Create and return the canonical directory layout for persisted data."""

    normalized = _normalize_path(base_dir)
    layout = {
        "base": normalized,
        "pamphlets": normalized / "pamphlets",
        "entries": normalized / "entries",
        "uploads": normalized / "uploads",
        "images": normalized / "images",
        "logs": normalized / "logs",
    }

    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)

    # 旧来の data/ 構成を参照するコードがあっても壊さないよう、空であれば作成しておく。
    legacy = normalized / "data"
    legacy.mkdir(parents=True, exist_ok=True)

    return layout
