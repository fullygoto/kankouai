"""Helpers for resolving application data directories."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Any, Iterable
import os


def _normalize_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    try:
        return path.expanduser().resolve()
    except Exception:
        return path.expanduser()


def default_data_base_dir(app_env: str | None) -> Path:
    """Return the default persistent data directory for the environment."""

    env = (app_env or os.environ.get("APP_ENV") or "").lower()
    if env in {"production", "prod", "staging", "stage"}:
        return Path("/var/data")
    return Path("./data")


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


def ensure_data_directories(
    base_dir: Path,
    *,
    pamphlet_dir: str | os.PathLike[str] | None = None,
    extra_dirs: Iterable[str | os.PathLike[str]] | None = None,
) -> None:
    candidates: list[Path] = [base_dir, base_dir / "data", base_dir / "data" / "images", base_dir / "logs", base_dir / "system"]

    if pamphlet_dir:
        candidates.append(_normalize_path(pamphlet_dir))

    if extra_dirs:
        for item in extra_dirs:
            candidates.append(_normalize_path(item))

    for path in candidates:
        path.mkdir(parents=True, exist_ok=True)
