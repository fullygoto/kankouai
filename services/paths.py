"""Utilities for determining writable data directories."""

from __future__ import annotations

import os
from pathlib import Path


def get_data_base_dir() -> Path:
    """Return the base directory for writable application data.

    The directory is resolved from the ``DATA_BASE_DIR`` environment variable when
    available. When unset, ``APP_ENV`` decides between the production mount point
    (``/var/data``) and a local ``./data`` folder that is writable in development
    and CI environments. The resolved path is created if missing so callers can
    assume it exists.
    """

    env = os.getenv("DATA_BASE_DIR")
    if env:
        base = Path(env).expanduser().resolve()
    else:
        app_env = os.getenv("APP_ENV", "").lower()
        if app_env in {"prod", "production"}:
            base = Path("/var/data").resolve()
        else:
            base = Path("./data").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base
