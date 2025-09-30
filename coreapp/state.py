"""Simple persistence helpers for per-user suspension flags."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator

import fcntl

from services.paths import get_data_base_dir

_STATE_DIR_NAME = "state"
_STATE_FILE_NAME = "suspend.json"


def _state_dir() -> Path:
    base = get_data_base_dir(None)
    directory = base / _STATE_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _state_file() -> Path:
    path = _state_dir() / _STATE_FILE_NAME
    if not path.exists():
        path.write_text("{}", encoding="utf-8")
    return path


def _hash_user_id(user_id: str) -> str:
    import hashlib

    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    return digest


def _load_state(handle) -> Dict[str, bool]:
    handle.seek(0)
    raw = handle.read()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned: Dict[str, bool] = {}
    for key, value in data.items():
        if isinstance(key, str):
            cleaned[key] = bool(value)
    return cleaned


def _dump_state(handle, state: Dict[str, bool]) -> None:
    handle.seek(0)
    json.dump(state, handle, ensure_ascii=False, sort_keys=True)
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())


@contextmanager
def _locked_state() -> Iterator[Dict[str, bool]]:
    path = _state_file()
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            state = _load_state(handle)
            yield state
            _dump_state(handle, state)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def suspend_user(user_id: str) -> None:
    if not user_id:
        return
    hashed = _hash_user_id(user_id)
    with _locked_state() as state:
        state[hashed] = True


def resume_user(user_id: str) -> None:
    if not user_id:
        return
    hashed = _hash_user_id(user_id)
    with _locked_state() as state:
        state[hashed] = False


def is_suspended(user_id: str) -> bool:
    if not user_id:
        return False
    path = _state_file()
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            state = _load_state(handle)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    hashed = _hash_user_id(user_id)
    return bool(state.get(hashed))


def reset_for_tests() -> None:
    path = _state_file()
    path.write_text("{}", encoding="utf-8")


__all__ = ["suspend_user", "resume_user", "is_suspended", "reset_for_tests"]
