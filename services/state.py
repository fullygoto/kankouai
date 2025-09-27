"""User state persistence for LINE control commands."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional

from pathlib import Path

from config import SYSTEM_BASE_DIR

LOGGER = logging.getLogger(__name__)

_DEFAULT_DB_PATH = str(Path(SYSTEM_BASE_DIR) / "user_state.sqlite")
_INIT_LOCK = threading.Lock()
_INITED_PATH: Optional[str] = None


def _db_path() -> str:
    return os.getenv("USER_STATE_DB_PATH", _DEFAULT_DB_PATH)


def _ensure_initialized() -> None:
    global _INITED_PATH
    path = _db_path()
    if _INITED_PATH == path:
        return
    with _INIT_LOCK:
        if _INITED_PATH == path:
            return
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_state(
                    user_id TEXT PRIMARY KEY,
                    paused_until INTEGER,
                    paused_reason TEXT,
                    updated_at INTEGER
                )
                """
            )
        finally:
            conn.close()
        _INITED_PATH = path


def _connect() -> sqlite3.Connection:
    _ensure_initialized()
    conn = sqlite3.connect(_db_path(), timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _exclusive_connection() -> sqlite3.Connection:
    conn = _connect()
    try:
        conn.execute("BEGIN EXCLUSIVE")
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


def _row_to_state(row: sqlite3.Row | None, *, user_id: str) -> Dict[str, Optional[int]]:
    if not row:
        return {
            "user_id": user_id,
            "paused_until": None,
            "paused_reason": None,
            "updated_at": None,
        }
    return {
        "user_id": user_id,
        "paused_until": row["paused_until"],
        "paused_reason": row["paused_reason"],
        "updated_at": row["updated_at"],
    }


def get_state(user_id: str) -> Dict[str, Optional[int]]:
    if not user_id:
        raise ValueError("user_id is required")
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT user_id, paused_until, paused_reason, updated_at FROM user_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    finally:
        conn.close()
    return _row_to_state(row, user_id=user_id)


def pause(
    user_id: str,
    ttl_sec: Optional[int],
    reason: str = "manual",
    *,
    now: Optional[int] = None,
) -> Dict[str, Optional[int]]:
    if not user_id:
        raise ValueError("user_id is required")
    ttl = int(ttl_sec or 0)
    if ttl <= 0:
        raise ValueError("ttl_sec must be positive")
    now_epoch = int(now if now is not None else time.time())
    paused_until = now_epoch + ttl
    with _exclusive_connection() as conn:
        previous = conn.execute(
            "SELECT paused_until, paused_reason, updated_at FROM user_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        conn.execute(
            """
            INSERT INTO user_state(user_id, paused_until, paused_reason, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET paused_until=excluded.paused_until,
                          paused_reason=excluded.paused_reason,
                          updated_at=excluded.updated_at
            """,
            (user_id, paused_until, reason, now_epoch),
        )
    state = get_state(user_id)
    old_paused_until = previous["paused_until"] if previous else None
    state["old_paused_until"] = old_paused_until
    state["old_paused_reason"] = previous["paused_reason"] if previous else None
    LOGGER.info(
        "CTRL action=pause uid=%s ts=%s ttl=%s old_paused_until=%s new_paused_until=%s reason=%s",
        user_id,
        now_epoch,
        ttl,
        old_paused_until,
        state["paused_until"],
        reason,
    )
    return state


def resume(
    user_id: str,
    reason: str = "manual",
    *,
    now: Optional[int] = None,
    log: bool = True,
) -> Dict[str, Optional[int]]:
    if not user_id:
        raise ValueError("user_id is required")
    now_epoch = int(now if now is not None else time.time())
    with _exclusive_connection() as conn:
        previous = conn.execute(
            "SELECT paused_until, paused_reason, updated_at FROM user_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        conn.execute(
            """
            INSERT INTO user_state(user_id, paused_until, paused_reason, updated_at)
            VALUES(?, NULL, NULL, ?)
            ON CONFLICT(user_id)
            DO UPDATE SET paused_until=NULL,
                          paused_reason=NULL,
                          updated_at=excluded.updated_at
            """,
            (user_id, now_epoch),
        )
    state = get_state(user_id)
    old_paused_until = previous["paused_until"] if previous else None
    state["old_paused_until"] = old_paused_until
    state["old_paused_reason"] = previous["paused_reason"] if previous else None
    was_paused = bool(old_paused_until and old_paused_until > now_epoch)
    state["was_paused"] = was_paused
    if log:
        LOGGER.info(
            "CTRL action=resume uid=%s ts=%s old_paused_until=%s new_paused_until=%s reason=%s",
            user_id,
            now_epoch,
            old_paused_until,
            state["paused_until"],
            reason,
        )
    return state


def resume_all(*, now: Optional[int] = None) -> None:
    now_epoch = int(now if now is not None else time.time())
    with _exclusive_connection() as conn:
        conn.execute(
            "UPDATE user_state SET paused_until=NULL, paused_reason=NULL, updated_at=?",
            (now_epoch,),
        )
    LOGGER.info("CTRL action=resume_all ts=%s", now_epoch)


def is_paused(user_id: str, now_epoch: Optional[int] = None) -> bool:
    if not user_id:
        return False
    now_val = int(now_epoch if now_epoch is not None else time.time())
    state = get_state(user_id)
    paused_until = state.get("paused_until")
    if paused_until is None:
        LOGGER.info(
            "STATE uid=%s is_paused=False check_ts=%s src=db paused_until=None",
            user_id,
            now_val,
        )
        return False
    if paused_until <= now_val:
        resume(user_id, reason="expired", now=now_val, log=False)
        LOGGER.info(
            "STATE uid=%s is_paused=False check_ts=%s src=db paused_until=%s (expired)",
            user_id,
            now_val,
            paused_until,
        )
        return False
    LOGGER.info(
        "STATE uid=%s is_paused=True check_ts=%s src=db paused_until=%s",
        user_id,
        now_val,
        paused_until,
    )
    return True


def reset_for_tests() -> None:
    with _exclusive_connection() as conn:
        conn.execute("DELETE FROM user_state")
