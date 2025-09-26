"""User state persistence for LINE control commands."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Dict, Optional

LOGGER = logging.getLogger(__name__)

_DEFAULT_DB_PATH = "/var/data/system/user_state.sqlite"
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
    if not row:
        return {"user_id": user_id, "paused_until": None, "paused_reason": None, "updated_at": None}
    return {
        "user_id": row["user_id"],
        "paused_until": row["paused_until"],
        "paused_reason": row["paused_reason"],
        "updated_at": row["updated_at"],
    }


def pause(user_id: str, ttl_sec: Optional[int], reason: str = "manual", *, now: Optional[int] = None) -> Dict[str, Optional[int]]:
    if not user_id:
        raise ValueError("user_id is required")
    ttl = int(ttl_sec or 0)
    if ttl <= 0:
        raise ValueError("ttl_sec must be positive")
    now_epoch = int(now if now is not None else time.time())
    paused_until = now_epoch + ttl
    with _exclusive_connection() as conn:
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
    LOGGER.info(
        "[user_state] action=pause user_id=%s ttl_sec=%s paused_until=%s reason=%s",  # noqa: G004
        user_id,
        ttl,
        paused_until,
        reason,
    )
    return {
        "user_id": user_id,
        "paused_until": paused_until,
        "paused_reason": reason,
        "updated_at": now_epoch,
    }


def resume(user_id: str, reason: str = "manual", *, now: Optional[int] = None, log: bool = True) -> Dict[str, Optional[int]]:
    if not user_id:
        raise ValueError("user_id is required")
    now_epoch = int(now if now is not None else time.time())
    with _exclusive_connection() as conn:
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
    if log:
        LOGGER.info(
            "[user_state] action=resume user_id=%s paused_until=None reason=%s",  # noqa: G004
            user_id,
            reason,
        )
    return {
        "user_id": user_id,
        "paused_until": None,
        "paused_reason": None,
        "updated_at": now_epoch,
    }


def resume_all(*, now: Optional[int] = None) -> None:
    now_epoch = int(now if now is not None else time.time())
    with _exclusive_connection() as conn:
        conn.execute(
            "UPDATE user_state SET paused_until=NULL, paused_reason=NULL, updated_at=?",
            (now_epoch,),
        )
    LOGGER.info("[user_state] action=resume_all updated_at=%s", now_epoch)  # noqa: G004


def is_paused(user_id: str, now_epoch: Optional[int] = None) -> bool:
    if not user_id:
        return False
    now_val = int(now_epoch if now_epoch is not None else time.time())
    state = get_state(user_id)
    paused_until = state.get("paused_until")
    if paused_until is None:
        return False
    if paused_until <= now_val:
        resume(user_id, reason="expired", now=now_val, log=False)
        return False
    return True


def reset_for_tests() -> None:
    with _exclusive_connection() as conn:
        conn.execute("DELETE FROM user_state")
