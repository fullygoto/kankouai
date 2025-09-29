"""Helpers for atomic file writes and filesystem-level locking."""

from __future__ import annotations

import datetime as _dt
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import portalocker

__all__ = [
    "file_lock",
    "atomic_write_bytes",
    "atomic_write_text",
]

_LOCK_TIMEOUT = float(os.getenv("FILE_LOCK_TIMEOUT", "10"))


@contextmanager
def file_lock(target: Path | str) -> Generator[None, None, None]:
    """Acquire an exclusive lock associated with ``target``."""

    path = Path(target)
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING
    try:
        with portalocker.Lock(str(lock_path), timeout=_LOCK_TIMEOUT, flags=flags):
            yield
    finally:
        try:
            if lock_path.exists() and lock_path.stat().st_size == 0:
                lock_path.unlink()
        except OSError:
            pass


def _fsync_directory(path: Path) -> None:
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_bytes(path: Path | str, data: bytes, *, create_backup: bool = False) -> None:
    """Atomically write ``data`` to ``path`` with an exclusive lock."""

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

        with file_lock(dest):
            if create_backup and dest.exists():
                timestamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                backup = dest.with_name(f"{dest.name}.bak-{timestamp}")
                try:
                    import shutil

                    shutil.copy2(dest, backup)
                except OSError:
                    pass
            os.replace(tmp_name, dest)
            _fsync_directory(dest.parent)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except OSError:
            pass


def atomic_write_text(
    path: Path | str,
    text: str,
    *,
    encoding: str = "utf-8",
    create_backup: bool = False,
) -> None:
    """Atomically write ``text`` encoded using ``encoding`` to ``path``."""

    atomic_write_bytes(Path(path), text.encode(encoding), create_backup=create_backup)
