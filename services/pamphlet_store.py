"""Utility helpers for managing pamphlet text files on disk."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List

import os
import shutil

from config import PAMPHLET_CITIES
from services.paths import get_data_base_dir


BASE = Path(os.getenv("PAMPHLET_BASE_DIR") or get_data_base_dir() / "pamphlets")
_DIRS_READY = False
_DIRS_LOCK = Lock()


def ensure_dirs() -> None:
    """Ensure the base directory and per-city folders exist."""

    global _DIRS_READY
    if _DIRS_READY:
        return
    with _DIRS_LOCK:
        if _DIRS_READY:
            return
        BASE.mkdir(parents=True, exist_ok=True)
        for slug in PAMPHLET_CITIES:
            (BASE / slug).mkdir(parents=True, exist_ok=True)
        _DIRS_READY = True


def _city_root(city: str) -> Path:
    if city not in PAMPHLET_CITIES:
        raise ValueError("対応していない市町です。")
    ensure_dirs()
    return (BASE / city).resolve()


def _sanitize_name(name: str) -> str:
    """Sanitize a filename while keeping non-ASCII characters."""

    base = os.path.basename(name or "")
    base = base.replace("/", "_").replace("\\", "_").replace("\x00", "")

    if not base:
        raise ValueError("ファイル名が空です。")

    if not base.lower().endswith(".txt"):
        raise ValueError("テキスト(.txt)ファイルのみアップロードできます。")

    stem, _ext = os.path.splitext(base)
    if not stem.strip():
        base = f"upload_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"

    return base


def _safe(city: str, name: str) -> Path:
    root = _city_root(city)
    safe = _sanitize_name(name or "")
    path = (root / safe).resolve()
    if not str(path).startswith(str(root)):
        raise ValueError("保存先を特定できません。")
    return path


def list_files(city: str) -> List[Dict[str, object]]:
    """List text files for the given city."""

    root = _city_root(city)
    items: List[Dict[str, object]] = []
    for path in sorted(root.glob("*.txt")):
        try:
            stat = path.stat()
        except OSError:
            continue
        items.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime": datetime.fromtimestamp(stat.st_mtime),
            }
        )
    return items


def save_file(city: str, filestorage) -> str:
    """Save a FileStorage object into the city folder."""

    if filestorage is None or not getattr(filestorage, "filename", ""):
        raise ValueError("ファイルが選択されていません。")
    dest = _safe(city, filestorage.filename or "")
    filestorage.save(dest)
    return str(dest)


def delete_file(city: str, name: str) -> None:
    """Delete a named text file from the city folder."""

    path = _safe(city, name)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("ファイルが見つかりませんでした。")
    path.unlink()


def read_text(city: str, name: str, max_bytes: int | None = None) -> tuple[str, int, float]:
    """Read a UTF-8 text file, optionally truncated for previews."""

    path = _safe(city, name)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("ファイルが見つかりませんでした。")

    stat = path.stat()
    size = stat.st_size
    with path.open("rb") as fh:
        if max_bytes is not None and max_bytes >= 0:
            data = fh.read(max_bytes)
        else:
            data = fh.read()

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")

    return text, size, stat.st_mtime


def write_text(
    city: str,
    name: str,
    text: str,
    *,
    expected_mtime: float | None = None,
    make_backup: bool = True,
) -> float:
    """Overwrite a pamphlet text file with optimistic locking and backup support."""

    path = _safe(city, name)
    path.parent.mkdir(parents=True, exist_ok=True)

    current_stat = None
    try:
        current_stat = path.stat()
    except FileNotFoundError:
        if expected_mtime is not None:
            raise ValueError("他の変更で競合しました。") from None

    if expected_mtime is not None and current_stat is not None:
        # Allow for filesystem precision differences (~1 microsecond)
        if abs(current_stat.st_mtime - expected_mtime) > 1e-6:
            raise ValueError("他の変更で競合しました。")

    if make_backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)

    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)

    return path.stat().st_mtime


def stat_file(city: str, name: str) -> dict:
    """Return size and mtime for a given pamphlet text file."""

    path = _safe(city, name)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("ファイルが見つかりませんでした。")

    stat = path.stat()
    return {"size": stat.st_size, "mtime": stat.st_mtime}


def get_file_path(city: str, name: str) -> Path:
    """Return the resolved path for a pamphlet text file."""

    path = _safe(city, name)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("ファイルが見つかりませんでした。")
    return path
