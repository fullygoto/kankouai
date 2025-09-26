"""Utility helpers for managing pamphlet text files on disk."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import os

from config import PAMPHLET_BASE_DIR, PAMPHLET_CITIES


BASE = Path(PAMPHLET_BASE_DIR)


def ensure_dirs() -> None:
    """Ensure the base directory and per-city folders exist."""

    BASE.mkdir(parents=True, exist_ok=True)
    for slug in PAMPHLET_CITIES:
        (BASE / slug).mkdir(parents=True, exist_ok=True)


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
