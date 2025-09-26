"""File storage helpers for pamphlet text management."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from werkzeug.utils import secure_filename

from config import PAMPHLET_BASE_DIR, PAMPHLET_CITIES


BASE = Path(PAMPHLET_BASE_DIR)


def ensure_dirs() -> None:
    """Ensure the base directory and per-city folders exist."""

    BASE.mkdir(parents=True, exist_ok=True)
    for slug in PAMPHLET_CITIES:
        (BASE / slug).mkdir(parents=True, exist_ok=True)


def _city_root(city: str) -> Path:
    if city not in PAMPHLET_CITIES:
        raise ValueError("unknown city")
    ensure_dirs()
    return (BASE / city).resolve()


def _safe(city: str, name: str) -> Path:
    root = _city_root(city)
    safe = secure_filename(name or "")
    if not safe:
        raise ValueError("invalid filename")
    path = (root / safe).resolve()
    if not str(path).startswith(str(root)):
        raise ValueError("bad path")
    if path.suffix.lower() != ".txt":
        raise ValueError("txt only")
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
        raise ValueError("file required")
    filename = filestorage.filename or ""
    if not filename.lower().endswith(".txt"):
        raise ValueError("txt only")
    dest = _safe(city, filename)
    filestorage.save(dest)
    return str(dest)


def delete_file(city: str, name: str) -> None:
    """Delete a named text file from the city folder."""

    path = _safe(city, name)
    if path.exists() and path.is_file():
        path.unlink()
