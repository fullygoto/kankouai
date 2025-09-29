"""Utility helpers for managing pamphlet text files on disk."""

from __future__ import annotations

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import os
import re
import unicodedata

from flask import current_app
from werkzeug.utils import secure_filename

from config import PAMPHLET_BASE_DIR, PAMPHLET_CITIES
from app_utils.storage import atomic_write_bytes, atomic_write_text, file_lock


BASE = Path(PAMPHLET_BASE_DIR).expanduser()


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


_SAFE_NAME_RE = re.compile(
    r"[^0-9A-Za-zぁ-んァ-ヶ一-龠々ー_\-\.\(\)\[\]（）【】「」『』・!！?？&＆:：;；,，.。 　]",
)


def _sanitize_name(name: str) -> str:
    """Sanitise filenames while keeping Japanese characters when possible."""

    raw = unicodedata.normalize("NFKC", name or "")
    raw = raw.replace("/", "_").replace("\\", "_").replace("\x00", "")
    raw = raw.strip()
    ascii_candidate = secure_filename(raw)
    chosen = raw if raw and _SAFE_NAME_RE.sub("", raw) else ascii_candidate
    base = (chosen or ascii_candidate or "").strip().lstrip(".")

    if not base:
        base = f"pamphlet_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    if not base.lower().endswith(".txt"):
        if raw.lower().endswith(".txt"):
            base = base + ("" if base.endswith(".txt") else ".txt")
        else:
            raise ValueError("テキスト(.txt)ファイルのみアップロードできます。")

    base = _SAFE_NAME_RE.sub("", base) or base
    base = base.strip()
    if not base:
        base = f"pamphlet_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"

    return base[:120]


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
    mimetype = (getattr(filestorage, "mimetype", "") or "").lower()
    allowed_mimes = {"text/plain", "text/markdown", "application/octet-stream"}
    if mimetype and mimetype not in allowed_mimes:
        raise ValueError("テキスト(.txt)ファイルのみアップロードできます。")

    max_bytes = current_app.config.get("PAMPHLET_UPLOAD_MAX_BYTES")
    if not max_bytes:
        max_bytes = current_app.config.get("MAX_CONTENT_LENGTH")
    if not max_bytes:
        max_bytes = 16 * 1024 * 1024

    data = filestorage.read()
    if len(data) > int(max_bytes):
        raise ValueError("ファイルサイズが大きすぎます。")

    atomic_write_bytes(dest, data)
    return str(dest)


def delete_file(city: str, name: str) -> None:
    """Delete a named text file from the city folder."""

    path = _safe(city, name)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("ファイルが見つかりませんでした。")
    with file_lock(path):
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

    atomic_write_text(path, text, create_backup=make_backup)

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
