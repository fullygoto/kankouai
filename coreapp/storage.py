"""Helpers for managing persistent storage directories."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import shutil
import tarfile
import tempfile
from typing import Dict, Iterable

from coreapp import config as cfg


BASE_DIR = Path(cfg.DATA_BASE_DIR)
PAMPHLETS_DIR = Path(cfg.PAMPHLET_BASE_DIR)
BACKUPS_DIR = BASE_DIR / "backups"
_LEGACY_SENTINEL = BASE_DIR / ".pamphlets_legacy_migrated"


def ensure_dirs() -> None:
    """Ensure persistent directories are present."""

    for directory in (BASE_DIR, PAMPHLETS_DIR, BACKUPS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Atomically write *text* to *path* preserving durability semantics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, encoding=encoding, dir=str(path.parent)
    ) as tmp:
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _copy_txt_files(src: Path, dest: Path) -> None:
    for item in src.rglob("*.txt"):
        if not item.is_file():
            continue
        try:
            rel = item.relative_to(src)
        except ValueError:
            rel = item.name
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                src_mtime = item.stat().st_mtime
                dest_mtime = target.stat().st_mtime
            except OSError:
                continue
            # Only overwrite when the source looks newer.
            if src_mtime <= dest_mtime + 1e-6:
                continue
        shutil.copy2(item, target)


def seed_from_repo_if_empty() -> None:
    """Seed pamphlets from the repository when the directory is empty."""

    ensure_dirs()
    seed_dir = Path(cfg.SEED_PAMPHLET_DIR)
    if not seed_dir.exists():
        return
    has_txt = any(PAMPHLETS_DIR.rglob("*.txt"))
    if has_txt:
        return
    _copy_txt_files(seed_dir, PAMPHLETS_DIR)


def migrate_from_legacy_paths() -> None:
    """Rescue pamphlet files from historical locations once."""

    ensure_dirs()
    if _LEGACY_SENTINEL.exists():
        return

    migrated = False
    candidates = [
        Path("pamphlets"),
        Path("data/pamphlets"),
        Path("static/pamphlets"),
    ]
    for src in candidates:
        if not src.exists() or not any(src.rglob("*.txt")):
            continue
        _copy_txt_files(src, PAMPHLETS_DIR)
        migrated = True

    try:
        if migrated:
            _LEGACY_SENTINEL.write_text(
                datetime.utcnow().isoformat(timespec="seconds"),
                encoding="utf-8",
            )
        else:
            _LEGACY_SENTINEL.touch()
    except OSError:
        # Failing to write the sentinel should not block startup.
        pass


def list_city_dirs() -> list[Path]:
    """Return available city directories under the pamphlet root."""

    if not PAMPHLETS_DIR.exists():
        return []
    return [p for p in PAMPHLETS_DIR.iterdir() if p.is_dir()]


def iter_pamphlet_files(pattern: str = "*.txt") -> Iterable[Path]:
    """Iterate over pamphlet files matching *pattern*."""

    if not PAMPHLETS_DIR.exists():
        return iter(())
    return PAMPHLETS_DIR.rglob(pattern)


def count_pamphlet_files() -> int:
    """Return the number of pamphlet text files."""

    return sum(1 for _ in iter_pamphlet_files())


def count_pamphlets_by_city() -> Dict[str, int]:
    """Return a mapping of city directory name to pamphlet file count."""

    counts: Dict[str, int] = {}
    if not PAMPHLETS_DIR.exists():
        return counts

    for city_dir in list_city_dirs():
        try:
            counts[city_dir.name] = sum(1 for _ in city_dir.rglob("*.txt"))
        except OSError:
            continue
    return counts


def create_pamphlet_backup(*, suffix: str | None = None) -> Path:
    """Create a tar.gz backup of the pamphlet directory."""

    ensure_dirs()
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M")
    if suffix:
        timestamp = f"{timestamp}-{suffix}"
    archive = BACKUPS_DIR / f"pamphlets-{timestamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(PAMPHLETS_DIR, arcname="pamphlets")
    return archive


__all__ = [
    "BASE_DIR",
    "PAMPHLETS_DIR",
    "BACKUPS_DIR",
    "ensure_dirs",
    "atomic_write_text",
    "seed_from_repo_if_empty",
    "migrate_from_legacy_paths",
    "list_city_dirs",
    "iter_pamphlet_files",
    "count_pamphlet_files",
    "count_pamphlets_by_city",
    "create_pamphlet_backup",
]

