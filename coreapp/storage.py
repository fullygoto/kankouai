"""Helpers for managing persistent storage directories."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import logging
import os
import shutil
import tarfile
import tempfile
from typing import Dict, Iterable

from coreapp import config as cfg


log = logging.getLogger(__name__)

# 互換用：外部から参照される可能性があるので残す（呼び出し時に同期される）
BASE_DIR = Path(getattr(cfg, "DATA_BASE_DIR", tempfile.gettempdir()))
PAMPHLETS_DIR = Path(getattr(cfg, "PAMPHLET_BASE_DIR", BASE_DIR / "pamphlets"))
BACKUPS_DIR = BASE_DIR / "backups"
_SYSTEM_DIR = BASE_DIR / "system"
_LEGACY_SENTINEL = BASE_DIR / ".pamphlets_legacy_migrated"


def _is_production() -> bool:
    return os.getenv("APP_ENV") in ("production", "prod")


def _resolve_base_dir() -> Path:
    """
    env -> cfg の順で解決。
    「ディレクトリじゃない」場合でも勝手に別パスへ逃げない（readyzが検出できるようにする）。
    """
    env = os.getenv("DATA_BASE_DIR")
    if env:
        return Path(env)
    return Path(getattr(cfg, "DATA_BASE_DIR", tempfile.gettempdir()))


def _resolve_pamphlets_dir(base_dir: Path) -> Path:
    env = os.getenv("PAMPHLET_BASE_DIR")
    if env:
        return Path(env)

    cfg_dir = Path(getattr(cfg, "PAMPHLET_BASE_DIR", base_dir / "pamphlets"))

    # env で base_dir を変えたのに cfg_dir が古い base 配下を指していそうなら揃える
    try:
        orig_base = Path(getattr(cfg, "DATA_BASE_DIR"))
        if orig_base != base_dir and str(cfg_dir).startswith(str(orig_base)):
            return base_dir / "pamphlets"
    except Exception:
        pass

    return cfg_dir


def _current_dirs() -> tuple[Path, Path, Path, Path, Path]:
    base_dir = _resolve_base_dir()
    pamphlets_dir = _resolve_pamphlets_dir(base_dir)
    backups_dir = base_dir / "backups"
    system_dir = base_dir / "system"
    legacy_sentinel = base_dir / ".pamphlets_legacy_migrated"

    global BASE_DIR, PAMPHLETS_DIR, BACKUPS_DIR, _SYSTEM_DIR, _LEGACY_SENTINEL
    BASE_DIR = base_dir
    PAMPHLETS_DIR = pamphlets_dir
    BACKUPS_DIR = backups_dir
    _SYSTEM_DIR = system_dir
    _LEGACY_SENTINEL = legacy_sentinel

    return base_dir, pamphlets_dir, backups_dir, system_dir, legacy_sentinel


def _mkdir_safe(directory: Path) -> bool:
    """
    directory を作成する。
    directory がファイルとして存在するなど異常でも、production 以外は起動を落とさず False。
    """
    try:
        if directory.exists() and not directory.is_dir():
            msg = f"storage.ensure_dirs: path exists but is not a directory: {directory}"
            if _is_production():
                raise FileExistsError(msg)
            log.warning(msg)
            return False

        directory.mkdir(parents=True, exist_ok=True)
        return True

    except FileExistsError:
        if _is_production():
            raise
        log.warning("storage.ensure_dirs: FileExistsError for %s", directory, exc_info=True)
        return False

    except Exception:
        if _is_production():
            raise
        log.warning("storage.ensure_dirs: failed for %s", directory, exc_info=True)
        return False


def ensure_dirs() -> None:
    """Ensure persistent directories are present (best-effort in non-prod)."""
    base_dir, pamphlets_dir, backups_dir, system_dir, _ = _current_dirs()

    # base が不正でも、readyz などが検出できるように「逃げない」
    _mkdir_safe(base_dir)
    _mkdir_safe(system_dir)
    _mkdir_safe(pamphlets_dir)
    _mkdir_safe(backups_dir)


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
            if src_mtime <= dest_mtime + 1e-6:
                continue
        shutil.copy2(item, target)


def seed_from_repo_if_empty() -> None:
    """Seed pamphlets from the repository when the directory is empty."""
    ensure_dirs()
    _, pamphlets_dir, _, _, _ = _current_dirs()

    seed_dir = Path(cfg.SEED_PAMPHLET_DIR)
    if not seed_dir.exists():
        return
    has_txt = any(pamphlets_dir.rglob("*.txt"))
    if has_txt:
        return
    _copy_txt_files(seed_dir, pamphlets_dir)


def migrate_from_legacy_paths() -> None:
    """Rescue pamphlet files from historical locations once."""
    ensure_dirs()
    _, pamphlets_dir, _, _, legacy_sentinel = _current_dirs()

    if legacy_sentinel.exists():
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
        _copy_txt_files(src, pamphlets_dir)
        migrated = True

    try:
        if migrated:
            legacy_sentinel.write_text(
                datetime.utcnow().isoformat(timespec="seconds"),
                encoding="utf-8",
            )
        else:
            legacy_sentinel.touch()
    except OSError:
        pass


def list_city_dirs() -> list[Path]:
    """Return available city directories under the pamphlet root."""
    _, pamphlets_dir, _, _, _ = _current_dirs()
    if not pamphlets_dir.exists():
        return []
    return [p for p in pamphlets_dir.iterdir() if p.is_dir()]


def iter_pamphlet_files(pattern: str = "*.txt") -> Iterable[Path]:
    """Iterate over pamphlet files matching *pattern*."""
    _, pamphlets_dir, _, _, _ = _current_dirs()
    if not pamphlets_dir.exists():
        return iter(())
    return pamphlets_dir.rglob(pattern)


def count_pamphlet_files() -> int:
    """Return the number of pamphlet text files."""
    return sum(1 for _ in iter_pamphlet_files())


def count_pamphlets_by_city() -> Dict[str, int]:
    """Return a mapping of city directory name to pamphlet file count."""
    counts: Dict[str, int] = {}
    _, pamphlets_dir, _, _, _ = _current_dirs()

    if not pamphlets_dir.exists():
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
    _, pamphlets_dir, backups_dir, _, _ = _current_dirs()

    _mkdir_safe(backups_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M")
    if suffix:
        timestamp = f"{timestamp}-{suffix}"
    archive = backups_dir / f"pamphlets-{timestamp}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(pamphlets_dir, arcname="pamphlets")
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
