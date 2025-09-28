"""Rollback automation helpers for Render/Flask deployment."""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

from services.paths import get_data_base_dir

from .manifest import BackupManifest, SnapshotEntry, ISO_FORMAT

LOGGER = logging.getLogger("rollback")


def _env_path(key: str, default: str) -> Path:
    return Path(os.getenv(key, default)).expanduser()


def _default_data_base_dir() -> Path:
    return get_data_base_dir()


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass
class RollbackSettings:
    data_base_dir: Path = dataclasses.field(default_factory=_default_data_base_dir)
    backup_dir: Path = dataclasses.field(default_factory=lambda: _env_path("BACKUP_DIR", "/var/tmp/backup"))
    retention: int = dataclasses.field(default_factory=lambda: _env_int("BACKUP_RETENTION", 14))
    ready_timeout: int = dataclasses.field(default_factory=lambda: _env_int("ROLLBACK_READY_TIMEOUT_SEC", 90))
    canary_enabled: bool = dataclasses.field(default_factory=lambda: os.getenv("ROLLBACK_CANARY_ENABLED", "true").lower() in {"1", "true", "yes"})
    manifest_name: str = "manifest.json"
    ready_url: str = os.getenv("ROLLBACK_READY_URL", os.getenv("READYZ_URL", "http://localhost:5000/readyz"))
    code_checkout_enabled: bool = dataclasses.field(default_factory=lambda: os.getenv("ROLLBACK_CODE_CHECKOUT_ENABLED", "1") not in {"0", "false", "no"})
    git_remote: str = os.getenv("ROLLBACK_GIT_REMOTE", "origin")
    health_poll_interval: int = dataclasses.field(default_factory=lambda: _env_int("ROLLBACK_READY_INTERVAL_SEC", 10))
    allowed_admin_ips: str = os.getenv("ALLOW_ADMIN_ROLLBACK_IPS", "")

    @property
    def manifest_path(self) -> Path:
        return self.backup_dir / self.manifest_name


class RollbackError(RuntimeError):
    pass


class RollbackManager:
    def __init__(self, settings: Optional[RollbackSettings] = None) -> None:
        self.settings = settings or RollbackSettings()
        self.settings.backup_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = BackupManifest(self.settings.manifest_path)
        self._ensure_logger()

    def _ensure_logger(self) -> None:
        if LOGGER.handlers:
            return
        log_dir = Path(os.getenv("ROLLBACK_LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_dir / "rollback.log")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(fmt)
        LOGGER.addHandler(handler)
        LOGGER.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Backup creation
    # ------------------------------------------------------------------
    def create_backup(self, notes: str = "") -> SnapshotEntry:
        snapshot_id = BackupManifest.build_snapshot_id("snapshot")
        LOGGER.info("backup start id=%s", snapshot_id)
        data_path = self._create_data_snapshot(snapshot_id)
        db_path = self._create_db_snapshot(snapshot_id)
        entry = SnapshotEntry(
            snapshot_id=snapshot_id,
            created_at=time.strftime(ISO_FORMAT, time.gmtime()),
            data_path=str(data_path),
            data_sha256=self._sha256_file(data_path),
            data_bytes=data_path.stat().st_size,
            db_path=str(db_path),
            db_sha256=self._sha256_file(db_path),
            db_bytes=db_path.stat().st_size,
            app_revision=self._current_git_revision(),
            alembic_revision=self._current_alembic_revision(),
            notes=notes,
        )
        self.manifest.append(entry)
        self._prune_retention()
        LOGGER.info("backup complete id=%s", snapshot_id)
        return entry

    def _prune_retention(self) -> None:
        entries = self.manifest.load()
        entries.sort(key=lambda e: e.created_at, reverse=True)
        keep = entries[: self.settings.retention]
        remove = entries[self.settings.retention :]
        if remove:
            LOGGER.info("retention pruning remove=%s", [r.snapshot_id for r in remove])
        self.manifest.save(keep)
        for entry in remove:
            for path in (entry.data_path, entry.db_path):
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception as exc:  # pragma: no cover - best effort cleanup
                    LOGGER.warning("failed to delete %s: %s", path, exc)

    def _create_data_snapshot(self, snapshot_id: str) -> Path:
        target_dir = self.settings.data_base_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self.settings.backup_dir / f"var-data-{snapshot_id}.tgz"
        with tarfile.open(archive_path, "w:gz") as tar:
            if any(target_dir.iterdir()):
                for item in target_dir.iterdir():
                    tar.add(item, arcname=item.name)
            else:
                temp_marker = target_dir / ".rollback_empty"
                temp_marker.touch(exist_ok=True)
                tar.add(temp_marker, arcname=temp_marker.name)
                temp_marker.unlink(missing_ok=True)
        os.chmod(archive_path, 0o600)
        return archive_path

    def _create_db_snapshot(self, snapshot_id: str) -> Path:
        db_url = os.getenv("DATABASE_URL", os.getenv("DB_URL", ""))
        archive_path = self.settings.backup_dir / f"db-{snapshot_id}.dump"
        if db_url.startswith("sqlite"):
            db_path = self._resolve_sqlite_path(db_url)
            if not db_path.exists():
                raise RollbackError(f"sqlite database not found at {db_path}")
            shutil.copy2(db_path, archive_path)
        elif db_url.startswith("postgres"):  # pragma: no cover - depends on pg_dump
            cmd = [
                "pg_dump",
                "--format=custom",
                "--file",
                str(archive_path),
                db_url,
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except FileNotFoundError as exc:
                raise RollbackError("pg_dump not available") from exc
        else:
            raise RollbackError(f"Unsupported DATABASE_URL: {db_url}")
        os.chmod(archive_path, 0o600)
        return archive_path

    # ------------------------------------------------------------------
    # Restore flow
    # ------------------------------------------------------------------
    def restore_latest_backup(self, *, reason: str, auto: bool = False) -> SnapshotEntry:
        entry = self.manifest.latest()
        if not entry:
            raise RollbackError("no backups available")
        return self.restore_snapshot(entry.snapshot_id, reason=reason, auto=auto)

    def restore_snapshot(self, snapshot_id: str, *, reason: str, auto: bool = False) -> SnapshotEntry:
        entry = self.manifest.get(snapshot_id)
        if not entry:
            raise RollbackError(f"snapshot {snapshot_id} not found")
        txn_id = f"rb-{uuid.uuid4().hex[:12]}"
        LOGGER.warning("rollback start txn=%s snapshot=%s auto=%s reason=%s", txn_id, snapshot_id, auto, reason)
        self._restore_db(entry, txn_id)
        self._restore_data(entry, txn_id)
        if self.settings.code_checkout_enabled:
            self._checkout_git_revision(entry.app_revision, txn_id)
        LOGGER.info("rollback complete txn=%s", txn_id)
        return entry

    def _restore_data(self, entry: SnapshotEntry, txn_id: str) -> None:
        base = self.settings.data_base_dir
        archive = Path(entry.data_path)
        if not archive.exists():
            raise RollbackError(f"data archive missing: {archive}")
        temp_dir = tempfile.mkdtemp(prefix=f"restore-{txn_id}-", dir=str(base.parent))
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=temp_dir)
        if base.exists():
            backup_dir = base.parent / f"{base.name}.pre-{txn_id}"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.move(base, backup_dir)
        shutil.move(temp_dir, base)

    def _restore_db(self, entry: SnapshotEntry, txn_id: str) -> None:
        db_url = os.getenv("DATABASE_URL", os.getenv("DB_URL", ""))
        archive = Path(entry.db_path)
        if db_url.startswith("sqlite"):
            db_path = self._resolve_sqlite_path(db_url)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            if db_path.exists():
                backup = db_path.with_suffix(f".pre-{txn_id}")
                if backup.exists():
                    backup.unlink()
                shutil.move(db_path, backup)
            shutil.copy2(archive, db_path)
        elif db_url.startswith("postgres"):  # pragma: no cover - depends on pg_restore
            cmd = [
                "pg_restore",
                "--clean",
                "--dbname",
                db_url,
                str(archive),
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except FileNotFoundError as exc:
                raise RollbackError("pg_restore not available") from exc
        else:
            raise RollbackError(f"Unsupported DATABASE_URL: {db_url}")
        self._run_alembic_downgrade(entry.alembic_revision)

    def _checkout_git_revision(self, revision: str, txn_id: str) -> None:
        try:
            subprocess.run(["git", "fetch", self.settings.git_remote, revision], check=False)
            subprocess.run(["git", "checkout", revision], check=True)
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            LOGGER.error("git checkout failed txn=%s: %s", txn_id, exc)
            raise RollbackError("git checkout failed") from exc

    def _run_alembic_downgrade(self, revision: str) -> None:
        if revision in {"", "unknown"}:
            return
        try:
            subprocess.run(["alembic", "downgrade", revision], check=True, capture_output=True)
        except FileNotFoundError:  # pragma: no cover
            LOGGER.warning("alembic command not available, skipping downgrade")
        except subprocess.CalledProcessError as exc:  # pragma: no cover
            LOGGER.error("alembic downgrade failed: %s", exc.stderr)
            raise RollbackError("alembic downgrade failed") from exc

    # ------------------------------------------------------------------
    # Canary logic
    # ------------------------------------------------------------------
    def run_canary_after_deploy(self, snapshot_id: Optional[str]) -> bool:
        if not self.settings.canary_enabled:
            return True
        deadline = time.time() + self.settings.ready_timeout
        last_error: Optional[str] = None
        while time.time() < deadline:
            try:
                response = requests.get(self.settings.ready_url, timeout=5)
                if response.status_code == 200:
                    info = {"status": "pass", "checked_at": time.strftime(ISO_FORMAT, time.gmtime())}
                    if snapshot_id:
                        self.manifest.update(snapshot_id, ready_check=info)
                    LOGGER.info("canary succeeded url=%s", self.settings.ready_url)
                    return True
                last_error = f"status={response.status_code}"
            except Exception as exc:  # pragma: no cover - network errors
                last_error = str(exc)
            time.sleep(self.settings.health_poll_interval)
        LOGGER.error("canary failed after timeout error=%s", last_error)
        if snapshot_id:
            self.manifest.update(snapshot_id, ready_check={"status": "fail", "error": last_error or "unknown"})
        self.restore_latest_backup(reason="readyz_fail", auto=True)
        return False

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _resolve_sqlite_path(url: str) -> Path:
        if url.startswith("sqlite:////"):
            return Path(url.replace("sqlite:////", "/", 1))
        if url.startswith("sqlite:///"):
            return Path(url.replace("sqlite:///", "", 1)).resolve()
        if url.startswith("sqlite://"):
            return Path(url.replace("sqlite://", "", 1)).resolve()
        raise RollbackError(f"Invalid sqlite URL: {url}")

    def _current_git_revision(self) -> str:
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True)
            return result.stdout.strip()
        except Exception:  # pragma: no cover
            return "unknown"

    def _current_alembic_revision(self) -> str:
        try:
            result = subprocess.run(["alembic", "current"], capture_output=True, text=True, check=True)
            return result.stdout.strip().split()[-1]
        except Exception:
            return "unknown"


__all__ = ["RollbackManager", "RollbackSettings", "RollbackError"]
