"""Backup manifest management for rollback system."""
from __future__ import annotations

import dataclasses
import datetime as dt
import fcntl
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ISO_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class SnapshotEntry:
    """Represents a single snapshot entry in the manifest."""

    snapshot_id: str
    created_at: str
    data_path: str
    data_sha256: str
    data_bytes: int
    db_path: str
    db_sha256: str
    db_bytes: int
    app_revision: str
    alembic_revision: str
    notes: str = ""
    ready_check: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "SnapshotEntry":
        return cls(**payload)  # type: ignore[arg-type]


class BackupManifest:
    """Thread/process safe manifest stored as JSON."""

    def __init__(self, manifest_path: os.PathLike[str] | str) -> None:
        self.path = Path(manifest_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_raw(self) -> Dict[str, object]:
        if not self.path.exists():
            return {"snapshots": []}
        with self.path.open("r", encoding="utf-8") as fh:
            content = fh.read().strip()
        if not content:
            return {"snapshots": []}
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"snapshots": []}

    def _save_raw(self, payload: Dict[str, object]) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        tmp_path.replace(self.path)

    def _lock_file(self):
        class _LockCtx:
            def __init__(self, file_path: Path) -> None:
                self._file_path = file_path
                self._fh = None

            def __enter__(self):
                self._file_path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self._file_path.open("a+")
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
                self._fh.seek(0)
                return self._fh

            def __exit__(self, exc_type, exc, tb):
                if self._fh:
                    try:
                        fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                    finally:
                        self._fh.close()

        return _LockCtx(self.path)

    def load(self) -> List[SnapshotEntry]:
        raw = self._load_raw()
        entries: Iterable[Dict[str, object]] = raw.get("snapshots", [])  # type: ignore[assignment]
        return [SnapshotEntry.from_dict(item) for item in entries]

    def save(self, entries: Iterable[SnapshotEntry]) -> None:
        payload = {"snapshots": [entry.to_dict() for entry in entries]}
        self._save_raw(payload)

    def append(self, entry: SnapshotEntry) -> None:
        with self._lock_file():
            data = self._load_raw()
            snapshots = data.setdefault("snapshots", [])
            snapshots.append(entry.to_dict())
            self._save_raw(data)

    def prune(self, retention: int) -> List[SnapshotEntry]:
        """Remove old entries keeping ``retention`` newest ones."""
        entries = self.load()
        entries.sort(key=lambda e: e.created_at, reverse=True)
        keep = entries[:retention]
        if len(entries) > retention:
            self.save(keep)
        return keep

    def update(self, snapshot_id: str, **fields: object) -> SnapshotEntry:
        entries = self.load()
        for entry in entries:
            if entry.snapshot_id == snapshot_id:
                for key, value in fields.items():
                    if hasattr(entry, key):
                        setattr(entry, key, value)
                self.save(entries)
                return entry
        raise KeyError(snapshot_id)

    def get(self, snapshot_id: str) -> Optional[SnapshotEntry]:
        for entry in self.load():
            if entry.snapshot_id == snapshot_id:
                return entry
        return None

    def latest(self) -> Optional[SnapshotEntry]:
        entries = self.load()
        if not entries:
            return None
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[0]

    @staticmethod
    def build_snapshot_id(prefix: str = "snapshot") -> str:
        now = dt.datetime.utcnow().strftime(ISO_FORMAT)
        safe = now.replace(":", "").replace("-", "")
        return f"{prefix}-{safe}"


__all__ = ["BackupManifest", "SnapshotEntry"]
