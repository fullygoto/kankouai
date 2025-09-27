from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.rollback_service import RollbackManager, RollbackSettings
from services.manifest import BackupManifest


@pytest.fixture
def rollback_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "var" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "users.json").write_text(json.dumps({"alice": 1}), encoding="utf-8")
    db_path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, name TEXT)")
    cur.execute("INSERT INTO items(name) VALUES ('first')")
    conn.commit()
    conn.close()

    monkeypatch.setenv("DATA_BASE_DIR", str(data_dir))
    backup_dir = tmp_path / "backups"
    monkeypatch.setenv("BACKUP_DIR", str(backup_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ROLLBACK_CODE_CHECKOUT_ENABLED", "0")

    settings = RollbackSettings(
        data_base_dir=data_dir,
        backup_dir=backup_dir,
        retention=5,
        ready_timeout=2,
        canary_enabled=True,
    )
    manager = RollbackManager(settings=settings)
    return manager, data_dir, db_path, backup_dir


def _read_db(path: Path) -> list[tuple[int, str]]:
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM items ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return rows


def test_backup_and_restore_roundtrip(rollback_env, monkeypatch):
    manager, data_dir, db_path, backup_dir = rollback_env
    entry = manager.create_backup(notes="initial")

    # mutate
    (data_dir / "users.json").write_text(json.dumps({"alice": 2}), encoding="utf-8")
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM items")
    conn.execute("INSERT INTO items(name) VALUES ('mutated')")
    conn.commit()
    conn.close()

    restored = manager.restore_latest_backup(reason="test", auto=False)
    assert restored.snapshot_id == entry.snapshot_id

    restored_data = json.loads((data_dir / "users.json").read_text(encoding="utf-8"))
    assert restored_data == {"alice": 1}
    assert _read_db(db_path)[0][1] == "first"

    manifest = BackupManifest(manager.settings.manifest_path)
    saved_entry = manifest.get(entry.snapshot_id)
    assert saved_entry is not None
    assert saved_entry.data_sha256 == entry.data_sha256
    assert saved_entry.db_sha256 == entry.db_sha256


def test_manifest_retention(tmp_path, monkeypatch):
    data_dir = tmp_path / "var" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "file.txt").write_text("seed", encoding="utf-8")
    db_path = tmp_path / "db.sqlite"
    sqlite3.connect(db_path).close()
    backup_dir = tmp_path / "backups"

    monkeypatch.setenv("DATA_BASE_DIR", str(data_dir))
    monkeypatch.setenv("BACKUP_DIR", str(backup_dir))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ROLLBACK_CODE_CHECKOUT_ENABLED", "0")

    settings = RollbackSettings(
        data_base_dir=data_dir,
        backup_dir=backup_dir,
        retention=2,
        ready_timeout=1,
        canary_enabled=False,
    )
    manager = RollbackManager(settings=settings)

    ids = []
    for idx in range(3):
        (data_dir / "file.txt").write_text(f"seed-{idx}", encoding="utf-8")
        ids.append(manager.create_backup(notes=f"run-{idx}").snapshot_id)
    manifest_ids = [entry.snapshot_id for entry in manager.manifest.load()]
    assert len(manifest_ids) == 2
    assert manifest_ids[0] == ids[-1]
    assert manifest_ids[1] == ids[-2]


def test_canary_failure_triggers_rollback(rollback_env, monkeypatch):
    manager, data_dir, db_path, backup_dir = rollback_env
    entry = manager.create_backup(notes="canary")

    class DummyResponse:
        status_code = 500

    manager.settings.ready_timeout = 1
    manager.settings.health_poll_interval = 0
    monkeypatch.setattr("services.rollback_service.requests.get", lambda *a, **k: DummyResponse())

    calls = {}

    def fake_restore_latest(self, reason: str, auto: bool):
        calls["reason"] = reason
        calls["auto"] = auto
        return entry

    monkeypatch.setattr(RollbackManager, "restore_latest_backup", fake_restore_latest)
    result = manager.run_canary_after_deploy(entry.snapshot_id)
    assert result is False
    assert calls["reason"] == "readyz_fail"
    assert calls["auto"] is True


def test_restore_invokes_alembic_downgrade(rollback_env, monkeypatch):
    manager, data_dir, db_path, backup_dir = rollback_env
    entry = manager.create_backup(notes="downgrade")

    called = {}

    def fake_downgrade(self, revision: str):
        called["revision"] = revision

    monkeypatch.setattr(RollbackManager, "_run_alembic_downgrade", fake_downgrade)
    manager.restore_snapshot(entry.snapshot_id, reason="test", auto=False)
    assert called["revision"] == entry.alembic_revision
