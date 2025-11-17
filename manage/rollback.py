"""CLI entrypoint for backup/rollback operations."""
from __future__ import annotations

import json
import click

from services.rollback_service import RollbackManager, RollbackError


@click.group()
def cli() -> None:
    """Rollback management commands."""


@cli.command("backup")
@click.option("--notes", default="", help="Optional notes for the snapshot")
def backup(notes: str) -> None:
    """Create data and database backup."""
    manager = RollbackManager()
    entry = manager.create_backup(notes=notes)
    click.echo(json.dumps(entry.to_dict(), indent=2))


@cli.command("restore")
@click.option("--snapshot", "snapshot_id", default=None, help="Snapshot ID to restore")
@click.option("--auto", is_flag=True, help="Mark rollback as automatic")
@click.option("--reason", default="manual", help="Reason for rollback")
def restore(snapshot_id: str | None, auto: bool, reason: str) -> None:
    """Restore snapshot."""
    manager = RollbackManager()
    if snapshot_id:
        entry = manager.restore_snapshot(snapshot_id, reason=reason, auto=auto)
    else:
        entry = manager.restore_latest_backup(reason=reason, auto=auto)
    click.echo(json.dumps(entry.to_dict(), indent=2))


@cli.command("canary")
@click.option("--snapshot", "snapshot_id", default=None)
def canary(snapshot_id: str | None) -> None:
    """Run health canary against /readyz."""
    manager = RollbackManager()
    success = manager.run_canary_after_deploy(snapshot_id)
    if not success:
        raise SystemExit(1)


@cli.command("verify")
@click.option("--snapshot", "snapshot_id", default=None, help="Snapshot ID to validate")
@click.option(
    "--create-backup/--no-create-backup",
    default=True,
    help="Create a fresh backup before validation",
)
@click.option("--notes", default="restore-verify", help="Optional notes for the validation backup")
def verify(snapshot_id: str | None, create_backup: bool, notes: str) -> None:
    """Validate backup archives without mutating live data directories."""

    manager = RollbackManager()
    if create_backup:
        entry = manager.create_backup(notes=notes)
        snapshot_id = entry.snapshot_id
    entry = manager.validate_snapshot(snapshot_id)
    click.echo(json.dumps({"snapshot_id": entry.snapshot_id, "status": "ok"}, indent=2))


if __name__ == "__main__":
    try:
        cli()
    except RollbackError as exc:  # pragma: no cover - click handles in tests
        raise SystemExit(str(exc))
