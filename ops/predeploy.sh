#!/usr/bin/env bash
set -euo pipefail

BACKUP_ROOT="/var/data"
BACKUP_DIR="$BACKUP_ROOT/_backups"
KEEP_GENERATIONS=5
TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
ARCHIVE_PATH="$BACKUP_DIR/${TIMESTAMP}.tar.gz"

log() {
  echo "[predeploy] $1"
}

log "Ensuring backup directory exists at $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

log "Creating archive $ARCHIVE_PATH"
if tar -C "$BACKUP_ROOT" --exclude="_backups" -czf "$ARCHIVE_PATH" .; then
  log "Backup completed: $ARCHIVE_PATH"
else
  log "Backup failed"
  exit 1
fi

log "Pruning backups to latest $KEEP_GENERATIONS generations"
mapfile -t EXISTING < <(ls -1t "$BACKUP_DIR"/*.tar.gz 2>/dev/null || true)
if [ "${#EXISTING[@]}" -gt "$KEEP_GENERATIONS" ]; then
  for OLD in "${EXISTING[@]:$KEEP_GENERATIONS}"; do
    log "Removing old backup $OLD"
    rm -f "$OLD"
  done
fi

log "Predeploy backup complete"
