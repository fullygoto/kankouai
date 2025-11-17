#!/usr/bin/env bash
# Validate that rollback backups can be restored by exercising the verify command.
# Intended for CI/cron. Creates a fresh backup by default and validates its archives
# without touching the live DATA_BASE_DIR contents.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "$PROJECT_ROOT"

python -m manage.rollback verify "$@"
