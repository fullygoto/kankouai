#!/usr/bin/env bash
# Wrapper script for migrate_to_var_data.py with the same options.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/migrate_to_var_data.py" "$@"
