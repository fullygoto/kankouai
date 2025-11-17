#!/usr/bin/env bash
# Emit /var/data usage metrics in Prometheus text format and optionally append to a log.
set -euo pipefail

DATA_DIR=${DATA_BASE_DIR:-/var/data}
LOG_FILE=${VAR_DATA_METRICS_LOG:-logs/var_data_metrics.log}
TIMESTAMP="$(date -u +%s)"

if [ ! -d "$DATA_DIR" ]; then
  echo "DATA_DIR does not exist: $DATA_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

USAGE_BYTES=$(du -sb "$DATA_DIR" | cut -f1)
AVAIL_BYTES=$(df -B1 "$DATA_DIR" | tail -1 | awk '{print $4}')

cat <<METRICS | tee -a "$LOG_FILE"
# HELP var_data_usage_bytes Total size of DATA_DIR
# TYPE var_data_usage_bytes gauge
var_data_usage_bytes{path="$DATA_DIR"} $USAGE_BYTES $TIMESTAMP
# HELP var_data_available_bytes Free bytes reported by the filesystem
# TYPE var_data_available_bytes gauge
var_data_available_bytes{path="$DATA_DIR"} $AVAIL_BYTES $TIMESTAMP
# HELP var_data_collection_timestamp_seconds Unix timestamp for the measurement
# TYPE var_data_collection_timestamp_seconds gauge
var_data_collection_timestamp_seconds $TIMESTAMP
METRICS
