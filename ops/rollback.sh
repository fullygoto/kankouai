#!/usr/bin/env bash
# Manual Render rollback helper
# Usage:
#   RENDER_API_KEY=... RENDER_SERVICE_ID=... PROD_BASE_URL=https://example.onrender.com \
#     ops/rollback.sh
# Requirements:
#   - Run from a git clone with access to the origin remote
#   - Branch ops/prod-pin stores ops/PROD_COMMIT with the last healthy SHA
set -euo pipefail

: "${RENDER_API_KEY:?RENDER_API_KEY must be set}"
: "${RENDER_SERVICE_ID:?RENDER_SERVICE_ID must be set}"
: "${PROD_BASE_URL:?PROD_BASE_URL must be set}"

RENDER_API_BASE="https://api.render.com/v1"
PIN_BRANCH="ops/prod-pin"

retry() {
  local attempts=$1
  shift
  local count=0
  local delay=2
  until "$@"; do
    exit_code=$?
    count=$((count + 1))
    if [ $count -ge $attempts ]; then
      return $exit_code
    fi
    sleep $((delay ** count))
  done
}

fetch_pin_branch() {
  if git ls-remote --exit-code origin "$PIN_BRANCH" >/dev/null 2>&1; then
    git fetch origin "$PIN_BRANCH:$PIN_BRANCH" --depth=1 >/dev/null 2>&1
  else
    echo "Pin branch $PIN_BRANCH not found" >&2
    exit 1
  fi
}

fetch_pin_branch
PREV_SHA=$(git show "$PIN_BRANCH:ops/PROD_COMMIT" 2>/dev/null | tr -d '\n')
if [ -z "$PREV_SHA" ]; then
  echo "ops/PROD_COMMIT is empty" >&2
  exit 1
fi

echo "Rolling back to $PREV_SHA"
BODY=$(jq -n --arg commit "$PREV_SHA" '{commitId: $commit, clearCache: false}')
DEPLOY_RESPONSE=$(retry 5 curl --fail-with-body -sS \
  -X POST "$RENDER_API_BASE/services/$RENDER_SERVICE_ID/deploys" \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d "$BODY")
DEPLOY_ID=$(echo "$DEPLOY_RESPONSE" | jq -r '.id')
if [ -z "$DEPLOY_ID" ] || [ "$DEPLOY_ID" = "null" ]; then
  echo "Failed to trigger rollback deploy" >&2
  echo "$DEPLOY_RESPONSE"
  exit 1
fi

echo "Triggered rollback deploy $DEPLOY_ID"
while true; do
  STATUS_RESPONSE=$(retry 5 curl --fail-with-body -sS \
    "$RENDER_API_BASE/services/$RENDER_SERVICE_ID/deploys/$DEPLOY_ID" \
    -H "Authorization: Bearer $RENDER_API_KEY")
  STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
  echo "Deploy status: $STATUS"
  case "$STATUS" in
    live|succeeded)
      break
      ;;
    failed|canceled)
      echo "$STATUS_RESPONSE"
      exit 1
      ;;
    *)
      sleep 10
      ;;
  esac
done

echo "Waiting for /readyz at $PROD_BASE_URL"
success=0
for attempt in $(seq 1 300); do
  if retry 3 curl -fsS "$PROD_BASE_URL/readyz" >/dev/null; then
    echo "readyz healthy on attempt $attempt"
    success=1
    break
  fi
  sleep 2
done
if [ "$success" -ne 1 ]; then
  echo "readyz did not become healthy" >&2
  exit 1
fi

retry 5 curl -fsS "$PROD_BASE_URL/healthz" >/dev/null

echo "Rollback completed"
