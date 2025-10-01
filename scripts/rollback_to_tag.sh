#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <tag>" >&2
  exit 1
fi

TAG_NAME="$1"

git fetch --tags

if ! git rev-parse "${TAG_NAME}" >/dev/null 2>&1; then
  echo "Tag ${TAG_NAME} does not exist" >&2
  exit 1
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "Resetting ${CURRENT_BRANCH} to ${TAG_NAME}" >&2
git reset --hard "${TAG_NAME}"

echo "Local branch ${CURRENT_BRANCH} now matches ${TAG_NAME}."
echo "Push the rollback with: git push --force-with-lease origin ${CURRENT_BRANCH}" >&2
