#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <tag> [ref]" >&2
  exit 1
fi

TAG_NAME="$1"
REF="${2:-}"

if git rev-parse "${TAG_NAME}" >/dev/null 2>&1; then
  echo "Tag ${TAG_NAME} already exists" >&2
  exit 1
fi

git fetch --tags

if [[ -n "${REF}" ]]; then
  git tag -a "${TAG_NAME}" "${REF}" -m "Release ${TAG_NAME}"
else
  git tag -a "${TAG_NAME}" -m "Release ${TAG_NAME}"
fi

git push origin "${TAG_NAME}"
