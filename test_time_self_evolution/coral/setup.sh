#!/usr/bin/env bash
# Clone & pin upstream CORAL.
# Source: https://github.com/Human-Agent-Society/CORAL
# Pinned to HEAD on 2026-04-27 (no upstream release tags).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_DIR="${ROOT_DIR}/external/coral"
REPO_URL="https://github.com/Human-Agent-Society/CORAL.git"
REF="${CORAL_REF:-61bc7619a05e4e0e36e3556f89756f095f36b8db}"

if [ ! -d "${TARGET_DIR}/.git" ]; then
  mkdir -p "$(dirname "${TARGET_DIR}")"
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

git -C "${TARGET_DIR}" fetch --tags
git -C "${TARGET_DIR}" checkout "${REF}"

# Install CORAL into the active Python environment. The hardened runner imports
# this package directly; Agent containers receive a separate broker-only
# `coral eval` shim and never execute this host CLI.
if [ -f "${TARGET_DIR}/pyproject.toml" ]; then
  python -m pip install -e "${TARGET_DIR}"
fi

echo "CORAL ready at ${TARGET_DIR} (${REF})"
CORAL_IMPORT_PATH="$(python -c 'import pathlib, coral; print(pathlib.Path(coral.__file__).resolve())')"
case "${CORAL_IMPORT_PATH}" in
  "${TARGET_DIR}"/*)
    echo "CORAL Python import: ${CORAL_IMPORT_PATH}"
    ;;
  *)
    echo "ERROR: Python imports CORAL from ${CORAL_IMPORT_PATH}, not ${TARGET_DIR}" >&2
    exit 1
    ;;
esac
