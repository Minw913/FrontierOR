#!/usr/bin/env bash
# Clone & pin upstream EoH (Liu et al., ICML 2024) for use as a baseline.
# Source: https://github.com/FeiLiu36/EoH
# Pinned to HEAD on 2026-04-25 (no upstream release tags).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_DIR="${ROOT_DIR}/external/eoh"
REPO_URL="https://github.com/FeiLiu36/EoH.git"
REF="${EOH_REF:-bc1d8810fd726149adeb08d9019c9c892355f6ec}"

if [ ! -d "${TARGET_DIR}/.git" ]; then
  mkdir -p "$(dirname "${TARGET_DIR}")"
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

git -C "${TARGET_DIR}" fetch --tags
git -C "${TARGET_DIR}" checkout "${REF}"

if [ -f "${TARGET_DIR}/eoh/setup.py" ] || [ -f "${TARGET_DIR}/eoh/pyproject.toml" ]; then
  python -m pip install -e "${TARGET_DIR}/eoh"
fi

echo "EoH ready at ${TARGET_DIR} (${REF})"
