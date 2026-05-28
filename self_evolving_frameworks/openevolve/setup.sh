#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TARGET_DIR="${ROOT_DIR}/external/openevolve"
PATCHES_DIR="${SCRIPT_DIR}/patches"
REPO_URL="https://github.com/algorithmicsuperintelligence/openevolve.git"
REF="${OPENEVOLVE_REF:-v0.2.27}"

if [ ! -d "${TARGET_DIR}/.git" ]; then
  mkdir -p "$(dirname "${TARGET_DIR}")"
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

git -C "${TARGET_DIR}" fetch --tags
git -C "${TARGET_DIR}" checkout "${REF}"

# Apply our local patches on top of the pinned upstream ref. Each .patch in
# patches/ is applied in lexicographic order. Idempotent — already-applied
# patches are skipped via the reverse-apply probe, so re-running setup.sh
# (or running it after working-tree edits) is safe.
if [ -d "${PATCHES_DIR}" ]; then
  for patch in "${PATCHES_DIR}"/*.patch; do
    [ -e "${patch}" ] || continue
    name="$(basename "${patch}")"
    if git -C "${TARGET_DIR}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
      echo "  [patch] ${name} already applied — skipping"
      continue
    fi
    if ! git -C "${TARGET_DIR}" apply --check "${patch}" >/dev/null 2>&1; then
      echo "  [patch] ERROR: ${name} cannot be applied cleanly to ${REF}." >&2
      echo "          Most likely the upstream ref shifted past the patch's hunk context." >&2
      exit 1
    fi
    git -C "${TARGET_DIR}" apply "${patch}"
    echo "  [patch] applied ${name}"
  done
fi

python -m pip install -e "${TARGET_DIR}"

echo "OpenEvolve ready at ${TARGET_DIR} (${REF})"
