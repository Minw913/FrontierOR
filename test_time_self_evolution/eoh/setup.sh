#!/usr/bin/env bash
# Isolated EOH environment; does not mutate CORAL/OpenEvolve environments.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET_DIR="${ROOT_DIR}/external/eoh"
REF="${EOH_REF:-bc1d8810fd726149adeb08d9019c9c892355f6ec}"
VENV_DIR="${EOH_VENV:-${ROOT_DIR}/.venv-eoh}"
if [ ! -d "${TARGET_DIR}/.git" ]; then
  git clone https://github.com/FeiLiu36/EoH.git "${TARGET_DIR}"
fi
git -C "${TARGET_DIR}" fetch origin "${REF}"
git -C "${TARGET_DIR}" checkout "${REF}"
if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PYTHON_BIN:-python3}" -m venv "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt" \
  joblib==1.5.3 numba==0.61.2 openai==3.42.0 pytest==8.4.2
# EvaluationResult is shared with the existing pinned OpenEvolve checkout.
if [ ! -f "${ROOT_DIR}/external/openevolve/pyproject.toml" ]; then
  PATH="${VENV_DIR}/bin:$PATH" bash "${ROOT_DIR}/test_time_self_evolution/openevolve/setup.sh"
fi
"${VENV_DIR}/bin/python" -m pip install -e "${TARGET_DIR}/eoh" -e "${ROOT_DIR}/external/openevolve"
"${VENV_DIR}/bin/python" -m pip check
printf 'EOH ready: %s\n' "${VENV_DIR}/bin/python"
