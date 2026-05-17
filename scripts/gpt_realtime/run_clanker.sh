#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
VENV_DIR="${CLANKER_VENV_DIR:-${ROOT_DIR}/.venv-clanker}"

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3.12}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
else
  echo "Missing python3. Install Python 3.12, then rerun this script." >&2
  exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install websocket-client sounddevice

cd "${ROOT_DIR}"
exec "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/run_clanker.py" "$@"
