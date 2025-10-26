#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/prototypes/.venv"
REQ_FILE="${ROOT_DIR}/prototypes/requirements.txt"

echo "[setup] using project root: ${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[setup] ERROR: ${PYTHON_BIN} not found on PATH" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv with ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m pip install --upgrade pip
  "${PYTHON_BIN}" -m pip install --upgrade uv
fi

echo "[setup] creating uv virtual environment at ${VENV_DIR}"
uv venv --clear "${VENV_DIR}"

echo "[setup] installing requirements from ${REQ_FILE}"
uv pip install --python "${VENV_DIR}/bin/python" -r "${REQ_FILE}"

echo "[setup] done. activate with:"
echo "source ${VENV_DIR}/bin/activate"
