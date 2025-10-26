#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/prototypes/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "[train-demo] virtual environment missing, bootstrapping..."
  bash "${ROOT_DIR}/prototypes/scripts/setup_uv_env.sh"
fi

source "${VENV_DIR}/bin/activate"

# Force CPU execution for the demo to avoid GPU capability mismatches.
export CUDA_VISIBLE_DEVICES=""

DATA_ROOT="${1:-${ROOT_DIR}/runs/dummy_eeg}"
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[train-demo] dataset not found at ${DATA_ROOT}, generating dummy data..."
  bash "${ROOT_DIR}/prototypes/scripts/create_dummy_dataset.sh" "${DATA_ROOT}"
fi

EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LIMIT_TRAIN="${LIMIT_TRAIN:-64}"
LIMIT_VAL="${LIMIT_VAL:-16}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/runs/eeg_demo}"

echo "[train-demo] starting training (epochs=${EPOCHS}, batch_size=${BATCH_SIZE})"
python -m prototypes.eeg_scale_ar_pipeline \
  --data-root "${DATA_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --limit-train-samples "${LIMIT_TRAIN}" \
  --limit-val-samples "${LIMIT_VAL}" \
  --checkpoint-dir "${CHECKPOINT_DIR}"

echo "[train-demo] finished. Check ${CHECKPOINT_DIR} for artifacts."
