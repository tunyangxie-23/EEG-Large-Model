#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/prototypes/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "[dummy] virtual environment not found. Run prototypes/scripts/setup_uv_env.sh first." >&2
  exit 1
fi

OUT_DIR="${1:-${ROOT_DIR}/runs/dummy_eeg}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-64}"
VAL_SAMPLES="${VAL_SAMPLES:-16}"
CHANNELS="${CHANNELS:-32}"
TIME_STEPS="${TIME_STEPS:-128}"
AMP_MIN="${AMP_MIN:-0.8}"
AMP_MAX="${AMP_MAX:-1.2}"
NOISE_STD="${NOISE_STD:-0.0}"
SEED="${SEED:-0}"

echo "[dummy] writing dataset to ${OUT_DIR}"
"${VENV_PYTHON}" "${ROOT_DIR}/prototypes/scripts/make_dummy_dataset.py" \
  "${OUT_DIR}" \
  --train-samples "${TRAIN_SAMPLES}" \
  --val-samples "${VAL_SAMPLES}" \
  --channels "${CHANNELS}" \
  --time-steps "${TIME_STEPS}" \
  --amp-min "${AMP_MIN}" \
  --amp-max "${AMP_MAX}" \
  --noise-std "${NOISE_STD}" \
  --seed "${SEED}"
