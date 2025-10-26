# EEG Autoregressive Prototype

This folder hosts a self‑contained prototype that adapts the VAR “next-scale” idea to EEG. Each cumulative frequency band (e.g., f₁, f₁+f₂, …) is treated as one stage in an autoregressive sequence.

## Layout
- `eeg_band_decomposer.py` – FIR helpers to split a raw EEG window into fixed bands and build cumulative stacks.
- `eeg_frequency_dataset.py` – loads `.npz` segments that contain `signal`, `cumulative_bands`, and `bands` arrays.
- `eeg_scale_ar_pipeline.py` – training / validation / inference loop for a lightweight Transformer.
- `requirements.txt` – minimal dependency set for the prototype.
- `scripts/` – convenience wrappers (env bootstrap, dummy data, training demo).

## Environment (uv)
Run once from the repository root (creates `.venv` under `prototypes/`):

```bash
bash prototypes/scripts/setup_uv_env.sh
source prototypes/.venv/bin/activate
```

The script installs [`uv`](https://github.com/astral-sh/uv) if needed, provisions the virtual environment, and installs dependencies from `requirements.txt`.

## Data expectations
Each EEG window must be saved as:

```python
np.savez_compressed(
    ".../sample_xxx.npz",
    signal=raw_channels_time,
    cumulative_bands=stacked_bands,  # shape: (num_bands, channels, time)
    bands=np.asarray([(low1, high1), ...], dtype=np.float32),
    sfreq=np.float32(100.0),
)
```

Place files under `DATA_ROOT/train/` (and optionally `DATA_ROOT/val/`). The preprocessing pipeline in `EEG_data/Process_500Hz.py` already emits this format.

### Quick synthetic dataset

For smoke testing without raw EEG, generate a toy dataset with deterministic band structure (center-frequency sinusoids plus optional noise):

```bash
source prototypes/.venv/bin/activate
bash prototypes/scripts/create_dummy_dataset.sh /tmp/eeg_dummy
```

Useful knobs (pass as environment variables before the script call):

```bash
TRAIN_SAMPLES=512 VAL_SAMPLES=128 \
AMP_MIN=0.9 AMP_MAX=1.1 NOISE_STD=0.0 \
bash prototypes/scripts/create_dummy_dataset.sh /tmp/eeg_struct
```

The generated rollouts follow a known pattern, so training loss decays sharply and the predictions overlay cleanly with ground truth.

## Training & Evaluation
Train the model (example with limited samples for a quick run):

```bash
source prototypes/.venv/bin/activate
python -m prototypes.eeg_scale_ar_pipeline \
    --data-root /tmp/eeg_dummy \
    --epochs 3 \
    --batch-size 4 \
    --limit-train-samples 32 \
    --limit-val-samples 8 \
    --checkpoint-dir runs/eeg_demo
```

Outputs:
- `runs/eeg_demo/best.pt` – checkpoint with best validation loss (contains config, optimizer state, etc.).
- `runs/eeg_demo/demo_prediction.pt` – autoregressive rollout tensor `(stages, channels, time)`.
- `runs/eeg_demo/metrics.json` – per-epoch train/val losses for plotting progress.

For a longer run that demonstrates convergence on the dummy data, try:

```bash
source prototypes/.venv/bin/activate
# Avoid GPU warnings from an unsupported device
export CUDA_VISIBLE_DEVICES=""
python -m prototypes.eeg_scale_ar_pipeline \
    --data-root /workspace/runs/dummy_eeg_long \
    --epochs 60 \
    --batch-size 8 \
    --checkpoint-dir runs/eeg_demo_long \
    --device cpu
```

The script will resume automatically if interrupted (`--resume`).

## Inference only

```bash
source prototypes/.venv/bin/activate
python -m prototypes.eeg_scale_ar_pipeline \
    --data-root /tmp/eeg_dummy \
    --epochs 1 \
    --batch-size 1 \
    --limit-train-samples 1 \
    --limit-val-samples 1 \
    --checkpoint-dir runs/eeg_demo \
    --resume runs/eeg_demo/best.pt
```

The script will load the checkpoint, run one validation pass, regenerate the rollout, and refresh `metrics.json`.

## Visualizing rollouts

Generate a simple per-stage plot of the autoregressive sample:

```bash
source prototypes/.venv/bin/activate
python -m prototypes.scripts.plot_rollout \
    runs/eeg_demo_long/demo_prediction.pt \
    runs/eeg_demo_long/rollout.png \
    --channel 0 \
    --json-out runs/eeg_demo_long/rollout_metrics.json
```

Add `--data-root /path/to/dataset --split val --sample-idx 0` to overlay the ground-truth stack and record per-stage MSEs in the JSON output.

> **GPU note**: passing `--device gpu` is supported, but the pipeline will fall back to CPU automatically if PyTorch reports that the detected GPU is not compatible with the installed wheel (e.g., unsupported compute capability).

## One-command demo

From the repo root:

```bash
bash prototypes/scripts/train_demo.sh
```

This script:
1. Sets up the uv environment (if missing).
2. Generates a dummy dataset under `runs/dummy_eeg`.
3. Trains for a few epochs with tight sample limits so it finishes quickly.
