# VAR-style EEG Prediction Foundation Model

A Visual Autoregressive Modeling (VAR) inspired foundation model for EEG signal prediction using next-scale frequency band synthesis.

## Overview

This project adapts the Visual Autoregressive (VAR) "next-scale" prediction paradigm to EEG signals. Instead of predicting images at progressively higher resolutions, we predict EEG signals with progressively more frequency content. Each cumulative frequency band (δ, δ+θ, δ+θ+α, ...) is treated as one stage in an autoregressive sequence, analogous to image scales in VAR.

### Key Concepts

- **Frequency-based Autoregression**: Like VAR predicts from coarse-to-fine image resolutions, our model predicts from low-to-high frequency bands
- **Cumulative Band Synthesis**: Each stage adds a new frequency component to the cumulative EEG signal
- **Transformer Architecture**: Lightweight Transformer encoder predicts next cumulative signal from current aggregate
- **Multi-scale Training**: Progressive training from simple (low-frequency) to complex (full-spectrum) signals

## Project Structure

```
.
├── VAR/                          # Original VAR codebase (adapted for EEG)
│   ├── models/                   # VAR model implementations
│   ├── train.py                  # Distributed training entry point
│   ├── trainer.py                # VARTrainer class
│   ├── dist.py                   # Distributed training utilities
│   └── requirements.txt          # VAR dependencies
│
├── prototypes/                   # EEG-specific prototype implementation
│   ├── eeg_scale_ar_pipeline.py # Lightweight training/inference pipeline
│   ├── eeg_band_decomposer.py   # FIR filter utilities
│   ├── eeg_frequency_dataset.py # EEG dataset loader
│   ├── scripts/                 # Setup and demo scripts
│   └── requirements.txt         # Prototype dependencies
│
└── utils/                        # Shared utilities
    └── data_eeg.py              # EEG data loading helpers
```

## Installation

### Quick Start (Prototypes)

For rapid experimentation with the lightweight prototype:

```bash
cd prototypes
bash scripts/setup_uv_env.sh
source .venv/bin/activate
```

This installs [`uv`](https://github.com/astral-sh/uv) package manager and creates a virtual environment with minimal dependencies.

### Full Installation (VAR Training)

For distributed training with the full VAR framework:

```bash
cd VAR
pip install -r requirements.txt
```

Requirements:
- Python 3.8+
- PyTorch 2.0+ with CUDA support (optional)
- `torch`, `torchvision`, `numpy`, `mne`, `scipy`

## Data Preparation

### EEG Data Format

EEG segments must be preprocessed into `.npz` files with frequency band decomposition:

```python
import numpy as np
from prototypes.eeg_band_decomposer import cumulative_frequency_levels

# Decompose raw EEG into cumulative frequency bands
signal = ...  # shape: (channels, time), e.g. (64, 500)
sfreq = 100.0  # sampling frequency in Hz

cumulative_bands = cumulative_frequency_levels(
    signal, sfreq,
    bands=[(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0)]
)

# Save to disk
np.savez_compressed(
    "sample_001.npz",
    signal=signal,
    cumulative_bands=cumulative_bands,  # shape: (num_bands, channels, time)
    bands=np.array([(0.5, 4.0), (4.0, 8.0), ...], dtype=np.float32),
    sfreq=np.float32(sfreq),
)
```

### Dataset Directory Structure

```
/path/to/dataset/
├── train/
│   ├── sample_001.npz
│   ├── sample_002.npz
│   └── ...
└── val/
    ├── sample_001.npz
    └── ...
```

### Generate Synthetic Test Data

For quick testing without real EEG data:

```bash
source prototypes/.venv/bin/activate
bash prototypes/scripts/create_dummy_dataset.sh /tmp/eeg_dummy
```

This creates deterministic synthetic EEG data with known frequency patterns for validation.

## Usage

### Option 1: Lightweight Prototype (Recommended for Starting)

The prototype provides a self-contained implementation ideal for experimentation:

#### Training

```bash
source prototypes/.venv/bin/activate

python -m prototypes.eeg_scale_ar_pipeline \
    --data-root /tmp/eeg_dummy \
    --epochs 60 \
    --batch-size 8 \
    --d-model 256 \
    --nhead 8 \
    --num-layers 4 \
    --lr 1e-4 \
    --checkpoint-dir runs/eeg_demo \
    --device cpu
```

**Key Arguments:**
- `--data-root`: Path to dataset containing `train/` and `val/` folders
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size per device
- `--d-model`: Transformer embedding dimension
- `--nhead`: Number of attention heads
- `--num-layers`: Number of transformer layers
- `--lr`: Learning rate
- `--checkpoint-dir`: Output directory for checkpoints and predictions
- `--device`: `cpu` or `cuda:0` (auto-detects GPU availability)

**Outputs:**
- `runs/eeg_demo/best.pt` - Best model checkpoint
- `runs/eeg_demo/demo_prediction.pt` - Autoregressive rollout sample
- `runs/eeg_demo/metrics.json` - Training curves

#### Quick Demo

For a complete end-to-end demo (dataset generation + training + inference):

```bash
bash prototypes/scripts/train_demo.sh
```

#### Inference Only

Load a trained checkpoint and generate predictions:

```bash
python -m prototypes.eeg_scale_ar_pipeline \
    --data-root /tmp/eeg_dummy \
    --epochs 1 \
    --batch-size 1 \
    --limit-train-samples 1 \
    --limit-val-samples 1 \
    --checkpoint-dir runs/eeg_demo \
    --resume runs/eeg_demo/best.pt
```

#### Visualize Predictions

```bash
python -m prototypes.scripts.plot_rollout \
    runs/eeg_demo/demo_prediction.pt \
    runs/eeg_demo/rollout.png \
    --channel 0 \
    --data-root /tmp/eeg_dummy \
    --split val \
    --sample-idx 0 \
    --json-out runs/eeg_demo/rollout_metrics.json
```

### Option 2: Full VAR Framework (For Large-Scale Training)

The full VAR implementation supports:
- Distributed multi-GPU/multi-node training
- VQVAE tokenization
- Advanced optimization (mixed precision, gradient accumulation)
- Conditional generation (class labels)

#### Single GPU Training

```bash
cd VAR
python train.py \
    --depth=16 \
    --bs=32 \
    --ep=100 \
    --fp16=1 \
    --data_path=/path/to/eeg_dataset
```

#### Multi-GPU Distributed Training

```bash
cd VAR
torchrun --nproc_per_node=8 train.py \
    --depth=20 \
    --bs=768 \
    --ep=200 \
    --fp16=1 \
    --alng=1e-3 \
    --wpe=0.1 \
    --data_path=/path/to/eeg_dataset
```

**Key Arguments:**
- `--depth`: Model depth (16, 20, 24, 30, 36)
- `--bs`: Global batch size
- `--ep`: Number of epochs
- `--fp16`: Enable mixed precision (1=enabled, 0=disabled)
- `--alng`: AdaLN initialization gamma
- `--wpe`: Warmup epochs fraction
- `--data_path`: Path to dataset root

**Outputs:**
- `local_output/ar-ckpt-last.pth` - Latest checkpoint
- `local_output/ar-ckpt-best.pth` - Best validation checkpoint
- `local_output/log.txt` - Training logs
- Tensorboard logs in `local_output/`

#### Resume Training

Training automatically resumes from the last checkpoint if interrupted:

```bash
# Just rerun the same command - it will auto-resume
torchrun --nproc_per_node=8 train.py --depth=20 --bs=768 --ep=200 ...
```

## Model Architecture

### EEG Scale Autoregressive Transformer

```python
from prototypes.eeg_scale_ar_pipeline import EEGScaleAutoregressiveModel

model = EEGScaleAutoregressiveModel(
    channels=64,        # Number of EEG channels
    time_steps=500,     # Temporal resolution
    d_model=256,        # Embedding dimension
    nhead=8,            # Attention heads
    num_layers=4,       # Transformer layers
    dropout=0.1         # Dropout rate
)

# Input: (batch, channels, time)
# Output: (batch, channels, time)
prediction = model(current_cumulative_band)
```

### Training Process

1. **Input**: Previous cumulative band signal (e.g., δ+θ)
2. **Projection**: Linear projection to d_model dimension
3. **Positional Encoding**: Learnable positional embeddings
4. **Transformer Encoding**: Multi-head self-attention layers
5. **Output Projection**: Linear projection back to channel space
6. **Target**: Next cumulative band signal (e.g., δ+θ+α)

Loss: Mean Squared Error (MSE) between predicted and target signals

### Autoregressive Inference

```python
from prototypes.eeg_scale_ar_pipeline import autoregressive_inference

# Generate full spectrum prediction stage-by-stage
rollout = autoregressive_inference(
    model=trained_model,
    num_stages=5,       # Number of frequency bands
    channels=64,
    time_steps=500,
    device='cpu'
)
# Output shape: (num_stages, channels, time_steps)
```

## Frequency Band Configuration

Default frequency bands follow standard EEG nomenclature:

```python
DEFAULT_BANDS = (
    (0.5, 4.0),    # Delta (δ)
    (4.0, 8.0),    # Theta (θ)
    (8.0, 13.0),   # Alpha (α)
    (13.0, 30.0),  # Beta (β)
    (30.0, 50.0),  # Gamma (γ)
)
```

The model learns to predict:
1. Stage 1: δ
2. Stage 2: δ + θ
3. Stage 3: δ + θ + α
4. Stage 4: δ + θ + α + β
5. Stage 5: δ + θ + α + β + γ (full spectrum)

## Advanced Configuration

### Custom Training Hyperparameters

```bash
python -m prototypes.eeg_scale_ar_pipeline \
    --data-root /path/to/data \
    --epochs 100 \
    --batch-size 16 \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --d-model 512 \
    --nhead 16 \
    --num-layers 8 \
    --dropout 0.2 \
    --grad-clip 1.0 \
    --num-workers 4 \
    --log-interval 50 \
    --checkpoint-dir runs/custom_experiment \
    --seed 42
```

### Limit Dataset Size (for debugging)

```bash
python -m prototypes.eeg_scale_ar_pipeline \
    --data-root /path/to/data \
    --limit-train-samples 100 \
    --limit-val-samples 20 \
    --epochs 5 \
    --checkpoint-dir runs/debug
```

### VAR Framework Advanced Options

See [VAR/README.md](VAR/README.md) for full documentation on:
- Progressive training (`--pg`, `--pg0`, `--pgwp`)
- Learning rate scheduling (`--sche`, `--tlr`, `--twd`)
- Model variants (`--depth`, `--saln`, `--anorm`)
- Optimization (`--opt`, `--tclip`, `--ac`)

## Performance Tips

### GPU Acceleration

```bash
# Enable mixed precision for faster training
python -m prototypes.eeg_scale_ar_pipeline \
    --device cuda:0 \
    --batch-size 32
```

The pipeline automatically falls back to CPU if GPU is unavailable.

### Data Loading

```bash
# Increase workers for faster data loading
python -m prototypes.eeg_scale_ar_pipeline \
    --num-workers 8
```

### Distributed Training

For large-scale experiments, use the VAR framework:

```bash
# 4 GPUs on single node
torchrun --nproc_per_node=4 VAR/train.py --depth=20 --bs=512

# Multi-node (e.g., 2 nodes, 8 GPUs each)
# Node 0:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=NODE0_IP --master_port=29500 \
    VAR/train.py --depth=24 --bs=1024

# Node 1:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=NODE0_IP --master_port=29500 \
    VAR/train.py --depth=24 --bs=1024
```

## Evaluation Metrics

### Training Metrics

- **Train Loss**: MSE on training set
- **Val Loss**: MSE on validation set
- **Gradient Norm**: For monitoring optimization stability

### Inference Metrics

When visualizing with ground truth:

```bash
python -m prototypes.scripts.plot_rollout \
    runs/eeg_demo/demo_prediction.pt \
    output.png \
    --data-root /path/to/data \
    --split val \
    --sample-idx 0 \
    --json-out metrics.json
```

Output `metrics.json` contains:
- **Per-stage MSE**: Reconstruction error for each frequency band
- **Cumulative MSE**: Overall reconstruction quality
- **Signal-to-Noise Ratio**: Quality of predictions

## Troubleshooting

### Common Issues

**1. No GPU detected despite having CUDA**

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Explicitly set device
python -m prototypes.eeg_scale_ar_pipeline --device cuda:0 ...
```

**2. Out of Memory (OOM)**

Reduce batch size or model size:

```bash
python -m prototypes.eeg_scale_ar_pipeline \
    --batch-size 4 \
    --d-model 128 \
    --num-layers 2
```

**3. Dataset not found**

Verify directory structure:

```bash
ls /path/to/dataset/train/*.npz
ls /path/to/dataset/val/*.npz
```

**4. Training loss not decreasing**

- Try lower learning rate: `--lr 1e-5`
- Check gradient clipping: `--grad-clip 0.5`
- Verify data preprocessing
- Test on synthetic data first

## Citation

This project is inspired by Visual Autoregressive Modeling (VAR):

```bibtex
@Article{VAR,
    title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction}, 
    author={Keyu Tian and Yi Jiang and Zehuan Yuan and Bingyue Peng and Liwei Wang},
    year={2024},
    eprint={2404.02905},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## License

This project follows the MIT License. See [VAR/LICENSE](VAR/LICENSE) for details.

## Contributing

For issues, questions, or contributions, please:
1. Check existing documentation in `prototypes/README.md` and `VAR/README.md`
2. Verify your setup with the synthetic data demo
3. Open an issue with reproducible examples

## Acknowledgments

- **VAR**: Original Visual Autoregressive Modeling framework ([GitHub](https://github.com/FoundationVision/VAR))
- **MNE-Python**: EEG filtering and preprocessing utilities
- **PyTorch**: Deep learning framework

