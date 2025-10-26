import argparse
import os
from pathlib import Path

import numpy as np

DEFAULT_BANDS = (
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 50.0),
)


def synthesize_sample(
    bands=DEFAULT_BANDS,
    channels: int = 32,
    time_steps: int = 128,
    amplitude_range=(0.8, 1.2),
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Create a structured EEG segment with cumulative frequency bands."""
    rng = rng or np.random.default_rng()
    time = np.linspace(0, 1, time_steps, dtype=np.float32)
    phases = np.linspace(0, 2 * np.pi, channels, endpoint=False, dtype=np.float32)
    band_signals = []
    for low, high in bands:
        freq = (low + high) * 0.5
        waveform = np.sin(2 * np.pi * freq * time + phases[:, None]).astype(np.float32)
        amps = rng.uniform(amplitude_range[0], amplitude_range[1], size=(channels, 1)).astype(np.float32)
        band = amps * waveform
        if noise_std > 0:
            band += rng.normal(0.0, noise_std, size=band.shape).astype(np.float32)
        band_signals.append(band)
    band_stack = np.stack(band_signals, axis=0)
    cumulative = np.cumsum(band_stack, axis=0)
    raw_signal = cumulative[-1]
    return {
        'signal': raw_signal,
        'cumulative_bands': cumulative,
        'bands': np.asarray(bands, dtype=np.float32),
        'sfreq': np.float32(100.0),
    }


def save_split(split_dir: Path, num_samples: int, seed: int | None = None, **kwargs):
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for idx in range(num_samples):
        sample = synthesize_sample(rng=rng, **kwargs)
        path = split_dir / f"sample_{idx:04d}.npz"
        np.savez_compressed(path, **sample)


def main():
    parser = argparse.ArgumentParser(description="Generate a dummy EEG dataset.")
    parser.add_argument("output_dir", type=Path, help="Directory to store the dataset.")
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--time-steps", type=int, default=128)
    parser.add_argument("--amp-min", type=float, default=0.8)
    parser.add_argument("--amp-max", type=float, default=1.2)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    kwargs = dict(
        channels=args.channels,
        time_steps=args.time_steps,
        amplitude_range=(args.amp_min, args.amp_max),
        noise_std=args.noise_std,
    )
    save_split(args.output_dir / "train", args.train_samples, seed=args.seed, **kwargs)
    if args.val_samples > 0:
        save_split(args.output_dir / "val", args.val_samples, seed=args.seed + 1, **kwargs)

    print(f"Dummy dataset ready at {args.output_dir} (train={args.train_samples}, val={args.val_samples})")


if __name__ == "__main__":
    main()
