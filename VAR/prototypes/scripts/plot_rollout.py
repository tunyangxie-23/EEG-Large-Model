#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from prototypes.eeg_frequency_dataset import EEGFrequencyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize autoregressive rollout.")
    parser.add_argument("rollout", type=Path, help="Path to demo_prediction.pt")
    parser.add_argument("output", type=Path, help="Output image file (e.g., rollout.png)")
    parser.add_argument("--data-root", type=str, default=None, help="Optional dataset root for ground-truth comparison.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index for ground truth.")
    parser.add_argument("--channel", type=int, default=0, help="Channel index to visualize.")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to use if --data-root is provided.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON metrics output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    recon = torch.load(args.rollout, map_location="cpu")
    if recon.ndim != 3:
        raise ValueError(f"Expected rollout tensor with shape (stages, channels, time); got {tuple(recon.shape)}")
    stages, channels, time_steps = recon.shape
    channel = max(0, min(args.channel, channels - 1))

    target = None
    mse_per_stage = None
    if args.data_root:
        dataset = EEGFrequencyDataset(args.data_root, split=args.split)
        sample, _ = dataset[min(args.sample_idx, len(dataset) - 1)]
        target = sample.to(dtype=recon.dtype)
        if target.shape != recon.shape:
            print(f"[warn] target shape {tuple(target.shape)} != rollout shape {tuple(recon.shape)}; skipping overlay")
            target = None
        else:
            mse = torch.mean((recon - target) ** 2, dim=(1, 2))
            mse_per_stage = mse.tolist()

    fig, axes = plt.subplots(stages, 1, figsize=(12, 2 * stages), sharex=True)
    if stages == 1:
        axes = [axes]
    time = torch.arange(time_steps)
    for idx in range(stages):
        ax = axes[idx]
        ax.plot(time, recon[idx, channel].numpy(), label="generated", color="C0")
        if target is not None:
            ax.plot(time, target[idx, channel].numpy(), label="target", color="C1", linestyle="--", alpha=0.7)
        ax.set_ylabel(f"Stage {idx}")
        if idx == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time step")
    os.makedirs(args.output.parent, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)
    print(f"[viz] saved figure to {args.output}")

    if args.json_out:
        stats = {
            "rollout_path": str(args.rollout),
            "channel": channel,
            "stages": stages,
            "time_steps": time_steps,
        }
        if mse_per_stage is not None:
            stats["mse_per_stage"] = mse_per_stage
        with open(args.json_out, "w", encoding="utf-8") as fp:
            json.dump(stats, fp, indent=2)
        print(f"[viz] saved metrics to {args.json_out}")


if __name__ == "__main__":
    main()
