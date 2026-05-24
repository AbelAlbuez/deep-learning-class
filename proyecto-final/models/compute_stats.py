"""Compute per-channel mean and std for the 5-channel advanced pipeline.

Usage (from proyecto-final/models/):
    python compute_stats.py /path/to/datasets/Training

The script uses identity normalization (mean=0, std=1) so raw fused values
[R, G, B, CLAHE, Sobel] flow through unchanged, giving accurate statistics.
Paste the printed constants into common/preprocessing/normalization.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.dataset import MRIDataset
from common.preprocessing import AdvancedMRIPreprocessing
from common.preprocessing.normalization import compute_dataset_stats

CHANNEL_NAMES = ["R", "G", "B", "CLAHE", "Sobel"]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python compute_stats.py /path/to/datasets/Training")
        sys.exit(1)

    train_dir = Path(sys.argv[1])
    if not train_dir.exists():
        print(f"Error: directory not found: {train_dir}")
        sys.exit(1)

    # Identity normalization keeps raw fused values in [0, 1]
    strategy = AdvancedMRIPreprocessing(
        mean=[0.0, 0.0, 0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0, 1.0, 1.0],
    )

    print(f"Loading dataset from: {train_dir}")
    dataset = MRIDataset(train_dir, strategy, training=False)
    print(f"Images found: {len(dataset)}")

    # num_workers=0 avoids MPS deadlocks on macOS
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Computing statistics (this may take a minute)…")
    mean, std = compute_dataset_stats(loader)

    print("\nPer-channel statistics:")
    for name, m, s in zip(CHANNEL_NAMES, mean, std):
        print(f"  {name:6s}  mean={m:.4f}  std={s:.4f}")

    print("\nPaste into common/preprocessing/normalization.py:")
    print(f"DEFAULT_MEAN_5CH = {[round(m, 4) for m in mean]}")
    print(f"DEFAULT_STD_5CH  = {[round(s, 4) for s in std]}")


if __name__ == "__main__":
    main()
