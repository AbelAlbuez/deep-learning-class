"""Shared utilities: seed, device, and EarlyStopping."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience:   Number of epochs to wait after last improvement.
        mode:       ``'max'`` (higher is better) or ``'min'`` (lower is better).
        min_delta:  Minimum change to qualify as an improvement.
    """

    def __init__(
        self, patience: int = 7, mode: str = "max", min_delta: float = 0.0
    ) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got: {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Return True if training should stop."""
        if self.best is None:
            self.best = score
            return False
        improved = (
            score > self.best + self.min_delta
            if self.mode == "max"
            else score < self.best - self.min_delta
        )
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
