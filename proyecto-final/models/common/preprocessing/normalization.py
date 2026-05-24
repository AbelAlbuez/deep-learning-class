"""Per-channel normalisation utilities for arbitrary-channel tensors."""
from __future__ import annotations

import torch

# Per-channel stats for the 5-channel input [R, G, B, CLAHE, Sobel].
# RGB channels use ImageNet statistics.
# CLAHE and Sobel defaults are dataset-agnostic placeholders; recompute
# with compute_dataset_stats() before final training runs.
DEFAULT_MEAN_5CH = [0.1791, 0.1791, 0.1791, 0.2634, 0.1022]
DEFAULT_STD_5CH  = [0.1884, 0.1884, 0.1884, 0.2465, 0.1424]


def normalize_tensor(
    tensor: torch.Tensor,
    mean: list[float] = DEFAULT_MEAN_5CH,
    std: list[float] = DEFAULT_STD_5CH,
) -> torch.Tensor:
    """Normalize a (C, H, W) tensor channel-wise with given mean and std.

    Args:
        tensor: float32 tensor of shape (C, H, W).
        mean:   per-channel means, length must equal C.
        std:    per-channel standard deviations, length must equal C.

    Returns:
        Normalised float32 tensor of shape (C, H, W).
    """
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1).clamp(min=1e-7)
    return (tensor - mean_t) / std_t


def compute_dataset_stats(
    loader: "torch.utils.data.DataLoader",
) -> tuple[list[float], list[float]]:
    """Compute per-channel mean and std over an entire DataLoader.

    Use this to derive accurate normalization constants from your training set
    before final training. Pass a DataLoader whose preprocessing strategy does
    NOT apply normalization (or use raw fused tensors).

    Returns:
        (mean_per_channel, std_per_channel) as plain Python lists.
    """
    channels = None
    n_pixels = 0

    for batch, _ in loader:
        # batch: (B, C, H, W)
        b, c, h, w = batch.shape
        if channels is None:
            channels = c
            sum_ = torch.zeros(c)
            sum_sq = torch.zeros(c)
        pixels = b * h * w
        flat = batch.view(b, c, -1)          # (B, C, H*W)
        sum_ += flat.sum(dim=(0, 2))
        sum_sq += (flat ** 2).sum(dim=(0, 2))
        n_pixels += pixels

    mean = (sum_ / n_pixels).tolist()
    std = ((sum_sq / n_pixels - torch.tensor(mean) ** 2).sqrt()).tolist()
    return mean, std

