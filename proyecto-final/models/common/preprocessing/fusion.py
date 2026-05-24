"""Multi-channel fusion: stack RGB + CLAHE + Sobel into a 5-channel array."""
from __future__ import annotations

import numpy as np


def fuse_channels(
    rgb: np.ndarray,
    clahe: np.ndarray,
    sobel: np.ndarray,
) -> np.ndarray:
    """Fuse RGB, CLAHE, and Sobel channels into a single (H, W, 5) tensor.

    Channel layout:
        0 — R
        1 — G
        2 — B
        3 — CLAHE (enhanced contrast, grayscale)
        4 — Sobel (edge magnitude)

    All channels are normalised to float32 [0, 1] before stacking.

    Args:
        rgb:   uint8 (H, W, 3) — resized RGB image.
        clahe: uint8 (H, W)    — CLAHE-enhanced grayscale channel.
        sobel: float32 (H, W)  — normalised Sobel edge magnitude [0, 1].

    Returns:
        float32 numpy array of shape (H, W, 5).
    """
    rgb_f = rgb.astype(np.float32) / 255.0          # (H, W, 3)
    clahe_f = clahe.astype(np.float32) / 255.0      # (H, W)
    # sobel is already float32 [0, 1]

    return np.concatenate(
        [
            rgb_f,                          # (H, W, 3)
            clahe_f[..., np.newaxis],       # (H, W, 1)
            sobel[..., np.newaxis],         # (H, W, 1)
        ],
        axis=-1,
    )
