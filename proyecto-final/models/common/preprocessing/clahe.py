"""CLAHE: Contrast Limited Adaptive Histogram Equalization for MRI images."""
from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE to an RGB image and return an enhanced grayscale channel.

    Args:
        rgb: uint8 numpy array of shape (H, W, 3).
        clip_limit: CLAHE contrast clip limit.
        tile_grid_size: Grid size for local histogram equalization.

    Returns:
        uint8 numpy array of shape (H, W) — enhanced contrast channel.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)
