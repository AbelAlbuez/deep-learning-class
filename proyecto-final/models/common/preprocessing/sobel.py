"""Sobel edge extraction for MRI structural gradient representation."""
from __future__ import annotations

import cv2
import numpy as np


def apply_sobel(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel edge magnitude from a grayscale image.

    Applies Sobel kernels in X and Y directions, computes the gradient
    magnitude G = sqrt(Gx^2 + Gy^2), and normalises to [0, 1].

    Args:
        gray: uint8 numpy array of shape (H, W) — typically the CLAHE output.

    Returns:
        float32 numpy array of shape (H, W) with values in [0, 1].
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    max_val = magnitude.max()
    if max_val > 0:
        magnitude = magnitude / max_val
    return magnitude.astype(np.float32)
