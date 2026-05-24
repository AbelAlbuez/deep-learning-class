"""AdvancedMRIPreprocessing: 5-channel multimodal pipeline for model3.

Pipeline (train):
    MRI → RGB → Resize → CLAHE → Sobel → Fusion → Augmentation → Tensor → Normalize

Pipeline (val/test):
    MRI → RGB → Resize → CLAHE → Sobel → Fusion → Tensor → Normalize

Output tensor shape: (5, 224, 224)
Channel layout: [R, G, B, CLAHE, Sobel]
"""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .clahe import apply_clahe
from .fusion import fuse_channels
from .normalization import DEFAULT_MEAN_5CH, DEFAULT_STD_5CH, normalize_tensor
from .sobel import apply_sobel

IMAGE_SIZE = 224


class AdvancedMRIPreprocessing:
    """Advanced MRI preprocessing producing 5-channel tensors.

    Combines RGB intensity, CLAHE local contrast enhancement, and Sobel
    edge gradients into a single multimodal tensor. Augmentation is applied
    identically across all 5 channels to preserve spatial alignment.

    Args:
        image_size:        Target spatial resolution (square).
        clahe_clip_limit:  CLAHE contrast clip limit.
        clahe_tile_grid:   CLAHE tile grid size.
        mean:              Per-channel means for normalisation (5 values).
        std:               Per-channel stds for normalisation (5 values).
        rotation_degrees:  Maximum rotation angle for training augmentation.
        brightness_range:  (min, max) multiplicative brightness jitter on RGB.
    """

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple[int, int] = (8, 8),
        mean: list[float] = DEFAULT_MEAN_5CH,
        std: list[float] = DEFAULT_STD_5CH,
        rotation_degrees: float = 15.0,
        brightness_range: tuple[float, float] = (0.8, 1.2),
    ) -> None:
        self.image_size = image_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.mean = mean
        self.std = std
        self.rotation_degrees = rotation_degrees
        self.brightness_range = brightness_range

    def __call__(self, image: Image.Image, train: bool = True) -> torch.Tensor:
        # 1. RGB conversion
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Resize
        img_np = np.array(image, dtype=np.uint8)
        img_np = cv2.resize(
            img_np,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )

        # 3. CLAHE on grayscale → (H, W) uint8
        clahe_ch = apply_clahe(img_np, self.clahe_clip_limit, self.clahe_tile_grid)

        # 4. Sobel on CLAHE output → (H, W) float32 [0, 1]
        sobel_ch = apply_sobel(clahe_ch)

        # 5. Multi-channel fusion → (H, W, 5) float32 [0, 1]
        fused = fuse_channels(img_np, clahe_ch, sobel_ch)

        # 6. Convert to tensor (5, H, W)
        tensor = torch.from_numpy(fused.transpose(2, 0, 1).copy())

        # 7. Augmentation — applied uniformly across all channels
        if train:
            tensor = self._augment(tensor)

        # 8. Channel-wise normalisation
        tensor = normalize_tensor(tensor, self.mean, self.std)

        return tensor

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply spatial and photometric augmentation preserving channel alignment."""
        # Random horizontal flip (all channels)
        if random.random() < 0.5:
            tensor = TF.hflip(tensor)

        # Random rotation ±rotation_degrees (all channels)
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        tensor = TF.rotate(tensor, angle)

        # Brightness jitter on RGB channels only (channels 0-2)
        factor = random.uniform(*self.brightness_range)
        tensor[:3] = torch.clamp(tensor[:3] * factor, 0.0, 1.0)

        return tensor
