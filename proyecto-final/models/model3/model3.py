"""Model3: MRIResNet — Residual CNN with Squeeze-and-Excitation attention.

Architecture overview:
    Input (5, 224, 224)          ← R, G, B, CLAHE, Sobel
    → Initial Conv Block         → (32, 112, 112)
    → Residual+SE Stage 1        → (32,  56,  56)
    → Residual+SE Stage 2        → (64,  28,  28)
    → Residual+SE Stage 3        → (128, 14,  14)
    → Residual+SE Stage 4        → (256, 14,  14)
    → Global Average Pooling     → (256,)
    → Classifier Head            → (4,)

Key design decisions:
- in_channels=5 to consume the multimodal preprocessing output.
- ResidualBlock preserves gradient flow with identity / projection shortcuts.
- SEBlock adaptively reweights channels after each residual stage.
- GAP replaces the large Flatten→Linear used in model1, reducing parameters.
- Kaiming initialization throughout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import DROPOUT, IN_CHANNELS, NUM_CLASSES, SE_REDUCTION


# ---------------------------------------------------------------------------
# Weight initialisation (same strategy as model1)
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Squeeze: GlobalAvgPool → one descriptor per channel.
    Excitation: FC → ReLU → FC → Sigmoid → per-channel weights in [0, 1].
    Reweight: element-wise multiplication of weights with input feature map.

    Args:
        channels:  Number of input (and output) channels.
        reduction: Bottleneck reduction ratio. Hidden units = channels // reduction.
    """

    def __init__(self, channels: int, reduction: int = SE_REDUCTION) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        s = self.squeeze(x)                        # (B, C, 1, 1)
        s = self.excitation(s)                     # (B, C)
        s = s.view(s.size(0), -1, 1, 1)           # (B, C, 1, 1)
        return x * s                               # channel-wise reweighting


class ResidualBlock(nn.Module):
    """Two-layer residual block: Conv-BN-ReLU-Conv-BN + skip.

    When in_channels != out_channels a 1×1 projection is applied to the
    shortcut path so dimensions match before addition.

    Structure:
        x → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → (+skip) → ReLU
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MRIResNet(nn.Module):
    """Residual CNN with SE attention for 5-channel MRI classification.

    Input  : (B, 5, 224, 224)
    Output : (B, num_classes)

    Args:
        in_channels:  Number of input channels (5 for the advanced pipeline).
        num_classes:  Number of output classes.
        dropout:      Dropout probability in the classifier head.
        se_reduction: SE bottleneck reduction ratio.
    """

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        se_reduction: int = SE_REDUCTION,
    ) -> None:
        super().__init__()

        # -- Initial conv block -------------------------------------------------
        # Accepts 5-channel multimodal input and maps to 32 feature maps.
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 → 112
        )

        # -- Residual + SE stages -----------------------------------------------
        self.stage1 = self._make_stage(32,  32,  n_blocks=2, reduction=se_reduction)  # → 56
        self.stage2 = self._make_stage(32,  64,  n_blocks=2, reduction=se_reduction)  # → 28
        self.stage3 = self._make_stage(64,  128, n_blocks=2, reduction=se_reduction)  # → 14
        self.stage4 = nn.Sequential(                                                   # → 14
            ResidualBlock(128, 256),
            SEBlock(256, se_reduction),
        )

        # -- Classifier head ----------------------------------------------------
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self.apply(_init_weights)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        reduction: int,
    ) -> nn.Sequential:
        """Build a stage: n residual+SE pairs followed by MaxPool."""
        layers: list[nn.Module] = [
            ResidualBlock(in_channels, out_channels),
            SEBlock(out_channels, reduction),
        ]
        for _ in range(n_blocks - 1):
            layers.extend([
                ResidualBlock(out_channels, out_channels),
                SEBlock(out_channels, reduction),
            ])
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)    # (B, 32, 112, 112)
        x = self.stage1(x)     # (B,  32,  56,  56)
        x = self.stage2(x)     # (B,  64,  28,  28)
        x = self.stage3(x)     # (B, 128,  14,  14)
        x = self.stage4(x)     # (B, 256,  14,  14)
        x = self.gap(x)        # (B, 256,   1,   1)
        return self.classifier(x)  # (B, num_classes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    in_channels: int = IN_CHANNELS,
    num_classes: int = NUM_CLASSES,
    dropout: float = DROPOUT,
    se_reduction: int = SE_REDUCTION,
) -> MRIResNet:
    return MRIResNet(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
        se_reduction=se_reduction,
    )
