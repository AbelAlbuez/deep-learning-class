"""Arquitecturas: DeepCNN_BN (Modelo 2) y DeepCNN_BN_GAP (variante con GAP).

Ambas comparten el mismo backbone convolucional VGG-style de 5 bloques
con BatchNorm en cada Conv2d y Dropout2d después de cada MaxPool. La
diferencia entre ambas está en el cabezal:

  * DeepCNN_BN     : AdaptiveAvgPool2d(2,2) + cabezal FC con Dropout.
  * DeepCNN_BN_GAP : Global Average Pooling + Linear(512, 4).

La hipótesis a contrastar es si el cabezal FC aporta capacidad
discriminativa real sobre el GAP, una vez que el backbone ya tiene
suficiente profundidad (10 capas conv frente a las 3 del baseline).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import DROPOUT2D_RATES, DROPOUT_FC, NUM_CLASSES


def _init_weights(module: nn.Module) -> None:
    """Inicialización He: Kaiming normal fan_out (Conv2d) / fan_in (Linear).

    Mismo criterio que el Modelo 1: fan_in en Linear evita preactivaciones
    explosivas al combinarse con BatchNorm.
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _conv_block(
    in_channels: int,
    out_channels: int,
    dropout2d: float,
    pool: bool = True,
) -> nn.Sequential:
    """Bloque VGG-style: 2x(Conv-BN-ReLU) + [MaxPool] + Dropout2d.

    Si pool=False, no se reduce la dimensión espacial. Esto se usa en el
    último bloque del backbone para mantener salida (14, 14) y permitir
    que AdaptiveAvgPool2d((2, 2)) sea divisible (requisito de MPS).
    """
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    layers.append(nn.Dropout2d(p=dropout2d))
    return nn.Sequential(*layers)


def _build_backbone() -> nn.Sequential:
    """Backbone compartido por ambas variantes: 5 bloques (10 capas conv).

    Los bloques 1-4 reducen espacialmente con MaxPool; el bloque 5 NO
    hace pool para que la salida sea 14x14 (en vez de 7x7). El motivo
    es que AdaptiveAvgPool2d((2, 2)) exige que el input sea divisible
    por el output bajo MPS, y 14/2=7 funciona mientras que 7/2 falla.

    Salida: (B, 512, 14, 14) para entrada (B, 3, 224, 224).
    """
    r1, r2, r3, r4, r5 = DROPOUT2D_RATES
    return nn.Sequential(
        _conv_block(3,   32,  r1),                  # (B,  32, 112, 112)
        _conv_block(32,  64,  r2),                  # (B,  64,  56,  56)
        _conv_block(64,  128, r3),                  # (B, 128,  28,  28)
        _conv_block(128, 256, r4),                  # (B, 256,  14,  14)
        _conv_block(256, 512, r5, pool=False),      # (B, 512,  14,  14)
    )


class DeepCNN_BN(nn.Module):
    """Modelo 2 principal: 10 capas Conv + BN + Dropout2d + cabezal FC.

    Input  : (B, 3, 224, 224)
    Output : (B, 4)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_fc: float = DROPOUT_FC,
    ) -> None:
        super().__init__()
        self.features = _build_backbone()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # (B, 512, 2, 2) -> 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(256, num_classes),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


class DeepCNN_BN_GAP(nn.Module):
    """Variante con Global Average Pooling en lugar de cabezal FC denso.

    Reduce drásticamente los parámetros del cabezal (~525K -> ~2K) sin
    tocar el backbone convolucional, permitiendo aislar el efecto del
    cabezal sobre la generalización.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_fc: float = DROPOUT_FC,
    ) -> None:
        super().__init__()
        self.features = _build_backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # GAP -> (B, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_fc),
            nn.Linear(512, num_classes),
        )
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Cuenta los parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str) -> nn.Module:
    """Factory de modelos.

    Args:
        name: 'deep' o 'deep_gap'.

    Returns:
        Instancia de nn.Module.
    """
    name = name.lower()
    if name == "deep":
        return DeepCNN_BN()
    if name == "deep_gap":
        return DeepCNN_BN_GAP()
    raise ValueError(f"Arquitectura desconocida: {name!r}. Usa 'deep' o 'deep_gap'.")
