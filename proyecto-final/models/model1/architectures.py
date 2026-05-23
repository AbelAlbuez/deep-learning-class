"""Arquitecturas: CNNBaseline (Modelo 1) y OkanNet (replicación con BatchNorm)."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import DROPOUT, NUM_CLASSES


def _init_weights(module: nn.Module) -> None:
    """Inicialización He: Kaiming normal fan_out para Conv2d, fan_in para Linear.

    Se usan modos distintos por capa:
      * Conv2d: fan_out preserva la varianza hacia delante.
      * Linear: fan_in evita preactivaciones explosivas en la FC grande
        (50176→128); fan_out daría std ≈ 0.125 ⇒ activaciones de orden 28,
        provocando divergencia (especialmente al combinar con BatchNorm).
    """
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


class CNNBaseline(nn.Module):
    """CNN Baseline: 3 bloques Conv-ReLU-MaxPool + cabezal FC.

    Input  : (B, 3, 224, 224)
    Output : (B, 4)
    Parámetros entrenables esperados: 6,446,756.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1: (B,3,224,224) -> (B,16,112,112)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloque 2: -> (B,32,56,56)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloque 3: -> (B,64,28,28)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class OkanNet(nn.Module):
    """OkanNet replicado: idéntica a CNNBaseline + BatchNorm2d tras cada Conv2d.

    Parámetros entrenables esperados: 6,446,980 (224 más que baseline).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloque 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloque 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Cuenta los parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str) -> nn.Module:
    """Factory de modelos.

    Args:
        name: 'baseline' o 'okannet'.

    Returns:
        Instancia de nn.Module.
    """
    name = name.lower()
    if name == "baseline":
        return CNNBaseline()
    if name == "okannet":
        return OkanNet()
    raise ValueError(f"Arquitectura desconocida: {name!r}. Usa 'baseline' o 'okannet'.")
