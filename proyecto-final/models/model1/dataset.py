"""Dataset y DataLoaders para clasificación de tumores cerebrales (MRI)."""
from __future__ import annotations

import warnings

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight as _sk_class_weight
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .config import (
    BATCH_SIZE,
    CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    NUM_WORKERS,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_SPLIT,
)


def _ensure_rgb(img):
    """Convierte PIL a RGB si viene en grayscale u otro modo."""
    return img.convert("RGB") if img.mode != "RGB" else img


def get_transforms(train: bool) -> transforms.Compose:
    """Pipeline de preprocesamiento. Con augmentation si train=True."""
    base = [
        transforms.Lambda(_ensure_rgb),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]
    if train:
        base.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2),
            ]
        )
    base.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(base)


def _validate_paths() -> None:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de entrenamiento: {TRAIN_DIR}"
        )
    if not TEST_DIR.exists():
        raise FileNotFoundError(f"No se encontró el directorio de test: {TEST_DIR}")
    for cls in CLASSES:
        if not (TRAIN_DIR / cls).exists():
            raise FileNotFoundError(f"Falta la clase en Training: {cls}")
        if not (TEST_DIR / cls).exists():
            raise FileNotFoundError(f"Falta la clase en Testing: {cls}")


def _safe_num_workers(requested: int) -> int:
    """En MPS, num_workers>0 puede colgar; usamos 0 con warning."""
    if torch.backends.mps.is_available() and requested > 0:
        warnings.warn(
            "MPS detectado: usando num_workers=0 para evitar bloqueos en DataLoader.",
            stacklevel=2,
        )
        return 0
    return requested


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Crea train/val/test DataLoaders. Split 80/20 sobre Training; Testing como holdout.

    val_loader y test_loader usan transforms SIN augmentation.
    """
    _validate_paths()
    workers = _safe_num_workers(num_workers)

    # Dataset completo con augmentation (para train)
    train_full = ImageFolder(str(TRAIN_DIR), transform=get_transforms(train=True))
    # Mismo dataset pero sin augmentation (para val: indexa los mismos archivos)
    val_full = ImageFolder(str(TRAIN_DIR), transform=get_transforms(train=False))

    # Verificar consistencia de clases
    if train_full.classes != CLASSES:
        raise ValueError(
            f"Clases descubiertas {train_full.classes} no coinciden con CLASSES esperadas {CLASSES}."
        )

    n_total = len(train_full)
    n_val = int(round(n_total * val_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        train_full, [n_train, n_val], generator=generator
    )
    # val_subset debe apuntar al dataset sin augmentation
    val_subset = Subset(val_full, val_subset.indices)

    test_dataset = ImageFolder(str(TEST_DIR), transform=get_transforms(train=False))

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )

    return train_loader, val_loader, test_loader


def compute_class_weights(train_loader: DataLoader) -> torch.Tensor:
    """Calcula pesos balanceados a partir de las etiquetas del train_loader."""
    dataset = train_loader.dataset
    # Resuelve labels independientemente de Subset
    if isinstance(dataset, Subset):
        base = dataset.dataset
        targets = [base.targets[i] for i in dataset.indices]
    else:
        targets = list(getattr(dataset, "targets", []))

    if not targets:
        raise ValueError("No se pudieron extraer las etiquetas del dataset de entrenamiento.")

    targets_np = np.asarray(targets)
    classes = np.unique(targets_np)
    weights = _sk_class_weight(class_weight="balanced", classes=classes, y=targets_np)
    return torch.tensor(weights, dtype=torch.float32)
