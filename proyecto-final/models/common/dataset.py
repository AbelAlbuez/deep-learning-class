"""Generic MRI dataset with injectable preprocessing strategy.

Usage:
    from common.dataset import get_dataloaders
    from common.preprocessing import AdvancedMRIPreprocessing

    train_loader, val_loader, test_loader = get_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        preprocessing_strategy=AdvancedMRIPreprocessing(),
    )
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight as _sk_class_weight
from torch.utils.data import DataLoader, Dataset, Subset, random_split

CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

BATCH_SIZE = 32
VAL_SPLIT = 0.2
NUM_WORKERS = 4
SEED = 42


class MRIDataset(Dataset):
    """MRI classification dataset with an injectable preprocessing strategy.

    The strategy is called as ``strategy(pil_image, train=self.training)``
    and must return a torch.Tensor.  Use BasePreprocessing for model1-style
    pipelines and AdvancedMRIPreprocessing for the 5-channel model3 pipeline.

    Args:
        root:                   Directory containing one sub-folder per class.
        preprocessing_strategy: Callable(PIL.Image, train: bool) → Tensor.
        training:               If True, strategy receives train=True (augment).
        classes:                Ordered list of class names (must match subdir names).
    """

    def __init__(
        self,
        root: str | Path,
        preprocessing_strategy,
        training: bool = True,
        classes: list[str] = CLASSES,
    ) -> None:
        self.root = Path(root)
        self.strategy = preprocessing_strategy
        self.training = training
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls in classes:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {cls_dir}")
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in VALID_EXTS:
                    self.samples.append((f, self.class_to_idx[cls]))

    @property
    def targets(self) -> list[int]:
        return [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        tensor = self.strategy(image, train=self.training)
        return tensor, label


def _safe_num_workers(requested: int) -> int:
    if torch.backends.mps.is_available() and requested > 0:
        warnings.warn(
            "MPS detected: using num_workers=0 to avoid DataLoader deadlocks.",
            stacklevel=2,
        )
        return 0
    return requested


def get_dataloaders(
    train_dir: str | Path,
    test_dir: str | Path,
    preprocessing_strategy,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
    classes: list[str] = CLASSES,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders with an injectable strategy.

    The training split receives ``train=True`` (augmentation enabled).
    The validation and test splits receive ``train=False`` (deterministic).

    An 80/20 random split (reproducible via seed) is applied to train_dir.
    test_dir is used as the held-out test set.
    """
    workers = _safe_num_workers(num_workers)

    # Two dataset objects over the same files: one augmented, one not.
    train_full = MRIDataset(train_dir, preprocessing_strategy, training=True, classes=classes)
    val_full = MRIDataset(train_dir, preprocessing_strategy, training=False, classes=classes)

    n_total = len(train_full)
    n_val = int(round(n_total * val_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_split, val_split_tmp = random_split(train_full, [n_train, n_val], generator=generator)
    # Re-index val_split_tmp against the non-augmented dataset.
    val_subset = Subset(val_full, val_split_tmp.indices)

    test_dataset = MRIDataset(test_dir, preprocessing_strategy, training=False, classes=classes)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_split,
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
    """Compute balanced class weights from a train DataLoader."""
    dataset = train_loader.dataset
    if isinstance(dataset, Subset):
        base = dataset.dataset
        targets = [base.targets[i] for i in dataset.indices]
    else:
        targets = list(getattr(dataset, "targets", []))

    if not targets:
        raise ValueError("No labels found in the training dataset.")

    targets_np = np.asarray(targets)
    classes = np.unique(targets_np)
    weights = _sk_class_weight(class_weight="balanced", classes=classes, y=targets_np)
    return torch.tensor(weights, dtype=torch.float32)
