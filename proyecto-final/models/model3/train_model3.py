"""Training script for Model3 (MRIResNet).

Usage (from proyecto-final/):
    python -m models.model3.train_model3
    python -m models.model3.train_model3 --epochs 50 --lr 1e-4 --batch_size 32
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "models.model3"

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    CLASSES,
    EARLY_STOPPING_PATIENCE,
    FIGURES_DIR,
    LEARNING_RATE,
    MIN_LR,
    NUM_EPOCHS,
    RESULTS_DIR,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    WEIGHT_DECAY,
)
from .model3 import MRIResNet, build_model, count_parameters

# Shared infrastructure from common
from ..common.dataset import compute_class_weights, get_dataloaders
from ..common.metrics import compute_metrics
from ..common.preprocessing import AdvancedMRIPreprocessing
from ..common.utils import EarlyStopping, get_device, set_seed
from ..common.visualization import plot_training_curves


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        n = labels.size(0)
        total_loss += loss.item() * n
        total_samples += n
        correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / total_samples, correct / total_samples


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        n = labels.size(0)
        total_loss += loss.item() * n
        total_samples += n
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    metrics = compute_metrics(np.asarray(y_true), np.asarray(y_pred), average="macro")
    return total_loss / total_samples, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MRIResNet (Model3).")
    p.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    p.add_argument("--lr",          type=float, default=LEARNING_RATE)
    p.add_argument("--weight_decay",type=float, default=WEIGHT_DECAY)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    set_seed(SEED)
    device = get_device()
    _log(f"Device: {device}")
    _log(f"epochs={args.epochs} | lr={args.lr} | wd={args.weight_decay} | bs={args.batch_size}")

    for d in (CHECKPOINTS_DIR, FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # -- Data ----------------------------------------------------------------
    _log("Building dataloaders...")
    strategy = AdvancedMRIPreprocessing()
    train_loader, val_loader, _ = get_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        preprocessing_strategy=strategy,
        batch_size=args.batch_size,
        classes=CLASSES,
        seed=SEED,
    )
    _log(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    class_weights = compute_class_weights(train_loader).to(device)
    _log(f"Class weights: {class_weights.cpu().numpy().round(4).tolist()}")

    # -- Model ---------------------------------------------------------------
    model = build_model().to(device)
    n_params = count_parameters(model)
    _log(f"Trainable parameters: {n_params:,}")

    # -- Optimisation --------------------------------------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="max")

    # -- Logging setup -------------------------------------------------------
    csv_path = RESULTS_DIR / "mriresnet_metrics.csv"
    fieldnames = [
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_precision", "val_recall", "val_f1", "lr",
    ]
    best_f1 = -1.0
    best_epoch = -1
    ckpt_path = CHECKPOINTS_DIR / "mriresnet_best.pth"
    rows: list[dict] = []
    t0 = time.time()

    # -- Training loop -------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        _log(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch":         epoch,
            "train_loss":    round(train_loss, 6),
            "train_acc":     round(train_acc, 6),
            "val_loss":      round(val_loss, 6),
            "val_acc":       round(val_metrics["accuracy"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall":    round(val_metrics["recall"], 6),
            "val_f1":        round(val_metrics["f1"], 6),
            "lr":            current_lr,
        }
        rows.append(row)

        _log(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['f1']:.4f} | lr={current_lr:.2e}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_f1": best_f1,
                },
                ckpt_path,
            )
            _log(f"  ✓ New best val F1: {best_f1:.4f} → checkpoint saved")

        scheduler.step(val_metrics["f1"])

        if early_stopper(val_metrics["f1"]):
            _log(f"Early stopping at epoch {epoch} (best epoch={best_epoch}).")
            break

    # -- Save artefacts ------------------------------------------------------
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _log(f"Metrics saved: {csv_path}")

    curves_path = FIGURES_DIR / "curves_mriresnet.png"
    plot_training_curves(csv_path, curves_path)
    _log(f"Training curves saved: {curves_path}")

    elapsed = time.time() - t0
    Path(RESULTS_DIR / "mriresnet_train_time.txt").write_text(
        f"{elapsed:.2f}\n", encoding="utf-8"
    )

    _log("=" * 60)
    _log(
        f"Summary | best_epoch={best_epoch} | best_val_f1={best_f1:.4f} "
        f"| elapsed={elapsed / 60:.2f} min"
    )


if __name__ == "__main__":
    main()
