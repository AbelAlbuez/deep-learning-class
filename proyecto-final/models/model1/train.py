"""Entrenamiento del Modelo 1 (CNNBaseline u OkanNet).

Uso:
    python models/model1/train.py --arch baseline
    python models/model1/train.py --arch okannet
"""
from __future__ import annotations

# Permitir ejecución directa: `python models/model1/train.py ...`
if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "models.model1"

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architectures import count_parameters, get_model
from .config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EARLY_STOPPING_PATIENCE,
    FIGURES_DIR,
    LEARNING_RATE,
    MIN_LR,
    NUM_EPOCHS,
    RESULTS_DIR,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SEED,
)
from .dataset import compute_class_weights, get_dataloaders
from .utils import (
    EarlyStopping,
    compute_metrics,
    get_device,
    plot_training_curves,
    set_seed,
)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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

        batch_n = labels.size(0)
        total_loss += loss.item() * batch_n
        total_samples += batch_n
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

        batch_n = labels.size(0)
        total_loss += loss.item() * batch_n
        total_samples += batch_n

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())

    metrics = compute_metrics(np.asarray(y_true), np.asarray(y_pred), average="macro")
    return total_loss / total_samples, metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena CNNBaseline u OkanNet.")
    p.add_argument("--arch", required=True, choices=["baseline", "okannet"])
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(SEED)
    device = get_device()
    _log(f"Device: {device}")
    _log(f"Arquitectura: {args.arch} | epochs={args.epochs} | lr={args.lr} | bs={args.batch_size}")

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _log("Cargando dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(batch_size=args.batch_size)
    _log(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    class_weights = compute_class_weights(train_loader).to(device)
    _log(f"Class weights: {class_weights.cpu().numpy().round(4).tolist()}")

    model = get_model(args.arch).to(device)
    n_params = count_parameters(model)
    _log(f"Parámetros entrenables: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode="max")

    csv_path = RESULTS_DIR / f"{args.arch}_metrics.csv"
    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_precision",
        "val_recall",
        "val_f1",
        "lr",
    ]

    best_f1 = -1.0
    best_epoch = -1
    ckpt_path = CHECKPOINTS_DIR / f"{args.arch}_best.pth"
    rows: list[dict] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        _log(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_metrics["accuracy"], 6),
            "val_precision": round(val_metrics["precision"], 6),
            "val_recall": round(val_metrics["recall"], 6),
            "val_f1": round(val_metrics["f1"], 6),
            "lr": current_lr,
        }
        rows.append(row)

        _log(
            f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} | lr={current_lr:.2e}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "arch": args.arch,
                    "val_f1": best_f1,
                },
                ckpt_path,
            )
            _log(f"  ✓ Mejor F1 val: {best_f1:.4f} → checkpoint guardado")

        scheduler.step(val_metrics["f1"])

        if early_stopper(val_metrics["f1"]):
            _log(f"Early stopping en epoch {epoch} (best epoch={best_epoch}).")
            break

    # Guardar CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _log(f"Métricas guardadas en: {csv_path}")

    # Curvas
    curves_path = FIGURES_DIR / f"curvas_{args.arch}.png"
    plot_training_curves(csv_path, curves_path)
    _log(f"Curvas guardadas en: {curves_path}")

    elapsed = time.time() - t0
    _log("=" * 60)
    _log(
        f"Resumen [{args.arch}] | mejor epoch={best_epoch} | "
        f"mejor F1 val={best_f1:.4f} | tiempo={elapsed/60:.2f} min"
    )

    # Tiempo total en archivo para uso por compare.py
    Path(RESULTS_DIR / f"{args.arch}_train_time.txt").write_text(
        f"{elapsed:.2f}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
