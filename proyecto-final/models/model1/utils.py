"""Utilidades: seed, device, early stopping, métricas y plots."""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def set_seed(seed: int = 42) -> None:
    """Fija semillas en random, numpy y torch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Devuelve el device disponible en orden: MPS -> CUDA -> CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EarlyStopping:
    """Early stopping monitoreando una métrica (modo 'max' o 'min')."""

    def __init__(self, patience: int = 7, mode: str = "max", min_delta: float = 0.0) -> None:
        if mode not in {"max", "min"}:
            raise ValueError(f"mode debe ser 'max' o 'min', recibido: {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Devuelve True si se debe detener el entrenamiento."""
        if self.best is None:
            self.best = score
            return False
        improved = (
            score > self.best + self.min_delta
            if self.mode == "max"
            else score < self.best - self.min_delta
        )
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> dict[str, float]:
    """Calcula accuracy, precision, recall, f1 (macro por defecto)."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def plot_training_curves(metrics_csv_path: str | Path, output_path: str | Path) -> None:
    """Genera figura 2x2 con curvas de loss, accuracy, F1 y learning rate."""
    df = pd.read_csv(metrics_csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Loss
    ax = axes[0, 0]
    ax.plot(df["epoch"], df["train_loss"], label="Train", marker="o", markersize=3)
    ax.plot(df["epoch"], df["val_loss"], label="Val", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(df["epoch"], df["train_acc"], label="Train", marker="o", markersize=3)
    ax.plot(df["epoch"], df["val_acc"], label="Val", marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)

    # F1
    ax = axes[1, 0]
    ax.plot(df["epoch"], df["val_f1"], label="Val F1 (macro)", marker="o",
            markersize=3, color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("F1-Score Validación")
    ax.legend()
    ax.grid(alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(df["epoch"], df["lr"], marker="o", markersize=3, color="purple")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning Rate")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Matriz de confusión normalizada por fila (porcentajes)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "%"},
        ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (% por clase real)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> str:
    """Guarda classification_report de sklearn como texto y lo devuelve."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(report, encoding="utf-8")
    return report
