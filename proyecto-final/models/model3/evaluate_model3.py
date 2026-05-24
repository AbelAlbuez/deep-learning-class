"""Evaluation script for Model3 (MRIResNet) on the held-out test set.

Usage (from proyecto-final/):
    python -m models.model3.evaluate_model3
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "models.model3"

from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from .config import (
    CHECKPOINTS_DIR,
    CLASSES,
    FIGURES_DIR,
    RESULTS_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from .model3 import build_model

from ..common.dataset import get_dataloaders
from ..common.metrics import compute_metrics
from ..common.preprocessing import AdvancedMRIPreprocessing
from ..common.utils import get_device
from ..common.visualization import plot_confusion_matrix, save_classification_report


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in tqdm(loader, desc="test"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def main() -> None:
    device = get_device()
    _log(f"Device: {device}")

    ckpt_path = CHECKPOINTS_DIR / "mriresnet_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train_model3.py first."
        )

    for d in (FIGURES_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    _log(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    _log(f"Checkpoint from epoch {ckpt['epoch']} (val F1={ckpt['val_f1']:.4f})")

    _log("Building test loader...")
    strategy = AdvancedMRIPreprocessing()
    _, _, test_loader = get_dataloaders(
        train_dir=TRAIN_DIR,
        test_dir=TEST_DIR,
        preprocessing_strategy=strategy,
        classes=CLASSES,
    )
    _log(f"Test batches: {len(test_loader)}")

    y_true, y_pred = run_inference(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, average="macro")
    _log(
        f"Test → accuracy={metrics['accuracy']:.4f}  "
        f"precision={metrics['precision']:.4f}  "
        f"recall={metrics['recall']:.4f}  "
        f"f1={metrics['f1']:.4f}"
    )

    report_path = RESULTS_DIR / "mriresnet_test_report.txt"
    report = save_classification_report(y_true, y_pred, CLASSES, report_path)
    _log(f"Classification report → {report_path}")

    cm_path = FIGURES_DIR / "confusion_mriresnet.png"
    plot_confusion_matrix(y_true, y_pred, CLASSES, cm_path)
    _log(f"Confusion matrix → {cm_path}")

    print()
    print("=" * 70)
    print("Test Results — MRIResNet (Model3)")
    print("=" * 70)
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Precision (macro)  : {metrics['precision']:.4f}")
    print(f"Recall    (macro)  : {metrics['recall']:.4f}")
    print(f"F1-Score  (macro)  : {metrics['f1']:.4f}")
    print()
    print(report)


if __name__ == "__main__":
    main()
