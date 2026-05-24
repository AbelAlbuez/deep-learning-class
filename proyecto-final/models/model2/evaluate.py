"""Evaluación sobre el conjunto de Testing del Modelo 2.

Uso:
    python models/model2/evaluate.py --arch deep
    python models/model2/evaluate.py --arch deep_gap
"""
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "models.model2"

import argparse
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

from .architectures import get_model
from .config import CHECKPOINTS_DIR, CLASSES, FIGURES_DIR, RESULTS_DIR
from .dataset import get_dataloaders
from .utils import (
    compute_metrics,
    get_device,
    plot_confusion_matrix,
    save_classification_report,
)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evalúa modelo entrenado sobre Testing/.")
    p.add_argument("--arch", required=True, choices=["deep", "deep_gap"])
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in tqdm(loader, desc="test"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def main() -> None:
    args = parse_args()
    device = get_device()
    _log(f"Device: {device}")

    ckpt_path = CHECKPOINTS_DIR / f"{args.arch}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No existe el checkpoint {ckpt_path}. Entrena primero con train.py."
        )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _log(f"Cargando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = get_model(args.arch).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    _log("Preparando test_loader...")
    _, _, test_loader = get_dataloaders()
    _log(f"Test batches: {len(test_loader)}")

    y_true, y_pred = run_inference(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred, average="macro")
    _log(
        f"Test → accuracy={metrics['accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )

    report_path = RESULTS_DIR / f"{args.arch}_test_report.txt"
    report = save_classification_report(y_true, y_pred, CLASSES, report_path)
    _log(f"Classification report → {report_path}")

    cm_path = FIGURES_DIR / f"confusion_{args.arch}.png"
    plot_confusion_matrix(y_true, y_pred, CLASSES, cm_path)
    _log(f"Matriz de confusión → {cm_path}")

    print()
    print("=" * 70)
    print(f"Resultados Test [{args.arch}]")
    print("=" * 70)
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Precision (macro)  : {metrics['precision']:.4f}")
    print(f"Recall    (macro)  : {metrics['recall']:.4f}")
    print(f"F1-Score  (macro)  : {metrics['f1']:.4f}")
    print()
    print(report)


if __name__ == "__main__":
    main()
