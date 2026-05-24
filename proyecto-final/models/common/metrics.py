"""Classification metrics helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1.

    Args:
        y_true:  Ground-truth class indices.
        y_pred:  Predicted class indices.
        average: Averaging strategy passed to sklearn (default: ``'macro'``).

    Returns:
        Dictionary with keys ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
