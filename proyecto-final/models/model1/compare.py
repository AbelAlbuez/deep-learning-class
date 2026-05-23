"""Comparación CNNBaseline vs OkanNet: figura combinada + tabla resumen."""
from __future__ import annotations

if __name__ == "__main__" and __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    __package__ = "models.model1"

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .architectures import CNNBaseline, OkanNet, count_parameters
from .config import FIGURES_DIR, RESULTS_DIR

ARCHS = ["baseline", "okannet"]
PRETTY = {"baseline": "CNNBaseline", "okannet": "OkanNet"}


def _read_metrics(arch: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{arch}_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta CSV de métricas: {path}")
    return pd.read_csv(path)


def _read_test_metrics(arch: str) -> dict[str, float]:
    """Extrae accuracy y macro avg (precision/recall/f1) del classification_report."""
    path = RESULTS_DIR / f"{arch}_test_report.txt"
    if not path.exists():
        raise FileNotFoundError(f"Falta test report: {path}")
    text = path.read_text(encoding="utf-8")

    acc_match = re.search(r"accuracy\s+([\d.]+)", text)
    macro_match = re.search(
        r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text
    )
    if not acc_match or not macro_match:
        raise ValueError(f"No se pudo parsear el reporte: {path}")

    return {
        "accuracy": float(acc_match.group(1)),
        "precision": float(macro_match.group(1)),
        "recall": float(macro_match.group(2)),
        "f1": float(macro_match.group(3)),
    }


def _read_train_time(arch: str) -> float:
    path = RESULTS_DIR / f"{arch}_train_time.txt"
    if not path.exists():
        return float("nan")
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return float("nan")


def _format_time(seconds: float) -> str:
    if seconds != seconds:  # NaN
        return "N/A"
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:>3d}m {s:02d}s"


def _delta_pct(a: float, b: float) -> str:
    """Δ = (okannet - baseline) en puntos porcentuales."""
    diff = (b - a) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2f}%"


def make_comparison_figure(metrics: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        df = metrics[arch]
        ax.plot(df["epoch"], df["val_f1"], marker="o", markersize=4, color="green",
                label="Val F1 (macro)")
        ax.plot(df["epoch"], df["val_acc"], marker="s", markersize=4, color="steelblue",
                label="Val Accuracy", alpha=0.7)
        ax.set_title(PRETTY[arch])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    fig.suptitle("Comparación CNNBaseline vs OkanNet — validación", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    test_metrics: dict[str, dict[str, float]],
    val_metrics: dict[str, pd.DataFrame],
    params: dict[str, int],
    times: dict[str, float],
) -> str:
    b = test_metrics["baseline"]
    o = test_metrics["okannet"]
    best_epoch_b = int(val_metrics["baseline"].loc[val_metrics["baseline"]["val_f1"].idxmax(), "epoch"])
    best_epoch_o = int(val_metrics["okannet"].loc[val_metrics["okannet"]["val_f1"].idxmax(), "epoch"])

    lines = [
        "Comparación CNNBaseline vs OkanNet (Modelo 1)",
        "==============================================",
        "",
        f"{'Métrica':<22}| {'CNNBaseline':^11} | {'OkanNet':^9} | Δ",
        f"{'-'*22}+{'-'*13}+{'-'*11}+{'-'*10}",
        f"{'Accuracy':<22}|   {b['accuracy']:.4f}    |  {o['accuracy']:.4f}  | {_delta_pct(b['accuracy'], o['accuracy'])}",
        f"{'Precision (macro)':<22}|   {b['precision']:.4f}    |  {o['precision']:.4f}  | {_delta_pct(b['precision'], o['precision'])}",
        f"{'Recall (macro)':<22}|   {b['recall']:.4f}    |  {o['recall']:.4f}  | {_delta_pct(b['recall'], o['recall'])}",
        f"{'F1-Score (macro)':<22}|   {b['f1']:.4f}    |  {o['f1']:.4f}  | {_delta_pct(b['f1'], o['f1'])}",
        f"{'Parámetros':<22}|  {params['baseline']:>9,} | {params['okannet']:>9,}| +{params['okannet']-params['baseline']}",
        f"{'Mejor epoch':<22}|     {best_epoch_b:>2}      |    {best_epoch_o:>2}    |",
        f"{'Tiempo entrenamiento':<22}|  {_format_time(times['baseline'])}  | {_format_time(times['okannet'])} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    val_metrics = {arch: _read_metrics(arch) for arch in ARCHS}
    test_metrics = {arch: _read_test_metrics(arch) for arch in ARCHS}
    times = {arch: _read_train_time(arch) for arch in ARCHS}
    params = {
        "baseline": count_parameters(CNNBaseline()),
        "okannet": count_parameters(OkanNet()),
    }

    fig_path = FIGURES_DIR / "comparacion_baseline_okannet.png"
    make_comparison_figure(val_metrics, fig_path)
    print(f"Figura comparativa → {fig_path}")

    summary = build_summary(test_metrics, val_metrics, params, times)
    summary_path = RESULTS_DIR / "comparison_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Tabla comparativa → {summary_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
