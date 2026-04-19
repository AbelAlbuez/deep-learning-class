"""Deep-dive analysis of best and worst grid runs.

Reads outputs/results.csv, loads the best and worst models by test_acc,
re-evaluates them on the test set and writes a per-class report to
outputs/analysis_report.txt.

Run with:
    source venv/bin/activate && python analyze_results.py
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_DIR = "/Users/abelalbuez/Documents/Maestria/Tercer Semestre/Aprendizaje Profundo/deep-learning-class/traffic-sign-cnn-classifier"
TEST_DIR = os.path.join(PROJECT_DIR, "datasets/test_dataset/test")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "results.csv")
REPORT_PATH = os.path.join(OUTPUTS_DIR, "analysis_report.txt")

IMG_SIZE = 32
F1_THRESHOLD = 0.85
SEED = 42


def tag(row):
    return f"{row['model']}_e{int(row['epochs'])}_bs{int(row['batch_size'])}"


def build_test_flow():
    gen = ImageDataGenerator(rescale=1.0 / 255)
    return gen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )


def format_confusion(cm, class_names):
    header_width = max(len(c) for c in class_names) + 2
    cell_width = max(len(str(cm.max())), max(len(c) for c in class_names)) + 2
    header = " " * header_width + "".join(c.rjust(cell_width) for c in class_names)
    lines = [header]
    for cls, row in zip(class_names, cm):
        lines.append(cls.ljust(header_width) + "".join(str(v).rjust(cell_width) for v in row))
    return "\n".join(lines)


def analyze_run(row, test_flow, class_names):
    run_tag = tag(row)
    model_path = os.path.join(MODELS_DIR, f"{run_tag}.keras")

    if not os.path.exists(model_path):
        return f"\nModelo {run_tag} no encontrado en {model_path}\n"

    print(f"Cargando {run_tag} desde {model_path}")
    model = tf.keras.models.load_model(model_path)

    test_flow.reset()
    probs = model.predict(test_flow, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = test_flow.classes

    report_text = classification_report(
        y_true, y_pred, target_names=class_names,
        zero_division=0, digits=4,
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names,
        zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)

    struggling = [
        (cls, report_dict[cls]["f1-score"], int(report_dict[cls]["support"]))
        for cls in class_names
        if report_dict[cls]["f1-score"] < F1_THRESHOLD
    ]
    struggling.sort(key=lambda x: x[1])

    section = []
    section.append("=" * 78)
    section.append(f"MODELO: {run_tag}")
    section.append("=" * 78)
    section.append(f"test_acc registrado en CSV: {row['test_acc']}")
    section.append(f"val_acc registrado en CSV:  {row['val_acc']}")
    section.append(f"f1_score registrado en CSV: {row['f1_score']}")
    section.append(f"training_time_sec:          {row['training_time_sec']}")
    section.append("")
    section.append("CLASSIFICATION REPORT (re-evaluado en test)")
    section.append("-" * 78)
    section.append(report_text)
    section.append("CONFUSION MATRIX (filas=real, columnas=predicho)")
    section.append("-" * 78)
    section.append(format_confusion(cm, class_names))
    section.append("")
    section.append(f"CLASES EN DIFICULTAD (f1 < {F1_THRESHOLD})")
    section.append("-" * 78)
    if not struggling:
        section.append("Ninguna — todas las clases superan el umbral.")
    else:
        for cls, f1, support in struggling:
            section.append(f"  {cls:<12}  f1={f1:.4f}  support={support}")
    section.append("")
    return "\n".join(section)


def main():
    if not os.path.exists(RESULTS_CSV):
        print(f"No existe {RESULTS_CSV}. Ejecuta main.py primero.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        print("results.csv está vacío.")
        sys.exit(1)

    df_sorted = df.sort_values("test_acc", ascending=False).reset_index(drop=True)
    best = df_sorted.iloc[0]
    worst = df_sorted.iloc[-1]

    print(f"Mejor por test_acc:  {tag(best)}  (test_acc={best['test_acc']})")
    print(f"Peor por test_acc:   {tag(worst)}  (test_acc={worst['test_acc']})")

    test_flow = build_test_flow()
    inv_class_indices = {v: k for k, v in test_flow.class_indices.items()}
    class_names = [inv_class_indices[i] for i in range(len(inv_class_indices))]

    header = [
        "ANÁLISIS DE RESULTADOS — TALLER 2",
        "=" * 78,
        f"Total de experimentos en results.csv: {len(df)}",
        f"Umbral de dificultad (f1): {F1_THRESHOLD}",
        "",
    ]
    best_section = analyze_run(best, test_flow, class_names)
    worst_section = analyze_run(worst, test_flow, class_names)

    comparison = [
        "=" * 78,
        "COMPARACIÓN RÁPIDA",
        "=" * 78,
        f"  Mejor:  {tag(best):<30}  test_acc={best['test_acc']}  f1={best['f1_score']}",
        f"  Peor:   {tag(worst):<30}  test_acc={worst['test_acc']}  f1={worst['f1_score']}",
        f"  Gap en test_acc: {best['test_acc'] - worst['test_acc']:.4f}",
        "",
    ]

    full = "\n".join(header) + best_section + worst_section + "\n".join(comparison)

    print("\n" + full)
    with open(REPORT_PATH, "w") as f:
        f.write(full)
    print(f"\nReporte guardado en: {REPORT_PATH}")


if __name__ == "__main__":
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    main()
