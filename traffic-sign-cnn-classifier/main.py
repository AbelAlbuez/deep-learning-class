"""Traffic sign CNN classifier — Taller 2.

Grid-based experimentation with resume support.

Run end-to-end with:
    python main.py
"""

import os
import csv
import time
import random
import shutil
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR = "/Users/abelalbuez/Documents/Maestria/Tercer Semestre/Aprendizaje Profundo/deep-learning-class/traffic-sign-cnn-classifier"
TRAIN_DIR = os.path.join(PROJECT_DIR, "datasets/train_dataset/train")
TEST_DIR = os.path.join(PROJECT_DIR, "datasets/test_dataset/test")

OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
EXPLORATION_DIR = os.path.join(OUTPUTS_DIR, "exploration")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
RESULTS_CSV = os.path.join(OUTPUTS_DIR, "results.csv")
SUMMARY_PATH = os.path.join(OUTPUTS_DIR, "summary.txt")

IMG_SIZE = 32
CHANNELS = 3
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, CHANNELS)

EXPERIMENT_GRID = {
    "epochs": [10, 20, 30],
    "batch_sizes": [4, 16, 32, 64],
    "models": ["model1", "model2"],
}

BONUS_EPOCHS = 40
SEED = 42

CSV_COLUMNS = [
    "model", "epochs", "batch_size",
    "train_acc", "val_acc", "test_acc",
    "precision", "recall", "f1_score",
    "training_time_sec",
]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

for d in (OUTPUTS_DIR, EXPLORATION_DIR, MODELS_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)


def banner(title):
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


# =============================================================================
# 1. DATASET EXPLORATION
# =============================================================================

banner("1. DATASET EXPLORATION")


def count_by_class(root_dir):
    counts = {}
    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if os.path.isdir(cls_path):
            counts[cls] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
    return counts


train_counts = count_by_class(TRAIN_DIR)
test_counts = count_by_class(TEST_DIR)

CLASS_NAMES = sorted(train_counts.keys())
NUM_CLASSES = len(CLASS_NAMES)

df_counts = pd.DataFrame({
    "clase": CLASS_NAMES,
    "train": [train_counts[c] for c in CLASS_NAMES],
    "test": [test_counts.get(c, 0) for c in CLASS_NAMES],
})
df_counts["total"] = df_counts["train"] + df_counts["test"]
print(df_counts.to_string(index=False))
print(f"\nTotal train: {df_counts['train'].sum()} — Total test: {df_counts['test'].sum()}")
print(f"Clases: {NUM_CLASSES} — {CLASS_NAMES}")

widths, heights, modes = [], [], []
for cls in CLASS_NAMES:
    cls_path = os.path.join(TRAIN_DIR, cls)
    files = os.listdir(cls_path)
    for f in random.sample(files, min(30, len(files))):
        with Image.open(os.path.join(cls_path, f)) as img:
            widths.append(img.size[0])
            heights.append(img.size[1])
            modes.append(img.mode)

print(f"\nAncho  — min: {min(widths)}, max: {max(widths)}, media: {np.mean(widths):.1f}")
print(f"Alto   — min: {min(heights)}, max: {max(heights)}, media: {np.mean(heights):.1f}")
print(f"Modos de imagen: {set(modes)}")

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].bar(CLASS_NAMES, [train_counts[c] for c in CLASS_NAMES], color="steelblue")
ax[0].set_title("Distribución de clases — Train")
ax[0].tick_params(axis="x", rotation=45)
ax[1].bar(CLASS_NAMES, [test_counts.get(c, 0) for c in CLASS_NAMES], color="indianred")
ax[1].set_title("Distribución de clases — Test")
ax[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(EXPLORATION_DIR, "class_distribution.png"), dpi=120)
plt.close()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax_, cls in zip(axes.flat, CLASS_NAMES):
    files = [f for f in os.listdir(os.path.join(TRAIN_DIR, cls))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sample_path = os.path.join(TRAIN_DIR, cls, random.choice(files))
    img = Image.open(sample_path)
    ax_.imshow(img)
    ax_.set_title(f"{cls}\n{img.size[0]}x{img.size[1]}", fontsize=10)
    ax_.axis("off")
    shutil.copy(sample_path, os.path.join(EXPLORATION_DIR, f"sample_{cls}.jpg"))
plt.suptitle("Muestra por clase — tamaño original")
plt.tight_layout()
plt.savefig(os.path.join(EXPLORATION_DIR, "samples_per_class.png"), dpi=120)
plt.close()

print(f"\nExploración guardada en: {EXPLORATION_DIR}")


# =============================================================================
# 2. DATASET CONSTRUCTION
# =============================================================================

banner("2. DATASET CONSTRUCTION")


def build_generators(batch_size):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_flow = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED,
        classes=CLASS_NAMES,
    )
    val_flow = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
        classes=CLASS_NAMES,
    )
    test_flow = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        classes=CLASS_NAMES,
    )
    return train_flow, val_flow, test_flow


tr_check, va_check, te_check = build_generators(batch_size=32)
print(f"Train muestras: {tr_check.samples}")
print(f"Val muestras:   {va_check.samples}")
print(f"Test muestras:  {te_check.samples}")
print(f"Mapeo clases:   {tr_check.class_indices}")


# =============================================================================
# MODEL BUILDERS
# =============================================================================

def build_model1():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(32, (5, 5), strides=1, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(5, 5)),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="cnn_simple")
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_model2():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(48, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(96, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="cnn_profunda")
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_model_bonus():
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Conv2D(48, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(96, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(100, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="cnn_bonus")
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


MODEL_BUILDERS = {
    "model1": build_model1,
    "model2": build_model2,
}


# =============================================================================
# PLOT HELPERS
# =============================================================================

def plot_history(history, title, filepath):
    h = history.history if hasattr(history, "history") else history
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(h["loss"], label="train")
    ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title(f"{title} — Loss"); ax[0].set_xlabel("época"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h["accuracy"], label="train")
    ax[1].plot(h["val_accuracy"], label="val")
    ax[1].set_title(f"{title} — Accuracy"); ax[1].set_xlabel("época"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close()


def plot_confusion(y_true, y_pred, title, filepath):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f"{title} — Confusion Matrix")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close()


# =============================================================================
# CSV / RESUME HELPERS
# =============================================================================

def load_completed_runs():
    if not os.path.exists(RESULTS_CSV):
        return set()
    try:
        df = pd.read_csv(RESULTS_CSV)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return set()
    if df.empty:
        return set()
    return {
        (str(r["model"]), int(r["epochs"]), int(r["batch_size"]))
        for _, r in df.iterrows()
    }


def append_result(row):
    new_file = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


# =============================================================================
# 3. EXPERIMENT LOOP
# =============================================================================

banner("3. EXPERIMENT LOOP")

completed = load_completed_runs()
if completed:
    print(f"Resume: {len(completed)} combinaciones ya completadas, se omitirán.")
else:
    print("No hay resultados previos — se ejecutará la grilla completa.")

combinations = [
    (model_name, epochs, batch_size)
    for epochs in EXPERIMENT_GRID["epochs"]
    for batch_size in EXPERIMENT_GRID["batch_sizes"]
    for model_name in EXPERIMENT_GRID["models"]
]
print(f"Combinaciones totales: {len(combinations)}")


def run_experiment(model_name, epochs, batch_size):
    tag = f"{model_name}_e{epochs}_bs{batch_size}"
    print(f"\n--- Ejecutando {tag} ---")

    tf.keras.backend.clear_session()
    tr, va, te = build_generators(batch_size=batch_size)
    model = MODEL_BUILDERS[model_name]()

    t0 = time.time()
    history = model.fit(tr, validation_data=va, epochs=epochs, verbose=2)
    training_time = time.time() - t0

    train_acc = float(history.history["accuracy"][-1])
    val_acc = float(history.history["val_accuracy"][-1])

    probs = model.predict(te, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = te.classes
    test_acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    model_path = os.path.join(MODELS_DIR, f"{tag}.keras")
    curves_path = os.path.join(PLOTS_DIR, f"{tag}_curves.png")
    cm_path = os.path.join(PLOTS_DIR, f"{tag}_cm.png")

    model.save(model_path)
    plot_history(history, tag, curves_path)
    plot_confusion(y_true, y_pred, tag, cm_path)

    row = {
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "test_acc": round(test_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "training_time_sec": round(training_time, 1),
    }
    append_result(row)

    print(f"    train_acc={row['train_acc']} val_acc={row['val_acc']} test_acc={row['test_acc']}")
    print(f"    precision={row['precision']} recall={row['recall']} f1={row['f1_score']}")
    print(f"    tiempo={row['training_time_sec']}s — guardado en {model_path}")

    return row


for model_name, epochs, batch_size in combinations:
    key = (model_name, int(epochs), int(batch_size))
    if key in completed:
        print(f"skip {model_name}_e{epochs}_bs{batch_size} (ya registrado)")
        continue
    run_experiment(model_name, epochs, batch_size)


# =============================================================================
# 4. SUMMARY REPORT
# =============================================================================

banner("4. SUMMARY REPORT")

results_df = pd.read_csv(RESULTS_CSV)

best_val = results_df.loc[results_df["val_acc"].idxmax()]
best_test = results_df.loc[results_df["test_acc"].idxmax()]
best_f1 = results_df.loc[results_df["f1_score"].idxmax()]

sorted_df = results_df.sort_values(
    ["test_acc", "f1_score", "val_acc"], ascending=False
).reset_index(drop=True)


def fmt_best(label, row):
    return (
        f"{label}: {row['model']} | epochs={int(row['epochs'])} | "
        f"batch_size={int(row['batch_size'])} | "
        f"val_acc={row['val_acc']:.4f} | test_acc={row['test_acc']:.4f} | "
        f"f1={row['f1_score']:.4f}"
    )


summary_lines = [
    "TALLER 2 — GRID EXPERIMENT SUMMARY",
    "=" * 78,
    "",
    f"Grid: {EXPERIMENT_GRID}",
    f"Total de experimentos: {len(results_df)}",
    "",
    "MEJORES COMBINACIONES",
    "-" * 78,
    fmt_best("Mejor por val_acc ", best_val),
    fmt_best("Mejor por test_acc", best_test),
    fmt_best("Mejor por f1_score", best_f1),
    "",
    "TABLA COMPLETA (ordenada por test_acc desc)",
    "-" * 78,
    sorted_df.to_string(index=False),
    "",
]

summary_text = "\n".join(summary_lines)
print(summary_text)

with open(SUMMARY_PATH, "w") as f:
    f.write(summary_text)

print(f"\nResumen guardado en: {SUMMARY_PATH}")


# =============================================================================
# 5. BONUS — MODEL 2 + BN + DROPOUT + DATA AUGMENTATION
# =============================================================================

banner("5. BONUS — MODEL 2 + BN + DROPOUT + AUGMENTATION")

bonus_path = os.path.join(MODELS_DIR, "model_bonus.keras")
if os.path.exists(bonus_path):
    print(f"Modelo bonus ya existe en {bonus_path} — se omite reentrenamiento.")
else:
    best_bs_bonus = int(
        results_df[results_df["model"] == "model2"]
        .sort_values("val_acc", ascending=False)
        .iloc[0]["batch_size"]
    )
    print(f"Usando batch_size={best_bs_bonus} (mejor de model2 por val_acc)")

    aug_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=(0.9, 1.1),
    )
    plain_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    test_datagen_bonus = ImageDataGenerator(rescale=1.0 / 255)

    tr_aug = aug_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=best_bs_bonus,
        class_mode="categorical", subset="training", shuffle=True, seed=SEED, classes=CLASS_NAMES)
    va_aug = plain_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=best_bs_bonus,
        class_mode="categorical", subset="validation", shuffle=False, seed=SEED, classes=CLASS_NAMES)
    te_aug = test_datagen_bonus.flow_from_directory(
        TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=32,
        class_mode="categorical", shuffle=False, classes=CLASS_NAMES)

    tf.keras.backend.clear_session()
    model_bonus = build_model_bonus()
    early = callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)

    history_bonus = model_bonus.fit(
        tr_aug, validation_data=va_aug,
        epochs=BONUS_EPOCHS, callbacks=[early, reduce_lr], verbose=2,
    )

    plot_history(history_bonus, "Modelo Bonus", os.path.join(PLOTS_DIR, "bonus_curves.png"))
    model_bonus.save(bonus_path)

    probs_b = model_bonus.predict(te_aug, verbose=0)
    y_pred_b = np.argmax(probs_b, axis=1)
    y_true_b = te_aug.classes
    plot_confusion(y_true_b, y_pred_b, "Modelo Bonus", os.path.join(PLOTS_DIR, "bonus_cm.png"))

    report_bonus = classification_report(y_true_b, y_pred_b,
                                         target_names=CLASS_NAMES, zero_division=0)
    print("\n--- Modelo Bonus ---\n" + report_bonus)
    print(f"\nmodelo bonus guardado en: {bonus_path}")


print("\nOK.")
