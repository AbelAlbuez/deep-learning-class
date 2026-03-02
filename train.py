import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

from models import get_model

os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------------------------------------------
# CONFIGURACION
# -----------------------------------------------------------
CSV_PATH   = "dataset_clean.csv"
CSV_TFIDF_PATH = "dataset_clean_tfidf.csv"
SEED       = 42
EPOCHS     = 200
BATCH_SIZE = 64
LR         = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)


# =============================================================
# TF-IDF + REGRESION LOGISTICA
# =============================================================
def cargar_datos_tfidf():
    df = pd.read_csv(CSV_TFIDF_PATH)
    X_text = df["text_all"].astype(str).values
    y = df["gender_label"].values.astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_text, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def entrenar_evaluar_tfidf():
    X_train, X_val, X_test, y_train, y_val, y_test = cargar_datos_tfidf()

    print(f"Train : {len(X_train)} filas")
    print(f"Val   : {len(X_val)} filas")
    print(f"Test  : {len(X_test)} filas\n")

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
    )

    X_train_word = word_vec.fit_transform(X_train)
    X_val_word = word_vec.transform(X_val)
    X_test_word = word_vec.transform(X_test)

    X_train_char = char_vec.fit_transform(X_train)
    X_val_char = char_vec.transform(X_val)
    X_test_char = char_vec.transform(X_test)

    X_train_all = hstack([X_train_word, X_train_char], format="csr")
    X_val_all = hstack([X_val_word, X_val_char], format="csr")
    X_test_all = hstack([X_test_word, X_test_char], format="csr")

    clf = LogisticRegression(
        max_iter=2500,
        solver="liblinear",
        C=2.0,
        random_state=SEED,
    )
    clf.fit(X_train_all, y_train)

    val_scores = clf.predict_proba(X_val_all)[:, 1]
    thresholds = np.linspace(0.30, 0.70, 81)
    best_threshold = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        val_pred = (val_scores >= thr).astype(int)
        f1 = f1_score(y_val, val_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)

    print(f"Mejor threshold en validacion (F1): {best_threshold:.3f}")

    test_scores = clf.predict_proba(X_test_all)[:, 1]
    y_pred = (test_scores >= best_threshold).astype(int)

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    print(f"\n  Accuracy : {metricas['accuracy']:.4f}")
    print(f"  Precision: {metricas['precision']:.4f}")
    print(f"  Recall   : {metricas['recall']:.4f}")
    print(f"  F1 Score : {metricas['f1']:.4f}")

    return np.array(y_test), np.array(y_pred), metricas


# =============================================================
# PUNTO 5: PARTICIONAMIENTO Y NORMALIZACION
# =============================================================
def cargar_datos():
    df = pd.read_csv(CSV_PATH)
    X  = df.drop(columns=["gender_label"]).values.astype(np.float32)
    y  = df["gender_label"].values.astype(np.float32)

    # Split 70 / 15 / 15 con stratify para mantener balance de clases
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    # Normalizacion: scaler ajustado SOLO sobre train
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"Train : {X_train.shape[0]} filas")
    print(f"Val   : {X_val.shape[0]} filas")
    print(f"Test  : {X_test.shape[0]} filas\n")

    def to_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        to_loader(X_train, y_train, shuffle=True),
        to_loader(X_val,   y_val),
        to_loader(X_test,  y_test),
        X_train.shape[1]
    )


# =============================================================
# PUNTO 6: ENTRENAMIENTO
# =============================================================
def entrenar(model, train_loader, val_loader, model_name):
    # El Perceptron se entrena con BCEWithLogitsLoss sobre la salida lineal
    # Los MLP se entrenan con BCELoss sobre la salida sigmoide
    es_perceptron = model_name == "perceptron"

    if es_perceptron:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    historia = {"train_loss": [], "val_loss": [],
                "train_acc":  [], "val_acc":  []}

    for epoch in range(1, EPOCHS + 1):
        # ── entrenamiento ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            if es_perceptron:
                logits = model.linear(X_batch).squeeze(-1)
                loss   = criterion(logits, y_batch.float())
                preds  = (logits >= 0.0).float()
            else:
                out   = model(X_batch).squeeze(-1)
                loss  = criterion(out, y_batch.float())
                preds = (out >= 0.5).float()

            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * len(y_batch)
            train_correct += (preds == y_batch).sum().item()
            train_total   += len(y_batch)

        # ── validacion ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                if es_perceptron:
                    logits = model.linear(X_batch).squeeze(-1)
                    loss   = criterion(logits, y_batch.float())
                    preds  = (logits >= 0.0).float()
                else:
                    out   = model(X_batch).squeeze(-1)
                    loss  = criterion(out, y_batch.float())
                    preds = (out >= 0.5).float()

                val_loss    += loss.item() * len(y_batch)
                val_correct += (preds == y_batch).sum().item()
                val_total   += len(y_batch)

        historia["train_loss"].append(train_loss / train_total)
        historia["val_loss"].append(val_loss   / val_total)
        historia["train_acc"].append(train_correct / train_total)
        historia["val_acc"].append(val_correct  / val_total)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | "
                  f"train_loss: {historia['train_loss'][-1]:.4f} | "
                  f"val_loss: {historia['val_loss'][-1]:.4f} | "
                  f"val_acc: {historia['val_acc'][-1]:.4f}")

    torch.save(model.state_dict(), f"models/{model_name}.pt")
    return historia


# =============================================================
# PUNTO 7: EVALUACION
# =============================================================
def evaluar(model, test_loader, model_name):
    model.eval()
    es_perceptron = model_name == "perceptron"
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            if es_perceptron:
                preds = (model.linear(X_batch).squeeze(-1) >= 0.0).float()
            else:
                preds = (model(X_batch).squeeze(-1) >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metricas = {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "f1"       : f1_score(y_true, y_pred, zero_division=0),
    }

    print(f"\n  Accuracy : {metricas['accuracy']:.4f}")
    print(f"  Precision: {metricas['precision']:.4f}")
    print(f"  Recall   : {metricas['recall']:.4f}")
    print(f"  F1 Score : {metricas['f1']:.4f}")

    return y_true, y_pred, metricas


# =============================================================
# GRAFICAS
# =============================================================
def plot_historia(historia, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Entrenamiento: {model_name}", fontweight="bold")

    axes[0].plot(historia["train_loss"], label="train")
    axes[0].plot(historia["val_loss"],   label="val")
    axes[0].set_title("Loss por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(historia["train_acc"], label="train")
    axes[1].plot(historia["val_acc"],   label="val")
    axes[1].set_title("Accuracy por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_historia.png", dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["male", "female"],
                yticklabels=["male", "female"])
    plt.title(f"Matriz de confusion: {model_name}", fontweight="bold")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_confusion.png", dpi=150)
    plt.close()


def plot_comparativo(resultados):
    nombres   = list(resultados.keys())
    metricas  = ["accuracy", "precision", "recall", "f1"]
    x         = np.arange(len(metricas))
    width     = 0.25
    colores   = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, nombre in enumerate(nombres):
        valores = [resultados[nombre][m] for m in metricas]
        ax.bar(x + i * width, valores, width, label=nombre, color=colores[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels(metricas)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Comparacion de modelos", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/comparativo.png", dpi=150)
    plt.close()
    print("\nGraficas guardadas en plots/")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    resultados = {}

    usar_tfidf = True
    modelos = ["mlp2"]
    nombres = {
        "mlp2": "MLP 2 capas ocultas",
    }

    if usar_tfidf:
        print("=" * 50)
        print("Entrenando: TF-IDF + Regresion Logistica")
        print("=" * 50)
        y_true, y_pred, metricas = entrenar_evaluar_tfidf()
        resultados["TF-IDF + LogReg"] = metricas
        plot_confusion(y_true, y_pred, "TF-IDF + LogReg")

    if modelos:
        train_loader, val_loader, test_loader, n_features = cargar_datos()

        for key in modelos:
            print("=" * 50)
            print(f"Entrenando: {nombres[key]}")
            print("=" * 50)

            model    = get_model(key, n_features)
            historia = entrenar(model, train_loader, val_loader, key)

            print(f"\nEvaluacion en test:")
            y_true, y_pred, metricas = evaluar(model, test_loader, key)

            resultados[nombres[key]] = metricas

            plot_historia(historia, nombres[key])
            plot_confusion(y_true, y_pred, nombres[key])

    if len(resultados) > 1:
        plot_comparativo(resultados)

    print("\n" + "=" * 50)
    print("RESUMEN FINAL")
    print("=" * 50)
    print(f"{'Modelo':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    for nombre, m in resultados.items():
        print(f"{nombre:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f}")
