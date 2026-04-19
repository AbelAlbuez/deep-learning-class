# =============================================================
# Taller 1 – Aprendizaje Profundo
# Punto 2: Comprensión del Dataset (EDA)
# Twitter User Gender Classification
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Estilo global de gráficas ──────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
COLORS = {"male": "#4C72B0", "female": "#DD8452"}
OUTPUT_DIR = "plots/"  # carpeta donde se guardan las gráficas

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================
# 1. CARGA DEL DATASET
# =============================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas\n")
    return df


# =============================================================
# 2. RESUMEN GENERAL
# =============================================================
def resumen_general(df: pd.DataFrame):
    print("=" * 60)
    print("RESUMEN GENERAL DEL DATASET")
    print("=" * 60)

    print("\n── Primeras filas ──")
    print(df.head(3).to_string())

    print("\n── Tipos de datos ──")
    print(df.dtypes.to_string())

    print("\n── Estadísticas descriptivas (numéricas) ──")
    print(df.describe().to_string())

    print("\n── Valores nulos por columna ──")
    nulos = df.isnull().sum()
    pct   = (nulos / len(df) * 100).round(2)
    resumen_nulos = pd.DataFrame({"nulos": nulos, "% nulos": pct})
    print(resumen_nulos[resumen_nulos["nulos"] > 0].to_string())


# =============================================================
# 3. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
# =============================================================
def plot_distribucion_genero(df: pd.DataFrame):
    print("\n── Distribución de clases (gender) ──")
    conteo = df["gender"].value_counts()
    print(conteo.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribución de la variable objetivo: gender", fontsize=14, fontweight="bold")

    # Barras — todas las clases originales
    conteo.plot(kind="bar", ax=axes[0], color=sns.color_palette("muted", len(conteo)), edgecolor="white")
    axes[0].set_title("Todas las clases")
    axes[0].set_xlabel("Género")
    axes[0].set_ylabel("Cantidad")
    axes[0].tick_params(axis="x", rotation=0)
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 50,
                     f"{int(bar.get_height()):,}",
                     ha="center", va="bottom", fontsize=9)

    # Pie — solo male / female (Opción B)
    df_bin = df[df["gender"].isin(["male", "female"])]
    conteo_bin = df_bin["gender"].value_counts()
    axes[1].pie(
        conteo_bin,
        labels=conteo_bin.index,
        autopct="%1.1f%%",
        colors=[COLORS["male"], COLORS["female"]],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Solo male / female (Opción B)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}01_distribucion_genero.png", dpi=150)
    plt.show()
    print(f"  → Guardada: {OUTPUT_DIR}01_distribucion_genero.png")


# =============================================================
# 4. FILTRADO OPCIÓN B  (male / female)
# =============================================================
def filtrar_binario(df: pd.DataFrame) -> pd.DataFrame:
    df_bin = df[df["gender"].isin(["male", "female"])].copy()
    df_bin["gender_label"] = (df_bin["gender"] == "female").astype(int)  # 0=male, 1=female
    print(f"\nDataset binario: {df_bin.shape[0]} filas")
    print(f"   male  : {(df_bin['gender']=='male').sum():,}")
    print(f"   female: {(df_bin['gender']=='female').sum():,}")
    return df_bin


# =============================================================
# 5. ANÁLISIS DE VARIABLES NUMÉRICAS
# =============================================================
def plot_numericas(df: pd.DataFrame):
    numericas = ["fav_number", "retweet_count", "tweet_count", "gender:confidence"]
    numericas = [c for c in numericas if c in df.columns]

    fig, axes = plt.subplots(len(numericas), 2, figsize=(14, len(numericas) * 3.5))
    fig.suptitle("Distribución de variables numéricas por género", fontsize=14, fontweight="bold")

    for i, col in enumerate(numericas):
        # Histograma por género
        for gen, color in COLORS.items():
            subset = df[df["gender"] == gen][col].dropna()
            # Cap outliers para visualización (percentil 99)
            cap = subset.quantile(0.99)
            subset_capped = subset[subset <= cap]
            axes[i, 0].hist(subset_capped, bins=40, alpha=0.6, color=color, label=gen)
        axes[i, 0].set_title(f"{col} — histograma")
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel("Frecuencia")
        axes[i, 0].legend()

        # Boxplot por género
        data_box = [df[df["gender"] == g][col].dropna() for g in COLORS]
        axes[i, 1].boxplot(data_box, labels=list(COLORS.keys()),
                           patch_artist=True,
                           boxprops=dict(facecolor="lightblue", color="steelblue"),
                           medianprops=dict(color="red", linewidth=2))
        axes[i, 1].set_title(f"{col} — boxplot")
        axes[i, 1].set_ylabel(col)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}02_numericas.png", dpi=150)
    plt.show()
    print(f"  → Guardada: {OUTPUT_DIR}02_numericas.png")


# =============================================================
# 6. ANÁLISIS DE VARIABLES CATEGÓRICAS
# =============================================================
def plot_categoricas(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Variables categóricas vs género", fontsize=14, fontweight="bold")

    # profile_yn
    if "profile_yn" in df.columns:
        ct = pd.crosstab(df["profile_yn"], df["gender"])
        ct.plot(kind="bar", ax=axes[0], color=[COLORS["male"], COLORS["female"]], edgecolor="white")
        axes[0].set_title("¿Tiene imagen de perfil? vs género")
        axes[0].set_xlabel("profile_yn")
        axes[0].set_ylabel("Cantidad")
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].legend(title="gender")

    # Top 10 user_timezone
    if "user_timezone" in df.columns:
        top_tz = df["user_timezone"].value_counts().head(10).index
        df_tz  = df[df["user_timezone"].isin(top_tz)]
        ct2    = pd.crosstab(df_tz["user_timezone"], df_tz["gender"])
        ct2.plot(kind="barh", ax=axes[1], color=[COLORS["male"], COLORS["female"]], edgecolor="white")
        axes[1].set_title("Top 10 user_timezone vs género")
        axes[1].set_xlabel("Cantidad")
        axes[1].legend(title="gender")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}03_categoricas.png", dpi=150)
    plt.show()
    print(f"  → Guardada: {OUTPUT_DIR}03_categoricas.png")


# =============================================================
# 7. MATRIZ DE CORRELACIÓN (variables numéricas)
# =============================================================
def plot_correlacion(df: pd.DataFrame):
    numericas = ["fav_number", "retweet_count", "tweet_count",
                 "gender:confidence", "gender_label"]
    numericas = [c for c in numericas if c in df.columns]

    corr = df[numericas].corr()

    plt.figure(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Matriz de correlación — variables numéricas", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}04_correlacion.png", dpi=150)
    plt.show()
    print(f"  → Guardada: {OUTPUT_DIR}04_correlacion.png")


# =============================================================
# 8. ANÁLISIS DE NULOS EN COLUMNAS CLAVE
# =============================================================
def plot_nulos(df: pd.DataFrame):
    nulos_pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    nulos_pct = nulos_pct[nulos_pct > 0]

    if nulos_pct.empty:
        print("\nNo hay valores nulos en el dataset.")
        return

    plt.figure(figsize=(10, 5))
    bars = plt.barh(nulos_pct.index, nulos_pct.values,
                    color=sns.color_palette("Reds_r", len(nulos_pct)))
    plt.xlabel("% de valores nulos")
    plt.title("Porcentaje de valores nulos por columna", fontsize=13, fontweight="bold")
    plt.gca().xaxis.set_major_formatter(mticker.PercentFormatter())
    for bar, val in zip(bars, nulos_pct.values):
        plt.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}05_nulos.png", dpi=150)
    plt.show()
    print(f"  → Guardada: {OUTPUT_DIR}05_nulos.png")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    # Ajusta esta ruta al CSV descargado de Kaggle
    CSV_PATH = "gender-classifier-DFE-791531.csv"

    df = load_data(CSV_PATH)
    resumen_general(df)

    plot_distribucion_genero(df)

    df_bin = filtrar_binario(df)

    plot_nulos(df_bin)
    plot_numericas(df_bin)
    plot_categoricas(df_bin)
    plot_correlacion(df_bin)

    print("\nEDA completado. Graficas guardadas en:", OUTPUT_DIR)