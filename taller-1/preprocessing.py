# =============================================================
# deep-learning-class - Taller 1
# Punto 3 y 4: Limpieza de datos y transformacion de variables
# =============================================================
# El dataset original (CSV) no se modifica en ningun momento.
# Se trabaja sobre una copia en memoria y el resultado se
# guarda en un archivo nuevo: dataset_clean.csv
# =============================================================

import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------
# CONFIGURACION
# -----------------------------------------------------------
CSV_PATH        = "gender-classifier-DFE-791531.csv"
OUTPUT_PATH     = "dataset_clean.csv"
OUTPUT_TFIDF_PATH = "dataset_clean_tfidf.csv"
REPORT_DIR      = "plots/"
os.makedirs(REPORT_DIR, exist_ok=True)


# =============================================================
# PASO 1: CARGA — se trabaja sobre una copia, nunca el original
# =============================================================
def load_copy(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1")
    df  = raw.copy()                          # copia en memoria
    print(f"Dataset original  : {raw.shape[0]} filas x {raw.shape[1]} columnas")
    print(f"Copia de trabajo  : {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Original intacto  : {raw.shape == df.shape}\n")
    return df


# =============================================================
# PASO 2: FILTRADO — clasificacion binaria (male / female)
# =============================================================
def filtrar_binario(df: pd.DataFrame) -> pd.DataFrame:
    antes = len(df)
    df = df[df["gender"].isin(["male", "female"])].copy()
    print(f"[Filtro binario] {antes} -> {len(df)} filas "
          f"(eliminadas: {antes - len(df)})")
    return df


# =============================================================
# PASO 3: ELIMINAR COLUMNAS SIN VALOR PREDICTIVO
# =============================================================
# Justificacion por columna:
#
# Metadatos del sistema CrowdFlower (no describen al usuario):
#   _unit_id, _golden, _unit_state, _trusted_judgments,
#   _last_judgment_at
#
# Columnas casi completamente vacias (>99% nulos):
#   gender_gold (99.7%), profile_yn_gold (99.7%),
#   tweet_coord (99.4%)
#
# Identificadores sin informacion:
#   tweet_id
#
# Fechas redundantes (tweet_created cubre lo mismo que created
# pero con menos informacion util):
#   tweet_created
#
# URLs no procesables por MLP:
#   profileimage
#
# gender:confidence es metadata del etiquetado, no del usuario.
# Incluirla seria "trampa" porque esta directamente ligada
# al proceso de asignacion de la etiqueta gender.
# =============================================================

COLUMNAS_ELIMINAR = [
    "_unit_id",
    "_golden",
    "_unit_state",
    "_trusted_judgments",
    "_last_judgment_at",
    "gender_gold",
    "profile_yn_gold",
    "tweet_coord",
    "tweet_id",
    "tweet_created",
    "profileimage",
    "gender:confidence",
    "profile_yn:confidence",    # metadata del etiquetado, no del usuario
]

def eliminar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    existentes = [c for c in COLUMNAS_ELIMINAR if c in df.columns]
    df = df.drop(columns=existentes)
    print(f"[Columnas eliminadas] {len(existentes)} columnas")
    print(f"  {existentes}")
    print(f"  Columnas restantes: {list(df.columns)}\n")
    return df


# =============================================================
# PASO 4: VARIABLE OBJETIVO
# =============================================================
def encodear_target(df: pd.DataFrame) -> pd.DataFrame:
    # 0 = male  |  1 = female
    df["gender_label"] = (df["gender"] == "female").astype(int)
    df = df.drop(columns=["gender"])
    print(f"[Target] gender_label creado  -> 0=male, 1=female")
    print(f"  male  : {(df['gender_label'] == 0).sum()}")
    print(f"  female: {(df['gender_label'] == 1).sum()}\n")
    return df


# =============================================================
# PASO 5: TRATAMIENTO DE NULOS
# =============================================================
# Estrategia por columna:
#
# user_timezone (34.1% nulos): se imputa con "Unknown" porque
#   eliminar filas perderia un tercio del dataset.
#
# tweet_location (32.2% nulos): texto muy ruidoso (ciudad,
#   pais, frases libres). Se elimina la columna completa ya
#   que no es procesable de forma confiable por MLP sin NLP.
#
# description (13.2% nulos): texto libre. Se imputa con string
#   vacio para que el feature derivado (longitud) sea 0.
#
# name (pocos nulos): se imputa con string vacio.
# =============================================================

def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    # tweet_location: muy ruidoso y 32% nulos -> eliminar
    if "tweet_location" in df.columns:
        df = df.drop(columns=["tweet_location"])
        print("[Nulos] tweet_location eliminada (32% nulos, texto no estructurado)")

    # user_timezone: imputar con categoria "Unknown"
    if "user_timezone" in df.columns:
        nulos_tz = df["user_timezone"].isnull().sum()
        df["user_timezone"] = df["user_timezone"].fillna("Unknown")
        print(f"[Nulos] user_timezone: {nulos_tz} nulos imputados con 'Unknown'")

    # description: imputar con string vacio
    if "description" in df.columns:
        nulos_desc = df["description"].isnull().sum()
        df["description"] = df["description"].fillna("")
        print(f"[Nulos] description: {nulos_desc} nulos imputados con ''")

    # name: imputar con string vacio
    if "name" in df.columns:
        nulos_name = df["name"].isnull().sum()
        df["name"] = df["name"].fillna("")
        print(f"[Nulos] name: {nulos_name} nulos imputados con ''")

    # text (ultimo tweet): imputar con string vacio
    if "text" in df.columns:
        nulos_text = df["text"].isnull().sum()
        df["text"] = df["text"].fillna("")
        print(f"[Nulos] text: {nulos_text} nulos imputados con ''")

    print()
    return df


# =============================================================
# PASO 6: INGENIERIA DE FEATURES DESDE TEXTO
# =============================================================
# En lugar de pasar texto crudo al MLP (imposible sin NLP),
# extraemos features numericas derivadas del texto que si
# pueden alimentar la red directamente.
# =============================================================

def features_desde_texto(df: pd.DataFrame) -> pd.DataFrame:
    # Longitud de la descripcion del perfil
    if "description" in df.columns:
        df["desc_length"] = df["description"].str.len()
        df = df.drop(columns=["description"])
        print("[Features texto] desc_length creada desde description")

    # Longitud del nombre de usuario
    if "name" in df.columns:
        df["name_length"] = df["name"].str.len()
        df = df.drop(columns=["name"])
        print("[Features texto] name_length creada desde name")

    # Longitud del ultimo tweet
    if "text" in df.columns:
        df["tweet_length"] = df["text"].str.len()
        df = df.drop(columns=["text"])
        print("[Features texto] tweet_length creada desde text")

    print()
    return df


def construir_dataset_tfidf(df: pd.DataFrame) -> pd.DataFrame:
    tfidf_df = df[["gender_label", "name", "description", "text"]].copy()
    tfidf_df["text_all"] = (
        tfidf_df["name"].astype(str) + " " +
        tfidf_df["description"].astype(str) + " " +
        tfidf_df["text"].astype(str)
    ).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    tfidf_df = tfidf_df[["gender_label", "text_all"]]
    return tfidf_df


# =============================================================
# PASO 7: TRANSFORMACION DE FEATURES — PUNTO 4 DEL TALLER
# =============================================================

# --- 7a. Colores hex -> entero decimal ---
# link_color y sidebar_color son strings hexadecimales (ej: "08C2C2")
# Se convierten a su valor entero equivalente para que la red
# pueda procesarlos numericamente.

def hex_a_decimal(hex_str):
    try:
        return int(str(hex_str).strip().lstrip("#"), 16)
    except (ValueError, TypeError):
        return 0

def transformar_colores(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["link_color", "sidebar_color"]:
        if col in df.columns:
            df[col] = df[col].apply(hex_a_decimal)
            print(f"[Colores] {col}: hex -> decimal")
    print()
    return df


# --- 7b. profile_yn: binary encoding ---
# yes -> 1  |  no -> 0
# Justificacion: variable binaria, no necesita One-Hot.

def transformar_profile_yn(df: pd.DataFrame) -> pd.DataFrame:
    if "profile_yn" in df.columns:
        df["profile_yn"] = df["profile_yn"].map({"yes": 1, "no": 0}).fillna(0).astype(int)
        print("[Categoricas] profile_yn: yes/no -> 1/0")
    print()
    return df


# --- 7c. created: antiguedad de la cuenta en dias ---
# Convertir fecha de creacion a numero de dias desde creacion
# hasta la fecha mas reciente del dataset.
# Justificacion: la red no puede procesar fechas directamente.

def transformar_fecha(df: pd.DataFrame) -> pd.DataFrame:
    if "created" in df.columns:
        df["created"] = pd.to_datetime(df["created"], errors="coerce")
        fecha_ref = df["created"].max()
        df["account_age_days"] = (fecha_ref - df["created"]).dt.days.fillna(0).astype(int)
        df = df.drop(columns=["created"])
        print(f"[Fechas] created -> account_age_days (ref: {fecha_ref.date()})")
    print()
    return df


# --- 7d. user_timezone: One-Hot Encoding ---
# Justificacion: variable nominal sin orden, One-Hot es la
# transformacion correcta. Se agrupan timezones con menos de
# 50 apariciones en la categoria "Other" para evitar explosion
# de dimensionalidad.

def transformar_timezone(df: pd.DataFrame) -> pd.DataFrame:
    if "user_timezone" in df.columns:
        conteo = df["user_timezone"].value_counts()
        timezones_frecuentes = conteo[conteo >= 50].index
        df["user_timezone"] = df["user_timezone"].apply(
            lambda x: x if x in timezones_frecuentes else "Other"
        )
        dummies = pd.get_dummies(df["user_timezone"], prefix="tz", dtype=int)
        df = pd.concat([df.drop(columns=["user_timezone"]), dummies], axis=1)
        print(f"[One-Hot] user_timezone -> {dummies.shape[1]} columnas dummy")
    print()
    return df


# =============================================================
# PASO 8: VERIFICACION FINAL
# =============================================================

def verificar(df: pd.DataFrame):
    print("=" * 60)
    print("DATASET LIMPIO - RESUMEN FINAL")
    print("=" * 60)
    print(f"Forma            : {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Nulos restantes  : {df.isnull().sum().sum()}")
    print(f"Tipos de datos   :\n{df.dtypes.value_counts().to_string()}")
    print(f"\nColumnas finales :\n{list(df.columns)}")
    print(f"\nDistribucion del target:")
    print(df["gender_label"].value_counts().to_string())


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESAMIENTO - Taller 1")
    print("=" * 60 + "\n")

    df = load_copy(CSV_PATH)

    df = filtrar_binario(df)
    df = eliminar_columnas(df)
    df = encodear_target(df)

    df = tratar_nulos(df)
    df_tfidf = construir_dataset_tfidf(df)

    df = features_desde_texto(df)
    df = transformar_colores(df)
    # df = transformar_profile_yn(df)
    df = transformar_fecha(df)

    # df = df.drop(columns=["user_timezone", "profile_yn" ])
    df = df.drop(columns=["profile_yn" ])
    df = transformar_timezone(df)
    # eliminar columnas user_timezone

    verificar(df)

    df.to_csv(OUTPUT_PATH, index=False)
    df_tfidf.to_csv(OUTPUT_TFIDF_PATH, index=False)
    print(f"\nDataset limpio guardado en: {OUTPUT_PATH}")
    print(f"Dataset TF-IDF guardado en: {OUTPUT_TFIDF_PATH}")
    print("El archivo original no fue modificado.")
