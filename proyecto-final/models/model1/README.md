# Modelo 1 — CNN Baseline + OkanNet

Implementación del Modelo 1 del proyecto final de Aprendizaje Profundo (MAIN
Javeriana). Incluye dos arquitecturas: la CNN Baseline propia (responsabilidad
de Abel) y una réplica de OkanNet (Uçar & Kurt, 2026) para comparación
académica del impacto de BatchNorm.

## Archivos

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Constantes centralizadas: rutas, hiperparámetros, seed |
| `dataset.py` | Transforms, DataLoaders con split 80/20, class_weights |
| `architectures.py` | `CNNBaseline` y `OkanNet` con factory `get_model()` |
| `utils.py` | Seed, device, EarlyStopping, métricas, plots |
| `train.py` | CLI de entrenamiento (`--arch baseline\|okannet`) |
| `evaluate.py` | CLI de evaluación sobre holdout Testing |
| `compare.py` | Genera comparativa baseline vs okannet |
| `requirements.txt` | Dependencias específicas del Modelo 1 |
| `checkpoints/` | Pesos entrenados (gitignored) |

## Setup

Desde la raíz del proyecto (`proyecto-final/`):

```bash
# 1. Crear venv con Python 3.11
python3.11 -m venv .venv

# 2. Activar venv
source .venv/bin/activate

# 3. Instalar dependencias del Modelo 1
pip install --upgrade pip
pip install -r models/model1/requirements.txt
```

## Ejecución paso a paso

Desde la raíz del proyecto, con el venv activado:

```bash
# Entrenar CNN Baseline (~15-25 min en Mac con MPS)
python models/model1/train.py --arch baseline

# Entrenar OkanNet (~15-25 min)
python models/model1/train.py --arch okannet

# Evaluar sobre Testing (holdout)
python models/model1/evaluate.py --arch baseline
python models/model1/evaluate.py --arch okannet

# Generar comparativa final
python models/model1/compare.py
```

## Outputs generados

```
models/model1/checkpoints/
├── baseline_best.pth          # Mejor checkpoint baseline (gitignored)
└── okannet_best.pth           # Mejor checkpoint okannet (gitignored)

outputs/figures/model-1/
├── curvas_baseline.png                    # Loss, Acc, F1, LR por época
├── confusion_baseline.png                 # Matriz confusión normalizada
├── curvas_okannet.png
├── confusion_okannet.png
└── comparacion_baseline_okannet.png       # Comparativa lado a lado

outputs/results/model-1/
├── baseline_metrics.csv                   # Métricas por época
├── baseline_test_report.txt               # classification_report sklearn
├── okannet_metrics.csv
├── okannet_test_report.txt
└── comparison_summary.txt                 # Tabla comparativa final
```

## Arquitectura del Baseline (para reporte LaTeX §III-B)

| # | Capa | Filtros | Kernel | Stride | Padding | Salida (C, H, W) | Parámetros |
|---|---|---|---|---|---|---|---|
| 1 | Conv2D | 16 | 3×3 | 1 | 1 | (16, 224, 224) | 448 |
| 2 | ReLU | — | — | — | — | (16, 224, 224) | 0 |
| 3 | MaxPool2D | — | 2×2 | 2 | 0 | (16, 112, 112) | 0 |
| 4 | Conv2D | 32 | 3×3 | 1 | 1 | (32, 112, 112) | 4,640 |
| 5 | ReLU | — | — | — | — | (32, 112, 112) | 0 |
| 6 | MaxPool2D | — | 2×2 | 2 | 0 | (32, 56, 56) | 0 |
| 7 | Conv2D | 64 | 3×3 | 1 | 1 | (64, 56, 56) | 18,496 |
| 8 | ReLU | — | — | — | — | (64, 56, 56) | 0 |
| 9 | MaxPool2D | — | 2×2 | 2 | 0 | (64, 28, 28) | 0 |
| 10 | Flatten | — | — | — | — | (50176,) | 0 |
| 11 | Linear | — | — | — | — | (128,) | 6,422,656 |
| 12 | ReLU | — | — | — | — | (128,) | 0 |
| 13 | Dropout(0.5) | — | — | — | — | (128,) | 0 |
| 14 | Linear | — | — | — | — | (4,) | 516 |

**Total parámetros entrenables: 6,446,756**

OkanNet es idéntico pero añade `BatchNorm2d` después de cada Conv2d, sumando 224
parámetros adicionales (total: 6,446,980).

## Hiperparámetros (para reporte LaTeX §IV-A)

| Hiperparámetro | Valor |
|---|---|
| Optimizador | Adam (betas=0.9, 0.999) |
| Learning rate inicial | 1e-3 |
| Scheduler | ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6) |
| Loss | CrossEntropyLoss(weight=class_weights) |
| Batch size | 32 |
| Epochs máx | 40 |
| Early stopping | patience=7 sobre F1-macro val |
| Split | 80/20 sobre Training |
| Seed | 42 |

## Tiempos esperados

| Hardware | Tiempo por arquitectura | Tiempo total (ambas + eval + compare) |
|---|---|---|
| Mac con MPS (M1/M2/M3) | ~15-25 min | ~40-60 min |
| CPU puro | ~60-90 min | ~2.5-4 h |
| Colab GPU T4 | ~5-10 min | ~15-25 min |
