# Modelo 2 — DeepCNN-BN + DeepCNN-BN-GAP

Implementación del Modelo 2 del proyecto final de Aprendizaje Profundo (MAIN
Javeriana). Una CNN profunda (10 capas convolucionales) con *Batch
Normalization* y *Dropout2d* en cada bloque, en dos variantes que
contrastan el cabezal denso clásico contra *Global Average Pooling*.

## Archivos

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Constantes centralizadas: rutas, hiperparámetros, seed |
| `dataset.py` | Transforms, DataLoaders con split 80/20, class_weights |
| `architectures.py` | `DeepCNN_BN` y `DeepCNN_BN_GAP` con factory `get_model()` |
| `utils.py` | Seed, device, EarlyStopping, métricas, plots |
| `train.py` | CLI de entrenamiento (`--arch deep\|deep_gap`) |
| `evaluate.py` | CLI de evaluación sobre holdout Testing |
| `compare.py` | Genera comparativa deep vs deep_gap |
| `requirements.txt` | Dependencias específicas del Modelo 2 |
| `checkpoints/` | Pesos entrenados (gitignored) |

## Setup

Desde la raíz del proyecto (`proyecto-final/`):

```bash
# 1. Crear venv con Python 3.11
python3.11 -m venv .venv

# 2. Activar venv
source .venv/bin/activate

# 3. Instalar dependencias del Modelo 2
pip install --upgrade pip
pip install -r models/model2/requirements.txt
```

## Ejecución paso a paso

Desde la raíz del proyecto, con el venv activado:

```bash
# Entrenar DeepCNN-BN (cabezal FC)
python models/model2/train.py --arch deep

# Entrenar DeepCNN-BN-GAP (Global Average Pooling)
python models/model2/train.py --arch deep_gap

# Evaluar sobre Testing (holdout)
python models/model2/evaluate.py --arch deep
python models/model2/evaluate.py --arch deep_gap

# Generar comparativa final
python models/model2/compare.py
```

## Outputs generados

```
models/model2/checkpoints/
├── deep_best.pth              # Mejor checkpoint DeepCNN-BN (gitignored)
└── deep_gap_best.pth          # Mejor checkpoint DeepCNN-BN-GAP (gitignored)

outputs/figures/model-2/
├── curvas_deep.png                       # Loss, Acc, F1, LR por época
├── confusion_deep.png                    # Matriz confusión normalizada
├── curvas_deep_gap.png
├── confusion_deep_gap.png
└── comparacion_deep_vs_deepgap.png       # Comparativa lado a lado

outputs/results/model-2/
├── deep_metrics.csv                      # Métricas por época
├── deep_test_report.txt                  # classification_report sklearn
├── deep_gap_metrics.csv
├── deep_gap_test_report.txt
└── comparison_summary.txt                # Tabla comparativa final
```

## Arquitectura del Modelo 2 (para reporte LaTeX §III-C)

Backbone VGG-style común a ambas variantes (entrada $(B, 3, 224, 224)$):

| # | Bloque | Filtros | Capas internas | Salida $(C,H,W)$ |
|---|---|---|---|---|
| 1 | Conv Block 1 | 32  | 2× [Conv3×3→BN→ReLU] + MaxPool + Dropout2d(0.10) | $(32,112,112)$ |
| 2 | Conv Block 2 | 64  | 2× [Conv3×3→BN→ReLU] + MaxPool + Dropout2d(0.15) | $(64, 56, 56)$ |
| 3 | Conv Block 3 | 128 | 2× [Conv3×3→BN→ReLU] + MaxPool + Dropout2d(0.20) | $(128, 28, 28)$ |
| 4 | Conv Block 4 | 256 | 2× [Conv3×3→BN→ReLU] + MaxPool + Dropout2d(0.25) | $(256, 14, 14)$ |
| 5 | Conv Block 5 | 512 | 2× [Conv3×3→BN→ReLU] + Dropout2d(0.30) (sin pool) | $(512, 14, 14)$ |

Cabezales:

| Variante | Cabezal | Salida |
|---|---|---|
| `deep` (DeepCNN-BN) | AdaptiveAvgPool2d(2,2) → Flatten(2048) → Linear(2048,256) → BN1d → ReLU → Dropout(0.5) → Linear(256,4) | $(B,4)$ |
| `deep_gap` (DeepCNN-BN-GAP) | AdaptiveAvgPool2d(1,1) → Flatten(512) → Dropout(0.5) → Linear(512,4) | $(B,4)$ |

**Parámetros entrenables:**
* DeepCNN-BN: **5,240,292**
* DeepCNN-BN-GAP: **4,716,260** (~524K menos: ahorra el FC denso)

> El bloque 5 omite MaxPool intencionalmente. El motivo es que
> `AdaptiveAvgPool2d((2,2))` exige bajo backend MPS que la entrada sea
> divisible por la salida (14/2=7 ✓; 7/2 ✗). Manteniendo 14×14 a la
> salida del backbone, ambas variantes funcionan sin transferir a CPU.

## Hiperparámetros (para reporte LaTeX §IV-A)

| Hiperparámetro | Valor |
|---|---|
| Optimizador | AdamW (betas=0.9, 0.999) |
| Learning rate inicial | 5e-4 |
| Weight decay | 1e-4 |
| Scheduler | ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6) |
| Loss | CrossEntropyLoss(weight=class_weights) |
| Batch size | 64 |
| Epochs máx | 50 |
| Early stopping | patience=8 sobre F1-macro val |
| Split | 80/20 sobre Training |
| Seed | 42 |

**Justificación frente al Modelo 1:** OkanNet (la variante con BN del
Modelo 1) colapsó con lr=1e-3 y batch=32 porque BatchNorm es sensible
a ese régimen: con *batch* pequeño las estadísticas por *mini-batch*
son ruidosas, y un lr alto amplifica esa inestabilidad. Para el Modelo
2 se baja lr a 5e-4 y se sube batch a 64 para dar a BN un régimen
estable, y se añade *weight decay* (1e-4) compensando la regularización
implícita que perdimos al pasar de SGD-momentum a Adam. AdamW separa
*weight decay* del gradiente, evitando el sesgo que tiene Adam clásico
con `weight_decay`.

## Tiempos esperados

| Hardware | Tiempo por arquitectura | Tiempo total (ambas + eval + compare) |
|---|---|---|
| Mac con MPS (M1/M2/M3) | ~25-40 min | ~60-90 min |
| CPU puro | ~2-3 h | ~5-7 h |
| Colab GPU T4 | ~8-15 min | ~25-40 min |

Modelo 2 es ~10× más profundo que el baseline (10 conv vs 3) por lo
que los tiempos son aproximadamente el doble que en Modelo 1.
