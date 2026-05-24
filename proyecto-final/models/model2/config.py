"""Constantes centralizadas para el Modelo 2 (DeepCNN-BN + DeepCNN-BN-GAP).

Aprendizajes del Modelo 1 (Abel) aplicados aquí:
  * OkanNet (BN sobre baseline) colapsó con lr=1e-3 y batch=32. El Modelo 2
    usa lr=5e-4 y batch=64 para darle a BatchNorm un régimen estable.
  * El cabezal FC del baseline concentra >99% de los parámetros. La variante
    DeepCNN-BN-GAP sustituye el FC gigante por Global Average Pooling.
"""
from __future__ import annotations

from pathlib import Path

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(
    "/Users/danielfeliperioscaro/Documents/Universidad/"
    "deep-learning-class/proyecto-final/datasets"
)
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "model2" / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "model-2"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "model-2"

# Reproducibilidad
SEED = 42

# Clases (orden alfabético que usa ImageFolder)
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
NUM_CLASSES = 4

# Preprocesamiento
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Hiperparámetros (ajustados para estabilidad con BatchNorm)
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
NUM_EPOCHS = 50
DROPOUT_FC = 0.5
DROPOUT2D_RATES = (0.10, 0.15, 0.20, 0.25, 0.30)  # uno por bloque conv
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 8
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# Split
VAL_SPLIT = 0.2

# DataLoader
NUM_WORKERS = 4
