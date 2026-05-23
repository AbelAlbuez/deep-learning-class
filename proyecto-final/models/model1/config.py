"""Constantes centralizadas para el Modelo 1 (CNNBaseline + OkanNet)."""
from __future__ import annotations

from pathlib import Path

# Rutas
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(
    "/Users/abelalbuez/Documents/Maestria/Tercer Semestre/"
    "Aprendizaje Profundo/deep-learning-class/proyecto-final/datasets"
)
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "model1" / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "model-1"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "model-1"

# Reproducibilidad
SEED = 42

# Clases (orden alfabético que usa ImageFolder)
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
NUM_CLASSES = 4

# Preprocesamiento
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Hiperparámetros
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 40
DROPOUT = 0.5
EARLY_STOPPING_PATIENCE = 7
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# Split
VAL_SPLIT = 0.2

# DataLoader
NUM_WORKERS = 4
