"""Configuration for Model3 (MRIResNet with SE attention)."""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Override with environment variable: export DATASET_DIR=/path/to/datasets
DATA_DIR = Path(os.environ.get("DATASET_DIR", str(PROJECT_ROOT / "datasets")))
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

CHECKPOINTS_DIR = PROJECT_ROOT / "models" / "model3" / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "model-3"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results" / "model-3"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Classes (alphabetical — matches ImageFolder order)
# ---------------------------------------------------------------------------
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
NUM_CLASSES = 4

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
IN_CHANNELS = 5  # R, G, B, CLAHE, Sobel

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
DROPOUT = 0.5
SE_REDUCTION = 8  # SE bottleneck ratio; keeps ≥4 hidden units at 32 channels

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
MIN_LR = 1e-6

# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------
VAL_SPLIT = 0.2
NUM_WORKERS = 4
