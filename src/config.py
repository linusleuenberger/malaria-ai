# ============================================================
# src/config.py
# Zentrale Konfigurationsdatei – alle Einstellungen an einem Ort
# ============================================================

from __future__ import annotations

import logging
import torch
from pathlib import Path
from typing import Dict, Tuple

# ── Projektpfade ──────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent

DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUGMENTED_DIR = DATA_DIR / "augmented"

MODEL_DIR      = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
FINAL_DIR      = MODEL_DIR / "final"

RESULTS_DIR     = BASE_DIR / "results"
PLOTS_DIR       = RESULTS_DIR / "plots"
METRICS_DIR     = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

LOG_FILE = RESULTS_DIR / "training.log"

# Alle nötigen Ordner automatisch erstellen
_DIRS_TO_CREATE = [
    RAW_DIR / "healthy",
    RAW_DIR / "infected",
    PROCESSED_DIR / "train" / "healthy",
    PROCESSED_DIR / "train" / "infected",
    PROCESSED_DIR / "val"   / "healthy",
    PROCESSED_DIR / "val"   / "infected",
    PROCESSED_DIR / "test"  / "healthy",
    PROCESSED_DIR / "test"  / "infected",
    AUGMENTED_DIR,
    CHECKPOINT_DIR,
    FINAL_DIR,
    PLOTS_DIR,
    METRICS_DIR,
    PREDICTIONS_DIR,
]

for _d in _DIRS_TO_CREATE:
    _d.mkdir(parents=True, exist_ok=True)

# ── Logging Setup ─────────────────────────────────────────────
LOG_LEVEL = logging.INFO

logging.basicConfig(
    level    = LOG_LEVEL,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Config geladen – Projektpfade initialisiert.")

# ── Bildverarbeitung ──────────────────────────────────────────
IMAGE_SIZE: Tuple[int, int] = (224, 224)
CHANNELS:   int             = 3

# Nach compute_dataset_stats() aus dataset.py durch eigene Werte ersetzen
# Bis dahin: ImageNet-Werte als Startwert
MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
STD:  Tuple[float, ...] = (0.229, 0.224, 0.225)

# ── Klassen ───────────────────────────────────────────────────
# 2 Klassen für binäre Klassifikation (healthy / infected)
# Für 5 Klassen später einfach erweitern:
# CLASS_NAMES = ["healthy", "falciparum", "vivax", "ovale", "malariae"]
CLASS_NAMES:  list[str]       = ["healthy", "infected"]
NUM_CLASSES:  int             = len(CLASS_NAMES)
CLASS_TO_IDX: Dict[str, int]  = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str]  = {i: name for i, name in enumerate(CLASS_NAMES)}

# ── Dataset Aufteilung ────────────────────────────────────────
TRAIN_SPLIT: float = 0.70
VAL_SPLIT:   float = 0.15
TEST_SPLIT:  float = 0.15
assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6, \
    "Splits müssen zusammen 1.0 ergeben!"

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE:    int   = 128  # RTX 5070 Ti: 128-256 optimal fuer ResNet50
EPOCHS:        int   = 50
LEARNING_RATE: float = 0.001
WEIGHT_DECAY:  float = 1e-4

# Gradient Clipping – verhindert explodierende Gradienten
MAX_GRAD_NORM: float = 1.0

# Label Smoothing – macht Modell robuster gegen unsichere Labels
LABEL_SMOOTHING: float = 0.1
# 0.0 = kein Smoothing, 0.1 = leichtes Smoothing

# Early Stopping
EARLY_STOPPING_PATIENCE: int = 10

# Learning Rate Scheduler
LR_SCHEDULER_PATIENCE: int   = 5
LR_SCHEDULER_FACTOR:   float = 0.5

# Experiment Tracking (optional, für später)
USE_WANDB:     bool = False
WANDB_PROJECT: str  = "malaria-ai"

# ── Modell ────────────────────────────────────────────────────
ARCHITECTURE:    str   = "resnet50"
# Optionen: "resnet50", "resnet101", "efficientnet_b0"

FREEZE_BACKBONE: bool  = True   # Backbone wird ab Epoch 5 schrittweise aufgetaut
DROPOUT_RATE:    float = 0.4
HIDDEN_SIZE:     int   = 256 if NUM_CLASSES <= 2 else 512

# ── Hardware ──────────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Gerät: {DEVICE.upper()}")

# ── Modellpfade ───────────────────────────────────────────────
BEST_MODEL_PATH:  Path = CHECKPOINT_DIR / "best_model.pth"
FINAL_MODEL_PATH: Path = FINAL_DIR      / "final_model.pth"

# ── Reproduzierbarkeit ────────────────────────────────────────
RANDOM_SEED: int = 42