# ============================================================
# src/config.py
# Zentrale Konfiguration - alle Einstellungen an einem Ort.
# ============================================================
#
# Hier stehen alle Pfade und Hyperparameter. Andere Module
# importieren nur aus dieser Datei, damit es keine doppelten
# oder widerspruechlichen Werte gibt.
# ============================================================

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

# ── Projektpfade ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"        # train / val / test (richtiger Split)
AUGMENTED_DIR = DATA_DIR / "augmented"        # optional, offline augmentiert

MODEL_DIR      = BASE_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"    # Zwischenstaende (resume)
FINAL_DIR      = MODEL_DIR / "final"          # bestes Modell + Statistiken

RESULTS_DIR     = BASE_DIR / "results"
PLOTS_DIR       = RESULTS_DIR / "plots"
METRICS_DIR     = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Nur die wirklich benoetigten Ausgabeordner anlegen.
for _d in (CHECKPOINT_DIR, FINAL_DIR, PLOTS_DIR, METRICS_DIR, PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Checkpoint-Pfade ──────────────────────────────────────────
# last.pth  : nach JEDER Epoche ueberschrieben -> Training fortsetzbar
# best.pth  : nur bei neuer Bestleistung ueberschrieben
# final     : am Ende eine Kopie des besten Modells (zum Testen/Deployen)
LAST_CKPT_PATH   = CHECKPOINT_DIR / "last.pth"
BEST_CKPT_PATH   = CHECKPOINT_DIR / "best.pth"
FINAL_MODEL_PATH = FINAL_DIR / "final_model.pth"

# Alte Namenskompatibilitaet (frueher hiess das best_model.pth)
BEST_MODEL_PATH = BEST_CKPT_PATH

# ── Bildverarbeitung ──────────────────────────────────────────
IMAGE_SIZE: Tuple[int, int] = (224, 224)
CHANNELS: int = 3

# Normalisierungswerte: falls dataset_stats.json existiert (aus dem
# Preprocessing), werden die echten Werte des Datensatzes genutzt.
# Sonst ImageNet-Standardwerte (gut fuer vortrainierte Modelle).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _load_dataset_stats() -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    stats_file = METRICS_DIR / "dataset_stats.json"
    if stats_file.exists():
        try:
            data = json.loads(stats_file.read_text())
            mean = tuple(data["mean"])
            std  = tuple(data["std"])
            if len(mean) == 3 and len(std) == 3:
                return mean, std
        except Exception as exc:  # pragma: no cover - defensiv
            logger.warning("dataset_stats.json unlesbar (%s) - nutze ImageNet.", exc)
    return _IMAGENET_MEAN, _IMAGENET_STD


MEAN, STD = _load_dataset_stats()

# ── Klassen ───────────────────────────────────────────────────
CLASS_NAMES = ["healthy", "infected"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# ── Datensatz-Split (nur fuer das Preprocessing relevant) ─────
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.70, 0.15, 0.15
assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6

# ── Modell ────────────────────────────────────────────────────
ARCHITECTURE = "resnet50"          # "resnet50" | "resnet101" | "efficientnet_b0"
FREEZE_BACKBONE = True             # Start: nur Kopf trainieren
DROPOUT_RATE = 0.4
HIDDEN_SIZE = 256

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE = 256                   # RTX 5070 Ti (16 GB): 256 lastet die GPU gut aus
EPOCHS = 40
LEARNING_RATE = 1e-3               # Lernrate fuer den neuen Kopf
BACKBONE_LR_FACTOR = 0.1           # Backbone lernt 10x langsamer
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1              # robuster gegen unsichere Labels
MAX_GRAD_NORM = 1.0                # Gradient Clipping

WARMUP_EPOCHS = 3                  # LR faehrt sanft hoch -> stabilerer Start
UNFREEZE_AT_EPOCH = 3             # ab hier Backbone schrittweise auftauen
UNFREEZE_N_LAYERS = 60            # wie viele der letzten Layer auftauen
EARLY_STOPPING_PATIENCE = 8       # Epochen ohne Verbesserung -> Stop

# Exponential Moving Average der Gewichte. Standardmaessig AUS: bei wenigen
# Schritten pro Epoche hinkt der EMA-Mittelwert stark hinterher und macht die
# Validierung unbrauchbar. Nur einschalten, wenn bewusst gewuenscht.
USE_EMA = False
EMA_DECAY = 0.999

# ── Hardware / Performance ────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 12                   # Ryzen 5 7600X: 12 Threads -> 12 Worker
PIN_MEMORY = (DEVICE == "cuda")
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

USE_AMP = (DEVICE == "cuda")       # Mixed Precision
AMP_DTYPE = torch.bfloat16         # bf16 ist auf Blackwell (50xx) stabil & schnell
CHANNELS_LAST = True               # bessere Tensor-Core-Auslastung
# torch.compile nur, wenn Triton verfuegbar ist (unter Windows meist nicht).
# Sonst wuerde der erste Forward-Pass mit "TritonMissing" abbrechen.
USE_COMPILE = (DEVICE == "cuda") and importlib.util.find_spec("triton") is not None

# ── Reproduzierbarkeit ────────────────────────────────────────
RANDOM_SEED = 42
