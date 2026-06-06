# ============================================================
# src/utils.py
# Kleine Hilfsfunktionen: Seed, Performance-Setup, Checks.
# ============================================================

from __future__ import annotations

import logging
import random

import numpy as np
import torch

from src.config import DEVICE, NUM_CLASSES, RANDOM_SEED

logger = logging.getLogger(__name__)


# ── Reproduzierbarkeit ────────────────────────────────────────
def set_seed(seed: int = RANDOM_SEED) -> None:
    """Alle Zufallsquellen auf denselben Wert setzen."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info("Seed gesetzt: %d", seed)


# ── Performance-/Determinismus-Schalter ───────────────────────
def setup_perf(deterministic: bool = False) -> None:
    """
    GPU-Verhalten einstellen.

    deterministic=False (Standard): cuDNN sucht die schnellsten Kernel
        -> beste Auslastung der GPU, minimale zufaellige Schwankungen.
    deterministic=True: exakt reproduzierbar (fuer einen Referenzlauf),
        dafuer langsamer.
    """
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    # TF32: schnellere Matrix-Multiplikationen auf modernen NVIDIA-GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("Performance-Setup: %s",
                "deterministisch (reproduzierbar)" if deterministic
                else "schnell (cudnn.benchmark an)")


# ── Hardware-Info ─────────────────────────────────────────────
def print_system_info() -> None:
    logger.info("PyTorch %s | CUDA verfuegbar: %s | Geraet: %s",
                torch.__version__, torch.cuda.is_available(), DEVICE.upper())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s (%.1f GB VRAM)", props.name, props.total_memory / 1e9)


# ── Sanity Check vor dem Training ─────────────────────────────
def sanity_check(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> bool:
    """Ein Batch durchschicken und auf grobe Fehler pruefen."""
    try:
        images, labels = next(iter(loader))
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model.eval()
        with torch.no_grad():
            out = model(images)
        if torch.isnan(out).any():
            logger.error("Sanity Check: NaN im Output."); return False
        if out.shape[1] != NUM_CLASSES:
            logger.error("Sanity Check: %d statt %d Klassen.", out.shape[1], NUM_CLASSES)
            return False
        if labels.min() < 0 or labels.max() >= NUM_CLASSES:
            logger.error("Sanity Check: Labels ausserhalb des gueltigen Bereichs.")
            return False
        logger.info("Sanity Check bestanden (Batch %s).", tuple(images.shape))
        return True
    except Exception as exc:
        logger.error("Sanity Check fehlgeschlagen: %s", exc)
        return False


# ── Zeitformat ────────────────────────────────────────────────
def format_time(seconds: float) -> str:
    s = int(seconds)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m else f"{s}s"
