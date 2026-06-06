# ============================================================
# src/model.py
# Modell-Architektur (Transfer Learning) + Speichern/Laden.
# ============================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from src.config import (
    ARCHITECTURE,
    CHANNELS_LAST,
    DEVICE,
    DROPOUT_RATE,
    FREEZE_BACKBONE,
    HIDDEN_SIZE,
    LABEL_SMOOTHING,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)

_SUPPORTED = {
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "efficientnet_b0": models.efficientnet_b0,
}


# ── Modell bauen ──────────────────────────────────────────────
def build_model(
    architecture: str = ARCHITECTURE,
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = FREEZE_BACKBONE,
    dropout_rate: float = DROPOUT_RATE,
    hidden_size: int = HIDDEN_SIZE,
    pretrained: bool = True,
) -> nn.Module:
    """Vortrainiertes CNN laden und den Klassifikationskopf ersetzen."""
    if architecture not in _SUPPORTED:
        raise ValueError(f"Unbekannte Architektur '{architecture}'. "
                         f"Optionen: {list(_SUPPORTED)}")

    model = _SUPPORTED[architecture](weights="DEFAULT" if pretrained else None)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    in_features = _in_features(model, architecture)
    head = nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, num_classes),
    )
    if "resnet" in architecture:
        model.fc = head
    else:  # efficientnet
        model.classifier = head

    model = model.to(DEVICE)
    if CHANNELS_LAST and DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)

    _log_params(model)
    return model


def get_loss_function() -> nn.Module:
    """CrossEntropy mit Label Smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


def unfreeze_layers(model: nn.Module, n_layers: int) -> int:
    """Die letzten n_layers Parameter-Tensoren wieder trainierbar machen."""
    params = list(model.parameters())
    for p in params[-n_layers:]:
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Aufgetaut: letzte %d Layer -> %s trainierbare Parameter",
                n_layers, f"{trainable:,}")
    return trainable


# ── Speichern / Laden ─────────────────────────────────────────
def save_checkpoint(state: dict, path: Path) -> None:
    """Beliebigen Trainings-Zustand speichern (atomar)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_model(path: str | Path, device: str = DEVICE) -> nn.Module:
    """
    Modell fuer Inferenz/Evaluation laden.

    Funktioniert mit allen Checkpoint-Formaten dieses Projekts
    (last/best/final). Bevorzugt die EMA-Gewichte, falls vorhanden,
    da diese in der Regel zuverlaessiger sind.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        arch = ckpt.get("architecture", ARCHITECTURE)
        ncls = ckpt.get("num_classes", NUM_CLASSES)
        state = ckpt.get("ema") or ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:  # altes Format
        arch = ckpt.get("architecture", ARCHITECTURE)
        ncls = ckpt.get("num_classes", NUM_CLASSES)
        state = ckpt["model_state_dict"]
    else:  # reines state_dict
        arch, ncls, state = ARCHITECTURE, NUM_CLASSES, ckpt

    model = build_model(architecture=arch, num_classes=ncls,
                        freeze_backbone=False)
    model.load_state_dict(state)
    model.eval()
    logger.info("Modell geladen: %s", path)
    return model


# ── Hilfsfunktionen ───────────────────────────────────────────
def _in_features(model: nn.Module, architecture: str) -> int:
    if "resnet" in architecture:
        return model.fc.in_features
    return model.classifier[1].in_features


def _log_params(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameter gesamt: %s | trainierbar: %s | eingefroren: %s",
                f"{total:,}", f"{trainable:,}", f"{total - trainable:,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = build_model()
    dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
    if CHANNELS_LAST and DEVICE == "cuda":
        dummy = dummy.to(memory_format=torch.channels_last)
    print("Output:", tuple(m(dummy).shape), "| Loss:", get_loss_function())
    print("[OK] model.py")
