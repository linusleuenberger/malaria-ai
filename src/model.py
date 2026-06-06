# ============================================================
# src/model.py
# KI-Architektur – flexibles Transfer Learning
# ============================================================

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torchvision import models

from src.config import (
    ARCHITECTURE,
    DEVICE,
    DROPOUT_RATE,
    FINAL_MODEL_PATH,
    FREEZE_BACKBONE,
    HIDDEN_SIZE,
    LABEL_SMOOTHING,
    NUM_CLASSES,
)

logger = logging.getLogger(__name__)


# ── Unterstützte Architekturen ────────────────────────────────
SUPPORTED_MODELS: Dict[str, object] = {
    "resnet50"       : models.resnet50,
    "resnet101"      : models.resnet101,
    "efficientnet_b0": models.efficientnet_b0,
}


# ── Modell bauen ──────────────────────────────────────────────
def build_model(
    architecture:    str   = ARCHITECTURE,
    num_classes:     int   = NUM_CLASSES,
    freeze_backbone: bool  = FREEZE_BACKBONE,
    dropout_rate:    float = DROPOUT_RATE,
    hidden_size:     int   = HIDDEN_SIZE,
) -> nn.Module:
    """
    Vortrainiertes Modell laden und für Malaria-Klassifikation anpassen.

    Args:
        architecture    : "resnet50", "resnet101" oder "efficientnet_b0"
        num_classes     : Anzahl Klassen (2 oder 5)
        freeze_backbone : Vortrainierte Layer einfrieren
        dropout_rate    : Dropout Wahrscheinlichkeit
        hidden_size     : Grösse des mittleren Layers

    Returns:
        Modell auf DEVICE verschoben
    """

    if architecture not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unbekannte Architektur: '{architecture}'\n"
            f"Optionen: {list(SUPPORTED_MODELS.keys())}"
        )

    # ── 1. Vortrainiertes Modell laden ────────────────────────
    model_fn = SUPPORTED_MODELS[architecture]
    model    = model_fn(weights="DEFAULT")
    logger.info(f"Modell geladen: {architecture}")

    # ── 2. Backbone einfrieren ────────────────────────────────
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        _set_bn_eval(model)
        logger.info("Backbone eingefroren.")

    # ── 3. Letzten Layer ersetzen ─────────────────────────────
    in_features = _get_in_features(model, architecture)

    new_head = nn.Sequential(
        nn.Linear(in_features, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(hidden_size, num_classes)
    )

    if "resnet" in architecture:
        model.fc = new_head
    elif "efficientnet" in architecture:
        model.classifier = new_head

    # ── 4. Auf GPU/CPU verschieben ────────────────────────────
    model = model.to(DEVICE)
    _log_parameter_count(model)

    return model


# ── Loss Funktion ─────────────────────────────────────────────
def get_loss_function() -> nn.Module:
    """
    CrossEntropyLoss mit Label Smoothing.
    Label Smoothing macht das Modell robuster gegen
    unsichere oder falsch gelabelte Bilder.
    """
    return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


# ── Schrittweises Auftauen ────────────────────────────────────
def unfreeze_layers(model: nn.Module, n_layers: int) -> None:
    """
    Letzte n_layers Layer trainierbar machen.

    Empfohlene Reihenfolge:
        Schritt 1: freeze_backbone=True  → nur Head trainieren
        Schritt 2: unfreeze_layers(20)   → letzte 20 Layer auftauen
        Schritt 3: unfreeze_layers(50)   → noch mehr auftauen

    Args:
        model    : Das Modell
        n_layers : Anzahl Layer die aufgetaut werden sollen
    """
    params = list(model.parameters())
    for param in params[-n_layers:]:
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    logger.info(
        f"Aufgetaut: letzte {n_layers} Layer "
        f"→ {trainable:,} trainierbare Parameter"
    )


# ── Modell speichern ──────────────────────────────────────────
def save_model(
    model:      nn.Module,
    path:       object = FINAL_MODEL_PATH,
    extra_info: Optional[dict] = None,
) -> None:
    """
    Modell speichern inkl. Metadaten.

    Args:
        model      : Das trainierte Modell
        path       : Speicherpfad
        extra_info : Zusätzliche Infos (Epoch, Accuracy etc.)
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture"    : ARCHITECTURE,
        "num_classes"     : NUM_CLASSES,
    }

    if extra_info:
        checkpoint.update(extra_info)

    torch.save(checkpoint, path)
    logger.info(f"Modell gespeichert: {path}")


# ── Modell laden ──────────────────────────────────────────────
def load_model(path: object = FINAL_MODEL_PATH) -> nn.Module:
    """
    Gespeichertes Modell laden.

    Args:
        path : Pfad zum gespeicherten Checkpoint

    Returns:
        Modell im eval() Modus auf DEVICE
    """
    checkpoint = torch.load(path, map_location=DEVICE)

    model = build_model(
        architecture    = checkpoint.get("architecture", ARCHITECTURE),
        num_classes     = checkpoint.get("num_classes",  NUM_CLASSES),
        freeze_backbone = False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(f"Modell geladen von: {path}")
    return model


# ── Hilfsfunktionen ───────────────────────────────────────────
def _get_in_features(model: nn.Module, architecture: str) -> int:
    """Gibt die Eingabegrösse des letzten Layers zurück."""
    if "resnet" in architecture:
        return model.fc.in_features
    elif "efficientnet" in architecture:
        return model.classifier[1].in_features
    raise ValueError(f"Unbekannte Architektur: {architecture}")


def _set_bn_eval(model: nn.Module) -> None:
    """
    BatchNorm Layer in eval Modus setzen wenn Backbone eingefroren.
    Verhindert dass BatchNorm Statistiken sich verändern.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def _log_parameter_count(model: nn.Module) -> None:
    """Logt Anzahl trainierbare und eingefrorene Parameter."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    logger.info(f"Parameter gesamt:      {total:>12,}")
    logger.info(f"Parameter trainierbar: {trainable:>12,}")
    logger.info(f"Parameter eingefroren: {total - trainable:>12,}")


# ── Quick-Test: python -m src.model ──────────────────────────
if __name__ == "__main__":
    model = build_model()
    print(model)

    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    out   = model(dummy)
    print(f"\nEingabe:  {list(dummy.shape)}")
    print(f"Ausgabe:  {list(out.shape)}")
    print(f"Klassen:  {NUM_CLASSES}")
    print(f"Loss:     {get_loss_function()}")
    print("\n✓ model.py funktioniert korrekt.")