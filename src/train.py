# ============================================================
# src/train.py
# Trainingsloop mit Early Stopping, LR Scheduler & Logging
# ============================================================

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.config import (
    BEST_MODEL_PATH,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    MAX_GRAD_NORM,
    RANDOM_SEED,
    USE_WANDB,
    WANDB_PROJECT,
    WEIGHT_DECAY,
)
from src.model import build_model, get_loss_function, save_model, unfreeze_layers

logger = logging.getLogger(__name__)


# ── Wandb Setup ───────────────────────────────────────────────
def _init_wandb(config: dict) -> None:
    """Wandb initialisieren falls aktiviert."""
    if USE_WANDB:
        import wandb
        wandb.init(
            project = WANDB_PROJECT,
            config  = config,
        )
        logger.info("Wandb initialisiert.")


def _log_wandb(metrics: dict) -> None:
    """Metriken zu Wandb loggen falls aktiviert."""
    if USE_WANDB:
        import wandb
        wandb.log(metrics)


# ── Early Stopping ────────────────────────────────────────────
class EarlyStopping:
    """
    Stoppt Training wenn Validierungs-Loss sich nicht verbessert.

    Args:
        patience : Anzahl Epochen ohne Verbesserung bevor gestoppt wird
        delta    : Minimale Verbesserung die zählt
    """

    def __init__(
        self,
        patience: int   = EARLY_STOPPING_PATIENCE,
        delta:    float = 1e-4,
    ) -> None:
        self.patience  = patience
        self.delta     = delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.stop      = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            # Verbesserung → Counter zurücksetzen
            self.best_loss = val_loss
            self.counter   = 0
        else:
            # Keine Verbesserung → Counter erhöhen
            self.counter += 1
            logger.info(
                f"Early Stopping: {self.counter}/{self.patience} "
                f"Epochen ohne Verbesserung"
            )
            if self.counter >= self.patience:
                self.stop = True
                logger.info("Early Stopping ausgelöst – Training gestoppt.")

        return self.stop


# ── Ein Epoch trainieren ──────────────────────────────────────
def _train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    scaler:     GradScaler,
) -> Tuple[float, float]:
    """
    Einen einzelnen Trainingsdurchlauf ausführen.
    Verwendet Mixed Precision Training auf GPU für mehr Geschwindigkeit.

    Returns:
        (loss, accuracy) für diese Epoch
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        # ── Mixed Precision ────────────────────────────────
        # autocast: float16 auf GPU → schneller & weniger RAM
        # autocast: float32 auf CPU → kein Unterschied
        with autocast(enabled=(DEVICE == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        # ── Rückwärtsdurchlauf mit Scaler ──────────────────
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        # ── Statistiken ────────────────────────────────────
        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct   / total

    return avg_loss, accuracy


# ── Ein Epoch validieren ──────────────────────────────────────
def _validate_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """
    Einen einzelnen Validierungsdurchlauf ausführen.

    Returns:
        (loss, accuracy) für diese Epoch
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with autocast(enabled=(DEVICE == "cuda")):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct   / total

    return avg_loss, accuracy


# ── Haupttraining ─────────────────────────────────────────────
def train(
    model:              nn.Module,
    train_loader:       DataLoader,
    val_loader:         DataLoader,
    epochs:             int   = EPOCHS,
    lr:                 float = LEARNING_RATE,
    unfreeze_at_epoch:  int   = 10,
    unfreeze_n_layers:  int   = 20,
) -> Dict[str, list]:
    """
    Vollständiger Trainingsloop mit:
    - Mixed Precision Training
    - Gradual Unfreezing
    - Early Stopping
    - LR Scheduling
    - Wandb Logging

    Args:
        model             : Das zu trainierende Modell
        train_loader      : DataLoader für Trainingsdaten
        val_loader        : DataLoader für Validierungsdaten
        epochs            : Maximale Anzahl Epochen
        lr                : Lernrate
        unfreeze_at_epoch : Nach welcher Epoch Backbone auftauen
        unfreeze_n_layers : Wie viele Layer auftauen

    Returns:
        history: Dict mit train_loss, val_loss, train_acc, val_acc
    """

    # Reproduzierbarkeit
    torch.manual_seed(RANDOM_SEED)

    # ── Wandb initialisieren ───────────────────────────────
    _init_wandb({
        "epochs"            : epochs,
        "learning_rate"     : lr,
        "batch_size"        : train_loader.batch_size,
        "weight_decay"      : WEIGHT_DECAY,
        "max_grad_norm"     : MAX_GRAD_NORM,
        "unfreeze_at_epoch" : unfreeze_at_epoch,
        "unfreeze_n_layers" : unfreeze_n_layers,
    })

    # ── Optimizer, Loss, Scheduler ─────────────────────────
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = lr,
        weight_decay = WEIGHT_DECAY,
    )
    criterion = get_loss_function()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        patience = LR_SCHEDULER_PATIENCE,
        factor   = LR_SCHEDULER_FACTOR,
    )
    early_stopping = EarlyStopping()

    # Mixed Precision Scaler – nur aktiv auf GPU
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # Verlauf speichern
    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss"  : [],
        "train_acc" : [],
        "val_acc"   : [],
    }

    best_val_loss = float("inf")

    logger.info("=" * 60)
    logger.info("Training gestartet")
    logger.info(f"Epochen:           {epochs}")
    logger.info(f"Lernrate:          {lr}")
    logger.info(f"Batch-Size:        {train_loader.batch_size}")
    logger.info(f"Mixed Precision:   {DEVICE == 'cuda'}")
    logger.info(f"Wandb:             {USE_WANDB}")
    logger.info(f"Unfreeze Epoch:    {unfreeze_at_epoch}")
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # ── Gradual Unfreezing ─────────────────────────────
        if epoch == unfreeze_at_epoch:
            unfreeze_layers(model, unfreeze_n_layers)
            # Optimizer neu erstellen mit aufgetauten Layern
            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr           = lr * 0.1,  # Kleinere LR für aufgetaute Layer
                weight_decay = WEIGHT_DECAY,
            )
            logger.info(
                f"Epoch {epoch}: Backbone aufgetaut – "
                f"letzte {unfreeze_n_layers} Layer trainierbar"
            )

        # ── Training ──────────────────────────────────────
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )

        # ── Validierung ───────────────────────────────────
        val_loss, val_acc = _validate_one_epoch(
            model, val_loader, criterion
        )

        # ── LR Scheduler ──────────────────────────────────
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Verlauf speichern ─────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"]  .append(val_loss)
        history["train_acc"] .append(train_acc)
        history["val_acc"]   .append(val_acc)

        # ── Wandb Logging ──────────────────────────────────
        _log_wandb({
            "epoch"     : epoch,
            "train_loss": train_loss,
            "train_acc" : train_acc,
            "val_loss"  : val_loss,
            "val_acc"   : val_acc,
            "lr"        : current_lr,
        })

        # ── Bestes Modell speichern ────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model,
                path       = BEST_MODEL_PATH,
                extra_info = {
                    "epoch"   : epoch,
                    "val_loss": val_loss,
                    "val_acc" : val_acc,
                }
            )
            logger.info(
                f"  → Neues bestes Modell gespeichert "
                f"(Val-Loss: {val_loss:.4f})"
            )

        # ── Epoch Zusammenfassung ─────────────────────────
        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
            f"LR: {current_lr:.6f} | "
            f"Zeit: {elapsed:.1f}s"
        )

        # ── Early Stopping prüfen ─────────────────────────
        if early_stopping(val_loss):
            logger.info(f"Training nach Epoch {epoch} gestoppt.")
            break

    logger.info("=" * 60)
    logger.info("Training abgeschlossen.")
    logger.info(f"Bester Val-Loss: {best_val_loss:.4f}")
    logger.info("=" * 60)

    if USE_WANDB:
        import wandb
        wandb.finish()

    return history


# ── Quick-Test: python -m src.train ──────────────────────────
if __name__ == "__main__":
    from src.config  import PROCESSED_DIR
    from src.dataset import get_dataloaders

    loaders = get_dataloaders(
        data_dir    = PROCESSED_DIR,
        batch_size  = 4,
        num_workers = 0,
        pin_memory  = False,
    )

    model = build_model()

    history = train(
        model             = model,
        train_loader      = loaders["train"],
        val_loader        = loaders["val"],
        epochs            = 2,
        unfreeze_at_epoch = 2,
        unfreeze_n_layers = 10,
    )

    print("\nVerlauf:")
    for key, values in history.items():
        print(f"  {key}: {[round(v, 4) for v in values]}")

    print("\n✓ train.py funktioniert korrekt.")