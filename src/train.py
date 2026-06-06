# ============================================================
# src/train.py
# Trainingsloop mit:
#   - sauberer Terminal-Ausgabe (tqdm, eine Zeile pro Epoche)
#   - fortsetzbaren Checkpoints (last.pth nach jeder Epoche)
#   - bestem Modell (best.pth) + Kopie nach models/final am Ende
#   - Warmup + Cosine LR, schrittweisem Auftauen, EMA, AMP, Early Stopping
# ============================================================

from __future__ import annotations

import copy
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    AMP_DTYPE,
    BACKBONE_LR_FACTOR,
    BEST_CKPT_PATH,
    CHANNELS_LAST,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EMA_DECAY,
    EPOCHS,
    FINAL_MODEL_PATH,
    LAST_CKPT_PATH,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MEAN,
    METRICS_DIR,
    STD,
    IMAGE_SIZE,
    ARCHITECTURE,
    NUM_CLASSES,
    USE_AMP,
    USE_COMPILE,
    USE_EMA,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from src.model import get_loss_function, save_checkpoint, unfreeze_layers

logger = logging.getLogger(__name__)

_USE_SCALER = USE_AMP and AMP_DTYPE == torch.float16


# ── Exponential Moving Average der Gewichte ───────────────────
class ModelEMA:
    """Haelt eine geglaettete Kopie der Gewichte - meist zuverlaessiger."""

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)


# ── Early Stopping ────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = EARLY_STOPPING_PATIENCE, delta: float = 1e-4,
                 best: float = float("inf"), counter: int = 0) -> None:
        self.patience, self.delta = patience, delta
        self.best, self.counter, self.stop = best, counter, False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.delta:
            self.best, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ── Optimizer / Scheduler ─────────────────────────────────────
def _build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Zwei Lernraten: Kopf schnell, Backbone langsam (typisch fuer Finetuning)."""
    head_keys = ("fc", "classifier")
    head, backbone = [], []
    for name, p in model.named_parameters():
        (head if name.split(".")[0] in head_keys else backbone).append(p)
    return torch.optim.AdamW([
        {"params": backbone, "lr": lr * BACKBONE_LR_FACTOR},
        {"params": head,     "lr": lr},
    ], weight_decay=WEIGHT_DECAY)


def _lr_factor(epoch: int, total: int, warmup: int) -> float:
    """Linearer Warmup, danach Cosine-Abfall auf ~0."""
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


# ── Eine Epoche ───────────────────────────────────────────────
def _move(images: torch.Tensor) -> torch.Tensor:
    images = images.to(DEVICE, non_blocking=True)
    if CHANNELS_LAST and DEVICE == "cuda":
        images = images.to(memory_format=torch.channels_last)
    return images


def _train_epoch(model, loader, optimizer, criterion, scaler, ema, epoch) -> Tuple[float, float]:
    model.train()
    total_loss = correct = total = 0
    bar = tqdm(loader, desc=f"Epoche {epoch:>2} | train", leave=False, dynamic_ncols=True)
    for images, labels in bar:
        images, labels = _move(images), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if _USE_SCALER:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (outputs.argmax(1) == labels).sum().item()
        total += bs
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")
    return total_loss / total, correct / total


@torch.no_grad()
def _validate(model, loader, criterion) -> Tuple[float, float]:
    model.eval()
    total_loss = correct = total = 0
    for images, labels in loader:
        images, labels = _move(images), labels.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (outputs.argmax(1) == labels).sum().item()
        total += bs
    return total_loss / total, correct / total


# ── Checkpoint-Bausteine ──────────────────────────────────────
def _base_meta() -> dict:
    return {"architecture": ARCHITECTURE, "num_classes": NUM_CLASSES,
            "mean": list(MEAN), "std": list(STD), "image_size": list(IMAGE_SIZE)}


def _empty_history() -> Dict[str, list]:
    return {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}


# ── Haupttraining ─────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    unfreeze_at_epoch: int = 3,
    unfreeze_n_layers: int = 60,
    resume: bool = False,
) -> Dict[str, list]:
    """
    Vollstaendiger Trainingslauf.

    Nach jeder Epoche wird last.pth geschrieben (kompletter Zustand) -> bei
    Abbruch kann mit resume=True nahtlos weitertrainiert werden. Das beste
    Modell (kleinster Val-Loss) landet in best.pth und am Ende als Kopie in
    models/final/final_model.pth samt Statistiken.
    """
    criterion = get_loss_function()
    optimizer = _build_optimizer(model, lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: _lr_factor(e, epochs, WARMUP_EPOCHS))
    scaler = torch.amp.GradScaler("cuda", enabled=_USE_SCALER)
    ema = ModelEMA(model) if USE_EMA else None

    # core = unkompiliertes Modell: dient fuer Checkpoints, EMA, Optimizer.
    # train_model = (optional) kompilierte Variante fuer schnellere Forward-Passes.
    core = model
    train_model = model
    if USE_COMPILE and DEVICE == "cuda":
        try:
            compiled = torch.compile(model)
            # Test-Forward: torch.compile kompiliert erst beim 1. Aufruf,
            # deshalb hier provozieren, um Fehler (z. B. fehlendes Triton)
            # abzufangen und sauber ohne Kompilierung weiterzulaufen.
            probe = torch.zeros(2, 3, *IMAGE_SIZE, device=DEVICE)
            if CHANNELS_LAST:
                probe = probe.to(memory_format=torch.channels_last)
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                compiled(probe)
            train_model = compiled
            logger.info("torch.compile aktiv.")
        except Exception as exc:  # pragma: no cover
            logger.warning("torch.compile nicht moeglich (%s) - ohne Kompilierung.",
                           type(exc).__name__)

    history = _empty_history()
    start_epoch, best_val = 1, float("inf")
    stopper = EarlyStopping()
    unfrozen = False

    # ── Fortsetzen ─────────────────────────────────────────────
    if resume and LAST_CKPT_PATH.exists():
        ck = torch.load(LAST_CKPT_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if _USE_SCALER and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema"):
            ema.module.load_state_dict(ck["ema"])
        history = ck.get("history", history)
        best_val = ck.get("best_val", best_val)
        start_epoch = ck.get("epoch", 0) + 1
        stopper = EarlyStopping(best=ck.get("best_val", float("inf")),
                                counter=ck.get("early_stop_counter", 0))
        unfrozen = ck.get("unfrozen", False)
        logger.info("Fortsetzen ab Epoche %d (bester Val-Loss bisher: %.4f)",
                    start_epoch, best_val)

    if unfrozen:  # nach resume Backbone wieder auftauen
        unfreeze_layers(model, unfreeze_n_layers)

    logger.info("=" * 64)
    logger.info("Training: Epochen %d-%d | Batch %d | bf16=%s | EMA=%s",
                start_epoch, epochs, train_loader.batch_size,
                USE_AMP and AMP_DTYPE == torch.bfloat16, USE_EMA)
    logger.info("=" * 64)

    t_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        if epoch == unfreeze_at_epoch and not unfrozen:
            unfreeze_layers(model, unfreeze_n_layers)
            unfrozen = True

        t0 = time.time()
        tr_loss, tr_acc = _train_epoch(train_model, train_loader, optimizer,
                                       criterion, scaler, ema, epoch)
        # Validierung immer am echten Modell (EMA wuerde bei wenigen Schritten
        # pro Epoche hinterherhinken und die Validierung verfaelschen).
        eval_model = core
        va_loss, va_acc = _validate(eval_model, val_loader, criterion)
        scheduler.step()
        cur_lr = optimizer.param_groups[-1]["lr"]

        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc, cur_lr)):
            history[k].append(v)

        improved = va_loss < best_val
        if improved:
            best_val = va_loss
            save_checkpoint({
                "model": model.state_dict(),
                "ema": ema.module.state_dict() if ema is not None else None,
                "epoch": epoch, "best_val": best_val,
                "metrics": {"val_loss": va_loss, "val_acc": va_acc},
                **_base_meta(),
            }, BEST_CKPT_PATH)

        # last.pth: kompletter Zustand zum Fortsetzen
        save_checkpoint({
            "model": model.state_dict(),
            "ema": ema.module.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if _USE_SCALER else None,
            "epoch": epoch, "best_val": best_val,
            "early_stop_counter": stopper.counter,
            "unfrozen": unfrozen, "history": history,
            **_base_meta(),
        }, LAST_CKPT_PATH)

        logger.info(
            "Epoche %2d/%d | train loss %.4f acc %.3f | val loss %.4f acc %.3f "
            "| lr %.2e | %4.0fs%s",
            epoch, epochs, tr_loss, tr_acc, va_loss, va_acc, cur_lr,
            time.time() - t0, "  <- bestes" if improved else "")

        if stopper.step(va_loss):
            logger.info("Early Stopping nach Epoche %d (kein Fortschritt seit %d).",
                        epoch, stopper.patience)
            break

    # ── Abschluss: bestes Modell nach final + Statistiken ──────
    _finalize(history, best_val, time.time() - t_start)
    return history


def _finalize(history: Dict[str, list], best_val: float, elapsed: float) -> None:
    """Bestes Modell nach models/final kopieren und Statistiken speichern."""
    if BEST_CKPT_PATH.exists():
        best = torch.load(BEST_CKPT_PATH, map_location="cpu", weights_only=False)
        save_checkpoint(best, FINAL_MODEL_PATH)
        logger.info("Bestes Modell -> %s", FINAL_MODEL_PATH)

    val_losses = history["val_loss"]
    best_epoch = (val_losses.index(min(val_losses)) + 1) if val_losses else 0
    summary = {
        "epochs_trained": len(val_losses),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val, 4),
        "best_val_acc": round(history["val_acc"][best_epoch - 1], 4) if best_epoch else None,
        "final_train_acc": round(history["train_acc"][-1], 4) if history["train_acc"] else None,
        "training_seconds": round(elapsed, 1),
    }
    (METRICS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))
    (METRICS_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Statistiken gespeichert: training_history.json, training_summary.json")
    _plot_history(history)


def _plot_history(history: Dict[str, list]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.config import PLOTS_DIR

        if not history["train_loss"]:
            return
        ep = range(1, len(history["train_loss"]) + 1)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        a1.plot(ep, history["train_loss"], label="Train"); a1.plot(ep, history["val_loss"], label="Val")
        a1.set_title("Loss"); a1.set_xlabel("Epoche"); a1.legend(); a1.grid(alpha=.3)
        a2.plot(ep, [a*100 for a in history["train_acc"]], label="Train")
        a2.plot(ep, [a*100 for a in history["val_acc"]], label="Val")
        a2.set_title("Accuracy (%)"); a2.set_xlabel("Epoche"); a2.legend(); a2.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(PLOTS_DIR / "training_history.png", dpi=150)
        plt.close(fig)
        logger.info("Plot gespeichert: training_history.png")
    except Exception as exc:  # pragma: no cover
        logger.warning("Trainings-Plot fehlgeschlagen: %s", exc)
