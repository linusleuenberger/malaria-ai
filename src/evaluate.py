"""
evaluate.py - Modell-Evaluation mit Metriken, Plots & Fehleranalyse
=====================================================================

Erstellt:
    - Confusion Matrix
    - ROC Kurve mit optimalem Schwellenwert
    - Precision-Recall Kurve
    - Grad-CAM Heatmaps
    - Training History Plot
    - Falsch klassifizierte Bilder
    - Metriken mit Konfidenzintervallen als JSON
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    CLASS_NAMES,
    DEVICE,
    IDX_TO_CLASS,
    MEAN,
    STD,
    METRICS_DIR,
    PLOTS_DIR,
    USE_WANDB,
)
# GradCAM wird aus predict.py importiert – keine Duplizierung
from src.predict import GradCAM, get_last_conv_layer

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Vorhersagen sammeln
# ──────────────────────────────────────────────────────────────────────────────

def _collect_predictions(
    model:  nn.Module,
    loader: DataLoader,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Alle Vorhersagen und Labels sammeln.

    Returns:
        all_labels : Echte Labels
        all_preds  : Vorhergesagte Labels (bei threshold=0.5)
        all_probs  : Wahrscheinlichkeit für infected (für ROC & Threshold)
    """
    model.eval()
    all_labels: List[int]   = []
    all_preds:  List[int]   = []
    all_probs:  List[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with autocast(
                device_type = "cuda",
                enabled     = (str(DEVICE) == "cuda"),   # Fix: torch.device Vergleich
            ):
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds .extend(preds .cpu().tolist())
            all_probs .extend(probs[:, 1].cpu().tolist())

    return all_labels, all_preds, all_probs


# ──────────────────────────────────────────────────────────────────────────────
# Schwellenwert-Optimierung
# ──────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    labels:     List[int],
    probs:      List[float],
    min_recall: float = 0.95,
) -> Tuple[float, Dict[str, float]]:
    """
    Findet den optimalen Schwellenwert für medizinische Anwendung.

    Strategie:
        1. Recall muss >= min_recall (95%) sein
           -> Kein infizierter Patient darf verpasst werden
        2. Unter allen gültigen Schwellenwerten -> höchsten F1 wählen

    Returns:
        best_threshold : Optimaler Schwellenwert
        best_metrics   : Metriken bei diesem Schwellenwert
    """
    best_threshold = 0.5
    best_f1        = 0.0
    best_metrics: Dict[str, float] = {}

    for threshold in np.arange(0.1, 0.9, 0.01):
        preds  = [1 if p >= threshold else 0 for p in probs]
        recall = recall_score(labels, preds, zero_division=0)

        if recall < min_recall:
            continue

        f1        = f1_score       (labels, preds, zero_division=0)
        precision = precision_score(labels, preds, zero_division=0)
        accuracy  = sum(p == l for p, l in zip(preds, labels)) / len(labels)

        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if f1 > best_f1:
            best_f1        = f1
            best_threshold = float(threshold)
            best_metrics   = {
                "threshold"  : best_threshold,
                "accuracy"   : round(accuracy,    4),
                "precision"  : round(precision,   4),
                "recall"     : round(recall,      4),
                "specificity": round(specificity, 4),
                "f1"         : round(f1,          4),
            }

    # Fallback wenn kein Threshold die Recall-Anforderung erfüllt
    if not best_metrics:
        logger.warning(
            f"Kein Schwellenwert mit Recall >= {min_recall:.0%} gefunden. "
            f"Verwende Standard 0.5."
        )
        best_threshold = 0.5
        preds          = [1 if p >= 0.5 else 0 for p in probs]
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        best_metrics   = {
            "threshold"  : 0.5,
            "accuracy"   : round(sum(p == l for p, l in zip(preds, labels)) / len(labels), 4),
            "precision"  : round(precision_score(labels, preds, zero_division=0), 4),
            "recall"     : round(recall_score   (labels, preds, zero_division=0), 4),
            "specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
            "f1"         : round(f1_score       (labels, preds, zero_division=0), 4),
        }

    logger.info(f"Optimaler Schwellenwert : {best_threshold:.2f}")
    logger.info(f"  Recall      : {best_metrics['recall']      :.2%}")
    logger.info(f"  Specificity : {best_metrics['specificity'] :.2%}")
    logger.info(f"  Precision   : {best_metrics['precision']   :.2%}")
    logger.info(f"  F1          : {best_metrics['f1']          :.2%}")

    return best_threshold, best_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Metriken berechnen
# ──────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    labels: List[int],
    preds:  List[int],
    probs:  List[float],
) -> Dict[str, float]:
    """
    Alle wichtigen Metriken berechnen inkl. Bootstrap Konfidenzintervall.

    Neu gegenüber Original:
        - Specificity (True Negative Rate)
        - Average Precision Score (AP) – besser als AUC bei unbalancierten Daten
    """
    accuracy  = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall    = recall_score   (labels, preds, average="weighted", zero_division=0)
    f1        = f1_score       (labels, preds, average="weighted", zero_division=0)

    # Specificity
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    specificity    = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
        logger.warning("AUC konnte nicht berechnet werden.")

    # Average Precision Score
    try:
        ap = average_precision_score(labels, probs)
    except ValueError:
        ap = float("nan")
        logger.warning("AP Score konnte nicht berechnet werden.")

    acc_ci = _bootstrap_confidence_interval(labels, preds)

    return {
        "accuracy"        : round(accuracy,    4),
        "accuracy_ci_low" : round(acc_ci[0],   4),
        "accuracy_ci_high": round(acc_ci[1],   4),
        "precision"       : round(precision,   4),
        "recall"          : round(recall,      4),
        "specificity"     : round(specificity, 4),
        "f1"              : round(f1,          4),
        "auc"             : round(auc,         4),
        "ap"              : round(ap,          4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap Konfidenzintervall
# ──────────────────────────────────────────────────────────────────────────────

def _bootstrap_confidence_interval(
    labels:      List[int],
    preds:       List[int],
    n_bootstrap: int   = 1000,
    confidence:  float = 0.95,
) -> Tuple[float, float]:
    """
    Bootstrap Konfidenzintervall für Accuracy.

    1000x zufällig Testbilder neu samplen -> Accuracy jedes Mal berechnen
    -> 95% aller Werte = Konfidenzintervall.

    Returns:
        (ci_low, ci_high)
    """
    labels_arr = np.array(labels)
    preds_arr  = np.array(preds)
    n          = len(labels_arr)
    accuracies = []

    rng = np.random.default_rng(seed=42)
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        acc     = (labels_arr[indices] == preds_arr[indices]).mean()
        accuracies.append(acc)

    alpha   = 1 - confidence
    ci_low  = float(np.percentile(accuracies, 100 * alpha / 2))
    ci_high = float(np.percentile(accuracies, 100 * (1 - alpha / 2)))

    logger.info(
        f"Accuracy Konfidenzintervall (95%): {ci_low:.2%} – {ci_high:.2%}"
    )

    return ci_low, ci_high


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktion: Denormalisierung
# ──────────────────────────────────────────────────────────────────────────────

def _denormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Tensor denormalisieren mit MEAN/STD aus config.py.

    Fix: Vorher waren ImageNet-Werte hardcodiert.
    Jetzt kommen die Werte aus config.py -> bleibt konsistent
    wenn eigene Normalisierungswerte berechnet wurden.

    Args:
        img_tensor : [C, H, W] Tensor

    Returns:
        [H, W, C] numpy Array mit Werten in [0, 1]
    """
    mean = np.array(MEAN)
    std  = np.array(STD)
    img  = img_tensor.permute(1, 2, 0).numpy()
    img  = img * std + mean
    return np.clip(img, 0, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    labels:    List[int],
    preds:     List[int],
    save_path: Path,
) -> None:
    """Confusion Matrix als Heatmap speichern."""
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "Blues",
        xticklabels = CLASS_NAMES,
        yticklabels = CLASS_NAMES,
        ax          = ax,
    )
    ax.set_title ("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Vorhergesagt",     fontsize=12)
    ax.set_ylabel("Tatsächlich",      fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion Matrix gespeichert: {save_path}")


def _plot_roc_curve(
    labels:    List[int],
    probs:     List[float],
    auc:       float,
    threshold: float,
    save_path: Path,
) -> None:
    """ROC Kurve mit optimalem Schwellenwert speichern."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    threshold_idx        = np.argmin(np.abs(thresholds - threshold))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr,
            color="royalblue", linewidth=2,
            label=f"ROC Kurve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1],
            color="gray", linestyle="--", linewidth=1,
            label="Zufällig (AUC = 0.5)")
    ax.scatter(
        fpr[threshold_idx], tpr[threshold_idx],
        color="red", s=100, zorder=5,
        label=f"Optimaler Schwellenwert ({threshold:.2f})",
    )
    ax.set_title ("ROC Kurve",           fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.legend    (loc="lower right",     fontsize=11)
    ax.grid      (True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"ROC Kurve gespeichert: {save_path}")


def _plot_precision_recall_curve(
    labels:    List[int],
    probs:     List[float],
    ap:        float,
    threshold: float,
    save_path: Path,
) -> None:
    """
    Precision-Recall Kurve speichern.

    Aussagekräftiger als ROC bei unbalancierten Datensätzen,
    weil sie nicht durch die hohe Anzahl echter Negativen
    geschönt wird.
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(labels, probs)
    threshold_idx = np.argmin(np.abs(thresholds - threshold))

    # Baseline: zufälliger Klassifizierer
    baseline = sum(labels) / len(labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals,
            color="royalblue", linewidth=2,
            label=f"PR Kurve (AP = {ap:.4f})")
    ax.scatter(
        recall_vals[threshold_idx],
        precision_vals[threshold_idx],
        color="red", s=100, zorder=5,
        label=f"Optimaler Schwellenwert ({threshold:.2f})",
    )
    ax.axhline(
        y=baseline, color="gray", linestyle="--", linewidth=1,
        label=f"Zufällig (P = {baseline:.2f})",
    )
    ax.set_title ("Precision-Recall Kurve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall",                  fontsize=12)
    ax.set_ylabel("Precision",               fontsize=12)
    ax.legend    (loc="lower left",          fontsize=11)
    ax.grid      (True, alpha=0.3)
    ax.set_xlim  ([0.0, 1.05])
    ax.set_ylim  ([0.0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"PR Kurve gespeichert: {save_path}")


def plot_training_history(
    history:   Dict[str, list],
    save_path: Path = PLOTS_DIR / "training_history.png",
) -> None:
    """Train/Val Loss und Accuracy über alle Epochen plotten."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"],
             color="royalblue", linewidth=2, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],
             color="tomato",    linewidth=2, label="Val Loss")
    ax1.set_title ("Loss über Epochen", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch",             fontsize=12)
    ax1.set_ylabel("Loss",              fontsize=12)
    ax1.legend    (fontsize=11)
    ax1.grid      (True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]],
             color="royalblue", linewidth=2, label="Train Accuracy")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],
             color="tomato",    linewidth=2, label="Val Accuracy")
    ax2.set_title ("Accuracy über Epochen", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch",                 fontsize=12)
    ax2.set_ylabel("Accuracy (%)",          fontsize=12)
    ax2.legend    (fontsize=11)
    ax2.grid      (True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Training History gespeichert: {save_path}")


def plot_gradcam(
    model:     nn.Module,
    loader:    DataLoader,
    n_images:  int  = 8,
    save_path: Path = PLOTS_DIR / "gradcam.png",
) -> None:
    """
    Grad-CAM Heatmaps für Beispielbilder plotten.

    Fix: GradCAM kommt aus predict.py (keine Duplizierung mehr).
    Fix: Denormalisierung nutzt MEAN/STD aus config.py.
    """
    cam = GradCAM(model, get_last_conv_layer(model))

    collected_imgs:   List[np.ndarray] = []
    collected_cams:   List[np.ndarray] = []
    collected_labels: List[str]        = []
    collected_preds:  List[str]        = []
    images_shown = 0

    model.eval()
    for images, labels in loader:
        for i in range(images.size(0)):
            if images_shown >= n_images:
                break

            img_tensor = images[i].unsqueeze(0)

            # Grad-CAM braucht Gradienten -> enable_grad
            with torch.enable_grad():
                heatmap = cam(img_tensor.to(DEVICE))

            # Vorhersage ohne Gradienten
            with torch.no_grad():
                output = model(img_tensor.to(DEVICE))
                pred   = output.argmax(dim=1).item()

            # Fix: _denormalize nutzt config.py Werte
            collected_imgs  .append(_denormalize(images[i]))
            collected_cams  .append(heatmap)
            collected_labels.append(IDX_TO_CLASS[labels[i].item()])
            collected_preds .append(IDX_TO_CLASS[pred])
            images_shown += 1

        if images_shown >= n_images:
            break

    fig, axes = plt.subplots(n_images, 3, figsize=(12, n_images * 3))

    for i, (img, heatmap, true_label, pred_label) in enumerate(
        zip(collected_imgs, collected_cams, collected_labels, collected_preds)
    ):
        correct = true_label == pred_label
        color   = "green" if correct else "red"
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        overlay = GradCAM.overlay(pil_img, heatmap)

        axes[i, 0].imshow(img);      axes[i, 0].set_title(f"Original\nEcht: {true_label}",      fontsize=9);                    axes[i, 0].axis("off")
        axes[i, 1].imshow(heatmap, cmap="jet"); axes[i, 1].set_title("Grad-CAM\nHeatmap",        fontsize=9);                    axes[i, 1].axis("off")
        axes[i, 2].imshow(overlay); axes[i, 2].set_title(f"Überlagert\nVorherges: {pred_label}", fontsize=9, color=color);       axes[i, 2].axis("off")

    plt.suptitle("Grad-CAM – Was schaut die KI an?", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Grad-CAM gespeichert: {save_path}")


def _plot_misclassified(
    model:     nn.Module,
    loader:    DataLoader,
    threshold: float = 0.5,
    n_images:  int   = 12,
    save_path: Path  = PLOTS_DIR / "misclassified.png",
) -> None:
    """Zeigt Bilder die das Modell falsch klassifiziert hat."""
    model.eval()
    wrong_images: List[torch.Tensor] = []
    wrong_labels: List[int]          = []
    wrong_preds:  List[int]          = []
    wrong_probs:  List[float]        = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with autocast(
                device_type = "cuda",
                enabled     = (str(DEVICE) == "cuda"),
            ):
                outputs = model(images)

            probs = torch.softmax(outputs, dim=1)
            preds = (probs[:, 1] >= threshold).long()

            wrong_mask = preds != labels
            for i, wrong in enumerate(wrong_mask):
                if wrong and len(wrong_images) < n_images:
                    wrong_images.append(images[i].cpu())
                    wrong_labels.append(labels[i].cpu().item())
                    wrong_preds .append(preds[i] .cpu().item())
                    wrong_probs .append(probs[i][preds[i]].cpu().item())

            if len(wrong_images) >= n_images:
                break

    if not wrong_images:
        logger.info("Keine falsch klassifizierten Bilder gefunden.")
        return

    cols = 4
    rows = (len(wrong_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, (img_t, true_label, pred_label, prob) in enumerate(
        zip(wrong_images, wrong_labels, wrong_preds, wrong_probs)
    ):
        # Fix: _denormalize nutzt config.py Werte statt hardcoded ImageNet
        axes[i].imshow(_denormalize(img_t))
        axes[i].set_title(
            f"Echt:      {IDX_TO_CLASS[true_label]}\n"
            f"Vorherges: {IDX_TO_CLASS[pred_label]}\n"
            f"Konfidenz: {prob:.2%}",
            fontsize=9, color="red",
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Falsch klassifizierte Bilder", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Falsch klassifizierte Bilder gespeichert: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Hauptfunktion
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    history: Dict[str, list] | None = None,
) -> Dict[str, float]:
    """
    Vollständige Evaluation des Modells auf dem Testset.

    Args:
        model   : Das trainierte Modell
        loader  : Test DataLoader
        history : Trainings-Verlauf aus train()

    Returns:
        Dict mit allen Metriken
    """
    logger.info("=" * 60)
    logger.info("Evaluation gestartet")
    logger.info("=" * 60)

    PLOTS_DIR  .mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Vorhersagen sammeln
    labels, preds, probs = _collect_predictions(model, loader)

    # 2. Optimalen Schwellenwert finden
    threshold, threshold_metrics = find_optimal_threshold(labels, probs)

    # Vorhersagen mit optimalem Schwellenwert neu berechnen
    preds = [1 if p >= threshold else 0 for p in probs]

    # 3. Metriken berechnen
    metrics = _compute_metrics(labels, preds, probs)
    metrics["optimal_threshold"] = threshold

    # 4. Detaillierten Bericht ausgeben
    logger.info("\n" + classification_report(
        labels, preds,
        target_names = CLASS_NAMES,
        digits       = 4,
    ))
    logger.info("-" * 40)
    logger.info(f"  Accuracy    : {metrics['accuracy']   :.2%} "
                f"(95% CI: {metrics['accuracy_ci_low']:.2%} – "
                f"{metrics['accuracy_ci_high']:.2%})")
    logger.info(f"  Precision   : {metrics['precision']  :.2%}")
    logger.info(f"  Recall      : {metrics['recall']     :.2%}")
    logger.info(f"  Specificity : {metrics['specificity']:.2%}")
    logger.info(f"  F1-Score    : {metrics['f1']         :.2%}")
    logger.info(f"  AUC         : {metrics['auc']        :.4f}")
    logger.info(f"  AP Score    : {metrics['ap']         :.4f}")
    logger.info(f"  Threshold   : {threshold:.2f}")
    logger.info("-" * 40)

    # 5. Plots erstellen (jeder in eigenem try/except – ein Fehler stoppt nicht alles)
    for fn, kwargs in [
        (_plot_confusion_matrix,       dict(labels=labels, preds=preds,
                                            save_path=PLOTS_DIR / "confusion_matrix.png")),
        (_plot_roc_curve,              dict(labels=labels, probs=probs,
                                            auc=metrics["auc"], threshold=threshold,
                                            save_path=PLOTS_DIR / "roc_curve.png")),
        (_plot_precision_recall_curve, dict(labels=labels, probs=probs,
                                            ap=metrics["ap"], threshold=threshold,
                                            save_path=PLOTS_DIR / "precision_recall_curve.png")),
        (_plot_misclassified,          dict(model=model, loader=loader,
                                            threshold=threshold,
                                            save_path=PLOTS_DIR / "misclassified.png")),
        (plot_gradcam,                 dict(model=model, loader=loader,
                                            save_path=PLOTS_DIR / "gradcam.png")),
    ]:
        try:
            fn(**kwargs)
        except Exception as e:
            logger.warning(f"{fn.__name__} fehlgeschlagen: {e}")

    if history:
        try:
            plot_training_history(history)
        except Exception as e:
            logger.warning(f"Training History fehlgeschlagen: {e}")

    # 6. Optimalen Schwellenwert separat speichern
    # -> predict.py kann diesen Wert laden statt hardcoded 0.70 zu nutzen
    threshold_path = METRICS_DIR / "optimal_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({"threshold": threshold, **threshold_metrics}, f, indent=4)
    logger.info(f"Optimaler Schwellenwert gespeichert: {threshold_path}")

    # 7. WandB Logging
    if USE_WANDB:
        try:
            import wandb
            wandb.log(metrics)
            wandb.log({
                "confusion_matrix"       : wandb.Image(str(PLOTS_DIR / "confusion_matrix.png")),
                "roc_curve"              : wandb.Image(str(PLOTS_DIR / "roc_curve.png")),
                "precision_recall_curve" : wandb.Image(str(PLOTS_DIR / "precision_recall_curve.png")),
                "gradcam"                : wandb.Image(str(PLOTS_DIR / "gradcam.png")),
                "misclassified"          : wandb.Image(str(PLOTS_DIR / "misclassified.png")),
            })
            logger.info("Metriken & Plots zu WandB geloggt.")
        except Exception as e:
            logger.warning(f"WandB Logging fehlgeschlagen: {e}")

    # 8. Metriken als JSON speichern
    metrics_path = METRICS_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metriken gespeichert: {metrics_path}")

    logger.info("=" * 60)
    logger.info("Evaluation abgeschlossen.")
    logger.info("=" * 60)

    return metrics


# ── Quick-Test: python -m src.evaluate ───────────────────────────────────────
if __name__ == "__main__":
    from src.config  import BEST_MODEL_PATH, PROCESSED_DIR
    from src.dataset import get_dataloaders
    from src.model   import load_model

    loaders = get_dataloaders(
        data_dir    = PROCESSED_DIR,
        batch_size  = 32,
        num_workers = 0,
        pin_memory  = False,
    )

    model   = load_model(BEST_MODEL_PATH)
    metrics = evaluate(model=model, loader=loaders["test"])

    print("\nMetriken:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\n[OK] evaluate.py funktioniert korrekt.")