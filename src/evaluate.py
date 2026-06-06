"""
src/evaluate.py - schnelle Evaluation nach dem Training.

Berechnet die Kernmetriken auf dem TEST-Split, bestimmt den fuer die
Medizin sinnvollen Schwellenwert (hoher Recall) und speichert
test_metrics.json + optimal_threshold.json sowie die wichtigsten Plots.

Eine ausfuehrliche Zuverlaessigkeits-Analyse macht analyze.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
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

from src.config import CLASS_NAMES, DEVICE, METRICS_DIR, PLOTS_DIR

logger = logging.getLogger(__name__)


# ── Vorhersagen sammeln ───────────────────────────────────────
@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader) -> Tuple[List[int], List[float]]:
    """Gibt (echte Labels, Wahrscheinlichkeit fuer 'infected') zurueck."""
    model.eval()
    labels: List[int] = []
    probs: List[float] = []
    for images, lab in loader:
        images = images.to(DEVICE, non_blocking=True)
        p = torch.softmax(model(images), dim=1)[:, 1]
        labels.extend(lab.tolist())
        probs.extend(p.float().cpu().tolist())
    return labels, probs


# ── Schwellenwert optimieren ──────────────────────────────────
def find_optimal_threshold(labels: List[int], probs: List[float],
                           min_recall: float = 0.95) -> Tuple[float, Dict[str, float]]:
    """Hoechster F1 unter der Bedingung Recall >= min_recall (kein Fall verpassen)."""
    best_t, best_f1, best = 0.5, -1.0, {}
    for t in np.arange(0.05, 0.95, 0.01):
        preds = [1 if p >= t else 0 for p in probs]
        rec = recall_score(labels, preds, zero_division=0)
        if rec < min_recall:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
            best = _threshold_metrics(labels, preds, best_t)
    if not best:  # kein Threshold erfuellt die Recall-Anforderung
        preds = [1 if p >= 0.5 else 0 for p in probs]
        best_t, best = 0.5, _threshold_metrics(labels, preds, 0.5)
    logger.info("Optimaler Schwellenwert: %.2f (Recall %.3f, F1 %.3f)",
                best_t, best["recall"], best["f1"])
    return best_t, best


def _threshold_metrics(labels, preds, t) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return {"threshold": round(t, 4), "accuracy": round(acc, 4),
            "precision": round(precision_score(labels, preds, zero_division=0), 4),
            "recall": round(recall_score(labels, preds, zero_division=0), 4),
            "specificity": round(spec, 4),
            "f1": round(f1_score(labels, preds, zero_division=0), 4)}


# ── Metriken ──────────────────────────────────────────────────
def compute_metrics(labels: List[int], preds: List[int], probs: List[float]) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ci_low, ci_high = _bootstrap_ci(labels, preds)
    return {
        "accuracy": round(sum(p == l for p, l in zip(preds, labels)) / len(labels), 4),
        "accuracy_ci_low": round(ci_low, 4),
        "accuracy_ci_high": round(ci_high, 4),
        "precision": round(precision_score(labels, preds, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(labels, preds, average="weighted", zero_division=0), 4),
        "specificity": round(spec, 4),
        "f1": round(f1_score(labels, preds, average="weighted", zero_division=0), 4),
        "auc": round(_safe(roc_auc_score, labels, probs), 4),
        "ap": round(_safe(average_precision_score, labels, probs), 4),
    }


def _safe(fn, labels, probs) -> float:
    try:
        return float(fn(labels, probs))
    except ValueError:
        return float("nan")


def _bootstrap_ci(labels, preds, n: int = 1000, conf: float = 0.95) -> Tuple[float, float]:
    la, pr = np.array(labels), np.array(preds)
    rng = np.random.default_rng(42)
    accs = [(la[idx] == pr[idx]).mean()
            for idx in (rng.integers(0, len(la), len(la)) for _ in range(n))]
    a = 1 - conf
    return float(np.percentile(accs, 100 * a / 2)), float(np.percentile(accs, 100 * (1 - a / 2)))


# ── Plots ─────────────────────────────────────────────────────
def _plot_confusion(labels, preds, path: Path) -> None:
    import matplotlib.pyplot as plt
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], CLASS_NAMES); ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Vorhergesagt"); ax.set_ylabel("Tatsaechlich")
    ax.set_title("Confusion Matrix"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_roc(labels, probs, auc, t, path: Path) -> None:
    import matplotlib.pyplot as plt
    fpr, tpr, thr = roc_curve(labels, probs)
    i = int(np.argmin(np.abs(thr - t)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Zufall")
    ax.scatter(fpr[i], tpr[i], color="red", zorder=5, label=f"Schwelle {t:.2f}")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-Kurve"); ax.legend(loc="lower right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_pr(labels, probs, ap, path: Path) -> None:
    import matplotlib.pyplot as plt
    prec, rec, _ = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"PR (AP={ap:.4f})")
    ax.axhline(sum(labels) / len(labels), ls="--", color="gray", label="Zufall")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall-Kurve"); ax.legend(loc="lower left"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ── Hauptfunktion ─────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader,
             history: Dict[str, list] | None = None) -> Dict[str, float]:
    """Evaluation auf dem TEST-Split; speichert Metriken + Plots."""
    logger.info("=" * 64)
    logger.info("Evaluation auf dem Test-Split")
    logger.info("=" * 64)

    labels, probs = collect_predictions(model, loader)
    threshold, thr_metrics = find_optimal_threshold(labels, probs)
    preds = [1 if p >= threshold else 0 for p in probs]
    metrics = compute_metrics(labels, preds, probs)
    metrics["optimal_threshold"] = round(threshold, 4)

    logger.info("\n%s", classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    logger.info("Accuracy %.2f%% (95%% CI %.2f-%.2f) | Recall %.2f%% | "
                "Specificity %.2f%% | F1 %.2f%% | AUC %.4f | AP %.4f",
                metrics["accuracy"] * 100, metrics["accuracy_ci_low"] * 100,
                metrics["accuracy_ci_high"] * 100, metrics["recall"] * 100,
                metrics["specificity"] * 100, metrics["f1"] * 100,
                metrics["auc"], metrics["ap"])

    for fn, args in [
        (_plot_confusion, (labels, preds, PLOTS_DIR / "confusion_matrix.png")),
        (_plot_roc, (labels, probs, metrics["auc"], threshold, PLOTS_DIR / "roc_curve.png")),
        (_plot_pr, (labels, probs, metrics["ap"], PLOTS_DIR / "precision_recall_curve.png")),
    ]:
        try:
            fn(*args)
        except Exception as exc:
            logger.warning("Plot %s fehlgeschlagen: %s", fn.__name__, exc)

    (METRICS_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=4))
    (METRICS_DIR / "optimal_threshold.json").write_text(json.dumps(thr_metrics, indent=4))
    logger.info("Gespeichert: test_metrics.json, optimal_threshold.json + Plots")
    return metrics
