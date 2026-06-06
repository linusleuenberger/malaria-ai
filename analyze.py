"""
analyze.py - Ausfuehrliche Zuverlaessigkeits-Analyse des finalen Modells.

Dieses Script ist getrennt vom Training und beantwortet die Frage:
"Wie zuverlaessig ist mein Modell wirklich?" - immer auf dem TEST-Split,
den das Modell waehrend des Trainings nie gesehen hat.

Es erzeugt:
  - results/metrics/evaluation_report.json   (alle Zahlen, maschinenlesbar)
  - results/evaluation_report.md             (lesbarer Bericht)
  - results/plots/  reliability_diagram, confidence_histogram,
                    confusion_matrix, roc_curve, precision_recall_curve,
                    gradcam, misclassified

Verwendung:
    python analyze.py                 # nutzt models/final/final_model.pth
    python analyze.py --tta           # mit Test-Time-Augmentation (etwas zuverlaessiger)
    python analyze.py --model models/checkpoints/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.config import (
    BEST_CKPT_PATH,
    CLASS_NAMES,
    DEVICE,
    FINAL_MODEL_PATH,
    IDX_TO_CLASS,
    MEAN,
    METRICS_DIR,
    PLOTS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    STD,
)
from src.dataset import get_dataloaders
from src.evaluate import (
    _plot_confusion,
    _plot_pr,
    _plot_roc,
    compute_metrics,
    find_optimal_threshold,
)
from src.model import load_model
from src.utils import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("analyze")


# ── Vorhersagen (optional mit Test-Time-Augmentation) ─────────
@torch.no_grad()
def collect(model, loader, tta: bool = False) -> Tuple[List[int], List[float]]:
    model.eval()
    labels: List[int] = []
    probs: List[float] = []
    for images, lab in loader:
        images = images.to(DEVICE, non_blocking=True)
        p = F.softmax(model(images), dim=1)
        if tta:  # Mittelwert ueber Original + horizontal/vertikal gespiegelt
            p = p + F.softmax(model(torch.flip(images, [3])), dim=1)
            p = p + F.softmax(model(torch.flip(images, [2])), dim=1)
            p = p / 3.0
        labels.extend(lab.tolist())
        probs.extend(p[:, 1].float().cpu().tolist())
    return labels, probs


# ── Kalibrierung (Expected Calibration Error) ─────────────────
def calibration(labels, probs, preds, n_bins: int = 10) -> Tuple[float, list]:
    """
    ECE: Stimmt die angegebene Sicherheit mit der echten Trefferquote ueberein?
    Confidence = Wahrscheinlichkeit der vorhergesagten Klasse.
    """
    conf = np.array([p if pr == 1 else 1 - p for p, pr in zip(probs, preds)])
    correct = np.array([int(pr == l) for pr, l in zip(preds, labels)])
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append((float((lo + hi) / 2), 0, 0.0, 0.0)); continue
        acc, avg_conf, w = correct[m].mean(), conf[m].mean(), m.mean()
        ece += w * abs(acc - avg_conf)
        rows.append((float((lo + hi) / 2), int(m.sum()), float(acc), float(avg_conf)))
    return float(ece), rows


def per_class_metrics(labels, preds) -> Dict[str, Dict[str, float]]:
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=[0, 1], zero_division=0)
    return {CLASS_NAMES[i]: {"precision": round(p[i], 4), "recall": round(r[i], 4),
                             "f1": round(f[i], 4), "support": int(s[i])} for i in range(2)}


def threshold_sweep(labels, probs) -> List[Dict[str, float]]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    rows = []
    for t in np.arange(0.1, 0.91, 0.1):
        preds = [1 if p >= t else 0 for p in probs]
        rows.append({"threshold": round(float(t), 2),
                     "precision": round(precision_score(labels, preds, zero_division=0), 4),
                     "recall": round(recall_score(labels, preds, zero_division=0), 4),
                     "f1": round(f1_score(labels, preds, zero_division=0), 4)})
    return rows


# ── Plots: Kalibrierung & Konfidenz ───────────────────────────
def plot_reliability(rows, ece, path: Path) -> None:
    import matplotlib.pyplot as plt
    centers = [r[0] for r in rows]; accs = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfekt kalibriert")
    ax.plot(centers, accs, "o-", lw=2, label=f"Modell (ECE={ece:.3f})")
    ax.set_xlabel("Angegebene Sicherheit"); ax.set_ylabel("Echte Trefferquote")
    ax.set_title("Reliability Diagram (Kalibrierung)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confidence_hist(labels, probs, preds, path: Path) -> None:
    import matplotlib.pyplot as plt
    conf = np.array([p if pr == 1 else 1 - p for p, pr in zip(probs, preds)])
    correct = np.array([pr == l for pr, l in zip(preds, labels)])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(conf[correct], bins=20, alpha=.7, label="richtig", color="seagreen")
    ax.hist(conf[~correct], bins=20, alpha=.7, label="falsch", color="tomato")
    ax.set_xlabel("Sicherheit"); ax.set_ylabel("Anzahl")
    ax.set_title("Konfidenz: richtige vs. falsche Vorhersagen"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _denorm(t: torch.Tensor) -> np.ndarray:
    img = t.permute(1, 2, 0).cpu().numpy() * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)


def plot_gradcam(model, loader, path: Path, n: int = 6) -> None:
    import matplotlib.pyplot as plt
    from src.predict import GradCAM, get_last_conv_layer
    from PIL import Image
    cam = GradCAM(model, get_last_conv_layer(model))
    images, labels = next(iter(loader))
    n = min(n, images.size(0))
    fig, axes = plt.subplots(n, 2, figsize=(7, n * 3))
    for i in range(n):
        t = images[i].unsqueeze(0)
        with torch.enable_grad():
            heat = cam(t)
        with torch.no_grad():
            pred = int(model(t.to(DEVICE)).argmax(1).item())
        base = _denorm(images[i])
        overlay = GradCAM.overlay(Image.fromarray((base * 255).astype(np.uint8)), heat)
        ok = pred == labels[i].item()
        axes[i, 0].imshow(base); axes[i, 0].axis("off")
        axes[i, 0].set_title(f"echt: {IDX_TO_CLASS[labels[i].item()]}", fontsize=9)
        axes[i, 1].imshow(overlay); axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Vorhersage: {IDX_TO_CLASS[pred]}", fontsize=9,
                             color="green" if ok else "red")
    fig.suptitle("Grad-CAM: worauf schaut das Modell?")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


@torch.no_grad()
def plot_misclassified(model, loader, threshold, path: Path, n: int = 12) -> None:
    import matplotlib.pyplot as plt
    model.eval()
    imgs, info = [], []
    for images, labels in loader:
        probs = F.softmax(model(images.to(DEVICE)), dim=1)[:, 1]
        preds = (probs >= threshold).long().cpu()
        for i in range(len(labels)):
            if preds[i] != labels[i] and len(imgs) < n:
                imgs.append(_denorm(images[i]))
                info.append((IDX_TO_CLASS[labels[i].item()],
                             IDX_TO_CLASS[int(preds[i])], float(probs[i])))
        if len(imgs) >= n:
            break
    if not imgs:
        logger.info("Keine Fehlklassifikationen gefunden."); return
    cols = 4; rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()
    for i, (img, (t, p, c)) in enumerate(zip(imgs, info)):
        axes[i].imshow(img); axes[i].axis("off")
        axes[i].set_title(f"echt: {t}\nVorhers.: {p} ({c:.2f})", fontsize=8, color="red")
    for j in range(len(imgs), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Falsch klassifizierte Bilder")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ── Markdown-Bericht ──────────────────────────────────────────
def write_markdown(report: Dict, path: Path) -> None:
    m, t = report["metrics"], report["optimal_threshold"]
    lines = [
        "# Zuverlaessigkeits-Analyse des Malaria-Modells", "",
        f"- Modell: `{report['model']}`",
        f"- Testbilder: {report['n_test']}  |  Test-Time-Augmentation: {report['tta']}",
        f"- Optimaler Schwellenwert: **{t}**  (Recall-Vorgabe >= 95 %)", "",
        "## Kernmetriken (Test-Split)", "",
        "| Metrik | Wert |", "|---|---|",
        f"| Accuracy | {m['accuracy']*100:.2f} % (95 % CI {m['accuracy_ci_low']*100:.2f}-{m['accuracy_ci_high']*100:.2f}) |",
        f"| Recall (Sensitivitaet) | {m['recall']*100:.2f} % |",
        f"| Specificity | {m['specificity']*100:.2f} % |",
        f"| Precision | {m['precision']*100:.2f} % |",
        f"| F1-Score | {m['f1']*100:.2f} % |",
        f"| AUC | {m['auc']:.4f} |",
        f"| AP | {m['ap']:.4f} |",
        f"| ECE (Kalibrierung, kleiner=besser) | {report['ece']:.4f} |", "",
        "## Pro Klasse", "",
        "| Klasse | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
    ]
    for cls, v in report["per_class"].items():
        lines.append(f"| {cls} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |")
    lines += ["", "## Schwellenwert-Analyse", "",
              "| Schwelle | Precision | Recall | F1 |", "|---|---|---|---|"]
    for r in report["threshold_sweep"]:
        lines.append(f"| {r['threshold']:.2f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    lines += ["", "## Plots", "",
              "Siehe `results/plots/`: reliability_diagram, confidence_histogram, "
              "confusion_matrix, roc_curve, precision_recall_curve, gradcam, misclassified.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Hauptablauf ───────────────────────────────────────────────
def run(model_path: Path, tta: bool) -> Dict:
    set_seed()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modell nicht gefunden: {model_path}\n"
            f"-> zuerst trainieren: python main.py --mode train")

    loaders = get_dataloaders(PROCESSED_DIR, splits=("test",), use_weighted_sampler=False)
    test_loader = loaders["test"]
    model = load_model(model_path)

    logger.info("Sammle Vorhersagen auf dem Test-Split (TTA=%s)...", tta)
    labels, probs = collect(model, test_loader, tta=tta)

    threshold, _ = find_optimal_threshold(labels, probs)
    preds = [1 if p >= threshold else 0 for p in probs]

    metrics = compute_metrics(labels, preds, probs)
    ece, rel_rows = calibration(labels, probs, preds)
    report = {
        "model": str(model_path), "n_test": len(labels), "tta": tta,
        "optimal_threshold": round(threshold, 4),
        "metrics": metrics, "ece": round(ece, 4),
        "per_class": per_class_metrics(labels, preds),
        "threshold_sweep": threshold_sweep(labels, probs),
    }

    # Plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    safe = [
        (plot_reliability, (rel_rows, ece, PLOTS_DIR / "reliability_diagram.png")),
        (plot_confidence_hist, (labels, probs, preds, PLOTS_DIR / "confidence_histogram.png")),
        (_plot_confusion, (labels, preds, PLOTS_DIR / "confusion_matrix.png")),
        (_plot_roc, (labels, probs, metrics["auc"], threshold, PLOTS_DIR / "roc_curve.png")),
        (_plot_pr, (labels, probs, metrics["ap"], PLOTS_DIR / "precision_recall_curve.png")),
        (plot_gradcam, (model, test_loader, PLOTS_DIR / "gradcam.png")),
        (plot_misclassified, (model, test_loader, threshold, PLOTS_DIR / "misclassified.png")),
    ]
    for fn, args in safe:
        try:
            fn(*args)
        except Exception as exc:
            logger.warning("Plot %s fehlgeschlagen: %s", fn.__name__, exc)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    write_markdown(report, RESULTS_DIR / "evaluation_report.md")

    logger.info("=" * 64)
    logger.info("Fertig. Accuracy %.2f%% | Recall %.2f%% | Specificity %.2f%% | "
                "AUC %.4f | ECE %.4f", metrics["accuracy"] * 100, metrics["recall"] * 100,
                metrics["specificity"] * 100, metrics["auc"], ece)
    logger.info("Bericht: results/evaluation_report.md")
    logger.info("=" * 64)
    return report


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Zuverlaessigkeits-Analyse des Modells")
    ap.add_argument("--model", type=str, default=None,
                    help="Pfad zum Modell (Standard: models/final, sonst best.pth)")
    ap.add_argument("--tta", action="store_true", help="Test-Time-Augmentation aktivieren")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.model:
        path = Path(args.model)
    else:
        path = FINAL_MODEL_PATH if FINAL_MODEL_PATH.exists() else BEST_CKPT_PATH
    run(path, tta=args.tta)
