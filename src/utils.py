# ============================================================
# src/utils.py
# Hilfsfunktionen die überall im Projekt verwendet werden
# ============================================================

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import (
    DEVICE,
    PLOTS_DIR,
    RANDOM_SEED,
    CLASS_NAMES,
    IDX_TO_CLASS,
)

logger = logging.getLogger(__name__)


# ── Reproduzierbarkeit ────────────────────────────────────────
def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Alle Zufallsgeneratoren auf denselben Seed setzen.

    Warum:
        Ohne Seed → jedes Training gibt andere Ergebnisse
        Mit Seed  → gleiche Ergebnisse bei jedem Durchlauf
                    wichtig für Reproduzierbarkeit & ETH-Präsentation

    Args:
        seed : Zufallszahl (Standard aus config.py)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    logger.info(f"Seed gesetzt: {seed}")


# ── Hardware Info ─────────────────────────────────────────────
def print_system_info() -> None:
    """
    Zeigt Informationen über verfügbare Hardware.

    Nützlich um zu prüfen ob GPU erkannt wird
    und wie viel VRAM verfügbar ist.
    """
    logger.info("=" * 50)
    logger.info("System Info")
    logger.info("=" * 50)
    logger.info(f"PyTorch Version : {torch.__version__}")
    logger.info(f"CUDA verfügbar  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"GPU             : {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM gesamt     : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        logger.info(
            f"VRAM verfügbar  : "
            f"{torch.cuda.memory_reserved(0) / 1e9:.1f} GB reserviert"
        )
    else:
        logger.info("GPU             : Keine GPU – verwende CPU")

    logger.info(f"Gerät           : {DEVICE.upper()}")
    logger.info("=" * 50)


# ── Datensatz Statistiken ─────────────────────────────────────
def print_dataset_info(
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    test_loader:  torch.utils.data.DataLoader,
) -> None:
    """
    Zeigt Übersicht über den Datensatz.

    Args:
        train_loader : DataLoader für Trainingsdaten
        val_loader   : DataLoader für Validierungsdaten
        test_loader  : DataLoader für Testdaten
    """
    total = (
        len(train_loader.dataset) +
        len(val_loader.dataset)   +
        len(test_loader.dataset)
    )

    logger.info("=" * 50)
    logger.info("Datensatz Übersicht")
    logger.info("=" * 50)
    logger.info(f"Training   : {len(train_loader.dataset):>6,} Bilder")
    logger.info(f"Validierung: {len(val_loader.dataset)  :>6,} Bilder")
    logger.info(f"Test       : {len(test_loader.dataset) :>6,} Bilder")
    logger.info(f"Total      : {total                    :>6,} Bilder")
    logger.info(f"Batch-Size : {train_loader.batch_size  :>6}")
    logger.info(f"Batches    : {len(train_loader)        :>6} (Training)")
    logger.info("=" * 50)


# ── Beispielbilder anzeigen ───────────────────────────────────
def plot_sample_images(
    loader:    torch.utils.data.DataLoader,
    n_images:  int  = 16,
    save_path: Path = PLOTS_DIR / "sample_images.png",
) -> None:
    """
    Beispielbilder aus dem DataLoader anzeigen.

    Nützlich um zu prüfen ob:
        - Bilder korrekt geladen werden
        - Labels korrekt vergeben sind
        - Augmentierung sinnvoll aussieht

    Args:
        loader    : DataLoader
        n_images  : Anzahl Bilder anzeigen
        save_path : Speicherpfad
    """
    images, labels = next(iter(loader))

    cols = 4
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(min(n_images, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225])
        img = img + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        label = IDX_TO_CLASS[labels[i].item()]
        color = "red" if label == "infected" else "green"

        axes[i].imshow(img)
        axes[i].set_title(label, fontsize=10, color=color,
                          fontweight="bold")
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Beispielbilder aus dem Datensatz",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Beispielbilder gespeichert: {save_path}")


# ── Klassenverteilung plotten ─────────────────────────────────
def plot_class_distribution(
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    test_loader:  torch.utils.data.DataLoader,
    save_path:    Path = PLOTS_DIR / "class_distribution.png",
) -> None:
    """
    Klassenverteilung aller Splits als Balkendiagramm.

    Args:
        train_loader : DataLoader Training
        val_loader   : DataLoader Validierung
        test_loader  : DataLoader Test
        save_path    : Speicherpfad
    """
    def _count_labels(
        loader: torch.utils.data.DataLoader,
    ) -> Dict[str, int]:
        counts = {name: 0 for name in CLASS_NAMES}
        for _, labels in loader:
            for label in labels:
                counts[IDX_TO_CLASS[label.item()]] += 1
        return counts

    train_counts = _count_labels(train_loader)
    val_counts   = _count_labels(val_loader)
    test_counts  = _count_labels(test_loader)

    x      = np.arange(len(CLASS_NAMES))
    width  = 0.25
    colors = ["royalblue", "tomato", "seagreen"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, train_counts.values(),
           width, label="Training",    color=colors[0], alpha=0.85)
    ax.bar(x,         val_counts.values(),
           width, label="Validierung", color=colors[1], alpha=0.85)
    ax.bar(x + width, test_counts.values(),
           width, label="Test",        color=colors[2], alpha=0.85)

    ax.set_title ("Klassenverteilung", fontsize=14, fontweight="bold")
    ax.set_xlabel("Klasse",            fontsize=12)
    ax.set_ylabel("Anzahl Bilder",     fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES,    fontsize=11)
    ax.legend    (fontsize=11)
    ax.grid      (True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Klassenverteilung gespeichert: {save_path}")


# ── Early Stopping Visualisierung ────────────────────────────
def plot_early_stopping(
    history:          Dict[str, list],
    stopped_at_epoch: int,
    save_path:        Path = PLOTS_DIR / "early_stopping.png",
) -> None:
    """
    Loss-Plot mit Markierung wo Early Stopping ausgelöst wurde.

    Zeigt:
        - Train & Val Loss über alle Epochen
        - Rote vertikale Linie = Punkt wo Training gestoppt wurde
        - Grüner Punkt        = Bestes Modell

    Args:
        history          : Dict aus train() mit train_loss & val_loss
        stopped_at_epoch : Epoch bei der Early Stopping ausgelöst wurde
        save_path        : Speicherpfad
    """
    epochs   = range(1, len(history["train_loss"]) + 1)
    best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1

    fig, ax = plt.subplots(figsize=(12, 6))

    # Loss Kurven
    ax.plot(epochs, history["train_loss"],
            color="royalblue", linewidth=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"],
            color="tomato",    linewidth=2, label="Val Loss")

    # Early Stopping Linie
    ax.axvline(
        x         = stopped_at_epoch,
        color     = "red",
        linestyle = "--",
        linewidth = 1.5,
        label     = f"Early Stopping (Epoch {stopped_at_epoch})",
    )

    # Bestes Modell markieren
    ax.scatter(
        best_epoch,
        history["val_loss"][best_epoch - 1],
        color  = "green",
        s      = 120,
        zorder = 5,
        label  = f"Bestes Modell (Epoch {best_epoch})",
    )

    ax.set_title ("Loss mit Early Stopping", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch",                   fontsize=12)
    ax.set_ylabel("Loss",                    fontsize=12)
    ax.legend    (fontsize=11)
    ax.grid      (True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Early Stopping Plot gespeichert: {save_path}")


# ── Sanity Check ──────────────────────────────────────────────
def sanity_check(
    model:  torch.nn.Module,
    loader: torch.utils.data.DataLoader,
) -> bool:
    """
    Schneller Check vor dem Training ob alles korrekt funktioniert.

    Prüft:
        ✓ Kann ein Batch durch das Modell?
        ✓ Gibt es NaN Werte in Outputs?
        ✓ Stimmen die Output-Dimensionen?
        ✓ Funktioniert GPU Transfer?
        ✓ Sind Labels im gültigen Bereich?

    Args:
        model  : Das Modell
        loader : DataLoader

    Returns:
        True wenn alles stimmt, False wenn Fehler gefunden
    """
    logger.info("=" * 50)
    logger.info("Sanity Check")
    logger.info("=" * 50)

    try:
        # ── 1. Batch laden ────────────────────────────────
        images, labels = next(iter(loader))
        logger.info(f"✓ Batch geladen: {list(images.shape)}")

        # ── 2. GPU Transfer ───────────────────────────────
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        logger.info(f"✓ GPU Transfer:  {DEVICE.upper()}")

        # ── 3. Vorwärtsdurchlauf ──────────────────────────
        model.eval()
        with torch.no_grad():
            outputs = model(images)
        logger.info(f"✓ Output Shape:  {list(outputs.shape)}")

        # ── 4. NaN Check ──────────────────────────────────
        if torch.isnan(outputs).any():
            logger.error("✗ NaN Werte in Outputs gefunden!")
            return False
        logger.info("✓ Keine NaN Werte")

        # ── 5. Output Dimensionen ─────────────────────────
        from src.config import NUM_CLASSES
        if outputs.shape[1] != NUM_CLASSES:
            logger.error(
                f"✗ Output hat {outputs.shape[1]} Klassen "
                f"aber {NUM_CLASSES} erwartet!"
            )
            return False
        logger.info(f"✓ Klassen: {NUM_CLASSES}")

        # ── 6. Labels Check ───────────────────────────────
        if labels.min() < 0 or labels.max() >= NUM_CLASSES:
            logger.error(
                f"✗ Labels ausserhalb gültigem Bereich: "
                f"min={labels.min()}, max={labels.max()}"
            )
            return False
        logger.info(f"✓ Labels gültig: 0 – {NUM_CLASSES - 1}")

        # ── 7. Kurzer Trainingsschritt ────────────────────
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if torch.isnan(loss):
            logger.error("✗ Loss ist NaN – Training würde fehlschlagen!")
            return False
        logger.info(f"✓ Trainingsschritt: Loss = {loss.item():.4f}")

        logger.info("=" * 50)
        logger.info("✓ Sanity Check bestanden – Training kann starten!")
        logger.info("=" * 50)
        return True

    except Exception as e:
        logger.error(f"✗ Sanity Check fehlgeschlagen: {e}")
        return False


# ── Requirements.txt Update ───────────────────────────────────
def update_requirements(
    output_path: Path,
) -> None:
    """
    Aktuelle pip Pakete automatisch in requirements.txt schreiben.

    Warum:
        Nach jedem pip install ändert sich die Umgebung
        → requirements.txt manuell updaten vergisst man
        → diese Funktion macht es automatisch

    Args:
        output_path : Pfad zur requirements.txt
    """
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output = True,
            text           = True,
            check          = True,
        )
        with open(output_path, "w") as f:
            f.write(result.stdout)

        # Anzahl Pakete zählen
        n_packages = len(result.stdout.strip().splitlines())
        logger.info(
            f"requirements.txt aktualisiert: "
            f"{n_packages} Pakete → {output_path}"
        )

    except subprocess.CalledProcessError as e:
        logger.error(f"requirements.txt Update fehlgeschlagen: {e}")


# ── Checkpoint Verwaltung ─────────────────────────────────────
def cleanup_checkpoints(
    checkpoint_dir: Path,
    keep_best:      int = 3,
) -> None:
    """
    Alte Checkpoints löschen, nur die besten behalten.

    Args:
        checkpoint_dir : Ordner mit Checkpoints
        keep_best      : Wie viele Checkpoints behalten
    """
    checkpoints = sorted(
        checkpoint_dir.glob("*.pth"),
        key     = os.path.getmtime,
        reverse = True,
    )

    to_delete = checkpoints[keep_best:]
    for ckpt in to_delete:
        ckpt.unlink()
        logger.info(f"Checkpoint gelöscht: {ckpt.name}")

    logger.info(
        f"Checkpoints aufgeräumt: "
        f"{len(to_delete)} gelöscht, "
        f"{min(len(checkpoints), keep_best)} behalten"
    )


# ── Metriken laden ────────────────────────────────────────────
def load_metrics(metrics_path: Path) -> Dict[str, float]:
    """
    Gespeicherte Metriken aus JSON laden.

    Args:
        metrics_path : Pfad zur JSON Datei

    Returns:
        Dict mit Metriken
    """
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metriken nicht gefunden: {metrics_path}\n"
            f"→ Zuerst evaluate.py ausführen."
        )

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    logger.info(f"Metriken geladen: {metrics_path}")
    return metrics


# ── Trainingszeit formatieren ─────────────────────────────────
def format_time(seconds: float) -> str:
    """
    Sekunden in lesbares Format umwandeln.

    Beispiele:
        45    → "45s"
        130   → "2m 10s"
        3700  → "1h 1m 40s"

    Args:
        seconds : Zeitdauer in Sekunden

    Returns:
        Formatierter String
    """
    seconds = int(seconds)
    hours   = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs    = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# ── Fortschrittsbalken ────────────────────────────────────────
def progress_bar(
    current: int,
    total:   int,
    width:   int = 40,
) -> str:
    """
    Einfacher Fortschrittsbalken für Terminal.

    Beispiel:
        [████████████████░░░░░░░░] 65% (13/20)

    Args:
        current : Aktueller Schritt
        total   : Gesamtschritte
        width   : Breite des Balkens

    Returns:
        Formatierter String
    """
    percent = current / total
    filled  = int(width * percent)
    bar     = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent:.0%} ({current}/{total})"


# ── Quick-Test: python -m src.utils ──────────────────────────
if __name__ == "__main__":
    from src.config import BASE_DIR, PROCESSED_DIR
    from src.dataset import get_dataloaders
    from src.model   import build_model

    # ── Seed testen ───────────────────────────────────────
    set_seed()

    # ── System Info ───────────────────────────────────────
    print_system_info()

    # ── Zeitformatierung testen ───────────────────────────
    print("\nZeitformatierung:")
    for secs in [45, 130, 3700]:
        print(f"  {secs}s → {format_time(secs)}")

    # ── Fortschrittsbalken testen ─────────────────────────
    print("\nFortschrittsbalken:")
    for i in [5, 10, 15, 20]:
        print(f"  {progress_bar(i, 20)}")

    # ── Sanity Check testen ───────────────────────────────
    print("\nSanity Check:")
    loaders = get_dataloaders(
        data_dir    = PROCESSED_DIR,
        batch_size  = 4,
        num_workers = 0,
        pin_memory  = False,
    )
    model = build_model()
    passed = sanity_check(model, loaders["train"])
    print(f"  Sanity Check: {'✓ bestanden' if passed else '✗ fehlgeschlagen'}")

    # ── Requirements updaten ──────────────────────────────
    update_requirements(BASE_DIR / "requirements.txt")

    print("\n✓ utils.py funktioniert korrekt.")