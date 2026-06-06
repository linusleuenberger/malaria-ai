# ============================================================
# preprocessing/normalization.py
# Pixelwerte normalisieren für stabiles Training
# ============================================================

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ── 0. Input Validierung ──────────────────────────────────────
def _validate_image(image: np.ndarray) -> None:
    """
    Prüft ob Bild gültig ist bevor normalisiert wird.

    Checks:
        ✓ Ist es ein NumPy Array?
        ✓ Hat es 3 Kanäle (RGB/BGR)?
        ✓ Ist es nicht leer?
        ✓ Ist der Wertebereich sinnvoll?

    Args:
        image : Zu prüfendes Bild

    Raises:
        TypeError  : Falls kein NumPy Array
        ValueError : Falls Dimensionen oder Werte falsch
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Bild muss ein NumPy Array sein, "
            f"nicht {type(image).__name__}"
        )

    if image.ndim != 3:
        raise ValueError(
            f"Bild muss 3 Dimensionen haben (H, W, C), "
            f"nicht {image.ndim}"
        )

    if image.shape[2] != 3:
        raise ValueError(
            f"Bild muss 3 Kanäle haben (RGB/BGR), "
            f"nicht {image.shape[2]}"
        )

    if image.size == 0:
        raise ValueError("Bild ist leer (size=0)")

    if np.all(image == 0):
        raise ValueError("Bild ist komplett schwarz")

    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() > 1.0 + 1e-6:
            logger.warning(
                f"Float-Bild hat Werte > 1.0 (max={image.max():.3f}). "
                f"Erwartet: 0.0–1.0"
            )


# ── 1. Min-Max Normalisierung ─────────────────────────────────
def normalize_minmax(
    image:   np.ndarray,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """
    Min-Max Normalisierung – Werte in festen Bereich bringen.

    Formel:
        normalized = (pixel - min) / (max - min)
        scaled     = normalized × (out_max - out_min) + out_min

    Beispiel (out_min=0, out_max=1):
        Pixel 0   → 0.0
        Pixel 128 → 0.502
        Pixel 255 → 1.0

    Wann verwenden:
        → Einfachste Normalisierung
        → Wenn Wertebereich bekannt (0–255)
        → Keine Annahmen über Verteilung nötig

    Nachteil:
        → Empfindlich auf Ausreisser
        → Ein sehr heller Pixel → alle anderen zu dunkel

    Args:
        image   : Bild als NumPy Array (beliebiger Wertebereich)
        out_min : Minimaler Ausgabewert (Standard: 0.0)
        out_max : Maximaler Ausgabewert (Standard: 1.0)

    Returns:
        Normalisiertes Bild als float32
    """
    _validate_image(image)

    image   = image.astype(np.float32)
    img_min = image.min()
    img_max = image.max()

    if img_max == img_min:
        logger.warning(
            "Bild hat konstante Pixelwerte – "
            "Normalisierung übersprungen."
        )
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - img_min) / (img_max - img_min)
    scaled     = normalized * (out_max - out_min) + out_min

    return scaled.astype(np.float32)


# ── 2. Z-Score Standardisierung ───────────────────────────────
def normalize_zscore(
    image:       np.ndarray,
    mean:        Optional[np.ndarray] = None,
    std:         Optional[np.ndarray] = None,
    per_channel: bool                 = True,
) -> np.ndarray:
    """
    Z-Score Standardisierung – Mittelwert 0, Standardabweichung 1.

    Formel:
        standardized = (pixel - mean) / std

    Beispiel (mean=0.5, std=0.2):
        Pixel 0.7 → (0.7 - 0.5) / 0.2 = +1.0
        Pixel 0.3 → (0.3 - 0.5) / 0.2 = -1.0
        Pixel 0.5 → (0.5 - 0.5) / 0.2 =  0.0

    Warum besser als Min-Max:
        → Robuster gegen Ausreisser
        → Neuronale Netze lernen besser mit
          zentrierten Werten um 0

    per_channel=True:
        → Jeden RGB-Kanal separat normalisieren
        → Farbinformationen bleiben erhalten
        → Empfohlen für Blutbilder

    Args:
        image       : Bild als NumPy Array (H, W, 3)
        mean        : Mittelwert (falls None → aus Bild berechnen)
        std         : Standardabweichung (falls None → aus Bild)
        per_channel : Pro Kanal oder global normalisieren

    Returns:
        Standardisiertes Bild als float32
    """
    _validate_image(image)

    image  = image.astype(np.float32)
    result = np.zeros_like(image)

    if per_channel:
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            ch_mean = mean[c] if mean is not None else channel.mean()
            ch_std  = std[c]  if std  is not None else channel.std()

            if ch_std < 1e-8:
                logger.warning(
                    f"Kanal {c} hat Standardabweichung ≈ 0 "
                    f"→ Kanal unverändert."
                )
                result[:, :, c] = channel - ch_mean
            else:
                result[:, :, c] = (channel - ch_mean) / ch_std
    else:
        img_mean = image.mean() if mean is None else mean
        img_std  = image.std()  if std  is None else std

        if img_std < 1e-8:
            return image - img_mean

        result = (image - img_mean) / img_std

    return result.astype(np.float32)


# ── 3. ImageNet Normalisierung ────────────────────────────────
def normalize_imagenet(
    image: np.ndarray,
) -> np.ndarray:
    """
    ImageNet Normalisierung – speziell für ResNet50.

    Warum diese Werte:
        ResNet50 auf 1.2 Millionen ImageNet Bilder trainiert.
        Diese Werte sind der Kanal-Durchschnitt aller Bilder.
        Modell erwartet GENAU diese Normalisierung.

    ImageNet Werte (RGB):
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    Voraussetzung:
        Bild muss bereits in 0.0–1.0 Bereich sein
        → zuerst normalize_minmax() aufrufen

    Args:
        image : BGR Bild als NumPy Array

    Returns:
        ImageNet-normalisiertes Bild als float32
    """
    _validate_image(image)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # BGR → RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_f   = image_rgb.astype(np.float32)

    if image_f.max() > 1.0:
        image_f /= 255.0

    normalized = (image_f - mean) / std

    return normalized.astype(np.float32)


# ── 4. Denormalisierung ───────────────────────────────────────
def denormalize_imagenet(
    image: np.ndarray,
) -> np.ndarray:
    """
    ImageNet Normalisierung rückgängig machen.

    Warum:
        Nach Normalisierung sind Pixelwerte negativ/> 1.0
        → Für Visualisierung (Grad-CAM, misclassified)
          muss man zurück zu 0–255 konvertieren

    Formel (Umkehrung):
        pixel = (normalized × std) + mean
        pixel = pixel × 255 → zurück zu 0–255

    Wann verwenden:
        → evaluate.py: Grad-CAM Heatmaps
        → evaluate.py: Falsch klassifizierte Bilder
        → Immer wenn normalisierte Tensoren angezeigt werden

    Args:
        image : ImageNet-normalisiertes Bild (float32)

    Returns:
        Bild als uint8 (0–255) im BGR Format
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Normalisierung rückgängig: pixel = (norm × std) + mean
    denormalized = image * std + mean

    # Auf 0–1 clippen (Rundungsfehler)
    denormalized = np.clip(denormalized, 0.0, 1.0)

    # 0–1 → 0–255
    denormalized = (denormalized * 255).astype(np.uint8)

    # RGB → BGR für OpenCV
    return cv2.cvtColor(denormalized, cv2.COLOR_RGB2BGR)


def denormalize_zscore(
    image: np.ndarray,
    mean:  np.ndarray,
    std:   np.ndarray,
) -> np.ndarray:
    """
    Z-Score Normalisierung rückgängig machen.

    Formel:
        pixel = (normalized × std) + mean

    Args:
        image : Z-Score normalisiertes Bild
        mean  : Original Mean (aus compute_channel_statistics)
        std   : Original Std  (aus compute_channel_statistics)

    Returns:
        Bild als float32 (0.0–1.0)
    """
    denormalized = image * std + mean
    return np.clip(denormalized, 0.0, 1.0).astype(np.float32)


# ── 5. Percentile Normalisierung ──────────────────────────────
def normalize_percentile(
    image:         np.ndarray,
    lower_percent: float = 2.0,
    upper_percent: float = 98.0,
) -> np.ndarray:
    """
    Percentile Normalisierung – robust gegen Ausreisser.

    Problem mit Min-Max:
        Ein einzelner sehr heller Pixel (Ausreisser)
        → alle anderen Pixel erscheinen zu dunkel

    Lösung Percentile:
        Statt Min/Max → 2. und 98. Percentile verwenden
        → Ausreisser werden ignoriert

    Beispiel:
        Werte: [10, 50, 100, 150, 200, 255, 255, 255]
        Min-Max:    255 als max → alles zu dunkel
        Percentile: 98% ≈ 200  → viel besser

    Args:
        image         : Bild als NumPy Array
        lower_percent : Unteres Percentile (Standard: 2%)
        upper_percent : Oberes  Percentile (Standard: 98%)

    Returns:
        Normalisiertes Bild als float32 (0.0–1.0)
    """
    _validate_image(image)

    image  = image.astype(np.float32)
    p_low  = np.percentile(image, lower_percent)
    p_high = np.percentile(image, upper_percent)

    if p_high == p_low:
        logger.warning("Percentile gleich – Min-Max als Fallback.")
        return normalize_minmax(image)

    clipped    = np.clip(image, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low)

    return normalized.astype(np.float32)


# ── 6. Datensatz-Statistiken berechnen ───────────────────────
def compute_channel_statistics(
    image_paths: List[Path],
    sample_size: int  = 1000,
    cache_path:  Path = Path("results/metrics/dataset_stats.json"),
) -> Dict[str, Tuple[float, float, float]]:
    """
    Kanal-Statistiken des Datensatzes berechnen mit Caching.

    Warum Caching:
        1000 Bilder laden dauert lange → nervt beim Entwickeln
        Einmal berechnen → als JSON speichern
        Beim nächsten Aufruf → einfach aus JSON laden
        → spart Zeit bei wiederholten Aufrufen

    Workflow:
        1. Erstes Mal: berechnen + in JSON speichern
        2. Ab zweitem Mal: aus JSON laden (sofort)
        3. Cache löschen falls neuer Datensatz

    Args:
        image_paths : Liste aller Bildpfade
        sample_size : Anzahl Bilder für Stichprobe
        cache_path  : Pfad für gecachte Statistiken

    Returns:
        Dict mit:
            mean : (R_mean, G_mean, B_mean)
            std  : (R_std,  G_std,  B_std)
    """
    # ── Cache prüfen ──────────────────────────────────────
    if cache_path.exists():
        with open(cache_path, "r") as f:
            stats = json.load(f)
        logger.info(
            f"Statistiken aus Cache geladen: {cache_path}\n"
            f"  mean: {stats['mean']}\n"
            f"  std:  {stats['std']}"
        )
        return stats

    # ── Neu berechnen ─────────────────────────────────────
    import random
    sample = random.sample(
        image_paths,
        min(sample_size, len(image_paths))
    )

    r_values, g_values, b_values = [], [], []

    logger.info(
        f"Berechne Statistiken aus "
        f"{len(sample)} Bildern..."
    )

    for path in sample:
        img = cv2.imread(str(path))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        r_values.append(img[:, :, 0].mean())
        g_values.append(img[:, :, 1].mean())
        b_values.append(img[:, :, 2].mean())

    stats = {
        "mean": (
            round(float(np.mean(r_values)), 4),
            round(float(np.mean(g_values)), 4),
            round(float(np.mean(b_values)), 4),
        ),
        "std": (
            round(float(np.std(r_values)), 4),
            round(float(np.std(g_values)), 4),
            round(float(np.std(b_values)), 4),
        ),
    }

    # ── Cache speichern ───────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"  mean (R,G,B) = {stats['mean']}")
    logger.info(f"  std  (R,G,B) = {stats['std']}")
    logger.info(f"  → Gespeichert: {cache_path}")
    logger.info("  → In config.py unter MEAN & STD eintragen!")

    return stats


# ── 7. Gesamte Normalisierungs-Pipeline ──────────────────────
def apply_normalization_pipeline(
    image:  np.ndarray,
    method: str            = "imagenet",
    mean:   Optional[Tuple] = None,
    std:    Optional[Tuple] = None,
) -> np.ndarray:
    """
    Normalisierungs-Pipeline – eine Methode wählen.

    Methoden:
        "imagenet"   → für ResNet50 (empfohlen)
        "zscore"     → mit eigenen Mean/Std Werten
        "minmax"     → einfach, 0.0–1.0
        "percentile" → robust gegen Ausreisser

    Args:
        image  : BGR Bild als NumPy Array
        method : Normalisierungsmethode
        mean   : Mean für zscore (falls None → aus Bild)
        std    : Std  für zscore (falls None → aus Bild)

    Returns:
        Normalisiertes Bild als float32
    """
    if method == "imagenet":
        return normalize_imagenet(image)

    elif method == "zscore":
        img_f    = image.astype(np.float32) / 255.0
        mean_arr = np.array(mean) if mean else None
        std_arr  = np.array(std)  if std  else None
        return normalize_zscore(img_f, mean_arr, std_arr)

    elif method == "minmax":
        return normalize_minmax(image)

    elif method == "percentile":
        return normalize_percentile(image)

    else:
        raise ValueError(
            f"Unbekannte Methode: '{method}'\n"
            f"Optionen: 'imagenet', 'zscore', 'minmax', 'percentile'"
        )


# ── 8. Visualisierung ─────────────────────────────────────────
def plot_normalization_comparison(
    image:     np.ndarray,
    save_path: Path = Path("results/plots/normalization_comparison.png"),
) -> None:
    """
    Alle Normalisierungsmethoden nebeneinander visualisieren.

    Zeigt pro Methode:
        Obere Reihe: Bild nach Normalisierung
        Untere Reihe: Histogramm der Pixelverteilung
            → Sofort sehen wie Verteilung sich verändert

    Nützlich für:
        → Verstehen was Normalisierung bewirkt
        → Richtige Methode wählen
        → ETH-Präsentation

    Args:
        image     : BGR Originalbild
        save_path : Speicherpfad
    """
    _validate_image(image)

    methods = {
        "Original\n(0–255)"  : image.astype(np.float32),
        "Min-Max\n(0–1)"     : normalize_minmax(image),
        "Percentile\n(0–1)"  : normalize_percentile(image),
        "Z-Score\n(um 0)"    : normalize_zscore(
                                    image.astype(np.float32) / 255.0
                               ),
        "ImageNet\n(ResNet)" : normalize_imagenet(image),
    }

    n_methods = len(methods)
    fig, axes = plt.subplots(
        2, n_methods,
        figsize=(n_methods * 3, 7)
    )

    for col, (name, img) in enumerate(methods.items()):

        # ── Obere Reihe: Bild ─────────────────────────────
        # Für Anzeige auf 0–1 clippen/skalieren
        display = img.copy()
        if display.max() > 1.0 or display.min() < 0.0:
            d_min   = display.min()
            d_max   = display.max()
            display = (display - d_min) / (d_max - d_min + 1e-8)

        display = np.clip(display, 0.0, 1.0)

        # BGR → RGB falls nötig
        if display.shape[2] == 3:
            display_rgb = display[:, :, ::-1]
        else:
            display_rgb = display

        axes[0, col].imshow(display_rgb)
        axes[0, col].set_title(name, fontsize=9, fontweight="bold")
        axes[0, col].axis("off")

        # ── Untere Reihe: Histogramm ──────────────────────
        flat = img.flatten()
        axes[1, col].hist(
            flat,
            bins   = 50,
            color  = "royalblue",
            alpha  = 0.75,
            density= True,
        )
        axes[1, col].set_xlabel(
            f"Wert\n[{flat.min():.2f}, {flat.max():.2f}]",
            fontsize=8
        )
        axes[1, col].set_ylabel("Dichte", fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

    plt.suptitle(
        "Normalisierungsmethoden – Vergleich",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Normalisierungsvergleich gespeichert: {save_path}")


# ── Quick-Test: python -m preprocessing.normalization ────────
if __name__ == "__main__":
    import sys
    from src.config import RAW_DIR, PLOTS_DIR, METRICS_DIR

    test_images = (
        list(RAW_DIR.rglob("*.png")) +
        list(RAW_DIR.rglob("*.jpg"))
    )

    if not test_images:
        logger.error("Keine Testbilder in data/raw/ gefunden.")
        sys.exit(1)

    img = cv2.imread(str(test_images[0]))
    logger.info(f"Original Shape:  {img.shape}")
    logger.info(f"Original Range:  {img.min()} – {img.max()}")

    # ── Validierung testen ─────────────────────────────────
    logger.info("Input Validierung:")
    try:
        _validate_image(img)
        logger.info("  ✓ Validierung bestanden")
    except (TypeError, ValueError) as e:
        logger.error(f"  ✗ Fehler: {e}")

    # ── Alle Methoden testen ───────────────────────────────
    logger.info("Normalisierung Tests:")
    tests = {
        "minmax"     : apply_normalization_pipeline(img, "minmax"),
        "percentile" : apply_normalization_pipeline(img, "percentile"),
        "imagenet"   : apply_normalization_pipeline(img, "imagenet"),
        "zscore"     : apply_normalization_pipeline(img, "zscore"),
    }

    for name, result in tests.items():
        logger.info(
            f"  ✓ {name:<12} "
            f"Range: [{result.min():.3f}, {result.max():.3f}]  "
            f"Mean: {result.mean():.3f}"
        )

    # ── Denormalisierung testen ────────────────────────────
    logger.info("Denormalisierung:")
    norm   = normalize_imagenet(img)
    denorm = denormalize_imagenet(norm)
    diff   = np.abs(img.astype(np.float32) - denorm.astype(np.float32))
    logger.info(f"  ✓ Max Differenz nach Denorm: {diff.max():.2f} Pixel")

    # ── Statistiken mit Cache testen ──────────────────────
    logger.info("Datensatz-Statistiken (mit Cache):")
    stats = compute_channel_statistics(
        test_images,
        sample_size = 50,
        cache_path  = METRICS_DIR / "dataset_stats.json",
    )
    logger.info(f"  mean: {stats['mean']}")
    logger.info(f"  std:  {stats['std']}")

    # Zweiter Aufruf → aus Cache laden
    stats_cached = compute_channel_statistics(
        test_images,
        cache_path = METRICS_DIR / "dataset_stats.json",
    )
    logger.info(f"  ✓ Cache funktioniert: {stats == stats_cached}")

    # ── Visualisierung testen ──────────────────────────────
    logger.info("Visualisierung:")
    plot_normalization_comparison(
        img,
        save_path = PLOTS_DIR / "normalization_comparison.png"
    )
    logger.info(
        f"  ✓ Gespeichert: "
        f"{PLOTS_DIR / 'normalization_comparison.png'}"
    )

    logger.info("✓ normalization.py funktioniert korrekt.")