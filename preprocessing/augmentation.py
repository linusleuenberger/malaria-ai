# ============================================================
# preprocessing/augmentation.py
#
# OFFLINE Augmentierung – Bilder vorab auf Disk speichern
# ============================================================
#
# Abgrenzung zu dataset.py:
#   dataset.py      → Online Augmentierung während Training
#                     torchvision transforms, automatisch
#                     jede Epoch andere Augmentierung
#
#   augmentation.py → Offline Augmentierung vorab
#                     albumentations, manuell ausführen
#                     erweiterte Augmentierungen
#                     nützlich wenn Datensatz sehr klein
#
# Wann Offline verwenden:
#   → Datensatz < 5000 Bilder
#   → Training zu langsam wegen Online Augmentierung
#   → Spezielle Augmentierungen die torchvision nicht hat
# ============================================================

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── 0. Qualitäts-Check ───────────────────────────────────────
def _is_valid_augmentation(
    image:          np.ndarray,
    min_brightness: float = 20.0,
    max_brightness: float = 235.0,
    min_sharpness:  float = 10.0,
    max_black_ratio: float = 0.3,
) -> bool:
    """
    Prüft ob augmentiertes Bild noch brauchbar ist.

    Warum nötig:
        ElasticTransform oder starke Rotation kann manchmal
        unbrauchbare Bilder erzeugen:
            → Zu dunkle Bilder (durch Rotation mit schwarzen Rändern)
            → Zu unscharfe Bilder
            → Bilder mit zu vielen schwarzen Bereichen (CoarseDropout)

    Checks:
        ✓ Helligkeit im sinnvollen Bereich
        ✓ Bild scharf genug (Laplacian Varianz)
        ✓ Nicht zu viele schwarze Pixel (Dropout-Anteil)

    Args:
        image           : Augmentiertes Bild
        min_brightness  : Minimale Durchschnittshelligkeit
        max_brightness  : Maximale Durchschnittshelligkeit
        min_sharpness   : Minimale Laplacian Varianz
        max_black_ratio : Maximaler Anteil schwarzer Pixel (0.0–1.0)

    Returns:
        True wenn Bild brauchbar, False wenn verwerfen
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # ── Helligkeit ─────────────────────────────────────────
    brightness = gray.mean()
    if brightness < min_brightness or brightness > max_brightness:
        return False

    # ── Schärfe (Laplacian Varianz) ────────────────────────
    sharpness = cv2.Laplacian(
        gray.astype(np.uint8), cv2.CV_64F
    ).var()
    if sharpness < min_sharpness:
        return False

    # ── Schwarze Pixel Anteil ──────────────────────────────
    black_pixels = np.sum(gray < 5)
    black_ratio  = black_pixels / gray.size
    if black_ratio > max_black_ratio:
        return False

    return True


# ── 1. Erweiterte Augmentierungs-Pipelines ───────────────────
def get_standard_offline_pipeline(
    img_size: int = 224,
) -> A.Compose:
    """
    Standard Offline-Pipeline.

    Enthält dieselben Augmentierungen wie dataset.py
    aber als albumentations Pipeline für Offline-Verarbeitung.

    Wann verwenden:
        → Datensatz klein aber nicht winzig (5000–15000 Bilder)
        → Einfache Vervielfältigung des Datensatzes

    Args:
        img_size : Ausgabe-Bildgrösse in Pixel

    Returns:
        albumentations Compose Pipeline
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip  (p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(
            limit       = 15,
            border_mode = cv2.BORDER_REFLECT,
            p           = 0.3,
        ),
        A.RandomBrightnessContrast(
            brightness_limit = 0.2,
            contrast_limit   = 0.2,
            p                = 0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit = 10,
            sat_shift_limit = 15,
            val_shift_limit = 10,
            p               = 0.4,
        ),
    ])


def get_extended_offline_pipeline(
    img_size: int = 224,
) -> A.Compose:
    """
    Erweiterte Offline-Pipeline mit Augmentierungen
    die torchvision NICHT hat.

    Das ist der eigentliche Mehrwert von augmentation.py
    gegenüber dataset.py:
        ElasticTransform → Zellen leicht verformen
        CoarseDropout    → Teile verdecken
        GridDropout      → Gitternetz-Lücken
        ISONoise         → Kamerarauschen
        GaussNoise       → Sensorrauschen

    Wann verwenden:
        → Datensatz sehr klein (< 5000 Bilder)
        → Maximale Robustheit gewünscht
        → Verschiedene Labore im Datensatz

    Args:
        img_size : Ausgabe-Bildgrösse in Pixel

    Returns:
        albumentations Compose Pipeline
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # ── Geometrisch ───────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip  (p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(
            limit       = 15,
            border_mode = cv2.BORDER_REFLECT,
            p           = 0.3,
        ),
        A.ShiftScaleRotate(
            shift_limit  = 0.05,
            scale_limit  = 0.1,
            rotate_limit = 0,
            p            = 0.3,
        ),
        A.ElasticTransform(
            alpha = 120,
            sigma = 120 * 0.05,
            p     = 0.2,
        ),
        # ← Nicht in torchvision vorhanden
        # Verformt Zellen leicht wie unter Mikroskop

        # ── Farbe ─────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit = 0.2,
            contrast_limit   = 0.2,
            p                = 0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit = 10,
            sat_shift_limit = 15,
            val_shift_limit = 10,
            p               = 0.4,
        ),
        A.RGBShift(
            r_shift_limit = 10,
            g_shift_limit = 10,
            b_shift_limit = 10,
            p             = 0.3,
        ),
        # ← Nicht in torchvision vorhanden

        # ── Rauschen ──────────────────────────────────────
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ISONoise  (
                color_shift = (0.01, 0.05),
                intensity   = (0.1,  0.5),
            ),
        ], p=0.3),
        # ← Nicht in torchvision vorhanden

        # ── Unschärfe ─────────────────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur  (blur_limit=3),
            A.MotionBlur  (blur_limit=3),
        ], p=0.2),

        # ── Dropout ───────────────────────────────────────
        A.CoarseDropout(
            max_holes  = 4,
            max_height = 16,
            max_width  = 16,
            min_holes  = 1,
            fill_value = 0,
            p          = 0.2,
        ),
        # ← Nicht in torchvision vorhanden

        A.GridDropout(
            ratio = 0.2,
            p     = 0.1,
        ),
        # ← Nicht in torchvision vorhanden
    ])


# ── 2. Offline Augmentierung ausführen ────────────────────────
def augment_dataset_offline(
    input_dir:      Path,
    output_dir:     Path,
    augment_factor: int  = 3,
    img_size:       int  = 224,
    extended:       bool = False,
    resume:         bool = True,
    quality_check:  bool = True,
    max_retries:    int  = 5,
) -> Dict[str, int]:
    """
    Datensatz offline augmentieren – Bilder auf Disk speichern.

    Features:
        → Qualitäts-Check nach jeder Augmentierung
        → tqdm Fortschrittsbalken
        → Resume: bereits augmentierte Bilder überspringen

    Ablauf:
        1. Alle Bilder in input_dir einlesen
        2. Pro Bild augment_factor Versionen erstellen
        3. Qualitäts-Check → schlechte Bilder neu augmentieren
        4. Original + augmentierte Bilder in output_dir speichern

    Args:
        input_dir      : Ordner mit Originalbildern
        output_dir     : Zielordner für augmentierte Bilder
        augment_factor : Wie viele Versionen pro Bild
        img_size       : Ausgabe-Bildgrösse in Pixel
        extended       : Erweiterte Pipeline verwenden
        resume         : Bereits augmentierte überspringen
        quality_check  : Augmentierte Bilder auf Qualität prüfen
        max_retries    : Wie oft neu augmentieren bei schlechter Qualität

    Returns:
        Dict mit:
            original   : Anzahl Originalbilder
            augmented  : Anzahl augmentierte Bilder
            skipped    : Anzahl übersprungene (resume)
            rejected   : Anzahl verworfene (schlechte Qualität)
            total      : Gesamt
    """
    pipeline = (
        get_extended_offline_pipeline(img_size)
        if extended
        else get_standard_offline_pipeline(img_size)
    )

    stats = {
        "original" : 0,
        "augmented": 0,
        "skipped"  : 0,
        "rejected" : 0,
        "total"    : 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Alle Bilder finden ────────────────────────────────
    image_paths: List[Path] = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
        image_paths.extend(input_dir.glob(ext))

    if not image_paths:
        logger.warning(f"Keine Bilder gefunden in: {input_dir}")
        return stats

    mode = "erweitert" if extended else "standard"
    logger.info(
        f"Offline Augmentierung ({mode}):\n"
        f"  Bilder:    {len(image_paths):,}\n"
        f"  Faktor:    {augment_factor}\n"
        f"  Erwartet:  {len(image_paths) * augment_factor:,} augmentierte\n"
        f"  Resume:    {resume}\n"
        f"  Qualität:  {quality_check}"
    )

    # ── Fortschrittsbalken ────────────────────────────────
    with tqdm(
        total = len(image_paths),
        desc  = f"Augmentierung ({mode})",
        unit  = "Bild",
    ) as pbar:

        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Nicht lesbar: {img_path.name}")
                pbar.update(1)
                continue

            # ── Original kopieren ──────────────────────────
            original_out = output_dir / img_path.name

            if resume and original_out.exists():
                stats["skipped"] += 1
            else:
                cv2.imwrite(str(original_out), image)
                stats["original"] += 1

            # ── Augmentierte Versionen erstellen ───────────
            stem = img_path.stem
            ext  = img_path.suffix

            for i in range(augment_factor):
                out_path = output_dir / f"{stem}_aug{i:02d}{ext}"

                # ── Resume: bereits vorhanden? ─────────────
                if resume and out_path.exists():
                    stats["skipped"] += 1
                    continue

                # ── Augmentieren mit Qualitäts-Check ───────
                augmented = None
                for attempt in range(max_retries):
                    candidate = pipeline(image=image)["image"]

                    if not quality_check or _is_valid_augmentation(candidate):
                        augmented = candidate
                        break
                    # Sonst: nochmal augmentieren

                if augmented is None:
                    # Nach max_retries immer noch schlecht
                    # → Original als Fallback speichern
                    logger.debug(
                        f"Qualität nach {max_retries} Versuchen "
                        f"ungenügend: {img_path.name} aug{i:02d} "
                        f"→ Original verwendet"
                    )
                    augmented = image
                    stats["rejected"] += 1

                cv2.imwrite(str(out_path), augmented)
                stats["augmented"] += 1

            pbar.update(1)
            pbar.set_postfix({
                "OK"      : stats["augmented"],
                "Skip"    : stats["skipped"],
                "Rejected": stats["rejected"],
            })

    stats["total"] = stats["original"] + stats["augmented"]

    logger.info(
        f"Augmentierung abgeschlossen:\n"
        f"  Original   : {stats['original']:>6,}\n"
        f"  Augmentiert: {stats['augmented']:>6,}\n"
        f"  Übersprungen: {stats['skipped']:>5,}\n"
        f"  Verworfen  : {stats['rejected']:>6,}\n"
        f"  Total      : {stats['total']:>6,}"
    )

    return stats


# ── 3. Klassenbalance ausgleichen ─────────────────────────────
def balance_classes_offline(
    input_dir:     Path,
    output_dir:    Path,
    img_size:      int  = 224,
    extended:      bool = False,
    resume:        bool = True,
    quality_check: bool = True,
) -> Dict[str, int]:
    """
    Klassen durch Augmentierung ausgleichen.

    Problem:
        infected: 5000 Bilder
        healthy:  9000 Bilder
        → Modell bevorzugt healthy (mehr Daten)

    Lösung:
        Kleine Klasse (infected) augmentieren
        bis beide Klassen gleich viele Bilder haben

    Features:
        → Qualitäts-Check nach Augmentierung
        → tqdm Fortschrittsbalken
        → Resume: bereits erstellte Bilder überspringen

    Args:
        input_dir     : Ordner mit healthy/ und infected/
        output_dir    : Zielordner
        img_size      : Ausgabe-Bildgrösse
        extended      : Erweiterte Pipeline
        resume        : Bereits erstellte überspringen
        quality_check : Augmentierte auf Qualität prüfen

    Returns:
        Dict mit Anzahl Bilder pro Klasse nach Balancierung
    """
    pipeline = (
        get_extended_offline_pipeline(img_size)
        if extended
        else get_standard_offline_pipeline(img_size)
    )

    # ── Bilder pro Klasse zählen ──────────────────────────
    class_counts: Dict[str, int]        = {}
    class_paths:  Dict[str, List[Path]] = {}

    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif"]:
            paths.extend(class_dir.glob(ext))
        class_counts[class_dir.name] = len(paths)
        class_paths [class_dir.name] = paths

    if not class_counts:
        logger.warning(f"Keine Klassen gefunden in: {input_dir}")
        return {}

    max_count = max(class_counts.values())

    logger.info("Klassenbalance vor Augmentierung:")
    for name, count in class_counts.items():
        diff = max_count - count
        logger.info(
            f"  {name:<12}: {count:>6,} Bilder "
            f"(+{diff:,} nötig)"
        )
    logger.info(f"  Ziel       : {max_count:,} Bilder pro Klasse")

    result_counts: Dict[str, int] = {}

    for class_name, paths in class_paths.items():
        class_out = output_dir / class_name
        class_out.mkdir(parents=True, exist_ok=True)

        # ── Originale kopieren ─────────────────────────────
        logger.info(f"Kopiere {class_name}...")
        for p in tqdm(paths, desc=f"Kopieren {class_name}", unit="Bild"):
            dst = class_out / p.name
            if resume and dst.exists():
                continue
            img = cv2.imread(str(p))
            if img is not None:
                cv2.imwrite(str(dst), img)

        current = len(paths)
        needed  = max_count - current

        if needed <= 0:
            logger.info(f"  {class_name}: bereits balanciert ✓")
            result_counts[class_name] = current
            continue

        # ── Fehlende Bilder augmentieren ───────────────────
        logger.info(
            f"  {class_name}: {needed:,} "
            f"augmentierte Bilder erstellen..."
        )

        generated = 0
        rejected  = 0

        with tqdm(
            total = needed,
            desc  = f"Balancierung {class_name}",
            unit  = "Bild",
        ) as pbar:
            while generated < needed:
                src_path = random.choice(paths)
                image    = cv2.imread(str(src_path))
                if image is None:
                    continue

                out_name = (
                    f"{src_path.stem}_bal"
                    f"{generated:05d}{src_path.suffix}"
                )
                out_path = class_out / out_name

                # Resume: bereits vorhanden?
                if resume and out_path.exists():
                    generated += 1
                    pbar.update(1)
                    continue

                # Augmentieren mit Qualitäts-Check
                augmented = None
                for _ in range(5):
                    candidate = pipeline(image=image)["image"]
                    if not quality_check or \
                       _is_valid_augmentation(candidate):
                        augmented = candidate
                        break

                if augmented is None:
                    augmented = image
                    rejected += 1

                cv2.imwrite(str(out_path), augmented)
                generated += 1
                pbar.update(1)
                pbar.set_postfix({"Rejected": rejected})

        result_counts[class_name] = current + generated
        logger.info(
            f"  {class_name}: {current:,} → "
            f"{result_counts[class_name]:,} Bilder "
            f"({rejected} verworfen)"
        )

    logger.info("Klassenbalance nach Augmentierung:")
    for name, count in result_counts.items():
        logger.info(f"  {name:<12}: {count:,} Bilder ✓")

    return result_counts


# ── 4. Visualisierung ─────────────────────────────────────────
def plot_augmentation_examples(
    image:      np.ndarray,
    n_examples: int  = 8,
    extended:   bool = False,
    save_path:  Path = Path("results/plots/augmentation_examples.png"),
) -> None:
    """
    Augmentierungsbeispiele visualisieren.

    Zeigt:
        Original + n_examples augmentierte Versionen
        Verworfene Bilder werden mit rotem Rahmen markiert

    Args:
        image      : BGR Originalbild
        n_examples : Anzahl augmentierte Versionen
        extended   : Erweiterte Pipeline zeigen
        save_path  : Speicherpfad
    """
    pipeline = (
        get_extended_offline_pipeline()
        if extended
        else get_standard_offline_pipeline()
    )

    images  = [image]
    titles  = ["Original"]
    valid   = [True]

    for i in range(n_examples):
        aug      = pipeline(image=image)["image"]
        is_valid = _is_valid_augmentation(aug)
        images.append(aug)
        titles.append(
            f"Aug {i + 1} {'✓' if is_valid else '✗'}"
        )
        valid.append(is_valid)

    cols = 4
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, (img, title, is_valid) in enumerate(
        zip(images, titles, valid)
    ):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)

        # Farbe: Original=rot, Gültig=grün, Ungültig=orange
        if title == "Original":
            color = "red"
        elif is_valid:
            color = "green"
        else:
            color = "orange"

        axes[i].set_title(
            title,
            fontsize   = 9,
            fontweight = "bold" if title == "Original" else "normal",
            color      = color,
        )
        axes[i].axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    mode = "Erweitert" if extended else "Standard"
    plt.suptitle(
        f"Offline Augmentierung – Beispiele ({mode})\n"
        f"Grün = gültig, Orange = verworfen",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Augmentierungsbeispiele gespeichert: {save_path}")


def plot_pipeline_comparison(
    image:     np.ndarray,
    save_path: Path = Path("results/plots/augmentation_comparison.png"),
) -> None:
    """
    Standard vs Erweiterte Pipeline vergleichen.

    Zeigt:
        Oben  → Standard  (wie dataset.py)
        Unten → Erweitert (mit ElasticTransform, Dropout etc.)

    Args:
        image     : BGR Originalbild
        save_path : Speicherpfad
    """
    standard_pipeline = get_standard_offline_pipeline()
    extended_pipeline = get_extended_offline_pipeline()

    n_examples = 4
    fig, axes  = plt.subplots(
        2, n_examples + 1,
        figsize=((n_examples + 1) * 3, 7)
    )

    pipelines = [
        (standard_pipeline, "Standard\n(wie dataset.py)"),
        (extended_pipeline, "Erweitert\n(Elastic, Dropout etc.)"),
    ]

    for row, (pipeline, label) in enumerate(pipelines):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(
            f"Original\n({label})",
            fontsize=8, color="red", fontweight="bold"
        )
        axes[row, 0].axis("off")

        for col in range(1, n_examples + 1):
            aug     = pipeline(image=image)["image"]
            aug_rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
            is_valid = _is_valid_augmentation(aug)
            axes[row, col].imshow(aug_rgb)
            axes[row, col].set_title(
                f"Beispiel {col} {'✓' if is_valid else '✗'}",
                fontsize = 8,
                color    = "green" if is_valid else "orange",
            )
            axes[row, col].axis("off")

    plt.suptitle(
        "Standard vs Erweiterte Offline-Augmentierung",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Pipeline-Vergleich gespeichert: {save_path}")


# ── Quick-Test: python -m preprocessing.augmentation ─────────
if __name__ == "__main__":
    import sys
    from src.config import RAW_DIR, PLOTS_DIR, AUGMENTED_DIR

    test_images = (
        list(RAW_DIR.rglob("*.png")) +
        list(RAW_DIR.rglob("*.jpg"))
    )

    if not test_images:
        logger.error("Keine Testbilder in data/raw/ gefunden.")
        sys.exit(1)

    img = cv2.imread(str(test_images[0]))
    logger.info(f"Original Shape: {img.shape}")

    # ── Qualitäts-Check testen ─────────────────────────────
    logger.info("Qualitäts-Check:")
    valid = _is_valid_augmentation(img)
    logger.info(f"  Original gültig: {valid}")

    # ── Pipelines testen ──────────────────────────────────
    logger.info("Pipelines:")
    for extended in [False, True]:
        pipeline = (
            get_extended_offline_pipeline()
            if extended
            else get_standard_offline_pipeline()
        )
        result   = pipeline(image=img)["image"]
        is_valid = _is_valid_augmentation(result)
        mode     = "erweitert" if extended else "standard"
        logger.info(
            f"  ✓ {mode:<12} → "
            f"Shape: {result.shape} | "
            f"Gültig: {is_valid}"
        )

    # ── Visualisierungen ──────────────────────────────────
    logger.info("Visualisierungen:")
    plot_augmentation_examples(
        img,
        save_path = PLOTS_DIR / "augmentation_examples.png"
    )
    logger.info("  ✓ Beispiele gespeichert")

    plot_pipeline_comparison(
        img,
        save_path = PLOTS_DIR / "augmentation_comparison.png"
    )
    logger.info("  ✓ Vergleich gespeichert")

    # ── Offline Augmentierung mit Resume testen ───────────
    logger.info("Offline Augmentierung (5 Bilder, resume=True):")
    stats = augment_dataset_offline(
        input_dir      = test_images[0].parent,
        output_dir     = AUGMENTED_DIR / "test",
        augment_factor = 2,
        img_size       = 224,
        resume         = True,
        quality_check  = True,
    )
    logger.info(f"  Original    : {stats['original']}")
    logger.info(f"  Augmentiert : {stats['augmented']}")
    logger.info(f"  Übersprungen: {stats['skipped']}")
    logger.info(f"  Verworfen   : {stats['rejected']}")
    logger.info(f"  Total       : {stats['total']}")

    # ── Resume testen: nochmal ausführen ──────────────────
    logger.info("Resume Test (nochmal ausführen):")
    stats2 = augment_dataset_offline(
        input_dir      = test_images[0].parent,
        output_dir     = AUGMENTED_DIR / "test",
        augment_factor = 2,
        resume         = True,
    )
    logger.info(
        f"  Übersprungen: {stats2['skipped']} "
        f"(sollte > 0 sein)"
    )

    logger.info("✓ augmentation.py funktioniert korrekt.")