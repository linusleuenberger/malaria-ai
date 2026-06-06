# ============================================================
# preprocessing/prepare_dataset.py
#
# Haupt-Pipeline: raw → processed
# Führt alle Preprocessing-Schritte zusammen:
#   1. Datensatz validieren
#   2. Bilder filtern (filter.py)
#   3. Normalisieren (normalization.py)
#   4. In Train/Val/Test aufteilen
#   5. Parallel verarbeiten
#   6. Integrity-Check
#   7. Report speichern
# ============================================================

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from preprocessing.filter        import apply_filter_pipeline, check_image_quality
from preprocessing.normalization import compute_channel_statistics
from src.config import (
    CLASS_NAMES,
    METRICS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

logger = logging.getLogger(__name__)


# ── 1. Datensatz validieren ───────────────────────────────────
def validate_raw_dataset(
    raw_dir: Path = RAW_DIR,
) -> Dict[str, int]:
    """
    Prüft ob der rohe Datensatz korrekt strukturiert ist.

    Erwartet:
        data/raw/
        ├── healthy/    ← Ordner muss existieren
        └── infected/   ← Ordner muss existieren

    Checks:
        ✓ Ordner healthy/ und infected/ vorhanden?
        ✓ Mindestens 100 Bilder pro Klasse?
        ✓ Bildformate unterstützt?
        ✓ Keine korrupten Bilder?
        ✓ Klassenbalance sinnvoll?

    Args:
        raw_dir : Pfad zum raw Ordner

    Returns:
        Dict mit Anzahl gültiger Bilder pro Klasse

    Raises:
        FileNotFoundError : Falls Ordner fehlen
        RuntimeError      : Falls zu wenige Bilder
    """
    logger.info("=" * 60)
    logger.info("Datensatz Validierung")
    logger.info("=" * 60)

    valid_counts: Dict[str, int] = {}

    for class_name in CLASS_NAMES:
        class_dir = raw_dir / class_name

        # ── Ordner Check ───────────────────────────────────
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Ordner nicht gefunden: {class_dir}\n"
                f"→ Datensatz herunterladen und in "
                f"data/raw/{class_name}/ ablegen"
            )

        # ── Bilder finden ──────────────────────────────────
        image_paths: List[Path] = []
        for ext in ["*.png", "*.jpg", "*.jpeg",
                    "*.tif", "*.tiff"]:
            image_paths.extend(class_dir.glob(ext))

        if len(image_paths) < 100:
            raise RuntimeError(
                f"Zu wenige Bilder in {class_name}/: "
                f"{len(image_paths)} "
                f"(Minimum: 100)"
            )

        # ── Korrupte Bilder prüfen (Stichprobe) ───────────
        sample    = random.sample(
            image_paths,
            min(50, len(image_paths))
        )
        corrupted = 0
        for p in sample:
            img = cv2.imread(str(p))
            if img is None:
                corrupted += 1

        if corrupted > 0:
            logger.warning(
                f"  {class_name}: {corrupted} korrupte "
                f"Bilder in Stichprobe gefunden"
            )

        valid_counts[class_name] = len(image_paths)
        logger.info(
            f"  {class_name:<12}: "
            f"{len(image_paths):>6,} Bilder ✓"
        )

    # ── Klassenbalance prüfen ─────────────────────────────
    counts = list(valid_counts.values())
    ratio  = max(counts) / min(counts) if min(counts) > 0 else float("inf")

    if ratio > 2.0:
        logger.warning(
            f"Starkes Klassenungleichgewicht: "
            f"Ratio = {ratio:.2f} "
            f"(> 2.0 kann problematisch sein)\n"
            f"→ WeightedSampler in dataset.py aktiv"
        )
    else:
        logger.info(f"  Klassenbalance: Ratio = {ratio:.2f} [OK]")

    logger.info("Validierung abgeschlossen [OK]")
    return valid_counts


# ── 2. Atomares Schreiben ─────────────────────────────────────
def _safe_write(
    image:      np.ndarray,
    output_path: Path,
) -> bool:
    """
    Bild sicher schreiben – entweder ganz oder gar nicht.

    Problem ohne atomares Schreiben:
        Falls Prozess während Schreiben abstürzt
        → halb-geschriebene Datei auf Disk
        → beim nächsten Start: Datei vorhanden aber korrupt
        → Training mit defekten Bildern

    Lösung:
        1. In temporäre .tmp Datei schreiben
        2. Erst wenn vollständig → umbenennen
        Umbenennen ist atomare Operation → kann nicht halb passieren

    Args:
        image       : Bild als NumPy Array
        output_path : Zielpfad

    Returns:
        True wenn erfolgreich, False bei Fehler
    """
    tmp_path = output_path.with_suffix(".tmp")
    try:
        # cv2.imwrite erkennt nur bekannte Extensions (.png, .jpg etc.)
        # .tmp ist unbekannt → daher erst als Bytes kodieren,
        # dann in .tmp schreiben, dann atomar umbenennen
        ext     = output_path.suffix  # z.B. ".png"
        success, buf = cv2.imencode(ext, image)
        if not success:
            raise RuntimeError(f"cv2.imencode fehlgeschlagen für {ext}")
        tmp_path.write_bytes(buf.tobytes())
        tmp_path.rename(output_path)
        return True
    except Exception as e:
        logger.error(f"Schreibfehler: {output_path.name}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


# ── 3. Einzelnes Bild verarbeiten ────────────────────────────
def _process_single_image(
    args: Tuple[Path, Path, bool],
) -> str:
    """
    Ein einzelnes Bild verarbeiten – für Parallelverarbeitung.

    Muss als separate Funktion existieren damit
    ProcessPoolExecutor sie in anderen Prozessen aufrufen kann.

    Args:
        args : Tuple aus (input_path, output_path, apply_filters)

    Returns:
        Status: "copied", "skipped" oder "failed"
    """
    input_path, output_path, apply_filters = args

    # Resume: bereits vorhanden → überspringen
    if output_path.exists():
        return "skipped"

    # Bild laden
    image = cv2.imread(str(input_path))
    if image is None:
        return "failed"

    # Qualitäts-Check
    quality = check_image_quality(image)
    if not quality["passed"]:
        return "failed"

    # Filter anwenden
    if apply_filters:
        image = apply_filter_pipeline(
            image,
            use_blur       = True,
            use_contrast   = True,
            use_sharpness  = True,
            use_artifacts  = True,
            use_background = False,
            use_stain_norm = True,
            use_macenko    = True,
        )

    # Sicher speichern
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = _safe_write(image, output_path)

    return "copied" if success else "failed"


# ── 4. Parallel kopieren & verarbeiten ───────────────────────
def _copy_and_process_parallel(
    image_paths:   List[Path],
    output_dir:    Path,
    apply_filters: bool = True,
    n_workers:     int  = 4,
    split_name:    str  = "",
    class_name:    str  = "",
) -> Dict[str, int]:
    """
    Bilder parallel verarbeiten mit ProcessPoolExecutor.

    Warum parallel:
        Filter + Macenko Normalisierung ist rechenintensiv
        → Auf 4 Kernen ~4× schneller
        → 27'558 Bilder: 2h → 30min

    Args:
        image_paths   : Liste der Bildpfade
        output_dir    : Zielordner
        apply_filters : Filterung anwenden
        n_workers     : Anzahl parallele Prozesse
        split_name    : Name des Splits (für tqdm)
        class_name    : Klassenname (für tqdm)

    Returns:
        Dict mit copied, skipped, failed Anzahl
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"copied": 0, "skipped": 0, "failed": 0}

    # Argumente für jeden Prozess vorbereiten
    args_list = [
        (
            img_path,
            output_dir / img_path.name,
            apply_filters,
        )
        for img_path in image_paths
    ]

    desc = f"{split_name}/{class_name}"

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_single_image, args): args
            for args in args_list
        }

        with tqdm(
            total = len(futures),
            desc  = desc,
            unit  = "Bild",
        ) as pbar:
            for future in as_completed(futures):
                status = future.result()
                stats[status] += 1
                pbar.update(1)
                pbar.set_postfix({
                    "OK"  : stats["copied"],
                    "Skip": stats["skipped"],
                    "Fail": stats["failed"],
                })

    return stats


# ── 5. Dataset aufteilen ──────────────────────────────────────
def split_dataset(
    image_paths: List[Path],
    train_split: float = TRAIN_SPLIT,
    val_split:   float = VAL_SPLIT,
    seed:        int   = RANDOM_SEED,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Bildpfade in Train/Val/Test aufteilen.

    Methode: Zufällige Aufteilung mit fixem Seed
    → Reproduzierbare Splits bei jedem Aufruf

    Verhältnis (aus config.py):
        Train : 70%
        Val   : 15%
        Test  : 15%

    Args:
        image_paths : Liste aller Bildpfade einer Klasse
        train_split : Anteil Training (0.0–1.0)
        val_split   : Anteil Validierung (0.0–1.0)
        seed        : Zufalls-Seed für Reproduzierbarkeit

    Returns:
        (train_paths, val_paths, test_paths)
    """
    paths = image_paths.copy()
    random.seed(seed)
    random.shuffle(paths)

    n       = len(paths)
    n_train = int(n * train_split)
    n_val   = int(n * val_split)

    train_paths = paths[:n_train]
    val_paths   = paths[n_train : n_train + n_val]
    test_paths  = paths[n_train + n_val:]

    return train_paths, val_paths, test_paths


# ── 6. Integrity-Check ───────────────────────────────────────
def verify_processed_dataset(
    processed_dir:  Path = PROCESSED_DIR,
    min_per_split:  int  = 100,
) -> bool:
    """
    Prüft ob processed/ Ordner vollständig und korrekt ist.

    Warum wichtig:
        Falls Pipeline abbricht → processed/ halb leer
        → Training startet trotzdem → schlechte Ergebnisse
        → Integrity-Check findet das sofort

    Checks:
        ✓ Alle Splits vorhanden (train/val/test)?
        ✓ Alle Klassen in jedem Split?
        ✓ Mindestens min_per_split Bilder pro Split/Klasse?
        ✓ Keine .tmp Dateien (abgebrochene Schreibvorgänge)?
        ✓ Keine korrupten Bilder (Stichprobe)?

    Args:
        processed_dir  : Pfad zum processed Ordner
        min_per_split  : Minimale Bilder pro Split/Klasse

    Returns:
        True wenn alles korrekt, False falls Problem gefunden
    """
    logger.info("=" * 60)
    logger.info("Integrity-Check")
    logger.info("=" * 60)

    all_ok = True

    for split_name in ["train", "val", "test"]:
        split_dir = processed_dir / split_name

        # ── Split vorhanden? ───────────────────────────────
        if not split_dir.exists():
            logger.error(f"  [FAIL] Split fehlt: {split_name}/")
            all_ok = False
            continue

        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name

            # ── Klasse vorhanden? ──────────────────────────
            if not class_dir.exists():
                logger.error(
                    f"  ✗ Klasse fehlt: "
                    f"{split_name}/{class_name}/"
                )
                all_ok = False
                continue

            # ── Bilder zählen ──────────────────────────────
            image_paths: List[Path] = []
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                image_paths.extend(class_dir.glob(ext))

            if len(image_paths) < min_per_split:
                logger.error(
                    f"  ✗ Zu wenige Bilder: "
                    f"{split_name}/{class_name}: "
                    f"{len(image_paths)} "
                    f"(Minimum: {min_per_split})"
                )
                all_ok = False
                continue

            # ── Temporäre Dateien ──────────────────────────
            tmp_files = list(class_dir.glob("*.tmp"))
            if tmp_files:
                logger.warning(
                    f"  ⚠ {len(tmp_files)} .tmp Dateien in "
                    f"{split_name}/{class_name}/ "
                    f"→ abgebrochener Schreibvorgang"
                )
                for tmp in tmp_files:
                    tmp.unlink()
                logger.info("    .tmp Dateien gelöscht [OK]")

            # ── Korrupte Bilder (Stichprobe) ───────────────
            sample    = random.sample(
                image_paths,
                min(20, len(image_paths))
            )
            corrupted = sum(
                1 for p in sample
                if cv2.imread(str(p)) is None
            )

            if corrupted > 0:
                logger.error(
                    f"  ✗ {corrupted} korrupte Bilder in "
                    f"{split_name}/{class_name}/"
                )
                all_ok = False
                continue

            logger.info(
                f"  ✓ {split_name}/{class_name}: "
                f"{len(image_paths):,} Bilder"
            )

    if all_ok:
        logger.info("\n[OK] Integrity-Check bestanden!")
    else:
        logger.error(
            "\n✗ Integrity-Check fehlgeschlagen!\n"
            "→ prepare_dataset(force=True) ausführen"
        )

    logger.info("=" * 60)
    return all_ok


# ── 7. Preprocessing-Report speichern ────────────────────────
def _save_report(
    split_stats:   Dict[str, Dict[str, int]],
    channel_stats: Optional[Dict],
    apply_filters: bool,
    duration_s:    float,
    save_path:     Path,
) -> None:
    """
    Vollständigen Preprocessing-Report als JSON speichern.

    Enthält:
        → Datum & Uhrzeit
        → Dauer der Pipeline
        → Bilder pro Split & Klasse
        → Kanal-Statistiken (Mean & Std)
        → Welche Filter angewendet wurden
        → Config-Parameter (Splits, Seed)

    Wichtig für:
        → ETH-Präsentation: Dokumentation der Methoden
        → Reproduzierbarkeit: gleiche Einstellungen später

    Args:
        split_stats   : Bilder pro Split und Klasse
        channel_stats : Mean & Std des Datensatzes
        apply_filters : Wurden Filter angewendet?
        duration_s    : Dauer in Sekunden
        save_path     : Speicherpfad
    """
    hours   = int(duration_s // 3600)
    minutes = int((duration_s % 3600) // 60)
    seconds = int(duration_s % 60)
    duration_str = f"{hours}h {minutes}m {seconds}s"

    report = {
        "meta": {
            "datum"        : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dauer"        : duration_str,
            "dauer_sekunden": round(duration_s, 1),
        },
        "config": {
            "train_split"  : TRAIN_SPLIT,
            "val_split"    : VAL_SPLIT,
            "test_split"   : TEST_SPLIT,
            "random_seed"  : RANDOM_SEED,
            "apply_filters": apply_filters,
        },
        "filter_pipeline": {
            "gaussian_blur"   : apply_filters,
            "clahe_contrast"  : apply_filters,
            "sharpness"       : apply_filters,
            "artifact_removal": apply_filters,
            "macenko_norm"    : apply_filters,
        },
        "splits": split_stats,
        "totals": {
            split_name: sum(classes.values())
            for split_name, classes in split_stats.items()
        },
        "channel_statistics": channel_stats or {},
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    logger.info(f"Report gespeichert: {save_path}")


# ── 8. Hauptfunktion ──────────────────────────────────────────
def prepare_dataset(
    raw_dir:       Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    apply_filters: bool = True,
    n_workers:     int  = 4,
    force:         bool = False,
) -> Dict[str, Dict[str, int]]:
    """
    Vollständige Preprocessing-Pipeline ausführen.

    Ablauf:
        1. Datensatz validieren
        2. Bilder aufteilen (Train/Val/Test)
        3. Parallel verarbeiten & kopieren
        4. Integrity-Check
        5. Kanal-Statistiken berechnen
        6. Report speichern

    Args:
        raw_dir       : Ordner mit Originalbildern
        processed_dir : Zielordner für verarbeitete Bilder
        apply_filters : filter.py Pipeline anwenden
        n_workers     : Anzahl parallele Prozesse
        force         : Bereits vorhandene Bilder neu verarbeiten

    Returns:
        Dict mit Statistiken pro Split und Klasse:
        {
            "train": {"healthy": 9645, "infected": 9645},
            "val"  : {"healthy": 2067, "infected": 2067},
            "test" : {"healthy": 2067, "infected": 2067},
        }
    """
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("prepare_dataset.py – Pipeline gestartet")
    logger.info(f"  Filter:    {apply_filters}")
    logger.info(f"  Workers:   {n_workers}")
    logger.info(f"  Force:     {force}")
    logger.info("=" * 60)

    # ── Schritt 1: Validieren ─────────────────────────────
    validate_raw_dataset(raw_dir)

    # ── Schritt 2: Aufteilen & Parallel verarbeiten ───────
    split_stats: Dict[str, Dict[str, int]] = {
        "train": {},
        "val"  : {},
        "test" : {},
    }

    for class_name in CLASS_NAMES:
        class_dir = raw_dir / class_name

        # Alle Bilder finden
        image_paths: List[Path] = []
        for ext in ["*.png", "*.jpg", "*.jpeg",
                    "*.tif", "*.tiff"]:
            image_paths.extend(class_dir.glob(ext))

        # Aufteilen
        train_paths, val_paths, test_paths = split_dataset(
            image_paths
        )

        logger.info(
            f"\n{class_name}:\n"
            f"  Train: {len(train_paths):,}\n"
            f"  Val  : {len(val_paths):,}\n"
            f"  Test : {len(test_paths):,}"
        )

        splits = {
            "train": train_paths,
            "val"  : val_paths,
            "test" : test_paths,
        }

        for split_name, paths in splits.items():
            output_dir = processed_dir / split_name / class_name

            # Force → vorhandene löschen
            if force and output_dir.exists():
                shutil.rmtree(output_dir)
                logger.info(
                    f"  {split_name}/{class_name}: "
                    f"gelöscht (force=True)"
                )

            # Parallel verarbeiten
            copy_stats = _copy_and_process_parallel(
                image_paths  = paths,
                output_dir   = output_dir,
                apply_filters = apply_filters,
                n_workers    = n_workers,
                split_name   = split_name,
                class_name   = class_name,
            )

            split_stats[split_name][class_name] = \
                copy_stats["copied"]

            logger.info(
                f"  {split_name}/{class_name}: "
                f"{copy_stats['copied']:,} kopiert | "
                f"{copy_stats['skipped']:,} übersprungen | "
                f"{copy_stats['failed']:,} fehlgeschlagen"
            )

    # ── Schritt 3: Integrity-Check ────────────────────────
    logger.info("\nIntegrity-Check nach Pipeline...")
    ok = verify_processed_dataset(processed_dir)
    if not ok:
        logger.error(
            "Pipeline abgeschlossen aber Integrity-Check "
            "fehlgeschlagen!\n"
            "→ prepare_dataset(force=True) ausführen"
        )

    # ── Schritt 4: Kanal-Statistiken berechnen ────────────
    logger.info("\nBerechne Kanal-Statistiken...")
    all_train_images: List[Path] = []
    for class_name in CLASS_NAMES:
        train_dir = processed_dir / "train" / class_name
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            all_train_images.extend(train_dir.glob(ext))

    channel_stats = None
    if all_train_images:
        channel_stats = compute_channel_statistics(
            image_paths = all_train_images,
            sample_size = 1000,
            cache_path  = METRICS_DIR / "dataset_stats.json",
        )
        logger.info(
            f"  mean: {channel_stats['mean']}\n"
            f"  std:  {channel_stats['std']}\n"
            f"  → In config.py unter MEAN & STD eintragen!"
        )

    # ── Schritt 5: Report speichern ───────────────────────
    duration = time.time() - start_time
    _save_report(
        split_stats   = split_stats,
        channel_stats = channel_stats,
        apply_filters = apply_filters,
        duration_s    = duration,
        save_path     = METRICS_DIR / "preprocessing_report.json",
    )

    # ── Schritt 6: Zusammenfassung ────────────────────────
    hours   = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline abgeschlossen [OK]")
    logger.info(f"Dauer: {hours}h {minutes}m {seconds}s")
    logger.info("=" * 60)
    logger.info(f"{'Split':<10} {'Klasse':<12} {'Bilder':>8}")
    logger.info("-" * 32)

    total = 0
    for split_name, classes in split_stats.items():
        for class_name, count in classes.items():
            logger.info(
                f"{split_name:<10} "
                f"{class_name:<12} "
                f"{count:>8,}"
            )
            total += count

    logger.info("-" * 32)
    logger.info(f"{'Total':<22} {total:>8,}")
    logger.info("=" * 60)

    return split_stats


# ── Quick-Test: python -m preprocessing.prepare_dataset ──────
if __name__ == "__main__":
    import sys

    # ── Validierung testen ─────────────────────────────────
    logger.info("Validierung:")
    try:
        counts = validate_raw_dataset()
        for name, count in counts.items():
            logger.info(f"  {name}: {count:,} Bilder")
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        logger.info(
            "→ Datensatz herunterladen: "
            "https://lhncbc.nlm.nih.gov/LHC-research/"
            "LHC-projects/image-processing/"
            "malaria-datasheet.html"
        )
        sys.exit(1)

    # ── Pipeline ausführen ────────────────────────────────
    logger.info("Pipeline starten:")
    split_stats = prepare_dataset(
        apply_filters = True,
        n_workers     = 4,
        force         = False,
    )

    # ── Integrity-Check ───────────────────────────────────
    logger.info("Integrity-Check:")
    ok = verify_processed_dataset()
    logger.info(f"  {'[OK] Bestanden' if ok else '[FAIL] Fehlgeschlagen'}")

    # ── Erge