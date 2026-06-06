# ============================================================
# main.py – Einstiegspunkt für das Malaria-KI Projekt
# ============================================================
#
# Modi:
#     train      → Modell trainieren
#     evaluate   → Modell auf Testset evaluieren
#     predict    → Einzelbild oder Ordner vorhersagen
#     preprocess → Rohdaten aufbereiten (raw → processed)
#     augment    → Offline-Augmentierung ausführen
#
# Verwendung:
#     python main.py --mode train
#     python main.py --mode train --resume
#     python main.py --mode train --eval-after
#     python main.py --mode evaluate
#     python main.py --mode predict --image data/raw/infected/cell_01.png
#     python main.py --mode predict --folder data/raw/infected/
#     python main.py --mode preprocess
#     python main.py --mode augment --level heavy --n 10
# ============================================================

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── Encoding-sicherer StreamHandler ───────────────────────────
class _SafeStreamHandler(logging.StreamHandler):
    """
    StreamHandler der UnicodeEncodeError auf Windows (cp1252) verhindert.
    Nicht darstellbare Zeichen werden durch '?' ersetzt statt zu crashen.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            try:
                enc  = getattr(self.stream, "encoding", None) or "ascii"
                safe = self.format(record).encode(enc, errors="replace").decode(enc)
                self.stream.write(safe + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
        except Exception:
            self.handleError(record)


# ── Logging Setup ─────────────────────────────────────────────
def setup_logging(log_dir: Path, debug: bool = False) -> None:
    """
    Logging in Konsole und Datei gleichzeitig.

    Level:
        DEBUG   - alle Details (nur mit --debug)
        INFO    - normaler Betrieb
        WARNING - Probleme die nicht zum Abbruch fuehren
        ERROR   - Fehler die zum Abbruch fuehren

    Args:
        log_dir (Path): Pfad zum Ordner fuer Log-Dateien.
        debug (bool): Ob Debug-Modus aktiviert werden soll.

    Returns:
        None
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.log"
    level    = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level    = level,
        format   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt  = "%Y-%m-%d %H:%M:%S",
        handlers = [
            _SafeStreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    # Externe Bibliotheken weniger verbose
    logging.getLogger("PIL")       .setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("albumentations").setLevel(logging.WARNING)

    logger.info(f"Logging aktiv -> {log_file}")


# ── Reproduzierbarkeit ────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """
    Alle Zufallsgeneratoren auf denselben Seed setzen.
    Gleicher Seed = gleiche Ergebnisse bei gleichem Code.
    
    Args:
        seed (int): Der gewünschte Random-Seed.
        
    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    logger.info(f"Seed gesetzt: {seed}")


# ── Device ────────────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Bestes verfügbares Gerät automatisch wählen.
    Priorität: CUDA → MPS (Apple) → CPU
    
    Returns:
        torch.device: Das ausgewählte Gerät.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name   = torch.cuda.get_device_name(0)
        mem    = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Gerät: {name} ({mem:.1f} GB VRAM)")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Gerät: Apple MPS (M-Chip)")

    else:
        device = torch.device("cpu")
        logger.info("Gerät: CPU")

    return device


# ── Hilfsfunktionen ───────────────────────────────────────────
def _check_processed_data(processed_dir: str) -> None:
    """
    Prüft ob aufbereitete Daten vorhanden sind.
    Gibt verständliche Fehlermeldung statt kryptischem Fehler.
    
    Args:
        processed_dir (str): Pfad zum Ordner mit aufbereiteten Daten.
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: Wenn der Ordner nicht existiert.
    """
    path = Path(processed_dir)

    if not path.exists():
        logger.error(
            f"Keine verarbeiteten Daten gefunden: {path}\n"
            f"  -> Zuerst ausfuehren: "
            f"python main.py --mode preprocess"
        )
        raise FileNotFoundError(f"Datenordner fehlt: {path}")

    for split in ["train", "val", "test"]:
        split_path = path / split
        if not split_path.exists():
            logger.error(
                f"Split-Ordner fehlt: {split_path}\n"
                f"  -> Zuerst ausfuehren: "
                f"python main.py --mode preprocess"
            )
            raise FileNotFoundError(f"Split-Ordner fehlt: {split_path}")


def _log_runtime(start: float, modus: str) -> None:
    """
    Laufzeit formatiert ausgeben.
    
    Args:
        start (float): Startzeitpunkt.
        modus (str): Name des ausgeführten Modus.
        
    Returns:
        None
    """
    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    logger.info(
        f"{modus} abgeschlossen in {minutes}m {seconds}s"
    )


# ── Modus: Preprocessing ──────────────────────────────────────
def mode_preprocess(args: argparse.Namespace) -> None:
    """
    raw → processed Pipeline ausführen.
    
    Args:
        args (argparse.Namespace): Die Kommandozeilenargumente.
        
    Returns:
        None
    """
    from preprocessing.prepare_dataset import (
        prepare_dataset,
        verify_processed_dataset,
    )
    logger.info("=" * 60)
    logger.info("MODUS: Preprocessing")
    logger.info("=" * 60)

    start = time.time()

    split_stats = prepare_dataset(
        raw_dir       = Path(args.raw_dir),
        processed_dir = Path(args.processed_dir),
        apply_filters = True,
        n_workers     = 4,
        force         = False,
    )

    # Integrity-Check nach Pipeline
    logger.info("Integrity-Check...")
    ok = verify_processed_dataset(Path(args.processed_dir))
    if not ok:
        logger.error(
            "Integrity-Check fehlgeschlagen!\n"
            "-> python main.py --mode preprocess erneut ausfuehren"
        )
        sys.exit(1)

    _log_runtime(start, "Preprocessing")


# ── Modus: Augmentierung ──────────────────────────────────────
def mode_augment(args: argparse.Namespace) -> None:
    """
    Offline-Augmentierung ausführen.
    
    Args:
        args (argparse.Namespace): Die Kommandozeilenargumente.
        
    Returns:
        None
    """
    from preprocessing.augmentation import (
        augment_dataset_offline,
    )
    logger.info("=" * 60)
    logger.info("MODUS: Augmentierung")
    logger.info("=" * 60)
    logger.info(f"  Level      : {args.level}")
    logger.info(f"  Pro Bild   : {args.n}")
    logger.info(f"  Nur Klasse : {args.only_class or 'alle'}")

    # Quelldaten prüfen
    _check_processed_data(args.processed_dir)

    start    = time.time()
    extended = args.level == "heavy"

    if args.only_class:
        # Nur eine Klasse augmentieren
        input_dir = (
            Path(args.processed_dir) / "train" / args.only_class
        )
        output_dir = (
            Path(args.augmented_dir) / args.only_class
        )
        stats = augment_dataset_offline(
            input_dir      = input_dir,
            output_dir     = output_dir,
            augment_factor = args.n,
            extended       = extended,
            resume         = True,
            quality_check  = True,
        )
        logger.info(
            f"  {args.only_class}: "
            f"{stats['total']:,} Bilder total"
        )

    else:
        # Alle Klassen balancieren
        from src.config import CLASS_NAMES
        for class_name in CLASS_NAMES:
            input_dir  = (
                Path(args.processed_dir) / "train" / class_name
            )
            output_dir = (
                Path(args.augmented_dir) / class_name
            )
            stats = augment_dataset_offline(
                input_dir      = input_dir,
                output_dir     = output_dir,
                augment_factor = args.n,
                extended       = extended,
                resume         = True,
                quality_check  = True,
            )
            logger.info(
                f"  {class_name}: "
                f"{stats['total']:,} Bilder total"
            )

    _log_runtime(start, "Augmentierung")


# ── Modus: Training ───────────────────────────────────────────
def mode_train(
    args:   argparse.Namespace,
    device: torch.device,
) -> None:
    """
    Modell trainieren.
    
    Args:
        args (argparse.Namespace): Die Kommandozeilenargumente.
        device (torch.device): Das zu verwendende Gerät (CPU/GPU).
        
    Returns:
        None
    """
    from src.config import (
        BATCH_SIZE,
        BEST_MODEL_PATH,
        EPOCHS,
        IMAGE_SIZE,
        LEARNING_RATE,
        PROCESSED_DIR,
        USE_WANDB,
        WANDB_PROJECT,
    )
    from src.dataset import get_dataloaders
    from src.model   import build_model, load_model
    from src.train   import train
    from src.utils   import (
        print_dataset_info,
        print_system_info,
        sanity_check,
    )
    logger.info("=" * 60)
    logger.info("MODUS: Training")
    logger.info("=" * 60)
    logger.info(f"  Epochen     : {EPOCHS}")
    logger.info(f"  Batch-Grösse: {BATCH_SIZE}")
    logger.info(f"  Lernrate    : {LEARNING_RATE}")
    logger.info(f"  Bildgrösse  : {IMAGE_SIZE[0]}px")
    logger.info(f"  Gerät       : {device}")

    # System Info anzeigen
    print_system_info()

    # Daten prüfen
    _check_processed_data(str(PROCESSED_DIR))

    # DataLoader erstellen
    # persistent_workers=True: Worker-Prozesse leben über Epochen hinaus → kein Overhead
    # prefetch_factor=2: CPU lädt nächsten Batch vor während GPU arbeitet
    loaders = get_dataloaders(
        data_dir           = PROCESSED_DIR,
        img_size           = IMAGE_SIZE[0],
        batch_size         = BATCH_SIZE,
        num_workers        = 8,
        pin_memory         = (device.type == "cuda"),
        persistent_workers = True,
        prefetch_factor    = 2,
    )

    # Datensatz-Übersicht
    print_dataset_info(
        loaders["train"],
        loaders["val"],
        loaders["test"],
    )

    # Modell erstellen
    model = build_model().to(device)
    logger.info(
        f"Modell: {model.__class__.__name__} | "
        f"Architektur: resnet50"
    )

    # Checkpoint laden falls --resume
    if args.resume:
        checkpoint_path = Path("models/checkpoints/best_model.pth")
        if checkpoint_path.exists():
            model = load_model(checkpoint_path)
            logger.info(f"Checkpoint geladen: {checkpoint_path}")
        else:
            logger.warning(
                "Kein Checkpoint gefunden – starte von vorne."
            )

    # Sanity Check vor Training
    logger.info("Sanity Check...")
    passed = sanity_check(model, loaders["train"])
    if not passed:
        logger.error(
            "Sanity Check fehlgeschlagen!\n"
            "-> Modell oder Daten pruefen"
        )
        sys.exit(1)
    logger.info("Sanity Check bestanden [OK]")

    # WandB initialisieren
    if USE_WANDB:
        try:
            import wandb
            wandb.init(
                project = WANDB_PROJECT,
                config  = {
                    "epochs"     : EPOCHS,
                    "batch_size" : BATCH_SIZE,
                    "lr"         : LEARNING_RATE,
                    "img_size"   : IMAGE_SIZE[0],
                    "device"     : str(device),
                },
            )
            logger.info("WandB initialisiert.")
        except Exception as e:
            logger.warning(
                f"WandB konnte nicht gestartet werden: {e}"
            )

    # Training starten
    start   = time.time()
    history = train(
        model        = model,
        train_loader = loaders["train"],
        val_loader   = loaders["val"],
    )
    _log_runtime(start, "Training")

    # WandB beenden
    if USE_WANDB:
        try:
            import wandb
            wandb.finish()
            logger.info("WandB beendet.")
        except Exception:
            pass

    # Direkt evaluieren falls --eval-after
    if args.eval_after:
        logger.info("Starte Evaluation nach Training...")
        args.model = str(BEST_MODEL_PATH)
        mode_evaluate(args, device, history=history)


# ── Modus: Evaluation ─────────────────────────────────────────
def mode_evaluate(
    args:    argparse.Namespace,
    device:  torch.device,
    history: Optional[dict] = None,
) -> None:
    """
    Modell auf Testset evaluieren.
    
    Args:
        args (argparse.Namespace): Die Kommandozeilenargumente.
        device (torch.device): Das zu verwendende Gerät (CPU/GPU).
        history (Optional[dict]): Trainingsverlauf (falls vorhanden).
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: Wenn das Modell nicht gefunden wird.
    """
    from src.config  import (
        BATCH_SIZE,
        IMAGE_SIZE,
        PROCESSED_DIR,
    )
    from src.dataset  import get_dataloaders
    from src.evaluate import evaluate
    from src.model    import load_model
    logger.info("=" * 60)
    logger.info("MODUS: Evaluation")
    logger.info("=" * 60)

    # Daten prüfen
    _check_processed_data(str(PROCESSED_DIR))

    # Modell prüfen
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Modell nicht gefunden: {model_path}")
        sys.exit(1)

    # DataLoader erstellen
    loaders = get_dataloaders(
        data_dir    = PROCESSED_DIR,
        img_size    = IMAGE_SIZE[0],
        batch_size  = BATCH_SIZE,
        num_workers = 4,
        pin_memory  = (device.type == "cuda"),
    )

    # Modell laden & evaluieren
    start   = time.time()
    model   = load_model(model_path)
    metrics = evaluate(
        model   = model,
        loader  = loaders["test"],
        history = history,
    )
    _log_runtime(start, "Evaluation")

    # Metriken ausgeben
    logger.info("\nMetriken:")
    for key, value in metrics.items():
        logger.info(f"  {key:<30}: {value}")


# ── Modus: Vorhersage ─────────────────────────────────────────
def mode_predict(
    args:   argparse.Namespace,
    device: torch.device,
) -> None:
    """
    Einzelbild oder Ordner vorhersagen.
    
    Args:
        args (argparse.Namespace): Die Kommandozeilenargumente.
        device (torch.device): Das zu verwendende Gerät (CPU/GPU).
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: Wenn Modell oder Bild/Ordner nicht gefunden werden.
    """
    from src.config import IMAGE_SIZE
    from src.model  import load_model
    logger.info("=" * 60)
    logger.info("MODUS: Vorhersage")
    logger.info("=" * 60)

    # Modell prüfen
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Modell nicht gefunden: {model_path}")
        sys.exit(1)

    model = load_model(model_path)

    # ── Einzelbild ─────────────────────────────────────────
    if args.image:
        from src.predict import preprocess_image, predict

        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Bild nicht gefunden: {image_path}")
            sys.exit(1)

        tensor, original = preprocess_image(image_path)
        result = predict(model, tensor, device)
        pred = result["predicted_idx"]
        confidence = result["confidence"]

        logger.info(f"Bild     : {image_path.name}")
        logger.info(
            f"Diagnose : "
            f"{'[INFECTED]' if pred == 1 else '[HEALTHY]'}"
        )
        logger.info(f"Konfidenz: {confidence:.2%}")

        # Grad-CAM anzeigen
        if args.gradcam:
            from src.predict import GradCAM, get_last_conv_layer
            import matplotlib.pyplot as plt
            cam = GradCAM(model, get_last_conv_layer(model))
            heatmap = cam(tensor.to(device))
            overlay = GradCAM.overlay(original, heatmap)
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original); axes[0].set_title("Original"); axes[0].axis("off")
            axes[1].imshow(overlay);  axes[1].set_title("Grad-CAM"); axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(Path("results/plots/gradcam_predict.png"), dpi=150)
            plt.close()

        # Ergebnis speichern
        if args.save:
            import json
            from src.config import PREDICTIONS_DIR
            result_data = {
                "image"     : str(image_path),
                "predicted" : result["predicted_class"],
                "confidence": round(confidence, 4),
            }
            save_path = PREDICTIONS_DIR / "prediction.json"
            with open(save_path, "w") as f:
                json.dump(result_data, f, indent=4)
            logger.info(f"Ergebnis gespeichert: {save_path}")

    # ── Ordner ─────────────────────────────────────────────
    elif args.folder:
        folder_path = Path(args.folder)
        if not folder_path.exists():
            logger.error(f"Ordner nicht gefunden: {folder_path}")
            sys.exit(1)

        from src.predict import preprocess_image, predict
        from src.config  import CLASS_NAMES, IDX_TO_CLASS, PREDICTIONS_DIR

        # Alle Bilder im Ordner finden
        image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif"]:
            image_paths.extend(folder_path.glob(ext))

        if not image_paths:
            logger.error(f"Keine Bilder in: {folder_path}")
            sys.exit(1)

        logger.info(f"{len(image_paths)} Bilder gefunden")

        results    = []
        infected   = 0
        healthy    = 0

        for img_path in image_paths:
            tensor, _ = preprocess_image(img_path)
            result_item = predict(model, tensor, device)
            pred       = result_item["predicted_idx"]
            confidence = result_item["confidence"]
            label      = result_item["predicted_class"]

            results.append({
                "image"     : img_path.name,
                "predicted" : label,
                "confidence": round(confidence, 4),
            })

            if pred == 1:
                infected += 1
            else:
                healthy  += 1

            logger.info(
                f"  {img_path.name:<30} -> "
                f"{'[INFECTED]' if pred == 1 else '[HEALTHY]'} "
                f"({confidence:.2%})"
            )

        # Zusammenfassung
        total = len(image_paths)
        logger.info("-" * 50)
        logger.info(f"  Gesamt  : {total}")
        logger.info(f"  Healthy : {healthy} ({healthy/total:.1%})")
        logger.info(f"  Infected: {infected} ({infected/total:.1%})")

        # Ergebnisse speichern
        if args.save:
            import json
            save_path = PREDICTIONS_DIR / "folder_predictions.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump({
                    "summary": {
                        "total"   : total,
                        "healthy" : healthy,
                        "infected": infected,
                    },
                    "predictions": results,
                }, f, indent=4)
            logger.info(f"Ergebnisse gespeichert: {save_path}")

    else:
        logger.error(
            "Kein Ziel angegeben.\n"
            "-> --image <pfad> oder --folder <pfad>"
        )
        sys.exit(1)


# ── Argument-Parser ───────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """
    Alle Kommandozeilen-Argumente definieren.
    
    Returns:
        argparse.Namespace: Die geparsten Argumente.
    """
    parser = argparse.ArgumentParser(
        description     = "Malaria-KI – Erkennung infizierter Blutzellen",
        formatter_class = argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action  = "version",
        version = "Malaria-KI v1.0.0"
    )

    # ── Pflichtargument ────────────────────────────────────
    parser.add_argument(
        "--mode",
        type    = str,
        required= True,
        choices = ["train", "evaluate", "predict",
                   "preprocess", "augment"],
        help    = "Betriebsmodus",
    )

    # ── Allgemein ──────────────────────────────────────────
    parser.add_argument(
        "--seed",
        type    = int,
        default = 42,
        help    = "Zufalls-Seed (Standard: 42)",
    )
    parser.add_argument(
        "--debug",
        action  = "store_true",
        help    = "Debug-Logging aktivieren",
    )
    parser.add_argument(
        "--log-dir",
        type    = str,
        default = "results/logs",
        help    = "Ordner für Log-Dateien",
    )

    # Standardpfad aus config.py
    try:
        from src.config import BEST_MODEL_PATH
        default_model = str(BEST_MODEL_PATH)
    except ImportError:
        default_model = "models/checkpoints/best_model.pth"

    parser.add_argument(
        "--model",
        type    = str,
        default = default_model,
        help    = "Pfad zum Modell",
    )

    # ── Preprocessing ──────────────────────────────────────
    parser.add_argument(
        "--raw-dir",
        type    = str,
        default = "data/raw",
        help    = "Ordner mit Originalbildern",
    )
    parser.add_argument(
        "--processed-dir",
        type    = str,
        default = "data/processed",
        help    = "Ordner für verarbeitete Bilder",
    )
    parser.add_argument(
        "--augmented-dir",
        type    = str,
        default = "data/augmented",
        help    = "Ordner für augmentierte Bilder",
    )

    # ── Augmentierung ──────────────────────────────────────
    parser.add_argument(
        "--level",
        type    = str,
        default = "medium",
        choices = ["light", "medium", "heavy"],
        help    = "Stärke der Augmentierung",
    )
    parser.add_argument(
        "--n",
        type    = int,
        default = 5,
        help    = "Augmentierte Kopien pro Originalbild",
    )
    parser.add_argument(
        "--only-class",
        type    = str,
        default = None,
        choices = ["infected", "healthy"],
        help    = "Nur eine Klasse augmentieren",
    )

    # ── Training ───────────────────────────────────────────
    parser.add_argument(
        "--resume",
        action  = "store_true",
        help    = "Training von letztem Checkpoint fortsetzen",
    )
    parser.add_argument(
        "--eval-after",
        action  = "store_true",
        help    = "Nach Training direkt evaluieren",
    )

    # ── Vorhersage ─────────────────────────────────────────
    parser.add_argument(
        "--image",
        type    = str,
        default = None,
        help    = "Pfad zu einem einzelnen Bild",
    )
    parser.add_argument(
        "--folder",
        type    = str,
        default = None,
        help    = "Pfad zu einem Ordner mit Bildern",
    )
    parser.add_argument(
        "--gradcam",
        action  = "store_true",
        help    = "Grad-CAM Heatmap anzeigen",
    )
    parser.add_argument(
        "--save",
        action  = "store_true",
        help    = "Ergebnisse als JSON speichern",
    )

    return parser.parse_args()


# ── Einstiegspunkt ────────────────────────────────────────────
def main() -> None:
    """
    Hauptfunktion des Skripts.
    Liest Argumente, initialisiert Umgebung und ruft Modus auf.

    Returns:
        None
    """
    args   = parse_args()
    setup_logging(Path(args.log_dir), debug=args.debug)
    set_seed(args.seed)
    logger.info("=" * 42)
    logger.info("         Malaria-KI gestartet")
    logger.info("=" * 42)

    device      = get_device()
    start_total = time.time()

    # Modus ausführen
    if   args.mode == "preprocess": mode_preprocess(args)
    elif args.mode == "augment":    mode_augment(args, )
    elif args.mode == "train":      mode_train(args, device)
    elif args.mode == "evaluate":   mode_evaluate(args, device)
    elif args.mode == "predict":    mode_predict(args, device)

    _log_runtime(start_total, "Gesamtlaufzeit")
    logger.info("=" * 42)
    logger.info("         Malaria-KI beendet")
    logger.info("=" * 42)


if __name__ == "__main__":
    main()