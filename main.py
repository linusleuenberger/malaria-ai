# ============================================================
# main.py - Einstiegspunkt fuer das Malaria-KI-Projekt
# ============================================================
#
# Modi:
#   preprocess : Rohdaten aufbereiten (raw -> processed, train/val/test)
#   train      : Modell trainieren (fortsetzbar mit --resume)
#   evaluate   : schnelle Evaluation auf dem Test-Split
#   analyze    : ausfuehrliche Zuverlaessigkeits-Analyse (siehe analyze.py)
#   predict    : Einzelbild oder Ordner vorhersagen
#
# Beispiele:
#   python main.py --mode train
#   python main.py --mode train --resume
#   python main.py --mode train --deterministic     # exakt reproduzierbar
#   python main.py --mode evaluate
#   python main.py --mode analyze --tta
#   python main.py --mode predict --folder data/processed/test/infected
# ============================================================

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("main")


# ── Logging (Windows-sicher) ──────────────────────────────────
class _SafeStreamHandler(logging.StreamHandler):
    """Ersetzt nicht darstellbare Zeichen statt zu crashen (Windows cp1252)."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.stream.write(self.format(record) + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            enc = getattr(self.stream, "encoding", None) or "ascii"
            self.stream.write(self.format(record).encode(enc, "replace").decode(enc)
                              + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(log_dir: Path, debug: bool = False) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[_SafeStreamHandler(sys.stdout),
                  logging.FileHandler(log_dir / "run.log", encoding="utf-8")],
    )
    for noisy in ("PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Modus: Preprocessing ──────────────────────────────────────
def mode_preprocess(args) -> None:
    from preprocessing.prepare_dataset import prepare_dataset, verify_processed_dataset
    logger.info("MODUS: Preprocessing")
    prepare_dataset(raw_dir=Path(args.raw_dir),
                    processed_dir=Path(args.processed_dir),
                    apply_filters=True, n_workers=4, force=False)
    if not verify_processed_dataset(Path(args.processed_dir)):
        logger.error("Integrity-Check fehlgeschlagen - bitte erneut ausfuehren.")
        sys.exit(1)


# ── Modus: Training ───────────────────────────────────────────
def mode_train(args) -> None:
    from src.config import (BATCH_SIZE, EPOCHS, IMAGE_SIZE, PROCESSED_DIR,
                            UNFREEZE_AT_EPOCH, UNFREEZE_N_LAYERS, USE_COMPILE, DEVICE)
    from src.dataset import get_dataloaders
    from src.model import build_model
    from src.train import train
    from src.utils import print_system_info, sanity_check

    logger.info("MODUS: Training | Epochen=%d Batch=%d", EPOCHS, BATCH_SIZE)
    print_system_info()
    _require_processed(PROCESSED_DIR)

    loaders = get_dataloaders(PROCESSED_DIR, img_size=IMAGE_SIZE[0], batch_size=BATCH_SIZE,
                              splits=("train", "val"))
    model = build_model()

    if not sanity_check(model, loaders["train"]):
        logger.error("Sanity Check fehlgeschlagen - Abbruch."); sys.exit(1)

    t0 = time.time()
    history = train(model, loaders["train"], loaders["val"],
                    unfreeze_at_epoch=UNFREEZE_AT_EPOCH,
                    unfreeze_n_layers=UNFREEZE_N_LAYERS, resume=args.resume)
    logger.info("Training fertig in %.0fs", time.time() - t0)

    if args.eval_after:
        _evaluate(history=history)


# ── Modus: Evaluation ─────────────────────────────────────────
def mode_evaluate(args) -> None:
    logger.info("MODUS: Evaluation")
    _evaluate(model_path=args.model)


def _evaluate(model_path: str | None = None, history=None) -> None:
    from src.config import (BATCH_SIZE, FINAL_MODEL_PATH, BEST_CKPT_PATH,
                            IMAGE_SIZE, PROCESSED_DIR)
    from src.dataset import get_dataloaders
    from src.evaluate import evaluate
    from src.model import load_model

    _require_processed(PROCESSED_DIR)
    if model_path:
        path = Path(model_path)
    else:
        path = FINAL_MODEL_PATH if FINAL_MODEL_PATH.exists() else BEST_CKPT_PATH
    if not path.exists():
        logger.error("Modell nicht gefunden: %s", path); sys.exit(1)

    loaders = get_dataloaders(PROCESSED_DIR, img_size=IMAGE_SIZE[0],
                              batch_size=BATCH_SIZE, splits=("test",),
                              use_weighted_sampler=False)
    evaluate(load_model(path), loaders["test"], history=history)


# ── Modus: Analyse ────────────────────────────────────────────
def mode_analyze(args) -> None:
    import analyze
    from src.config import FINAL_MODEL_PATH, BEST_CKPT_PATH
    path = Path(args.model) if args.model else (
        FINAL_MODEL_PATH if FINAL_MODEL_PATH.exists() else BEST_CKPT_PATH)
    analyze.run(path, tta=args.tta)


# ── Modus: Vorhersage ─────────────────────────────────────────
def mode_predict(args) -> None:
    import json
    from src.config import FINAL_MODEL_PATH, BEST_CKPT_PATH, PREDICTIONS_DIR
    from src.model import load_model
    from src.predict import predict, predict_folder, preprocess_image, load_threshold

    logger.info("MODUS: Vorhersage")
    path = Path(args.model) if args.model else (
        FINAL_MODEL_PATH if FINAL_MODEL_PATH.exists() else BEST_CKPT_PATH)
    if not path.exists():
        logger.error("Modell nicht gefunden: %s", path); sys.exit(1)
    model = load_model(path)
    threshold = load_threshold()

    if args.image:
        tensor, _ = preprocess_image(args.image)
        r = predict(model, tensor, threshold)
        logger.info("%s -> %s (%.2f%%)", Path(args.image).name,
                    r["predicted_class"].upper(), r["confidence"] * 100)
        if args.save:
            (PREDICTIONS_DIR / "prediction.json").write_text(json.dumps(r, indent=2))
    elif args.folder:
        results = predict_folder(model, args.folder, threshold=threshold)
        inf = sum(1 for r in results if r["predicted"] == "infected")
        logger.info("%d Bilder | healthy=%d infected=%d",
                    len(results), len(results) - inf, inf)
        if args.save:
            (PREDICTIONS_DIR / "folder_predictions.json").write_text(
                json.dumps({"summary": {"total": len(results),
                                        "healthy": len(results) - inf, "infected": inf},
                            "predictions": results}, indent=2))
    else:
        logger.error("Bitte --image <pfad> oder --folder <pfad> angeben."); sys.exit(1)


# ── Hilfen ────────────────────────────────────────────────────
def _require_processed(processed_dir) -> None:
    path = Path(processed_dir)
    for split in ("train", "val", "test"):
        if not (path / split).exists():
            logger.error("Split fehlt: %s\n-> python main.py --mode preprocess", path / split)
            sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Malaria-KI - Erkennung infizierter Blutzellen")
    p.add_argument("--mode", required=True,
                   choices=["preprocess", "train", "evaluate", "analyze", "predict"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--deterministic", action="store_true",
                   help="Exakt reproduzierbar (langsamer)")
    p.add_argument("--log-dir", type=str, default="results/logs")
    p.add_argument("--model", type=str, default=None, help="Pfad zum Modell")
    # Preprocessing
    p.add_argument("--raw-dir", type=str, default="data/raw")
    p.add_argument("--processed-dir", type=str, default="data/processed")
    # Training
    p.add_argument("--resume", action="store_true", help="Von last.pth fortsetzen")
    p.add_argument("--eval-after", action="store_true", help="Nach dem Training evaluieren")
    # Analyse / Vorhersage
    p.add_argument("--tta", action="store_true", help="Test-Time-Augmentation (analyze)")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--folder", type=str, default=None)
    p.add_argument("--save", action="store_true", help="Ergebnis als JSON speichern")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(Path(args.log_dir), debug=args.debug)

    from src.utils import set_seed, setup_perf
    set_seed(args.seed)
    setup_perf(deterministic=args.deterministic)

    start = time.time()
    {"preprocess": mode_preprocess, "train": mode_train, "evaluate": mode_evaluate,
     "analyze": mode_analyze, "predict": mode_predict}[args.mode](args)
    logger.info("Gesamtlaufzeit: %.0fs", time.time() - start)


if __name__ == "__main__":
    main()
