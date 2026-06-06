# ============================================================
# preprocessing/__init__.py
# Paket-Markierung, gebündelte Imports & Fehlerbehandlung
# ============================================================

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Paket-Metadaten ───────────────────────────────────────────
__version__ = "1.0.0"
__author__  = "Malaria-AI Team"
__date__    = "2026"

# ── Verfügbarkeits-Flags ──────────────────────────────────────
# Werden auf True gesetzt falls Import erfolgreich
_FILTER_AVAILABLE        = False
_NORMALIZATION_AVAILABLE = False
_AUGMENTATION_AVAILABLE  = False
_PREPARE_AVAILABLE       = False


# ── filter.py ────────────────────────────────────────────────
try:
    from preprocessing.filter import (
        apply_filter_pipeline,
        apply_gaussian_blur,
        apply_median_blur,
        check_image_quality,
        enhance_contrast_clahe,
        enhance_sharpness,
        normalize_staining_macenko,
        plot_filter_comparison,
        process_batch,
        process_image_file,
        remove_artifacts,
        remove_background,
    )
    _FILTER_AVAILABLE = True

except ImportError as e:
    logger.warning(
        f"filter.py konnte nicht geladen werden: {e}\n"
        f"→ pip install opencv-python"
    )


# ── normalization.py ──────────────────────────────────────────
try:
    from preprocessing.normalization import (
        apply_normalization_pipeline,
        compute_channel_statistics,
        denormalize_imagenet,
        denormalize_zscore,
        normalize_imagenet,
        normalize_minmax,
        normalize_percentile,
        normalize_zscore,
        plot_normalization_comparison,
    )
    _NORMALIZATION_AVAILABLE = True

except ImportError as e:
    logger.warning(
        f"normalization.py konnte nicht geladen werden: {e}\n"
        f"→ pip install opencv-python numpy matplotlib"
    )


# ── augmentation.py ───────────────────────────────────────────
try:
    from preprocessing.augmentation import (
        augment_dataset_offline,
        balance_classes_offline,
        get_extended_offline_pipeline,
        get_standard_offline_pipeline,
        plot_augmentation_examples,
        plot_pipeline_comparison,
    )
    _AUGMENTATION_AVAILABLE = True

except ImportError as e:
    logger.warning(
        f"augmentation.py konnte nicht geladen werden: {e}\n"
        f"→ pip install albumentations"
    )


# ── prepare_dataset.py ────────────────────────────────────────
try:
    from preprocessing.prepare_dataset import (
        prepare_dataset,
        split_dataset,
        validate_raw_dataset,
        verify_processed_dataset,
    )
    _PREPARE_AVAILABLE = True

except ImportError as e:
    logger.warning(
        f"prepare_dataset.py konnte nicht geladen werden: {e}\n"
        f"→ Alle Abhängigkeiten installieren"
    )


# ── Lazy Loading Hilfsfunktionen ──────────────────────────────
def get_filter_pipeline():
    """
    Lazy Loading für apply_filter_pipeline.

    Warum Lazy Loading:
        cv2, numpy etc. werden erst geladen wenn
        diese Funktion aufgerufen wird – nicht beim Import.
        → Schnellerer Programmstart falls Filter nicht gebraucht.

    Returns:
        apply_filter_pipeline Funktion

    Raises:
        ImportError : Falls cv2 nicht installiert
    """
    if not _FILTER_AVAILABLE:
        raise ImportError(
            "filter.py nicht verfügbar.\n"
            "→ pip install opencv-python"
        )
    from preprocessing.filter import apply_filter_pipeline
    return apply_filter_pipeline


def get_normalization_pipeline():
    """
    Lazy Loading für apply_normalization_pipeline.

    Returns:
        apply_normalization_pipeline Funktion

    Raises:
        ImportError : Falls Abhängigkeiten fehlen
    """
    if not _NORMALIZATION_AVAILABLE:
        raise ImportError(
            "normalization.py nicht verfügbar.\n"
            "→ pip install opencv-python numpy"
        )
    from preprocessing.normalization import apply_normalization_pipeline
    return apply_normalization_pipeline


def get_augmentation_pipeline(extended: bool = False):
    """
    Lazy Loading für Augmentierungs-Pipeline.

    Args:
        extended : Erweiterte Pipeline (True) oder Standard (False)

    Returns:
        albumentations Pipeline

    Raises:
        ImportError : Falls albumentations nicht installiert
    """
    if not _AUGMENTATION_AVAILABLE:
        raise ImportError(
            "augmentation.py nicht verfügbar.\n"
            "→ pip install albumentations"
        )
    if extended:
        from preprocessing.augmentation import get_extended_offline_pipeline
        return get_extended_offline_pipeline()
    else:
        from preprocessing.augmentation import get_standard_offline_pipeline
        return get_standard_offline_pipeline()


# ── Status-Übersicht ──────────────────────────────────────────
def print_status() -> None:
    """
    Zeigt welche Module verfügbar sind.

    Nützlich zum Debuggen falls Imports fehlschlagen:
        from preprocessing import print_status
        print_status()

    Ausgabe:
        preprocessing v1.0.0
        ✓ filter.py
        ✓ normalization.py
        ✗ augmentation.py  ← albumentations fehlt
        ✓ prepare_dataset.py
    """
    print(f"\npreprocessing v{__version__}")
    print("-" * 35)
    modules = {
        "filter.py"          : _FILTER_AVAILABLE,
        "normalization.py"   : _NORMALIZATION_AVAILABLE,
        "augmentation.py"    : _AUGMENTATION_AVAILABLE,
        "prepare_dataset.py" : _PREPARE_AVAILABLE,
    }
    for name, available in modules.items():
        status = "✓" if available else "✗"
        color  = "" if available else " ← fehlt!"
        print(f"  {status} {name}{color}")
    print()


# ── __all__ ───────────────────────────────────────────────────
__all__ = [
    # Metadaten
    "__version__",
    "__author__",
    "__date__",

    # Status
    "print_status",

    # Lazy Loading
    "get_filter_pipeline",
    "get_normalization_pipeline",
    "get_augmentation_pipeline",

    # Verfügbarkeits-Flags
    "_FILTER_AVAILABLE",
    "_NORMALIZATION_AVAILABLE",
    "_AUGMENTATION_AVAILABLE",
    "_PREPARE_AVAILABLE",

    # filter.py
    "apply_filter_pipeline",
    "apply_gaussian_blur",
    "apply_median_blur",
    "check_image_quality",
    "enhance_contrast_clahe",
    "enhance_sharpness",
    "normalize_staining_macenko",
    "plot_filter_comparison",
    "process_batch",
    "process_image_file",
    "remove_artifacts",
    "remove_background",

    # normalization.py
    "apply_normalization_pipeline",
    "compute_channel_statistics",
    "denormalize_imagenet",
    "denormalize_zscore",
    "normalize_imagenet",
    "normalize_minmax",
    "normalize_percentile",
    "normalize_zscore",
    "plot_normalization_comparison",

    # augmentation.py
    "augment_dataset_offline",
    "balance_classes_offline",
    "get_extended_offline_pipeline",
    "get_standard_offline_pipeline",
    "plot_augmentation_examples",
    "plot_pipeline_comparison",

    # prepare_dataset.py
    "prepare_dataset",
    "split_dataset",
    "validate_raw_dataset",
    "verify_processed_dataset",
]