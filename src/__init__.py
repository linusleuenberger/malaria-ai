# ============================================================
# src/__init__.py – pragmatische Version
# ============================================================

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__author__  = "Malaria-AI Team"

# ── Imports mit Fehlerbehandlung ──────────────────────────────
try:
    from src.model   import build_model, load_model
    from src.train   import train, EarlyStopping
    from src.evaluate import evaluate, plot_training_history
    from src.utils   import (
        set_seed,
        sanity_check,
        print_system_info,
        format_time,
    )
except ImportError as e:
    logger.warning(
        f"src konnte nicht vollständig geladen werden: {e}\n"
        f"→ pip install torch torchvision scikit-learn"
    )

# ── Status ────────────────────────────────────────────────────
def print_status() -> None:
    """Zeigt welche Module verfügbar sind."""
    modules = {
        "model.py"   : "build_model"        in dir(),
        "train.py"   : "train"              in dir(),
        "evaluate.py": "evaluate"           in dir(),
        "utils.py"   : "set_seed"           in dir(),
    }
    print(f"\nsrc v{__version__}")
    print("-" * 30)
    for name, ok in modules.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print()

__all__ = [
    "__version__",
    "print_status",
    "build_model",
    "load_model",
    "train",
    "EarlyStopping",
    "evaluate",
    "plot_training_history",
    "set_seed",
    "sanity_check",
    "print_system_info",
    "format_time",
]