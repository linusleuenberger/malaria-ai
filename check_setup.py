# ============================================================
# check_setup.py - schnelle Pruefung vor dem Training
# Verwendung: python check_setup.py
# ============================================================

import sys
from pathlib import Path

OK, FAIL, WARN = "[OK]  ", "[FAIL]", "[WARN]"


def check(label, cond, fix=""):
    print(f"  {OK if cond else FAIL}  {label}")
    if not cond and fix:
        print(f"        -> {fix}")
    return cond


def section(title):
    print(f"\n{'-'*52}\n  {title}\n{'-'*52}")


# ── Python ────────────────────────────────────────────────────
section("Python")
v = sys.version_info
check(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 9), "Python 3.9+ noetig")

# ── Bibliotheken ──────────────────────────────────────────────
section("Bibliotheken")
GPU_HINT = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
libs = {
    "torch": GPU_HINT, "torchvision": GPU_HINT,
    "PIL": "pip install Pillow", "cv2": "pip install opencv-python",
    "numpy": "pip install numpy", "sklearn": "pip install scikit-learn",
    "matplotlib": "pip install matplotlib", "tqdm": "pip install tqdm",
}
for lib, fix in libs.items():
    try:
        __import__(lib)
        check(lib, True)
    except ImportError:
        check(lib, False, fix)

# ── GPU ───────────────────────────────────────────────────────
section("GPU & CUDA")
try:
    import torch
    cuda = torch.cuda.is_available()
    check("CUDA verfuegbar", cuda,
          "GPU nicht erkannt -> CPU-Training waere sehr langsam.\n        " + GPU_HINT)
    print(f"        PyTorch {torch.__version__}", end="")
    if cuda:
        p = torch.cuda.get_device_properties(0)
        print(f" | CUDA {torch.version.cuda} | {p.name} ({p.total_memory/1e9:.1f} GB)")
        check("bfloat16-Unterstuetzung", torch.cuda.is_bf16_supported())
    else:
        print(" | nur CPU")
except ImportError:
    check("torch importierbar", False, GPU_HINT)

# ── Daten ─────────────────────────────────────────────────────
section("Daten")
base = Path(__file__).parent
for c in ("healthy", "infected"):
    d = base / "data" / "raw" / c
    n = len(list(d.glob("*.*"))) if d.exists() else 0
    check(f"data/raw/{c}/ ({n} Bilder)", n > 0, "Datensatz nach data/raw/ legen")

processed = base / "data" / "processed"
if processed.exists():
    for split in ("train", "val", "test"):
        n = sum(1 for f in (processed / split).rglob("*.*") if f.is_file()) \
            if (processed / split).exists() else 0
        check(f"data/processed/{split}/ ({n} Bilder)", n > 0,
              "python main.py --mode preprocess")
else:
    print(f"  {WARN}  data/processed/ fehlt -> python main.py --mode preprocess")

# ── Naechste Schritte ─────────────────────────────────────────
section("Naechste Schritte")
print("  1. python main.py --mode preprocess     (nur einmal noetig)")
print("  2. python main.py --mode train --eval-after")
print("  3. python analyze.py                     (Zuverlaessigkeits-Analyse)")
print()
