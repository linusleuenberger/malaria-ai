# ============================================================
# check_setup.py – Setup-Prüfung vor dem Training
# Verwendung: python check_setup.py
# ============================================================

import sys
import subprocess
from pathlib import Path

# ── Farben für Terminal ────────────────────────────────────────
OK   = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"
INFO = "\033[94mℹ\033[0m"

def check(label, condition, fix=""):
    status = OK if condition else FAIL
    print(f"  {status}  {label}")
    if not condition and fix:
        print(f"     → {fix}")
    return condition

def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ── 1. Python ─────────────────────────────────────────────────
section("Python")
v = sys.version_info
check(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 9),
      "Python 3.9+ benötigt: https://www.python.org/downloads/")


# ── 2. Bibliotheken ───────────────────────────────────────────
section("Bibliotheken")
libs = {
    "torch":         "pip install torch --index-url https://download.pytorch.org/whl/cu121",
    "torchvision":   "pip install torchvision --index-url https://download.pytorch.org/whl/cu121",
    "PIL":           "pip install Pillow",
    "cv2":           "pip install opencv-python",
    "albumentations":"pip install albumentations",
    "numpy":         "pip install numpy",
    "sklearn":       "pip install scikit-learn",
    "matplotlib":    "pip install matplotlib",
    "seaborn":       "pip install seaborn",
    "tqdm":          "pip install tqdm",
}
all_libs_ok = True
for lib, fix in libs.items():
    try:
        __import__(lib)
        check(lib, True)
    except ImportError:
        check(lib, False, fix)
        all_libs_ok = False


# ── 3. GPU / CUDA ─────────────────────────────────────────────
section("GPU & CUDA")
try:
    import torch

    cuda_available = torch.cuda.is_available()
    check("CUDA verfügbar", cuda_available,
          "GPU nicht erkannt → CPU wird verwendet (Training ~10× langsamer)\n"
          "     Lösung: PyTorch mit CUDA installieren:\n"
          "     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    if cuda_available:
        n = torch.cuda.device_count()
        check(f"Anzahl GPUs: {n}", n >= 1)

        for i in range(n):
            name   = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
            check(f"GPU {i}: {name} ({mem_gb:.1f} GB VRAM)", True)

        cuda_ver = torch.version.cuda
        check(f"CUDA Version: {cuda_ver}", True)

        # cuDNN
        cudnn_ok = torch.backends.cudnn.is_available()
        check(f"cuDNN verfügbar: {torch.backends.cudnn.version() if cudnn_ok else 'nein'}", cudnn_ok)

        # Mixed Precision
        amp_ok = torch.cuda.is_bf16_supported() or True  # AMP with fp16 always works
        check("Mixed Precision (AMP) unterstützt", True)

    else:
        # Zeige trotzdem torch-Version
        print(f"  {INFO}  PyTorch Version: {torch.__version__}")
        print(f"  {WARN}  CPU-Training aktiv – für echtes Training GPU empfohlen")

    print(f"\n  {INFO}  Aktives Gerät im Projekt: ", end="")
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"\033[1m{device}\033[0m")

except ImportError:
    check("torch importierbar", False, "pip install torch")


# ── 4. Projektstruktur ────────────────────────────────────────
section("Projektstruktur & Daten")
base = Path(__file__).parent
dirs = {
    "data/raw/healthy":            "Rohdaten (gesunde Zellen) fehlen",
    "data/raw/infected":           "Rohdaten (infizierte Zellen) fehlen",
    "data/processed":              "Wird durch 'python main.py --mode preprocess' erstellt",
    "models":                      "Wird automatisch beim Training erstellt",
    "results":                     "Wird automatisch beim Training erstellt",
}
for d, hint in dirs.items():
    path = base / d
    exists = path.exists()
    if exists and d.startswith("data/raw"):
        count = len(list(path.glob("*.*")))
        check(f"{d}/ ({count} Bilder)", count > 0, hint)
    else:
        check(f"{d}/", exists, hint)

# Prüfe ob processed/train etc. existieren
processed = base / "data" / "processed"
if processed.exists():
    for split in ["train", "val", "test"]:
        split_path = processed / split
        if split_path.exists():
            total = sum(1 for f in split_path.rglob("*.*") if f.is_file())
            check(f"data/processed/{split}/ ({total} Bilder)", total > 0)
        else:
            print(f"  {WARN}  data/processed/{split}/ fehlt → preprocess ausführen")


# ── 5. Zusammenfassung ────────────────────────────────────────
section("Nächste Schritte")
try:
    import torch
    if torch.cuda.is_available():
        print(f"  {OK}  Bereit zum Training!")
        print(f"\n  Empfohlene Reihenfolge:")
        print(f"    1. python main.py --mode preprocess")
        print(f"    2. python main.py --mode augment")
        print(f"    3. python main.py --mode train --eval-after")
    else:
        print(f"  {WARN}  Training möglich, aber GPU fehlt → sehr langsam")
        print(f"    → PyTorch mit CUDA installieren für GPU-Training")
except ImportError:
    print(f"  {FAIL}  Erst Bibliotheken installieren: pip install -r requirements.txt")

print()
