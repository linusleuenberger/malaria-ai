"""
dataset.py – MalariaDataset Klasse und DataLoader-Erstellung
=============================================================

Erwartet folgende Ordnerstruktur (nach prepare_dataset.py):

    data/processed/
    ├── train/
    │   ├── infected/
    │   └── healthy/
    ├── val/
    │   ├── infected/
    │   └── healthy/
    └── test/
        ├── infected/
        └── healthy/
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────────────────
# Klassen-Mapping
# ──────────────────────────────────────────────────────────────────────────────

CLASS_TO_IDX: Dict[str, int] = {"healthy": 0, "infected": 1}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

VALID_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-Klasse
# ──────────────────────────────────────────────────────────────────────────────

class MalariaDataset(Dataset):
    """
    PyTorch Dataset für Malaria-Zellbilder.

    Lädt alle Bildpfade beim Start (lazy loading) – die Pixel
    werden erst in __getitem__ gelesen. Spart RAM bei großen Datensätzen.

    Args:
        root_dir   : Pfad zum Split-Ordner, z.B. data/processed/train
        transform  : torchvision.transforms Pipeline
        extensions : Erlaubte Dateiendungen

    Beispiel:
        >>> ds = MalariaDataset("data/processed/train", transform=my_tf)
        >>> img, label = ds[0]   # label: 0=healthy, 1=infected
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        extensions: Optional[frozenset[str]] = None,
    ) -> None:
        self.root_dir   = Path(root_dir)
        self.transform  = transform
        self.extensions = extensions or VALID_EXTENSIONS

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Verzeichnis nicht gefunden: {self.root_dir}\n"
                f"→ Zuerst preprocessing/prepare_dataset.py ausführen."
            )

        self.samples: List[Tuple[Path, int]] = []
        self._scan_directory()

        if not self.samples:
            raise RuntimeError(
                f"Keine Bilder in '{self.root_dir}' gefunden.\n"
                f"Erlaubte Endungen: {self.extensions}\n"
                f"Erwartete Unterordner: 'infected/' und 'healthy/'"
            )

    def _scan_directory(self) -> None:
        """Durchsucht root_dir/infected/ und root_dir/healthy/ nach Bildern."""
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name

            if not class_dir.is_dir():
                print(f"[WARN] Fehlender Klassenordner: {class_dir}")
                continue

            found = 0
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((img_path, class_idx))
                    found += 1

            if found == 0:
                print(f"[WARN] Keine Bilder in: {class_dir}")

    # ── Dataset API ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Immer als RGB öffnen → 3 Kanäle garantiert (auch bei Graustufenbildern)
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    # ── Hilfsmethoden ─────────────────────────────────────────────────────────

    def get_class_counts(self) -> Dict[str, int]:
        """Gibt Anzahl Bilder pro Klasse zurück."""
        counts = {name: 0 for name in CLASS_TO_IDX}
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """
        Berechnet Sample-Gewichte für WeightedRandomSampler.

        Seltenere Klassen bekommen höheres Gewicht → Klassenungleichgewicht
        wird beim Training automatisch ausgeglichen.

        Formel: weight = total / (n_classes × count_in_class)

        Returns:
            1-D Float-Tensor, ein Gewicht pro Sample.
        """
        counts    = self.get_class_counts()
        n_total   = len(self.samples)
        n_classes = len(CLASS_TO_IDX)

        weight_map: Dict[str, float] = {}
        for class_name, count in counts.items():
            if count == 0:
                print(f"[WARN] Klasse '{class_name}' hat 0 Samples!")
                weight_map[class_name] = 0.0
            else:
                weight_map[class_name] = n_total / (n_classes * count)

        return torch.tensor(
            [weight_map[IDX_TO_CLASS[label]] for _, label in self.samples],
            dtype=torch.float32,
        )

    def get_labels(self) -> List[int]:
        """Alle Labels als Liste (nützlich für sklearn-Metriken)."""
        return [label for _, label in self.samples]

    def __repr__(self) -> str:
        c = self.get_class_counts()
        ratio = (
            f"{c['infected'] / c['healthy']:.2f}" if c["healthy"] > 0 else "inf"
        )
        return (
            f"MalariaDataset(\n"
            f"  root     = {self.root_dir}\n"
            f"  total    = {len(self.samples):,}\n"
            f"  healthy  = {c['healthy']:,}\n"
            f"  infected = {c['infected']:,}\n"
            f"  ratio    = {ratio}  (infected/healthy)\n"
            f")"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Transform-Pipelines
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(
    img_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std:  Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Augmentierte Pipeline für das Training.

    Schritte:
        Resize → HFlip → VFlip → Rotation → ColorJitter → Affine
        → ToTensor → Normalize
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),       # Zellen haben kein Oben/Unten
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.05,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.92, 1.08),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transforms(
    img_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std:  Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> transforms.Compose:
    """
    Minimale Pipeline für Validation und Test.
    Keine Augmentierung – nur Resize und Normalize.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 8,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    use_weighted_sampler: bool = True,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std:  Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Dict[str, DataLoader]:
    """
    Erstellt DataLoader fuer train, val und test in einem Schritt.

    Args:
        data_dir             : Pfad zu data/processed/
        img_size             : Bildgroesse in Pixel (quadratisch)
        batch_size           : Bilder pro Batch (128-256 fuer RTX optimal)
        num_workers          : Parallele Worker (8 fuer schnellen Transfer)
        pin_memory           : Schnellerer GPU-Transfer (nur mit GPU sinnvoll)
        persistent_workers   : Worker zwischen Epochen am Leben lassen
        prefetch_factor      : Batches die CPU vorlaedt waehrend GPU arbeitet
        use_weighted_sampler : Klassenungleichgewicht automatisch ausgleichen
        mean                 : RGB-Mittelwerte (am besten mit compute_dataset_stats())
        std                  : RGB-Standardabweichungen

    Returns:
        Dict mit Keys 'train', 'val', 'test'
    """
    data_dir = Path(data_dir)

    # persistent_workers braucht num_workers > 0
    _persistent = persistent_workers and num_workers > 0
    _prefetch   = prefetch_factor if num_workers > 0 else None

    train_tf = get_train_transforms(img_size, mean, std)
    eval_tf  = get_eval_transforms(img_size, mean, std)

    datasets: Dict[str, MalariaDataset] = {
        "train": MalariaDataset(data_dir / "train", transform=train_tf),
        "val"  : MalariaDataset(data_dir / "val",   transform=eval_tf),
        "test" : MalariaDataset(data_dir / "test",  transform=eval_tf),
    }

    # WeightedRandomSampler nur fuer Training
    train_sampler: Optional[WeightedRandomSampler] = None
    if use_weighted_sampler:
        weights = datasets["train"].get_class_weights()
        train_sampler = WeightedRandomSampler(
            weights     = weights,
            num_samples = len(weights),
            replacement = True,
        )

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            datasets["train"],
            batch_size         = batch_size,
            sampler            = train_sampler,
            shuffle            = (train_sampler is None),
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            persistent_workers = _persistent,
            prefetch_factor    = _prefetch,
            drop_last          = True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size         = batch_size,
            shuffle            = False,
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            persistent_workers = _persistent,
            prefetch_factor    = _prefetch,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size         = batch_size,
            shuffle            = False,
            num_workers        = num_workers,
            pin_memory         = pin_memory,
            persistent_workers = _persistent,
            prefetch_factor    = _prefetch,
        ),
    }

    # Uebersicht ausgeben (ASCII-sicher)
    print("\n" + "=" * 52)
    print("  DataLoader - Uebersicht")
    print("=" * 52)
    print(f"  {'Split':<12s}  {'Gesamt':>6}  {'Healthy':>8}  {'Infected':>8}")
    print("-" * 52)
    for split, ds in datasets.items():
        c = ds.get_class_counts()
        print(f"  {split:<12s}  {len(ds):>6,}  {c['healthy']:>8,}  {c['infected']:>8,}")
    sampler_str = "WeightedSampler [OK]" if use_weighted_sampler else "RandomShuffle"
    print("-" * 52)
    print(f"  Batch: {batch_size}  Workers: {num_workers}  Sampling: {sampler_str}")
    print("=" * 52 + "\n")

    return loaders


# ──────────────────────────────────────────────────────────────────────────────
# Datensatz-Statistiken berechnen
# ──────────────────────────────────────────────────────────────────────────────

def compute_dataset_stats(
    data_dir: str | Path,
    img_size: int = 224,
    num_workers: int = 4,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Berechnet pixelgenaue Mean & Std des Trainingsdatensatzes (kanalweise, RGB).

    Nur einmal ausführen → Werte in config.py unter MEAN und STD eintragen.

    Returns:
        (mean_rgb, std_rgb) – je ein Tupel mit 3 Float-Werten für R, G, B

    Beispiel:
        >>> mean, std = compute_dataset_stats("data/processed")
        >>> # → Werte in config.py eintragen
    """
    base_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0, 1], noch keine Normalisierung
    ])

    dataset = MalariaDataset(Path(data_dir) / "train", transform=base_tf)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=num_workers)

    channel_sum    = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    n_pixels = 0

    print("Berechne Datensatz-Statistiken...")
    for images, _ in loader:
        B, C, H, W  = images.shape
        n_pixels    += B * H * W
        channel_sum    += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

    mean = channel_sum / n_pixels
    std  = torch.sqrt(channel_sum_sq / n_pixels - mean ** 2)

    mean_t = tuple(round(v.item(), 4) for v in mean)
    std_t  = tuple(round(v.item(), 4) for v in std)

    print(f"  mean (R, G, B) = {mean_t}")
    print(f"  std  (R, G, B) = {std_t}")
    print("  → In config.py unter MEAN und STD eintragen!\n")

    return mean_t, std_t


# ──────────────────────────────────────────────────────────────────────────────
# Quick-Test:  python -m src.dataset
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    base_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

    if not base_dir.exists():
        print(f"\n[ERROR] Nicht gefunden: {base_dir}")
        print("→ Zuerst 'python preprocessing/prepare_dataset.py' ausführen.\n")
        sys.exit(1)

    # Einmalig Statistiken berechnen (danach auskommentieren):
    # compute_dataset_stats(base_dir)

    loaders = get_dataloaders(
        data_dir             = base_dir,
        batch_size           = 32,
        num_workers          = 0,      # 0 für lokalen Test
        pin_memory           = False,  # False ohne GPU
        use_weighted_sampler = True,
    )

    images, labels = next(iter(loaders["train"]))
    print(f"Batch-Shape  : {list(images.shape)}")   # [32, 3, 224, 224]
    print(f"Label-Shape  : {list(labels.shape)}")   # [32]
    print(f"Tensor-Range : [{images.min():.3f}, {images.max():.3f}]")
    print(f"Erste Labels : {[IDX_TO_CLASS[l.item()] for l in labels[:6]]}")
    print("\n✓ dataset.py funktioniert korrekt.")