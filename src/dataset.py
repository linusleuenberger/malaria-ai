"""
src/dataset.py - Dataset & DataLoader

Erwartete Ordnerstruktur (vom Preprocessing erzeugt):

    data/processed/
        train/  {healthy, infected}    <- Training
        val/    {healthy, infected}    <- Validierung (Modellauswahl)
        test/   {healthy, infected}    <- nur am Ende: ehrlicher Test

Jeder Schritt nutzt seinen eigenen Split, damit das Ergebnis
nicht beschoenigt wird.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.config import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    MEAN,
    STD,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
)

logger = logging.getLogger(__name__)

VALID_EXT = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})


# ── Dataset ───────────────────────────────────────────────────
class MalariaDataset(Dataset):
    """Liest Bildpfade beim Start ein, laedt Pixel erst bei Bedarf."""

    def __init__(self, root_dir: str | Path, transform: Optional[Callable] = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Verzeichnis fehlt: {self.root_dir}\n"
                f"-> zuerst 'python main.py --mode preprocess' ausfuehren.")

        self.samples: List[Tuple[Path, int]] = []
        for name, idx in CLASS_TO_IDX.items():
            class_dir = self.root_dir / name
            if not class_dir.is_dir():
                logger.warning("Klassenordner fehlt: %s", class_dir)
                continue
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in VALID_EXT:
                    self.samples.append((p, idx))

        if not self.samples:
            raise RuntimeError(f"Keine Bilder in {self.root_dir} gefunden.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def class_counts(self) -> Dict[str, int]:
        counts = {n: 0 for n in CLASS_TO_IDX}
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return counts

    def sample_weights(self) -> torch.Tensor:
        """Gewicht pro Sample fuer den WeightedRandomSampler (Klassenbalance)."""
        counts = self.class_counts()
        n_total, n_cls = len(self.samples), len(CLASS_TO_IDX)
        per_class = {c: (n_total / (n_cls * n) if n else 0.0) for c, n in counts.items()}
        return torch.tensor([per_class[IDX_TO_CLASS[l]] for _, l in self.samples],
                            dtype=torch.float32)

    def labels(self) -> List[int]:
        return [l for _, l in self.samples]


# ── Transforms ────────────────────────────────────────────────
def get_train_transforms(img_size: int = 224, mean=MEAN, std=STD) -> transforms.Compose:
    """Augmentierte Pipeline fuer das Training (Online-Augmentierung)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),     # Zellen haben kein Oben/Unten
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.15, 0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.92, 1.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_eval_transforms(img_size: int = 224, mean=MEAN, std=STD) -> transforms.Compose:
    """Pipeline fuer Val/Test/Inferenz - keine Augmentierung."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── DataLoader ────────────────────────────────────────────────
def get_dataloaders(
    data_dir: str | Path,
    img_size: int = 224,
    batch_size: int = 256,
    num_workers: int = NUM_WORKERS,
    use_weighted_sampler: bool = True,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, DataLoader]:
    """Erstellt DataLoader fuer die gewuenschten Splits."""
    data_dir = Path(data_dir)
    train_tf, eval_tf = get_train_transforms(img_size), get_eval_transforms(img_size)

    persistent = PERSISTENT_WORKERS and num_workers > 0
    prefetch = PREFETCH_FACTOR if num_workers > 0 else None

    def _loader(split: str) -> DataLoader:
        is_train = split == "train"
        ds = MalariaDataset(data_dir / split, transform=train_tf if is_train else eval_tf)
        sampler = None
        if is_train and use_weighted_sampler:
            w = ds.sample_weights()
            sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(is_train and sampler is None),
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            drop_last=is_train,
        )

    loaders = {s: _loader(s) for s in splits}

    logger.info("DataLoader bereit (Batch=%d, Workers=%d):", batch_size, num_workers)
    for s in splits:
        c = loaders[s].dataset.class_counts()
        logger.info("  %-5s %6d Bilder | healthy=%d infected=%d",
                    s, len(loaders[s].dataset), c["healthy"], c["infected"])
    return loaders


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.config import PROCESSED_DIR
    dl = get_dataloaders(PROCESSED_DIR, batch_size=8, num_workers=0)
    x, y = next(iter(dl["train"]))
    print("Batch:", tuple(x.shape), "Labels:", y[:8].tolist())
    print("[OK] dataset.py")
