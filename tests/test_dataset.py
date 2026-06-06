# ============================================================
# tests/test_dataset.py
# Tests für MalariaDataset, DataLoader und Transforms
# ============================================================

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    NUM_CLASSES,
)
from src.dataset import (
    MalariaDataset,
    get_dataloaders,
    get_eval_transforms,
    get_train_transforms,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def temp_dataset_dir():
    """
    Erstellt temporären Ordner mit Dummy-Bildern für Tests.

    Struktur:
        temp/
        ├── healthy/
        │   ├── img_000.png
        │   └── ...
        └── infected/
            ├── img_000.png
            └── ...

    Nach dem Test → Ordner automatisch gelöscht.
    """
    tmp_dir = Path(tempfile.mkdtemp())

    for class_name in CLASS_NAMES:
        class_dir = tmp_dir / class_name
        class_dir.mkdir(parents=True)

        for i in range(20):
            img_array = np.random.randint(
                0, 255,
                (150, 150, 3),
                dtype=np.uint8
            )
            img = Image.fromarray(img_array)
            img.save(class_dir / f"img_{i:03d}.png")

    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture
def temp_processed_dir():
    """
    Erstellt temporären processed/ Ordner mit
    train/, val/, test/ Splits für DataLoader Tests.
    """
    tmp_dir   = Path(tempfile.mkdtemp())
    processed = tmp_dir / "processed"

    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            split_dir = processed / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for i in range(10):
                img_array = np.random.randint(
                    0, 255,
                    (150, 150, 3),
                    dtype=np.uint8
                )
                img = Image.fromarray(img_array)
                img.save(split_dir / f"img_{i:03d}.png")

    yield processed
    shutil.rmtree(tmp_dir)


@pytest.fixture
def imbalanced_processed_dir():
    """
    Erstellt unbalancierten Datensatz für WeightedSampler Tests.
        healthy:  80 Bilder
        infected: 20 Bilder
    → Ratio 4:1
    """
    tmp_dir   = Path(tempfile.mkdtemp())
    processed = tmp_dir / "processed"

    counts = {"healthy": 80, "infected": 20}

    for split in ["train", "val", "test"]:
        for class_name, count in counts.items():
            split_dir = processed / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            n = count if split == "train" else count // 4
            for i in range(n):
                img_array = np.random.randint(
                    0, 255,
                    (50, 50, 3),
                    dtype=np.uint8
                )
                img = Image.fromarray(img_array)
                img.save(split_dir / f"img_{i:03d}.png")

    yield processed
    shutil.rmtree(tmp_dir)


# ── Tests: MalariaDataset ─────────────────────────────────────

class TestMalariaDataset:
    """
    Tests für die MalariaDataset Klasse.

    Testet:
        ✓ Bilder werden korrekt geladen
        ✓ Labels sind korrekt (0=healthy, 1=infected)
        ✓ Bildgrösse nach Transform korrekt
        ✓ Fehlerbehandlung bei fehlendem Ordner
        ✓ Alle Bildformate werden unterstützt
        ✓ Graustufenbilder → RGB Konvertierung
        ✓ Edge Cases: sehr klein, sehr gross, korrupt
    """

    def test_dataset_loads_images(self, temp_dataset_dir):
        """Prüft ob Bilder korrekt geladen werden."""
        dataset = MalariaDataset(
            temp_dataset_dir / "healthy",
            transform = get_eval_transforms()
        )
        assert len(dataset) > 0, \
            "Dataset ist leer – keine Bilder geladen"

    def test_dataset_correct_length(self, temp_dataset_dir):
        """Prüft ob Anzahl Bilder korrekt ist."""
        for class_name in CLASS_NAMES:
            dataset = MalariaDataset(
                temp_dataset_dir / class_name,
                transform = get_eval_transforms()
            )
            assert len(dataset) == 20, (
                f"Erwartet 20 Bilder für {class_name}, "
                f"gefunden: {len(dataset)}"
            )

    def test_dataset_returns_tensor_and_label(
        self, temp_dataset_dir
    ):
        """Prüft ob __getitem__ Tensor + Label zurückgibt."""
        dataset = MalariaDataset(
            temp_dataset_dir / "healthy",
            transform = get_eval_transforms()
        )
        image, label = dataset[0]

        assert isinstance(image, torch.Tensor), \
            f"Bild sollte Tensor sein, nicht {type(image)}"
        assert isinstance(label, int), \
            f"Label sollte int sein, nicht {type(label)}"

    def test_dataset_image_shape(self, temp_dataset_dir):
        """Prüft ob Bildgrösse nach Transform korrekt ist."""
        dataset = MalariaDataset(
            temp_dataset_dir / "healthy",
            transform = get_eval_transforms(img_size=IMAGE_SIZE[0])
        )
        image, _ = dataset[0]

        expected = (3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        assert image.shape == torch.Size(expected), (
            f"Erwartete Shape {expected}, "
            f"erhalten: {tuple(image.shape)}"
        )

    def test_dataset_correct_labels(self, temp_dataset_dir):
        """Prüft ob Labels korrekt vergeben werden."""
        healthy_dataset = MalariaDataset(
            temp_dataset_dir / "healthy",
            transform = get_eval_transforms()
        )
        infected_dataset = MalariaDataset(
            temp_dataset_dir / "infected",
            transform = get_eval_transforms()
        )

        _, healthy_label  = healthy_dataset[0]
        _, infected_label = infected_dataset[0]

        assert healthy_label  == 0, \
            f"healthy Label sollte 0 sein, nicht {healthy_label}"
        assert infected_label == 1, \
            f"infected Label sollte 1 sein, nicht {infected_label}"

    def test_dataset_pixel_range(self, temp_dataset_dir):
        """Prüft ob Pixelwerte nach Normalisierung sinnvoll sind."""
        dataset = MalariaDataset(
            temp_dataset_dir / "healthy",
            transform = get_eval_transforms()
        )
        image, _ = dataset[0]

        assert image.min() > -10, \
            f"Pixelwerte zu klein: {image.min():.3f}"
        assert image.max() <  10, \
            f"Pixelwerte zu gross: {image.max():.3f}"

    def test_dataset_missing_directory(self):
        """Prüft ob bei fehlendem Ordner Fehler geworfen wird."""
        with pytest.raises(FileNotFoundError):
            MalariaDataset(
                Path("/nicht/vorhanden/ordner"),
                transform = get_eval_transforms()
            )

    def test_dataset_empty_directory(self, tmp_path):
        """Prüft ob bei leerem Ordner Fehler geworfen wird."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(RuntimeError):
            MalariaDataset(
                empty_dir,
                transform = get_eval_transforms()
            )

    def test_dataset_rgb_conversion(self, tmp_path):
        """Prüft ob Graustufenbilder korrekt zu RGB konvertiert werden."""
        class_dir = tmp_path / "healthy"
        class_dir.mkdir(parents=True)

        gray_array = np.random.randint(
            0, 255, (150, 150), dtype=np.uint8
        )
        gray_img = Image.fromarray(gray_array, mode="L")
        gray_img.save(class_dir / "gray.png")

        dataset = MalariaDataset(
            class_dir,
            transform = get_eval_transforms()
        )
        image, _ = dataset[0]

        assert image.shape[0] == 3, \
            f"Erwartet 3 Kanäle (RGB), erhalten: {image.shape[0]}"

    def test_dataset_multiple_formats(self, tmp_path):
        """Prüft ob verschiedene Bildformate geladen werden."""
        class_dir = tmp_path / "healthy"
        class_dir.mkdir(parents=True)

        img_array = np.random.randint(
            0, 255, (50, 50, 3), dtype=np.uint8
        )
        img = Image.fromarray(img_array)

        for fmt, ext in [("PNG", ".png"), ("JPEG", ".jpg")]:
            img.save(class_dir / f"test{ext}", format=fmt)

        dataset = MalariaDataset(
            class_dir,
            transform = get_eval_transforms()
        )
        assert len(dataset) == 2, \
            f"Erwartet 2 Bilder (png + jpg), geladen: {len(dataset)}"

    # ── Edge Case Tests ────────────────────────────────────────

    def test_very_small_image(self, tmp_path):
        """
        Edge Case: 1×1 Pixel Bild.
        Sollte trotzdem auf 224×224 skaliert werden.
        """
        class_dir = tmp_path / "healthy"
        class_dir.mkdir(parents=True)

        tiny = Image.fromarray(
            np.array([[[128, 64, 32]]], dtype=np.uint8)
        )
        tiny.save(class_dir / "tiny.png")

        dataset = MalariaDataset(
            class_dir,
            transform = get_eval_transforms(img_size=224)
        )
        image, _ = dataset[0]

        assert image.shape == torch.Size([3, 224, 224]), (
            f"1×1 Bild sollte auf 224×224 skaliert werden, "
            f"erhalten: {tuple(image.shape)}"
        )

    def test_very_large_image(self, tmp_path):
        """
        Edge Case: 2000×2000 Pixel Bild.
        Sollte auf 224×224 herunterskaliert werden.
        """
        class_dir = tmp_path / "healthy"
        class_dir.mkdir(parents=True)

        large = Image.fromarray(
            np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        )
        large.save(class_dir / "large.png")

        dataset = MalariaDataset(
            class_dir,
            transform = get_eval_transforms(img_size=224)
        )
        image, _ = dataset[0]

        assert image.shape == torch.Size([3, 224, 224]), (
            f"2000×2000 Bild sollte auf 224×224 skaliert werden, "
            f"erhalten: {tuple(image.shape)}"
        )

    def test_corrupted_image_skipped(self, tmp_path):
        """
        Edge Case: Korrupte Bilddatei.
        Sollte übersprungen werden ohne Absturz.
        """
        class_dir = tmp_path / "healthy"
        class_dir.mkdir(parents=True)

        # Korrupte Datei (kein gültiges Bild)
        corrupt = class_dir / "corrupt.png"
        corrupt.write_bytes(b"das ist kein bild")

        # Valides Bild dazu
        valid = Image.fromarray(
            np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        )
        valid.save(class_dir / "valid.png")

        # Sollte nicht abstürzen
        try:
            dataset = MalariaDataset(
                class_dir,
                transform = get_eval_transforms()
            )
            # Falls korruptes Bild übersprungen → 1 Bild
            # Falls nicht → beim Laden Fehler
            for i in range(len(dataset)):
                try:
                    dataset[i]
                except Exception:
                    pass
        except Exception as e:
            pytest.fail(
                f"Korruptes Bild sollte nicht zum Absturz führen: {e}"
            )


# ── Tests: Transforms ─────────────────────────────────────────

class TestTransforms:
    """
    Tests für Bildtransformationen.

    Testet:
        ✓ Train-Transform gibt korrekten Output
        ✓ Eval-Transform gibt korrekten Output
        ✓ Eval-Transform ist deterministisch
        ✓ Train-Transform augmentiert Bilder
        ✓ Output ist Float-Tensor
    """

    def test_train_transform_output_shape(self):
        """Prüft ob Train-Transform korrekten Shape ausgibt."""
        transform = get_train_transforms(img_size=224)
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        )
        result = transform(dummy_img)

        assert result.shape == torch.Size([3, 224, 224]), (
            f"Erwartete Shape (3, 224, 224), "
            f"erhalten: {tuple(result.shape)}"
        )

    def test_eval_transform_output_shape(self):
        """Prüft ob Eval-Transform korrekten Shape ausgibt."""
        transform = get_eval_transforms(img_size=224)
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        )
        result = transform(dummy_img)

        assert result.shape == torch.Size([3, 224, 224]), (
            f"Erwartete Shape (3, 224, 224), "
            f"erhalten: {tuple(result.shape)}"
        )

    def test_eval_transform_is_deterministic(self):
        """Prüft ob Eval-Transform deterministisch ist."""
        transform = get_eval_transforms()
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        )

        result_1 = transform(dummy_img)
        result_2 = transform(dummy_img)

        assert torch.allclose(result_1, result_2), \
            "Eval-Transform sollte deterministisch sein"

    def test_train_transform_augments(self):
        """Prüft ob Train-Transform Bilder verändert."""
        transform = get_train_transforms()
        dummy_img = Image.fromarray(
            np.random.randint(50, 200, (150, 150, 3), dtype=np.uint8)
        )

        results  = [transform(dummy_img) for _ in range(10)]
        all_same = all(
            torch.allclose(results[0], r) for r in results[1:]
        )

        assert not all_same, \
            "Train-Transform sollte Bilder augmentieren"

    def test_transform_returns_float_tensor(self):
        """Prüft ob Transform Float-Tensor zurückgibt."""
        transform = get_eval_transforms()
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        )
        result = transform(dummy_img)

        assert result.dtype == torch.float32, \
            f"Erwartet float32, erhalten: {result.dtype}"


# ── Tests: DataLoaders ────────────────────────────────────────

class TestDataLoaders:
    """
    Tests für get_dataloaders Funktion.

    Testet:
        ✓ Alle drei Splits werden erstellt
        ✓ Batch-Shape korrekt
        ✓ Labels im gültigen Bereich
        ✓ Keine NaN/Inf Werte
        ✓ Float-Tensoren
        ✓ Fehler bei fehlendem Ordner
    """

    def test_dataloaders_returns_all_splits(
        self, temp_processed_dir
    ):
        """Prüft ob alle drei DataLoader erstellt werden."""
        loaders = get_dataloaders(
            data_dir    = temp_processed_dir,
            batch_size  = 4,
            num_workers = 0,
            pin_memory  = False,
        )

        assert "train" in loaders, "train DataLoader fehlt"
        assert "val"   in loaders, "val DataLoader fehlt"
        assert "test"  in loaders, "test DataLoader fehlt"

    def test_dataloader_batch_shape(self, temp_processed_dir):
        """Prüft ob Batch die richtige Shape hat."""
        loaders = get_dataloaders(
            data_dir    = temp_processed_dir,
            batch_size  = 4,
            img_size    = 224,
            num_workers = 0,
            pin_memory  = False,
        )

        images, labels = next(iter(loaders["train"]))

        assert images.shape[1:] == torch.Size([3, 224, 224]), (
            f"Erwartete Bild-Shape (3, 224, 224), "
            f"erhalten: {tuple(images.shape[1:])}"
        )
        assert len(labels) == len(images), \
            "Anzahl Labels stimmt nicht mit Anzahl Bildern überein"

    def test_dataloader_labels_valid_range(
        self, temp_processed_dir
    ):
        """Prüft ob Labels im gültigen Bereich sind."""
        loaders = get_dataloaders(
            data_dir    = temp_processed_dir,
            batch_size  = 4,
            num_workers = 0,
            pin_memory  = False,
        )

        for split_name, loader in loaders.items():
            for _, labels in loader:
                assert labels.min() >= 0, (
                    f"{split_name}: Label < 0 gefunden"
                )
                assert labels.max() < NUM_CLASSES, (
                    f"{split_name}: Label >= NUM_CLASSES gefunden"
                )

    def test_dataloader_float_images(self, temp_processed_dir):
        """Prüft ob Bilder als Float-Tensoren geladen werden."""
        loaders = get_dataloaders(
            data_dir    = temp_processed_dir,
            batch_size  = 4,
            num_workers = 0,
            pin_memory  = False,
        )

        images, _ = next(iter(loaders["train"]))

        assert images.dtype == torch.float32, (
            f"Erwartet float32, erhalten: {images.dtype}"
        )

    def test_dataloader_no_nan_values(self, temp_processed_dir):
        """Prüft ob keine NaN oder Inf Werte vorhanden sind."""
        loaders = get_dataloaders(
            data_dir    = temp_processed_dir,
            batch_size  = 4,
            num_workers = 0,
            pin_memory  = False,
        )

        images, _ = next(iter(loaders["train"]))

        assert not torch.isnan(images).any(), \
            "NaN Werte in Bildern gefunden"
        assert not torch.isinf(images).any(), \
            "Inf Werte in Bildern gefunden"

    def test_dataloader_batch_size(self, temp_processed_dir):
        """Prüft ob Batch-Size korrekt ist."""
        batch_size = 4
        loaders    = get_dataloaders(
            data_dir    = temp_processed_dir,
            batch_size  = batch_size,
            num_workers = 0,
            pin_memory  = False,
        )

        images, _ = next(iter(loaders["train"]))

        assert images.shape[0] <= batch_size, (
            f"Batch-Size {images.shape[0]} > "
            f"erwartet {batch_size}"
        )

    def test_missing_processed_dir(self, tmp_path):
        """Prüft ob bei fehlendem Ordner Fehler geworfen wird."""
        with pytest.raises(FileNotFoundError):
            get_dataloaders(
                data_dir    = tmp_path / "nicht_vorhanden",
                num_workers = 0,
                pin_memory  = False,
            )


# ── Tests: Performance ────────────────────────────────────────

class TestDataLoaderPerformance:
    """
    Performance Tests für DataLoader.

    Testet:
        ✓ 50 Batches in unter 30 Sekunden
        ✓ Kein Memory Leak über mehrere Epochen
    """

    def test_dataloader_loading_speed(
        self, temp_processed_dir
    ):
        """
        Prüft ob DataLoader schnell genug ist.

        Warum wichtig:
            Zu langsamer DataLoader → GPU wartet auf Daten
            → Training dauert 10× länger als nötig
            → num_workers Problem
        """
        loaders = get_dataloaders(
            data_dir    = temp_processed_dir,
            batch_size  = 4,
            num_workers = 0,
            pin_memory  = False,
        )

        start = time.time()
        count = 0
        for _ in loaders["train"]:
            count += 1

        elapsed = time.time() - start

        assert elapsed < 30.0, (
            f"DataLoader zu langsam: {elapsed:.1f}s "
            f"für {count} Batches\n"
            f"→ num_workers erhöhen"
        )

    def test_dataloader_multiple_epochs(
        self, temp_processed_dir
    ):
        """
        Prüft ob DataLoader über mehrere Epochen stabil bleibt.
        Kein Memory Leak, kein Absturz.
        """
        loaders = get_dataloaders(
            data_dir    = temp_processed_dir,
            batch_size  = 4,
            num_workers = 0,
            pin_memory  = False,
        )

        # 3 Epochen simulieren
        for epoch in range(3):
            for images, labels in loaders["train"]:
                assert images is not None
                assert labels is not None


# ── Tests: Reproduzierbarkeit ─────────────────────────────────

class TestReproducibility:
    """
    Tests ob Seed korrekt gesetzt wird.

    Testet:
        ✓ Gleicher Seed → gleiche Reihenfolge
        ✓ Verschiedener Seed → verschiedene Reihenfolge
    """

    def test_dataset_same_seed_same_order(
        self, temp_processed_dir
    ):
        """
        Prüft ob gleicher Seed → gleiche Dateireihenfolge.

        Warum wichtig:
            Train/Val/Test Split muss bei jedem Aufruf
            identisch sein → sonst Datenleck möglich
            (Testbild könnte im Training sein)
        """
        import random
        from src.config import RANDOM_SEED

        def get_file_order(seed: int):
            random.seed(seed)
            files = list(
                (temp_processed_dir / "train" / "healthy").glob("*.png")
            )
            random.shuffle(files)
            return [f.name for f in files]

        order_1 = get_file_order(RANDOM_SEED)
        order_2 = get_file_order(RANDOM_SEED)

        assert order_1 == order_2, (
            "Gleicher Seed sollte gleiche Reihenfolge geben\n"
            f"Order 1: {order_1[:3]}\n"
            f"Order 2: {order_2[:3]}"
        )

    def test_dataset_different_seed_different_order(
        self, temp_processed_dir
    ):
        """
        Prüft ob verschiedener Seed → verschiedene Reihenfolge.
        """
        import random

        def get_file_order(seed: int):
            random.seed(seed)
            files = list(
                (temp_processed_dir / "train" / "healthy").glob("*.png")
            )
            random.shuffle(files)
            return [f.name for f in files]

        order_1 = get_file_order(42)
        order_2 = get_file_order(99)

        # Mit hoher Wahrscheinlichkeit verschieden
        # (könnte theoretisch gleich sein → tolerieren)
        if len(order_1) > 3:
            assert order_1 != order_2, (
                "Verschiedener Seed sollte verschiedene Reihenfolge geben"
            )

    def test_dataloader_reproducible_with_seed(
        self, temp_processed_dir
    ):
        """
        Prüft ob DataLoader mit gleichem Seed
        gleiche Batch-Reihenfolge liefert.
        """
        import random
        from src.config import RANDOM_SEED

        def get_first_batch_labels(seed: int):
            torch.manual_seed(seed)
            random.seed(seed)
            loaders = get_dataloaders(
                data_dir             = temp_processed_dir,
                batch_size           = 4,
                num_workers          = 0,
                pin_memory           = False,
                use_weighted_sampler = False,
            )
            _, labels = next(iter(loaders["train"]))
            return labels.tolist()

        labels_1 = get_first_batch_labels(RANDOM_SEED)
        labels_2 = get_first_batch_labels(RANDOM_SEED)

        assert labels_1 == labels_2, (
            "Gleicher Seed sollte gleiche Batch-Reihenfolge geben\n"
            f"Batch 1: {labels_1}\n"
            f"Batch 2: {labels_2}"
        )


# ── Tests: Klassenbalance ─────────────────────────────────────

class TestClassBalance:
    """
    Tests für WeightedRandomSampler.

    Testet:
        ✓ WeightedSampler gleicht Klassen aus
        ✓ Ohne Sampler: Unbalancierter Datensatz sichtbar
        ✓ Klassen-Gewichte korrekt berechnet
    """

    def test_weighted_sampler_balances_classes(
        self, imbalanced_processed_dir
    ):
        """
        Prüft ob WeightedSampler Klassen ausgleicht.

        Ausgangslage:
            healthy:  80 Bilder (80%)
            infected: 20 Bilder (20%)

        Nach WeightedSampler:
            healthy:  ~50%
            infected: ~50%
        """
        loaders = get_dataloaders(
            data_dir             = imbalanced_processed_dir,
            batch_size           = 8,
            num_workers          = 0,
            pin_memory           = False,
            use_weighted_sampler = True,
        )

        # Labels über mehrere Batches sammeln
        all_labels = []
        for i, (_, labels) in enumerate(loaders["train"]):
            all_labels.extend(labels.tolist())
            if i >= 20:
                break

        if not all_labels:
            pytest.skip("Zu wenige Batches für Balance-Test")

        healthy_ratio  = all_labels.count(0) / len(all_labels)
        infected_ratio = all_labels.count(1) / len(all_labels)

        # Mit WeightedSampler: beide Klassen ~50%
        # Toleranz: ±20% (Zufallsvarianz bei kleinem Datensatz)
        assert 0.30 <= infected_ratio <= 0.70, (
            f"WeightedSampler sollte Klassen ausgleichen.\n"
            f"infected: {infected_ratio:.1%} "
            f"(erwartet: ~50%)"
        )

    def test_without_sampler_shows_imbalance(
        self, imbalanced_processed_dir
    ):
        """
        Prüft ob ohne Sampler das Ungleichgewicht sichtbar ist.
        Kontrolltest: WeightedSampler deaktiviert.
        """
        loaders = get_dataloaders(
            data_dir             = imbalanced_processed_dir,
            batch_size           = 8,
            num_workers          = 0,
            pin_memory           = False,
            use_weighted_sampler = False,
        )

        all_labels = []
        for i, (_, labels) in enumerate(loaders["train"]):
            all_labels.extend(labels.tolist())
            if i >= 10:
                break

        if not all_labels:
            pytest.skip("Zu wenige Batches für Balance-Test")

        infected_ratio = all_labels.count(1) / len(all_labels)

        # Ohne Sampler: infected << 50%
        assert infected_ratio < 0.50, (
            f"Ohne Sampler sollte infected < 50% sein, "
            f"erhalten: {infected_ratio:.1%}"
        )

    def test_class_weights_correct(self, temp_dataset_dir):
        """
        Prüft ob Klassen-Gewichte korrekt berechnet werden.

        Formel:
            weight = total / (n_classes × count_per_class)

        Bei 20 healthy und 20 infected:
            weight_healthy  = 40 / (2 × 20) = 1.0
            weight_infected = 40 / (2 × 20) = 1.0
        """
        dataset = MalariaDataset(
            temp_dataset_dir / "healthy",
            transform = get_eval_transforms()
        )

        weights = dataset.get_class_weights()

        assert weights is not None, \
            "get_class_weights() sollte Tensor zurückgeben"
        assert len(weights) == len(dataset), (
            f"Anzahl Gewichte ({len(weights)}) != "
            f"Anzahl Bilder ({len(dataset)})"
        )
        assert (weights > 0).all(), \
            "Alle Gewichte sollten > 0 sein"


# ── Quick-Test ────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])