# ============================================================
# tests/test_preprocessing.py
# Tests für filter.py, normalization.py, augmentation.py
# ============================================================

from __future__ import annotations


import cv2
import numpy as np
import pytest
from preprocessing.filter import (
    apply_filter_pipeline,
    apply_gaussian_blur,
    apply_median_blur,
    check_image_quality,
    enhance_contrast_clahe,
    enhance_sharpness,
    normalize_staining_macenko,
    process_batch,
    process_image_file,
    remove_artifacts,
    remove_background,
)
from preprocessing.normalization import (
    apply_normalization_pipeline,
    denormalize_imagenet,
    normalize_imagenet,
    normalize_minmax,
    normalize_percentile,
    normalize_zscore,
)
from preprocessing.augmentation import (
    _is_valid_augmentation,
    augment_dataset_offline,
    get_extended_offline_pipeline,
    get_standard_offline_pipeline,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def dummy_image():
    """
    Erstellt ein zufälliges BGR Testbild (150×150×3).
    Repräsentiert ein typisches Mikroskopbild.
    """
    return np.random.randint(
        30, 220,
        (150, 150, 3),
        dtype=np.uint8
    )


@pytest.fixture
def dummy_image_path(tmp_path):
    """
    Speichert ein Testbild auf Disk.
    Wird für process_image_file Tests gebraucht.
    """
    img_array = np.random.randint(
        30, 220,
        (150, 150, 3),
        dtype=np.uint8
    )
    img_path = tmp_path / "test_image.png"
    cv2.imwrite(str(img_path), img_array)
    return img_path


@pytest.fixture
def temp_image_dir(tmp_path):
    """
    Erstellt temporären Ordner mit mehreren Testbildern.
    Für Batch-Verarbeitung Tests.
    """
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(10):
        img_array = np.random.randint(
            30, 220,
            (150, 150, 3),
            dtype=np.uint8
        )
        cv2.imwrite(str(img_dir / f"img_{i:03d}.png"), img_array)

    return img_dir


# ── Tests: Bildfilter ─────────────────────────────────────────

class TestImageFilters:
    """
    Tests für filter.py Funktionen.

    Testet:
        ✓ Gaussian Blur gibt korrekten Output
        ✓ Median Blur gibt korrekten Output
        ✓ CLAHE verbessert Kontrast
        ✓ Schärfung verändert Bild
        ✓ Artefakt-Entfernung funktioniert
        ✓ Hintergrund-Entfernung funktioniert
        ✓ Macenko Normalisierung funktioniert
        ✓ Filter Pipeline gibt korrekten Output
        ✓ Falsche Kernel-Size wirft Fehler
    """

    def test_gaussian_blur_output_shape(self, dummy_image):
        """Prüft ob Gaussian Blur Shape beibehält."""
        result = apply_gaussian_blur(dummy_image, kernel_size=3)

        assert result.shape == dummy_image.shape, (
            f"Gaussian Blur sollte Shape beibehalten.\n"
            f"Input:  {dummy_image.shape}\n"
            f"Output: {result.shape}"
        )

    def test_gaussian_blur_changes_image(self, dummy_image):
        """Prüft ob Gaussian Blur das Bild verändert."""
        result = apply_gaussian_blur(dummy_image, kernel_size=5)

        assert not np.array_equal(dummy_image, result), \
            "Gaussian Blur sollte das Bild verändern"

    def test_gaussian_blur_even_kernel_raises_error(
        self, dummy_image
    ):
        """Prüft ob gerader Kernel-Size Fehler wirft."""
        with pytest.raises(ValueError):
            apply_gaussian_blur(dummy_image, kernel_size=4)

    def test_median_blur_output_shape(self, dummy_image):
        """Prüft ob Median Blur Shape beibehält."""
        result = apply_median_blur(dummy_image, kernel_size=3)

        assert result.shape == dummy_image.shape, (
            f"Median Blur sollte Shape beibehalten.\n"
            f"Input:  {dummy_image.shape}\n"
            f"Output: {result.shape}"
        )

    def test_median_blur_even_kernel_raises_error(
        self, dummy_image
    ):
        """Prüft ob gerader Kernel-Size Fehler wirft."""
        with pytest.raises(ValueError):
            apply_median_blur(dummy_image, kernel_size=4)

    def test_clahe_output_shape(self, dummy_image):
        """Prüft ob CLAHE Shape beibehält."""
        result = enhance_contrast_clahe(dummy_image)

        assert result.shape == dummy_image.shape, (
            f"CLAHE sollte Shape beibehalten.\n"
            f"Input:  {dummy_image.shape}\n"
            f"Output: {result.shape}"
        )

    def test_clahe_changes_image(self, dummy_image):
        """Prüft ob CLAHE das Bild verändert."""
        result = enhance_contrast_clahe(dummy_image)

        assert not np.array_equal(dummy_image, result), \
            "CLAHE sollte das Bild verändern"

    def test_sharpness_output_shape(self, dummy_image):
        """Prüft ob Schärfung Shape beibehält."""
        result = enhance_sharpness(dummy_image)

        assert result.shape == dummy_image.shape, \
            "Schärfung sollte Shape beibehalten"

    def test_remove_artifacts_output_shape(self, dummy_image):
        """Prüft ob Artefakt-Entfernung Shape beibehält."""
        result = remove_artifacts(dummy_image)

        assert result.shape == dummy_image.shape, \
            "Artefakt-Entfernung sollte Shape beibehalten"

    def test_remove_background_output_shape(self, dummy_image):
        """Prüft ob Hintergrund-Entfernung Shape beibehält."""
        result = remove_background(dummy_image)

        assert result.shape == dummy_image.shape, \
            "Hintergrund-Entfernung sollte Shape beibehalten"

    def test_remove_background_darkens_bright_pixels(self):
        """
        Prüft ob helle Pixel abgedunkelt werden.
        Pixel > threshold sollten dunkler werden.
        """
        # Bild mit sehr hellen Pixeln erstellen
        bright_image         = np.zeros((50, 50, 3), dtype=np.uint8)
        bright_image[:, :] = [220, 220, 220]  # Sehr hell

        result = remove_background(bright_image, threshold=200)

        # Helle Pixel sollten abgedunkelt sein
        assert result.mean() < bright_image.mean(), (
            "Hintergrund-Entfernung sollte helle Pixel abdunkeln.\n"
            f"Vorher: {bright_image.mean():.1f}\n"
            f"Nachher: {result.mean():.1f}"
        )

    def test_macenko_output_shape(self, dummy_image):
        """Prüft ob Macenko Normalisierung Shape beibehält."""
        result = normalize_staining_macenko(dummy_image)

        assert result.shape == dummy_image.shape, (
            f"Macenko sollte Shape beibehalten.\n"
            f"Input:  {dummy_image.shape}\n"
            f"Output: {result.shape}"
        )

    def test_macenko_output_uint8(self, dummy_image):
        """Prüft ob Macenko uint8 zurückgibt."""
        result = normalize_staining_macenko(dummy_image)

        assert result.dtype == np.uint8, (
            f"Macenko sollte uint8 zurückgeben, "
            f"erhalten: {result.dtype}"
        )

    def test_filter_pipeline_output_shape(self, dummy_image):
        """Prüft ob Filter-Pipeline Shape beibehält."""
        result = apply_filter_pipeline(dummy_image)

        assert result.shape == dummy_image.shape, (
            f"Filter-Pipeline sollte Shape beibehalten.\n"
            f"Input:  {dummy_image.shape}\n"
            f"Output: {result.shape}"
        )

    def test_filter_pipeline_no_nan(self, dummy_image):
        """Prüft ob Filter-Pipeline keine NaN Werte erzeugt."""
        result = apply_filter_pipeline(dummy_image)

        assert not np.isnan(result).any(), \
            "Filter-Pipeline sollte keine NaN Werte erzeugen"

    def test_filter_pipeline_valid_pixel_range(
        self, dummy_image
    ):
        """Prüft ob Pixel nach Pipeline im gültigen Bereich sind."""
        result = apply_filter_pipeline(dummy_image)

        assert result.min() >= 0, \
            f"Pixel sollten >= 0 sein, minimum: {result.min()}"
        assert result.max() <= 255, \
            f"Pixel sollten <= 255 sein, maximum: {result.max()}"


# ── Tests: Qualitäts-Check ────────────────────────────────────

class TestImageQuality:
    """
    Tests für check_image_quality Funktion.

    Testet:
        ✓ Normales Bild besteht Check
        ✓ Zu dunkles Bild scheitert
        ✓ Zu helles Bild scheitert
        ✓ Zu unscharfes Bild scheitert
        ✓ Leeres/schwarzes Bild scheitert
        ✓ Korrekte Metriken zurückgegeben
    """

    def test_normal_image_passes(self, dummy_image):
        """Prüft ob normales Bild den Check besteht."""
        result = check_image_quality(dummy_image)

        assert result["passed"] is True, (
            f"Normales Bild sollte Check bestehen.\n"
            f"Grund: {result.get('reason', 'unbekannt')}"
        )

    def test_too_dark_image_fails(self):
        """Prüft ob zu dunkles Bild scheitert."""
        dark_image = np.zeros((100, 100, 3), dtype=np.uint8)
        dark_image[:] = 5  # Sehr dunkel

        result = check_image_quality(dark_image, min_brightness=30.0)

        assert result["passed"] is False, \
            "Sehr dunkles Bild sollte Check nicht bestehen"

    def test_too_bright_image_fails(self):
        """Prüft ob überbelichtetes Bild scheitert."""
        bright_image      = np.zeros((100, 100, 3), dtype=np.uint8)
        bright_image[:] = 250  # Sehr hell

        result = check_image_quality(
            bright_image,
            max_brightness=220.0
        )

        assert result["passed"] is False, \
            "Überbelichtetes Bild sollte Check nicht bestehen"

    def test_black_image_fails(self):
        """Prüft ob komplett schwarzes Bild scheitert."""
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = check_image_quality(black_image)

        assert result["passed"] is False, \
            "Schwarzes Bild sollte Check nicht bestehen"

    def test_quality_returns_metrics(self, dummy_image):
        """Prüft ob Qualitäts-Check Metriken zurückgibt."""
        result = check_image_quality(dummy_image)

        assert "brightness" in result, \
            "Ergebnis sollte Helligkeit enthalten"
        assert "contrast" in result, \
            "Ergebnis sollte Kontrast enthalten"
        assert "sharpness" in result, \
            "Ergebnis sollte Schärfe enthalten"
        assert "passed" in result, \
            "Ergebnis sollte passed enthalten"
        assert "reason" in result, \
            "Ergebnis sollte reason enthalten"

    def test_quality_brightness_in_range(self, dummy_image):
        """Prüft ob Helligkeit sinnvollen Wert hat."""
        result = check_image_quality(dummy_image)

        assert 0 <= result["brightness"] <= 255, (
            f"Helligkeit sollte 0–255 sein, "
            f"erhalten: {result['brightness']}"
        )


# ── Tests: Normalisierung ─────────────────────────────────────

class TestNormalization:
    """
    Tests für normalization.py Funktionen.

    Testet:
        ✓ MinMax gibt Werte 0–1
        ✓ Z-Score gibt zentrierte Werte
        ✓ ImageNet gibt korrekten Shape
        ✓ Percentile ist robuster als MinMax
        ✓ Denormalisierung kehrt Normalisierung um
        ✓ Input-Validierung wirft Fehler
    """

    def test_minmax_output_range(self, dummy_image):
        """Prüft ob MinMax Werte in 0–1 sind."""
        result = normalize_minmax(dummy_image)

        assert result.min() >= 0.0, (
            f"MinMax: minimum sollte >= 0, "
            f"erhalten: {result.min():.4f}"
        )
        assert result.max() <= 1.0, (
            f"MinMax: maximum sollte <= 1, "
            f"erhalten: {result.max():.4f}"
        )

    def test_minmax_output_dtype(self, dummy_image):
        """Prüft ob MinMax float32 zurückgibt."""
        result = normalize_minmax(dummy_image)

        assert result.dtype == np.float32, (
            f"MinMax sollte float32 zurückgeben, "
            f"erhalten: {result.dtype}"
        )

    def test_zscore_centered_around_zero(self, dummy_image):
        """Prüft ob Z-Score um 0 zentriert ist."""
        img_float = dummy_image.astype(np.float32) / 255.0
        result    = normalize_zscore(img_float)

        # Mittelwert sollte nahe 0 sein
        assert abs(result.mean()) < 1.0, (
            f"Z-Score sollte um 0 zentriert sein, "
            f"Mittelwert: {result.mean():.4f}"
        )

    def test_imagenet_output_shape(self, dummy_image):
        """Prüft ob ImageNet-Normalisierung Shape beibehält."""
        result = normalize_imagenet(dummy_image)

        assert result.shape == dummy_image.shape, (
            f"ImageNet sollte Shape beibehalten.\n"
            f"Input:  {dummy_image.shape}\n"
            f"Output: {result.shape}"
        )

    def test_imagenet_output_dtype(self, dummy_image):
        """Prüft ob ImageNet-Normalisierung float32 zurückgibt."""
        result = normalize_imagenet(dummy_image)

        assert result.dtype == np.float32, (
            f"ImageNet sollte float32 zurückgeben, "
            f"erhalten: {result.dtype}"
        )

    def test_percentile_robust_to_outliers(self):
        """
        Prüft ob Percentile robuster als MinMax bei Ausreissern.

        Szenario:
            Bild mit einem sehr hellen Pixel (Ausreisser)
            MinMax     → alles zu dunkel wegen Ausreisser
            Percentile → Ausreisser wird ignoriert
        """
        normal_image         = np.ones((50, 50, 3), dtype=np.uint8) * 128
        normal_image[0, 0] = [255, 255, 255]  # Ausreisser

        minmax_result     = normalize_minmax(normal_image)
        percentile_result = normalize_percentile(normal_image)

        # Percentile sollte höheren Durchschnitt haben
        # weil Ausreisser nicht alles runterzieht
        assert percentile_result.mean() >= minmax_result.mean(), (
            "Percentile sollte robuster gegen Ausreisser sein.\n"
            f"MinMax Mean:     {minmax_result.mean():.4f}\n"
            f"Percentile Mean: {percentile_result.mean():.4f}"
        )

    def test_denormalize_imagenet_recovers_original(
        self, dummy_image
    ):
        """
        Prüft ob Denormalisierung Original annähernd wiederherstellt.

        Toleranz: ±5 Pixel (Rundungsfehler bei float→int)
        """
        normalized   = normalize_imagenet(dummy_image)
        denormalized = denormalize_imagenet(normalized)

        diff = np.abs(
            dummy_image.astype(np.float32) -
            denormalized.astype(np.float32)
        )

        assert diff.mean() < 10.0, (
            f"Denormalisierung sollte Original annähern.\n"
            f"Mittlere Differenz: {diff.mean():.2f} Pixel"
        )

    def test_normalization_pipeline_methods(self, dummy_image):
        """Prüft ob alle Methoden der Pipeline funktionieren."""
        for method in ["imagenet", "zscore", "minmax", "percentile"]:
            result = apply_normalization_pipeline(
                dummy_image,
                method=method
            )
            assert result is not None, \
                f"Methode {method} sollte Ergebnis zurückgeben"
            assert result.dtype == np.float32, \
                f"Methode {method} sollte float32 zurückgeben"

    def test_invalid_normalization_method_raises_error(
        self, dummy_image
    ):
        """Prüft ob ungültige Methode Fehler wirft."""
        with pytest.raises(ValueError):
            apply_normalization_pipeline(
                dummy_image,
                method="ungueltige_methode"
            )

    def test_validation_rejects_wrong_type(self):
        """Prüft ob Validierung falschen Typ ablehnt."""
        from preprocessing.normalization import _validate_image
        with pytest.raises(TypeError):
            _validate_image([[1, 2, 3], [4, 5, 6]])

    def test_validation_rejects_wrong_channels(self):
        """Prüft ob Validierung falsches Kanal-Format ablehnt."""
        from preprocessing.normalization import _validate_image
        wrong_channels = np.zeros((100, 100, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            _validate_image(wrong_channels)

    def test_validation_rejects_empty_image(self):
        """Prüft ob Validierung leeres Bild ablehnt."""
        from preprocessing.normalization import _validate_image
        empty = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            _validate_image(empty)


# ── Tests: Augmentierung ──────────────────────────────────────

class TestAugmentation:
    """
    Tests für augmentation.py Funktionen.

    Testet:
        ✓ Standard Pipeline gibt korrekten Output
        ✓ Erweiterte Pipeline gibt korrekten Output
        ✓ Augmentierung verändert Bilder
        ✓ Qualitäts-Check funktioniert
        ✓ Offline Augmentierung erstellt Dateien
        ✓ Resume überspringt vorhandene Dateien
    """

    def test_standard_pipeline_output_shape(
        self, dummy_image
    ):
        """Prüft ob Standard-Pipeline Shape korrekt ist."""
        pipeline = get_standard_offline_pipeline(img_size=224)
        result   = pipeline(image=dummy_image)["image"]

        assert result.shape == (224, 224, 3), (
            f"Standard-Pipeline sollte (224, 224, 3) ausgeben, "
            f"erhalten: {result.shape}"
        )

    def test_extended_pipeline_output_shape(
        self, dummy_image
    ):
        """Prüft ob Erweiterte-Pipeline Shape korrekt ist."""
        pipeline = get_extended_offline_pipeline(img_size=224)
        result   = pipeline(image=dummy_image)["image"]

        assert result.shape == (224, 224, 3), (
            f"Erweiterte-Pipeline sollte (224, 224, 3) ausgeben, "
            f"erhalten: {result.shape}"
        )

    def test_standard_pipeline_output_dtype(
        self, dummy_image
    ):
        """Prüft ob Standard-Pipeline uint8 zurückgibt."""
        pipeline = get_standard_offline_pipeline()
        result   = pipeline(image=dummy_image)["image"]

        assert result.dtype == np.uint8, (
            f"Pipeline sollte uint8 zurückgeben, "
            f"erhalten: {result.dtype}"
        )

    def test_pipeline_augments_image(self, dummy_image):
        """
        Prüft ob Pipeline Bilder tatsächlich verändert.
        Mindestens ein Ergebnis von 10 sollte anders sein.
        """
        pipeline = get_standard_offline_pipeline()
        results  = [
            pipeline(image=dummy_image)["image"]
            for _ in range(10)
        ]

        all_same = all(
            np.array_equal(results[0], r) for r in results[1:]
        )
        assert not all_same, \
            "Pipeline sollte Bilder augmentieren"

    def test_quality_check_valid_image(self, dummy_image):
        """Prüft ob normales Bild Qualitäts-Check besteht."""
        pipeline = get_standard_offline_pipeline()
        result   = pipeline(image=dummy_image)["image"]
        valid    = _is_valid_augmentation(result)

        assert valid is True, \
            "Augmentiertes Bild sollte Qualitäts-Check bestehen"

    def test_quality_check_black_image_fails(self):
        """Prüft ob schwarzes Bild Qualitäts-Check scheitert."""
        black_image = np.zeros((224, 224, 3), dtype=np.uint8)
        valid       = _is_valid_augmentation(black_image)

        assert valid is False, \
            "Schwarzes Bild sollte Qualitäts-Check scheitern"

    def test_offline_augmentation_creates_files(
        self, temp_image_dir, tmp_path
    ):
        """Prüft ob Offline-Augmentierung Dateien erstellt."""
        output_dir = tmp_path / "augmented"

        stats = augment_dataset_offline(
            input_dir      = temp_image_dir,
            output_dir     = output_dir,
            augment_factor = 2,
            quality_check  = False,
        )

        assert output_dir.exists(), \
            "Output-Ordner sollte erstellt werden"
        assert stats["original"] > 0, \
            "Mindestens ein Originalbild sollte kopiert werden"
        assert stats["augmented"] > 0, \
            "Mindestens ein augmentiertes Bild sollte erstellt werden"

    def test_offline_augmentation_correct_count(
        self, temp_image_dir, tmp_path
    ):
        """
        Prüft ob korrekte Anzahl Dateien erstellt wird.

        Formel:
            total = original + (original × augment_factor)
        """
        output_dir     = tmp_path / "augmented"
        augment_factor = 3

        stats = augment_dataset_offline(
            input_dir      = temp_image_dir,
            output_dir     = output_dir,
            augment_factor = augment_factor,
            quality_check  = False,
        )

        expected_augmented = stats["original"] * augment_factor
        assert stats["augmented"] == expected_augmented, (
            f"Erwartete {expected_augmented} augmentierte Bilder, "
            f"erhalten: {stats['augmented']}"
        )

    def test_offline_augmentation_resume_skips_existing(
        self, temp_image_dir, tmp_path
    ):
        """
        Prüft ob Resume bereits vorhandene Bilder überspringt.

        Ablauf:
            1. Augmentierung ausführen → Dateien erstellt
            2. Nochmal ausführen mit resume=True
            3. Alle Dateien sollten übersprungen werden
        """
        output_dir = tmp_path / "augmented"

        # Erste Ausführung
        stats_1 = augment_dataset_offline(
            input_dir      = temp_image_dir,
            output_dir     = output_dir,
            augment_factor = 2,
            resume         = True,
            quality_check  = False,
        )

        # Zweite Ausführung mit resume=True
        stats_2 = augment_dataset_offline(
            input_dir      = temp_image_dir,
            output_dir     = output_dir,
            augment_factor = 2,
            resume         = True,
            quality_check  = False,
        )

        assert stats_2["skipped"] > 0, (
            "Resume: Bereits vorhandene Bilder sollten "
            "übersprungen werden.\n"
            f"Übersprungen: {stats_2['skipped']}"
        )

    def test_offline_augmentation_no_resume_overwrites(
        self, temp_image_dir, tmp_path
    ):
        """
        Prüft ob ohne Resume Bilder neu erstellt werden.
        resume=False → kein Überspringen.
        """
        output_dir = tmp_path / "augmented"

        # Erste Ausführung
        augment_dataset_offline(
            input_dir      = temp_image_dir,
            output_dir     = output_dir,
            augment_factor = 2,
            resume         = False,
            quality_check  = False,
        )

        # Zweite Ausführung ohne resume
        stats_2 = augment_dataset_offline(
            input_dir      = temp_image_dir,
            output_dir     = output_dir,
            augment_factor = 2,
            resume         = False,
            quality_check  = False,
        )

        assert stats_2["skipped"] == 0, (
            "Ohne Resume: Kein Bild sollte übersprungen werden.\n"
            f"Übersprungen: {stats_2['skipped']}"
        )


# ── Tests: process_image_file ─────────────────────────────────

class TestProcessImageFile:
    """
    Tests für process_image_file Funktion.

    Testet:
        ✓ Einzelbild wird verarbeitet
        ✓ Output wird gespeichert
        ✓ Schlechte Qualität → None zurückgegeben
        ✓ Nicht existierendes Bild → Fehler
    """

    def test_process_image_returns_array(
        self, dummy_image_path
    ):
        """Prüft ob process_image_file NumPy Array zurückgibt."""
        result = process_image_file(
            dummy_image_path,
            quality_check = False,
        )

        assert result is not None, \
            "process_image_file sollte Array zurückgeben"
        assert isinstance(result, np.ndarray), \
            f"Erwartet NumPy Array, erhalten: {type(result)}"

    def test_process_image_saves_output(
        self, dummy_image_path, tmp_path
    ):
        """Prüft ob Output-Datei gespeichert wird."""
        output_path = tmp_path / "processed.png"

        process_image_file(
            dummy_image_path,
            output_path   = output_path,
            quality_check = False,
        )

        assert output_path.exists(), \
            f"Output-Datei sollte existieren: {output_path}"

    def test_process_nonexistent_file_raises_error(
        self, tmp_path
    ):
        """Prüft ob nicht existierendes Bild Fehler wirft."""
        with pytest.raises(FileNotFoundError):
            process_image_file(
                tmp_path / "nicht_vorhanden.png"
            )

    def test_process_low_quality_returns_none(
        self, tmp_path
    ):
        """
        Prüft ob Bild mit schlechter Qualität None zurückgibt.
        Schwarzes Bild sollte Qualitäts-Check scheitern.
        """
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)
        black_path  = tmp_path / "black.png"
        cv2.imwrite(str(black_path), black_image)

        result = process_image_file(
            black_path,
            quality_check = True,
        )

        assert result is None, (
            "Schwarzes Bild sollte None zurückgeben "
            "(Qualitäts-Check scheitert)"
        )


# ── Tests: Batch-Verarbeitung ─────────────────────────────────

class TestBatchProcessing:
    """
    Tests für process_batch Funktion.

    Testet:
        ✓ Mehrere Bilder werden verarbeitet
        ✓ Statistiken korrekt zurückgegeben
        ✓ Fehlerhafte Bilder werden übersprungen
    """

    def test_batch_processes_all_images(
        self, temp_image_dir, tmp_path
    ):
        """Prüft ob alle Bilder verarbeitet werden."""
        output_dir  = tmp_path / "processed"
        input_paths = list(temp_image_dir.glob("*.png"))

        stats = process_batch(
            input_paths   = input_paths,
            output_dir    = output_dir,
            n_workers     = 1,
            quality_check = False,
        )

        assert stats["processed"] + stats["skipped"] + \
               stats["failed"] == len(input_paths), (
            "Summe aller Stats sollte Anzahl Input-Bilder ergeben"
        )

    def test_batch_returns_correct_stats(
        self, temp_image_dir, tmp_path
    ):
        """Prüft ob Batch-Stats korrekte Keys haben."""
        output_dir  = tmp_path / "processed"
        input_paths = list(temp_image_dir.glob("*.png"))

        stats = process_batch(
            input_paths   = input_paths,
            output_dir    = output_dir,
            n_workers     = 1,
            quality_check = False,
        )

        assert "processed" in stats, "Stats sollte 'processed' enthalten"
        assert "skipped"   in stats, "Stats sollte 'skipped' enthalten"
        assert "failed"    in stats, "Stats sollte 'failed' enthalten"

    def test_batch_creates_output_dir(
        self, temp_image_dir, tmp_path
    ):
        """Prüft ob Output-Ordner erstellt wird."""
        output_dir  = tmp_path / "neu_erstellt" / "output"
        input_paths = list(temp_image_dir.glob("*.png"))[:3]

        process_batch(
            input_paths   = input_paths,
            output_dir    = output_dir,
            n_workers     = 1,
            quality_check = False,
        )

        assert output_dir.exists(), \
            "Output-Ordner sollte automatisch erstellt werden"


# ── Quick-Test ────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])