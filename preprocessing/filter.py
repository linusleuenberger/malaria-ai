#Funktion Filter    Was sie macht
#cv2.GaussianBlur   Weichzeichnen
#cv2.medianBlur     Median Filter
#cv2.cvtColor       Farbraum wechseln
#cv2.createCLAHE    Lokaler Kontrast
#cv2.morphologyEx   Artefakte entfernen
#cv2.equalizeHist   Histogramm ausgleichen

# ============================================================
# preprocessing/filter.py
# Bildfilter für Mikroskopische Blutbilder
# ============================================================

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# ── 1. Rauschen entfernen ─────────────────────────────────────
def apply_gaussian_blur(
    image:       np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Gaussian Blur – weichzeichnen um Rauschen zu reduzieren.

    Wie es funktioniert:
        Jeden Pixel durch gewichteten Durchschnitt
        seiner Nachbarpixel ersetzen.
        Pixel in der Mitte → höheres Gewicht
        Pixel am Rand      → niedrigeres Gewicht

        Kernel 3x3 Beispiel:
        [1, 2, 1]
        [2, 4, 2]  ÷ 16  → gewichteter Durchschnitt
        [1, 2, 1]

    Args:
        image       : Bild als NumPy Array (H, W, 3)
        kernel_size : Grösse des Weichzeichners
                      3 = leicht, 5 = mittel, 7 = stark
                      Muss immer ungerade sein!

    Returns:
        Geglättetes Bild
    """
    if kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size muss ungerade sein, nicht {kernel_size}"
        )
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_median_blur(
    image:       np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Median Blur – entfernt Salt-and-Pepper Rauschen.

    Unterschied zu Gaussian Blur:
        Gaussian → gewichteter Durchschnitt → weich
        Median   → mittlerer Wert           → erhält Kanten besser

    Args:
        image       : Bild als NumPy Array
        kernel_size : Muss ungerade sein

    Returns:
        Gefiltertes Bild
    """
    if kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size muss ungerade sein, nicht {kernel_size}"
        )
    return cv2.medianBlur(image, kernel_size)


# ── 2. Kontrast verbessern ────────────────────────────────────
def enhance_contrast_clahe(
    image:      np.ndarray,
    clip_limit: float          = 2.0,
    tile_size:  Tuple[int,int] = (8, 8),
) -> np.ndarray:
    """
    CLAHE – Contrast Limited Adaptive Histogram Equalization.

    Was ist CLAHE:
        Teilt Bild in kleine Kacheln (8x8) auf und
        verbessert Kontrast lokal in jeder Kachel.
        Viel besser als globales Histogram Equalization
        für Mikroskopbilder mit ungleicher Belichtung.

    Args:
        image      : BGR Bild als NumPy Array
        clip_limit : Maximale Kontrastverstärkung
        tile_size  : Grösse der lokalen Kacheln

    Returns:
        Kontrastverstärktes Bild
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe      = cv2.createCLAHE(clipLimit=clip_limit,
                                  tileGridSize=tile_size)
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def enhance_sharpness(
    image:    np.ndarray,
    strength: float = 1.5,
) -> np.ndarray:
    """
    Schärfe verbessern – Unsharp Masking.

    Formel: scharf = original + strength × (original - unscharf)

    Args:
        image    : Bild als NumPy Array
        strength : Schärfestärke (1.0 = leicht, 2.0 = stark)

    Returns:
        Geschärftes Bild
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)


# ── 3. Artefakte entfernen ────────────────────────────────────
def remove_artifacts(
    image:       np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Morphologisches Opening – kleine Artefakte entfernen.

    Opening = Erosion gefolgt von Dilation:
        Erosion:  kleine Artefakte verschwinden
        Dilation: grössere Strukturen bleiben erhalten

    Args:
        image       : Bild als NumPy Array
        kernel_size : 3 = kleine, 5 = grössere Artefakte

    Returns:
        Bild ohne kleine Artefakte
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def remove_background(
    image:     np.ndarray,
    threshold: int = 200,
) -> np.ndarray:
    """
    Hintergrund abdunkeln – Fokus auf Zellen legen.

    Pixel heller als threshold → abgedunkelt
    → Kontrast zwischen Zellen und Hintergrund erhöht

    Args:
        image     : Bild als NumPy Array
        threshold : Helligkeitsschwelle (0–255)

    Returns:
        Bild mit abgedunkeltem Hintergrund
    """
    gray            = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background_mask = gray > threshold
    result          = image.copy()
    result[background_mask] = (
        result[background_mask] * 0.5
    ).astype(np.uint8)
    return result


# ── 4. Stain Normalisierung ───────────────────────────────────
def normalize_staining_macenko(
    image:      np.ndarray,
    beta:       float = 0.15,
    alpha:      float = 1.0,
    light_intensity: int = 240,
) -> np.ndarray:
    """
    Macenko Stain Normalization – Industriestandard für Blutbilder.

    Problem:
        Verschiedene Labore → unterschiedliche Färbung
        Labor A: blau-violett   Labor B: rosa-rot
        → KI lernt Farbe statt Struktur

    Macenko Methode:
        1. Optische Dichte berechnen (wie stark Licht absorbiert)
        2. SVD (Singular Value Decomposition) auf OD-Matrix
        3. Stain-Vektoren extrahieren
        4. Auf Referenz-Stain normalisieren
        → Alle Bilder sehen farblich gleich aus

    Warum Macenko:
        → Peer-reviewed, publiziert 2009
        → Standard in medizinischer Bildverarbeitung
        → Robuster als einfache HSV Normalisierung

    Args:
        image           : BGR Bild als NumPy Array
        beta            : OD Schwellenwert (Standard: 0.15)
        alpha           : Percentile für Robustheit (Standard: 1.0)
        light_intensity : Maximale Lichtintensität (Standard: 240)

    Returns:
        Farblich normalisiertes Bild
    """
    # ── Schritt 1: BGR → RGB → optische Dichte ────────────
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64)

    # Schwarze Pixel vermeiden (Division durch 0)
    image_rgb = np.clip(image_rgb, 1, 255)

    # Optische Dichte: OD = -log(I / I_max)
    # Hohe OD = viel Farbstoff absorbiert (dunkle Stelle)
    OD = -np.log(image_rgb / light_intensity)

    # ── Schritt 2: Schwache Pixel entfernen ───────────────
    # Nur Pixel nehmen die genug Farbstoff absorbieren
    OD_flat   = OD.reshape(-1, 3)
    OD_thresh = OD_flat[np.all(OD_flat > beta, axis=1)]

    if OD_thresh.shape[0] == 0:
        logger.warning("Macenko: Keine Pixel über Schwellenwert – "
                       "Original zurückgegeben.")
        return image

    # ── Schritt 3: SVD – Stain Vektoren finden ────────────
    # SVD findet die Hauptrichtungen der Farbvariation
    _, _, V = np.linalg.svd(OD_thresh, full_matrices=False)

    # Erste zwei Singularvektoren = die zwei Hauptfarben
    # (Hämatoxylin und Eosin bei H&E, oder Giemsa-Komponenten)
    plane = V[:2, :]

    # ── Schritt 4: Winkel im Farbraum berechnen ───────────
    # Projektion auf die Farbebene
    proj   = OD_thresh @ plane.T
    angles = np.arctan2(proj[:, 1], proj[:, 0])

    # Extremwerte = die zwei Stain-Richtungen
    phi_min = np.percentile(angles, alpha)
    phi_max = np.percentile(angles, 100 - alpha)

    # Stain-Vektoren aus Winkeln berechnen
    v1 = plane.T @ np.array([np.cos(phi_min), np.sin(phi_min)])
    v2 = plane.T @ np.array([np.cos(phi_max), np.sin(phi_max)])

    # Positiv ausrichten
    if v1[0] < 0: v1 *= -1
    if v2[0] < 0: v2 *= -1

    # Stain Matrix: Zeilen = Stain-Vektoren
    # Stain mit höherem Rotanteil zuerst (Hämatoxylin/Hauptfarbstoff)
    if v1[0] > v2[0]:
        stain_matrix = np.array([v1, v2])
    else:
        stain_matrix = np.array([v2, v1])

    # ── Schritt 5: Referenz Stain Matrix ─────────────────
    # Typische Giemsa-Referenzwerte
    # (aus publizierten Referenz-Blutbildern berechnet)
    stain_ref = np.array([
        [0.5626, 0.7201, 0.4062],
        [0.2159, 0.8012, 0.5581],
    ])

    # ── Schritt 6: Konzentrationen berechnen ─────────────
    # Wie viel von jedem Farbstoff ist in jedem Pixel?
    # OD = stain_matrix.T @ concentrations
    # → concentrations = (stain_matrix.T)^(-1) @ OD
    try:
        concentrations = np.linalg.lstsq(
            stain_matrix.T, OD_flat.T, rcond=None
        )[0]
    except np.linalg.LinAlgError:
        logger.warning("Macenko: Lineare Gleichung nicht lösbar – "
                       "Original zurückgegeben.")
        return image

    # ── Schritt 7: Mit Referenz neu berechnen ────────────
    OD_norm = stain_ref.T @ concentrations
    OD_norm = OD_norm.T.reshape(OD.shape)

    # ── Schritt 8: Zurück zu RGB ──────────────────────────
    image_norm = light_intensity * np.exp(-OD_norm)
    image_norm = np.clip(image_norm, 0, 255).astype(np.uint8)

    return cv2.cvtColor(image_norm, cv2.COLOR_RGB2BGR)


def normalize_staining_simple(
    image: np.ndarray,
) -> np.ndarray:
    """
    Einfache HSV Normalisierung als Fallback.
    Weniger präzise als Macenko aber schneller.

    Args:
        image : BGR Bild als NumPy Array

    Returns:
        Farblich normalisiertes Bild
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_normalized   = cv2.equalizeHist(v)
    hsv_normalized = cv2.merge([h, s, v_normalized])
    return cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)


# ── 5. Qualitäts-Check ────────────────────────────────────────
def check_image_quality(
    image:              np.ndarray,
    min_brightness:     float = 30.0,
    max_brightness:     float = 220.0,
    min_contrast:       float = 20.0,
    min_sharpness:      float = 50.0,
    min_resolution:     Tuple[int, int] = (50, 50),
) -> Dict[str, bool | float | str]:
    """
    Prüft ob Bild gut genug für Training ist.

    Checks:
        ✓ Auflösung ausreichend?
        ✓ Bild nicht zu dunkel?
        ✓ Bild nicht zu hell/überbelichtet?
        ✓ Genug Kontrast vorhanden?
        ✓ Bild scharf genug?
        ✓ Bild nicht leer/schwarz?

    Args:
        image          : Bild als NumPy Array
        min_brightness : Minimale Durchschnittshelligkeit
        max_brightness : Maximale Durchschnittshelligkeit
        min_contrast   : Minimale Standardabweichung der Helligkeit
        min_sharpness  : Minimale Laplacian Varianz (Schärfe)
        min_resolution : Minimale (Höhe, Breite) in Pixel

    Returns:
        Dict mit:
            passed     : True wenn alle Checks bestanden
            reason     : Grund falls Check fehlgeschlagen
            brightness : Gemessene Helligkeit
            contrast   : Gemessener Kontrast
            sharpness  : Gemessene Schärfe
    """
    result: Dict[str, bool | float | str] = {
        "passed"    : True,
        "reason"    : "",
        "brightness": 0.0,
        "contrast"  : 0.0,
        "sharpness" : 0.0,
    }

    # ── Auflösung ──────────────────────────────────────────
    h, w = image.shape[:2]
    if h < min_resolution[0] or w < min_resolution[1]:
        result["passed"] = False
        result["reason"] = (
            f"Auflösung zu klein: {w}x{h} "
            f"(Minimum: {min_resolution[1]}x{min_resolution[0]})"
        )
        return result

    # Graustufenbild für weitere Checks
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # ── Helligkeit ─────────────────────────────────────────
    brightness = float(gray.mean())
    result["brightness"] = round(brightness, 2)

    if brightness < min_brightness:
        result["passed"] = False
        result["reason"] = (
            f"Bild zu dunkel: {brightness:.1f} "
            f"(Minimum: {min_brightness})"
        )
        return result

    if brightness > max_brightness:
        result["passed"] = False
        result["reason"] = (
            f"Bild überbelichtet: {brightness:.1f} "
            f"(Maximum: {max_brightness})"
        )
        return result

    # ── Kontrast ───────────────────────────────────────────
    contrast = float(gray.std())
    result["contrast"] = round(contrast, 2)

    if contrast < min_contrast:
        result["passed"] = False
        result["reason"] = (
            f"Kontrast zu schwach: {contrast:.1f} "
            f"(Minimum: {min_contrast})"
        )
        return result

    # ── Schärfe (Laplacian Varianz) ────────────────────────
    # Laplacian = zweite Ableitung → hohe Varianz = scharf
    # Unscharfe Bilder haben kleine Laplacian Varianz
    sharpness = float(cv2.Laplacian(
        gray.astype(np.uint8), cv2.CV_64F
    ).var())
    result["sharpness"] = round(sharpness, 2)

    if sharpness < min_sharpness:
        result["passed"] = False
        result["reason"] = (
            f"Bild zu unscharf: {sharpness:.1f} "
            f"(Minimum: {min_sharpness})"
        )
        return result

    # ── Leer/Schwarz Check ─────────────────────────────────
    if np.all(gray < 5):
        result["passed"] = False
        result["reason"] = "Bild ist leer oder komplett schwarz"
        return result

    return result


# ── 6. Gesamte Filter-Pipeline ────────────────────────────────
def apply_filter_pipeline(
    image:          np.ndarray,
    use_blur:       bool  = True,
    use_contrast:   bool  = True,
    use_sharpness:  bool  = True,
    use_artifacts:  bool  = True,
    use_background: bool  = False,
    use_stain_norm: bool  = True,
    use_macenko:    bool  = True,
    blur_kernel:    int   = 3,
    contrast_clip:  float = 2.0,
    sharpness:      float = 1.5,
) -> np.ndarray:
    """
    Komplette Vorverarbeitungs-Pipeline auf ein Bild anwenden.

    Reihenfolge ist wichtig:
        1. Rauschen entfernen  → saubereres Bild
        2. Artefakte entfernen → grobe Störungen weg
        3. Kontrast verbessern → Zellen besser sichtbar
        4. Schärfe verbessern  → Details klarer
        5. Hintergrund         → optional
        6. Farbnormalisierung  → optional, bei mehreren Laboren

    Args:
        image          : BGR Bild als NumPy Array
        use_blur       : Gaussian Blur anwenden
        use_contrast   : CLAHE Kontrast anwenden
        use_sharpness  : Schärfe verbessern
        use_artifacts  : Artefakte entfernen
        use_background : Hintergrund abdunkeln
        use_stain_norm : Färbung normalisieren
        use_macenko    : Macenko (True) oder einfache Methode (False)
        blur_kernel    : Kernel für Blur
        contrast_clip  : Clip Limit für CLAHE
        sharpness      : Stärke der Schärfung

    Returns:
        Vorverarbeitetes Bild
    """
    result = image.copy()

    if use_blur:
        result = apply_gaussian_blur(result, blur_kernel)

    if use_artifacts:
        result = remove_artifacts(result)

    if use_contrast:
        result = enhance_contrast_clahe(result, contrast_clip)

    if use_sharpness:
        result = enhance_sharpness(result, sharpness)

    if use_background:
        result = remove_background(result)

    if use_stain_norm:
        if use_macenko:
            result = normalize_staining_macenko(result)
        else:
            result = normalize_staining_simple(result)

    return result


# ── 7. Bild laden und Pipeline anwenden ──────────────────────
def process_image_file(
    input_path:   Path,
    output_path:  Optional[Path] = None,
    quality_check: bool          = True,
    **pipeline_kwargs,
) -> Optional[np.ndarray]:
    """
    Bild laden, Qualität prüfen, Pipeline anwenden, speichern.

    Args:
        input_path    : Pfad zum Originalbild
        output_path   : Speicherpfad (optional)
        quality_check : Qualitäts-Check vor Pipeline
        **pipeline_kwargs : Argumente für apply_filter_pipeline

    Returns:
        Vorverarbeitetes Bild oder None falls Qualität ungenügend
    """
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {input_path}")

    # Qualitäts-Check
    if quality_check:
        quality = check_image_quality(image)
        if not quality["passed"]:
            logger.warning(
                f"Bild übersprungen ({input_path.name}): "
                f"{quality['reason']}"
            )
            return None

    processed = apply_filter_pipeline(image, **pipeline_kwargs)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), processed)

    return processed


# ── 8. Batch-Verarbeitung ─────────────────────────────────────
def process_batch(
    input_paths:  List[Path],
    output_dir:   Path,
    n_workers:    int  = 4,
    quality_check: bool = True,
    **pipeline_kwargs,
) -> Dict[str, int]:
    """
    Mehrere Bilder parallel verarbeiten.

    Mit 4 CPU-Kernen → ~4x schneller als einzeln verarbeiten.
    Zeigt Fortschritt im Terminal.

    Args:
        input_paths   : Liste aller Bildpfade
        output_dir    : Zielordner für verarbeitete Bilder
        n_workers     : Anzahl parallele CPU-Kerne
        quality_check : Bilder auf Qualität prüfen
        **pipeline_kwargs : Argumente für apply_filter_pipeline

    Returns:
        Dict mit:
            processed : Anzahl erfolgreich verarbeitete Bilder
            skipped   : Anzahl übersprungene Bilder (schlechte Qualität)
            failed    : Anzahl fehlgeschlagene Bilder (Fehler)
    """
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    total = len(input_paths)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Hilfsfunktion für parallele Verarbeitung
    def _process_one(input_path: Path) -> str:
        output_path = output_dir / input_path.name
        try:
            result = process_image_file(
                input_path    = input_path,
                output_path   = output_path,
                quality_check = quality_check,
                **pipeline_kwargs,
            )
            return "skipped" if result is None else "processed"
        except Exception as e:
            logger.error(f"Fehler bei {input_path.name}: {e}")
            return "failed"

    # Parallel verarbeiten
    logger.info(
        f"Batch-Verarbeitung: {total} Bilder "
        f"mit {n_workers} Worker(s)"
    )

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_process_one, p): p
            for p in input_paths
        }
        for i, future in enumerate(as_completed(futures), 1):
            status = future.result()
            stats[status] += 1

            # Fortschritt
            if i % 100 == 0 or i == total:
                logger.info(
                    f"  Fortschritt: {i}/{total} | "
                    f"OK: {stats['processed']} | "
                    f"Übersprungen: {stats['skipped']} | "
                    f"Fehler: {stats['failed']}"
                )

    logger.info(
        f"Batch abgeschlossen: "
        f"{stats['processed']} verarbeitet, "
        f"{stats['skipped']} übersprungen, "
        f"{stats['failed']} fehlgeschlagen"
    )
    return stats


# ── 9. Vorher/Nachher Visualisierung ─────────────────────────
def plot_filter_comparison(
    image:     np.ndarray,
    save_path: Path = Path("results/plots/filter_comparison.png"),
) -> None:
    """
    Alle Filter nebeneinander visualisieren.

    Zeigt:
        Original | Blur | CLAHE | Scharf | Artefakte | Pipeline

    Nützlich für:
        → Prüfen ob Filter sinnvoll sind
        → ETH-Präsentation
        → Debugging

    Args:
        image     : BGR Originalbild
        save_path : Speicherpfad
    """
    # Alle Filter anwenden
    filters = {
        "Original"        : image,
        "Gaussian Blur"   : apply_gaussian_blur(image),
        "CLAHE Kontrast"  : enhance_contrast_clahe(image),
        "Schärfung"       : enhance_sharpness(image),
        "Artefakte weg"   : remove_artifacts(image),
        "Macenko Norm."   : normalize_staining_macenko(image),
        "Volle Pipeline"  : apply_filter_pipeline(image),
    }

    n      = len(filters)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 4))

    for ax, (name, img) in zip(axes, filters.items()):
        # BGR → RGB für matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        "Vorher / Nachher – Filtervergleich",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Filtervergleich gespeichert: {save_path}")


# ── Quick-Test: python -m preprocessing.filter ───────────────
if __name__ == "__main__":
    import sys
    from src.config import RAW_DIR, PLOTS_DIR

    test_images = (
        list(RAW_DIR.rglob("*.png")) +
        list(RAW_DIR.rglob("*.jpg"))
    )

    if not test_images:
        logger.error("Keine Testbilder in data/raw/ gefunden.")
        logger.error("→ Zuerst Datensatz herunterladen.")
        sys.exit(1)

    test_image = test_images[0]
    logger.info(f"Teste mit: {test_image}")

    img = cv2.imread(str(test_image))
    logger.info(f"Original Shape: {img.shape}")

    # ── Qualitäts-Check ───────────────────────────────────
    logger.info("Qualitäts-Check:")
    quality = check_image_quality(img)
    tests = {
        "gaussian_blur"      : apply_gaussian_blur(img),
        "median_blur"        : apply_median_blur(img),
        "clahe_contrast"     : enhance_contrast_clahe(img),
        "sharpness"          : enhance_sharpness(img),
        "remove_artifacts"   : remove_artifacts(img),
        "macenko_norm"       : normalize_staining_macenko(img),
        "full_pipeline"      : apply_filter_pipeline(img),
    }

    for name, result in tests.items():
        logger.info(f"  ✓ {name}: {result.shape}")

    # ── Visualisierung testen ──────────────────────────────
    logger.info("Visualisierung:")
    plot_filter_comparison(
        img,
        save_path = PLOTS_DIR / "filter_comparison.png"
    )
    logger.info(f"  ✓ Gespeichert: {PLOTS_DIR / 'filter_comparison.png'}")

    # ── Batch-Test mit 5 Bildern ───────────────────────────
    logger.info("Batch-Test (5 Bilder):")
    stats = process_batch(
        input_paths = test_images[:5],
        output_dir  = RAW_DIR.parent / "processed_test",
        n_workers   = 2,
    )
    logger.info(f"  Verarbeitet : {stats['processed']}")
    logger.info(f"  Übersprungen: {stats['skipped']}")
    logger.info(f"  Fehler      : {stats['failed']}")

    logger.info("✓ filter.py funktioniert korrekt.")