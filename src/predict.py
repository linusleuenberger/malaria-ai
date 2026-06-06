"""
predict.py – Einzelbild-Vorhersage (erweitert)
================================================

Features:
    - Einzelbild & Batch-Vorhersage
    - Grad-CAM Visualisierung
    - Unsicherheitserkennung
    - Ergebnisse als CSV/JSON speichern
    - Gradio Web-Interface

Verwendung:
    python -m src.predict --image path/to/cell.png
    python -m src.predict --image path/to/cell.png --gradcam
    python -m src.predict --folder path/to/folder/
    python -m src.predict --gui
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.dataset import IDX_TO_CLASS, get_eval_transforms

# ──────────────────────────────────────────────────────────────────────────────
# Konstanten
# ──────────────────────────────────────────────────────────────────────────────

UNCERTAINTY_THRESHOLD = 0.70   # unter 70% → "unsicher"
RESULTS_DIR = Path("results/predictions")
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Zeigt welche Bildregionen das Modell für die Entscheidung
    genutzt hat – als Heatmap über das Originalbild gelegt.

    Funktioniert indem Gradienten der Zielklasse bezüglich
    der letzten Conv-Schicht berechnet werden. Regionen mit
    hohen Gradienten sind für die Entscheidung wichtig.

    Args:
        model      : Trainiertes Modell
        target_layer: Letzte Conv-Schicht (z.B. model.features[-1])

    Beispiel:
        >>> cam = GradCAM(model, model.features[-1])
        >>> heatmap = cam(tensor, class_idx=1)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Hooks registrieren
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Berechnet die Grad-CAM Heatmap.

        Args:
            image_tensor : Vorverarbeiteter Tensor [1, 3, H, W]
            class_idx    : Zielklasse (None = vorhergesagte Klasse)

        Returns:
            Heatmap als numpy Array [H, W], Werte in [0, 1]
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass nur für Zielklasse
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Gewichte = globaler Durchschnitt der Gradienten
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Gewichtete Summe der Aktivierungen
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # Auf Bildgrösse hochskalieren und normalisieren
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    @staticmethod
    def overlay(
        original: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> Image.Image:
        """
        Legt die Heatmap über das Originalbild.

        Args:
            original : Originalbild als PIL Image
            heatmap  : Heatmap Array [H, W] in [0, 1]
            alpha    : Transparenz der Heatmap (0=unsichtbar, 1=nur Heatmap)

        Returns:
            Überlagertes Bild als PIL Image
        """
        import matplotlib.pyplot as plt

        # Heatmap einfärben (warm = wichtig, kalt = unwichtig)
        colormap  = plt.get_cmap("jet")
        colored   = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(colored).resize(original.size, Image.BILINEAR)

        # Überlagern
        return Image.blend(original.convert("RGB"), heatmap_img, alpha=alpha)


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Findet automatisch die letzte Conv2d-Schicht im Modell.
    Fallback falls der Layer nicht manuell angegeben wird.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Keine Conv2d-Schicht im Modell gefunden.")
    return last_conv


# ──────────────────────────────────────────────────────────────────────────────
# Modell laden
# ──────────────────────────────────────────────────────────────────────────────

def load_model(
    model_path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """
    Lädt ein gespeichertes Modell von der Festplatte.

    Erkennt automatisch ob der Checkpoint nur Gewichte enthält
    oder auch Metadaten wie Epoche und Validation-Accuracy.
    """
    from src.model import MalariaNet

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"  Checkpoint Epoche : {checkpoint.get('epoch', '?')}")
        print(f"  Val Accuracy      : {checkpoint.get('val_acc', '?'):.4f}")
    else:
        state_dict = checkpoint

    model = MalariaNet()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# ──────────────────────────────────────────────────────────────────────────────
# Bild vorbereiten
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    image_path: str | Path,
    img_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std:  Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Lädt ein Bild und bereitet es für das Modell vor.

    Returns:
        (tensor [1, 3, H, W], original PIL-Bild)
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    original  = Image.open(image_path).convert("RGB")
    transform = get_eval_transforms(img_size, mean, std)
    tensor    = transform(original).unsqueeze(0)  # [1, 3, H, W]

    return tensor, original


# ──────────────────────────────────────────────────────────────────────────────
# Einzelvorhersage
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> Dict:
    """
    Führt die Vorhersage für ein einzelnes Bild durch.

    Gibt zusätzlich eine Unsicherheitswarnung aus wenn
    die Confidence unter UNCERTAINTY_THRESHOLD liegt.

    Returns:
        Dict mit predicted_class, confidence, probs, uncertain, inference_ms
    """
    image_tensor = image_tensor.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(image_tensor)
        probs  = F.softmax(logits, dim=1)
    elapsed_ms = (time.perf_counter() - start) * 1000

    prob_healthy  = probs[0][0].item()
    prob_infected = probs[0][1].item()
    predicted_idx = probs.argmax(dim=1).item()
    confidence    = probs[0][predicted_idx].item()

    # Unsicherheitserkennung
    uncertain = confidence < UNCERTAINTY_THRESHOLD

    return {
        "predicted_class" : IDX_TO_CLASS[predicted_idx],
        "predicted_idx"   : predicted_idx,
        "confidence"      : confidence,
        "prob_healthy"    : prob_healthy,
        "prob_infected"   : prob_infected,
        "uncertain"       : uncertain,
        "inference_ms"    : round(elapsed_ms, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Batch-Inferenz
# ──────────────────────────────────────────────────────────────────────────────

def predict_batch(
    model: torch.nn.Module,
    image_paths: List[Path],
    device: torch.device,
    img_size: int = 224,
    batch_size: int = 32,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std:  Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> List[Dict]:
    """
    Verarbeitet mehrere Bilder effizient in Batches.

    Deutlich schneller als einzelne Vorhersagen auf GPU,
    da der Overhead pro Batch nur einmal anfällt.

    Args:
        model       : Geladenes Modell
        image_paths : Liste von Bildpfaden
        device      : CPU oder GPU
        batch_size  : Bilder pro Batch

    Returns:
        Liste von Ergebnis-Dicts (gleiche Struktur wie predict())
    """
    from torch.utils.data import DataLoader, Dataset

    class SimpleImageDataset(Dataset):
        def __init__(self, paths, transform):
            self.paths     = paths
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), str(self.paths[idx])

    transform = get_eval_transforms(img_size, mean, std)
    dataset   = SimpleImageDataset(image_paths, transform)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    results = []
    model.eval()

    with torch.no_grad():
        for tensors, paths in loader:
            tensors = tensors.to(device)
            logits  = model(tensors)
            probs   = F.softmax(logits, dim=1)

            for i in range(len(paths)):
                prob_h        = probs[i][0].item()
                prob_inf      = probs[i][1].item()
                predicted_idx = probs[i].argmax().item()
                confidence    = probs[i][predicted_idx].item()

                results.append({
                    "image_path"      : paths[i],
                    "predicted_class" : IDX_TO_CLASS[predicted_idx],
                    "predicted_idx"   : predicted_idx,
                    "confidence"      : confidence,
                    "prob_healthy"    : prob_h,
                    "prob_infected"   : prob_inf,
                    "uncertain"       : confidence < UNCERTAINTY_THRESHOLD,
                    "inference_ms"    : None,  # Batch → keine Einzelzeit
                })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Ergebnisse speichern
# ──────────────────────────────────────────────────────────────────────────────

def save_results(results: List[Dict], prefix: str = "predict") -> Dict[str, Path]:
    """
    Speichert Vorhersage-Ergebnisse als CSV und JSON.

    Dateien landen in results/predictions/ mit Zeitstempel
    damit ältere Ergebnisse nicht überschrieben werden.

    Args:
        results : Liste von Ergebnis-Dicts
        prefix  : Dateinamen-Präfix

    Returns:
        Dict mit Pfaden der gespeicherten Dateien
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path  = RESULTS_DIR / f"{prefix}_{timestamp}.csv"
    json_path = RESULTS_DIR / f"{prefix}_{timestamp}.json"

    # CSV
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Gespeichert: {csv_path}")
    print(f"Gespeichert: {json_path}")

    return {"csv": csv_path, "json": json_path}


# ──────────────────────────────────────────────────────────────────────────────
# Ergebnis ausgeben
# ──────────────────────────────────────────────────────────────────────────────

def print_result(result: Dict, image_path: str | Path) -> None:
    """Gibt das Ergebnis übersichtlich in der Konsole aus."""
    label      = result["predicted_class"].upper()
    confidence = result["confidence"] * 100
    bar_filled = int(confidence / 5)
    bar        = "█" * bar_filled + "░" * (20 - bar_filled)

    print()
    print("╔══════════════════════════════════════════╗")
    print("║           Malaria-KI Ergebnis            ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Bild      : {Path(image_path).name:<28s}║")
    print(f"║  Diagnose  : {label:<28s}║")
    print(f"║  Confidence: {confidence:>5.1f}%                        ║")
    print(f"║  [{bar}]   ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  P(healthy)  = {result['prob_healthy']:>6.4f}                  ║")
    print(f"║  P(infected) = {result['prob_infected']:>6.4f}                  ║")
    print(f"║  Inferenzzeit: {result['inference_ms']:>5.1f} ms                 ║")

    if result["uncertain"]:
        print("╠══════════════════════════════════════════╣")
        print("║  ⚠️  UNSICHER – bitte manuell prüfen      ║")

    print("╚══════════════════════════════════════════╝")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Ordner-Vorhersage
# ──────────────────────────────────────────────────────────────────────────────

def predict_folder(
    model: torch.nn.Module,
    folder_path: str | Path,
    device: torch.device,
    img_size: int = 224,
    batch_size: int = 32,
    save: bool = True,
) -> List[Dict]:
    """
    Batch-Vorhersage für alle Bilder in einem Ordner.

    Nutzt predict_batch() für effiziente GPU-Auslastung.
    Speichert Ergebnisse optional als CSV/JSON.
    """
    folder_path = Path(folder_path)
    image_paths = [p for p in sorted(folder_path.rglob("*"))
                   if p.suffix.lower() in VALID_EXTENSIONS]

    if not image_paths:
        print(f"[WARN] Keine Bilder in {folder_path}")
        return []

    print(f"Verarbeite {len(image_paths)} Bilder (Batch-Grösse: {batch_size})...")

    results = predict_batch(model, image_paths, device, img_size, batch_size)

    # Konsolenausgabe
    for r in results:
        flag   = "⚠️ " if r["uncertain"] else ""
        status = "🔴 INFECTED" if r["predicted_idx"] == 1 else "🟢 HEALTHY"
        name   = Path(r["image_path"]).name
        print(f"  {name:<40s} {flag}{status}  ({r['confidence']*100:.1f}%)")

    # Zusammenfassung
    n_infected  = sum(1 for r in results if r["predicted_idx"] == 1)
    n_uncertain = sum(1 for r in results if r["uncertain"])
    print(f"\nGesamt: {len(results)} | Healthy: {len(results)-n_infected} "
          f"| Infected: {n_infected} | Unsicher: {n_uncertain}")

    if save:
        save_results(results)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Gradio Web-Interface
# ──────────────────────────────────────────────────────────────────────────────

def launch_gui(
    model: torch.nn.Module,
    device: torch.device,
    gradcam: bool = True,
) -> None:
    """
    Startet ein Gradio Web-Interface für interaktive Vorhersagen.

    Bilder per Drag-and-Drop hochladen → sofortiges Ergebnis
    inklusive Grad-CAM Visualisierung.

    Args:
        model   : Geladenes Modell
        device  : CPU oder GPU
        gradcam : Grad-CAM Heatmap anzeigen
    """
    try:
        import gradio as gr
    except ImportError:
        print("[ERROR] Gradio nicht installiert. Bitte: pip install gradio")
        return

    cam = GradCAM(model, get_last_conv_layer(model)) if gradcam else None

    def run_prediction(pil_image: Image.Image):
        if pil_image is None:
            return None, "Kein Bild hochgeladen."

        transform = get_eval_transforms()
        tensor    = transform(pil_image.convert("RGB")).unsqueeze(0)
        result    = predict(model, tensor, device)

        label      = result["predicted_class"].upper()
        confidence = result["confidence"] * 100
        uncertain  = " ⚠️ UNSICHER" if result["uncertain"] else ""

        text = (
            f"**Diagnose: {label}**{uncertain}\n\n"
            f"Confidence:  {confidence:.1f}%\n"
            f"P(healthy):  {result['prob_healthy']*100:.2f}%\n"
            f"P(infected): {result['prob_infected']*100:.2f}%\n"
            f"Inferenz:    {result['inference_ms']} ms"
        )

        # Grad-CAM Bild
        overlay_img = None
        if cam is not None:
            heatmap     = cam(tensor.to(device), result["predicted_idx"])
            overlay_img = GradCAM.overlay(pil_image.convert("RGB"), heatmap)

        return overlay_img, text

    with gr.Blocks(title="Malaria-KI") as demo:
        gr.Markdown("## 🔬 Malaria-Zell Klassifikation")
        with gr.Row():
            input_img  = gr.Image(type="pil", label="Zellbild hochladen")
            output_img = gr.Image(label="Grad-CAM Visualisierung")
        output_text = gr.Markdown()
        btn = gr.Button("Analysieren", variant="primary")
        btn.click(run_prediction, inputs=input_img, outputs=[output_img, output_text])

    demo.launch(share=False)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Malaria-Zell Vorhersage")
    parser.add_argument("--image",      type=str, default=None)
    parser.add_argument("--folder",     type=str, default=None)
    parser.add_argument("--model",      type=str, default="models/final/model.pth")
    parser.add_argument("--img-size",   type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradcam",    action="store_true")
    parser.add_argument("--save",       action="store_true")
    parser.add_argument("--gui",        action="store_true")
    parser.add_argument("--device",     type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Einstiegspunkt
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = torch.device(args.device)

    print(f"Gerät  : {device}")
    print(f"Modell : {args.model}")

    model = load_model(args.model, device)
    print("Modell geladen ✓\n")

    # GUI
    if args.gui:
        launch_gui(model, device, gradcam=args.gradcam)

    # Einzelbild
    elif args.image:
        tensor, original = preprocess_image(args.image, args.img_size)
        result = predict(model, tensor, device)
        print_result(result, args.image)

        if args.gradcam:
            import matplotlib.pyplot as plt
            cam     = GradCAM(model, get_last_conv_layer(model))
            heatmap = cam(tensor.to(device), result["predicted_idx"])
            overlay = GradCAM.overlay(original, heatmap)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original);          axes[0].set_title("Original");  axes[0].axis("off")
            axes[1].imshow(overlay);           axes[1].set_title("Grad-CAM");  axes[1].axis("off")
            plt.tight_layout()
            plt.show()

        if args.save:
            result["image_path"] = str(args.image)
            save_results([result])

    # Ordner
    elif args.folder:
        predict_folder(model, args.folder, device, args.img_size, args.batch_size, save=args.save)

    else:
        print("Kein Modus angegeben. Beispiele:")
        print("  python -m src.predict --image cell.png")
        print("  python -m src.predict --image cell.png --gradcam --save")
        print("  python -m src.predict --folder data/raw/infected/ --save")
        print("  python -m src.predict --gui")