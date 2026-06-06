"""
src/predict.py - Vorhersage fuer Einzelbilder und Ordner.

Enthaelt ausserdem Grad-CAM (Heatmap: worauf schaut das Modell?),
das auch von evaluate.py / analyze.py genutzt wird.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.config import (
    DEVICE,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    MEAN,
    METRICS_DIR,
    STD,
)
from src.dataset import VALID_EXT, get_eval_transforms

logger = logging.getLogger(__name__)


# ── Optimaler Schwellenwert (aus der Evaluation) ──────────────
def load_threshold(default: float = 0.5) -> float:
    """Liest den in der Evaluation bestimmten Schwellenwert, sonst 0.5."""
    f = METRICS_DIR / "optimal_threshold.json"
    if f.exists():
        try:
            return float(json.loads(f.read_text())["threshold"])
        except Exception:
            pass
    return default


# ── Grad-CAM ──────────────────────────────────────────────────
class GradCAM:
    """Gradient-weighted Class Activation Mapping (Heatmap der wichtigen Regionen)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _m, _i, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _m, _gi, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        image_tensor = image_tensor.to(DEVICE).requires_grad_(True)
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
        cam = cam.squeeze().float().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    @staticmethod
    def overlay(original: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
        import matplotlib.pyplot as plt
        colored = (plt.get_cmap("jet")(heatmap)[:, :, :3] * 255).astype(np.uint8)
        heat = Image.fromarray(colored).resize(original.size, Image.BILINEAR)
        return Image.blend(original.convert("RGB"), heat, alpha=alpha)


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """Letzte Conv2d-Schicht finden (Ziel-Layer fuer Grad-CAM)."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("Keine Conv2d-Schicht gefunden.")
    return last


# ── Bild vorbereiten ──────────────────────────────────────────
def preprocess_image(image_path: str | Path) -> Tuple[torch.Tensor, Image.Image]:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    original = Image.open(image_path).convert("RGB")
    tensor = get_eval_transforms(IMAGE_SIZE[0], MEAN, STD)(original).unsqueeze(0)
    return tensor, original


# ── Einzelvorhersage ──────────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module, image_tensor: torch.Tensor,
            threshold: Optional[float] = None) -> Dict:
    """Wahrscheinlichkeit fuer 'infected' wird mit threshold verglichen."""
    if threshold is None:
        threshold = load_threshold()
    probs = F.softmax(model(image_tensor.to(DEVICE)), dim=1)[0]
    p_infected = probs[1].item()
    idx = 1 if p_infected >= threshold else 0
    return {
        "predicted_class": IDX_TO_CLASS[idx],
        "predicted_idx": idx,
        "confidence": probs[idx].item(),
        "prob_healthy": probs[0].item(),
        "prob_infected": p_infected,
        "threshold": threshold,
    }


# ── Ordner-Vorhersage (Batch) ─────────────────────────────────
class _PathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), str(self.paths[i])


@torch.no_grad()
def predict_folder(model: nn.Module, folder: str | Path, batch_size: int = 64,
                   threshold: Optional[float] = None) -> List[Dict]:
    if threshold is None:
        threshold = load_threshold()
    folder = Path(folder)
    paths = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in VALID_EXT]
    if not paths:
        logger.warning("Keine Bilder in %s", folder)
        return []

    loader = DataLoader(_PathDataset(paths, get_eval_transforms(IMAGE_SIZE[0], MEAN, STD)),
                        batch_size=batch_size, num_workers=4)
    results: List[Dict] = []
    model.eval()
    for tensors, names in loader:
        probs = F.softmax(model(tensors.to(DEVICE)), dim=1)
        for p, name in zip(probs, names):
            p_inf = p[1].item()
            idx = 1 if p_inf >= threshold else 0
            results.append({"image": Path(name).name,
                            "predicted": IDX_TO_CLASS[idx],
                            "confidence": round(p[idx].item(), 4),
                            "prob_infected": round(p_inf, 4)})
    return results
