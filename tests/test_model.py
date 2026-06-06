"""Tests fuer den Modellaufbau (ohne vortrainierte Gewichte -> schnell)."""
import pytest

torch = pytest.importorskip("torch")


def test_build_model_output_shape():
    from src.model import build_model
    from src.config import NUM_CLASSES
    model = build_model(freeze_backbone=False, pretrained=False).to("cpu")
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert tuple(out.shape) == (2, NUM_CLASSES)


def test_loss_function():
    from src.model import get_loss_function
    assert isinstance(get_loss_function(), torch.nn.Module)


def test_unfreeze_layers():
    from src.model import build_model, unfreeze_layers
    model = build_model(freeze_backbone=True, pretrained=False)
    trainable = unfreeze_layers(model, 10)
    assert trainable > 0
