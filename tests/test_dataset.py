"""Tests fuer die Transform-Pipelines."""
import pytest

pytest.importorskip("torch")
from PIL import Image  # noqa: E402


def test_eval_transform_shape():
    from src.dataset import get_eval_transforms
    img = Image.new("RGB", (130, 90), (120, 60, 60))
    out = get_eval_transforms(224)(img)
    assert tuple(out.shape) == (3, 224, 224)


def test_train_transform_shape():
    from src.dataset import get_train_transforms
    img = Image.new("RGB", (130, 90), (120, 60, 60))
    out = get_train_transforms(224)(img)
    assert tuple(out.shape) == (3, 224, 224)


def test_valid_extensions():
    from src.dataset import VALID_EXT
    assert ".png" in VALID_EXT and ".jpg" in VALID_EXT
