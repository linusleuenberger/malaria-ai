"""Tests fuer Konfiguration & Datensatz-Split."""
import pytest

pytest.importorskip("torch")  # braucht src.config -> torch


def test_splits_sum_to_one():
    from src.config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
    assert abs(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT - 1.0) < 1e-6


def test_class_definitions():
    from src.config import CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
    assert CLASS_NAMES == ["healthy", "infected"]
    assert NUM_CLASSES == 2
    assert CLASS_TO_IDX["healthy"] == 0 and CLASS_TO_IDX["infected"] == 1
    assert IDX_TO_CLASS[0] == "healthy" and IDX_TO_CLASS[1] == "infected"


def test_mean_std_length():
    from src.config import MEAN, STD
    assert len(MEAN) == 3 and len(STD) == 3
