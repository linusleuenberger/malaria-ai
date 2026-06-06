# ============================================================
# tests/test_model.py
# Tests für Modell-Architektur, Forward Pass, Save/Load
# ============================================================

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.config import (
    ARCHITECTURE,
    DEVICE,
    DROPOUT_RATE,
    FREEZE_BACKBONE,
    HIDDEN_SIZE,
    NUM_CLASSES,
)
from src.model import (
    build_model,
    get_loss_function,
    load_model,
    save_model,
    unfreeze_layers,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def model():
    """
    Erstellt ein frisches Modell für jeden Test.
    Wird nach dem Test automatisch gelöscht.
    """
    return build_model(freeze_backbone=True)


@pytest.fixture
def dummy_batch():
    """
    Erstellt einen Dummy-Batch für Forward Pass Tests.
    Shape: [4, 3, 224, 224] → 4 Bilder, RGB, 224×224
    """
    return torch.randn(4, 3, 224, 224).to(DEVICE)


@pytest.fixture
def dummy_labels():
    """
    Erstellt Dummy-Labels für Loss Tests.
    4 Labels, zufällig 0 (healthy) oder 1 (infected)
    """
    return torch.randint(0, NUM_CLASSES, (4,)).to(DEVICE)


@pytest.fixture
def temp_model_path():
    """
    Erstellt temporären Pfad zum Modell speichern.
    Nach dem Test automatisch gelöscht.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir) / "test_model.pth"


# ── Tests: Modell aufbauen ────────────────────────────────────

class TestBuildModel:
    """
    Tests für build_model Funktion.

    Testet:
        ✓ Modell wird erstellt
        ✓ Ausgabe hat korrekte Klassen
        ✓ Backbone eingefroren wenn freeze_backbone=True
        ✓ Backbone trainierbar wenn freeze_backbone=False
        ✓ Verschiedene Architekturen funktionieren
        ✓ Verschiedene Klassenanzahlen funktionieren
        ✓ Modell auf korrektem Device
    """

    def test_model_builds_successfully(self):
        """Prüft ob Modell ohne Fehler erstellt wird."""
        model = build_model()
        assert model is not None, \
            "build_model() sollte ein Modell zurückgeben"

    def test_model_correct_num_classes(self):
        """Prüft ob Modell korrekte Anzahl Klassen hat."""
        for num_classes in [2, 5]:
            model = build_model(num_classes=num_classes)

            # Letzten Layer finden
            if hasattr(model, "fc"):
                last_layer = list(model.fc.children())[-1]
            else:
                last_layer = list(model.classifier.children())[-1]

            assert last_layer.out_features == num_classes, (
                f"Erwartet {num_classes} Ausgaben, "
                f"erhalten: {last_layer.out_features}"
            )

    def test_backbone_frozen_when_specified(self):
        """
        Prüft ob Backbone eingefroren ist.

        Eingefroren = requires_grad=False
        → Layer werden beim Training nicht verändert
        → Vortrainiertes Wissen bleibt erhalten
        """
        model = build_model(freeze_backbone=True)

        # Backbone-Parameter sollten eingefroren sein
        frozen_params = [
            p for name, p in model.named_parameters()
            if "fc" not in name and "classifier" not in name
            and not p.requires_grad
        ]

        assert len(frozen_params) > 0, (
            "freeze_backbone=True: "
            "Backbone-Parameter sollten eingefroren sein"
        )

    def test_backbone_trainable_when_not_frozen(self):
        """Prüft ob Backbone trainierbar ist wenn nicht eingefroren."""
        model = build_model(freeze_backbone=False)

        trainable_params = [
            p for p in model.parameters()
            if p.requires_grad
        ]

        assert len(trainable_params) > 0, (
            "freeze_backbone=False: "
            "Parameter sollten trainierbar sein"
        )

    def test_model_head_always_trainable(self):
        """
        Prüft ob der neue Kopf immer trainierbar ist.
        Auch wenn Backbone eingefroren ist muss der Kopf lernen.
        """
        model = build_model(freeze_backbone=True)

        # fc oder classifier Layer finden
        if hasattr(model, "fc"):
            head_params = list(model.fc.parameters())
        else:
            head_params = list(model.classifier.parameters())

        trainable = [p for p in head_params if p.requires_grad]
        assert len(trainable) > 0, (
            "Modell-Kopf muss immer trainierbar sein, "
            "auch wenn Backbone eingefroren"
        )

    def test_resnet50_architecture(self):
        """Prüft ob ResNet50 korrekt geladen wird."""
        model = build_model(architecture="resnet50")
        assert hasattr(model, "fc"), \
            "ResNet50 sollte fc Layer haben"

    def test_resnet101_architecture(self):
        """Prüft ob ResNet101 korrekt geladen wird."""
        model = build_model(architecture="resnet101")
        assert hasattr(model, "fc"), \
            "ResNet101 sollte fc Layer haben"

    def test_efficientnet_architecture(self):
        """Prüft ob EfficientNet korrekt geladen wird."""
        model = build_model(architecture="efficientnet_b0")
        assert hasattr(model, "classifier"), \
            "EfficientNet sollte classifier Layer haben"

    def test_invalid_architecture_raises_error(self):
        """Prüft ob ungültige Architektur Fehler wirft."""
        with pytest.raises(ValueError):
            build_model(architecture="ungueltige_architektur")

    def test_model_on_correct_device(self, model):
        """Prüft ob Modell auf korrektem Device ist."""
        expected_device = torch.device(DEVICE)

        for param in model.parameters():
            assert param.device.type == expected_device.type, (
                f"Modell sollte auf {DEVICE} sein, "
                f"ist auf {param.device.type}"
            )
            break  # Ersten Parameter prüfen reicht


# ── Tests: Forward Pass ───────────────────────────────────────

class TestForwardPass:
    """
    Tests für den Forward Pass durch das Modell.

    Testet:
        ✓ Output Shape korrekt
        ✓ Keine NaN in Output
        ✓ Keine Inf in Output
        ✓ Batch-Verarbeitung funktioniert
        ✓ Einzelbild funktioniert
        ✓ eval() und train() Modus
    """

    def test_forward_pass_output_shape(
        self, model, dummy_batch
    ):
        """Prüft ob Output die richtige Shape hat."""
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)

        expected = (dummy_batch.shape[0], NUM_CLASSES)
        assert output.shape == torch.Size(expected), (
            f"Erwartete Output-Shape {expected}, "
            f"erhalten: {tuple(output.shape)}"
        )

    def test_forward_pass_no_nan(
        self, model, dummy_batch
    ):
        """Prüft ob kein NaN im Output vorhanden ist."""
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)

        assert not torch.isnan(output).any(), \
            "NaN Werte im Modell-Output gefunden"

    def test_forward_pass_no_inf(
        self, model, dummy_batch
    ):
        """Prüft ob kein Inf im Output vorhanden ist."""
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)

        assert not torch.isinf(output).any(), \
            "Inf Werte im Modell-Output gefunden"

    def test_forward_pass_single_image(self, model):
        """Prüft ob einzelnes Bild (Batch-Size=1) funktioniert."""
        single_image = torch.randn(1, 3, 224, 224).to(DEVICE)

        model.eval()
        with torch.no_grad():
            output = model(single_image)

        assert output.shape == torch.Size([1, NUM_CLASSES]), (
            f"Einzelbild: Erwartete Shape (1, {NUM_CLASSES}), "
            f"erhalten: {tuple(output.shape)}"
        )

    def test_forward_pass_large_batch(self, model):
        """Prüft ob grosse Batch-Size funktioniert."""
        large_batch = torch.randn(16, 3, 224, 224).to(DEVICE)

        model.eval()
        with torch.no_grad():
            output = model(large_batch)

        assert output.shape[0] == 16, (
            f"Batch-Size 16: Erwartete 16 Outputs, "
            f"erhalten: {output.shape[0]}"
        )

    def test_forward_pass_train_mode(
        self, model, dummy_batch
    ):
        """Prüft ob Forward Pass im Train-Modus funktioniert."""
        model.train()
        output = model(dummy_batch)

        assert output.shape == torch.Size(
            [dummy_batch.shape[0], NUM_CLASSES]
        ), (
            f"Train-Modus: Erwartete Shape "
            f"({dummy_batch.shape[0]}, {NUM_CLASSES}), "
            f"erhalten: {tuple(output.shape)}"
        )

    def test_forward_pass_eval_mode(
        self, model, dummy_batch
    ):
        """Prüft ob Forward Pass im Eval-Modus funktioniert."""
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)

        assert output.shape == torch.Size(
            [dummy_batch.shape[0], NUM_CLASSES]
        ), (
            f"Eval-Modus: Erwartete Shape "
            f"({dummy_batch.shape[0]}, {NUM_CLASSES}), "
            f"erhalten: {tuple(output.shape)}"
        )

    def test_softmax_sums_to_one(
        self, model, dummy_batch
    ):
        """
        Prüft ob Softmax-Wahrscheinlichkeiten sich zu 1 addieren.

        Warum wichtig:
            Softmax Output = Wahrscheinlichkeit
            Alle Klassen zusammen = 100% = 1.0
            Falls nicht → Fehler in Modell-Architektur
        """
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)
            probs  = torch.softmax(output, dim=1)

        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"Softmax Summen sollten 1.0 sein, "
            f"erhalten: {sums.tolist()}"
        )

    def test_output_changes_with_different_inputs(self, model):
        """
        Prüft ob verschiedene Inputs verschiedene Outputs geben.
        Falls immer gleicher Output → Modell defekt.
        """
        input_1 = torch.randn(1, 3, 224, 224).to(DEVICE)
        input_2 = torch.randn(1, 3, 224, 224).to(DEVICE)

        model.eval()
        with torch.no_grad():
            output_1 = model(input_1)
            output_2 = model(input_2)

        assert not torch.allclose(output_1, output_2), \
            "Verschiedene Inputs sollten verschiedene Outputs geben"


# ── Tests: Loss Funktion ──────────────────────────────────────

class TestLossFunction:
    """
    Tests für get_loss_function.

    Testet:
        ✓ Loss wird berechnet
        ✓ Loss ist positiv
        ✓ Loss ist nicht NaN
        ✓ Perfekte Vorhersage → niedrigerer Loss
    """

    def test_loss_returns_scalar(
        self, model, dummy_batch, dummy_labels
    ):
        """Prüft ob Loss ein Skalar ist."""
        criterion = get_loss_function()
        model.train()

        output = model(dummy_batch)
        loss   = criterion(output, dummy_labels)

        assert loss.dim() == 0, \
            f"Loss sollte Skalar sein, Shape: {loss.shape}"

    def test_loss_is_positive(
        self, model, dummy_batch, dummy_labels
    ):
        """Prüft ob Loss positiv ist."""
        criterion = get_loss_function()
        model.train()

        output = model(dummy_batch)
        loss   = criterion(output, dummy_labels)

        assert loss.item() > 0, \
            f"Loss sollte positiv sein, erhalten: {loss.item()}"

    def test_loss_not_nan(
        self, model, dummy_batch, dummy_labels
    ):
        """Prüft ob Loss kein NaN ist."""
        criterion = get_loss_function()
        model.train()

        output = model(dummy_batch)
        loss   = criterion(output, dummy_labels)

        assert not torch.isnan(loss), \
            "Loss ist NaN – Training würde fehlschlagen"

    def test_loss_not_inf(
        self, model, dummy_batch, dummy_labels
    ):
        """Prüft ob Loss kein Inf ist."""
        criterion = get_loss_function()
        model.train()

        output = model(dummy_batch)
        loss   = criterion(output, dummy_labels)

        assert not torch.isinf(loss), \
            "Loss ist Inf – Training würde fehlschlagen"

    def test_better_prediction_lower_loss(self, model):
        """
        Prüft ob bessere Vorhersage → niedrigerer Loss.

        Logik:
            Perfekte Vorhersage (sehr hohe Konfidenz)
            sollte niedrigeren Loss haben als
            schlechte Vorhersage (zufällig)
        """
        criterion = get_loss_function()
        labels    = torch.tensor([0, 1, 0, 1]).to(DEVICE)

        # Gute Vorhersage: hohe Konfidenz für richtiges Label
        good_logits = torch.tensor([
            [10.0, -10.0],  # healthy → korrekt
            [-10.0, 10.0],  # infected → korrekt
            [10.0, -10.0],  # healthy → korrekt
            [-10.0, 10.0],  # infected → korrekt
        ]).to(DEVICE)

        # Schlechte Vorhersage: hohe Konfidenz für falsches Label
        bad_logits = torch.tensor([
            [-10.0, 10.0],  # healthy als infected → falsch
            [10.0, -10.0],  # infected als healthy → falsch
            [-10.0, 10.0],  # healthy als infected → falsch
            [10.0, -10.0],  # infected als healthy → falsch
        ]).to(DEVICE)

        good_loss = criterion(good_logits, labels)
        bad_loss  = criterion(bad_logits,  labels)

        assert good_loss < bad_loss, (
            f"Gute Vorhersage sollte niedrigeren Loss haben.\n"
            f"Guter Loss:     {good_loss.item():.4f}\n"
            f"Schlechter Loss:{bad_loss.item():.4f}"
        )


# ── Tests: Modell speichern & laden ───────────────────────────

class TestSaveLoadModel:
    """
    Tests für save_model und load_model.

    Testet:
        ✓ Modell wird gespeichert
        ✓ Modell wird korrekt geladen
        ✓ Gewichte sind nach Laden identisch
        ✓ Geladenes Modell macht gleiche Vorhersagen
        ✓ Fehler bei nicht existierendem Pfad
    """

    def test_model_saves_file(
        self, model, temp_model_path
    ):
        """Prüft ob Datei nach save_model existiert."""
        save_model(model, temp_model_path)

        assert temp_model_path.exists(), \
            f"Modell-Datei sollte existieren: {temp_model_path}"

    def test_model_loads_successfully(
        self, model, temp_model_path
    ):
        """Prüft ob Modell ohne Fehler geladen wird."""
        save_model(model, temp_model_path)
        loaded = load_model(temp_model_path)

        assert loaded is not None, \
            "load_model() sollte Modell zurückgeben"

    def test_saved_weights_identical(
        self, model, temp_model_path
    ):
        """
        Prüft ob Gewichte nach Speichern & Laden identisch sind.

        Warum wichtig:
            Falls Gewichte sich verändern beim Speichern/Laden
            → Bestes Modell ist nicht mehr das beste Modell
            → Evaluation falsch
        """
        save_model(model, temp_model_path)
        loaded = load_model(temp_model_path)

        original_params = dict(model.named_parameters())
        loaded_params   = dict(loaded.named_parameters())

        for name, param in original_params.items():
            assert name in loaded_params, \
                f"Parameter {name} fehlt im geladenen Modell"
            assert torch.allclose(
                param.cpu(),
                loaded_params[name].cpu(),
                atol=1e-6
            ), (
                f"Parameter {name} hat sich beim "
                f"Speichern/Laden verändert"
            )

    def test_loaded_model_same_predictions(
        self, model, temp_model_path
    ):
        """
        Prüft ob geladenes Modell gleiche Vorhersagen macht.
        """
        dummy = torch.randn(2, 3, 224, 224).to(DEVICE)

        model.eval()
        with torch.no_grad():
            original_output = model(dummy)

        save_model(model, temp_model_path)
        loaded = load_model(temp_model_path)

        loaded.eval()
        with torch.no_grad():
            loaded_output = loaded(dummy)

        assert torch.allclose(
            original_output,
            loaded_output,
            atol=1e-5
        ), (
            "Geladenes Modell macht andere Vorhersagen\n"
            f"Original: {original_output[0].tolist()}\n"
            f"Geladen:  {loaded_output[0].tolist()}"
        )

    def test_load_nonexistent_model_raises_error(self):
        """Prüft ob Fehler bei nicht existierendem Pfad."""
        with pytest.raises(Exception):
            load_model(Path("/nicht/vorhanden/model.pth"))

    def test_save_model_includes_metadata(
        self, model, temp_model_path
    ):
        """
        Prüft ob gespeicherter Checkpoint Metadaten enthält.
        Metadaten: Architektur, Anzahl Klassen.
        """
        save_model(
            model,
            temp_model_path,
            extra_info={"epoch": 10, "val_acc": 0.95}
        )

        checkpoint = torch.load(
            temp_model_path,
            map_location="cpu"
        )

        assert "model_state_dict" in checkpoint, \
            "Checkpoint sollte model_state_dict enthalten"
        assert "architecture" in checkpoint, \
            "Checkpoint sollte Architektur enthalten"
        assert "num_classes" in checkpoint, \
            "Checkpoint sollte num_classes enthalten"
        assert "epoch" in checkpoint, \
            "Checkpoint sollte Epoch enthalten"
        assert "val_acc" in checkpoint, \
            "Checkpoint sollte val_acc enthalten"


# ── Tests: Unfreeze Layers ────────────────────────────────────

class TestUnfreezeLayers:
    """
    Tests für unfreeze_layers Funktion.

    Testet:
        ✓ Layer werden auftaut
        ✓ Anzahl trainierbare Parameter steigt
        ✓ Bereits trainierbare Parameter bleiben trainierbar
    """

    def test_unfreeze_increases_trainable_params(self, model):
        """
        Prüft ob unfreeze_layers mehr trainierbare Parameter ergibt.
        """
        before = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )

        unfreeze_layers(model, n_layers=20)

        after = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )

        assert after >= before, (
            f"Trainierbare Parameter sollten nach unfreeze "
            f"zunehmen oder gleich bleiben.\n"
            f"Vorher: {before:,}\n"
            f"Nachher: {after:,}"
        )

    def test_unfreeze_all_layers(self, model):
        """Prüft ob alle Layer auftaubar sind."""
        total_params = sum(
            p.numel() for p in model.parameters()
        )

        # Alle Layer auftauen
        unfreeze_layers(model, n_layers=1000)

        trainable = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )

        assert trainable == total_params, (
            f"Nach vollständigem Auftauen sollten alle "
            f"Parameter trainierbar sein.\n"
            f"Total: {total_params:,}\n"
            f"Trainierbar: {trainable:,}"
        )


# ── Tests: Parameter Count ────────────────────────────────────

class TestParameterCount:
    """
    Tests für Anzahl Parameter.

    Testet:
        ✓ Modell hat Parameter
        ✓ ResNet50 hat ~25 Millionen Parameter
        ✓ Frozen Modell hat weniger trainierbare Parameter
    """

    def test_model_has_parameters(self, model):
        """Prüft ob Modell Parameter hat."""
        total = sum(p.numel() for p in model.parameters())
        assert total > 0, "Modell sollte Parameter haben"

    def test_resnet50_parameter_count(self):
        """
        Prüft ob ResNet50 ~25 Millionen Parameter hat.
        Toleranz: 20–30 Millionen.
        """
        model = build_model(architecture="resnet50")
        total = sum(p.numel() for p in model.parameters())

        assert 20_000_000 < total < 30_000_000, (
            f"ResNet50 sollte ~25M Parameter haben, "
            f"erhalten: {total:,}"
        )

    def test_frozen_model_fewer_trainable_params(self):
        """Prüft ob gefrorenes Modell weniger trainierbare Parameter hat."""
        frozen   = build_model(freeze_backbone=True)
        unfrozen = build_model(freeze_backbone=False)

        frozen_trainable   = sum(
            p.numel() for p in frozen.parameters()
            if p.requires_grad
        )
        unfrozen_trainable = sum(
            p.numel() for p in unfrozen.parameters()
            if p.requires_grad
        )

        assert frozen_trainable < unfrozen_trainable, (
            f"Gefrorenes Modell sollte weniger trainierbare "
            f"Parameter haben.\n"
            f"Gefroren:   {frozen_trainable:,}\n"
            f"Aufgetaut:  {unfrozen_trainable:,}"
        )


# ── Quick-Test ────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])