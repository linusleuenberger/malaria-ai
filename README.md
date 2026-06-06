# Malaria-KI - Erkennung infizierter Blutzellen

Transfer-Learning (ResNet50) zur Klassifikation von Blutzellbildern in
`healthy` / `infected`.

## Datensatz pro Schritt (wichtig)

Jeder Schritt nutzt bewusst seinen eigenen Split, damit das Ergebnis nicht
beschoenigt wird:

| Schritt      | Datensatz                  | Zweck                              |
|--------------|----------------------------|------------------------------------|
| Training     | `data/processed/train`     | Lernen (mit Online-Augmentierung)  |
| Validierung  | `data/processed/val`       | Modellauswahl / Early Stopping     |
| Test         | `data/processed/test`      | ehrliche, einmalige Endbewertung   |

Die Augmentierung passiert live waehrend des Trainings (in `dataset.py`).
Der separate Ordner `data/augmented/` wird dadurch nicht benoetigt.

## Installation

GPU (RTX 5070 Ti, CUDA 12.8) zuerst:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python check_setup.py
```

## Ablauf

```
python main.py --mode preprocess              # einmalig: raw -> processed
python main.py --mode train --eval-after      # trainieren + danach Test
python analyze.py --tta                        # ausfuehrliche Zuverlaessigkeits-Analyse
```

### Training fortsetzen

Nach jeder Epoche wird `models/checkpoints/last.pth` (kompletter Zustand)
geschrieben. Bei Abbruch nahtlos weitertrainieren:

```
python main.py --mode train --resume
```

Das beste Modell (kleinster Val-Loss) landet in `models/checkpoints/best.pth`
und am Ende als Kopie in `models/final/final_model.pth`.

### Exakt reproduzierbarer Lauf

Standardmaessig laeuft das Training schnell (volle GPU-Auslastung). Fuer einen
bitgenau wiederholbaren Referenzlauf:

```
python main.py --mode train --deterministic
```

## Ausgaben

- `models/checkpoints/last.pth` - Zwischenstand (resume)
- `models/checkpoints/best.pth` - bestes Modell waehrend des Trainings
- `models/final/final_model.pth` - finales Modell (Kopie des besten)
- `results/metrics/` - `training_summary.json`, `test_metrics.json`,
  `optimal_threshold.json`, `evaluation_report.json`
- `results/evaluation_report.md` - lesbarer Zuverlaessigkeits-Bericht
- `results/plots/` - Loss/Accuracy, Confusion Matrix, ROC, PR,
  Reliability-Diagram, Konfidenz-Histogramm, Grad-CAM, Fehlklassifikationen

## Projektstruktur

```
main.py            Einstiegspunkt (preprocess | train | evaluate | analyze | predict)
analyze.py         separate, ausfuehrliche Zuverlaessigkeits-Analyse
check_setup.py     prueft Installation & Daten
src/
  config.py        alle Pfade & Hyperparameter
  dataset.py       Dataset, Transforms, DataLoader
  model.py         Modellaufbau, Speichern/Laden
  train.py         Trainingsloop (Checkpoints, EMA, Cosine-LR, AMP)
  evaluate.py      schnelle Evaluation auf dem Test-Split
  predict.py       Vorhersage + Grad-CAM
  utils.py         Seed, Performance-Setup, Sanity Check
preprocessing/     raw -> processed (Filter, Split, Statistiken)
notebooks/         Datenexploration & Visualisierung
tests/             pytest-Tests
```
