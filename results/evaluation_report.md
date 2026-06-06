# Zuverlaessigkeits-Analyse des Malaria-Modells

- Modell: `D:\Maturaarbeit\malaria-ai\models\final\final_model.pth`
- Testbilder: 4136  |  Test-Time-Augmentation: True
- Optimaler Schwellenwert: **0.42**  (Recall-Vorgabe >= 95 %)

## Kernmetriken (Test-Split)

| Metrik | Wert |
|---|---|
| Accuracy | 75.68 % (95 % CI 74.30-77.06) |
| Recall (Sensitivitaet) | 75.68 % |
| Specificity | 56.19 % |
| Precision | 80.28 % |
| F1-Score | 74.72 % |
| AUC | 0.9200 |
| AP | 0.9241 |
| ECE (Kalibrierung, kleiner=besser) | 0.2573 |

## Pro Klasse

| Klasse | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| healthy | 0.9208 | 0.5619 | 0.6979 | 2068 |
| infected | 0.6848 | 0.9516 | 0.7964 | 2068 |

## Schwellenwert-Analyse

| Schwelle | Precision | Recall | F1 |
|---|---|---|---|
| 0.10 | 0.5000 | 1.0000 | 0.6667 |
| 0.20 | 0.5002 | 1.0000 | 0.6669 |
| 0.30 | 0.5228 | 0.9937 | 0.6851 |
| 0.40 | 0.6469 | 0.9666 | 0.7751 |
| 0.50 | 0.8606 | 0.8482 | 0.8544 |
| 0.60 | 0.9783 | 0.3922 | 0.5599 |
| 0.70 | 0.9706 | 0.0319 | 0.0618 |
| 0.80 | 0.0000 | 0.0000 | 0.0000 |
| 0.90 | 0.0000 | 0.0000 | 0.0000 |

## Plots

Siehe `results/plots/`: reliability_diagram, confidence_histogram, confusion_matrix, roc_curve, precision_recall_curve, gradcam, misclassified.
