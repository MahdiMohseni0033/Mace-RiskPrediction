# Gradual training report — Within1Yr

## 1. Setup

- Cohort: n=94 (positives=17, negatives=77, prevalence=0.181).
- Feature ranking source: `Random_Forest/results/Within1Yr/final_selected_features.csv` (sorted by Random-Forest impurity importance).
- Number of ranked features: **33**.
- Sweep range: k = 1 … 33 (step 1), 33 models trained.
- Hyperparameter strategy: **per_fold_grid**.
- Outer CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) — 15 held-out folds per k.
- Inner CV: StratifiedKFold(n_splits=5), scoring=roc_auc.
- Random seed: 42. Wall time: 299.2s.

## 2. Peak performance per metric

Each row reports the feature count that maximised (or minimised, for Brier) the per-fold mean of the metric. `Pooled @ k*` is the metric computed on the concatenated out-of-fold predictions at that same k.

| Metric | Best k | Mean ± SD across folds | Pooled @ k* |
|---|---:|---|---:|
| ROC AUC (max) | 13 | 0.867 ± 0.109 | 0.830 |
| Average Precision (max) | 13 | 0.711 ± 0.196 | 0.615 |
| Balanced Accuracy (max) | 6 | 0.723 ± 0.161 | 0.721 |
| F1 Score (max) | 14 | 0.511 ± 0.219 | 0.517 |
| Sensitivity (TPR) (max) | 1 | 0.583 ± 0.317 | 0.588 |
| Specificity (TNR) (max) | 33 | 0.987 ± 0.037 | 0.987 |
| Precision (max) | 15 | 0.713 ± 0.328 | 0.600 |
| Accuracy (max) | 19 | 0.855 ± 0.073 | 0.855 |
| Brier Score (min) | 31 | 0.109 ± 0.028 | 0.109 |

## 3. Top-ranked features (used in the sweep)

These are the features that the published main-pipeline model selected via stability selection on the full data, ranked by Random-Forest impurity importance. The k-th model in the sweep uses the first k of them.

| Rank | Feature | Stability frequency | RF importance |
|---:|---|---:|---:|
| 1 | `162_U-PAR` | 0.85 | 0.0676 |
| 2 | `LAMP3` | 0.84 | 0.0596 |
| 3 | `153_THBS2` | 0.71 | 0.0572 |
| 4 | `123_LTBR` | 0.85 | 0.0495 |
| 5 | `EZR` | 0.74 | 0.0469 |
| 6 | `LILRB4` | 0.80 | 0.0464 |
| 7 | `LAG3` | 0.91 | 0.0426 |
| 8 | `181_BNP` | 0.71 | 0.0424 |
| 9 | `CST5` | 0.80 | 0.0393 |
| 10 | `CDH15` | 0.79 | 0.0363 |
| 11 | `185_CTSZ` | 0.61 | 0.0359 |
| 12 | `FCRL6` | 0.65 | 0.0328 |
| 13 | `159_TNFSF13B` | 0.68 | 0.0302 |
| 14 | `NEFL` | 0.99 | 0.0286 |
| 15 | `KLRD1` | 0.60 | 0.0278 |
| 16 | `IL6` | 0.66 | 0.0259 |
| 17 | `108_ADAM-TS13` | 0.61 | 0.0249 |
| 18 | `PD-L1` | 0.74 | 0.0245 |
| 19 | `148_TR-AP` | 0.88 | 0.0242 |
| 20 | `184_PD-L2` | 0.77 | 0.0240 |
| 21 | `138_DLK-1` | 0.68 | 0.0214 |
| 22 | `LY75` | 0.83 | 0.0213 |
| 23 | `sFRP-3` | 0.70 | 0.0212 |
| 24 | `124_IL1RL2` | 0.64 | 0.0211 |
| 25 | `156_MMP-2` | 0.76 | 0.0206 |
| 26 | `TGF-alpha` | 0.65 | 0.0200 |
| 27 | `DUSP3` | 0.76 | 0.0192 |
| 28 | `169_IL-1RT2` | 0.71 | 0.0190 |
| 29 | `RNF31` | 0.81 | 0.0184 |
| 30 | `HNMT` | 0.70 | 0.0178 |
| … | _3 more_ | | |

## 4. Files in this directory

- `metrics_per_k.csv` — one row per k with mean/std across folds + pooled metrics.
- `cv_fold_metrics_per_k.csv` — long format, one row per (k, fold).
- `cv_predictions_per_k.csv` — every OOF prediction at every k.
- `best_hyperparameters_per_k.csv` — inner-CV winning hyperparameters.
- `final_selected_features_used.csv` — exact feature ranking that drove the sweep.
- `peak_summary.json` — machine-readable peak-per-metric table.
- `plots/` — every metric vs. k, multi-panel summary, per-fold curves, heatmaps, peak-marker bar chart, feature-ranking bar chart, ROC overlay at selected k.
