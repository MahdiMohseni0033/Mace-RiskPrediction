# Gradual training report — Within7Yr

## 1. Setup

- Cohort: n=94 (positives=56, negatives=38, prevalence=0.596).
- Feature ranking source: `Random_Forest/results/Within7Yr/final_selected_features.csv` (sorted by Random-Forest impurity importance).
- Number of ranked features: **49**.
- Sweep range: k = 1 … 49 (step 1), 49 models trained.
- Hyperparameter strategy: **per_fold_grid**.
- Outer CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) — 15 held-out folds per k.
- Inner CV: StratifiedKFold(n_splits=5), scoring=roc_auc.
- Random seed: 42. Wall time: 454.4s.

## 2. Peak performance per metric

Each row reports the feature count that maximised (or minimised, for Brier) the per-fold mean of the metric. `Pooled @ k*` is the metric computed on the concatenated out-of-fold predictions at that same k.

| Metric | Best k | Mean ± SD across folds | Pooled @ k* |
|---|---:|---|---:|
| ROC AUC (max) | 26 | 0.898 ± 0.054 | 0.888 |
| Average Precision (max) | 21 | 0.936 ± 0.031 | 0.930 |
| Balanced Accuracy (max) | 19 | 0.810 ± 0.046 | 0.810 |
| F1 Score (max) | 46 | 0.863 ± 0.049 | 0.863 |
| Sensitivity (TPR) (max) | 46 | 0.934 ± 0.054 | 0.935 |
| Specificity (TNR) (max) | 20 | 0.746 ± 0.099 | 0.746 |
| Precision (max) | 20 | 0.837 ± 0.055 | 0.834 |
| Accuracy (max) | 19 | 0.826 ± 0.042 | 0.826 |
| Brier Score (min) | 14 | 0.138 ± 0.019 | 0.138 |

## 3. Top-ranked features (used in the sweep)

These are the features that the published main-pipeline model selected via stability selection on the full data, ranked by Random-Forest impurity importance. The k-th model in the sweep uses the first k of them.

| Rank | Feature | Stability frequency | RF importance |
|---:|---|---:|---:|
| 1 | `135_GDF-15` | 0.97 | 0.0610 |
| 2 | `CKAP4` | 0.86 | 0.0609 |
| 3 | `162_U-PAR` | 0.85 | 0.0550 |
| 4 | `179_EGFR` | 0.98 | 0.0510 |
| 5 | `114_CSTB` | 0.77 | 0.0486 |
| 6 | `ATP6V1F` | 0.99 | 0.0421 |
| 7 | `IFI30` | 0.83 | 0.0415 |
| 8 | `BCAN` | 0.98 | 0.0324 |
| 9 | `107_PlGF` | 0.81 | 0.0312 |
| 10 | `177_FABP2` | 1.00 | 0.0257 |
| 11 | `159_XCL1` | 0.87 | 0.0237 |
| 12 | `139_FGF-23` | 0.68 | 0.0228 |
| 13 | `FLRT2` | 0.61 | 0.0225 |
| 14 | `NRP2` | 0.66 | 0.0195 |
| 15 | `PSG1` | 0.97 | 0.0190 |
| 16 | `116_IDUA` | 0.93 | 0.0184 |
| 17 | `ANXA10` | 0.77 | 0.0175 |
| 18 | `CD200R1` | 0.83 | 0.0174 |
| 19 | `TNFRSF13C` | 0.67 | 0.0172 |
| 20 | `RPS6KB1` | 0.93 | 0.0170 |
| 21 | `167_MMP-7` | 0.71 | 0.0170 |
| 22 | `IL-10RB` | 0.61 | 0.0168 |
| 23 | `Alpha-2-MRAP` | 0.95 | 0.0163 |
| 24 | `140_SPON2` | 0.68 | 0.0163 |
| 25 | `IL10` | 0.81 | 0.0161 |
| 26 | `IL-17C` | 0.73 | 0.0161 |
| 27 | `DPEP1` | 0.69 | 0.0159 |
| 28 | `WFIKKN1` | 0.77 | 0.0150 |
| 29 | `169_IL-1RT2` | 0.83 | 0.0143 |
| 30 | `DFFA` | 0.61 | 0.0134 |
| … | _19 more_ | | |

## 4. Files in this directory

- `metrics_per_k.csv` — one row per k with mean/std across folds + pooled metrics.
- `cv_fold_metrics_per_k.csv` — long format, one row per (k, fold).
- `cv_predictions_per_k.csv` — every OOF prediction at every k.
- `best_hyperparameters_per_k.csv` — inner-CV winning hyperparameters.
- `final_selected_features_used.csv` — exact feature ranking that drove the sweep.
- `peak_summary.json` — machine-readable peak-per-metric table.
- `plots/` — every metric vs. k, multi-panel summary, per-fold curves, heatmaps, peak-marker bar chart, feature-ranking bar chart, ROC overlay at selected k.
