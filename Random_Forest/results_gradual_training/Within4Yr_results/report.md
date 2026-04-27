# Gradual training report — Within4Yr

## 1. Setup

- Cohort: n=94 (positives=43, negatives=51, prevalence=0.457).
- Feature ranking source: `Random_Forest/results/Within4Yr/final_selected_features.csv` (sorted by Random-Forest impurity importance).
- Number of ranked features: **38**.
- Sweep range: k = 1 … 38 (step 1), 38 models trained.
- Hyperparameter strategy: **per_fold_grid**.
- Outer CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) — 15 held-out folds per k.
- Inner CV: StratifiedKFold(n_splits=5), scoring=roc_auc.
- Random seed: 42. Wall time: 350.2s.

## 2. Peak performance per metric

Each row reports the feature count that maximised (or minimised, for Brier) the per-fold mean of the metric. `Pooled @ k*` is the metric computed on the concatenated out-of-fold predictions at that same k.

| Metric | Best k | Mean ± SD across folds | Pooled @ k* |
|---|---:|---|---:|
| ROC AUC (max) | 16 | 0.892 ± 0.048 | 0.887 |
| Average Precision (max) | 15 | 0.902 ± 0.046 | 0.891 |
| Balanced Accuracy (max) | 17 | 0.813 ± 0.085 | 0.813 |
| F1 Score (max) | 17 | 0.785 ± 0.106 | 0.790 |
| Sensitivity (TPR) (max) | 17 | 0.743 ± 0.133 | 0.744 |
| Specificity (TNR) (max) | 27 | 0.890 ± 0.097 | 0.889 |
| Precision (max) | 27 | 0.856 ± 0.120 | 0.840 |
| Accuracy (max) | 17 | 0.819 ± 0.082 | 0.819 |
| Brier Score (min) | 16 | 0.148 ± 0.021 | 0.148 |

## 3. Top-ranked features (used in the sweep)

These are the features that the published main-pipeline model selected via stability selection on the full data, ranked by Random-Forest impurity importance. The k-th model in the sweep uses the first k of them.

| Rank | Feature | Stability frequency | RF importance |
|---:|---|---:|---:|
| 1 | `BTN3A2` | 0.94 | 0.0905 |
| 2 | `162_U-PAR` | 0.96 | 0.0607 |
| 3 | `CD83` | 1.00 | 0.0497 |
| 4 | `MCP-3` | 0.97 | 0.0456 |
| 5 | `143_RETN` | 0.67 | 0.0383 |
| 6 | `179_EGFR` | 0.72 | 0.0383 |
| 7 | `TGF-alpha` | 0.71 | 0.0380 |
| 8 | `159_XCL1` | 0.74 | 0.0355 |
| 9 | `NAAA` | 0.86 | 0.0354 |
| 10 | `CST5` | 0.75 | 0.0338 |
| 11 | `188_ICAM-2` | 1.00 | 0.0330 |
| 12 | `LAG3` | 0.98 | 0.0320 |
| 13 | `157_PRELP` | 0.89 | 0.0316 |
| 14 | `IL-5R-alpha` | 0.86 | 0.0297 |
| 15 | `CD200R1` | 0.95 | 0.0295 |
| 16 | `148_TR-AP` | 0.95 | 0.0288 |
| 17 | `196_CCL16` | 0.80 | 0.0284 |
| 18 | `BCAN` | 0.93 | 0.0257 |
| 19 | `PSG1` | 0.96 | 0.0248 |
| 20 | `WFIKKN1` | 0.67 | 0.0227 |
| 21 | `CD63` | 0.81 | 0.0217 |
| 22 | `SH2D1A` | 0.96 | 0.0202 |
| 23 | `170_DCN` | 0.69 | 0.0201 |
| 24 | `ITGB6` | 0.71 | 0.0198 |
| 25 | `CCL20` | 0.77 | 0.0192 |
| 26 | `DSG3` | 0.89 | 0.0191 |
| 27 | `LY75` | 0.86 | 0.0175 |
| 28 | `IRF9` | 0.68 | 0.0159 |
| 29 | `PTS` | 0.63 | 0.0157 |
| 30 | `128_TLT-2` | 0.72 | 0.0147 |
| … | _8 more_ | | |

## 4. Files in this directory

- `metrics_per_k.csv` — one row per k with mean/std across folds + pooled metrics.
- `cv_fold_metrics_per_k.csv` — long format, one row per (k, fold).
- `cv_predictions_per_k.csv` — every OOF prediction at every k.
- `best_hyperparameters_per_k.csv` — inner-CV winning hyperparameters.
- `final_selected_features_used.csv` — exact feature ranking that drove the sweep.
- `peak_summary.json` — machine-readable peak-per-metric table.
- `plots/` — every metric vs. k, multi-panel summary, per-fold curves, heatmaps, peak-marker bar chart, feature-ranking bar chart, ROC overlay at selected k.
