# Gradual training report — Within2Yr

## 1. Setup

- Cohort: n=94 (positives=32, negatives=62, prevalence=0.340).
- Feature ranking source: `Random_Forest/results/Within2Yr/final_selected_features.csv` (sorted by Random-Forest impurity importance).
- Number of ranked features: **50**.
- Sweep range: k = 1 … 50 (step 1), 50 models trained.
- Hyperparameter strategy: **per_fold_grid**.
- Outer CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) — 15 held-out folds per k.
- Inner CV: StratifiedKFold(n_splits=5), scoring=roc_auc.
- Random seed: 42. Wall time: 463.5s.

## 2. Peak performance per metric

Each row reports the feature count that maximised (or minimised, for Brier) the per-fold mean of the metric. `Pooled @ k*` is the metric computed on the concatenated out-of-fold predictions at that same k.

| Metric | Best k | Mean ± SD across folds | Pooled @ k* |
|---|---:|---|---:|
| ROC AUC (max) | 42 | 0.878 ± 0.082 | 0.869 |
| Average Precision (max) | 20 | 0.801 ± 0.133 | 0.769 |
| Balanced Accuracy (max) | 3 | 0.748 ± 0.117 | 0.750 |
| F1 Score (max) | 3 | 0.650 ± 0.168 | 0.670 |
| Sensitivity (TPR) (max) | 4 | 0.660 ± 0.266 | 0.667 |
| Specificity (TNR) (max) | 46 | 0.941 ± 0.058 | 0.941 |
| Precision (max) | 39 | 0.790 ± 0.218 | 0.783 |
| Accuracy (max) | 14 | 0.790 ± 0.104 | 0.791 |
| Brier Score (min) | 13 | 0.145 ± 0.040 | 0.144 |

## 3. Top-ranked features (used in the sweep)

These are the features that the published main-pipeline model selected via stability selection on the full data, ranked by Random-Forest impurity importance. The k-th model in the sweep uses the first k of them.

| Rank | Feature | Stability frequency | RF importance |
|---:|---|---:|---:|
| 1 | `SKR3` | 0.86 | 0.0615 |
| 2 | `103_ADM` | 0.73 | 0.0434 |
| 3 | `LILRB4` | 0.76 | 0.0426 |
| 4 | `MAPT` | 0.83 | 0.0409 |
| 5 | `162_U-PAR` | 0.74 | 0.0394 |
| 6 | `185_CTSZ` | 0.96 | 0.0381 |
| 7 | `EDA2R` | 0.64 | 0.0352 |
| 8 | `186_MMP-3` | 0.95 | 0.0346 |
| 9 | `CD6` | 0.61 | 0.0342 |
| 10 | `BTN3A2` | 0.80 | 0.0337 |
| 11 | `LAG3` | 0.95 | 0.0306 |
| 12 | `184_PON3` | 0.70 | 0.0303 |
| 13 | `153_THBS2` | 0.78 | 0.0264 |
| 14 | `IKZF2` | 0.96 | 0.0222 |
| 15 | `157_PRELP` | 0.95 | 0.0218 |
| 16 | `Flt3L` | 0.79 | 0.0215 |
| 17 | `CD200` | 0.84 | 0.0204 |
| 18 | `TGF-alpha` | 0.64 | 0.0192 |
| 19 | `BCAN` | 0.67 | 0.0176 |
| 20 | `145_PAPPA` | 0.95 | 0.0174 |
| 21 | `FCRL6` | 0.70 | 0.0169 |
| 22 | `SNCG` | 0.65 | 0.0169 |
| 23 | `190_CA5A` | 0.82 | 0.0168 |
| 24 | `CD200R1` | 0.76 | 0.0164 |
| 25 | `177_FABP2` | 0.62 | 0.0162 |
| 26 | `FGF-5` | 0.71 | 0.0162 |
| 27 | `164_CTSD` | 0.71 | 0.0159 |
| 28 | `HMOX2` | 0.61 | 0.0156 |
| 29 | `CXCL11` | 0.74 | 0.0147 |
| 30 | `170_DCN` | 0.66 | 0.0147 |
| … | _20 more_ | | |

## 4. Files in this directory

- `metrics_per_k.csv` — one row per k with mean/std across folds + pooled metrics.
- `cv_fold_metrics_per_k.csv` — long format, one row per (k, fold).
- `cv_predictions_per_k.csv` — every OOF prediction at every k.
- `best_hyperparameters_per_k.csv` — inner-CV winning hyperparameters.
- `final_selected_features_used.csv` — exact feature ranking that drove the sweep.
- `peak_summary.json` — machine-readable peak-per-metric table.
- `plots/` — every metric vs. k, multi-panel summary, per-fold curves, heatmaps, peak-marker bar chart, feature-ranking bar chart, ROC overlay at selected k.
