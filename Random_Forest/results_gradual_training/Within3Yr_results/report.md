# Gradual training report — Within3Yr

## 1. Setup

- Cohort: n=94 (positives=36, negatives=58, prevalence=0.383).
- Feature ranking source: `Random_Forest/results/Within3Yr/final_selected_features.csv` (sorted by Random-Forest impurity importance).
- Number of ranked features: **48**.
- Sweep range: k = 1 … 48 (step 1), 48 models trained.
- Hyperparameter strategy: **per_fold_grid**.
- Outer CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) — 15 held-out folds per k.
- Inner CV: StratifiedKFold(n_splits=5), scoring=roc_auc.
- Random seed: 42. Wall time: 445.0s.

## 2. Peak performance per metric

Each row reports the feature count that maximised (or minimised, for Brier) the per-fold mean of the metric. `Pooled @ k*` is the metric computed on the concatenated out-of-fold predictions at that same k.

| Metric | Best k | Mean ± SD across folds | Pooled @ k* |
|---|---:|---|---:|
| ROC AUC (max) | 34 | 0.898 ± 0.064 | 0.887 |
| Average Precision (max) | 23 | 0.855 ± 0.073 | 0.830 |
| Balanced Accuracy (max) | 23 | 0.752 ± 0.077 | 0.754 |
| F1 Score (max) | 24 | 0.683 ± 0.100 | 0.688 |
| Sensitivity (TPR) (max) | 6 | 0.656 ± 0.160 | 0.657 |
| Specificity (TNR) (max) | 47 | 0.929 ± 0.085 | 0.931 |
| Precision (max) | 40 | 0.848 ± 0.163 | 0.817 |
| Accuracy (max) | 23 | 0.787 ± 0.070 | 0.787 |
| Brier Score (min) | 24 | 0.150 ± 0.020 | 0.150 |

## 3. Top-ranked features (used in the sweep)

These are the features that the published main-pipeline model selected via stability selection on the full data, ranked by Random-Forest impurity importance. The k-th model in the sweep uses the first k of them.

| Rank | Feature | Stability frequency | RF importance |
|---:|---|---:|---:|
| 1 | `BTN3A2` | 0.71 | 0.0563 |
| 2 | `LAG3` | 1.00 | 0.0530 |
| 3 | `186_MMP-3` | 0.96 | 0.0497 |
| 4 | `179_EGFR` | 0.83 | 0.0480 |
| 5 | `CD83` | 0.96 | 0.0461 |
| 6 | `103_ADM` | 0.65 | 0.0415 |
| 7 | `185_CTSZ` | 0.93 | 0.0401 |
| 8 | `FCRL6` | 0.80 | 0.0350 |
| 9 | `BCAN` | 0.98 | 0.0343 |
| 10 | `MAPT` | 0.73 | 0.0338 |
| 11 | `HGF` | 0.66 | 0.0307 |
| 12 | `CD200R1` | 0.98 | 0.0295 |
| 13 | `138_CTRC` | 0.62 | 0.0277 |
| 14 | `157_PRELP` | 0.91 | 0.0275 |
| 15 | `188_ICAM-2` | 0.97 | 0.0274 |
| 16 | `CTSC` | 0.76 | 0.0255 |
| 17 | `CD200` | 0.95 | 0.0236 |
| 18 | `WFIKKN1` | 0.88 | 0.0235 |
| 19 | `CST5` | 0.67 | 0.0229 |
| 20 | `164_CTSD` | 0.68 | 0.0197 |
| 21 | `PSG1` | 0.97 | 0.0195 |
| 22 | `PPP3R1` | 0.69 | 0.0185 |
| 23 | `LY75` | 0.95 | 0.0182 |
| 24 | `HMOX2` | 0.69 | 0.0173 |
| 25 | `170_DCN` | 0.69 | 0.0154 |
| 26 | `102_LDL receptor` | 0.62 | 0.0143 |
| 27 | `FGF-5` | 0.74 | 0.0141 |
| 28 | `Flt3L` | 0.74 | 0.0141 |
| 29 | `AOC1` | 0.65 | 0.0140 |
| 30 | `145_PAPPA` | 0.84 | 0.0128 |
| … | _18 more_ | | |

## 4. Files in this directory

- `metrics_per_k.csv` — one row per k with mean/std across folds + pooled metrics.
- `cv_fold_metrics_per_k.csv` — long format, one row per (k, fold).
- `cv_predictions_per_k.csv` — every OOF prediction at every k.
- `best_hyperparameters_per_k.csv` — inner-CV winning hyperparameters.
- `final_selected_features_used.csv` — exact feature ranking that drove the sweep.
- `peak_summary.json` — machine-readable peak-per-metric table.
- `plots/` — every metric vs. k, multi-panel summary, per-fold curves, heatmaps, peak-marker bar chart, feature-ranking bar chart, ROC overlay at selected k.
