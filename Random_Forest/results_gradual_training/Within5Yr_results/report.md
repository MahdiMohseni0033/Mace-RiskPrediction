# Gradual training report — Within5Yr

## 1. Setup

- Cohort: n=94 (positives=47, negatives=47, prevalence=0.500).
- Feature ranking source: `Random_Forest/results/Within5Yr/final_selected_features.csv` (sorted by Random-Forest impurity importance).
- Number of ranked features: **46**.
- Sweep range: k = 1 … 46 (step 1), 46 models trained.
- Hyperparameter strategy: **per_fold_grid**.
- Outer CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) — 15 held-out folds per k.
- Inner CV: StratifiedKFold(n_splits=5), scoring=roc_auc.
- Random seed: 42. Wall time: 426.4s.

## 2. Peak performance per metric

Each row reports the feature count that maximised (or minimised, for Brier) the per-fold mean of the metric. `Pooled @ k*` is the metric computed on the concatenated out-of-fold predictions at that same k.

| Metric | Best k | Mean ± SD across folds | Pooled @ k* |
|---|---:|---|---:|
| ROC AUC (max) | 46 | 0.878 ± 0.067 | 0.869 |
| Average Precision (max) | 46 | 0.905 ± 0.052 | 0.889 |
| Balanced Accuracy (max) | 8 | 0.804 ± 0.098 | 0.805 |
| F1 Score (max) | 8 | 0.807 ± 0.097 | 0.808 |
| Sensitivity (TPR) (max) | 8 | 0.821 ± 0.132 | 0.823 |
| Specificity (TNR) (max) | 33 | 0.833 ± 0.158 | 0.830 |
| Precision (max) | 39 | 0.838 ± 0.138 | 0.815 |
| Accuracy (max) | 8 | 0.806 ± 0.096 | 0.805 |
| Brier Score (min) | 8 | 0.150 ± 0.034 | 0.150 |

## 3. Top-ranked features (used in the sweep)

These are the features that the published main-pipeline model selected via stability selection on the full data, ranked by Random-Forest impurity importance. The k-th model in the sweep uses the first k of them.

| Rank | Feature | Stability frequency | RF importance |
|---:|---|---:|---:|
| 1 | `162_U-PAR` | 1.00 | 0.0726 |
| 2 | `BTN3A2` | 0.68 | 0.0541 |
| 3 | `NAAA` | 0.91 | 0.0518 |
| 4 | `143_RETN` | 0.83 | 0.0350 |
| 5 | `MCP-3` | 0.86 | 0.0347 |
| 6 | `179_EGFR` | 0.85 | 0.0340 |
| 7 | `167_MMP-7` | 0.64 | 0.0334 |
| 8 | `IL-5R-alpha` | 0.94 | 0.0334 |
| 9 | `BCAN` | 0.97 | 0.0301 |
| 10 | `177_FABP2` | 1.00 | 0.0278 |
| 11 | `PSG1` | 1.00 | 0.0260 |
| 12 | `140_SPON2` | 0.63 | 0.0260 |
| 13 | `CST5` | 0.77 | 0.0256 |
| 14 | `185_CTSZ` | 0.68 | 0.0253 |
| 15 | `186_MMP-3` | 0.84 | 0.0240 |
| 16 | `CD200R1` | 0.78 | 0.0238 |
| 17 | `LAG3` | 0.94 | 0.0238 |
| 18 | `116_IDUA` | 0.71 | 0.0237 |
| 19 | `WFIKKN1` | 0.73 | 0.0222 |
| 20 | `159_XCL1` | 0.79 | 0.0217 |
| 21 | `CD83` | 0.96 | 0.0213 |
| 22 | `CTF1` | 0.69 | 0.0212 |
| 23 | `DSG3` | 0.98 | 0.0210 |
| 24 | `170_DCN` | 0.67 | 0.0204 |
| 25 | `188_ICAM-2` | 0.93 | 0.0187 |
| 26 | `148_TR-AP` | 0.85 | 0.0187 |
| 27 | `RPS6KB1` | 0.76 | 0.0183 |
| 28 | `CCL20` | 0.71 | 0.0181 |
| 29 | `SH2D1A` | 1.00 | 0.0178 |
| 30 | `CXCL12` | 0.78 | 0.0176 |
| … | _16 more_ | | |

## 4. Files in this directory

- `metrics_per_k.csv` — one row per k with mean/std across folds + pooled metrics.
- `cv_fold_metrics_per_k.csv` — long format, one row per (k, fold).
- `cv_predictions_per_k.csv` — every OOF prediction at every k.
- `best_hyperparameters_per_k.csv` — inner-CV winning hyperparameters.
- `final_selected_features_used.csv` — exact feature ranking that drove the sweep.
- `peak_summary.json` — machine-readable peak-per-metric table.
- `plots/` — every metric vs. k, multi-panel summary, per-fold curves, heatmaps, peak-marker bar chart, feature-ranking bar chart, ROC overlay at selected k.
