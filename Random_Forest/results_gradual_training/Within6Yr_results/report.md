# Gradual training report — Within6Yr

## 1. Setup

- Cohort: n=94 (positives=53, negatives=41, prevalence=0.564).
- Feature ranking source: `Random_Forest/results/Within6Yr/final_selected_features.csv` (sorted by Random-Forest impurity importance).
- Number of ranked features: **50**.
- Sweep range: k = 1 … 50 (step 1), 50 models trained.
- Hyperparameter strategy: **per_fold_grid**.
- Outer CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=3) — 15 held-out folds per k.
- Inner CV: StratifiedKFold(n_splits=5), scoring=roc_auc.
- Random seed: 42. Wall time: 466.3s.

## 2. Peak performance per metric

Each row reports the feature count that maximised (or minimised, for Brier) the per-fold mean of the metric. `Pooled @ k*` is the metric computed on the concatenated out-of-fold predictions at that same k.

| Metric | Best k | Mean ± SD across folds | Pooled @ k* |
|---|---:|---|---:|
| ROC AUC (max) | 41 | 0.876 ± 0.093 | 0.868 |
| Average Precision (max) | 41 | 0.920 ± 0.063 | 0.904 |
| Balanced Accuracy (max) | 13 | 0.796 ± 0.112 | 0.796 |
| F1 Score (max) | 13 | 0.824 ± 0.105 | 0.826 |
| Sensitivity (TPR) (max) | 32 | 0.844 ± 0.137 | 0.843 |
| Specificity (TNR) (max) | 13 | 0.756 ± 0.136 | 0.756 |
| Precision (max) | 13 | 0.818 ± 0.099 | 0.816 |
| Accuracy (max) | 13 | 0.801 ± 0.113 | 0.801 |
| Brier Score (min) | 11 | 0.154 ± 0.035 | 0.153 |

## 3. Top-ranked features (used in the sweep)

These are the features that the published main-pipeline model selected via stability selection on the full data, ranked by Random-Forest impurity importance. The k-th model in the sweep uses the first k of them.

| Rank | Feature | Stability frequency | RF importance |
|---:|---|---:|---:|
| 1 | `143_RETN` | 0.87 | 0.0715 |
| 2 | `107_PlGF` | 0.95 | 0.0667 |
| 3 | `162_U-PAR` | 0.72 | 0.0466 |
| 4 | `SKR3` | 0.72 | 0.0415 |
| 5 | `179_EGFR` | 0.97 | 0.0378 |
| 6 | `ATP6V1F` | 0.90 | 0.0355 |
| 7 | `159_XCL1` | 0.93 | 0.0337 |
| 8 | `116_IDUA` | 0.95 | 0.0287 |
| 9 | `177_FABP2` | 0.98 | 0.0287 |
| 10 | `103_ADM` | 0.64 | 0.0282 |
| 11 | `IL-20RA` | 0.98 | 0.0271 |
| 12 | `DFFA` | 0.84 | 0.0263 |
| 13 | `BCAN` | 0.84 | 0.0249 |
| 14 | `140_SPON2` | 0.93 | 0.0248 |
| 15 | `CST5` | 0.78 | 0.0238 |
| 16 | `HGF` | 0.63 | 0.0225 |
| 17 | `170_DCN` | 0.70 | 0.0212 |
| 18 | `CD200R1` | 0.89 | 0.0211 |
| 19 | `184_PON3` | 0.69 | 0.0211 |
| 20 | `PSG1` | 0.99 | 0.0198 |
| 21 | `CLEC4D` | 0.64 | 0.0197 |
| 22 | `167_MMP-7` | 0.62 | 0.0185 |
| 23 | `IL10` | 0.80 | 0.0174 |
| 24 | `MCP-3` | 0.71 | 0.0160 |
| 25 | `CLM-6` | 0.68 | 0.0160 |
| 26 | `169_IL-1RT2` | 0.83 | 0.0156 |
| 27 | `185_CTSZ` | 0.78 | 0.0150 |
| 28 | `148_TR-AP` | 0.86 | 0.0150 |
| 29 | `Alpha-2-MRAP` | 0.91 | 0.0149 |
| 30 | `RPS6KB1` | 0.93 | 0.0144 |
| … | _20 more_ | | |

## 4. Files in this directory

- `metrics_per_k.csv` — one row per k with mean/std across folds + pooled metrics.
- `cv_fold_metrics_per_k.csv` — long format, one row per (k, fold).
- `cv_predictions_per_k.csv` — every OOF prediction at every k.
- `best_hyperparameters_per_k.csv` — inner-CV winning hyperparameters.
- `final_selected_features_used.csv` — exact feature ranking that drove the sweep.
- `peak_summary.json` — machine-readable peak-per-metric table.
- `plots/` — every metric vs. k, multi-panel summary, per-fold curves, heatmaps, peak-marker bar chart, feature-ranking bar chart, ROC overlay at selected k.
