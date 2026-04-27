# Gradual Training — MACE Risk Prediction

This document explains the **gradual training** analysis added on top of the main Random-Forest pipeline and reports its findings across all configured outcome horizons.

## 1. Motivation

The main pipeline (`Random_Forest/src/train_rf.py`) selects between roughly 30 and 50 features per horizon via stability selection and trains a Random Forest on those. A natural follow-up question for a publication is: **how many of those features are actually doing the work?** If the curve of performance vs. feature count plateaus (or peaks) early, the published model could be slimmed down, both for biological interpretability and to reduce the chance that the larger feature set is over-fitting noise on n=94 patients. This analysis answers that question with a held-out, cross-validated sweep.

## 2. Method

### 2.1 Inputs

For every configured horizon `Within{k}Yr` we read the published feature ranking from

```
Random_Forest/results/Within{k}Yr/final_selected_features.csv
```

produced by the main pipeline. The file lists every feature that survived stability selection on the full dataset, with two scores: `stability_frequency` (the L1-logistic stability-selection score, Meinshausen & Bühlmann 2010) and `rf_importance` (mean decrease in impurity from the final Random Forest fitted on all n=94 patients with the most frequent winning hyperparameters from the outer CV). **We sort by `rf_importance` (descending)** and call this the *feature ranking*.

### 2.2 The sweep

Let N be the size of the feature ranking for that outcome. For k = 1, 2, …, N we train a separate Random-Forest classifier using only the top-k features. Each training run uses the **same nested cross-validation protocol as the main pipeline**:

- Outer loop: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` — 15   held-out estimates per k.
- Inner loop: `StratifiedKFold(n_splits=5)` for hyperparameter tuning.
- Random seed: 42 (fixed end-to-end).

Class imbalance is handled by `class_weight="balanced"` inside the Random Forest, exactly as in the main pipeline.

### 2.3 Hyperparameter tuning

Two strategies are supported in `config.py` (`HYPERPARAM_STRATEGY`):

1. **`per_fold_grid` (default).** Inside each outer fold, run a    `GridSearchCV` over the same Random-Forest grid used by the main pipeline    (`n_estimators ∈ {300}`, `max_depth ∈ {None, 5}`,    `min_samples_leaf ∈ {1, 3}`, `max_features ∈ {"sqrt", 0.3}`), scored on    `roc_auc`, on the 5-fold stratified inner CV. The best estimator is    refit and used to score the held-out outer fold. This is the most    rigorous option and matches the main pipeline exactly.
2. **`fixed_best`.** Reuse the most-frequent winning combination from the    main pipeline's `best_hyperparameters.json` — no inner-CV search. Useful    for isolating the feature-count effect from hyperparameter variance, and    considerably faster.

Per-fold winning hyperparameters at every k are saved to `best_hyperparameters_per_k.csv` so the choices can be inspected.

### 2.4 Why use the full-data feature ranking, not refit it inside each fold?

Strictly speaking, computing the feature ordering on the full dataset before the outer CV introduces a small amount of look-ahead — the ordering is informed by every patient's outcome. We use this ordering deliberately, for two reasons. First, this is the *published* ordering: a clinician asking "do I need to measure all 50 of these proteins, or just the top five?" is asking about that exact list. Second, we are not making a generalisation claim about the ranking itself; for each k the *model* is still trained and evaluated under fold-respecting nested CV, so the k-vs-performance curve is a held-out estimate. The interaction is mild — at small k the ranking has almost no leverage to memorise individual patients — but it should be interpreted as **"how performance varies as we walk down the published ranking"**, not as fully nested feature selection.

### 2.5 Metrics recorded

At every k we record per-fold and pooled values of:

- ROC AUC, Average Precision (PR AUC),
- Balanced accuracy, accuracy, F1, precision, recall,
- Sensitivity (TPR) and specificity (TNR) at threshold 0.5,
- Brier score (lower is better),
- Pooled balanced accuracy and F1 at the Youden-optimal threshold.

All metric definitions are imported from the main pipeline's `metrics_utils.py` to guarantee identical maths across the two analyses.

### 2.6 Definition of "best k"

For each metric we report the k at which the **mean across the 15 outer folds** is best (highest, except for Brier where lowest). Per-fold standard deviation, the pooled OOF value at that same k, and the inner-CV winning hyperparameters are stored alongside.

## 3. Results

### 3.1 Cross-outcome summary

| Outcome | n | prev. | total feats | Best k (AUC) | Best AUC | Best k (BalAcc) | Best BalAcc | Best k (F1) | Best F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Within1Yr | 94 | 0.18 | 33 | 13 | 0.867 | 6 | 0.723 | 14 | 0.511 |
| Within2Yr | 94 | 0.34 | 50 | 42 | 0.878 | 3 | 0.748 | 3 | 0.650 |
| Within3Yr | 94 | 0.38 | 48 | 34 | 0.898 | 23 | 0.752 | 24 | 0.683 |
| Within4Yr | 94 | 0.46 | 38 | 16 | 0.892 | 17 | 0.813 | 17 | 0.785 |
| Within5Yr | 94 | 0.50 | 46 | 46 | 0.878 | 8 | 0.804 | 8 | 0.807 |
| Within6Yr | 94 | 0.56 | 50 | 41 | 0.876 | 13 | 0.796 | 13 | 0.824 |
| Within7Yr | 94 | 0.60 | 49 | 26 | 0.898 | 19 | 0.810 | 46 | 0.863 |

### 3.2 Per-outcome detail

Each `Within{k}Yr_results/` folder contains:

- `report.md` — same numbers as above, plus the actual top-30 features used.
- `metrics_per_k.csv` and `cv_fold_metrics_per_k.csv` — every metric,   every k, every fold.
- `cv_predictions_per_k.csv` — every patient's OOF probability at every k.
- `best_hyperparameters_per_k.csv` — inner-CV winners at every (k, fold).
- `peak_summary.json` — machine-readable peak-per-metric.
- `plots/` — the 20+ plots described below.

### 3.3 Plots produced (per outcome)

- `{metric}_vs_n_features.png` — for every metric, mean ± SD band, pooled   curve overlay, peak marker. Ten metrics × one plot each.
- `all_metrics_vs_n_features.png` — eight-panel summary on one figure.
- `per_fold_{metric}_vs_n_features.png` — every outer fold's curve in grey,   mean curve in red. Shows fold-to-fold variability.
- `heatmap_{metric}_per_fold.png` — fold × k heatmap; colour shows where the   signal is concentrated.
- `peak_marker_summary.png` — bar chart of best k per metric.
- `feature_ranking_used.png` — the top-30 features fed into the sweep.
- `roc_pooled_selected_k.png` — pooled ROC at k=1, peak-AUC k, and full k.

## 4. How to reproduce

```bash
# all configured outcomes (see Random_Forest/src_gradual_training/config.py)
.venv/bin/python Random_Forest/src_gradual_training/gradual_train.py

# subset
.venv/bin/python Random_Forest/src_gradual_training/gradual_train.py --outcomes Within3Yr Within5Yr
```

Total runtime depends on `HYPERPARAM_STRATEGY` and the number of features in each ranking. With `per_fold_grid` and the default grid, expect a few minutes per outcome on a modern multi-core machine.

## 5. Caveats

- **n = 94 is small.** The k-vs-performance curves are bumpy because each   point is the mean of only 15 outer-fold estimates. Standard-deviation   bands are wide; a difference of 0.02 AUC between two adjacent k values   should not be over-interpreted.
- **Feature ordering uses full-data RF importance.** See §2.4. The intent is   to characterise the published ranking, not to perform fully nested   selection.
- **No external validation.** The same caveats from `Report.md` apply.
