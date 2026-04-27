# CLAUDE.md — MACE Risk-Prediction Project

This file gives Claude Code (and future contributors) the project context, the
implementation expectations that have already been agreed, and the development
guidance that should be followed for every change.

---

## 1. Project context

**Goal.** Build, evaluate and document machine-learning models that predict the
occurrence of a Major Adverse Cardiovascular Event (MACE) within 1, 2, …, 7
years after the index visit, from a baseline proteomics panel plus a few
clinical features.

**Dataset.** `datasets/imputed_proteomics_base.csv`
- 94 patients (rows), 559 columns
- Identifier: `EuroSCOREPatient ID` (string, **never** used as a feature)
- Seven binary outcomes: `Within1Yr` … `Within7Yr` ∈ {0, 1}
- Remaining 551 numeric columns are candidate predictors (Olink panels +
  derived analytes)
- Already imputed; no further missing-value handling required

**Why this is hard.** Small-N, high-dimensional medical data (n ≪ p), with a
class imbalance that varies markedly by horizon (≈18 % positives at 1 year,
~50 % at 5–7 years). Statistical leakage and overfitting are the dominant
failure modes — every choice in the pipeline is made with that in mind.

**Output.** This work is intended for a peer-reviewed publication, so all
metrics, plots and selected-feature lists must be reproducible from the code
in this repository.

---

## 2. Repository layout

```
Mace-RiskPrediction/
├── datasets/                              # input data (do not modify)
│   └── imputed_proteomics_base.csv        # primary input
├── Random_Forest/                         # Random-Forest model family
│   ├── src/                               # main pipeline
│   │   ├── config.py                      # CV / selection / RF hyperparameters
│   │   ├── stability_selection.py         # L1-logistic stability selection
│   │   ├── metrics_utils.py               # metrics + plotting helpers
│   │   └── train_rf.py                    # main entry point
│   ├── results/                           # main pipeline outputs
│   │   ├── summary_metrics.csv            # one row per outcome
│   │   ├── summary_metrics.json           # same + full configuration
│   │   └── Within{k}Yr/                   # one folder per horizon
│   │       ├── metrics.json
│   │       ├── cv_fold_metrics.csv
│   │       ├── cv_predictions.csv
│   │       ├── selected_features_cv.csv
│   │       ├── final_selected_features.csv
│   │       ├── best_hyperparameters.json
│   │       ├── model.joblib
│   │       └── plots/
│   ├── src_gradual_training/              # gradual-training sweep
│   │   ├── config.py                      # outcomes, hyperparam strategy, k range
│   │   └── gradual_train.py               # entry point
│   └── results_gradual_training/          # gradual sweep outputs
│       ├── Report.md                      # cross-outcome scientific report
│       ├── summary_gradual_training.csv
│       ├── summary_best_k.png
│       ├── summary_best_metric_value.png
│       └── Within{k}Yr_results/
│           ├── report.md                  # per-outcome report
│           ├── peak_summary.json
│           ├── metrics_per_k.csv
│           ├── cv_fold_metrics_per_k.csv
│           ├── cv_predictions_per_k.csv
│           ├── best_hyperparameters_per_k.csv
│           ├── final_selected_features_used.csv
│           └── plots/                     # 20+ plots per outcome
├── Report.md                              # scientific report (publication-style)
├── README.md                              # how to run + outputs
└── CLAUDE.md                              # this file
```

If a new model family is added later (XGBoost, SVM, Logistic Regression, …),
**create a new sibling folder** (`XGBoost/`, `SVM/`, `LogisticRegression/`, …)
following the same `src/` + `results/` structure. Do **not** mix model
families inside an existing folder.

---

## 3. Implementation expectations

These are not negotiable — they were agreed up-front and are the basis on
which the results will be defended in a paper.

### Per-target framing
- Train **seven independent binary classifiers**, one per `Within{k}Yr`.
- The patient ID is a record identifier only; it is never a feature.
- When training the model for `Within{k}Yr`, **all other** outcome columns
  are removed from the feature matrix. This rules out a particularly
  insidious form of label leakage (e.g. using `Within2Yr` to predict
  `Within1Yr`).

### Validation
- **Nested cross-validation** is mandatory.
  - Outer loop: `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` — gives
    15 unbiased held-out estimates per outcome.
  - Inner loop: `StratifiedKFold(n_splits=5)` for hyperparameter tuning.
- Stratification on the (binary) target is mandatory because of class
  imbalance.
- Metrics aggregated **per fold** (mean ± sd) and **pooled** across all
  held-out predictions, both at the 0.50 cut-off and at the Youden-optimal
  threshold.

### Feature selection
- Done **inside the outer-training fold only** — never on the full data
  before splitting.
- Method: stability selection over an L1-regularised logistic regression
  (saga solver, `l1_ratio=1`, `class_weight="balanced"`), 100 subsamples at
  75 % size, three regularisation strengths `C ∈ {0.01, 0.1, 1.0}`, the
  per-feature stability score is the **maximum** selection frequency across
  the C-grid (randomised-lasso variant of Meinshausen & Bühlmann 2010).
- Threshold 0.60; if fewer than 5 features pass, fall back to the top-5 by
  frequency. Hard cap at 50 selected features.

### Modelling
- For now: scikit-learn `RandomForestClassifier` with
  `class_weight="balanced"` and `random_state` fixed.
- Hyperparameters tuned on the inner CV with `roc_auc` as the optimisation
  metric. Grid is intentionally small (avoid overfitting CV scores at small
  N): `n_estimators ∈ {300}`, `max_depth ∈ {None, 5}`,
  `min_samples_leaf ∈ {1, 3}`, `max_features ∈ {"sqrt", 0.3}`.
- Trees are scale-invariant — the `StandardScaler` exists only to feed the
  L1 selector. Random-Forest training uses the raw selected columns.

### Reproducibility
- Single `RANDOM_STATE = 42` propagated to every stochastic component.
- All artifacts are written to disk; nothing important lives only in memory.
- If you change the pipeline in a way that affects results, bump nothing
  silently — update `Report.md` accordingly.

---

## 4. Saved artifacts (per outcome)

For every `Within{k}Yr/` folder the pipeline writes:

| File | What it contains |
|---|---|
| `metrics.json` | summary stats: per-fold mean ± sd, pooled metrics at 0.5 and at the Youden threshold, confusion matrices, hyperparameter choice, runtime |
| `cv_fold_metrics.csv` | one row per outer fold with every metric |
| `cv_predictions.csv` | per-patient out-of-fold prediction (id, fold, y_true, y_prob, y_pred@0.5) — the source of truth for any further analysis |
| `selected_features_cv.csv` | how often each feature passed stability selection across the 15 outer training folds |
| `final_selected_features.csv` | features chosen by stability selection on the *full* dataset, with their RF importances — this is the published feature list |
| `best_hyperparameters.json` | inner-CV best params per fold |
| `model.joblib` | final fitted RF + scaler + selected feature names + Youden threshold |
| `plots/roc_curves.png` | per-fold ROC + pooled curve |
| `plots/pr_curve.png` | precision-recall on pooled OOF predictions |
| `plots/confusion_matrix_thr0.5.png` and `_youden.png` | confusion matrices at both thresholds |
| `plots/rf_feature_importance.png` | top-20 RF importances on the final model |
| `plots/stability_selection_frequency.png` | top-20 by stability frequency |

The repository-level `Random_Forest/results/summary_metrics.csv` collects one
row per outcome for at-a-glance comparison.

---

## 5. How to run

```bash
# from the project root — main pipeline
.venv/bin/python Random_Forest/src/train_rf.py            # all 7 horizons
.venv/bin/python Random_Forest/src/train_rf.py --outcomes Within1Yr Within3Yr

# gradual-training sweep (depends on the main run's results/)
.venv/bin/python Random_Forest/src_gradual_training/gradual_train.py
.venv/bin/python Random_Forest/src_gradual_training/gradual_train.py --outcomes Within3Yr
```

The first invocation creates `Random_Forest/results/` and overwrites previous
runs. The gradual script writes to `Random_Forest/results_gradual_training/`.
Set `OMP_NUM_THREADS=1` if you intend to launch multiple training sessions in
parallel.

### 5.1 Gradual-training rules

- **Always run the main pipeline first.** The gradual sweep reads each
  outcome's `final_selected_features.csv` from `Random_Forest/results/`. If
  that file is missing, the outcome is skipped with a warning.
- **Feature ordering is by `rf_importance` (descending)** — the same column
  that the main pipeline writes. Don't switch to `stability_frequency`
  silently; if you want to compare orderings, add a config flag.
- **Same nested CV as the main pipeline.** Outer
  `RepeatedStratifiedKFold(5, 3)` and inner `StratifiedKFold(5)` are reused
  so the curve at k = N is directly comparable to the main pipeline's
  headline numbers for that outcome.
- **`HYPERPARAM_STRATEGY` in `src_gradual_training/config.py` controls
  tuning.** `per_fold_grid` runs `GridSearchCV` per fold per k (the
  publication default). `fixed_best` reuses the most-frequent winning
  hyperparameters from the main run — faster, but the curves answer a
  slightly different question (feature-count effect with hyperparameters
  held fixed).
- **Don't recompute the feature ranking inside each fold.** The point of
  this analysis is to characterise the *published* ranking; refitting it
  per fold is a different experiment that belongs in a separate module.
  Document this caveat in any paper figure caption.

---

## 6. Development guidance

- **Don't introduce leakage.** Any preprocessing that learns from the data —
  scaling, imputation, feature selection, target encoding — must live inside
  the outer fold, not be fit once on the full dataset and reused.
- **Don't expand the hyperparameter grid without a reason.** With n=94, every
  extra grid point widens the inner-CV optimism gap. Add complexity only
  when the inner-CV best score is clearly capped by the current grid.
- **Don't change `RANDOM_STATE` to chase a better number.** That is the most
  common form of accidental p-hacking on small datasets.
- **Do save everything.** Reviewers will ask for per-patient OOF predictions
  and per-fold metrics; both are already written out.
- **Do keep the seven outcomes consistent.** They share one feature matrix
  and one config. If you tune anything, tune it for all seven.
- **Do prefer editing the existing files over adding new ones.**
  The four-file structure (`config.py`, `stability_selection.py`,
  `metrics_utils.py`, `train_rf.py`) is intentional.
- **When adding a new model family** (XGBoost, SVM, …), reuse the *same*
  validation protocol, the *same* feature-selection step, and the *same*
  output schema. Otherwise cross-model comparison will be apples-to-oranges.

---

## 7. Out-of-scope (for now)

- External validation on a separate cohort — not available.
- Time-to-event / Cox modelling — outcomes are coded as binary horizons.
- Calibration recalibration (Platt, isotonic) — Brier score is reported but
  no calibration plot is produced yet.
- SHAP / per-patient explanations — RF feature importance is the only
  global explainer for now.
