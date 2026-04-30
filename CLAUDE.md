# CLAUDE.md — MACE Risk-Prediction Project

This file gives Claude Code (and future contributors) the project context, the
implementation expectations that have already been agreed, and the development
guidance that should be followed for every change.

---

## 1. Project context

**Goal.** Build, evaluate and document machine-learning models that predict the
occurrence of a Major Adverse Cardiovascular Event (MACE) within 1, 2, …, 7
years after the index visit, from a baseline proteomics panel plus a few
clinical features. The work is intended as a peer-reviewed publication, so the
project compares **four model families** under a single, fixed validation
protocol: **Random Forest**, **XGBoost**, **Logistic Regression**, and
**SVM with an RBF kernel**.

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
│   ├── src_gradual_training/              # gradual-training sweep
│   │   ├── config.py
│   │   └── gradual_train.py
│   └── results_gradual_training/          # gradual sweep outputs
├── XGBoost/                               # XGBoost model family (same layout)
│   ├── src/{config.py, stability_selection.py, metrics_utils.py, train_xgb.py}
│   ├── src_gradual_training/{config.py, gradual_train.py}
│   ├── results/                           # populated by train_xgb.py
│   └── results_gradual_training/
├── LogisticRegression/                    # L1/L2 logistic regression
│   ├── src/{config.py, stability_selection.py, metrics_utils.py, train_lr.py}
│   ├── src_gradual_training/{config.py, gradual_train.py}
│   ├── results/
│   └── results_gradual_training/
├── SVM/                                   # SVC with RBF kernel
│   ├── src/{config.py, stability_selection.py, metrics_utils.py, train_svm.py}
│   ├── src_gradual_training/{config.py, gradual_train.py}
│   ├── results/
│   └── results_gradual_training/
├── Report.md                              # scientific report (publication-style)
├── README.md                              # how to run + outputs
└── CLAUDE.md                              # this file
```

Every model family follows an **identical** internal layout — `src/` for the
main pipeline, `src_gradual_training/` for the per-k feature-count sweep,
`results/` and `results_gradual_training/` for outputs. Each `Within{k}Yr/`
results folder contains the same files (`metrics.json`, `cv_fold_metrics.csv`,
`cv_predictions.csv`, `selected_features_cv.csv`, `final_selected_features.csv`,
`best_hyperparameters.json`, `model.joblib`, `plots/`).

If a new model family is added later, **create a new sibling folder** following
the same `src/` + `src_gradual_training/` structure. Do **not** mix model
families inside an existing folder. Each model's `results/` and
`results_gradual_training/` directories are git-ignored — the pipelines
regenerate them deterministically from `RANDOM_STATE = 42`.

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

All four families share the *same* feature-selection step, the *same* nested
CV protocol, the *same* output schema, and `roc_auc` as the inner-CV scoring
metric. The classifier and its grid differ; everything else is held fixed so
cross-model comparison is apples-to-apples.

- **Random Forest** — `RandomForestClassifier` with `class_weight="balanced"`.
  Grid: `n_estimators ∈ {300}`, `max_depth ∈ {None, 5}`,
  `min_samples_leaf ∈ {1, 3}`, `max_features ∈ {"sqrt", 0.3}`. Trees are
  scale-invariant; the `StandardScaler` is used only inside L1 selection.
- **XGBoost** — `XGBClassifier(objective="binary:logistic",
  tree_method="hist")` with `scale_pos_weight = #neg / #pos` per fold. Grid:
  `n_estimators ∈ {200, 400}`, `max_depth ∈ {3, 5}`,
  `learning_rate ∈ {0.05, 0.1}`, `subsample = 0.8`, `colsample_bytree = 0.8`,
  `reg_lambda = 1.0`, `min_child_weight = 1`. Trees are scale-invariant; the
  raw selected columns are passed in.
- **Logistic Regression** — `Pipeline(StandardScaler, LogisticRegression)`
  with `class_weight="balanced"`. The pipeline is mandatory — LR is
  scale-sensitive. Grid covers both penalties: `{penalty=l2, C ∈ {0.01,
  0.1, 1.0, 10.0}, solver=lbfgs}` and `{penalty=l1, C ∈ {0.1, 1.0, 10.0},
  solver=liblinear}`. `max_iter=5000`.
- **SVM (RBF)** — `Pipeline(StandardScaler, SVC(kernel="rbf",
  probability=True, class_weight="balanced"))`. Grid:
  `C ∈ {0.1, 1.0, 10.0}`, `gamma ∈ {"scale", 0.01, 0.1}`. Probabilities are
  Platt-scaled by sklearn (`probability=True`) for downstream metrics.

### Final-model feature importance

The "importance" column in each model's `final_selected_features.csv` is the
ranking that drives that model's gradual-training sweep:

| Model | Importance column | Definition |
|---|---|---|
| Random Forest | `rf_importance` | mean decrease in impurity |
| XGBoost | `xgb_importance` | gain-based importance (sklearn API default) |
| Logistic Regression | `lr_importance` | `|standardised coefficient|` from the fitted Pipeline |
| SVM (RBF) | `svm_importance` | permutation importance (drop in ROC AUC, `n_repeats=30`) on the final fitted Pipeline |

Permutation importance is used for SVM because RBF kernels have no native
feature_importances_. For RF/XGB/LR, the model-native importance is the
canonical choice.

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

Each model family has its own main entry point and its own gradual-training
entry point. The gradual-training scripts depend on the main pipeline's
`results/Within{k}Yr/final_selected_features.csv`, so always run main first.

```bash
# from the project root — main pipelines
.venv/bin/python Random_Forest/src/train_rf.py
.venv/bin/python XGBoost/src/train_xgb.py
.venv/bin/python LogisticRegression/src/train_lr.py
.venv/bin/python SVM/src/train_svm.py

# gradual-training sweeps (depend on each main run's results/)
.venv/bin/python Random_Forest/src_gradual_training/gradual_train.py
.venv/bin/python XGBoost/src_gradual_training/gradual_train.py
.venv/bin/python LogisticRegression/src_gradual_training/gradual_train.py
.venv/bin/python SVM/src_gradual_training/gradual_train.py

# subset:
.venv/bin/python XGBoost/src/train_xgb.py --outcomes Within1Yr Within3Yr
```

Each invocation creates / overwrites that model's `results/` (or
`results_gradual_training/`) tree. Set `OMP_NUM_THREADS=1` if you intend to
launch multiple model families in parallel — the inner GridSearchCV already
parallelises over folds via `N_JOBS=-1`.

### 5.1 Gradual-training rules

- **Always run that model's main pipeline first.** The gradual sweep reads
  each outcome's `final_selected_features.csv` from
  `<Model>/results/Within{k}Yr/`. If that file is missing, the outcome is
  skipped with a warning.
- **Feature ordering is by the model's own importance column**
  (`rf_importance`, `xgb_importance`, `lr_importance`, `svm_importance`).
  Each model's gradual sweep characterises *its own* published ranking,
  not someone else's. Don't switch to `stability_frequency` silently; if
  you want to compare orderings, add a config flag.
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
