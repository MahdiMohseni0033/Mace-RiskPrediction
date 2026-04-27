# MACE Risk Prediction from Baseline Proteomics

Predicting Major Adverse Cardiovascular Events (MACE) within 1 to 7 years
of follow-up, from a baseline proteomics panel of 551 candidate analytes,
in a cohort of 94 patients.

This repository contains a fully reproducible machine-learning pipeline
intended for a peer-reviewed publication. The current model family is
**Random Forest**; the code is structured so that additional model families
(XGBoost, SVM, Logistic Regression, …) can be added later as sibling folders
without touching the existing one.

---

## 1. Project structure

```
Mace-RiskPrediction/
├── datasets/
│   └── imputed_proteomics_base.csv     # primary input (94 × 559)
├── Random_Forest/
│   ├── src/                            # main pipeline code
│   │   ├── config.py
│   │   ├── stability_selection.py
│   │   ├── metrics_utils.py
│   │   └── train_rf.py
│   ├── results/                        # populated by the main pipeline
│   │   ├── summary_metrics.csv         # one row per outcome
│   │   ├── summary_metrics.json
│   │   └── Within{1..7}Yr/
│   │       ├── metrics.json
│   │       ├── cv_fold_metrics.csv
│   │       ├── cv_predictions.csv
│   │       ├── selected_features_cv.csv
│   │       ├── final_selected_features.csv
│   │       ├── best_hyperparameters.json
│   │       ├── model.joblib
│   │       └── plots/  (ROC, PR, confusion-matrix, feature plots)
│   ├── src_gradual_training/           # gradual-training analysis
│   │   ├── config.py                   # which outcomes, hyperparam strategy
│   │   └── gradual_train.py            # top-k sweep entry point
│   └── results_gradual_training/       # populated by the gradual pipeline
│       ├── Report.md                   # cross-outcome scientific report
│       ├── summary_gradual_training.csv
│       ├── summary_best_k.png
│       ├── summary_best_metric_value.png
│       └── Within{1..7}Yr_results/
│           ├── report.md               # per-outcome report (peak per metric)
│           ├── peak_summary.json
│           ├── metrics_per_k.csv
│           ├── cv_fold_metrics_per_k.csv
│           ├── cv_predictions_per_k.csv
│           ├── best_hyperparameters_per_k.csv
│           ├── final_selected_features_used.csv
│           └── plots/  (per-metric vs k, heatmaps, peak markers, ROC overlay)
├── Report.md                           # scientific report
├── README.md                           # this file
└── CLAUDE.md                           # implementation notes / dev guide
```

---

## 2. Inputs

`datasets/imputed_proteomics_base.csv` — 94 rows × 559 columns:

| Column type | Count | Used as |
|---|---|---|
| `EuroSCOREPatient ID` (string) | 1 | Record identifier — never a feature |
| `Within1Yr`, `Within2Yr`, …, `Within7Yr` | 7 | Binary outcomes (one classifier per column) |
| All other columns (numeric proteomics + clinical) | 551 | Candidate predictors |

The dataset is already imputed (one missing value in the original `G-CSF`
column was filled before this stage). When training the model for
`Within{k}Yr`, the **other six** outcome columns are dropped from the
feature matrix to prevent label leakage.

Class balance per horizon:

| Horizon | Negatives | Positives | Prevalence |
|---|---|---|---|
| Within1Yr | 77 | 17 | 18 % |
| Within2Yr | 62 | 32 | 34 % |
| Within3Yr | 58 | 36 | 38 % |
| Within4Yr | 51 | 43 | 46 % |
| Within5Yr | 47 | 47 | 50 % |
| Within6Yr | 41 | 53 | 56 % |
| Within7Yr | 38 | 56 | 60 % |

---

## 3. Method (one-paragraph summary)

For each of the seven horizons we train a Random-Forest classifier inside a
**nested cross-validation** loop. The outer loop is a 5-fold
`RepeatedStratifiedKFold` repeated 3 times (15 unbiased held-out estimates).
Inside every outer training fold, **stability selection** with an
L1-regularised logistic regression is run independently — 100 random
75 % subsamples × 3 regularisation strengths — and only features whose
maximum across-C selection frequency exceeds 0.60 (with a min of 5 / max of
50) are passed to the Random Forest. Random-Forest hyperparameters are
tuned by **exhaustive grid search** (`sklearn.model_selection.GridSearchCV`,
`scoring="roc_auc"`, `refit=True`) over the small discrete grid below, on
the 5-fold stratified inner CV. The final published model is fitted by
re-running stability selection on the full data and refitting the Random
Forest with the most-frequent winning hyperparameter combination from the
outer folds. Full details and the rationale are in `Report.md`.

---

## 4. Outputs

For each outcome horizon the pipeline writes one folder
`Random_Forest/results/Within{k}Yr/` containing:

| File | What you can do with it |
|---|---|
| `metrics.json` | Headline numbers — per-fold mean ± sd, pooled metrics at the 0.50 cut-off and at the Youden-optimal threshold, confusion matrices, hyperparameter choices, runtime. |
| `cv_fold_metrics.csv` | Every metric for every outer fold — drop into a spreadsheet to recompute summaries. |
| `cv_predictions.csv` | Per-patient out-of-fold prediction (`patient id, fold, y_true, y_prob, y_pred@0.5`). Source of truth for any downstream analysis (calibration, decision curves, subgroup analyses, …). |
| `selected_features_cv.csv` | For each candidate feature, the mean stability-selection frequency across the 15 outer training folds and the number of folds in which it survived selection — the basis for any "is feature X reproducibly informative?" claim. |
| `final_selected_features.csv` | The published feature list: features chosen by stability selection on the full dataset, with their RF impurity-based importances. |
| `best_hyperparameters.json` | Inner-CV best hyperparameters per outer fold. |
| `model.joblib` | The deployable artifact: fitted Random Forest, the scaler used during selection, the selected feature names, and the Youden threshold. |
| `plots/roc_curves.png` | Per-fold ROC curves overlaid on the pooled OOF curve. |
| `plots/pr_curve.png` | Precision-recall on pooled OOF predictions, with prevalence baseline. |
| `plots/confusion_matrix_thr0.5.png` & `_youden.png` | Confusion matrices at both decision thresholds. |
| `plots/rf_feature_importance.png` | Top-20 features by RF importance on the final model. |
| `plots/stability_selection_frequency.png` | Top-20 features by stability-selection frequency across outer folds. |

`Random_Forest/results/summary_metrics.csv` collects one row per outcome
for at-a-glance comparison; the JSON sibling also stores the full
configuration that produced those numbers.

---

## 5. Reproducing the results

### Environment

A Python 3.11 virtual environment is provided at `.venv/`. The pipeline
needs only the standard scientific stack:

```bash
.venv/bin/pip install pandas scikit-learn numpy matplotlib seaborn joblib
```

(These are already installed in the supplied venv.)

### Run the pipeline

```bash
# all seven horizons (default)
.venv/bin/python Random_Forest/src/train_rf.py

# a subset
.venv/bin/python Random_Forest/src/train_rf.py --outcomes Within1Yr Within3Yr
```

The pipeline is deterministic given `RANDOM_STATE = 42` (set in
`Random_Forest/src/config.py`). Total runtime is on the order of a few
minutes on a multi-core machine; stability selection and the inner CV use
all available cores via `joblib`.

### Run the gradual-training analysis

After the main pipeline has finished (the gradual sweep reads its
`final_selected_features.csv` files), run:

```bash
# all outcomes configured in src_gradual_training/config.py
.venv/bin/python Random_Forest/src_gradual_training/gradual_train.py

# subset
.venv/bin/python Random_Forest/src_gradual_training/gradual_train.py --outcomes Within3Yr Within5Yr
```

This trains, for each outcome, one Random Forest per `k = 1, 2, ..., N`
features (where N is the size of the published feature ranking) under the
same nested CV protocol as the main pipeline, and writes per-k metrics,
per-fold predictions, and a rich plot suite to
`Random_Forest/results_gradual_training/`. The cross-outcome scientific
report is written to `Random_Forest/results_gradual_training/Report.md`.

### Loading a trained model

```python
import joblib, pandas as pd
bundle = joblib.load("Random_Forest/results/Within3Yr/model.joblib")

df = pd.read_csv("datasets/imputed_proteomics_base.csv")
X = df[bundle["selected_feature_names"]].to_numpy(float)

probs = bundle["model"].predict_proba(X)[:, 1]
preds = (probs >= bundle["youden_threshold"]).astype(int)
```

The bundle stores: `model`, `scaler_for_l1`, `selected_feature_names`,
`all_feature_names`, `outcome`, `best_params`, `youden_threshold`.

---

## 6. Adding a new model family

Create a sibling top-level folder, copy the `src/` skeleton, and reuse
`stability_selection.py` and `metrics_utils.py` so that cross-model
comparison is fair:

```
Mace-RiskPrediction/
├── Random_Forest/
├── XGBoost/                   # new
│   ├── src/
│   └── results/
└── SVM/                       # new
    ├── src/
    └── results/
```

The validation protocol, feature-selection step, and output schema must be
held constant across model families — only the classifier and its
hyperparameter grid should change. See `CLAUDE.md` for the rules.

---

## 7. Limitations

- Single cohort (n = 94) — external validation on an independent dataset is
  not available.
- Outcomes are encoded as binary horizons rather than time-to-event;
  patients censored before each horizon are not separately accounted for.
- Calibration is reported (Brier score) but no isotonic / Platt
  recalibration is applied.
- See the *Limitations and outlook* section of `Report.md` for the full list.

---

## 8. Citation / contact

Methods, results and per-feature interpretation are described in
`Report.md`. For questions about the code, see `CLAUDE.md`.
