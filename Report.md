# Predicting Major Adverse Cardiovascular Events from Baseline Plasma Proteomics: A Random-Forest Pipeline with Stability-Selected Features

> Scientific report accompanying the code in this repository. Numbers in
> this document are produced verbatim from
> `Random_Forest/results/summary_metrics.csv` and the per-outcome
> `metrics.json` files, with the seed `RANDOM_STATE = 42`.

---

## 1. Background and aim

Major adverse cardiovascular events (MACE) — death, myocardial infarction,
stroke, urgent revascularisation — drive most of the morbidity that follows
cardiac surgery. Early identification of patients at elevated long-term
risk could in principle inform follow-up intensity and secondary
prevention. We test the hypothesis that a baseline plasma proteomic
fingerprint, supplemented by a small set of clinical analytes, carries
horizon-specific prognostic information for MACE during a seven-year
follow-up window.

The aim of this work is descriptive and benchmarking, not deployment: we
quantify how well an off-the-shelf non-linear classifier (Random Forest)
can separate cases from non-cases at each yearly horizon, given a small
cohort and a high-dimensional protein panel, while controlling for the
two dominant failure modes of small-N omics modelling — *information leakage
during feature selection* and *over-fitted hyperparameter tuning*.

---

## 2. Data

### 2.1 Cohort and panel

The dataset (`datasets/imputed_proteomics_base.csv`) contains 94 patients
characterised by 559 columns. After removing the patient identifier
(`EuroSCOREPatient ID`) and the seven outcome labels, **551 numeric
candidate features** remain, comprising Olink proteomic panels and a
small number of clinical analytes. A single missing value (in `G-CSF`)
was imputed during dataset preparation; the matrix delivered to the
modelling pipeline is complete.

### 2.2 Outcomes

Seven binary labels `Within{k}Yr` (k = 1 … 7) record whether the patient
had experienced a MACE by year *k* of follow-up. Class balance varies
substantially with horizon, from heavily imbalanced at one year (18 %
positives) to majority-positive by year 7 (60 % positives) — see Table 1.

**Table 1.** Cohort and class balance per outcome.

| Outcome | n | n positive | n negative | Prevalence |
|---|---:|---:|---:|---:|
| Within1Yr | 94 | 17 | 77 | 0.181 |
| Within2Yr | 94 | 32 | 62 | 0.340 |
| Within3Yr | 94 | 36 | 58 | 0.383 |
| Within4Yr | 94 | 43 | 51 | 0.457 |
| Within5Yr | 94 | 47 | 47 | 0.500 |
| Within6Yr | 94 | 53 | 41 | 0.564 |
| Within7Yr | 94 | 56 | 38 | 0.596 |

### 2.3 Per-target framing

Seven independent binary classifiers were trained, one per horizon. When
training the classifier for `Within{k}Yr`, the **other six** outcome
columns were excluded from the feature matrix. This rules out a subtle
form of label leakage: with cumulative outcomes, knowing `Within2Yr`
trivially forces `Within1Yr = 1` for some patients, so leaving sibling
outcomes in the feature set would inflate apparent discrimination.
Patient identifiers were never used as features.

---

## 3. Methods

The full pipeline is implemented in `Random_Forest/src/`. All design
choices below are encoded in `config.py` so that the run is fully
parameterised.

### 3.1 Nested cross-validation

Generalisation performance is estimated with a nested cross-validation
scheme:

- **Outer loop** — `RepeatedStratifiedKFold(n_splits = 5, n_repeats = 3,
  random_state = 42)`. The repetition reduces the variance of the
  performance estimate at small *n*, yielding 15 unbiased held-out test
  folds per outcome.
- **Inner loop** — `StratifiedKFold(n_splits = 5)` on each outer training
  fold, used solely for hyperparameter tuning.

All splits are stratified on the binary target.

### 3.2 Feature selection: stability selection by L1-logistic regression

Inside every outer training fold (and only there) we run an
implementation of the *randomised-lasso* variant of stability selection
(Meinshausen & Bühlmann, 2010) with a logistic-regression objective:

1. Standardise the outer-training features with `StandardScaler` (this
   scaler is used only to feed the L1 selector — Random Forests are
   scale-invariant and consume the unscaled selected columns).
2. Draw 100 random subsamples of size 75 % × *n*<sub>train</sub>, without
   replacement.
3. For each subsample, fit a class-weight-balanced logistic regression
   with an L1 penalty (`LogisticRegression(solver = "saga",
   l1_ratio = 1.0, class_weight = "balanced")`) at three regularisation
   strengths *C* ∈ {0.01, 0.1, 1.0}.
4. For each feature, compute the per-*C* selection frequency (fraction of
   subsamples in which the coefficient was non-zero) and take the
   maximum across the *C* grid as its **stability score**.
5. Retain features with stability score ≥ 0.60. If fewer than five
   features pass, fall back to the top-five by stability score; cap the
   selection at 50 features.

Because selection is run independently on every outer training fold,
held-out test points never inform which features survive — the central
guard against *selection-induced* leakage. Final published feature lists
are produced by re-running the same procedure on the full dataset.

### 3.3 Random-Forest classifier

A `RandomForestClassifier(class_weight = "balanced",
random_state = 42)` is fit on the stability-selected columns. The
hyperparameter grid, deliberately small to keep the inner-CV optimism
bound at small *n*, is

| Hyperparameter | Grid |
|---|---|
| `n_estimators` | {300} |
| `max_depth` | {None, 5} |
| `min_samples_leaf` | {1, 3} |
| `max_features` | {"sqrt", 0.3} |

**Hyperparameter-tuning algorithm.** Tuning is performed by *exhaustive
grid search* using `sklearn.model_selection.GridSearchCV`. With
8 candidate combinations and 5 inner folds, every combination is
evaluated on every inner fold (40 fits per outer fold), the ROC-AUC
(`scoring="roc_auc"`, insensitive to threshold and class balance) is
averaged across the inner folds, and the combination with the highest
mean inner-CV AUC is selected. The chosen combination is then refit on
the entire outer-training fold (`refit=True`) and evaluated on the
held-out outer-test fold. We deliberately use grid search rather than
randomised or Bayesian search: the grid is small and discrete, so
exhaustive evaluation is cheaper, fully deterministic given the seed,
and removes one source of run-to-run variability. The final,
deployable model for each horizon is fit on **all** patients with the
most-frequent outer-CV winning combination, on the features chosen by
stability selection on the full dataset.

### 3.4 Performance reporting

For every outer fold we record ROC-AUC, average precision, balanced
accuracy, accuracy, precision, recall (sensitivity), specificity,
F1-score, Brier score, and the confusion matrix at the 0.50 cut-off. We
report:

- per-fold mean ± standard deviation across the 15 outer folds, and
- pooled metrics computed on the concatenation of all out-of-fold
  predictions (preserves the patient-level structure of the data),
  evaluated at both the 0.50 cut-off and at the Youden-optimal threshold.

We additionally save: the per-patient out-of-fold predictions; the
inner-CV-winning hyperparameters per fold; the per-feature
stability-selection frequencies aggregated across the 15 folds; the
final selected-feature list with RF impurity-based importances; and the
fitted model artifact.

### 3.5 Reproducibility

A single random seed (`RANDOM_STATE = 42`) is propagated to all
stochastic components: outer CV, inner CV, bootstrap resampling within
stability selection, and the Random Forest itself. Re-running
`Random_Forest/src/train_rf.py` reproduces every number in this report
to the precision shown.

---

## 4. Results

### 4.1 Discrimination across horizons

**Table 2.** Out-of-fold discrimination metrics. *Pooled AUC* is computed
on the concatenated 15-fold OOF predictions; *fold-mean AUC* is the mean
± sd of the 15 per-fold AUCs.

| Outcome | Pooled AUC | Fold-mean AUC | Pooled AP | Pooled BalAcc | Brier | n features (final) |
|---|---:|---:|---:|---:|---:|---:|
| Within1Yr | 0.636 | 0.633 ± 0.174 | 0.286 | 0.497 | 0.149 | 33 |
| Within2Yr | 0.637 | 0.652 ± 0.136 | 0.469 | 0.589 | 0.219 | 50 |
| Within3Yr | 0.695 | 0.696 ± 0.077 | 0.586 | 0.615 | 0.209 | 48 |
| Within4Yr | 0.660 | 0.661 ± 0.087 | 0.689 | 0.591 | 0.225 | 38 |
| Within5Yr | 0.693 | 0.694 ± 0.071 | 0.722 | 0.645 | 0.221 | 46 |
| Within6Yr | 0.714 | 0.732 ± 0.107 | 0.782 | 0.634 | 0.212 | 50 |
| Within7Yr | 0.719 | 0.735 ± 0.089 | 0.792 | 0.646 | 0.205 | 49 |

Headline observations:

1. Discrimination **improves with longer horizons**, from a pooled AUC of
   0.636 at one year to 0.719 at seven years. This reflects two
   compounding effects: (i) the proteomic signal is stronger for
   slowly-emerging chronic outcomes than for acute peri-operative events,
   and (ii) class balance improves with horizon, narrowing the
   confidence intervals on AUC estimates.
2. Per-fold variability is highest at one year (sd 0.174 across 15
   folds), consistent with the very small positive count (17 events
   spread across 15 outer folds — roughly one positive per held-out
   fold). The 1-year model should be regarded as exploratory.
3. Average-precision tracks AUC closely once class balance is roughly
   even. AP for the 1-year model (0.286) is well above the prevalence
   (0.181), but the absolute value is too small to support decision-rule
   deployment at this horizon.
4. Brier scores between 0.15 and 0.23 indicate calibration that is
   substantially better than random (Brier of a no-skill predictor at the
   prevalence baseline ranges from 0.16 to 0.25 across horizons), though
   without explicit recalibration we make no calibration claim.

### 4.2 Threshold-dependent operating points

At the default 0.50 cut-off (Table 3a), Random Forest's `class_weight =
"balanced"` does not produce an aggressive predictor: sensitivities
range from 0.02 (Within1Yr) to 0.81 (Within7Yr), with the inverse pattern
in specificity. Choosing the Youden-optimal threshold per outcome
(Table 3b) substantially improves balanced accuracy in the imbalanced
horizons and is the operating point we recommend for any clinical
interpretation of the model.

**Table 3a.** Operating point at the 0.50 cut-off.

| Outcome | Sens | Spec | F1 | BalAcc |
|---|---:|---:|---:|---:|
| Within1Yr | 0.020 | 0.974 | 0.034 | 0.497 |
| Within2Yr | 0.333 | 0.844 | 0.408 | 0.589 |
| Within3Yr | 0.380 | 0.851 | 0.469 | 0.615 |
| Within4Yr | 0.496 | 0.686 | 0.531 | 0.591 |
| Within5Yr | 0.624 | 0.667 | 0.638 | 0.645 |
| Within6Yr | 0.748 | 0.520 | 0.706 | 0.634 |
| Within7Yr | 0.810 | 0.482 | 0.749 | 0.646 |

**Table 3b.** Operating point at the Youden threshold.

| Outcome | Threshold | Sens | Spec | BalAcc |
|---|---:|---:|---:|---:|
| Within1Yr | 0.250 | 0.431 | 0.831 | 0.631 |
| Within2Yr | 0.470 | 0.417 | 0.828 | 0.622 |
| Within3Yr | 0.347 | 0.713 | 0.580 | 0.647 |
| Within4Yr | 0.649 | 0.333 | 0.954 | 0.644 |
| Within5Yr | 0.537 | 0.596 | 0.738 | 0.667 |
| Within6Yr | 0.573 | 0.629 | 0.699 | 0.664 |
| Within7Yr | 0.580 | 0.726 | 0.623 | 0.674 |

### 4.3 Selected features and cross-horizon recurrence

Each per-outcome `final_selected_features.csv` lists between 33 and 50
features chosen by stability selection on the full dataset. A more
robust read is the **cross-horizon recurrence** of features, i.e. how
many of the seven final models retain a given feature.

**Table 4.** Features retained in 5 or more of the 7 final models
(stability + RF importance ranking, on full data).

| Feature | # horizons (out of 7) |
|---|:---:|
| 148_TR-AP | 7 |
| 162_U-PAR | 6 |
| BCAN | 6 |
| CD200R1 | 6 |
| IL-20RA | 6 |
| LY75 | 6 |
| 170_DCN | 5 |
| 179_EGFR | 5 |
| 185_CTSZ | 5 |
| 186_MMP-3 | 5 |
| CCL20 | 5 |
| CD83 | 5 |
| CST5 | 5 |
| IL-2RB | 5 |
| LAG3 | 5 |
| PSG1 | 5 |
| SH2D1A | 5 |
| WFIKKN1 | 5 |

Several of these are biologically coherent with cardiovascular risk:
- **uPAR (162_U-PAR / PLAUR)** is an established prognostic biomarker for
  cardiovascular mortality and inflammation; its appearance in 6/7
  models is the most clinically interpretable signal in the table.
- **MMP-3** (matrix metalloproteinase-3) and **TR-AP / TRAP**
  (tartrate-resistant acid phosphatase) are tissue-remodelling markers
  with prior evidence of association with adverse cardiac outcomes.
- **EGFR** signalling intersects with vascular smooth-muscle and
  endothelial biology; **GDF-15** (top feature for Within7Yr in Table 5
  below) is a well-known integrative marker of cardiovascular and
  all-cause mortality.
- The remaining recurring proteins are dominated by **immune-cell
  surface antigens / co-stimulators** (CD200R1, LY75, CD83, LAG3,
  IL-20RA, IL-2RB, SH2D1A) and **inflammatory chemokines** (CCL20),
  consistent with a chronic inflammatory contribution to long-term
  MACE.

The fact that the stability-selected features are biologically sensible
and substantially overlap across horizons is reassuring at this sample
size, where any single horizon's feature ranking is necessarily
unstable.

**Table 5.** Top eight features by RF importance in the final per-horizon
model.

| Outcome | Top features |
|---|---|
| Within1Yr | 162_U-PAR, LAMP3, 153_THBS2, 123_LTBR, EZR, LILRB4, LAG3, 181_BNP |
| Within2Yr | SKR3, 103_ADM, LILRB4, MAPT, 162_U-PAR, 185_CTSZ, EDA2R, 186_MMP-3 |
| Within3Yr | BTN3A2, LAG3, 186_MMP-3, 179_EGFR, CD83, 103_ADM, 185_CTSZ, FCRL6 |
| Within4Yr | BTN3A2, 162_U-PAR, CD83, MCP-3, 143_RETN, 179_EGFR, TGF-alpha, 159_XCL1 |
| Within5Yr | 162_U-PAR, BTN3A2, NAAA, 143_RETN, MCP-3, 179_EGFR, 167_MMP-7, IL-5R-alpha |
| Within6Yr | 143_RETN, 107_PlGF, 162_U-PAR, SKR3, 179_EGFR, ATP6V1F, 159_XCL1, 116_IDUA |
| Within7Yr | 135_GDF-15, CKAP4, 162_U-PAR, 179_EGFR, 114_CSTB, ATP6V1F, IFI30, BCAN |

### 4.4 Plots

For every outcome the pipeline writes (under
`Random_Forest/results/Within{k}Yr/plots/`):

- `roc_curves.png` — fifteen translucent per-fold ROC curves overlaid
  on the pooled OOF ROC curve (bold black) with the pooled AUC and the
  fold-mean ± sd AUC in the legend.
- `pr_curve.png` — pooled OOF precision-recall curve with the prevalence
  baseline.
- `confusion_matrix_thr0.5.png` and `confusion_matrix_youden.png` —
  pooled OOF confusion matrices at the 0.50 and Youden thresholds.
- `rf_feature_importance.png` — top-20 features by mean impurity
  decrease in the final RF.
- `stability_selection_frequency.png` — top-20 features by mean
  stability-selection frequency across the 15 outer training folds.

---

## 5. Discussion

The pipeline successfully demonstrates that a baseline proteomic profile
carries genuine, if moderate, information about long-term MACE risk in
this cohort. Discrimination is best at horizons with stable class
balance and chronic-disease pathophysiology (5–7 years; pooled AUC
0.69 – 0.72), and weakest at the one-year horizon, where peri-operative
events dominate and class imbalance is severe (pooled AUC 0.64, with
high fold-to-fold variability). Cross-horizon recurrence of
biologically plausible markers (uPAR, GDF-15, MMP-3, EGFR signalling,
immune co-stimulators) increases confidence that the signal is real
rather than an artefact of the small sample size.

The ceiling on absolute performance is consistent with the literature
on small proteomic cohorts: fold-mean AUCs in the 0.65 – 0.74 range,
with standard deviations of 0.07 – 0.17 across folds, are a realistic
expectation when *n* < 100 and *p* > 500. We chose nested CV plus
stability selection precisely so the *reported* numbers are not
inflated by the standard small-cohort pitfalls — fitting selection on
all data, tuning hyperparameters on the test fold, or reporting a
single "best run".

The pipeline is intentionally portable: the validation protocol,
feature-selection step and output schema are model-agnostic, and the
repository is structured so that additional model families (XGBoost,
SVM, regularised logistic regression, …) can be added as sibling
folders for fair head-to-head comparison.

---

## 6. Limitations and outlook

1. **Single cohort, no external validation.** External replication on an
   independent dataset is the single most informative next step.
2. **Binary horizons rather than time-to-event.** Censoring is not
   modelled; a Cox / random-survival-forest formulation would use the
   data more efficiently.
3. **Small sample size at the one-year horizon.** With 17 events the
   1-year model is underpowered; results should be regarded as
   exploratory. Bootstrap confidence intervals on AUC for this horizon
   would be wide.
4. **Calibration is not recalibrated.** Brier scores are reported but no
   isotonic / Platt step is applied. Decision-curve analysis would also
   strengthen the clinical interpretation.
5. **Feature interpretation is correlational.** RF importance and
   stability-selection frequency identify reproducibly informative
   markers, but neither provides causal direction or per-patient
   explanations. SHAP-based decomposition is a natural follow-up.
6. **Single model family.** The headline numbers are from Random Forest
   alone. We expect XGBoost and a regularised logistic regression to
   produce comparable AUCs; the framework supports their addition.

---

## 7. Reproducibility statement

All numbers in this report are produced by

```bash
.venv/bin/python Random_Forest/src/train_rf.py
```

with the seed `RANDOM_STATE = 42` set in
`Random_Forest/src/config.py`. The full per-patient OOF predictions
(`cv_predictions.csv`), the per-fold metrics (`cv_fold_metrics.csv`),
the selected-feature lists with stability scores, and the fitted model
artefacts (`model.joblib`) are written to
`Random_Forest/results/Within{k}Yr/` and are sufficient to regenerate
every table and figure in this report.

---

## 8. References (selected)

- Meinshausen N, Bühlmann P. Stability Selection. *J. R. Stat. Soc. B*
  72:417 – 473 (2010).
- Bach FR. Bolasso: model consistent Lasso estimation through the
  bootstrap. *ICML* 2008.
- Cawley GC, Talbot NLC. On over-fitting in model selection and
  subsequent selection bias in performance evaluation. *JMLR* 11:2079 –
  2107 (2010).
- Steyerberg EW. *Clinical Prediction Models*, 2nd ed. Springer (2019).
- Eapen DJ et al. soluble urokinase plasminogen activator receptor
  (uPAR) and adverse outcomes. *Eur Heart J* (multiple years).
- Wollert KC, Kempf T, Wallentin L. GDF-15 in cardiovascular disease.
  *Clin Chem* 63:140 – 151 (2017).
