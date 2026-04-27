"""Configuration for the gradual-training pipeline.

This pipeline depends on the artefacts written by `Random_Forest/src/train_rf.py`.
For each outcome listed in `OUTCOMES_TO_RUN`, the script reads the published
feature ranking (`final_selected_features.csv`, sorted by Random-Forest impurity
importance), then trains a sequence of Random Forest classifiers that use the
top-1, top-2, ..., top-N features and records held-out performance for each k.

The aim is to answer: how many of the published features does the model
actually need to reach its peak performance?
"""
from pathlib import Path

# ----- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "datasets" / "imputed_proteomics_base.csv"

# Source of truth for the per-outcome ranked feature list (produced by the
# main pipeline). The gradual script reads each outcome's
# `final_selected_features.csv` from this directory.
MAIN_RESULTS_DIR = PROJECT_ROOT / "Random_Forest" / "results"

# Where this script writes its artefacts.
RESULTS_DIR = PROJECT_ROOT / "Random_Forest" / "results_gradual_training"

# ----- Schema (mirrors the main pipeline) ------------------------------------
ID_COLUMN = "EuroSCOREPatient ID"
ALL_OUTCOMES = [f"Within{k}Yr" for k in range(1, 8)]

# ----- Which outcomes to run -------------------------------------------------
# Edit this list to control the run. Use the full set for a publication run.
OUTCOMES_TO_RUN = [
    "Within1Yr",
    "Within2Yr",
    "Within3Yr",
    "Within4Yr",
    "Within5Yr",
    "Within6Yr",
    "Within7Yr",
]

# ----- Cross-validation (kept identical to the main pipeline so the
# gradual-training curves are directly comparable to the headline numbers) ---
OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 3
INNER_N_SPLITS = 5
RANDOM_STATE = 42

# ----- Random-Forest hyperparameter tuning -----------------------------------
# Strategy options:
#   "per_fold_grid" — rerun GridSearchCV inside every outer fold, for every k.
#                     Most rigorous; matches the main pipeline's protocol.
#   "fixed_best"    — reuse the most-frequent best hyperparameters from the
#                     main pipeline's `best_hyperparameters.json`. Much
#                     faster, isolates the feature-count effect from
#                     hyperparameter variance.
HYPERPARAM_STRATEGY = "per_fold_grid"

# Used when HYPERPARAM_STRATEGY == "per_fold_grid" — same grid as the main run.
RF_PARAM_GRID = {
    "n_estimators": [300],
    "max_depth": [None, 5],
    "min_samples_leaf": [1, 3],
    "max_features": ["sqrt", 0.3],
}

# Inner-CV scoring metric (matches the main pipeline).
INNER_SCORING = "roc_auc"

# ----- Sweep settings --------------------------------------------------------
# Range of k (number of top features) to evaluate.
#   K_MIN: smallest k to evaluate (>= 1).
#   K_MAX: if None, runs up to the number of features in
#          final_selected_features.csv for that outcome.
#   K_STEP: stride (1 = every k, 2 = every other k, ...).
K_MIN = 1
K_MAX = None
K_STEP = 1

# ----- Misc ------------------------------------------------------------------
N_JOBS = -1
VERBOSE = True
