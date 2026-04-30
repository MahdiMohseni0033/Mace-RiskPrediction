"""Configuration for the XGBoost MACE risk-prediction pipeline.

Mirrors Random_Forest/src/config.py one-for-one so cross-model results are
directly comparable. Only the modelling block (XGB hyperparameter grid) and
the output directory differ.
"""
from pathlib import Path

# ----- Paths -----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "datasets" / "imputed_proteomics_base.csv"
RESULTS_DIR = PROJECT_ROOT / "XGBoost" / "results"

# ----- Schema -----
ID_COLUMN = "EuroSCOREPatient ID"
OUTCOME_COLUMNS = [f"Within{k}Yr" for k in range(1, 8)]

# ----- Cross-validation -----
OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 3
INNER_N_SPLITS = 5
RANDOM_STATE = 42

# ----- Stability selection (L1-logistic) — kept identical to RF pipeline
# so feature selection cannot bias the cross-model comparison.
STABILITY_N_SUBSAMPLES = 100
STABILITY_SUBSAMPLE_FRAC = 0.75
STABILITY_C_GRID = (0.01, 0.1, 1.0)
STABILITY_THRESHOLD = 0.60
MIN_SELECTED_FEATURES = 5
MAX_SELECTED_FEATURES = 50

# ----- XGBoost hyperparameter grid (inner CV) -----
# Kept deliberately small: with n=94 and 15 outer folds, every extra grid
# point widens the inner-CV optimism gap.
XGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [1.0],
    "min_child_weight": [1],
}

# ----- Misc -----
N_JOBS = -1
