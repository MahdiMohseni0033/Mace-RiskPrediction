"""Configuration for the XGBoost gradual-training pipeline.

Mirrors Random_Forest/src_gradual_training/config.py one-for-one. The feature
ranking comes from XGBoost's own `final_selected_features.csv` so the sweep
characterises *the XGBoost model's published ranking*, not the RF one.
"""
from pathlib import Path

# ----- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "datasets" / "imputed_proteomics_base.csv"

MAIN_RESULTS_DIR = PROJECT_ROOT / "XGBoost" / "results"
RESULTS_DIR = PROJECT_ROOT / "XGBoost" / "results_gradual_training"

# ----- Schema (mirrors the main pipeline) ------------------------------------
ID_COLUMN = "EuroSCOREPatient ID"
ALL_OUTCOMES = [f"Within{k}Yr" for k in range(1, 8)]

# ----- Which outcomes to run -------------------------------------------------
OUTCOMES_TO_RUN = [
    "Within1Yr",
    "Within2Yr",
    "Within3Yr",
    "Within4Yr",
    "Within5Yr",
    "Within6Yr",
    "Within7Yr",
]

# ----- Cross-validation (kept identical to the main pipeline) ---------------
OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 3
INNER_N_SPLITS = 5
RANDOM_STATE = 42

# ----- Hyperparameter strategy -----------------------------------------------
HYPERPARAM_STRATEGY = "per_fold_grid"

XGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [1.0],
    "min_child_weight": [1],
}

INNER_SCORING = "roc_auc"

# ----- Sweep settings --------------------------------------------------------
K_MIN = 1
K_MAX = None
K_STEP = 1

# ----- Misc ------------------------------------------------------------------
N_JOBS = -1
VERBOSE = True

# Column name in final_selected_features.csv that gives the ranking score.
RANKING_COLUMN = "xgb_importance"

# Family display name (used in plot titles and report headings).
MODEL_FAMILY = "XGBoost"
