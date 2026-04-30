"""Configuration for the Logistic-Regression gradual-training pipeline.

Mirrors Random_Forest/src_gradual_training/config.py one-for-one. The
feature ranking comes from the LR pipeline's `final_selected_features.csv`
(sorted by `lr_importance`, i.e. |standardised coefficient|).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "datasets" / "imputed_proteomics_base.csv"

MAIN_RESULTS_DIR = PROJECT_ROOT / "LogisticRegression" / "results"
RESULTS_DIR = PROJECT_ROOT / "LogisticRegression" / "results_gradual_training"

ID_COLUMN = "EuroSCOREPatient ID"
ALL_OUTCOMES = [f"Within{k}Yr" for k in range(1, 8)]

OUTCOMES_TO_RUN = [
    "Within1Yr", "Within2Yr", "Within3Yr", "Within4Yr",
    "Within5Yr", "Within6Yr", "Within7Yr",
]

OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 3
INNER_N_SPLITS = 5
RANDOM_STATE = 42

HYPERPARAM_STRATEGY = "per_fold_grid"

LR_PARAM_GRID = [
    {
        "clf__penalty": ["l2"],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs"],
    },
    {
        "clf__penalty": ["l1"],
        "clf__C": [0.1, 1.0, 10.0],
        "clf__solver": ["liblinear"],
    },
]
LR_MAX_ITER = 5000

INNER_SCORING = "roc_auc"

K_MIN = 1
K_MAX = None
K_STEP = 1

N_JOBS = -1
VERBOSE = True

RANKING_COLUMN = "lr_importance"
MODEL_FAMILY = "Logistic Regression"
