"""Configuration for the SVM (RBF) gradual-training pipeline.

Mirrors Random_Forest/src_gradual_training/config.py one-for-one. The
feature ranking comes from the SVM pipeline's `final_selected_features.csv`
(sorted by `svm_importance`, i.e. permutation importance).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "datasets" / "imputed_proteomics_base.csv"

MAIN_RESULTS_DIR = PROJECT_ROOT / "SVM" / "results"
RESULTS_DIR = PROJECT_ROOT / "SVM" / "results_gradual_training"

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

SVM_PARAM_GRID = {
    "clf__C": [0.1, 1.0, 10.0],
    "clf__gamma": ["scale", 0.01, 0.1],
}

INNER_SCORING = "roc_auc"

K_MIN = 1
K_MAX = None
K_STEP = 1

N_JOBS = -1
VERBOSE = True

RANKING_COLUMN = "svm_importance"
MODEL_FAMILY = "SVM-RBF"
