"""Configuration for the Logistic-Regression MACE risk-prediction pipeline.

Mirrors Random_Forest/src/config.py one-for-one. The classifier and its grid
differ; everything else (CV protocol, stability-selection feature step,
output schema) is held identical to the RF pipeline so cross-model
comparison is fair.
"""
from pathlib import Path

# ----- Paths -----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "datasets" / "imputed_proteomics_base.csv"
RESULTS_DIR = PROJECT_ROOT / "LogisticRegression" / "results"

# ----- Schema -----
ID_COLUMN = "EuroSCOREPatient ID"
OUTCOME_COLUMNS = [f"Within{k}Yr" for k in range(1, 8)]

# ----- Cross-validation -----
OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 3
INNER_N_SPLITS = 5
RANDOM_STATE = 42

# ----- Stability selection (L1-logistic) -----
STABILITY_N_SUBSAMPLES = 100
STABILITY_SUBSAMPLE_FRAC = 0.75
STABILITY_C_GRID = (0.01, 0.1, 1.0)
STABILITY_THRESHOLD = 0.60
MIN_SELECTED_FEATURES = 5
MAX_SELECTED_FEATURES = 50

# ----- Logistic-Regression hyperparameter grid (inner CV) -----
# Pipeline order: StandardScaler -> LogisticRegression. Grid keys are
# prefixed with `clf__` to address the LR step inside the Pipeline.
# We support both L2 (lbfgs) and L1 (liblinear) — the L1 path can produce
# additional sparsity *on top of* stability selection's pre-filter.
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

# ----- Misc -----
N_JOBS = -1
