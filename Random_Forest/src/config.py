"""Configuration for the Random Forest MACE risk-prediction pipeline."""
from pathlib import Path

# ----- Paths -----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "datasets" / "imputed_proteomics_base.csv"
RESULTS_DIR = PROJECT_ROOT / "Random_Forest" / "results"

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
STABILITY_THRESHOLD = 0.60      # selection frequency cutoff
MIN_SELECTED_FEATURES = 5
MAX_SELECTED_FEATURES = 50

# ----- Random Forest hyperparameter grid (inner CV) -----
RF_PARAM_GRID = {
    "n_estimators": [300],
    "max_depth": [None, 5],
    "min_samples_leaf": [1, 3],
    "max_features": ["sqrt", 0.3],
}

# ----- Misc -----
N_JOBS = -1
