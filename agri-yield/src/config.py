"""
Configuration settings for Agri Predictive Analytics Model.
Standardizes global random seed, file paths, and default parameters.
"""

from pathlib import Path

# Global Random Seed for reproducibility everywhere
RANDOM_SEED = 42

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Default Model Parameters with Seed 42
DEFAULT_MODEL_PARAMS = {
    "random_state": RANDOM_SEED,
}

# LightGBM default configuration with random seed
LIGHTGBM_DEFAULT_PARAMS = {
    "random_state": RANDOM_SEED,
    "seed": RANDOM_SEED,
    "objective": "regression",
    "verbose": -1,
}
