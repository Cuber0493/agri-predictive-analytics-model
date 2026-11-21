"""
Utility and helper functions for Agri Predictive Analytics Model.
Includes seed setting across random, numpy, os, and ML algorithms.
"""

import os
import random
import numpy as np
from .config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Enforces deterministic behavior by setting the random seed across
    Python built-in random, NumPy, and environment variables.

    Parameters
    ----------
    seed : int, optional
        The seed number to use, by default 42 (from config.RANDOM_SEED).
    """
    # 1. Python built-in random module
    random.seed(seed)

    # 2. Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3. NumPy random generator
    np.random.seed(seed)

    # Note: If deep learning libraries (torch / tensorflow) are added in future,
    # seed them here conditionally.
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_random_state(seed: int = RANDOM_SEED) -> int:
    """
    Returns the standard random seed integer to pass into
    scikit-learn, LightGBM, Optuna samplers, etc.
    """
    return seed
