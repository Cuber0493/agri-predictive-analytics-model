"""
Agri-Yield: Agricultural Predictive Analytics Model Package.
"""

from .config import RANDOM_SEED
from .utils import set_seed

# Initialize random seed on package import
set_seed(RANDOM_SEED)

__all__ = ["RANDOM_SEED", "set_seed"]
