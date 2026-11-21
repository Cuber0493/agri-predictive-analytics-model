"""
Root prediction launcher for Agri Predictive Analytics Model.
Runs agri-yield/src/predict.py.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent / "agri-yield"
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import main

if __name__ == "__main__":
    main()
