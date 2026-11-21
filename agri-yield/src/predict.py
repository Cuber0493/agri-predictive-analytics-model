"""
Example: end-of-August yield forecast with the saved model.

The model needs the columns listed in models/model_meta.json ("features"), built by src/features.py from data
available on 31 August (rainfall to August, soil test, history of previous seasons). It returns yield in quintal/ha.
Run:  python src/predict.py
"""
import sys, json
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.anchored import AnchoredRegressor  # noqa: F401  (needed to unpickle the model)


def load_model(path: Path = MODELS_DIR / "final_model.joblib"):
    return joblib.load(path)


def predict_yield(records: pd.DataFrame, path: Path = MODELS_DIR / "final_model.joblib") -> np.ndarray:
    """records: DataFrame with the columns in model_meta.json['features'] -> predicted yield (q/ha)."""
    model = load_model(path)
    return model.predict(records[list(model.feature_names_in_)])


def main():
    meta = json.load(open(MODELS_DIR / "model_meta.json"))
    df = pd.read_csv(PROCESSED_DATA_DIR / "agri_yield_features.csv")
    sample = df[(df.year == 2024) & (df.is_outlier == 0)].dropna(subset=[meta["anchor"]]).sample(8, random_state=42)
    out = sample[["district", "crop", "year"]].copy()
    out["predicted_q_per_ha"] = predict_yield(sample).round(2)
    out["actual_q_per_ha"] = sample[meta["target"]].round(2)
    print("Synthetic-data demo: 8 random 2024 records\n")
    print(out.to_string(index=False))
    print(f"\nFeatures required ({len(meta['features'])}): {meta['features']}")


if __name__ == "__main__":
    main()
