"""
AnchoredRegressor: predicts the yield RATIO  (final yield / "normal yield anchor")  and converts it
back to quintal/ha by multiplying with the anchor.

Why: yields rise ~1.1%/year (technology trend). Tree models cannot extrapolate a trend, so a model that
predicts raw yield under-predicts in later years. The anchor (75th percentile of the previous 5 seasons for
that district-crop, built only from past data) already carries the trend and the local productivity level, so
the model only has to learn the *weather/soil-driven deviation* from normal.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class AnchoredRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator=None, anchor_col="yield_anchor_5yr_p75", weighted=True):
        self.estimator = estimator
        self.anchor_col = anchor_col
        self.weighted = weighted

    def fit(self, X, y):
        X = pd.DataFrame(X)
        anchor = X[self.anchor_col].astype(float).values
        ratio = np.asarray(y, dtype=float) / anchor
        # weight by anchor so errors are optimised in q/ha (business units), not in % terms
        w = anchor if self.weighted else None
        if w is not None:
            self.estimator.fit(X, ratio, **{self._wkey(): w})
        else:
            self.estimator.fit(X, ratio)
        self.feature_names_in_ = np.array(X.columns)
        return self

    def _wkey(self):
        # Pipeline -> last step is called "model"
        return "model__sample_weight" if hasattr(self.estimator, "named_steps") else "sample_weight"

    def predict(self, X):
        X = pd.DataFrame(X)
        anchor = X[self.anchor_col].astype(float).values
        return self.estimator.predict(X) * anchor

    def predict_ratio(self, X):
        return self.estimator.predict(pd.DataFrame(X))
