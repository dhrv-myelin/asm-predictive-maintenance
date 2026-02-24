import pandas as pfrom sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from .isolation_forest import PdMHealthModel
from sklearn.ensemble import IsolationForest

class PdMHealthWrapper(BaseEstimator):
    """
    Sklearn-compatible wrapper for PdMHealthModel.
    """

    def __init__(self, **kwargs):
        self.model = PdMHealthModel(
            anomaly_model_cls=IsolationForest,
            anomaly_model_kwargs=kwargs
        )

    def fit(self, X, y=None):
        """
        X expected as numpy array.
        Convert to DataFrame internally.
        """
        df = pd.DataFrame(X)
        self.model.fit(df)
        return self

    def predict(self, X):
        """
        Return smoothed health score.
        """
        df = pd.DataFrame(X)
        _, sma_h, _ = self.model.predict_health(df)
        return sma_h.values
