from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
import numpy as np


class XGBWindowForecaster(BaseEstimator, RegressorMixin):

    def __init__(self, window_length=6, predict_horizon=5, **xgb_params):
        self.window_length = window_length
        self.predict_horizon = predict_horizon
        self.xgb_params = xgb_params
        self.model = XGBRegressor(**xgb_params)

    def _make_training_data(self, X, y):
        X_rows, y_vals = [], []

        n = len(X)

        for i in range(self.window_length - 1, n):
            target_idx = i + self.predict_horizon
            if target_idx >= n:
                break

            window = X[i - self.window_length + 1 : i + 1]
            X_rows.append(window.flatten())
            y_vals.append(y[target_idx])

        return np.array(X_rows), np.array(y_vals)

    def fit(self, X, y):
        X_train, y_train = self._make_training_data(X, y)

        if len(X_train) < 2:
            raise ValueError("Not enough data for window training")

        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        """
        X expected shape:
        (window_length, num_features)
        """
        if len(X.shape) == 2:
            X = X.flatten().reshape(1, -1)

        return self.model.predict(X)
