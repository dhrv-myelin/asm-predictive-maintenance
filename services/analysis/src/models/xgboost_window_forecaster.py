import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor

class XGBWindowForecaster(BaseEstimator, RegressorMixin):
    """
    Multivariate time-series forecaster using window flattening.

    Learns mapping:
        past window_length timesteps -> target at t + predict_horizon

    fit(X, y)
    predict(window)
    """

    def __init__(
        self,
        window_length=6,
        predict_horizon=5,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ):
        self.window_length = window_length
        self.predict_horizon = predict_horizon

        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=random_state,
            verbosity=0,
        )

    # --------------------------------------------------
    # BUILD WINDOWED TRAINING DATA
    # --------------------------------------------------

    def _build_training_matrix(self, X, y):
        """
        X: (N, num_features)
        y: (N,)

        Returns:
            X_train (samples, window_length * num_features)
            y_train (samples,)
        """

        n = len(X)
        X_rows = []
        y_vals = []

        for i in range(self.window_length - 1, n):
            target_idx = i + self.predict_horizon
            if target_idx >= n:
                break

            window = X[i - self.window_length + 1 : i + 1]
            X_rows.append(window.flatten())
            y_vals.append(y[target_idx])

        return np.array(X_rows), np.array(y_vals)

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------

    def fit(self, X, y):
        """
        X: (N, num_features)
        y: (N,)
        """

        X_train, y_train = self._build_training_matrix(X, y)

        if len(X_train) < 2:
            raise ValueError("Not enough data to build windowed samples.")

        self.model.fit(X_train, y_train)

        return self

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(self, X):
        """
        X expected shape:
            (window_length, num_features)

        Returns:
            (1,) prediction for t + predict_horizon
        """

        if X.shape[0] != self.window_length:
            raise ValueError(
                f"Expected {self.window_length} rows for prediction window."
            )

        X_flat = X.flatten().reshape(1, -1)

        return self.model.predict(X_flat)
