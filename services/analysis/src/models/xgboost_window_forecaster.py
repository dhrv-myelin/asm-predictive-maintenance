import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor


class XGBWindowForecaster(BaseEstimator, RegressorMixin):
    """
    Multivariate time-series forecaster using window flattening.

    Learns mapping:
        past window_length timesteps -> target at t + predict_horizon

    fit(X, y)
    predict(X)
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
    # BUILD WINDOWED TRAINING DATA (used for raw timeseries input)
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
        X: (N, window_length * num_features)  <- pre-flattened by SklearnBackend
        y: (N,)  or  (N, 1)
        """
        y = np.array(y).ravel()

        if len(X) < 2:
            raise ValueError("Not enough data to fit the model.")

        self.model.fit(X, y)
        return self

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(self, X):
        """
        X expected shape:
            (n_samples, window_length * num_features)  <- batch from SklearnBackend
            OR
            (window_length, num_features)              <- real-time single window

        Returns:
            (n_samples,) predictions
        """
        X = np.array(X)

        # Real-time single window: (window_length, num_features) -> flatten
        if X.ndim == 2 and X.shape[0] == self.window_length:
            X = X.flatten().reshape(1, -1)

        # Batch: (n_samples, flat_features) — pass through directly
        return self.model.predict(X)
