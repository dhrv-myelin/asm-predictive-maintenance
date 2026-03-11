"""
models/var_forecaster.py
========================
Multivariate time-series forecaster using Vector Autoregression (VAR).

Mirrors the sklearn-compatible interface of XGBWindowForecaster so it
slots cleanly into SklearnBackend in model.py.

VAR contract (important — differs from XGBoost)
------------------------------------------------
- fit(X, y)   : X is (T, n_features) — a flat 2D sequence of T timesteps.
                y is accepted but ignored; VAR is self-supervised.
- predict(X)  : X is either
                  (n_samples, lag_order * n_features)  — batch (SklearnBackend)
                  (lag_order, n_features)               — real-time single window
                Returns (n_samples,) with the target column's value at
                predict_horizon steps ahead.

The caller (SklearnBackend / model.py) must use the VAR_MODELS reshape path
(see integration notes at bottom of file) so that X arrives as a flat 2D
sequence rather than being further flattened across the seq_len axis.
"""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.vector_ar.var_model import VAR


class VARForecaster(BaseEstimator, RegressorMixin):
    """
    VAR-based multivariate forecaster.

    Parameters
    ----------
    maxlags : int | None
        VAR lag order p. If None, auto-selected by AIC up to maxlags_search.
    maxlags_search : int
        Upper bound for AIC lag search (only used when maxlags is None).
    predict_horizon : int
        Steps ahead to forecast (h in statsmodels VAR.forecast).
    target_col_idx : int
        Which column in the feature matrix is the prediction target.
        VAR forecasts all variables — this picks the one that gets returned.
    difference : bool
        First-difference the series before fitting and invert at predict time.
        Recommended if your series has a unit root (fails ADF stationarity test).
    """

    def __init__(
        self,
        maxlags: int | None = 5,
        maxlags_search: int = 10,
        predict_horizon: int = 1,
        target_col_idx: int = 0,
        difference: bool = False,
    ):
        self.maxlags = maxlags
        self.maxlags_search = maxlags_search
        self.predict_horizon = predict_horizon
        self.target_col_idx = target_col_idx
        self.difference = difference

        # Populated after fit()
        self._fitted_model = None  # statsmodels VARResultsWrapper
        self._lag_order: int = None  # resolved lag order p
        self._fit_tail: np.ndarray = None  # last p rows of training data
        self._last_level: np.ndarray = None  # last undifferenced row (for inversion)
        self._n_features: int = None

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y=None) -> "VARForecaster":
        """
        Fit the VAR model on the full training sequence.

        Parameters
        ----------
        X : (T, n_features)
            Full 2D sequence. The VAR_MODELS path in model.py ensures this
            arrives as (N * seq_len, n_features) — i.e. the entire unrolled
            time series, not independent windowed batches.
        y : ignored — VAR is self-supervised.
        """
        X = np.array(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                f"VARForecaster.fit expects 2D array (T, n_features), got shape {X.shape}. "
                "Make sure 'var_forecast' is listed in VAR_MODELS in model.py so the "
                "correct reshape path is used."
            )

        self._n_features = X.shape[1]

        if self.difference:
            self._last_level = X[-1].copy()
            X = np.diff(X, axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var_model = VAR(X)

            if self.maxlags is None:
                sel = var_model.select_order(maxlags=self.maxlags_search)
                self._lag_order = int(sel.aic)
                print(f"[VAR] Auto-selected lag order (AIC): {self._lag_order}")
            else:
                self._lag_order = int(self.maxlags)

            self._fitted_model = var_model.fit(maxlags=self._lag_order, ic=None)

        # Store forecast anchor: last p rows of (possibly differenced) series
        self._fit_tail = X[-self._lag_order :].copy()

        print(
            f"[VAR] Fit complete | lag_order={self._lag_order} | "
            f"n_features={self._n_features} | T={X.shape[0]}"
        )
        return self

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forecast predict_horizon steps ahead and return the target column.

        Parameters
        ----------
        X : one of:
            (n_samples, lag_order * n_features)  — flattened batch from SklearnBackend
            (lag_order, n_features)               — real-time single window
            (1, lag_order * n_features)           — single-sample batch

        Returns
        -------
        preds : (n_samples,)
            Forecasted value of the target column at t + predict_horizon.
        """
        if self._fitted_model is None:
            raise RuntimeError("VARForecaster must be fit() before predict().")

        X = np.array(X, dtype=float)
        n_features = self._n_features

        # ── normalise to (n_samples, lag_order, n_features) ──────────────
        if X.ndim == 2 and X.shape == (self._lag_order, n_features):
            # Real-time single window: (p, F) -> (1, p, F)
            windows = X[np.newaxis, ...]
        elif X.ndim == 2 and X.shape[1] == self._lag_order * n_features:
            # Batch from SklearnBackend: (n_samples, p*F) -> (n_samples, p, F)
            windows = X.reshape(-1, self._lag_order, n_features)
        elif X.ndim == 3:
            # Already (n_samples, p, F)
            windows = X
        else:
            raise ValueError(
                f"VARForecaster.predict: unexpected X shape {X.shape}. "
                f"Expected (lag_order={self._lag_order}, n_features={n_features}), "
                f"(n_samples, {self._lag_order * n_features}), or "
                f"(n_samples, {self._lag_order}, {n_features})."
            )

        preds = []
        for window in windows:  # window: (p, F)
            anchor = window.copy()

            if self.difference:
                anchor = np.diff(
                    np.vstack([self._last_level[np.newaxis, :], anchor]), axis=0
                )

            # statsmodels VAR.forecast(anchor, steps=h) -> (h, F)
            forecast = self._fitted_model.forecast(anchor, steps=self.predict_horizon)

            # Extract target column at the requested horizon step
            target_val = float(forecast[self.predict_horizon - 1, self.target_col_idx])

            if self.difference and self._last_level is not None:
                # Invert differencing
                target_val = self._last_level[self.target_col_idx] + target_val

            preds.append(target_val)

        return np.array(preds)
