import numpy as np
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


class IsolationForestHealth(BaseEstimator):
    """
    Unsupervised health scoring model.

    Learns healthy baseline distribution using IsolationForest
    and centroid distance.

    fit(X)     -> learn healthy behavior
    predict(X) -> return health score per row (0-100)

    NaN handling: median imputation is applied before all fitting and
    prediction so that missing sensor readings don't propagate through
    to the scores.
    """

    def __init__(
        self,
        n_estimators=100,
        contamination="auto",
        drift_percentile=5,
        breakdown_sensitivity=3.0,
        smoothing_window=20,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.drift_percentile = drift_percentile
        self.breakdown_sensitivity = breakdown_sensitivity
        self.smoothing_window = smoothing_window
        self.random_state = random_state

        # Impute missing sensor values with per-feature median before scaling
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.iso = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------

    def fit(self, X, y=None):
        """
        X: (N, num_features)
        Assumed to be healthy training data.
        y is ignored (unsupervised), accepted for sklearn API compatibility.
        """
        X = np.array(X, dtype=float)

        nan_count = np.sum(np.isnan(X))
        if nan_count > 0:
            print(
                f"  [IsolationForestHealth] Imputing {nan_count} NaN values in training data"
            )

        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        self.iso.fit(X_scaled)

        # Store training anomaly score distribution for z-score normalisation
        self.train_scores_ = self.iso.decision_function(X_scaled)
        self.score_mean_ = np.mean(self.train_scores_)
        self.score_std_ = max(np.std(self.train_scores_), 1e-6)

        # Healthy centroid and distance thresholds
        self.centroid_ = np.mean(X_scaled, axis=0)
        distances = np.linalg.norm(X_scaled - self.centroid_, axis=1)
        self.d_anchor_ = np.percentile(distances, 100 - self.drift_percentile)
        self.d_limit_ = self.d_anchor_ * self.breakdown_sensitivity

        return self

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(self, X):
        """
        X: (M, num_features)

        Returns:
            health_scores (M,) in range [0, 100]
        """
        X = np.array(X, dtype=float)

        nan_count = np.sum(np.isnan(X))
        if nan_count > 0:
            print(
                f"  [IsolationForestHealth] Imputing {nan_count} NaN values in prediction data"
            )

        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        # Distance-based component
        distances = np.linalg.norm(X_scaled - self.centroid_, axis=1)
        h_dist = np.where(
            distances <= self.d_anchor_,
            100.0,
            np.clip(
                100.0
                - (
                    (distances - self.d_anchor_)
                    / (self.d_limit_ - self.d_anchor_ + 1e-6)
                )
                * 100.0,
                0.0,
                100.0,
            ),
        )

        # IsolationForest pattern-based component
        iso_scores = self.iso.decision_function(X_scaled)
        z_scores = (self.score_mean_ - iso_scores) / self.score_std_
        h_iso = np.clip(100.0 - (z_scores * 15.0), 0.0, 100.0)

        raw_health = (h_dist + h_iso) / 2.0

        # Rolling average smoothing with edge-safe padding
        w = self.smoothing_window
        if w > 1 and len(raw_health) >= w:
            kernel = np.ones(w) / w
            valid = np.convolve(raw_health, kernel, mode="valid")
            # valid has length = len(raw_health) - w + 1
            # Pad edges with the nearest valid value to restore original length
            pad_left = w // 2
            pad_right = w - 1 - pad_left
            smooth = np.concatenate(
                [
                    np.full(pad_left, valid[0]),
                    valid,
                    np.full(pad_right, valid[-1]),
                ]
            )
            return smooth[: len(raw_health)]

        return raw_health
