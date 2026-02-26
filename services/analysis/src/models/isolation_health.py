import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


class IsolationForestHealth(BaseEstimator):
    """
    Unsupervised health scoring model.

    Learns healthy baseline distribution using IsolationForest
    and centroid distance.

    fit(X)     -> learn healthy behavior
    predict(X) -> return health score per row (0–100)
    """

    def __init__(
        # TODO: find actual good values for what goes into the models.
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
        """

        X_scaled = self.scaler.fit_transform(X)

        # Train isolation forest
        self.iso.fit(X_scaled)

        # Store training anomaly score distribution
        self.train_scores = self.iso.decision_function(X_scaled)
        self.score_mean = np.mean(self.train_scores)
        self.score_std = max(np.std(self.train_scores), 1e-6)

        # Compute healthy centroid
        self.centroid = np.mean(X_scaled, axis=0)

        # Compute distance thresholds
        distances = np.linalg.norm(X_scaled - self.centroid, axis=1)

        self.d_anchor = np.percentile(distances, 100 - self.drift_percentile)
        self.d_limit = self.d_anchor * self.breakdown_sensitivity

        return self

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------

    def predict(self, X):
        """
        X: (M, num_features)

        Returns:
            health_scores (M,)
        """

        X_scaled = self.scaler.transform(X)

        # Distance-based health
        distances = np.linalg.norm(X_scaled - self.centroid, axis=1)

        h_dist = np.where(
            distances <= self.d_anchor,
            100.0,
            np.clip(
                100
                - ((distances - self.d_anchor) / (self.d_limit - self.d_anchor + 1e-6))
                * 100,
                0,
                100,
            ),
        )

        # Pattern-based health
        scores = self.iso.decision_function(X_scaled)

        z_scores = (self.score_mean - scores) / self.score_std
        h_iso = np.clip(100 - (z_scores * 15), 0, 100)

        raw_health = (h_dist + h_iso) / 2

        # Rolling smoothing (manual)
        if self.smoothing_window > 1:
            smooth = np.convolve(
                raw_health,
                np.ones(self.smoothing_window) / self.smoothing_window,
                mode="same",
            )
            return smooth

        return raw_health
