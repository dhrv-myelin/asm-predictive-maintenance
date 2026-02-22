import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import percentileofscore


class PdMHealthModel:
    """
    Generic Predictive Maintenance Health Model.

    Usage:
        model = PdMHealthModel(IsolationForest, {...})
        model.fit(train_df)
        raw_h, sma_h, drifts = model.predict_health(monitor_df)
        y_fut, rul_cycles, rul_minutes, idx = model.forecast_rul(sma_h)

    train_df and monitor_df must be:
        - Pandas DataFrame
        - Indexed by timestamp
        - Columns = numeric metrics
    """

    def __init__(
        self,
        anomaly_model_cls,
        anomaly_model_kwargs=None,
        drift_percentile=5,
        breakdown_sensitivity=3.0,
        smoothing_window=20,
        rul_activate_health=95,
        rul_target_health=0,
    ):

        self.anomaly_model_cls = anomaly_model_cls
        self.anomaly_model_kwargs = anomaly_model_kwargs or {}

        self.drift_percentile = drift_percentile
        self.breakdown_sensitivity = breakdown_sensitivity
        self.smoothing_window = smoothing_window
        self.rul_activate_health = rul_activate_health
        self.rul_target_health = rul_target_health

        self.scaler = StandardScaler()
        self.anomaly_model = None

        # learned during fit
        self.centroid = None
        self.d_anchor = None
        self.d_limit = None
        self.train_iso_scores = None
        self.t_iso_mean = None
        self.t_iso_std = None
        self.train_means = None
        self.train_stds = None

    # -------------------------------------------------
    # FIT
    # -------------------------------------------------
    def fit(self, train_df: pd.DataFrame):
        """
        Fit on healthy training window.
        """

        train_scaled = self.scaler.fit_transform(train_df)

        self.anomaly_model = self.anomaly_model_cls(**self.anomaly_model_kwargs)
        self.anomaly_model.fit(train_scaled)

        # Isolation-style decision baseline
        self.train_iso_scores = self.anomaly_model.decision_function(train_scaled)
        self.t_iso_mean = np.mean(self.train_iso_scores)
        self.t_iso_std = max(np.std(self.train_iso_scores), 0.001)

        # Drift baseline
        self.centroid = np.mean(train_scaled, axis=0)
        train_distances = np.linalg.norm(train_scaled - self.centroid, axis=1)

        self.d_anchor = np.percentile(train_distances, 100 - self.drift_percentile)
        self.d_limit = self.d_anchor * self.breakdown_sensitivity

        # Metric statistics for RCA
        self.train_means = train_df.mean()
        self.train_stds = np.maximum(train_df.std().values, 0.01)

        return self

    # -------------------------------------------------
    # HEALTH PREDICTION
    # -------------------------------------------------
    def predict_health(self, monitor_df: pd.DataFrame):
        """
        Returns:
            raw_health_series
            sma_health_series
            metric_drifts (z-scores for RCA)
        """

        monitor_scaled = self.scaler.transform(monitor_df)

        # --- distance-based health ---
        monitor_distances = np.linalg.norm(monitor_scaled - self.centroid, axis=1)

        h_dist = np.where(
            monitor_distances <= self.d_anchor,
            100.0,
            np.clip(
                100
                - (
                    (monitor_distances - self.d_anchor)
                    / (self.d_limit - self.d_anchor + 1e-6)
                )
                * 100,
                0,
                100,
            ),
        )

        # --- pattern-based health ---
        monitor_iso_scores = self.anomaly_model.decision_function(monitor_scaled)

        h_iso_pct = np.array(
            [percentileofscore(self.train_iso_scores, s) for s in monitor_iso_scores]
        )

        h_iso = np.where(h_iso_pct > 50, 100, h_iso_pct * 2.0).clip(0, 100)

        raw_h = pd.Series(
            (h_dist + h_iso) / 2,
            index=monitor_df.index,
        ).clip(0, 100)

        sma_h = raw_h.rolling(
            window=self.smoothing_window,
            min_periods=1,
        ).mean()

        # --- RCA metric drifts ---
        metric_drifts = (monitor_df - self.train_means) / self.train_stds

        metric_drifts["Pattern_Anomaly"] = (
            self.t_iso_mean - monitor_iso_scores
        ) / self.t_iso_std

        return raw_h, sma_h, metric_drifts

    # -------------------------------------------------
    # RUL FORECAST
    # -------------------------------------------------
    def forecast_rul(self, sma_series: pd.Series):
        """
        Returns:
            forecast_health_1hr,
            rul_cycles,
            rul_minutes,
            regression_window_index
        """

        curr_h = sma_series.iloc[-1]

        if curr_h > self.rul_activate_health:
            return None, 0, 0, None

        t_now = sma_series.index[-1]
        t_lookback = t_now - pd.Timedelta(hours=1)
        y_window = sma_series.loc[t_lookback:t_now]

        if len(y_window) < 10:
            return None, 0, 0, None

        x_vals = (
            (y_window.index - y_window.index[0]).total_seconds().values.reshape(-1, 1)
        )

        reg = LinearRegression().fit(x_vals, y_window.values)
        slope = reg.coef_[0]

        if slope >= -0.0001:
            return None, 0, 0, None

        sec_to_zero = (self.rul_target_health - curr_h) / slope

        avg_cycle_sec = (y_window.index[-1] - y_window.index[0]).total_seconds() / len(
            y_window
        )

        rul_cycles = int(sec_to_zero / avg_cycle_sec)
        rul_minutes = int(sec_to_zero / 60)

        x_future = np.array([x_vals[-1] + 3600]).reshape(-1, 1)
        y_fut_end = reg.predict(x_future)[0]

        return (
            y_fut_end,
            max(0, rul_cycles),
            max(0, rul_minutes),
            y_window.index,
        )

