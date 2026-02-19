"""
models/inferencers/mamba.py
============================
Inference logic for all analysis types using a loaded Mamba_TS model.

Responsibilities:
  - Accept a LoadedModel + InferenceDataset
  - Validate input schema against training schema
  - Run the appropriate inference path per analysis_type
  - Return a standardised InferenceOutput dataclass

Classes
-------
MambaInferencer     Dispatcher — routes to the correct sub-method.
_ForecastRunner     Multi-step recursive forecast using Mamba_TS.
_AnomalyRunner      Z-score | WE rules | Isolation Forest on latest window.
_HealthRunner       Linear trend health score (0-100).
_RULRunner          Linear trend remaining useful life.

Dataclass
---------
InferenceOutput     Standardised result for all analysis types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest

from models.architectures.mamba import Mamba_TS
from data.data_loader import InferenceDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceOutput:
    """
    Unified output structure for all analysis types.
    Only the fields relevant to the analysis_type will be populated.
    The `payload` dict always contains the full breakdown for DB storage.
    """
    analysis_type   : str
    method_used     : str

    # Forecast fields
    forecast_values : Optional[list[float]] = None
    forecast_horizon: Optional[int]         = None
    lower_bound     : Optional[list[float]] = None
    upper_bound     : Optional[list[float]] = None

    # Anomaly fields
    is_anomaly      : Optional[bool]        = None
    anomaly_score   : Optional[float]       = None

    # Health fields
    health_score    : Optional[float]       = None

    # RUL fields
    rul_cycles      : Optional[float]       = None
    rul_seconds     : Optional[float]       = None

    confidence      : Optional[float]       = None
    data_points_used: Optional[int]         = None

    # Full rich payload for DB JSON column
    payload         : dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Schema validator
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    pass


def _validate_schema(dataset: InferenceDataset, expected_features: list[str]) -> None:
    """
    Confirms the inference dataset has exactly the features the model was
    trained on, in the correct order.  Raises SchemaValidationError if not.
    """
    if dataset.feature_names != expected_features:
        missing  = set(expected_features) - set(dataset.feature_names)
        extra    = set(dataset.feature_names) - set(expected_features)
        raise SchemaValidationError(
            f"Feature schema mismatch. Missing: {missing}. Unexpected: {extra}"
        )


# ---------------------------------------------------------------------------
# Forecast runner
# ---------------------------------------------------------------------------

class _ForecastRunner:
    """
    Multi-step recursive forecast using the trained Mamba_TS model.

    Strategy:
      1. Take the last seq_len rows from InferenceDataset.X as the seed window.
      2. Feed through model → get pred[0] (next step).
      3. Roll the window forward (drop oldest row, append feature row for pred).
      4. Repeat for forecast_horizon steps.
      5. Compute naive confidence interval from reconstruction error on seed.
    """

    def run(
        self,
        model   : Mamba_TS,
        dataset : InferenceDataset,
        horizon : int,
        device  : torch.device,
    ) -> InferenceOutput:

        seq_len    = model.seq_len
        X          = dataset.X                         # (window_size, n_features)
        seed_window = X[-seq_len:].copy()              # (seq_len, n_features)

        forecasts: list[float] = []
        model.eval()

        with torch.no_grad():
            window = seed_window.copy()

            # Compute reconstruction error on seed as uncertainty proxy
            seed_t      = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
            recon, _    = model(seed_t)
            recon_err   = float(torch.mean((recon - seed_t) ** 2).item())
            sigma       = np.sqrt(recon_err) * 2.0     # naive CI width

            for _ in range(horizon):
                inp   = torch.tensor(window[np.newaxis], dtype=torch.float32).to(device)
                _, pred = model(inp)
                pred_val = float(pred[0, 0].item())    # scalar (scaler space)
                forecasts.append(pred_val)

                # Shift window: drop oldest, append new row
                new_row        = window[-1].copy()
                new_row[0]     = pred_val              # update first feature (target proxy)
                window         = np.vstack([window[1:], new_row])

        # Inverse-transform forecasts using the dataset scaler
        # The scaler was fit on the full feature matrix, so we need to
        # inverse-transform only the target feature (index 0)
        # We do this by constructing a full-width row with zeros elsewhere.
        n_feats     = dataset.X.shape[1]
        fc_array    = np.zeros((len(forecasts), n_feats), dtype=np.float32)
        fc_array[:, 0] = forecasts
        fc_inv      = dataset.scaler.inverse_transform(fc_array)[:, 0].tolist()

        # CI in original scale (rough: ±1.96σ)
        dummy_lo = np.zeros((1, n_feats), dtype=np.float32)
        dummy_lo[0, 0] = -sigma
        dummy_hi = np.zeros((1, n_feats), dtype=np.float32)
        dummy_hi[0, 0] = sigma
        sigma_inv = float(
            dataset.scaler.inverse_transform(dummy_hi)[0, 0] -
            dataset.scaler.inverse_transform(dummy_lo)[0, 0]
        ) / 2.0

        lower = [v - 1.96 * abs(sigma_inv) for v in fc_inv]
        upper = [v + 1.96 * abs(sigma_inv) for v in fc_inv]

        return InferenceOutput(
            analysis_type    = 'forecast',
            method_used      = 'mamba',
            forecast_values  = [round(v, 6) for v in fc_inv],
            forecast_horizon = horizon,
            lower_bound      = [round(v, 6) for v in lower],
            upper_bound      = [round(v, 6) for v in upper],
            confidence       = max(0.0, min(1.0, 1.0 - recon_err * 10)),
            data_points_used = len(X),
            payload          = {
                'recon_error_seed' : round(recon_err, 8),
                'sigma_inv'        : round(abs(sigma_inv), 6),
                'horizon'          : horizon,
            },
        )


# ---------------------------------------------------------------------------
# Anomaly runner
# ---------------------------------------------------------------------------

class _AnomalyRunner:
    """
    Runs the configured anomaly detection method on the latest observation.

    Methods
    -------
    zscore            Rolling or baseline z-score on target column.
    western_electric  WE rules 1-4 on the last N observations.
    isolation_forest  IF trained on recent feature window.

    The `method` argument comes from metric_cfg['analysis']['anomaly_method'].
    Falls back to `anomaly_fallback` if the primary method cannot run.
    """

    WE_MIN_SAMPLES = 15

    def run(
        self,
        dataset    : InferenceDataset,
        method     : str,
        fallback   : str,
        metric_cfg : dict,
    ) -> InferenceOutput:
        try:
            if method == 'western_electric':
                return self._western_electric(dataset, metric_cfg)
            elif method == 'isolation_forest':
                return self._isolation_forest(dataset, metric_cfg)
            else:
                return self._zscore(dataset, metric_cfg)
        except Exception as exc:
            logger.warning("_AnomalyRunner: %s failed (%s), falling back to %s", method, exc, fallback)
            try:
                return self._zscore(dataset, metric_cfg)
            except Exception as exc2:
                logger.error("_AnomalyRunner: fallback also failed: %s", exc2)
                return InferenceOutput(
                    analysis_type = 'anomaly',
                    method_used   = 'none',
                    is_anomaly    = False,
                    anomaly_score = 0.0,
                    confidence    = 0.0,
                    payload       = {'error': str(exc2)},
                )

    # ---- Z-score --------------------------------------------------------

    def _zscore(self, dataset: InferenceDataset, metric_cfg: dict) -> InferenceOutput:
        # Use the raw (inverse-transformed) target column from the scaler
        # We reconstruct it from the first feature column
        n_feats     = dataset.X.shape[1]
        dummy       = np.zeros((len(dataset.X), n_feats), dtype=np.float32)
        dummy[:, 0] = dataset.X[:, 0]
        target_orig = dataset.scaler.inverse_transform(dummy)[:, 0]

        mean  = float(np.mean(target_orig))
        std   = float(np.std(target_orig)) or 1e-9
        last  = float(target_orig[-1])
        z     = (last - mean) / std

        threshold  = 3.0
        is_anomaly = abs(z) > threshold

        return InferenceOutput(
            analysis_type = 'anomaly',
            method_used   = 'zscore',
            is_anomaly    = is_anomaly,
            anomaly_score = round(abs(z), 4),
            confidence    = 0.85,
            data_points_used = len(target_orig),
            payload       = {
                'anomaly_flags': {'zscore_exceeded': is_anomaly},
                'z_score'      : round(z, 4),
                'threshold'    : threshold,
                'mean'         : round(mean, 6),
                'std'          : round(std, 6),
                'last_value'   : round(last, 6),
            },
        )

    # ---- Western Electric -----------------------------------------------

    def _western_electric(self, dataset: InferenceDataset, metric_cfg: dict) -> InferenceOutput:
        n_feats     = dataset.X.shape[1]
        dummy       = np.zeros((len(dataset.X), n_feats), dtype=np.float32)
        dummy[:, 0] = dataset.X[:, 0]
        vals        = dataset.scaler.inverse_transform(dummy)[:, 0]

        if len(vals) < self.WE_MIN_SAMPLES:
            raise ValueError(f"Insufficient samples for WE rules: {len(vals)}")

        mean = float(np.mean(vals))
        std  = float(np.std(vals)) or 1e-9
        z    = (vals - mean) / std

        flags: dict[str, bool] = {}
        flags['rule_1_beyond_3sigma']              = bool(abs(z[-1]) > 3.0)
        if len(z) >= 3:
            last3 = z[-3:]
            flags['rule_2_two_of_three_beyond_2sigma'] = bool(
                np.sum(last3 > 2.0) >= 2 or np.sum(last3 < -2.0) >= 2
            )
        if len(z) >= 5:
            last5 = z[-5:]
            flags['rule_3_four_of_five_beyond_1sigma'] = bool(
                np.sum(last5 > 1.0) >= 4 or np.sum(last5 < -1.0) >= 4
            )
        if len(z) >= 8:
            last8 = z[-8:]
            flags['rule_4_eight_same_side'] = bool(np.all(last8 > 0) or np.all(last8 < 0))

        triggered  = [k for k, v in flags.items() if v]
        is_anomaly = len(triggered) > 0
        score      = len(triggered) / max(len(flags), 1)

        return InferenceOutput(
            analysis_type    = 'anomaly',
            method_used      = 'western_electric',
            is_anomaly       = is_anomaly,
            anomaly_score    = round(score, 4),
            confidence       = 0.9,
            data_points_used = len(vals),
            payload          = {
                'anomaly_flags'  : flags,
                'rules_triggered': triggered,
                'last_z_score'   : round(float(z[-1]), 4),
                'mean'           : round(mean, 6),
                'std'            : round(std, 6),
            },
        )

    # ---- Isolation Forest -----------------------------------------------

    def _isolation_forest(self, dataset: InferenceDataset, metric_cfg: dict) -> InferenceOutput:
        X = dataset.X     # already scaled

        if len(X) < 20:
            raise ValueError(f"Insufficient samples for Isolation Forest: {len(X)}")

        contamination = metric_cfg.get('anomaly', {}).get('contamination', 0.05)
        clf = IsolationForest(
            n_estimators  = 100,
            contamination = contamination,
            random_state  = 42,
            n_jobs        = -1,
        )
        clf.fit(X)

        last_row   = X[[-1]]
        pred       = int(clf.predict(last_row)[0])
        raw_score  = float(clf.decision_function(last_row)[0])
        norm_score = float(1 / (1 + np.exp(raw_score * 5)))    # sigmoid normalisation
        is_anomaly = pred == -1

        return InferenceOutput(
            analysis_type    = 'anomaly',
            method_used      = 'isolation_forest',
            is_anomaly       = is_anomaly,
            anomaly_score    = round(norm_score, 4),
            confidence       = 0.85,
            data_points_used = len(X),
            payload          = {
                'anomaly_flags'      : {'isolation_forest_flag': is_anomaly},
                'raw_decision_score' : round(raw_score, 6),
                'normalised_score'   : round(norm_score, 4),
            },
        )


# ---------------------------------------------------------------------------
# Health runner
# ---------------------------------------------------------------------------

class _HealthRunner:
    """
    Computes a 0-100 health score using linear regression on the target series.

    Components:
      degradation_score  : penalises positive slope (metric trending up = worse)
      anomaly_score      : penalises high fraction of recent points >2σ
      trend_score        : penalises strong R² with positive slope
    """

    def run(self, dataset: InferenceDataset, metric_cfg: dict) -> InferenceOutput:
        health_cfg = metric_cfg.get('health', {})
        n_feats    = dataset.X.shape[1]
        dummy      = np.zeros((len(dataset.X), n_feats), dtype=np.float32)
        dummy[:, 0] = dataset.X[:, 0]
        vals       = dataset.scaler.inverse_transform(dummy)[:, 0]

        if len(vals) < 5:
            return InferenceOutput(
                analysis_type = 'health',
                method_used   = 'linear_regression',
                health_score  = 100.0,
                confidence    = 0.0,
                payload       = {'error': 'insufficient data'},
            )

        x       = np.arange(len(vals), dtype=float)
        slope, intercept, r, p, se = scipy_stats.linregress(x, vals)
        r2      = r ** 2
        mean    = float(np.mean(vals))
        std     = float(np.std(vals)) or 1e-9

        degrad_weight  = float(health_cfg.get('degradation_weight', 0.4))
        anomaly_weight = float(health_cfg.get('anomaly_weight', 0.3))
        trend_weight   = float(health_cfg.get('trend_weight', 0.3))

        # Degradation: how fast is the metric rising relative to its mean?
        degrad_rate  = abs(slope) / (abs(mean) or 1.0)
        degrad_score = float(np.clip(1.0 - min(degrad_rate * 10, 1.0), 0.0, 1.0))

        # Anomaly rate: fraction of recent 20 points >2σ
        recent_z     = abs((vals[-min(20, len(vals)):] - mean) / std)
        anomaly_frac = float((recent_z > 2.0).mean())
        anomaly_score_component = float(np.clip(1.0 - anomaly_frac * 2, 0.0, 1.0))

        # Trend: penalise strong R² with positive slope
        trend_score = float(np.clip(1.0 - r2 * max(0.0, float(np.sign(slope))), 0.0, 1.0))

        health = (degrad_weight * degrad_score
                  + anomaly_weight * anomaly_score_component
                  + trend_weight * trend_score) * 100.0
        health = round(float(np.clip(health, 0.0, 100.0)), 2)

        return InferenceOutput(
            analysis_type    = 'health',
            method_used      = 'linear_regression',
            health_score     = health,
            confidence       = min(0.9, float(r2 + 0.3)),
            data_points_used = len(vals),
            payload          = {
                'score_components': {
                    'degradation_score': round(degrad_score * 100, 2),
                    'anomaly_score'    : round(anomaly_score_component * 100, 2),
                    'trend_score'      : round(trend_score * 100, 2),
                },
                'slope'            : round(float(slope), 8),
                'r_squared'        : round(float(r2), 4),
                'degradation_rate' : round(float(degrad_rate), 6),
            },
        )


# ---------------------------------------------------------------------------
# RUL runner
# ---------------------------------------------------------------------------

class _RULRunner:
    """
    Estimates Remaining Useful Life via linear extrapolation.

    Projects the current trend forward to failure_threshold.
    If no positive slope is detected, RUL is indeterminate (None).
    """

    def run(self, dataset: InferenceDataset, metric_cfg: dict) -> InferenceOutput:
        health_cfg        = metric_cfg.get('health', {})
        failure_threshold = health_cfg.get('failure_threshold')

        n_feats = dataset.X.shape[1]
        dummy   = np.zeros((len(dataset.X), n_feats), dtype=np.float32)
        dummy[:, 0] = dataset.X[:, 0]
        vals    = dataset.scaler.inverse_transform(dummy)[:, 0]

        if len(vals) < 5:
            return InferenceOutput(
                analysis_type = 'rul',
                method_used   = 'linear_regression',
                rul_cycles    = None,
                rul_seconds   = None,
                confidence    = 0.0,
                payload       = {'error': 'insufficient data'},
            )

        x = np.arange(len(vals), dtype=float)
        slope, intercept, r, _, _ = scipy_stats.linregress(x, vals)
        r2    = r ** 2
        mean  = float(np.mean(vals))

        if failure_threshold is None:
            failure_threshold = mean * 1.5

        current_value = float(vals[-1])

        if slope <= 0 or current_value >= failure_threshold:
            rul_cycles = 0.0 if current_value >= failure_threshold else None
        else:
            remaining  = failure_threshold - current_value
            rul_cycles = float(remaining / slope)

        # Convert cycles to seconds using timestamp spacing
        rul_seconds: Optional[float] = None
        if rul_cycles is not None and rul_cycles > 0:
            ts = dataset.timestamps
            if len(ts) >= 2 and hasattr(ts.iloc[0], 'total_seconds'):
                dt_median = ts.diff().median()
                rul_seconds = rul_cycles * dt_median.total_seconds()
            elif len(ts) >= 2:
                try:
                    delta = (pd.to_datetime(ts.iloc[-1]) - pd.to_datetime(ts.iloc[-2]))
                    rul_seconds = rul_cycles * delta.total_seconds()
                except Exception:
                    pass

        return InferenceOutput(
            analysis_type    = 'rul',
            method_used      = 'linear_regression',
            rul_cycles       = round(rul_cycles, 1) if rul_cycles is not None else None,
            rul_seconds      = round(rul_seconds, 1) if rul_seconds is not None else None,
            confidence       = min(0.85, float(r2 + 0.2)),
            data_points_used = len(vals),
            payload          = {
                'slope'             : round(float(slope), 8),
                'r_squared'         : round(float(r2), 4),
                'failure_threshold' : failure_threshold,
                'current_value'     : round(float(current_value), 6),
                'degradation_rate'  : round(abs(float(slope)) / (abs(mean) or 1.0), 6),
            },
        )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

import pandas as pd   # noqa: E402  (needed by _RULRunner)


class MambaInferencer:
    """
    Dispatcher that routes an InferenceDataset to the correct analysis runner.

    Parameters
    ----------
    model       : Mamba_TS       Loaded model (already on correct device).
    feature_names: list[str]     Feature schema logged at training time.
    metric_cfg  : dict           Single metric config block.
    device      : str            "cpu" or "cuda:N".
    """

    def __init__(
        self,
        model         : Mamba_TS,
        feature_names : list[str],
        metric_cfg    : dict,
        device        : str = "cpu",
    ) -> None:
        self._model         = model
        self._feature_names = feature_names
        self._metric_cfg    = metric_cfg
        self._device        = torch.device(device)

        self._forecast_runner = _ForecastRunner()
        self._anomaly_runner  = _AnomalyRunner()
        self._health_runner   = _HealthRunner()
        self._rul_runner      = _RULRunner()

    def predict(
        self,
        dataset      : InferenceDataset,
        analysis_type: str,
    ) -> InferenceOutput:
        """
        Run inference for a single analysis type.

        Parameters
        ----------
        dataset       : InferenceDataset from DataLoader.load_inference_window()
        analysis_type : one of 'forecast' | 'anomaly' | 'health' | 'rul'

        Returns
        -------
        InferenceOutput  — always returned, never raises.
        """
        # Schema validation
        try:
            _validate_schema(dataset, self._feature_names)
        except SchemaValidationError as exc:
            logger.error("MambaInferencer: schema mismatch for %s: %s", analysis_type, exc)
            return InferenceOutput(
                analysis_type = analysis_type,
                method_used   = 'none',
                confidence    = 0.0,
                payload       = {'error': f'schema_mismatch: {exc}'},
            )

        try:
            if analysis_type == 'forecast':
                horizon = self._metric_cfg.get('analysis', {}).get('forecast_horizon', 10)
                return self._forecast_runner.run(
                    self._model, dataset, horizon, self._device
                )

            elif analysis_type == 'anomaly':
                method   = self._metric_cfg.get('analysis', {}).get('anomaly_method', 'zscore')
                fallback = self._metric_cfg.get('analysis', {}).get('anomaly_fallback', 'zscore')
                return self._anomaly_runner.run(dataset, method, fallback, self._metric_cfg)

            elif analysis_type == 'health':
                return self._health_runner.run(dataset, self._metric_cfg)

            elif analysis_type == 'rul':
                return self._rul_runner.run(dataset, self._metric_cfg)

            else:
                logger.error("MambaInferencer: unknown analysis_type '%s'", analysis_type)
                return InferenceOutput(
                    analysis_type = analysis_type,
                    method_used   = 'none',
                    confidence    = 0.0,
                    payload       = {'error': f'unknown analysis_type: {analysis_type}'},
                )

        except Exception as exc:
            logger.error("MambaInferencer: unhandled error in %s: %s", analysis_type, exc,
                         exc_info=True)
            return InferenceOutput(
                analysis_type = analysis_type,
                method_used   = 'none',
                confidence    = 0.0,
                payload       = {'error': str(exc)},
            )
