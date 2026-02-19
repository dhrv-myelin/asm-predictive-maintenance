"""
models/inferencers/mamba.py
─────────────────────────────
Inference logic for the Mamba_TS model.

Responsibilities
----------------
1. Receive a LoadedModel (model in memory, on device) + InferenceDataset.
2. Validate input schema against the feature_schema logged at training time.
3. Run model.forward() in no_grad context.
4. Denormalise prediction using the stored scaler.
5. Produce multi-step forecast via recursive rollout.
6. Return a structured ForecastOutput dataclass.

The class is intentionally thin — it does NOT touch the DB, MLflow, or the
training pipeline. It only knows how to call model.forward() and interpret
the result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from config.config_schema import MetricConfig
from data.data_loader import InferenceDataset
from models.architectures.mamba import Mamba_TS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ForecastOutput:
    """Structured output from MambaInferencer.predict()."""
    station_name      : str
    metric_name       : str
    target_column     : str

    # Forecast
    forecast_values   : list[float]     # predicted values (denormalised), length = horizon
    forecast_horizon  : int
    lower_bound       : list[float]     # 95% CI lower
    upper_bound       : list[float]     # 95% CI upper

    # Meta
    latest_observed   : float           # most recent raw value in the input window
    model_version     : str             # MLflow model version string
    mlflow_run_id     : str
    feature_names     : list[str]
    confidence        : float           # 0–1, based on val_r2 logged at training time
    data_points_used  : int

    # Anomaly signal derived from last prediction vs observed
    predicted_vs_observed_delta : float
    is_forecast_anomaly         : bool  # True if delta > 2σ of training residuals

    error             : Optional[str] = None


# ---------------------------------------------------------------------------
# Inferencer
# ---------------------------------------------------------------------------

class MambaInferencer:
    """
    Runs inference with a loaded Mamba_TS model.

    Parameters
    ----------
    model        : Mamba_TS instance (already on device, eval mode)
    feature_schema : dict loaded from feature_schema.json artifact
                     keys: feature_names, target_col, seq_len, n_features
    scaler       : fitted sklearn scaler loaded from scaler.pkl artifact
    config       : MetricConfig for this station+metric
    model_version: MLflow version string (for traceability in output)
    mlflow_run_id: MLflow run id (for traceability)
    device       : "cpu" or "cuda"
    training_residual_std : std of training prediction errors (for CI width)
                           loaded from model_config.json if available
    """

    def __init__(
        self,
        model                  : Mamba_TS,
        feature_schema         : dict,
        scaler,
        config                 : MetricConfig,
        model_version          : str = "unknown",
        mlflow_run_id          : str = "unknown",
        device                 : str = "cpu",
        training_residual_std  : float = 0.1,
    ) -> None:
        self._model          = model
        self._schema         = feature_schema
        self._scaler         = scaler
        self._config         = config
        self._model_version  = model_version
        self._mlflow_run_id  = mlflow_run_id
        self._device         = device
        self._residual_std   = training_residual_std

        # Target column is always index 0 in feature matrix (DataLoader contract)
        self._target_idx = 0

    # ──────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────

    def predict(self, dataset: InferenceDataset) -> ForecastOutput:
        """
        Run multi-step forecast.

        Steps
        -----
        1. Validate that dataset.feature_names matches the training schema.
        2. Convert dataset.X to a torch tensor on the correct device.
        3. Roll out forecast_horizon predictions recursively:
           each predicted value is fed back as the next input step.
        4. Denormalise predictions using the stored scaler.
        5. Compute 95% CI as pred ± 1.96 * residual_std (in scaled space,
           then denormalised).
        6. Return ForecastOutput.
        """
        cfg     = self._config
        horizon = cfg.mamba.seq_len   # default horizon = seq_len; override via config if needed

        # Schema validation
        schema_err = self._validate_schema(dataset)
        if schema_err:
            return self._error_output(schema_err, dataset)

        # Build input tensor: (1, seq_len, n_features) → extract target channel
        X_tensor = torch.FloatTensor(dataset.X).to(self._device)  # (1, seq_len, n_features)

        # Recursive multi-step rollout
        forecasts_scaled = self._rollout(X_tensor, horizon)

        # Denormalise — scaler was fit on all features; target is index 0
        forecasts_denorm = self._denormalise(forecasts_scaled)
        ci_half          = 1.96 * abs(self._residual_std)   # in scaled space
        lower_denorm     = self._denormalise(
            [f - ci_half for f in forecasts_scaled]
        )
        upper_denorm     = self._denormalise(
            [f + ci_half for f in forecasts_scaled]
        )

        # Anomaly check: compare first predicted step vs latest observed
        delta  = abs(forecasts_denorm[0] - dataset.latest_value)
        # Threshold: 2 × denormalised residual std
        thresh = abs(self._denormalise([self._residual_std])[0] - self._denormalise([0.0])[0]) * 2
        is_anom = delta > thresh

        return ForecastOutput(
            station_name     = cfg.station,
            metric_name      = cfg.metric,
            target_column    = cfg.target_column,
            forecast_values  = [round(v, 6) for v in forecasts_denorm],
            forecast_horizon = horizon,
            lower_bound      = [round(v, 6) for v in lower_denorm],
            upper_bound      = [round(v, 6) for v in upper_denorm],
            latest_observed  = dataset.latest_value,
            model_version    = self._model_version,
            mlflow_run_id    = self._mlflow_run_id,
            feature_names    = dataset.feature_names,
            confidence       = self._estimate_confidence(),
            data_points_used = dataset.X.shape[1],
            predicted_vs_observed_delta = round(delta, 6),
            is_forecast_anomaly         = is_anom,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _rollout(self, X_tensor: torch.Tensor, horizon: int) -> list[float]:
        """
        Recursive single-step-ahead rollout.

        At each step:
          - Pass the current window through the model.
          - Take pred[:, 0] as the forecast for step t+1.
          - Shift the window left by 1, appending the new prediction.
          - Repeat.

        All arithmetic in scaled space; denormalisation happens after.
        """
        self._model.eval()
        window  = X_tensor.clone()   # (1, seq_len, n_features)
        results = []

        with torch.no_grad():
            for _ in range(horizon):
                x_target         = window[:, :, self._target_idx]   # (1, seq_len)
                _, pred           = self._model(x_target)
                pred_val          = float(pred.squeeze().item())
                results.append(pred_val)

                # Shift window: drop oldest step, append new step
                new_step         = window[:, -1:, :].clone()         # (1, 1, n_features)
                new_step[:, 0, self._target_idx] = pred_val
                window           = torch.cat([window[:, 1:, :], new_step], dim=1)

        return results

    def _denormalise(self, scaled_values: list[float]) -> list[float]:
        """
        Inverse-transform a list of scaled target values.
        The scaler was fit on the full feature matrix; target is column 0.
        We create a dummy row to call inverse_transform correctly.
        """
        n_features = self._schema["n_features"]
        dummy      = np.zeros((len(scaled_values), n_features), dtype=np.float32)
        dummy[:, self._target_idx] = scaled_values

        restored   = self._scaler.inverse_transform(dummy)
        return restored[:, self._target_idx].tolist()

    def _validate_schema(self, dataset: InferenceDataset) -> Optional[str]:
        """
        Returns an error string if the dataset doesn't match the training schema,
        None if everything is fine.
        """
        required = set(self._schema["feature_names"])
        provided = set(dataset.feature_names)
        missing  = required - provided
        if missing:
            return f"Schema mismatch: missing features {missing}"
        if dataset.X.shape[1] != self._schema["seq_len"]:
            return (
                f"Sequence length mismatch: expected {self._schema['seq_len']}, "
                f"got {dataset.X.shape[1]}"
            )
        return None

    def _estimate_confidence(self) -> float:
        """
        Simple confidence proxy: lower residual_std → higher confidence.
        Maps [0, 0.5] std range to [1.0, 0.5] confidence (clipped).
        """
        return float(np.clip(1.0 - self._residual_std * 2, 0.5, 1.0))

    def _error_output(self, error: str, dataset: InferenceDataset) -> ForecastOutput:
        cfg = self._config
        return ForecastOutput(
            station_name     = cfg.station,
            metric_name      = cfg.metric,
            target_column    = cfg.target_column,
            forecast_values  = [],
            forecast_horizon = 0,
            lower_bound      = [],
            upper_bound      = [],
            latest_observed  = dataset.latest_value,
            model_version    = self._model_version,
            mlflow_run_id    = self._mlflow_run_id,
            feature_names    = dataset.feature_names,
            confidence       = 0.0,
            data_points_used = 0,
            predicted_vs_observed_delta = 0.0,
            is_forecast_anomaly         = False,
            error            = error,
        )
