"""
config/config_schema.py
─────────────────────────
Pydantic v2 models that represent a validated, merged config entry.
ConfigLoader reads analysis_config.yaml and produces MetricConfig instances.

The merge logic:  global defaults → metric-level overrides (deep merge).
Consumers always read from MetricConfig — never from raw YAML dicts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class DBConfig(BaseModel):
    database_url_env : str = "DATABASE_URL"
    table            : str = "process_metrics"
    schema_name      : str = Field("public", alias="schema")

    @property
    def database_url(self) -> str:
        url = os.environ.get(self.database_url_env)
        if not url:
            raise EnvironmentError(
                f"Environment variable '{self.database_url_env}' is not set."
            )
        return url

    model_config = {"populate_by_name": True}


class DataConfig(BaseModel):
    training_lookback_days    : int   = 7
    inference_lookback_cycles : int   = 50
    min_train_samples         : int   = 30
    val_split                 : float = 0.2
    scaler                    : Literal["robust", "minmax", "standard"] = "robust"


class TrainingConfig(BaseModel):
    retrain_trigger         : Literal["drift", "manual", "both"] = "both"
    drift_threshold_psi     : float = 0.2
    drift_threshold_ks_pvalue: float = 0.05
    max_model_age_days      : int   = 14


class MambaConfig(BaseModel):
    seq_len           : int   = 12
    hidden_dim        : int   = 64
    d_state           : int   = 16
    d_conv            : int   = 4
    num_layers        : int   = 2
    dropout           : float = 0.2
    learning_rate     : float = 0.001
    num_epochs        : int   = 200
    loss_weight_recon : float = 0.3
    device            : Literal["auto", "cpu", "cuda"] = "auto"


class MLflowConfig(BaseModel):
    tracking_uri_env  : str = "MLFLOW_TRACKING_URI"
    artifact_root_env : str = "MLFLOW_ARTIFACT_ROOT"

    @property
    def tracking_uri(self) -> str:
        return os.environ.get(self.tracking_uri_env, "http://mlflow:5000")

    @property
    def artifact_root(self) -> str:
        return os.environ.get(self.artifact_root_env, "/mlflow/artifacts")


class AnomalyConfig(BaseModel):
    method             : Literal["zscore", "western_electric", "isolation_forest"] = "zscore"
    zscore_threshold   : float      = 3.0
    zscore_window      : int        = 50
    we_rules           : list[int]  = Field(default_factory=lambda: [1, 2, 3, 4])
    if_contamination   : float      = 0.05
    if_n_estimators    : int        = 100


class HealthRULConfig(BaseModel):
    degradation_weight            : float = 0.4
    anomaly_weight                : float = 0.3
    trend_weight                  : float = 0.3
    failure_threshold_multiplier  : float = 1.5

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "HealthRULConfig":
        total = self.degradation_weight + self.anomaly_weight + self.trend_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"health_rul weights must sum to 1.0, got {total:.3f}"
            )
        return self


# ---------------------------------------------------------------------------
# Top-level per-metric config
# ---------------------------------------------------------------------------

class MetricConfig(BaseModel):
    """
    Fully resolved config for one station+metric pair.
    Produced by ConfigLoader.load() after merging global defaults
    with metric-level overrides.
    """
    station          : str
    metric           : str
    target_column    : str
    analysis_types   : list[Literal["forecast", "anomaly", "health_score", "rul"]]
    required_features: list[str]

    db        : DBConfig
    data      : DataConfig
    training  : TrainingConfig
    mamba     : MambaConfig
    mlflow    : MLflowConfig
    anomaly   : AnomalyConfig
    health_rul: HealthRULConfig

    @property
    def experiment_name(self) -> str:
        """MLflow experiment namespace: '<station>/<metric>'"""
        return f"{self.station}/{self.metric}"

    @property
    def model_registry_name(self) -> str:
        """MLflow model registry key: '<station>__<metric>'"""
        return f"{self.station}__{self.metric}"


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent / "analysis_config.yaml"


class ConfigLoader:
    """
    Reads analysis_config.yaml once, validates every metric entry,
    and caches the list of MetricConfig objects.

    Usage:
        loader = ConfigLoader()
        cfg = loader.load("system", "cycle_time")
        all_cfgs = loader.list_all()
    """

    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        self._path   = config_path
        self._cache  : dict[str, MetricConfig] = {}
        self._reload()

    def _reload(self) -> None:
        with open(self._path) as f:
            raw = yaml.safe_load(f)

        global_raw = raw.get("global", {})
        self._cache = {}

        for entry in raw.get("metrics", []):
            key = f"{entry['station']}/{entry['metric']}"
            cfg = self._merge_and_validate(global_raw, entry)
            self._cache[key] = cfg

    def reload(self) -> None:
        """Force re-read from disk (call after editing the YAML)."""
        self._reload()

    def load(self, station: str, metric: str) -> MetricConfig:
        key = f"{station}/{metric}"
        if key not in self._cache:
            raise KeyError(
                f"No config found for '{key}'. "
                f"Available: {list(self._cache.keys())}"
            )
        return self._cache[key]

    def list_all(self) -> list[MetricConfig]:
        return list(self._cache.values())

    # ── merge helper ────────────────────────────────────────────────────────

    @staticmethod
    def _merge_and_validate(global_raw: dict, metric_raw: dict) -> MetricConfig:
        """
        Deep-merge global defaults with metric-level overrides.
        metric_raw keys win over global_raw keys at every sub-config level.
        """
        def _merge(base: dict, override: dict) -> dict:
            result = dict(base)
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(result.get(k), dict):
                    result[k] = _merge(result[k], v)
                else:
                    result[k] = v
            return result

        merged = _merge(global_raw, metric_raw)

        return MetricConfig(
            station           = merged["station"],
            metric            = merged["metric"],
            target_column     = merged["target_column"],
            analysis_types    = merged["analysis_types"],
            required_features = merged.get("required_features", [merged["target_column"]]),
            db                = DBConfig(**merged.get("db", {})),
            data              = DataConfig(**merged.get("data", {})),
            training          = TrainingConfig(**merged.get("training", {})),
            mamba             = MambaConfig(**merged.get("mamba", {})),
            mlflow            = MLflowConfig(**merged.get("mlflow", {})),
            anomaly           = AnomalyConfig(**merged.get("anomaly", {})),
            health_rul        = HealthRULConfig(**merged.get("health_rul", {})),
        )
