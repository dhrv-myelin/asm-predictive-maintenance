"""
registry/model_loader.py
──────────────────────────
Connects MLflow Model Registry → local filesystem → in-memory model cache.

Responsibilities
----------------
1. Query MLflow for the "Production" model version for a given station+metric.
2. Download/locate artifacts (model.pt, scaler.pkl, feature_schema.json,
   model_config.json) from the local artifact path stored in MLflow.
3. Deserialise: load state_dict into Mamba_TS, unpickle scaler, parse schema.
4. Place model on the correct device (CPU or CUDA) via DeviceManager.
5. Cache the loaded model so the same instance is reused across inference calls
   (one cache entry per station+metric — shared where architectures match).
6. Expose reload() so the orchestrator can hot-swap a newly promoted model.

Cache key: "<station>/<metric>"  — one entry per station+metric pair.
This means the same Mamba_TS architecture is shared across analysis types
(forecast output is reused for anomaly scoring based on reconstruction error).
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import torch
from mlflow.tracking import MlflowClient

from config.config_schema import MetricConfig
from models.architectures.mamba import Mamba_TS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """Everything needed to run inference for one station+metric."""
    model          : Mamba_TS
    scaler         : object             # fitted sklearn scaler
    feature_schema : dict               # from feature_schema.json
    model_config   : dict               # from model_config.json
    model_version  : str
    mlflow_run_id  : str
    device         : str
    artifact_dir   : Path
    loaded_at      : datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def training_residual_std(self) -> float:
        """Residual std logged at training time, used for CI width in inferencer."""
        return float(self.model_config.get("val_rmse", 0.1))


@dataclass
class LoadedModelSummary:
    station        : str
    metric         : str
    model_version  : str
    mlflow_run_id  : str
    device         : str
    loaded_at      : str
    n_parameters   : int


# ---------------------------------------------------------------------------
# DeviceManager
# ---------------------------------------------------------------------------

class DeviceManager:
    """
    Centralised CUDA device assignment.
    Detects available GPUs at startup and round-robins assignments.
    """

    def __init__(self) -> None:
        self._n_gpus    = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._rr_index  = 0
        if self._n_gpus > 0:
            logger.info("DeviceManager: %d CUDA device(s) available.", self._n_gpus)
        else:
            logger.info("DeviceManager: no CUDA devices found; using CPU.")

    def get_device(self, config: MetricConfig) -> str:
        """
        Return a device string for the given metric config.

        Rules
        -----
        "cpu"  → always CPU
        "cuda" → CUDA (error if unavailable)
        "auto" → CUDA if available (round-robin across GPUs), else CPU
        """
        pref = config.mamba.device
        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            if self._n_gpus == 0:
                logger.warning("CUDA requested for %s/%s but no GPU found; using CPU.",
                               config.station, config.metric)
                return "cpu"
            return self._next_cuda()
        # auto
        if self._n_gpus > 0:
            return self._next_cuda()
        return "cpu"

    def place(self, model: Mamba_TS, device: str) -> Mamba_TS:
        """Move model to device and set eval mode."""
        return model.to(device).eval()

    def _next_cuda(self) -> str:
        device = f"cuda:{self._rr_index % self._n_gpus}"
        self._rr_index += 1
        return device


# ---------------------------------------------------------------------------
# ModelLoader
# ---------------------------------------------------------------------------

class ModelLoader:
    """
    Manages the lifecycle of loaded Mamba_TS models.

    Parameters
    ----------
    mlflow_tracking_uri : URI of the self-hosted MLflow server
    artifact_base       : local filesystem root where MLflow stores artifacts
                          (must match MLFLOW_ARTIFACT_ROOT in the MLflow container)
    device_manager      : DeviceManager instance (shared across all loaders)
    """

    def __init__(
        self,
        mlflow_tracking_uri : str,
        artifact_base       : str,
        device_manager      : Optional[DeviceManager] = None,
    ) -> None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self._client        = MlflowClient()
        self._artifact_base = Path(artifact_base)
        self._device_mgr    = device_manager or DeviceManager()
        self._cache         : dict[str, LoadedModel] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def get_or_load(self, config: MetricConfig) -> Optional[LoadedModel]:
        """
        Return a cached LoadedModel if available, otherwise load from
        the MLflow Model Registry.

        Returns None if no Production model exists yet for this station+metric.
        """
        key = self._cache_key(config)
        if key in self._cache:
            return self._cache[key]
        return self._load_from_registry(config)

    def reload(self, config: MetricConfig) -> Optional[LoadedModel]:
        """
        Evict the cache entry and reload from MLflow.
        Called after a new model version is promoted to Production.
        """
        key = self._cache_key(config)
        if key in self._cache:
            del self._cache[key]
            logger.info("ModelLoader: evicted cache for %s/%s", config.station, config.metric)
        return self._load_from_registry(config)

    def is_model_available(self, config: MetricConfig) -> bool:
        """
        True if a Production model exists in the MLflow registry
        for this station+metric.
        """
        try:
            mv = self._get_production_version(config.model_registry_name)
            return mv is not None
        except Exception:
            return False

    def list_loaded_models(self) -> list[LoadedModelSummary]:
        summaries = []
        for key, lm in self._cache.items():
            station, metric = key.split("/", 1)
            summaries.append(LoadedModelSummary(
                station       = station,
                metric        = metric,
                model_version = lm.model_version,
                mlflow_run_id = lm.mlflow_run_id,
                device        = lm.device,
                loaded_at     = lm.loaded_at.isoformat(),
                n_parameters  = lm.model.n_parameters,
            ))
        return summaries

    # ──────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────

    def _load_from_registry(self, config: MetricConfig) -> Optional[LoadedModel]:
        """
        Full load sequence:
          1. Query MLflow for Production version.
          2. Resolve local artifact path.
          3. Load scaler, feature_schema, model_config.
          4. Instantiate Mamba_TS from model_config, load state_dict.
          5. Place on device, set eval mode.
          6. Cache and return.
        """
        registry_name = config.model_registry_name
        try:
            mv = self._get_production_version(registry_name)
        except Exception as exc:
            logger.warning("MLflow registry query failed for %s: %s", registry_name, exc)
            return None

        if mv is None:
            logger.info("No Production model in registry for %s", registry_name)
            return None

        # Resolve artifact directory
        # MLflow stores artifact_uri as  file:///mlflow/artifacts/<station>/<metric>
        # We strip the scheme and use the path directly.
        art_uri  = mv.source                          # e.g. file:///mlflow/artifacts/...
        art_path = Path(art_uri.replace("file://", ""))
        if not art_path.exists():
            # Fallback: construct path from our known artifact base
            art_path = self._artifact_base / config.station / config.metric
        if not art_path.exists():
            logger.error("Artifact directory not found for %s: %s", registry_name, art_path)
            return None

        # Deserialise artifacts
        try:
            scaler         = self._load_scaler(art_path)
            feature_schema = self._load_json(art_path / "feature_schema.json")
            model_config   = self._load_json(art_path / "model_config.json")
            model          = self._load_model(art_path, model_config, config)
        except Exception as exc:
            logger.error("Failed to load artifacts for %s: %s", registry_name, exc)
            return None

        device = self._device_mgr.get_device(config)
        model  = self._device_mgr.place(model, device)

        loaded = LoadedModel(
            model          = model,
            scaler         = scaler,
            feature_schema = feature_schema,
            model_config   = model_config,
            model_version  = mv.version,
            mlflow_run_id  = mv.run_id,
            device         = device,
            artifact_dir   = art_path,
        )

        key              = self._cache_key(config)
        self._cache[key] = loaded
        logger.info(
            "ModelLoader: loaded %s/%s v%s on %s",
            config.station, config.metric, mv.version, device,
        )
        return loaded

    def _load_model(
        self,
        art_path     : Path,
        model_config : dict,
        config       : MetricConfig,
    ) -> Mamba_TS:
        """Instantiate Mamba_TS from saved config and load state_dict."""
        model = Mamba_TS(
            seq_len    = model_config.get("seq_len",    config.mamba.seq_len),
            hidden_dim = model_config.get("hidden_dim", config.mamba.hidden_dim),
            d_state    = model_config.get("d_state",    config.mamba.d_state),
            d_conv     = model_config.get("d_conv",     config.mamba.d_conv),
            num_layers = model_config.get("num_layers", config.mamba.num_layers),
            dropout    = model_config.get("dropout",    config.mamba.dropout),
        )
        state_dict = torch.load(art_path / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def _load_scaler(self, art_path: Path):
        with open(art_path / "scaler.pkl", "rb") as f:
            return pickle.load(f)

    def _load_json(self, path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    def _get_production_version(self, registry_name: str):
        """Return the Production ModelVersion object, or None."""
        try:
            versions = self._client.get_latest_versions(
                registry_name, stages=["Production"]
            )
            return versions[0] if versions else None
        except mlflow.exceptions.MlflowException:
            return None

    @staticmethod
    def _cache_key(config: MetricConfig) -> str:
        return f"{config.station}/{config.metric}"
