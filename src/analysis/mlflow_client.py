"""
tracking/mlflow_client.py
───────────────────────────
Thin wrapper around the MLflow Python SDK.

All MLflow interactions in the system go through this class.
It handles experiment creation, run logging, artifact association,
model registration, and stage promotion.

Self-hosted MLflow server is expected at MLFLOW_TRACKING_URI.
Artifacts are stored on the local filesystem at MLFLOW_ARTIFACT_ROOT,
which is a volume mounted by both the MLflow container and this container.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from config.config_schema import MetricConfig
from models.trainers.mamba import TrainResult

logger = logging.getLogger(__name__)


class MLflowTrackingClient:
    """
    Parameters
    ----------
    tracking_uri  : URI of self-hosted MLflow server, e.g. "http://mlflow:5000"
    artifact_root : local path where MLflow writes artifact files,
                    e.g. "/mlflow/artifacts" (must match server config)
    """

    def __init__(self, tracking_uri: str, artifact_root: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._client        = MlflowClient()
        self._artifact_root = Path(artifact_root)

    # ──────────────────────────────────────────────────────────────────────
    # Experiment management
    # ──────────────────────────────────────────────────────────────────────

    def get_or_create_experiment(self, config: MetricConfig) -> str:
        """
        Return the experiment_id for "<station>/<metric>".
        Creates the experiment if it doesn't exist.
        """
        name = config.experiment_name
        exp  = mlflow.get_experiment_by_name(name)
        if exp is not None:
            return exp.experiment_id

        artifact_location = str(self._artifact_root / config.station / config.metric)
        exp_id = mlflow.create_experiment(name, artifact_location=artifact_location)
        logger.info("Created MLflow experiment '%s' (id=%s)", name, exp_id)
        return exp_id

    # ──────────────────────────────────────────────────────────────────────
    # Training run logging
    # ──────────────────────────────────────────────────────────────────────

    def log_training_run(
        self,
        config      : MetricConfig,
        train_result: TrainResult,
    ) -> str:
        """
        Log a completed training run to MLflow.

        Logs
        ----
        Params  : all hyperparameters from train_result.hyperparameters
        Metrics : val_loss, val_mae, val_rmse, val_r2, n_epochs, training_seconds
        Tags    : station, metric, stopped_early
        Artifacts: the directory at train_result.artifact_dir
                   (model.pt, scaler.pkl, feature_schema.json, model_config.json)

        Returns
        -------
        MLflow run_id string.
        """
        exp_id = self.get_or_create_experiment(config)

        with mlflow.start_run(experiment_id=exp_id) as run:
            run_id = run.info.run_id

            # Params
            mlflow.log_params(train_result.hyperparameters)

            # Metrics
            mlflow.log_metrics({
                "val_loss"         : train_result.val_loss,
                "val_mae"          : train_result.val_mae,
                "val_rmse"         : train_result.val_rmse,
                "val_r2"           : train_result.val_r2,
                "n_epochs_trained" : train_result.n_epochs_trained,
                "training_seconds" : train_result.training_seconds,
            })

            # Epoch-level loss curves (logged as individual steps)
            for step, (tr_l, vl_l) in enumerate(
                zip(train_result.train_loss_history, train_result.val_loss_history)
            ):
                mlflow.log_metric("train_loss", tr_l, step=step)
                mlflow.log_metric("val_loss_epoch", vl_l, step=step)

            # Tags
            mlflow.set_tags({
                "station"      : config.station,
                "metric"       : config.metric,
                "stopped_early": str(train_result.stopped_early),
                "architecture" : "Mamba_TS",
            })

            # Artifact directory
            mlflow.log_artifacts(train_result.artifact_dir, artifact_path="model")

        logger.info(
            "Logged training run %s for %s/%s (val_rmse=%.4f, val_r2=%.4f)",
            run_id, config.station, config.metric,
            train_result.val_rmse, train_result.val_r2,
        )
        return run_id

    # ──────────────────────────────────────────────────────────────────────
    # Model registration
    # ──────────────────────────────────────────────────────────────────────

    def register_model(
        self,
        config        : MetricConfig,
        run_id        : str,
        artifact_path : str = "model",
    ) -> "mlflow.entities.model_registry.ModelVersion":
        """
        Register the model logged in `run_id` to the MLflow Model Registry
        under the name config.model_registry_name, in "Staging" stage.

        Returns the new ModelVersion object.
        """
        name       = config.model_registry_name
        model_uri  = f"runs:/{run_id}/{artifact_path}"

        mv = mlflow.register_model(model_uri=model_uri, name=name)
        logger.info(
            "Registered model '%s' version %s (run=%s)",
            name, mv.version, run_id,
        )
        return mv

    def promote_to_production(
        self,
        config  : MetricConfig,
        version : str,
    ) -> None:
        """
        Transition `version` of config.model_registry_name to "Production".
        Archives any existing Production version automatically.
        """
        name = config.model_registry_name
        self._client.transition_model_version_stage(
            name    = name,
            version = version,
            stage   = "Production",
            archive_existing_versions = True,
        )
        logger.info("Promoted '%s' v%s → Production", name, version)

    # ──────────────────────────────────────────────────────────────────────
    # Inference logging (lightweight — not a full MLflow run)
    # ──────────────────────────────────────────────────────────────────────

    def log_inference_event(
        self,
        config        : MetricConfig,
        model_version : str,
        mlflow_run_id : str,
        output_summary: dict,
    ) -> None:
        """
        Log a lightweight record of an inference event.
        Writes a JSON file to the artifact store; does not start an MLflow run.
        Useful for auditing which model version served which prediction.
        """
        event = {
            "timestamp"     : datetime.now(timezone.utc).isoformat(),
            "station"       : config.station,
            "metric"        : config.metric,
            "model_version" : model_version,
            "train_run_id"  : mlflow_run_id,
            **output_summary,
        }
        log_path = (
            self._artifact_root
            / config.station
            / config.metric
            / "inference_log.jsonl"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    def get_production_model_info(self, config: MetricConfig) -> Optional[dict]:
        """
        Return {version, run_id, artifact_uri, registered_at} for the current
        Production model, or None if no Production model exists.
        """
        try:
            versions = self._client.get_latest_versions(
                config.model_registry_name, stages=["Production"]
            )
        except Exception:
            return None

        if not versions:
            return None

        mv = versions[0]
        return {
            "version"       : mv.version,
            "run_id"        : mv.run_id,
            "artifact_uri"  : mv.source,
            "registered_at" : mv.creation_timestamp,
        }
