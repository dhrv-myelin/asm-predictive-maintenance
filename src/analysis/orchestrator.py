"""
orchestrator/orchestrator.py
──────────────────────────────
Central router for the inference system. Contains no ML logic itself.

Decision tree for each station+metric (called per inference run):
  1. Is a Production model available in MLflow?
       No  → _train()
  2. Is the model older than max_model_age_days?
       Yes → _train()
  3. Does drift_detector report drift AND retrain_trigger includes "drift"?
       Yes → _train()
  4. Otherwise → _infer()

After training, the new model is registered, promoted to Production,
and hot-reloaded into the ModelLoader cache before inference continues.

All results are persisted to the inference_results table (DB) and
an InferenceRun row (summary) is created/updated.
"""

from __future__ import annotations

import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from config.config_schema import ConfigLoader, MetricConfig
from data.data_loader import DataLoader, TrainingDataset, InferenceDataset
from drift.drift_detector import DriftDetector, DriftReport
from models.inferencers.mamba import MambaInferencer, ForecastOutput
from models.trainers.mamba import MambaTrainer, TrainResult
from registry.model_loader import ModelLoader, LoadedModel
from tracking.mlflow_client import MLflowTrackingClient

# Import DB models from the existing models.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from db.models import (
    AnalysisType, ErrorLog, InferenceResult, InferenceRun,
    MethodName, RunStatus, Severity,
)

logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    """
    Parameters
    ----------
    session_factory     : callable → context-manager yielding SQLAlchemy Session
    config_loader       : ConfigLoader (pre-loaded from analysis_config.yaml)
    model_loader        : ModelLoader (shared model cache)
    mlflow_client       : MLflowTrackingClient
    artifact_base       : local path for training artifacts (passed to MambaTrainer)
    max_workers         : parallel workers for station+metric processing
    """

    def __init__(
        self,
        session_factory : object,
        config_loader   : ConfigLoader,
        model_loader    : ModelLoader,
        mlflow_client   : MLflowTrackingClient,
        artifact_base   : str  = "/mlflow/artifacts",
        max_workers     : int  = 4,
    ) -> None:
        self._sf            = session_factory
        self._config_loader = config_loader
        self._model_loader  = model_loader
        self._mlflow        = mlflow_client
        self._artifact_base = artifact_base
        self._max_workers   = max_workers

    # ──────────────────────────────────────────────────────────────────────
    # Public entry points
    # ──────────────────────────────────────────────────────────────────────

    def run(self, triggered_by: str = "db_poll") -> str:
        """
        Execute a full inference cycle over all configured station+metric pairs.
        Returns the run_id of the created InferenceRun.
        """
        run_id = str(uuid.uuid4())
        now    = datetime.now(timezone.utc)

        self._create_run_row(run_id, triggered_by, now)
        logger.info("InferenceRun %s started (trigger=%s)", run_id, triggered_by)

        configs = self._config_loader.list_all()
        if not configs:
            logger.warning("No metric configs loaded; nothing to do.")
            self._finalise_run(run_id, RunStatus.COMPLETED)
            return run_id

        results: list[Optional[bool]] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(self._process_metric, run_id, cfg): cfg
                for cfg in configs
            }
            for future in as_completed(futures):
                cfg = futures[future]
                try:
                    future.result()
                    results.append(True)
                except Exception as exc:
                    logger.error(
                        "Worker failed for %s/%s: %s", cfg.station, cfg.metric, exc
                    )
                    self._db_log_error(
                        module       = "orchestrator.worker",
                        message      = f"{cfg.station}/{cfg.metric}: {exc}",
                        run_id       = run_id,
                        station_name = cfg.station,
                        metric_name  = cfg.metric,
                        tb           = traceback.format_exc(),
                    )
                    results.append(False)

        n_failed  = results.count(False)
        n_total   = len(results)
        status    = (
            RunStatus.FAILED   if n_failed == n_total and n_total > 0 else
            RunStatus.PARTIAL  if n_failed > 0 else
            RunStatus.COMPLETED
        )
        self._finalise_run(run_id, status)
        logger.info(
            "InferenceRun %s done: %d/%d succeeded",
            run_id, n_total - n_failed, n_total,
        )
        return run_id

    def retrain_manual(self, station: str, metric: str) -> Optional[str]:
        """
        Force retrain for a specific station+metric, bypassing drift check.
        Returns the MLflow run_id, or None on failure.
        """
        try:
            cfg = self._config_loader.load(station, metric)
        except KeyError as exc:
            logger.error("Manual retrain: %s", exc)
            return None

        logger.info("Manual retrain triggered for %s/%s", station, metric)
        return self._train(cfg, run_id="manual")

    # ──────────────────────────────────────────────────────────────────────
    # Per-metric pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _process_metric(self, run_id: str, cfg: MetricConfig) -> None:
        """
        Full pipeline for one station+metric.
        Decides train vs infer; persists results.
        """
        data_loader = DataLoader(self._sf, cfg)
        loaded_model = self._model_loader.get_or_load(cfg)

        # ── Decision: train or infer? ──────────────────────────────────────
        should_train = False
        train_reason = ""

        if loaded_model is None:
            should_train = True
            train_reason = "no Production model in registry"

        elif self._model_is_stale(loaded_model, cfg):
            should_train = True
            train_reason = f"model age > {cfg.training.max_model_age_days} days"

        elif cfg.training.retrain_trigger in ("drift", "both"):
            drift_data = data_loader.load_drift_window()
            if drift_data is not None:
                report = DriftDetector(cfg).check(drift_data)
                if report.drifted:
                    should_train = True
                    train_reason = f"drift detected: {report.drift_reason}"
                    logger.info(
                        "Drift for %s/%s: %s", cfg.station, cfg.metric, report.drift_reason
                    )

        # ── Train ──────────────────────────────────────────────────────────
        if should_train:
            logger.info("Retraining %s/%s — reason: %s", cfg.station, cfg.metric, train_reason)
            mlflow_run_id = self._train(cfg, run_id=run_id)
            if mlflow_run_id is None:
                return
            # Hot-reload the newly promoted model
            loaded_model = self._model_loader.reload(cfg)
            if loaded_model is None:
                logger.error(
                    "Model reload failed after training for %s/%s", cfg.station, cfg.metric
                )
                return

        # ── Infer ──────────────────────────────────────────────────────────
        if loaded_model is None:
            logger.error("No model available for %s/%s after train attempt.", cfg.station, cfg.metric)
            return

        self._infer(run_id, cfg, loaded_model, data_loader)

    # ──────────────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────────────

    def _train(self, cfg: MetricConfig, run_id: str) -> Optional[str]:
        """
        Full training pipeline for one station+metric.
        Returns the MLflow run_id on success, None on failure.
        """
        data_loader = DataLoader(self._sf, cfg)
        dataset     = data_loader.load_training_window()

        if dataset is None:
            logger.warning(
                "Skipping training for %s/%s — insufficient data.", cfg.station, cfg.metric
            )
            return None

        trainer = MambaTrainer(cfg, artifact_base=self._artifact_base)
        try:
            result: TrainResult = trainer.train(dataset)
        except Exception as exc:
            logger.error("Training failed for %s/%s: %s", cfg.station, cfg.metric, exc)
            self._db_log_error(
                module       = "orchestrator.train",
                message      = str(exc),
                run_id       = run_id,
                station_name = cfg.station,
                metric_name  = cfg.metric,
                tb           = traceback.format_exc(),
            )
            return None

        # Log to MLflow
        try:
            mlflow_run_id = self._mlflow.log_training_run(cfg, result)
            mv            = self._mlflow.register_model(cfg, mlflow_run_id)
            self._mlflow.promote_to_production(cfg, mv.version)
            logger.info(
                "Trained + promoted %s/%s v%s (val_rmse=%.4f)",
                cfg.station, cfg.metric, mv.version, result.val_rmse,
            )
            return mlflow_run_id
        except Exception as exc:
            logger.error(
                "MLflow logging failed for %s/%s: %s", cfg.station, cfg.metric, exc
            )
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────

    def _infer(
        self,
        run_id      : str,
        cfg         : MetricConfig,
        loaded_model: LoadedModel,
        data_loader : DataLoader,
    ) -> None:
        """
        Run inference for all configured analysis_types and persist results.
        """
        feat_names = loaded_model.feature_schema["feature_names"]
        inf_dataset = data_loader.load_inference_window(
            scaler        = loaded_model.scaler,
            feature_names = feat_names,
        )
        if inf_dataset is None:
            logger.warning("No inference data for %s/%s", cfg.station, cfg.metric)
            return

        for analysis_type_str in cfg.analysis_types:
            try:
                result_row = self._run_single_analysis(
                    run_id, cfg, loaded_model, inf_dataset, analysis_type_str
                )
                if result_row:
                    self._persist_result(result_row)
            except Exception as exc:
                logger.error(
                    "Inference failed for %s/%s/%s: %s",
                    cfg.station, cfg.metric, analysis_type_str, exc,
                )
                self._db_log_error(
                    module       = f"orchestrator.infer.{analysis_type_str}",
                    message      = str(exc),
                    run_id       = run_id,
                    station_name = cfg.station,
                    metric_name  = cfg.metric,
                    tb           = traceback.format_exc(),
                )

    def _run_single_analysis(
        self,
        run_id          : str,
        cfg             : MetricConfig,
        loaded_model    : LoadedModel,
        inf_dataset     : InferenceDataset,
        analysis_type   : str,
    ) -> Optional[InferenceResult]:
        """
        Run one analysis type and return an unsaved InferenceResult ORM row.
        """
        if analysis_type == "forecast":
            inferencer = MambaInferencer(
                model                 = loaded_model.model,
                feature_schema        = loaded_model.feature_schema,
                scaler                = loaded_model.scaler,
                config                = cfg,
                model_version         = loaded_model.model_version,
                mlflow_run_id         = loaded_model.mlflow_run_id,
                device                = loaded_model.device,
                training_residual_std = loaded_model.training_residual_std,
            )
            output: ForecastOutput = inferencer.predict(inf_dataset)

            return InferenceResult(
                id               = str(uuid.uuid4()),
                run_id           = run_id,
                station_name     = cfg.station,
                metric_name      = cfg.metric,
                analysis_type    = AnalysisType.FORECAST,
                method_used      = MethodName.NEURAL_NETWORK,
                is_anomaly       = output.is_forecast_anomaly,
                anomaly_score    = abs(output.predicted_vs_observed_delta),
                confidence       = output.confidence,
                data_points_used = output.data_points_used,
                payload          = {
                    "forecast_values"              : output.forecast_values,
                    "forecast_horizon"             : output.forecast_horizon,
                    "lower_bound"                  : output.lower_bound,
                    "upper_bound"                  : output.upper_bound,
                    "latest_observed"              : output.latest_observed,
                    "predicted_vs_observed_delta"  : output.predicted_vs_observed_delta,
                    "model_version"                : output.model_version,
                    "mlflow_run_id"                : output.mlflow_run_id,
                    "features_used"                : output.feature_names,
                    "error"                        : output.error,
                },
            )

        # Anomaly, health_score, rul — these will be implemented in their own
        # inferencer classes (zscore, western_electric, isolation_forest,
        # linear_regression). The orchestrator will dispatch here once those
        # inferencers are built. For now, return None to skip gracefully.
        logger.debug(
            "Analysis type '%s' inferencer not yet wired in orchestrator for %s/%s",
            analysis_type, cfg.station, cfg.metric,
        )
        return None

    # ──────────────────────────────────────────────────────────────────────
    # DB persistence helpers
    # ──────────────────────────────────────────────────────────────────────

    def _persist_result(self, result: InferenceResult) -> None:
        with self._sf() as session:
            session.add(result)
            session.commit()

    def _create_run_row(self, run_id: str, triggered_by: str, now: datetime) -> None:
        with self._sf() as session:
            session.add(InferenceRun(
                id           = run_id,
                triggered_by = triggered_by,
                status       = RunStatus.RUNNING,
                started_at   = now,
            ))
            session.commit()

    def _finalise_run(self, run_id: str, status: RunStatus) -> None:
        with self._sf() as session:
            run = session.get(InferenceRun, run_id)
            if run:
                run.status       = status
                run.completed_at = datetime.now(timezone.utc)
                session.commit()

    def _db_log_error(
        self,
        module       : str,
        message      : str,
        run_id       : Optional[str] = None,
        station_name : Optional[str] = None,
        metric_name  : Optional[str] = None,
        tb           : Optional[str] = None,
    ) -> None:
        try:
            with self._sf() as session:
                session.add(ErrorLog(
                    severity     = Severity.ERROR,
                    module       = module,
                    message      = message,
                    run_id       = run_id,
                    station_name = station_name,
                    metric_name  = metric_name,
                    traceback    = tb,
                ))
                session.commit()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _model_is_stale(loaded_model: LoadedModel, cfg: MetricConfig) -> bool:
        age_days = (
            datetime.now(timezone.utc) - loaded_model.loaded_at
        ).total_seconds() / 86400
        return age_days > cfg.training.max_model_age_days
