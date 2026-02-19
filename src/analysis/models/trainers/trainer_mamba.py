"""
models/trainers/mamba.py
=========================
Training logic for the Mamba_TS model.

Responsibilities:
  - Build the model from config + dataset shape
  - Run training loop with reconstruction + prediction dual loss
  - Early stopping on validation loss
  - Save all artifacts to local filesystem
  - Log everything to MLflow via the tracking client

Classes
-------
MambaTrainer   Full training orchestration for one station+metric.

Dataclasses
-----------
TrainResult    Everything the orchestrator needs after training completes.
ArtifactBundle Paths of all saved artifacts for a training run.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import TensorDataset

from models.architectures.mamba import Mamba_TS
from data.data_loader import TrainingDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Returned by MambaTrainer.train() — everything the orchestrator needs."""
    val_loss       : float
    val_mae        : float
    val_rmse       : float
    train_loss     : float
    best_epoch     : int
    epochs_run     : int
    training_time_s: float
    model          : Mamba_TS
    artifact_bundle: "ArtifactBundle"
    metrics        : dict[str, float]  = field(default_factory=dict)


@dataclass
class ArtifactBundle:
    """Local filesystem paths for all artifacts saved after training."""
    artifact_dir      : str   # root directory for this training run
    weights_path      : str   # model.pt   — state_dict
    architecture_path : str   # arch.json  — model config
    scaler_path       : str   # scaler.pkl
    feature_schema_path: str  # feature_schema.json
    metrics_path      : str   # metrics.json
    config_snapshot_path: str # config_snapshot.json


# ---------------------------------------------------------------------------
# MambaTrainer
# ---------------------------------------------------------------------------

class MambaTrainer:
    """
    Trains a Mamba_TS model for one station+metric configuration.

    Usage
    -----
    trainer = MambaTrainer(
        metric_cfg    = cfg,           # single metric block from analysis_config
        global_cfg    = global_cfg,
        artifact_root = "/models",
        device        = "cuda:0",
        mlflow_client = mlflow_client, # optional, can be None
    )
    result = trainer.train(dataset)    # TrainingDataset from DataLoader
    """

    def __init__(
        self,
        metric_cfg    : dict,
        global_cfg    : dict,
        artifact_root : str,
        device        : str            = "cpu",
        mlflow_client               = None,   # tracking.mlflow_client.MLflowTrackingClient
    ) -> None:
        self._metric_cfg   = metric_cfg
        self._global_cfg   = global_cfg
        self._artifact_root = Path(artifact_root)
        self._device       = torch.device(device)
        self._mlflow       = mlflow_client

        # Derived identifiers
        self._line         = metric_cfg['line']
        self._target       = metric_cfg['target_column']
        self._mamba_cfg    = {**global_cfg.get('mamba', {}), **metric_cfg.get('mamba', {})}

    # ------------------------------------------------------------------
    # Public: train
    # ------------------------------------------------------------------

    def train(self, dataset: TrainingDataset) -> Optional[TrainResult]:
        """
        Full training run.

        1. Builds sequences from the feature matrix.
        2. Instantiates Mamba_TS with correct input_size from dataset.
        3. Runs training loop with dual loss (recon + pred).
        4. Evaluates on val set after each epoch.
        5. Applies early stopping.
        6. Saves artifacts.
        7. Optionally logs to MLflow.

        Returns None if the dataset is too small to train.
        """
        cfg     = self._mamba_cfg
        seq_len = int(cfg.get('seq_len', 12))

        X_seqs_tr, y_seqs_tr = self._build_sequences(dataset.X_train, dataset.y_train, seq_len)
        X_seqs_va, y_seqs_va = self._build_sequences(dataset.X_val,   dataset.y_val,   seq_len)

        if len(X_seqs_tr) < 10:
            logger.warning("MambaTrainer: too few sequences (%d) for %s/%s",
                           len(X_seqs_tr), self._line, self._target)
            return None

        input_size = X_seqs_tr.shape[2]     # n_features

        model = Mamba_TS(
            input_size       = input_size,
            seq_len          = seq_len,
            hidden_dim       = int(cfg.get('hidden_dim',  64)),
            d_state          = int(cfg.get('d_state',     16)),
            d_conv           = int(cfg.get('d_conv',       4)),
            num_layers       = int(cfg.get('num_layers',   2)),
            dropout          = float(cfg.get('dropout',  0.2)),
            forecast_horizon = 1,           # trainer always predicts 1 step ahead
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr           = float(cfg.get('learning_rate', 1e-3)),
            weight_decay = 1e-4,
        )
        criterion        = nn.MSELoss()
        recon_weight     = float(cfg.get('loss_weight_recon', 0.3))
        pred_weight      = 1.0 - recon_weight
        grad_clip        = float(cfg.get('grad_clip', 1.0))
        num_epochs       = int(cfg.get('num_epochs', 200))
        patience         = int(cfg.get('early_stopping_patience', 20))
        batch_size       = int(cfg.get('batch_size', 32))

        # DataLoader for batching
        train_loader = self._make_loader(X_seqs_tr, y_seqs_tr, batch_size, shuffle=True)

        best_val_loss   = float('inf')
        best_state_dict = None
        best_epoch      = 0
        no_improve      = 0
        train_losses    : list[float] = []
        val_losses      : list[float] = []

        t0 = time.time()

        for epoch in range(1, num_epochs + 1):
            # ── training ──────────────────────────────────────────────
            model.train()
            epoch_loss = 0.0
            for X_b, y_b in train_loader:
                X_b = X_b.to(self._device)
                y_b = y_b.to(self._device)
                optimizer.zero_grad()

                recon, pred = model(X_b)

                # Reconstruction loss: model should reproduce the input sequence
                loss_recon = criterion(recon, X_b)
                # Prediction loss: predict the next target value
                loss_pred  = criterion(pred.squeeze(-1), y_b)

                loss = recon_weight * loss_recon + pred_weight * loss_pred
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                epoch_loss += loss.item() * len(X_b)

            epoch_loss /= len(X_seqs_tr)
            train_losses.append(epoch_loss)

            # ── validation ────────────────────────────────────────────
            val_loss, val_mae, val_rmse = self._evaluate(
                model, X_seqs_va, y_seqs_va, criterion, recon_weight, pred_weight
            )
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss   = val_loss
                best_epoch      = epoch
                no_improve      = 0
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info("MambaTrainer: early stopping at epoch %d (best=%d)", epoch, best_epoch)
                break

            if epoch % 50 == 0 or epoch == 1:
                logger.debug("Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  val_mae=%.4f",
                             epoch, num_epochs, epoch_loss, val_loss, val_mae)

        training_time = time.time() - t0

        # Restore best weights
        if best_state_dict:
            model.load_state_dict(best_state_dict)
        model.eval()

        # Final validation metrics
        _, final_mae, final_rmse = self._evaluate(
            model, X_seqs_va, y_seqs_va, criterion, recon_weight, pred_weight
        )

        metrics = {
            'val_loss'        : round(best_val_loss, 6),
            'val_mae'         : round(final_mae, 6),
            'val_rmse'        : round(final_rmse, 6),
            'train_loss_final': round(train_losses[-1], 6),
            'best_epoch'      : best_epoch,
            'epochs_run'      : len(train_losses),
            'training_time_s' : round(training_time, 2),
            'n_train_sequences': len(X_seqs_tr),
            'n_val_sequences'  : len(X_seqs_va),
        }

        # Save artifacts
        artifact_bundle = self._save_artifacts(
            model          = model,
            dataset        = dataset,
            metrics        = metrics,
        )

        # MLflow logging (optional)
        if self._mlflow is not None:
            try:
                self._log_to_mlflow(metrics, artifact_bundle)
            except Exception as exc:
                logger.warning("MambaTrainer: MLflow logging failed (non-fatal): %s", exc)

        return TrainResult(
            val_loss        = best_val_loss,
            val_mae         = final_mae,
            val_rmse        = final_rmse,
            train_loss      = train_losses[-1],
            best_epoch      = best_epoch,
            epochs_run      = len(train_losses),
            training_time_s = training_time,
            model           = model,
            artifact_bundle = artifact_bundle,
            metrics         = metrics,
        )

    # ------------------------------------------------------------------
    # Internal: sequence building
    # ------------------------------------------------------------------

    def _build_sequences(
        self,
        X      : np.ndarray,   # (n_samples, n_features)
        y      : np.ndarray,   # (n_samples,)
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Converts flat (n_samples, n_features) into sliding-window
        (n_sequences, seq_len, n_features) and corresponding targets.
        Target y[i] = value at step i+seq_len (next step after window).
        """
        X_seqs, y_seqs = [], []
        for i in range(len(X) - seq_len):
            X_seqs.append(X[i: i + seq_len])
            y_seqs.append(y[i + seq_len])

        if not X_seqs:
            return torch.empty(0), torch.empty(0)

        return (
            torch.tensor(np.array(X_seqs), dtype=torch.float32),
            torch.tensor(np.array(y_seqs), dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Internal: evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(
        self,
        model        : Mamba_TS,
        X_seqs       : torch.Tensor,
        y_seqs       : torch.Tensor,
        criterion    : nn.Module,
        recon_weight : float,
        pred_weight  : float,
    ) -> tuple[float, float, float]:
        """Returns (combined_loss, mae, rmse) on the given sequences."""
        if len(X_seqs) == 0:
            return 0.0, 0.0, 0.0

        model.eval()
        X_d = X_seqs.to(self._device)
        y_d = y_seqs.to(self._device)

        recon, pred = model(X_d)
        loss = (recon_weight * criterion(recon, X_d) +
                pred_weight  * criterion(pred.squeeze(-1), y_d))

        preds_np  = pred.squeeze(-1).cpu().numpy()
        target_np = y_d.cpu().numpy()

        mae  = float(np.mean(np.abs(preds_np - target_np)))
        rmse = float(np.sqrt(np.mean((preds_np - target_np) ** 2)))

        return float(loss.item()), mae, rmse

    # ------------------------------------------------------------------
    # Internal: torch DataLoader
    # ------------------------------------------------------------------

    @staticmethod
    def _make_loader(
        X_seqs    : torch.Tensor,
        y_seqs    : torch.Tensor,
        batch_size: int,
        shuffle   : bool,
    ) -> TorchDataLoader:
        return TorchDataLoader(
            TensorDataset(X_seqs, y_seqs),
            batch_size = batch_size,
            shuffle    = shuffle,
            drop_last  = False,
        )

    # ------------------------------------------------------------------
    # Internal: artifact saving
    # ------------------------------------------------------------------

    def _save_artifacts(
        self,
        model  : Mamba_TS,
        dataset: TrainingDataset,
        metrics: dict,
    ) -> ArtifactBundle:
        """
        Saves four artifacts to local filesystem:
          model.pt            PyTorch state_dict
          arch.json           Model architecture config
          scaler.pkl          Fitted sklearn scaler
          feature_schema.json Column names + order (must match at inference time)
          metrics.json        Training metrics
          config_snapshot.json Copy of the full metric config used for this run

        Directory: <artifact_root>/<line>/<target_column>/<timestamp>/
        """
        import time as _time
        run_ts  = int(_time.time())
        art_dir = self._artifact_root / self._line / self._target / str(run_ts)
        art_dir.mkdir(parents=True, exist_ok=True)

        # 1. Model weights
        weights_path = str(art_dir / 'model.pt')
        torch.save(model.state_dict(), weights_path)

        # 2. Architecture config
        arch_path = str(art_dir / 'arch.json')
        with open(arch_path, 'w') as f:
            json.dump(model.get_config(), f, indent=2)

        # 3. Scaler
        scaler_path = str(art_dir / 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(dataset.scaler, f)

        # 4. Feature schema — exact ordered list of columns the model expects
        schema_path = str(art_dir / 'feature_schema.json')
        with open(schema_path, 'w') as f:
            json.dump({'feature_names': dataset.feature_names}, f, indent=2)

        # 5. Metrics
        metrics_path = str(art_dir / 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # 6. Config snapshot
        cfg_path = str(art_dir / 'config_snapshot.json')
        with open(cfg_path, 'w') as f:
            json.dump(self._metric_cfg, f, indent=2, default=str)

        logger.info("MambaTrainer: artifacts saved to %s", art_dir)

        return ArtifactBundle(
            artifact_dir       = str(art_dir),
            weights_path       = weights_path,
            architecture_path  = arch_path,
            scaler_path        = scaler_path,
            feature_schema_path= schema_path,
            metrics_path       = metrics_path,
            config_snapshot_path= cfg_path,
        )

    # ------------------------------------------------------------------
    # Internal: MLflow logging
    # ------------------------------------------------------------------

    def _log_to_mlflow(self, metrics: dict, bundle: ArtifactBundle) -> None:
        """Log params, metrics, and artifacts to MLflow for this training run."""
        exp_name  = f"{self._line}/{self._target}"
        exp_id    = self._mlflow.get_or_create_experiment(self._line, self._target)

        run_id = self._mlflow.log_training_run(
            experiment_id = exp_id,
            config        = self._metric_cfg,
            train_result_metrics = metrics,
            artifact_path = bundle.artifact_dir,
        )

        self._mlflow.register_model(
            run_id       = run_id,
            model_name   = exp_name,
            artifact_path= bundle.artifact_dir,
        )

        logger.info("MambaTrainer: MLflow run logged — exp=%s, run_id=%s", exp_name, run_id)
