"""
models/trainers/mamba.py
──────────────────────────
Training loop for the Mamba_TS model.

Responsibilities
----------------
1. Receive a TrainingDataset (already sequenced and scaled by DataLoader).
2. Instantiate Mamba_TS from config hyperparameters.
3. Move model to the correct device (CUDA if available and configured).
4. Run training loop with dual loss (reconstruction + prediction).
5. Validate every epoch; apply early stopping.
6. Save all artifacts (weights, scaler, feature_schema, config snapshot)
   to a local directory that MLflowTrackingClient will then log.
7. Return a TrainResult dataclass with metrics and artifact path.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.config_schema import MetricConfig
from data.data_loader import TrainingDataset
from models.architectures.mamba import Mamba_TS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass returned to the orchestrator / MLflow client
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    val_loss          : float
    val_mae           : float
    val_rmse          : float
    val_r2            : float
    train_loss_history: list[float]
    val_loss_history  : list[float]
    n_epochs_trained  : int
    training_seconds  : float
    artifact_dir      : str                 # local path; MLflow logs from here
    model             : Mamba_TS            # in-memory model (also saved to disk)
    feature_names     : list[str]
    hyperparameters   : dict                # snapshot of what was used
    stopped_early     : bool = False


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MambaTrainer:
    """
    Trains a Mamba_TS model for a single station+metric.

    Parameters
    ----------
    config       : validated MetricConfig
    artifact_base: root directory where per-run artifact folders are created.
                   e.g. /mlflow/artifacts  → per-run folder created inside.
    """

    def __init__(
        self,
        config        : MetricConfig,
        artifact_base : str = "/mlflow/artifacts",
    ) -> None:
        self._config        = config
        self._artifact_base = Path(artifact_base)
        self._device        = self._resolve_device()

    # ──────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────

    def train(self, dataset: TrainingDataset) -> TrainResult:
        """
        Full training run.

        Workflow
        --------
        1. Build model from config hyperparameters.
        2. Run training loop (dual loss: recon + pred).
        3. Validate each epoch; apply early stopping.
        4. Save artifacts to disk.
        5. Return TrainResult.
        """
        cfg         = self._config
        mamba_cfg   = cfg.mamba
        t_start     = time.time()

        # ── 1. Build model ─────────────────────────────────────────────────
        model = Mamba_TS(
            seq_len    = mamba_cfg.seq_len,
            hidden_dim = mamba_cfg.hidden_dim,
            d_state    = mamba_cfg.d_state,
            d_conv     = mamba_cfg.d_conv,
            num_layers = mamba_cfg.num_layers,
            dropout    = mamba_cfg.dropout,
        ).to(self._device)

        logger.info(
            "Training Mamba for %s/%s on %s | params=%d | epochs=%d",
            cfg.station, cfg.metric, self._device,
            model.n_parameters, mamba_cfg.num_epochs,
        )

        # ── 2. DataLoaders ─────────────────────────────────────────────────
        X_tr = torch.FloatTensor(dataset.X_train).to(self._device)
        y_tr = torch.FloatTensor(dataset.y_train).to(self._device)
        X_vl = torch.FloatTensor(dataset.X_val).to(self._device)
        y_vl = torch.FloatTensor(dataset.y_val).to(self._device)

        # Mamba_TS.forward() expects (batch, seq_len) for the target column only.
        # Extract target column index (always index 0 per DataLoader contract).
        target_idx = 0

        train_loader = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size = max(8, min(32, len(X_tr) // 4)),
            shuffle    = False,   # chronological order matters for time series
        )

        # ── 3. Optimiser & loss ────────────────────────────────────────────
        optimizer   = torch.optim.AdamW(
            model.parameters(),
            lr           = mamba_cfg.learning_rate,
            weight_decay = 1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
        )
        criterion = nn.MSELoss()

        w_recon = mamba_cfg.loss_weight_recon
        w_pred  = 1.0 - w_recon

        # ── 4. Training loop ───────────────────────────────────────────────
        best_val_loss      = float("inf")
        best_state_dict    = None
        patience_counter   = 0
        early_stop_patience = max(20, mamba_cfg.num_epochs // 10)

        train_losses: list[float] = []
        val_losses  : list[float] = []

        for epoch in range(mamba_cfg.num_epochs):
            # — Train —
            model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                # X_batch: (B, seq_len, n_features)
                # Mamba_TS forward takes the target channel as (B, seq_len)
                x_target = X_batch[:, :, target_idx]

                optimizer.zero_grad()
                recon, pred = model(x_target)

                loss_recon = criterion(recon, x_target)
                loss_pred  = criterion(pred.squeeze(-1), y_batch)
                loss       = w_recon * loss_recon + w_pred * loss_pred

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train = epoch_loss / len(train_loader)
            train_losses.append(avg_train)

            # — Validate —
            model.eval()
            with torch.no_grad():
                x_val_target = X_vl[:, :, target_idx]
                recon_v, pred_v = model(x_val_target)
                val_loss = (
                    w_recon * criterion(recon_v, x_val_target)
                    + w_pred * criterion(pred_v.squeeze(-1), y_vl)
                ).item()
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            # — Early stopping —
            if val_loss < best_val_loss - 1e-6:
                best_val_loss    = val_loss
                best_state_dict  = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stop_patience:
                logger.info(
                    "Early stop at epoch %d/%d for %s/%s (best_val=%.6f)",
                    epoch + 1, mamba_cfg.num_epochs, cfg.station, cfg.metric, best_val_loss,
                )
                n_epochs = epoch + 1
                stopped_early = True
                break
        else:
            n_epochs      = mamba_cfg.num_epochs
            stopped_early = False

        # Restore best weights
        if best_state_dict:
            model.load_state_dict(best_state_dict)
        model.eval()

        # ── 5. Compute validation metrics ─────────────────────────────────
        val_metrics = self._compute_val_metrics(model, X_vl, y_vl, target_idx)

        # ── 6. Save artifacts ──────────────────────────────────────────────
        artifact_dir = self._save_artifacts(
            model         = model,
            dataset       = dataset,
            hyperparameters = self._hyperparameter_snapshot(),
        )

        training_secs = time.time() - t_start
        logger.info(
            "Training complete for %s/%s | val_rmse=%.4f | val_r2=%.4f | %.1fs",
            cfg.station, cfg.metric,
            val_metrics["rmse"], val_metrics["r2"], training_secs,
        )

        return TrainResult(
            val_loss           = best_val_loss,
            val_mae            = val_metrics["mae"],
            val_rmse           = val_metrics["rmse"],
            val_r2             = val_metrics["r2"],
            train_loss_history = train_losses,
            val_loss_history   = val_losses,
            n_epochs_trained   = n_epochs,
            training_seconds   = training_secs,
            artifact_dir       = str(artifact_dir),
            model              = model,
            feature_names      = dataset.feature_names,
            hyperparameters    = self._hyperparameter_snapshot(),
            stopped_early      = stopped_early,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Artifact persistence
    # ──────────────────────────────────────────────────────────────────────

    def _save_artifacts(
        self,
        model           : Mamba_TS,
        dataset         : TrainingDataset,
        hyperparameters : dict,
    ) -> Path:
        """
        Save all four artifact files to a run-specific directory.

        Artifact layout
        ---------------
        <artifact_base>/<station>/<metric>/
            model.pt              — PyTorch state dict
            scaler.pkl            — fitted sklearn scaler
            feature_schema.json   — {feature_names, target_col, dtypes}
            model_config.json     — hyperparameter snapshot + model metadata
        """
        cfg      = self._config
        art_dir  = self._artifact_base / cfg.station / cfg.metric
        art_dir.mkdir(parents=True, exist_ok=True)

        # model weights
        torch.save(model.state_dict(), art_dir / "model.pt")

        # scaler
        with open(art_dir / "scaler.pkl", "wb") as f:
            pickle.dump(dataset.scaler, f)

        # feature schema
        schema = {
            "feature_names" : dataset.feature_names,
            "target_col"    : dataset.target_col,
            "seq_len"       : cfg.mamba.seq_len,
            "n_features"    : len(dataset.feature_names),
        }
        with open(art_dir / "feature_schema.json", "w") as f:
            json.dump(schema, f, indent=2)

        # model config snapshot
        with open(art_dir / "model_config.json", "w") as f:
            json.dump(hyperparameters, f, indent=2)

        logger.debug("Artifacts saved to %s", art_dir)
        return art_dir

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_val_metrics(
        self,
        model      : Mamba_TS,
        X_val      : torch.Tensor,
        y_val      : torch.Tensor,
        target_idx : int,
    ) -> dict:
        model.eval()
        with torch.no_grad():
            _, pred = model(X_val[:, :, target_idx])
            preds = pred.squeeze(-1).cpu().numpy()
            actuals = y_val.cpu().numpy()

        mae  = float(np.mean(np.abs(preds - actuals)))
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2   = float(1 - ss_res / (ss_tot + 1e-8))
        return {"mae": mae, "rmse": rmse, "r2": r2}

    def _resolve_device(self) -> str:
        device_cfg = self._config.mamba.device
        if device_cfg == "cpu":
            return "cpu"
        if device_cfg == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available; falling back to CPU.")
                return "cpu"
            return "cuda"
        # auto
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _hyperparameter_snapshot(self) -> dict:
        cfg = self._config
        return {
            "station"         : cfg.station,
            "metric"          : cfg.metric,
            "target_column"   : cfg.target_column,
            "architecture"    : "Mamba_TS",
            **cfg.mamba.model_dump(),
        }
