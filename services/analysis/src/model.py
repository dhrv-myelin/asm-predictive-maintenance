import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

matplotlib.use("Agg")

from models.isolation_health import IsolationForestHealth
from models.mamba import Mamba_TS
from models.xgboost_window_forecaster import XGBWindowForecaster

# ============================================================
# Registry
# ============================================================

models = {
    "mamba": Mamba_TS,
    "health_score": IsolationForestHealth,
    "xgboost_forecast": XGBWindowForecaster,
}

# Unsupervised: y is ignored entirely for fit and eval
UNSUPERVISED_MODELS = {"health_score"}

# Row-wise: DO NOT flatten seq_len into columns.
# Instead unroll (N, seq_len, F) -> (N*seq_len, F) so the model
# scores each timestep as an independent sample.
ROWWISE_MODELS = {"health_score"}


# ============================================================
# Main Wrapper
# ============================================================


class Model:
    """
    Universal wrapper for PyTorch and sklearn models.

    Orchestrator interface:
        train(X, y)
        real_time_inference(window_df)
    """

    def __init__(self, data_handler, model, config, target_name):
        self.data_handler = data_handler
        self.model_name = model
        self.config = config
        self.target_name = target_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = self.config["model_type"]
        self.backend = self._init_backend()

    def _init_backend(self):
        if self.model_type == "torch":
            return TorchBackend(self.config, self.device)
        elif self.model_type == "sklearn":
            return SklearnBackend(self.config)
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def train(self, X, y):
        """
        X: (N, seq_len, num_features)
        y: (N, num_targets)
        """
        if self.model_type == "tabular":
            N, seq_len, num_features = X.shape
            method = self.config["method"]

            if method in ROWWISE_MODELS:
                # Each timestep is its own sample — unroll the seq_len dimension
                # (N, seq_len, F) -> (N*seq_len, F)
                X = X.reshape(N * seq_len, num_features)
                y = np.repeat(
                    y, seq_len, axis=0
                )  # keep shape consistent (ignored anyway)
            else:
                # Flatten each window into one long row
                # (N, seq_len, F) -> (N, seq_len*F)
                X = X.reshape(N, seq_len * num_features)

        self.backend.train(X, y)

    # --------------------------------------------------------
    # Real-time inference
    # --------------------------------------------------------

    def real_time_inference(self, window_df):
        """
        window_df: DataFrame of shape (seq_len, num_features + timestamp)
        """
        X = window_df.drop(
            columns=["timestamp", self.target_name], errors="ignore"
        ).to_numpy()

        method = self.config.get("method", "")

        if self.model_type == "tabular":
            if method in ROWWISE_MODELS:
                pass  # pass (seq_len, F) as-is; model scores row by row
            else:
                X = X.reshape(1, -1)  # flatten to (1, seq_len*F)
        elif self.model_type == "sequence":
            X = X[np.newaxis, ...]  # add batch dim -> (1, seq_len, F)

        preds = self.backend.predict(X)
        return np.array(preds).flatten().tolist()


# ============================================================
# Torch Backend
# ============================================================


class TorchBackend:

    def __init__(self, config, device):
        self.device = device
        self.config = config
        self.model = models[config["method"]](**config.get("arch", {}))
        self.model.to(self.device)

        # INFO: training params from config are added in here
        self.train_cfg = self.config.get("train_params", {})

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_cfg.get("learning_rate", 1e-3)
        )
        mlflow.pytorch.autolog()

    def train(self, X, y):
        print("Starting training...")

        train_split = self.train_cfg.get("train_split", 0.7)
        val_split = self.train_cfg.get("train_split", 0.2)
        test_split = self.train_cfg.get("train_split", 0.1)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        N = X.shape[0]
        train_end = int(N * 0.7)
        val_end = train_end + int(N * 0.15)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        epochs = self.config.get("num_epochs", 200)
        lr = self.config.get("learning_rate", 1e-3)

        with mlflow.start_run():
            mlflow.log_param("num_epochs", epochs)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("model_name", self.config["method"])

            for epoch in range(epochs):
                self.model.train()
                self.optimizer.zero_grad()
                recon, pred = self.model(X_train)
                train_loss = self.criterion(pred, y_train)
                train_loss.backward()
                self.optimizer.step()

                self.model.eval()
                with torch.no_grad():
                    _, pred_val = self.model(X_val)
                    val_loss = self.criterion(pred_val, y_val)

                mlflow.log_metric("train_loss", train_loss.item(), step=epoch)
                mlflow.log_metric("val_loss", val_loss.item(), step=epoch)
                print(
                    f"Epoch {epoch+1}/{epochs} "
                    f"| Train Loss: {train_loss.item():.6f} "
                    f"| Val Loss: {val_loss.item():.6f}"
                )

            self.model.eval()
            with torch.no_grad():
                _, pred_test = self.model(X_test)
                test_loss = self.criterion(pred_test, y_test)

            mlflow.log_metric("test_loss", test_loss.item())
            print(f"\nFinal Test Loss: {test_loss.item():.6f}")

            _log_pred_vs_real_plot(
                y_true=y_test.cpu().numpy().flatten(),
                y_pred=pred_test.cpu().numpy().flatten(),
                title=f"{self.config['method']} — Test: Predicted vs Real",
            )

            mlflow.pytorch.log_model(self.model, "model")

    def predict(self, X):
        if self.config.get("load_path"):
            print(f"Loading model from {self.config['load_path']}")
            self.model = torch.load(
                self.config["load_path"], map_location=self.device, weights_only=False
            )
            self.model.to(self.device)
            self.model.eval()

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, pred = self.model(X)
        return pred.cpu().numpy()


# ============================================================
# Sklearn Backend
# ============================================================


class SklearnBackend:

    def __init__(self, config):
        self.config = config
        self.is_unsupervised = config["method"] in UNSUPERVISED_MODELS

        if config["method"] not in models:
            raise ValueError(f"Unknown sklearn model: {config['method']}")

        self.model = models[config["method"]](**config.get("arch", {}))

        # IMPORTANT: do NOT enable autolog for unsupervised models.
        # autolog tries to log training labels that don't exist, produces NaN
        # metrics, and then mlflow.sklearn.log_model tries to log those same
        # NaN metrics again -> SQLite UNIQUE constraint crash.
        if not self.is_unsupervised:
            mlflow.sklearn.autolog()

    def train(self, X, y):
        print("Starting sklearn training...")

        y = np.array(y).ravel()  # (N,1) -> (N,)

        N = X.shape[0]
        train_end = int(N * 0.7)
        val_end = train_end + int(N * 0.15)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        print(
            f"  Split -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
        )

        with mlflow.start_run():
            mlflow.log_param("model_name", self.config["method"])
            mlflow.log_params(self.config.get("arch", {}))

            # Fit
            if self.is_unsupervised:
                self.model.fit(X_train)
            else:
                self.model.fit(X_train, y_train)

            # Validation
            y_val_pred = self.model.predict(X_val)
            print(
                f"  val_pred shape: {np.array(y_val_pred).shape}, any NaN: {np.any(np.isnan(y_val_pred))}"
            )

            if self.is_unsupervised:
                val_mean = float(np.nanmean(y_val_pred)) if len(y_val_pred) > 0 else 0.0
                mlflow.log_metric("val_mean_health_score", val_mean)
                print(f"Validation mean health score: {val_mean:.2f}")
            else:
                val_loss = mean_squared_error(y_val, y_val_pred)
                mlflow.log_metric("val_loss", val_loss)
                print(f"Validation Loss: {val_loss:.6f}")

            # Test
            y_test_pred = self.model.predict(X_test)

            if self.is_unsupervised:
                test_mean = (
                    float(np.nanmean(y_test_pred)) if len(y_test_pred) > 0 else 0.0
                )
                mlflow.log_metric("test_mean_health_score", test_mean)
                print(f"Test mean health score: {test_mean:.2f}")

                _log_health_score_plot(
                    scores=y_test_pred,
                    title=f"{self.config['method']} — Test Health Score",
                )
            else:
                test_loss = mean_squared_error(y_test, y_test_pred)
                mlflow.log_metric("test_loss", test_loss)
                print(f"Final Test Loss: {test_loss:.6f}")

                _log_pred_vs_real_plot(
                    y_true=y_test,
                    y_pred=y_test_pred,
                    title=f"{self.config['method']} — Test: Predicted vs Real",
                )

            mlflow.sklearn.log_model(self.model, "model")

    def predict(self, X):
        if self.config.get("load_path"):
            print(f"Loading sklearn model from {self.config['load_path']}")
            self.model = mlflow.sklearn.load_model(self.config["load_path"])
        return self.model.predict(X)


# ============================================================
# Plot helpers — saved as MLflow artifacts under plots/
# ============================================================


def _log_pred_vs_real_plot(y_true, y_pred, title="Predicted vs Real"):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    ax1.plot(y_true, label="Real", color="#2196F3", linewidth=1.5)
    ax1.plot(y_pred, label="Predicted", color="#FF5722", linewidth=1.5, linestyle="--")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    residuals = y_pred - y_true
    ax2.bar(range(len(residuals)), residuals, color="#9C27B0", alpha=0.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Residuals (Predicted − Real)", fontsize=11)
    ax2.set_xlabel("Sample index")
    ax2.set_ylabel("Error")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_log(fig, "pred_vs_real.png")


def _log_health_score_plot(scores, title="Health Score"):
    scores = np.array(scores).flatten()
    x = np.arange(len(scores))

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.axhspan(75, 100, alpha=0.08, color="green", label="Healthy (75–100)")
    ax.axhspan(40, 75, alpha=0.08, color="orange", label="Degraded (40–75)")
    ax.axhspan(0, 40, alpha=0.08, color="red", label="Critical (0–40)")

    ax.plot(x, scores, color="#1565C0", linewidth=1.5, zorder=3)
    ax.fill_between(x, scores, alpha=0.15, color="#1565C0")

    ax.axhline(75, color="green", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(40, color="red", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Health Score (0–100)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_and_log(fig, "health_score.png")


def _save_and_log(fig, filename):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    fig.savefig(tmp_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(tmp_path, artifact_path="plots")
    os.unlink(tmp_path)
