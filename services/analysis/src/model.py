import os
import joblib
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


# ============================================================
# Constants
# ============================================================

SPLIT = {"train": 0.70, "val": 0.15}  # test gets the remainder


# ============================================================
# Helpers
# ============================================================


def _split(X, y=None):
    """Split arrays into train / val / test."""
    N = len(X)
    train_end = int(N * SPLIT["train"])
    val_end = train_end + int(N * SPLIT["val"])

    if y is None:
        return X[:train_end], X[train_end:val_end], X[val_end:]

    return (
        X[:train_end],
        y[:train_end],
        X[train_end:val_end],
        y[train_end:val_end],
        X[val_end:],
        y[val_end:],
    )


# ============================================================
# Model (orchestrator)
# ============================================================


class Model:
    """
    Universal wrapper for PyTorch and sklearn models.

    Public interface:
        train(X, y)                     X: (N, seq_len, F), y: (N, num_targets)
        real_time_inference(window_df)  -> list[float]
    """

    def __init__(self, data_handler, model, config, target_name):
        self.data_handler = data_handler
        self.model_name = model
        self.config = config
        self.target_name = target_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = config["model_type"]
        self.backend = self._init_backend()

    def _init_backend(self):
        if self.model_type == "torch":
            return TorchBackend(self.config, self.device)
        if self.model_type == "sklearn":
            return SklearnBackend(self.config)
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def train(self, X, y, timestamps=None):
        if self.model_type == "sklearn":
            N, seq_len, F = X.shape
            if self.config["method"] in UNSUPERVISED_MODELS:
                # Unsupervised (health_score): windows are seq_len=1 by config.
                # Squeeze to (N, F) so IsolationForest sees flat feature rows.
                # y is meaningless — SklearnBackend.train() will ignore it.
                X = X.reshape(N * seq_len, F)
                self.train_timestamps = timestamps  # ← store for later use
            elif self.config["method"] in ROWWISE_MODELS:
                X = X.reshape(N * seq_len, F)
                y = np.repeat(y, seq_len, axis=0)
            else:
                X = X.reshape(N, seq_len * F)

        self.backend.train(X, y)

    def real_time_inference(self, window_df):
        X = window_df.drop(columns=["timestamp"], errors="ignore").to_numpy()
        print(
            f"[DEBUG] Columns used for inference ({X.shape[1]}): {list(window_df.drop(columns=['timestamp'], errors='ignore').columns)}"
        )
        print(f"[DEBUG] X shape after drop: {X.shape}")
        if self.model_type == "sklearn" and self.config["method"] not in ROWWISE_MODELS:
            X = X.reshape(1, -1)  # flatten to (1, seq_len*F)
        elif self.model_type == "torch":
            X = X[np.newaxis, ...]  # add batch dim -> (1, seq_len, F)
        return np.array(self.backend.predict(X)).flatten().tolist()


# ============================================================
# Torch Backend
# ============================================================


class TorchBackend:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        train_params = config.get("train_params", {})
        self.epochs = train_params.get("num_epochs", 200)
        self.lr = train_params.get("learning_rate", 1e-3)

        self.model = models[config["method"]](**config.get("arch", {})).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _eval_loss(self, X, y):
        """Eval-mode forward pass, returns scalar loss. Used 3x in train."""
        self.model.eval()
        with torch.no_grad():
            _, pred = self.model(X)
        return self.criterion(pred, y).item()

    def train(self, X, y):
        # 1. Split
        X_train, y_train, X_val, y_val, X_test, y_test = _split(
            torch.tensor(X, dtype=torch.float32).to(self.device),
            torch.tensor(y, dtype=torch.float32).to(self.device),
        )

        with mlflow.start_run():
            # 2. Log hyperparams
            mlflow.log_params(
                {
                    "num_epochs": self.epochs,
                    "learning_rate": self.lr,
                    "model": self.config["method"],
                }
            )

            # 3. Train loop
            for epoch in range(self.epochs):
                self.model.train()
                self.optimizer.zero_grad()
                _, pred = self.model(X_train)
                loss = self.criterion(pred, y_train)
                loss.backward()
                self.optimizer.step()

                # 4. Validate each epoch
                val_loss = self._eval_loss(X_val, y_val)
                mlflow.log_metrics(
                    {"train_loss": loss.item(), "val_loss": val_loss}, step=epoch
                )
                print(
                    f"Epoch {epoch+1}/{self.epochs} | train: {loss.item():.6f} | val: {val_loss:.6f}"
                )

            # 5. Final test evaluation
            test_loss = self._eval_loss(X_test, y_test)
            mlflow.log_metric("test_loss", test_loss)
            print(f"Test loss: {test_loss:.6f}")

            # 6. Log model
            # mlflow.pytorch.log_model(self.model, "model")
            save_dir = f"saved_models/{self.config['save_name']}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.model, os.path.join(save_dir, "model.pt"))
            mlflow.log_param("local_model_path", save_dir)

    def predict(self, X):
        # Optionally load a saved checkpoint
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

    def train(self, X, y):
        y = np.array(y).ravel()

        # 1. Split
        if self.is_unsupervised:
            X_train, X_val, X_test = _split(X)
            y_train, y_val, y_test = None, None, None
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = _split(X, y)
        print(f"Split -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

        with mlflow.start_run():
            # 2. Log params
            mlflow.log_params(
                {"model": self.config["method"], **self.config.get("arch", {})}
            )

            # 3. Fit
            (
                self.model.fit(X_train)
                if self.is_unsupervised
                else self.model.fit(X_train, y_train)
            )

            # 4. Evaluate
            for name, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
                pred = self.model.predict(X_s)
                if self.is_unsupervised:
                    metric = float(np.nanmean(pred))
                    mlflow.log_metric(f"{name}_mean_score", metric)
                    print(f"{name} mean score: {metric:.4f}")
                else:
                    loss = mean_squared_error(y_s, pred)
                    mlflow.log_metric(f"{name}_loss", loss)
                    print(f"{name} loss: {loss:.6f}")

            # 5. Log model
            # mlflow.sklearn.log_model(self.model, "model")

            save_dir = f"saved_models/{self.config['save_name']}"
            os.makedirs(save_dir, exist_ok=True)
            joblib.dump(self.model, os.path.join(save_dir, "model.joblib"))
            mlflow.log_param("local_model_path", save_dir)

    def predict(self, X):
        if self.config.get("load_path"):
            print(f"Loading sklearn model from {self.config['load_path']}")
            # self.model = mlflow.sklearn.load_model(self.config["load_path"])
            self.model = joblib.load(self.config["load_path"])
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
