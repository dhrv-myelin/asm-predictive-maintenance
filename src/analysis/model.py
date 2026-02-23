import numpy as np
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import yaml


from models.mamba import Mamba_TS

# ============================================================
# Main Wrapper
# ============================================================

models = {
    "mamba": Mamba_TS,
}


class Model:
    """
    Simple universal wrapper for:
        - PyTorch models
        - Sklearn models

    Expected by orchestrator:
        train(X, y)
        real_time_inference(window_df)
    """

    def __init__(self, data_handler, model, config, target_name):

        # basic inputs

        # datahandler class
        self.data_handler = data_handler
        # HACK: method str from config
        self.model_name = model
        # config of just that method
        self.config = config
        self.target_name = target_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_type = self.config["model_type"]

        # used to init either sklearn or torch
        self.backend = self._init_backend()

    # --------------------------------------------------------
    # Backend Selection
    # --------------------------------------------------------

    def _init_backend(self):

        if self.model_type == "sequence":
            return TorchBackend(self.config, self.device)

        elif self.model_type == "tabular":
            return SklearnBackend(self.config)

        else:
            raise ValueError(f"Unsupported method: {self.model_name}")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    def train(self, X, y):
        """
        X: (N, seq_len, num_features)
        y: (N, num_targets)
        """

        if self.model_type == "tabular":
            # Flatten sequence dimension
            N, seq_len, num_features = X.shape
            X = X.reshape(N, seq_len * num_features)

        # sequence models keep shape (N, seq_len, num_features)

        self.backend.train(X, y)

    # --------------------------------------------------------
    # Real-time inference
    # --------------------------------------------------------

    def real_time_inference(self, window_df):
        """
        window_df: DataFrame (seq_len, num_features + timestamp)
        """

        # Drop timestamp + target if present
        X = window_df.drop(
            columns=["timestamp", self.target_name], errors="ignore"
        ).to_numpy()

        if self.model_type == "tabular":
            # Flatten entire window into one row
            X = X.reshape(1, -1)

        elif self.model_type == "sequence":
            # Add batch dimension
            X = X[np.newaxis, ...]

        preds = self.backend.predict(X)

        return np.array(preds).flatten().tolist()


# ============================================================
# Torch Backend
# ============================================================


class TorchBackend:

    def __init__(self, config, device):

        # cuda device if possible
        self.device = device

        # config only contains required params
        self.config = config

        # 👇 YOU plug in your torch model class here
        # Replace YourTorchModel with your actual model
        self.model = models[config["method"]](**config.get("arch", {}))

        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.get("lr", 1e-3)
        )

        mlflow.pytorch.autolog()

    def train(self, X, y):

        print("Starting training...")

        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        N = X.shape[0]

        # ---------------------------------------------------------
        # SPLIT (leave your config placeholders)
        # ---------------------------------------------------------

        # train_ratio = self.config["train_ratio"]
        # val_ratio = self.config["val_ratio"]
        # test_ratio = self.config["test_ratio"]

        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15

        train_end = int(N * train_ratio)
        val_end = train_end + int(N * val_ratio)

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        epochs = self.config.get("epochs", 10)
        lr = self.config.get("lr", 1e-3)

        # ---------------------------------------------------------
        # MLflow Run
        # ---------------------------------------------------------

        with mlflow.start_run():

            # Log hyperparameters
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("model_name", self.config["method"])

            for epoch in range(epochs):

                self.model.train()
                self.optimizer.zero_grad()

                recon, pred = self.model(X_train)
                train_loss = self.criterion(pred, y_train)

                train_loss.backward()
                self.optimizer.step()

                # Validation
                self.model.eval()
                with torch.no_grad():
                    _, pred_val = self.model(X_val)
                    val_loss = self.criterion(pred_val, y_val)

                # Log metrics per epoch
                mlflow.log_metric("train_loss", train_loss.item(), step=epoch)
                mlflow.log_metric("val_loss", val_loss.item(), step=epoch)

                print(
                    f"Epoch {epoch+1}/{epochs} "
                    f"| Train Loss: {train_loss.item():.6f} "
                    f"| Val Loss: {val_loss.item():.6f}"
                )

        # -----------------------------------------------------
        # Test Evaluation
        # -----------------------------------------------------

        self.model.eval()
        with torch.no_grad():
            _, pred_test = self.model(X_test)
            test_loss = self.criterion(pred_test, y_test)

        mlflow.log_metric("test_loss", test_loss.item())

        print(f"\nFinal Test Loss: {test_loss.item():.6f}")

        # -----------------------------------------------------
        # Log Model Artifact
        # -----------------------------------------------------

        mlflow.pytorch.log_model(self.model, "model")

    def predict(self, X):

        self.model.eval()

        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            recon, pred = self.model(X)

        return pred.cpu().numpy()


# ============================================================
# Sklearn Backend
# ============================================================


# TODO:
class SklearnBackend:

    def __init__(self, config) -> None:
        pass

    def train(self):
        pass

    def predict(self):
        pass

    pass


if __name__ == "__main__":

    pass

    # from data_handler import DataHandler

    # import os

    # config path
    # CONFIG_PATH = os.getcwd() + "/config/analysis_config.yaml"

    # def load_config(path):

    # with open(path, "r") as f:
    # data = yaml.safe_load(f)
    # return data

    # data_handler = DataHandler

    # config = load_config(CONFIG_PATH)

    # model = Model(
    #     data_handler=data_handler,
    #     model="mamba",
    #     # model_name="xgboost",
    #     config=config,
    #     target_name="system__cycle_time",
    # )
