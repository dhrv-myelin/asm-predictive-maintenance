import os

import torch
import numpy
import yaml

# config path
CONFIG_PATH = os.getcwd() + "/config/analysis_config.yaml"


def load_config(path):

    with open(path, "r") as f:
        data = yaml.safe_load(f)

        # test_config = data["target"]["system__cycle_time"][0]
        # print(test_config)

    return data


# model class
class Model:

    # takes in the config, the target and the model to use

    def __init__(self, config, model_name, target_name):

        self.model_name = model_name
        self.target_name = target_name

        all_configs = config["target"][target_name]

        # find config matching model_name
        self.config = next(c for c in all_configs if c["method"] == model_name)

        # Extract config fields
        self.method = self.config["method"]
        # required features from wide table
        self.required_features = self.config["required_features"]
        self.history_window = self.config["history_window"]
        self.prediction_window = self.config["prediction_window"]
        self.mode = self.config.get("mode", "offline")

    def train_model(self):
        pass

    def real_time_inference(self):
        pass


class Model:
    """
    Generic ML model wrapper.

    Supports:
        - sequence models (PyTorch: Mamba, LSTM, Transformer)
        - tabular models (sklearn: XGBoost, RF, Linear, etc)

    Public API:
        train_model()
        real_time_inference()
    """

    # ─────────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────────
    def __init__(self, data_handler, model, config, target_name):

        self.data_handler = data_handler
        self.model = model
        self.config = config
        self.target_name = target_name

        self.method = config["method"]
        self.model_type = config["model_type"]

        self.history_window = config.get("history_window", None)
        self.prediction_window = config.get("prediction_window", 1)

        print(config)

    # ─────────────────────────────────────────────
    # PUBLIC TRAIN ENTRY
    # ─────────────────────────────────────────────
    def train_model(self):

        if self.model_type == "tabular":
            self._train_tabular()

        elif self.model_type == "sequence":
            self._train_sequence()

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    # ─────────────────────────────────────────────
    # TABULAR TRAINING (sklearn style)
    # ─────────────────────────────────────────────
    def _train_tabular(self):

        df = self.data_handler.df.copy()

        if df.empty:
            raise RuntimeError("No training data available")

        X = df.drop(columns=[self.target_name, "timestamp"], errors="ignore")
        y = df[self.target_name]

        print(f"[TABULAR TRAIN] Rows: {len(df)}, Features: {X.shape[1]}")

        self.model.fit(X.values, y.values)

    # ─────────────────────────────────────────────
    # SEQUENCE TRAINING (PyTorch style)
    # ─────────────────────────────────────────────
    def _train_sequence(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.train()

        train_params = self.config.get("train_params", {})
        lr = train_params.get("learning_rate", 1e-3)
        epochs = train_params.get("num_epochs", 10)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        print(f"[SEQUENCE TRAIN] Epochs={epochs}, Device={device}")

        for epoch in range(epochs):

            ts = None
            total_loss = 0
            steps = 0

            while True:

                out = self.data_handler.fetch_next_window(ts, for_training=True)
                if out is None:
                    break

                X_df, y_df = out

                # remove timestamp
                X_df = X_df.drop(columns=["timestamp"], errors="ignore")

                X = torch.tensor(
                    X_df.values, dtype=torch.float32, device=device
                ).unsqueeze(0)
                y = torch.tensor(
                    y_df.values, dtype=torch.float32, device=device
                ).unsqueeze(0)

                optimizer.zero_grad()

                pred = self.model(X)

                loss = loss_fn(pred.squeeze(), y.squeeze())

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1

                ts = (
                    X_df.index[0]
                    if "timestamp" not in X_df
                    else X_df.iloc[0]["timestamp"]
                )

            avg_loss = total_loss / max(steps, 1)
            print(f"Epoch {epoch+1}/{epochs}  Loss={avg_loss:.5f}")

    # ─────────────────────────────────────────────
    # PUBLIC INFERENCE ENTRY
    # ─────────────────────────────────────────────
    def real_time_inference(self):

        if self.model_type == "tabular":
            return self._inference_tabular()

        elif self.model_type == "sequence":
            return self._inference_sequence()

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    # ─────────────────────────────────────────────
    # TABULAR INFERENCE
    # ─────────────────────────────────────────────
    def _inference_tabular(self):

        df = self.data_handler.df

        if df.empty:
            return None

        X = df.tail(1).drop(columns=[self.target_name, "timestamp"], errors="ignore")

        pred = self.model.predict(X.values)

        return np.asarray(pred)

    # ─────────────────────────────────────────────
    # SEQUENCE INFERENCE
    # ─────────────────────────────────────────────
    def _inference_sequence(self):

        df = self.data_handler.df

        if len(df) < self.history_window:
            return None

        X_df = df.tail(self.history_window).drop(columns=["timestamp"], errors="ignore")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X = torch.tensor(X_df.values, dtype=torch.float32, device=device).unsqueeze(0)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)

        return pred.squeeze().cpu().numpy()


if __name__ == "__main__":

    config = load_config(CONFIG_PATH)

    model = Model(config, "mamba", "system__cycle_time")
