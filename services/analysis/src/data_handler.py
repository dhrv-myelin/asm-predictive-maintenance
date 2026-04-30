from __future__ import annotations

import logging
from datetime import date

import numpy as np

# import pandas as pd

import polars as pl

logger = logging.getLogger(__name__)


class NewDataHandler:

    def __init__(self, config: dict, target_name: str):

        self.config = config
        self.target_name = target_name

        # way to hold a cycle
        self._cycle_buffer: dict[tuple[date, int], dict] = {}

        # decide new configs

        # this is to decide what data frame the model needs.
        # some models need wide, some need time stamp some dont blah blah
        self.format = config.get("format", None)

    def ingest_process_metric_rows(self, rows):

        if not rows:
            return None

        fo


class DataHandler:
    def __init__(self, config: dict, target_name: str):
        self.config = config
        self.target_name = target_name

        self.required_features: list[str] = config["required_features"]
        print("Required features : ", self.required_features)
        self.format = config.get("format", None)
        self.history_window: int = config["history_window"]
        self.stride: int = config.get("stride", 0) or 0
        self.prediction_window: int = config["prediction_window"]

        self._feature_keys = [k for k in self.required_features if k != "timestamp"]
        self._feature_set = set(self._feature_keys)

        self._cycle_buffer: dict[tuple[date, int], dict] = {}
        self._max_cycle_per_day: dict[date, int] = {}

        self.df: pd.DataFrame = pd.DataFrame(columns=self.required_features)
        self.last_timestamp = None

    def ingest(self, rows) -> pd.DataFrame | None:
        print(f"[DEBUG] Ingesting rows: {len(rows)} rows")
        if not rows:
            return None

        for r in rows:
            ts = pd.to_datetime(r.timestamp)
            nmn = f"{r.station_name}__{r.metric_name}"
            val = r.value
            cycle_count = int(r.cycle_count) if r.cycle_count is not None else None
            if cycle_count is None:
                continue
            day = ts.date()
            key = (day, cycle_count)
            if key not in self._cycle_buffer:
                self._cycle_buffer[key] = {"timestamp": ts}
            self._cycle_buffer[key][nmn] = val
            if ts > self._cycle_buffer[key]["timestamp"]:
                self._cycle_buffer[key]["timestamp"] = ts

            if (
                day not in self._max_cycle_per_day
                or cycle_count > self._max_cycle_per_day[day]
            ):
                self._max_cycle_per_day[day] = cycle_count

        self._flush_completed_cycles()
        return self.df if not self.df.empty else None

    def _build_row(self, row_dict: dict) -> dict:
        merged = {}
        for col in self.required_features:
            v = row_dict.get(col, np.nan)
            if v is pd.NA:
                v = np.nan
            merged[col] = v

        if "timestamp" in row_dict and row_dict["timestamp"] is not None:
            self.last_timestamp = row_dict["timestamp"]

        return merged

    def _append_rows(self, rows_to_append: list[dict]) -> None:
        if not rows_to_append:
            return

        new_df = pd.DataFrame(rows_to_append, columns=self.required_features)

        if not self.df.empty:
            self.df = pd.concat([self.df, new_df], ignore_index=True)
        else:
            self.df = new_df

        self.df.drop_duplicates(inplace=True)

    def _flush_completed_cycles(self) -> None:
        keys_to_flush = []
        for key in self._cycle_buffer:
            day, cycle = key
            max_for_day = self._max_cycle_per_day.get(day, 0)
            if cycle < max_for_day:
                keys_to_flush.append(key)

        keys_to_flush.sort()

        rows_to_append = []
        for key in keys_to_flush:
            row_dict = self._cycle_buffer.pop(key)
            rows_to_append.append(self._build_row(row_dict))

        self._append_rows(rows_to_append)

        if rows_to_append:
            print(
                f"[DEBUG] Flushed {len(rows_to_append)} cycles, total df rows: {len(self.df)}"
            )

    def flush_remaining(self) -> None:
        if not self._cycle_buffer:
            return

        keys_sorted = sorted(self._cycle_buffer.keys())
        rows_to_append = []
        for key in keys_sorted:
            row_dict = self._cycle_buffer.pop(key)
            rows_to_append.append(self._build_row(row_dict))

        self._append_rows(rows_to_append)

        if rows_to_append:
            print(
                f"[DEBUG] Force-flushed {len(rows_to_append)} remaining cycles, "
                f"total df rows: {len(self.df)}"
            )

    ##############################################################################################
    # PUBLIC API. THIS SHOULD NOT BE CHANGED IN TERMS OF FUNCTIONALITY
    def fetch_next_window(
        self,
        curr_first_timestamp,
        for_training: bool = False,
    ):
        df = self.df
        print("[DEBUG] self.df.shape", self.df.shape)
        if df.empty:
            return None

        if curr_first_timestamp is None:
            start = 0
        else:
            idx = df.index[df["timestamp"] == curr_first_timestamp]
            if len(idx) == 0:
                return None
            start = idx[0] + self.stride

        end = start + self.history_window

        if end > len(df):
            return None

        X = df.iloc[start:end]

        if for_training:
            y_start = end - 1
            y_end = y_start + self.prediction_window
            if y_end > len(df):
                return None
            y = df.iloc[y_start:y_end][[self.target_name]]
            return X, y

        print(f"[DEBUG] Found Inference Window : {X}")

        return X

    def fetch_train_data(self):
        if self.df.empty:
            print("[DEBUG] DataFrame is empty, no training data available.")
            return None, None, None

        is_unsupervised = self.target_name not in self.df.columns

        X_list = []
        Y_list = []
        timestamps_list = []
        curr_first_timestamp = None

        curr_idx = 0

        while True:
            if is_unsupervised:
                start = curr_idx
                end = start + self.history_window
                if end > len(self.df):
                    break
                X_df = self.df.iloc[start:end]
                timestamps = (
                    X_df["timestamp"].to_numpy()
                    if "timestamp" in X_df.columns
                    else None
                )
                X_seq = X_df.drop(columns=["timestamp"], errors="ignore").to_numpy()
                X_list.append(X_seq)
                if timestamps is not None:
                    timestamps_list.append(timestamps)
                curr_idx += self.stride if self.stride > 0 else 1
            else:
                out = self.fetch_next_window(
                    curr_first_timestamp,
                    for_training=True,
                )
                if out is None:
                    break
                X_df, y_df = out
                X_seq = X_df.drop(columns=["timestamp"]).to_numpy()
                y_val = y_df.drop(columns=["timestamp"], errors="ignore").to_numpy()
                if y_val.ndim > 1:
                    y_val = y_val[-1]
                X_list.append(X_seq)
                Y_list.append(y_val)
                curr_first_timestamp = X_df.iloc[0]["timestamp"]
                print(f"[DEBUG] Advancing to timestamp: {curr_first_timestamp}")

        print(f"[DEBUG] Total windows built: {len(X_list)}")

        if not X_list:
            return None, None, None

        X_train = np.stack(X_list)
        Y_train = np.stack(Y_list) if Y_list else None

        print(
            f"[DEBUG] X_train shape: {X_train.shape}, "
            f"Y_train: {'None (unsupervised)' if Y_train is None else Y_train.shape}"
        )

        timestamps_out = np.stack(timestamps_list) if timestamps_list else None
        return X_train, Y_train, timestamps_out


if __name__ == "__main__":
    import time

    import yaml
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    logging.basicConfig(level=logging.INFO)

    DATABASE_URL = "postgresql://postgres:<password>@localhost:5432/glue-dispenser-db"

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)

    def session_factory():
        return SessionLocal()

    def load_config(path="../../config/analysis_config.yaml"):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    test_config = load_config()["target"]["system__cycle_time"][0]
    print(test_config)
    handler = DataHandler(
        session_factory=session_factory,
        config=test_config,
        target_name="system__cycle_time",
    )
    curr_first_timestamp = None
    while True:
        start = time.time()
        X = handler.fetch_next_window(curr_first_timestamp, for_training=False)
        print("Time taken: ", time.time() - start)
        print("===================================================================")
        curr_first_timestamp = X.iloc[0]["timestamp"]

        if X is None:
            print("❌ Not enough data for window")
            break
        else:
            print("✅ Window shape:", X.shape)
            print(X)
