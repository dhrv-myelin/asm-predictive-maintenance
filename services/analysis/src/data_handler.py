from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataHandler:
    def __init__(self, config: dict, target_name: str):
        self.config = config
        self.target_name = target_name
        self.format = config.get("format", "cycle")

        self.history_window = config["history_window"]
        self.stride = config.get("stride", 0)
        self.prediction_window = config["prediction_window"]
        self.required_features = config["required_features"]

        # ----------------------------
        # Buffers
        # ----------------------------
        if self.format == "cycle":
            # one row per (day, cycle)
            self._buffer: dict[tuple[date, int], dict] = {}
            self._max_cycle_per_day: dict[date, int] = {}

        elif self.format == "time":
            # multiple rows (events) per bucket
            self._time_bucket_seconds = config.get("bucket_duration", 60)
            self._buffer: dict[int, list[dict]] = {}
            self._latest_bucket_seen = -1

        # dataframe stays wide schema, but time-mode will be sparse
        self.df = pd.DataFrame(columns=["timestamp"] + self.required_features)

    # ----------------------------
    # Keying
    # ----------------------------
    def _get_buffer_key(self, ts, cycle_count):
        if self.format == "cycle":
            return (ts.date(), cycle_count)
        else:
            return int(ts.timestamp() // self._time_bucket_seconds)

    # ----------------------------
    # Completeness logic
    # ----------------------------
    def _is_complete(self, key) -> bool:
        if self.format == "cycle":
            day, cycle = key
            max_cycle = self._max_cycle_per_day.get(day, 0)
            return cycle < max_cycle
        else:
            return key < self._latest_bucket_seen

    # ----------------------------
    # Ingest
    # ----------------------------
    def ingest(self, rows):
        if not rows:
            return None

        for r in rows:
            ts = pd.to_datetime(r.timestamp)
            nmn = f"{r.station_name}__{r.metric_name}"
            cycle_count = r.cycle_count

            key = self._get_buffer_key(ts, cycle_count)

            if self.format == "cycle":
                # --- wide aggregation ---
                if key not in self._buffer:
                    self._buffer[key] = {"timestamp": ts}

                self._buffer[key][nmn] = r.value

                # track cycle completeness
                day = ts.date()
                if (
                    day not in self._max_cycle_per_day
                    or cycle_count > self._max_cycle_per_day[day]
                ):
                    self._max_cycle_per_day[day] = cycle_count

            else:
                # --- event-level storage ---
                if key not in self._buffer:
                    self._buffer[key] = []

                self._buffer[key].append({"timestamp": ts, nmn: r.value})

                # track latest bucket seen
                if key > self._latest_bucket_seen:
                    self._latest_bucket_seen = key

        self._flush_completed()

        return self.df if not self.df.empty else None

    # ----------------------------
    # Flush
    # ----------------------------
    def _flush_completed(self):
        keys_to_flush = [k for k in self._buffer if self._is_complete(k)]
        keys_to_flush.sort()

        for key in keys_to_flush:
            if self.format == "cycle":
                row = self._buffer.pop(key)
                self._append_row(row)

            else:
                rows = self._buffer.pop(key)

                # enforce ordering within bucket
                rows.sort(key=lambda x: x["timestamp"])

                for row in rows:
                    self._append_row(row)

    def flush_remaining(self):
        for key in sorted(self._buffer.keys()):
            if self.format == "cycle":
                self._append_row(self._buffer.pop(key))
            else:
                rows = self._buffer.pop(key)
                rows.sort(key=lambda x: x["timestamp"])
                for row in rows:
                    self._append_row(row)

    # ----------------------------
    # Append
    # ----------------------------
    def _append_row(self, row_dict):
        # ensure all columns exist (sparse-safe)
        row = {col: row_dict.get(col, None) for col in self.df.columns}
        self.df.loc[len(self.df)] = row

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
    from types import SimpleNamespace

    def make_rows(df: pd.DataFrame):
        """
        Convert dataframe rows → objects with attribute access
        """
        return [
            SimpleNamespace(
                timestamp=row["timestamp"],
                station_name=row["station_name"],
                metric_name=row["metric_name"],
                value=row["value"],
                cycle_count=row.get("cycle_count", 0),
            )
            for _, row in df.iterrows()
        ]

    def run_test(csv_path: str, format_mode: str):
        print(f"\n=== Running test | format = {format_mode} ===")

        config = {
            "format": format_mode,
            "history_window": 10,
            "prediction_window": 2,
            "bucket_duration": 60,
            "required_features": [
                "rbw_station__z_axis_positioning_wait",
                "rbw_station__pre_camera_positioning_wait",
                "rbw_station__marking_galvo_positioning_time",
                "rbw_station__post_homing_wait",
                "system__m1_position_actual",
                "rbw_station__z_axis_target_height",
                "rbw_station__pre_image_save_wait",
                "rbw_station__m2_position_target",
                "rbw_station__between_cluster_gantry_positioning_wait",
                "rbw_station__gantry_move_z",
                "rbw_station__reinspection_time",
                "rbw_station__gantry_move_m2",
                "rbw_station__marking_init_time",
                "rbw_station__double_exposure_positioning_time",
                "rbw_station__z_axis_position_target",
                "rbw_station__marking_result_wait",
                "rbw_station__clamping_time",
                "rbw_station__product_release_wait",
                "rbw_station__pre_double_exposure_wait_time",
                "rbw_station__pre_marking_wait",
                "rbw_station__gantry_positioning_time",
                "rbw_station__camera_positioning_time",
                "rbw_station__gantry_move_x",
                "rbw_station__galvo_1_position",
                "rbw_station__camera_image_save_time",
                "rbw_station__unclamping_time",
                "rbw_station__marking_repeat_wait",
                "system__x_axis_position_actual",
                "system__x_axis_position_target",
                "rbw_station__galvo_2_position",
                "rbw_station__marking_to_reinspection_wait",
                "rbw_station__gantry_move_m1",
                "rbw_station__m2_position_actual",
                "rbw_station__upstream_waiting_time",
                "rbw_station__pre_data_handshake_wait",
                "rbw_station__data_handshake_time",
                "rbw_station__z_axis_positioning_time",
                "rbw_station__z_axis_homing_time",
                "system__m1_position_target",
                "rbw_station__z_axis_position_actual",
            ],
        }

        handler = DataHandler(config=config, target_name="dummy")

        # ----------------------------
        # Load + sort
        # ----------------------------
        df = pd.read_csv(csv_path, nrows=200)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Sanity check (optional but useful)
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("Timestamps are not monotonic after sorting")

        # ----------------------------
        # Ingest by timestamp (KEY FIX)
        # ----------------------------
        for ts, group in df.groupby("timestamp", sort=True):
            rows = make_rows(group)

            print(f"\n--- Ingesting timestamp {ts} ({len(group)} rows) ---")
            out = handler.ingest(rows)

            if out is not None:
                print("Flushed rows:")
                print(out.tail(3))
            else:
                print("No flush yet")

        # ----------------------------
        # Final flush
        # ----------------------------
        print("\n--- Final flush ---")
        handler.flush_remaining()

        if not handler.df.empty:
            print(handler.df.tail(5))

    csv_path = "/home/dhruvkumarjiguda/code/asm-predictive-maintenance/services/analysis/dataset_gen/process_metrics.csv"

    # Run both modes
    run_test(csv_path, format_mode="cycle")
    run_test(csv_path, format_mode="time")
