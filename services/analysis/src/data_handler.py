from __future__ import annotations
import copy
import logging
from collections import deque

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Static key-sets used to classify each new_metric_name
# ─────────────────────────────────────────────────────────────────────────────
# All known shuttle/fmt bare keys (regardless of required_features)
_ALL_FMT_BARE_KEYS: set[str] = {""}

# All known end/summary keys
_ALL_END_KEYS: set[str] = {
    "system__Mode_value",
    "system__Online_value",
    "system__Priority_value",
    "system__TestSercycle_timeiesID_value",
    "system__TestSeriesID_value",
    "system__cam_uv_glue_sn",
    "system__cavity_1_dispenser_start_point_a_value",
    "system__cavity_1_dispenser_start_point_x_value",
    "system__cavity_1_dispenser_start_point_y_value",
    "system__cavity_2_dispenser_start_point_a_value",
    "system__cavity_2_dispenser_start_point_x_value",
    "system__cavity_2_dispenser_start_point_y_value",
    "system__cavity_3_dispenser_start_point_a_value",
    "system__cavity_3_dispenser_start_point_x_value",
    "system__cavity_3_dispenser_start_point_y_value",
    "system__cavity_4_dispenser_start_point_a_value",
    "system__cavity_4_dispenser_start_point_x_value",
    "system__cavity_4_dispenser_start_point_y_value",
    "system__cavity_5_dispenser_start_point_a_value",
    "system__cavity_5_dispenser_start_point_x_value",
    "system__cavity_5_dispenser_start_point_y_value",
    "system__cavity_6_dispenser_start_point_a_value",
    "system__cavity_6_dispenser_start_point_x_value",
    "system__cavity_6_dispenser_start_point_y_value",
    "system__cm_vendor_value",
    "system__cycle_time_value",
    "system__dispense_height_value",
    "system__dispense_speed_value",
    "system__dispense_voltage_value",
    "system__gantry_cpk_x_lower",
    "system__gantry_cpk_x_upper",
    "system__gantry_cpk_x_value",
    "system__gantry_cpk_y_lower",
    "system__gantry_cpk_y_upper",
    "system__gantry_cpk_y_value",
    "system__glue_weight_lower",
    "system__glue_weight_upper",
    "system__glue_weight_value",
    "system__main_valve_temp",
    "system__nozzle_temp_lower",
    "system__nozzle_temp_upper",
    "system__nozzle_temp_value",
    "system__operator_id_value",
    "system__pressure_value",
    "system__pulse_value",
    "system__raising_time_value",
    "system__single_dot_valve1_lower",
    "system__single_dot_valve1_upper",
    "system__single_dot_valve1_value",
    "system__single_dot_valve2_lower",
    "system__single_dot_valve2_upper",
    "system__single_dot_valve2_value",
    "system__striking_time_value",
    "system__sub_valve_temp",
    "system__tossing_lower",
    "system__tossing_upper",
    "system__tossing_value",
    "system__unique_pallet_count",
    "timestamp",
}

# All known L1 keys (excluding 'timestamp')
_ALL_L1_KEYS: set[str] = {
    "ced_station__barcode_scanning_time",
    "ced_station__cavity_1_dispensing_time",
    "ced_station__cavity_2_dispensing_time",
    "ced_station__cavity_3_dispensing_time",
    "ced_station__cavity_4_dispensing_time",
    "ced_station__cavity_5_dispensing_time",
    "ced_station__cavity_6_dispensing_time",
    "ced_station__downstream_waiting_time",
    "ced_station__entry_stopper_eval_delay",
    "ced_station__entry_stopper_lowering_time",
    "ced_station__entry_stopper_raising_time",
    "ced_station__exit_stopper_lowering_time",
    "ced_station__exit_stopper_raising_time",
    "ced_station__inspection_time",
    "ced_station__movein_to_entry_stopper_up_delay",
    "ced_station__pallet_clamping_time",
    "ced_station__pallet_lifting_time",
    "ced_station__pallet_lowering_time",
    "ced_station__pallet_movein_time",
    "ced_station__pallet_moveout_time",
    "ced_station__pallet_unclamping_time",
    "ced_station__pdca_conn_time",
    "ced_station__pdca_upload_time",
    "ced_station__post_pallet_lifting_delay",
    "ced_station__pre_cavity_2_dispensing_delay",
    "ced_station__pre_cavity_3_dispensing_delay",
    "ced_station__pre_cavity_4_dispensing_delay",
    "ced_station__pre_cavity_5_dispensing_delay",
    "ced_station__pre_cavity_6_dispensing_delay",
    "ced_station__pre_clamping_delay",
    "ced_station__pre_dispensing_delay",
    "ced_station__pre_inspection_delay",
    "ced_station__pre_pallet_lifting_delay",
    "ced_station__pre_pallet_lowering_delay",
    "ced_station__pre_sfc_query_delay",
    "ced_station__sfc_conn_time",
    "ced_station__sfc_query_processing_time",
    "ced_station__upstream_waiting_time",
    "gantry_positioning__gantry_safety_positioning_time",
}

# All known L2 keys (excluding 'timestamp')
_ALL_L2_KEYS: set[str] = {
    "ced_maintenance__maint_between_glue_purge_delay",
    "ced_maintenance__maint_between_nozzle_clean_delay",
    "ced_maintenance__maint_glue_purging_time",
    "ced_maintenance__maint_move_to_safe_time",
    "ced_maintenance__maint_nozzle_cleaning_time",
    "ced_maintenance__maint_post_glue_purge_delay",
    "ced_maintenance__maint_post_nozzle_clean_delay",
    "ced_maintenance__maintenance_waiting_time",
}


def _build_template(keys: list[str], sentinel=-1) -> dict:
    """Return an ordered dict with every key set to the sentinel value."""
    return {k: sentinel for k in keys}


class DataHandler:
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict, target_name: str):
        self.config = config
        self.target_name = target_name

        # ── config extraction ──────────────────────────────────────────
        self.required_features: list[str] = config["required_features"]
        print("Required features : ", self.required_features)
        self.format = config.get("format", None)  # 'wide' | None
        self.history_window: int = config["history_window"]
        self.stride: int = config.get("stride", 0) or 0
        self.prediction_window: int = config["prediction_window"]

        # ── derive templates from required_features ────────────────────
        #    Each group only includes keys that appear in required_features
        #    (plus 'timestamp' for l1/l2 which is always tracked internally).
        self._fmt_keys = [k for k in self.required_features if k in _ALL_FMT_BARE_KEYS]
        print("FMT KEYS : ", self._fmt_keys)
        self._end_keys = [k for k in self.required_features if k in _ALL_END_KEYS]
        print("END KEYS : ", self._end_keys)
        self._l1_keys = [k for k in self.required_features if k in _ALL_L1_KEYS]
        print("L1 KEYS : ", self._l1_keys)
        self._l2_keys = [k for k in self.required_features if k in _ALL_L2_KEYS]
        print("L2 KEYS : ", self._l2_keys)

        # 'timestamp' is always tracked internally for L1/L2/end even if not
        # in required_features (used for queue ordering). IMPORTANT: keep
        # _end_keys as the pure feature list — _try_flush uses it to decide
        # whether an end queue entry is required. Only _end_internal_keys
        # (used to build _active_end) gets the timestamp appended.
        self._fmt_internal_keys = self._fmt_keys + (
            ["timestamp"] if "timestamp" not in self._fmt_keys else []
        )
        self._end_internal_keys = self._end_keys + (
            ["timestamp"] if "timestamp" not in self._end_keys else []
        )
        self._l1_internal_keys = self._l1_keys + (
            ["timestamp"] if "timestamp" not in self._l1_keys else []
        )
        self._l2_internal_keys = self._l2_keys + (
            ["timestamp"] if "timestamp" not in self._l2_keys else []
        )

        # ── lookup sets ────────────────────────────────────────────────
        self._fmt_bare_set = set(self._fmt_keys)
        self._end_set = set(self._end_internal_keys)  # includes timestamp
        self._l1_set = set(self._l1_keys)
        self._l2_set = set(self._l2_keys)

        # ── persistent active dicts (survive across ingest() calls) ───
        self._active_fmt = _build_template(self._fmt_internal_keys)
        self._active_l1 = _build_template(self._l1_internal_keys)
        self._active_l2 = _build_template(self._l2_internal_keys)
        self._active_end = _build_template(self._end_internal_keys)

        # ── completed-dict queues (FIFO) ───────────────────────────────
        self._fmt_queue: deque[dict] = deque()
        self._l1_queue: deque[dict] = deque()
        self._l2_queue: deque[dict] = deque()
        self._end_queue: deque[dict] = deque()

        # ── output row accumulators ────────────────────────────────────
        self._l1_rows: list[dict] = []
        self._l2_rows: list[dict] = []

        # ── final wide DataFrame (grows as cycles complete) ────────────
        self.df = pd.DataFrame(columns=self.required_features)

        # Last timestamp seen across any completed cycle (used by inference_loop
        # for models that don't store timestamp in self.df)
        self.last_timestamp = None

    # ------------------------------------------------------------------
    # Public: ingest rows from the poller
    # ------------------------------------------------------------------

    def ingest(self, rows) -> pd.DataFrame | None:
        """
        Accept one or more raw poller rows, update the active dicts, flush
        any newly completed cycles into self.df, and return self.df.
        """
        print(f"[DEBUG] Ingesting rows: {len(rows)} rows")
        if not rows:
            return None

        for r in rows:
            ts = pd.to_datetime(r.timestamp)
            nmn = f"{r.station_name}__{r.metric_name}"
            val = r.value
            self._process_one(nmn, val, ts)
        # print(f"[DEBUG] Recieved :: {ts} : {nmn} : {val}")
        return self.df if not self.df.empty else None

    # ------------------------------------------------------------------
    # Internal: process a single (new_metric_name, value, timestamp)
    # ------------------------------------------------------------------
    def _is_complete(self, d: dict) -> bool:
        return all(v != -1 for v in d.values())

    def _process_one(self, nmn: str, val, ts: pd.Timestamp) -> None:

        # ── (a) shuttle / fmt metrics ──────────────────────────────────
        if nmn in self._fmt_bare_set:
            # print(f"[DEBUG] Processing fmt metric: {nmn} with value {val} at timestamp {ts}")
            if self._active_fmt.get(nmn, -1) == -1:
                self._active_fmt[nmn] = val
                # print(f"[DEBUG] Updated active_fmt: {self._active_fmt}")
            self._active_fmt["timestamp"] = ts  # always keep latest ts
            if self._fmt_keys and self._is_complete(self._active_fmt):
                # print("[DEBUG] Completed fmt dict: ", self._active_fmt)
                self._fmt_queue.append(copy.copy(self._active_fmt))
                self._active_fmt = _build_template(self._fmt_internal_keys)
                self._try_flush()
            return

        # ── (b) end / summary metrics ──────────────────────────────────
        if nmn in self._end_set:
            # print(f"[DEBUG] Processing fmt metric: {nmn} with value {val} at timestamp {ts}")
            if self._active_end.get(nmn, -1) == -1:
                self._active_end[nmn] = val
                # print(f"[DEBUG] Updated active_end: {self._active_end}")
            self._active_end["timestamp"] = ts
            if self._end_keys and self._is_complete(self._active_end):
                self._end_queue.append(copy.copy(self._active_end))
                self._active_end = _build_template(self._end_keys)
                self._try_flush()
            return

        # ── (c) L1 metrics ─────────────────────────────────────────────
        if nmn in self._l1_set:
            if self._active_l1.get(nmn, -1) == -1:
                self._active_l1[nmn] = val
            self._active_l1["timestamp"] = ts  # always keep latest ts
            if self._l1_keys and self._is_complete(self._active_l1):
                # print("[DEBUG] Completed L1 dict: ", self._active_l1)
                self._l1_queue.append(copy.copy(self._active_l1))
                self._active_l1 = _build_template(self._l1_internal_keys)
                self._try_flush()
            return

        # ── (d) L2 metrics ─────────────────────────────────────────────
        if nmn in self._l2_set:
            # print(f"[DEBUG] Processing fmt metric: {nmn} with value {val} at timestamp {ts}")
            if self._active_l2.get(nmn, -1) == -1:
                self._active_l2[nmn] = val
                # print(f"[DEBUG] Updated active_l2: {self._active_l2}")
            self._active_l2["timestamp"] = ts
            if self._l2_keys and self._is_complete(self._active_l2):
                self._l2_queue.append(copy.copy(self._active_l2))
                self._active_l2 = _build_template(self._l2_internal_keys)
                self._try_flush()
            return

    # ------------------------------------------------------------------
    # Internal: flush completed cycles into output rows
    # ------------------------------------------------------------------

    def _try_flush(self) -> None:
        """
        Merge one entry from each queue into a single wide row and append
        it to self.df.  Keeps flushing as long as all queues have entries.

        If fmt_keys or end_keys are empty (not in required_features),
        those dicts are treated as always-satisfied (empty dict).
        """
        while True:
            fmt_ready = bool(self._fmt_queue) or not self._fmt_keys
            end_ready = bool(self._end_queue) or not self._end_keys
            line_has_keys = bool(self._l1_keys) or bool(self._l2_keys)
            line_ready = (
                bool(self._l1_queue) or bool(self._l2_queue)
            ) or not line_has_keys

            if not (fmt_ready and end_ready and line_ready):
                break

            fmt_part = self._fmt_queue.popleft() if self._fmt_keys else {}
            end_part = self._end_queue.popleft() if self._end_keys else {}

            # Only attempt to pop a line entry if there are line keys at all
            if line_has_keys:
                if self._l1_queue and self._l2_queue:
                    ts1 = self._l1_queue[0].get("timestamp", pd.NaT)
                    ts2 = self._l2_queue[0].get("timestamp", pd.NaT)
                    use_l1 = ts1 <= ts2
                else:
                    use_l1 = bool(self._l1_queue)
                line_part = (
                    self._l1_queue.popleft() if use_l1 else self._l2_queue.popleft()
                )
            else:
                line_part = {}

            merged = {**fmt_part, **line_part, **end_part}

            # Track the latest timestamp even if not in required_features
            if "timestamp" in merged and merged["timestamp"] is not None:
                self.last_timestamp = merged["timestamp"]

            # Build a one-row DataFrame with only required_features columns
            row_df = pd.DataFrame([merged])
            # Keep only required_features (drops 'timestamp' if not requested)
            for col in self.required_features:
                if col not in row_df.columns:
                    row_df[col] = pd.NA
            row_df = row_df[self.required_features]

            # Replace stray sentinels with NA
            row_df = row_df.replace(-1, pd.NA)
            # row_df = row_df.fillna(-1)
            print(f"[DEBUG] Completed one cycle (End timestamp : {merged['timestamp']}), appending to DataHandler df")
            self.df = (
                pd.concat([self.df, row_df], ignore_index=True)
                if not self.df.empty
                else row_df.copy()
            )
            self.df.drop_duplicates(inplace=True)

    # ------------------------------------------------------------------
    # Public: sliding-window fetch (inference)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public: return ALL available (X, y) training pairs
    # ------------------------------------------------------------------

    def fetch_train_data(self):
        """
        Build training data in sequence format:
        X.shape = (N, seq_len, num_features)
        Y.shape = (N, num_targets), or None for unsupervised models.

        Unsupervised detection: if target_name is not a column in self.df
        (e.g. health_score targets like "l1_buffer_a__health_score" are never
        ingested as feature columns), y is returned as None. Callers must
        check for this and route to an unsupervised training path.
        """
        if self.df.empty:
            print("[DEBUG] DataFrame is empty, no training data available.")
            return None, None, None

        # health_score: target_name is e.g. "l1_buffer_a__health_score" —
        # never a column in df, which only holds the 4 buffer feature cols.
        is_unsupervised = self.target_name not in self.df.columns

        X_list = []
        Y_list = []
        timestamps_list = []
        curr_first_timestamp = None

        # For unsupervised models without a timestamp column, track position
        # by integer index to avoid KeyError in fetch_next_window.
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
                )  # ← capture timestamps
                X_seq = X_df.drop(columns=["timestamp"], errors="ignore").to_numpy()
                X_list.append(X_seq)
                if timestamps is not None:
                    timestamps_list.append(timestamps)  # ← store them
                curr_idx += self.stride
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
            # print("[DEBUG] No training windows could be built from the data.")
            return None, None, None

        X_train = np.stack(X_list)  # (N, seq_len, num_features)
        Y_train = np.stack(Y_list) if Y_list else None  # None for unsupervised

        print(
            f"[DEBUG] X_train shape: {X_train.shape}, "
            f"Y_train: {'None (unsupervised)' if Y_train is None else Y_train.shape}"
        )

        timestamps_out = np.stack(timestamps_list) if timestamps_list else None
        return X_train, Y_train, timestamps_out


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import yaml
    import time

    logging.basicConfig(level=logging.INFO)

    # --------------------------------------------------
    # DB setup
    # --------------------------------------------------

    DATABASE_URL = "postgresql://postgres:<password>@localhost:5432/glue-dispenser-db"
    # ⬆️ change to your real DB URL

    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)

    def session_factory():
        return SessionLocal()

    def load_config(path="../../config/analysis_config.yaml"):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    # --------------------------------------------------
    # Minimal config for testing
    # --------------------------------------------------
    test_config = load_config()["target"]["system__cycle_time"][0]
    # print(test_config)
    # --------------------------------------------------
    # Init handler
    # --------------------------------------------------
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
        # print(X)
        print("Time taken: ", time.time() - start)
        print("===================================================================")
        curr_first_timestamp = X.iloc[0]["timestamp"]

        if X is None:
            print("❌ Not enough data for window")
            break
        else:
            print("✅ Window shape:", X.shape)
            print(X)

        # --------------------------------------------------
        # Training window test
        # --------------------------------------------------
        # start = time.time()
        # XY = handler.fetch_next_window(curr_first_timestamp=None, for_training=True)
        # print("Time taken: ", time.time()-start)

        # if XY is None:
        #     print("❌ Not enough data for training window")
        # else:
        #     X_tr, y_tr = XY
        #     print("\nTraining X shape:", X_tr.shape)
        #     print("Training y shape:", y_tr.shape)
        #     print("\nX sample:")
        #     print(X_tr.head())
        #     print("\ny sample:")
        #     print(y_tr.head())
        # print("===================================================================")
        # curr_first_timestamp = XY[0].iloc[0]['timestamp']

    # print("\n✅ DataHandler verification complete")
