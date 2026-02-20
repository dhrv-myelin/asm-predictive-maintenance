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
_SHUTTLE_STATION_PREFIX = "shuttle_station__"

# All known shuttle/fmt bare keys (regardless of required_features)
_ALL_FMT_BARE_KEYS: set[str] = {
    "system__count",
    "shuttle_pre_scan_time",
    "shuttle_scan_success_time",
    "shuttle_post_scan_delay",
    "shuttle_carrier_motion_time",
    "shuttle_stopper_lowering_time",
    "shuttle_pallet_release_time",
    "shuttle_stopper_rising_time",
}

# All known end/summary keys
_ALL_END_KEYS: set[str] = {
    "system__cycle_time",
    "system__total_time",
    "system__rolling_uph",
    "system__ng_pallets",
    "system__yield",
    "system__total_units",
    "system__total_pallets",
    "system__ok_pallets",
}

# All known L1 keys (excluding 'timestamp')
_ALL_L1_KEYS: set[str] = {
    "l1_buffer_a__buffer_holding_time",
    "l1_buffer_a__buffer_stopper_lowering_time",
    "l1_buffer_a__buffer_pallet_release_time",
    "l1_buffer_a__buffer_stopper_rising_time",
    "l1_dispenser__dispenser_entry_unclamp_time",
    "l1_dispenser__dispenser_delay_post_entry_unclamp",
    "l1_dispenser__dispenser_pallet_lifting_1_time",
    "l1_dispenser__dispenser_entry_clamping_time",
    "l1_dispenser__dispenser_delay_post_clamping",
    "l1_dispenser__dispenser_pallet_stopper_releasing_time",
    "l1_dispenser__dispenser_delay_pre_gantry_positioning",
    "l1_dispenser__dispenser_gantry_positioning_time",
    "l1_dispenser__dispenser_dispensing_time",
    "l1_dispenser__dispenser_pallet_unpositioning_time",
    "l1_dispenser__dispenser_pallet_stopper_locking_time",
    "l1_dispenser__dispenser_pre_exit_unclamping",
    "l1_dispenser__dispenser_exit_unclamping",
    "l1_dispenser__dispenser_post_exit_unclamp",
    "l1_dispenser__dispenser_pallet_lowering_1",
    "l1_dispenser__dispenser_stopper_lowering",
    "l1_dispenser__dispenser_pallet_releasing",
    "l1_dispenser__dispenser_stopper_rasing",
    "l1_inspection__inspection_lifter_rising_time",
    "l1_inspection__inspection_gantry_positioning_time",
    "l1_inspection__inspection_core_vision_system_time",
    "l1_inspection__inspection_post_vision_check_delay",
    "l1_inspection__inspection_lifter_lowering_time",
    "l1_inspection__inspection_pallet_releasing_time",
}

# All known L2 keys (excluding 'timestamp')
_ALL_L2_KEYS: set[str] = {
    "l2_dispenser__dispenser_entry_unclamp_time",
    "l2_dispenser__dispenser_delay_post_entry_unclamp",
    "l2_dispenser__dispenser_pallet_lifting_1_time",
    "l2_dispenser__dispenser_entry_clamping_time",
    "l2_dispenser__dispenser_delay_post_clamping",
    "l2_dispenser__dispenser_pallet_stopper_releasing_time",
    "l2_dispenser__dispenser_delay_pre_gantry_positioning",
    "l2_dispenser__dispenser_gantry_positioning_time",
    "l2_dispenser__dispenser_dispensing_time",
    "l2_dispenser__dispenser_pallet_unpositioning_time",
    "l2_dispenser__dispenser_pallet_stopper_locking_time",
    "l2_dispenser__dispenser_pre_exit_unclamping",
    "l2_dispenser__dispenser_exit_unclamping",
    "l2_dispenser__dispenser_post_exit_unclamp",
    "l2_dispenser__dispenser_pallet_lowering_1",
    "l2_dispenser__dispenser_stopper_lowering",
    "l2_dispenser__dispenser_pallet_releasing",
    "l2_dispenser__dispenser_stopper_rasing",
    "l2_inspection__inspection_lifter_rising_time",
    "l2_inspection__inspection_gantry_positioning_time",
    "l2_inspection__inspection_core_vision_system_time",
    "l2_inspection__inspection_post_vision_check_delay",
    "l2_inspection__inspection_lifter_lowering_time",
    "l2_inspection__inspection_pallet_releasing_time",
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
        self.format = config.get("format", None)        # 'wide' | None
        self.history_window: int = config["history_window"]
        self.stride: int = config.get("stride", 0) or 0
        self.prediction_window: int = config["prediction_window"]

        # ── derive templates from required_features ────────────────────
        #    Each group only includes keys that appear in required_features
        #    (plus 'timestamp' for l1/l2 which is always tracked internally).
        self._fmt_keys  = [k for k in self.required_features if k in _ALL_FMT_BARE_KEYS]
        self._end_keys  = [k for k in self.required_features if k in _ALL_END_KEYS]
        self._l1_keys   = [k for k in self.required_features if k in _ALL_L1_KEYS]
        self._l2_keys   = [k for k in self.required_features if k in _ALL_L2_KEYS]

        # 'timestamp' is always tracked internally for L1/L2 even if not in
        # required_features (used for queue ordering); it is removed from the
        # final output if absent from required_features.
        self._l1_internal_keys = self._l1_keys + (
            ["timestamp"] if "timestamp" not in self._l1_keys else []
        )
        self._l2_internal_keys = self._l2_keys + (
            ["timestamp"] if "timestamp" not in self._l2_keys else []
        )

        # ── lookup sets ────────────────────────────────────────────────
        self._fmt_bare_set = set(self._fmt_keys)
        self._end_set      = set(self._end_keys)
        self._l1_set       = set(self._l1_keys)
        self._l2_set       = set(self._l2_keys)

        # ── persistent active dicts (survive across ingest() calls) ───
        self._active_fmt = _build_template(self._fmt_keys)
        self._active_l1  = _build_template(self._l1_internal_keys)
        self._active_l2  = _build_template(self._l2_internal_keys)
        self._active_end = _build_template(self._end_keys)

        # ── completed-dict queues (FIFO) ───────────────────────────────
        self._fmt_queue: deque[dict] = deque()
        self._l1_queue:  deque[dict] = deque()
        self._l2_queue:  deque[dict] = deque()
        self._end_queue: deque[dict] = deque()

        # ── output row accumulators ────────────────────────────────────
        self._l1_rows: list[dict] = []
        self._l2_rows: list[dict] = []

        # ── final wide DataFrame (grows as cycles complete) ────────────
        self.df = pd.DataFrame(columns=self.required_features)

    # ------------------------------------------------------------------
    # Public: ingest rows from the poller
    # ------------------------------------------------------------------

    def ingest(self, rows) -> pd.DataFrame | None:
        """
        Accept one or more raw poller rows, update the active dicts, flush
        any newly completed cycles into self.df, and return self.df.
        """
        if not rows:
            return None

        for r in rows:
            ts  = pd.to_datetime(r.timestamp)
            nmn = f"{r.station_name}__{r.metric_name}"
            val = r.value
            self._process_one(nmn, val, ts)
        # print(f"[DEBUG] Recieved :: {ts} : {nmn} : {val}")
        return self.df if not self.df.empty else None

    # ------------------------------------------------------------------
    # Internal: process a single (new_metric_name, value, timestamp)
    # ------------------------------------------------------------------

    def _bare(self, nmn: str) -> str:
        if nmn.startswith(_SHUTTLE_STATION_PREFIX):
            return nmn[len(_SHUTTLE_STATION_PREFIX):]
        return nmn

    def _is_complete(self, d: dict) -> bool:
        return all(v != -1 for v in d.values())

    def _process_one(self, nmn: str, val, ts: pd.Timestamp) -> None:
        bare = self._bare(nmn)

        # ── (a) shuttle / fmt metrics ──────────────────────────────────
        if bare in self._fmt_bare_set:
            if self._active_fmt.get(bare, -1) == -1:
                self._active_fmt[bare] = val
            if self._fmt_keys and self._is_complete(self._active_fmt):
                self._fmt_queue.append(copy.copy(self._active_fmt))
                self._active_fmt = _build_template(self._fmt_keys)
            return

        # ── (b) end / summary metrics ──────────────────────────────────
        if nmn in self._end_set:
            if self._active_end.get(nmn, -1) == -1:
                self._active_end[nmn] = val
            if self._end_keys and self._is_complete(self._active_end):
                self._end_queue.append(copy.copy(self._active_end))
                self._active_end = _build_template(self._end_keys)
                self._try_flush()
            return

        # ── (c) L1 metrics ─────────────────────────────────────────────
        if nmn in self._l1_set:
            if self._active_l1.get(nmn, -1) == -1:
                self._active_l1[nmn] = val
            self._active_l1["timestamp"] = ts       # always keep latest ts
            if self._l1_keys and self._is_complete(self._active_l1):
                self._l1_queue.append(copy.copy(self._active_l1))
                self._active_l1 = _build_template(self._l1_internal_keys)
                self._try_flush()
            return

        # ── (d) L2 metrics ─────────────────────────────────────────────
        if nmn in self._l2_set:
            if self._active_l2.get(nmn, -1) == -1:
                self._active_l2[nmn] = val
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
            line_ready = bool(self._l1_queue) or bool(self._l2_queue)

            if not (fmt_ready and end_ready and line_ready):
                break

            fmt_part = self._fmt_queue.popleft() if self._fmt_keys else {}
            end_part = self._end_queue.popleft() if self._end_keys else {}

            # Choose oldest line entry
            if self._l1_queue and self._l2_queue:
                ts1 = self._l1_queue[0].get("timestamp", pd.NaT)
                ts2 = self._l2_queue[0].get("timestamp", pd.NaT)
                use_l1 = ts1 <= ts2
            else:
                use_l1 = bool(self._l1_queue)

            line_part = self._l1_queue.popleft() if use_l1 else self._l2_queue.popleft()

            merged = {**fmt_part, **line_part, **end_part}

            # Build a one-row DataFrame with only required_features columns
            row_df = pd.DataFrame([merged])
            # Keep only required_features (drops 'timestamp' if not requested)
            for col in self.required_features:
                if col not in row_df.columns:
                    row_df[col] = pd.NA
            row_df = row_df[self.required_features]

            # Replace stray sentinels with NA
            row_df = row_df.replace(-1, pd.NA)
            print("[DEBUG] Completed one cycle, appending to DataHandler df")
            self.df = (
                pd.concat([self.df, row_df], ignore_index=True)
                if not self.df.empty
                else row_df.copy()
            )

    # ------------------------------------------------------------------
    # Public: sliding-window fetch (inference)
    # ------------------------------------------------------------------

    def fetch_next_window(
        self,
        curr_first_timestamp,
        for_training: bool = False,
    ):
        df = self.df
        if df.empty:
            return None

        if curr_first_timestamp is None:
            start_idx = 0
        else:
            idx = df.index[df["timestamp"] == curr_first_timestamp]
            if len(idx) == 0:
                return None
            start_idx = idx[0]

        start = start_idx + self.stride
        end   = start + self.history_window

        if end > len(df):
            return None

        X = df.iloc[start:end]

        if for_training:
            y_start = end - 1
            y_end   = y_start + self.prediction_window
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
        Y.shape = (N, num_targets)
        """
        if self.df.empty:
            return None, None

        X_list = []
        Y_list = []

        curr_first_timestamp = None

        while True:
            out = self.fetch_next_window(
                curr_first_timestamp,
                for_training=True
            )

            if out is None:
                break

            X_df, y_df = out   # DataFrames

            # ---- convert to numpy sequences ----
            X_seq = X_df.drop(columns=["timestamp",self.target_name]).to_numpy()
            y_val = y_df.drop(columns=["timestamp"], errors="ignore").to_numpy()

            # if prediction_window > 1 → take last step target
            if y_val.ndim > 1:
                y_val = y_val[-1]

            X_list.append(X_seq)
            Y_list.append(y_val)

            # advance window
            curr_first_timestamp = X_df.iloc[0]["timestamp"]

        if not X_list:
            return None, None

        X_train = np.stack(X_list)   # (N, seq_len, num_features)
        Y_train = np.stack(Y_list)   # (N, num_targets)

        return X_train, Y_train



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
    test_config = load_config()['target']['system__cycle_time'][0]
    # print(test_config)
    # --------------------------------------------------
    # Init handler
    # --------------------------------------------------
    print(test_config)
    handler = DataHandler(
        session_factory=session_factory,
        config=test_config,
        target_name="system__cycle_time"
    )
    curr_first_timestamp = None
    while True:
        start = time.time()
        X = handler.fetch_next_window(curr_first_timestamp, for_training=False)
        # print(X)
        print("Time taken: ", time.time()-start)
        print("===================================================================")
        curr_first_timestamp = X.iloc[0]['timestamp']
        
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

