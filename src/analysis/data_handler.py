from __future__ import annotations
import pandas as pd
from sqlalchemy import text
import logging


import copy
from collections import deque
logger = logging.getLogger(__name__)


DICT_FMT_TEMPLATE = {
    'system__count':                  -1,
    'shuttle_pre_scan_time':          -1,
    'shuttle_scan_success_time':      -1,
    'shuttle_post_scan_delay':        -1,
    'shuttle_carrier_motion_time':    -1,
    'shuttle_stopper_lowering_time':  -1,
    'shuttle_pallet_release_time':    -1,
    'shuttle_stopper_rising_time':    -1,
}

DICT_L1_TEMPLATE = {
    'l1_buffer_a__buffer_holding_time':                     -1,
    'l1_buffer_a__buffer_stopper_lowering_time':            -1,
    'l1_buffer_a__buffer_pallet_release_time':              -1,
    'l1_buffer_a__buffer_stopper_rising_time':              -1,
    'l1_dispenser__dispenser_entry_unclamp_time':           -1,
    'l1_dispenser__dispenser_delay_post_entry_unclamp':     -1,
    'l1_dispenser__dispenser_pallet_lifting_1_time':        -1,
    'l1_dispenser__dispenser_entry_clamping_time':          -1,
    'l1_dispenser__dispenser_delay_post_clamping':          -1,
    'l1_dispenser__dispenser_pallet_stopper_releasing_time':-1,
    'l1_dispenser__dispenser_delay_pre_gantry_positioning': -1,
    'l1_dispenser__dispenser_gantry_positioning_time':      -1,
    'l1_dispenser__dispenser_dispensing_time':              -1,
    'l1_dispenser__dispenser_pallet_unpositioning_time':    -1,
    'l1_dispenser__dispenser_pallet_stopper_locking_time':  -1,
    'l1_dispenser__dispenser_pre_exit_unclamping':          -1,
    'l1_dispenser__dispenser_exit_unclamping':              -1,
    'l1_dispenser__dispenser_post_exit_unclamp':            -1,
    'l1_dispenser__dispenser_pallet_lowering_1':            -1,
    'l1_dispenser__dispenser_stopper_lowering':             -1,
    'l1_dispenser__dispenser_pallet_releasing':             -1,
    'l1_dispenser__dispenser_stopper_rasing':               -1,
    'l1_inspection__inspection_lifter_rising_time':         -1,
    'l1_inspection__inspection_gantry_positioning_time':    -1,
    'l1_inspection__inspection_core_vision_system_time':    -1,
    'l1_inspection__inspection_post_vision_check_delay':    -1,
    'l1_inspection__inspection_lifter_lowering_time':       -1,
    'l1_inspection__inspection_pallet_releasing_time':      -1,
    'timestamp':                                            -1,
}

DICT_L2_TEMPLATE = {
    'l2_dispenser__dispenser_entry_unclamp_time':           -1,
    'l2_dispenser__dispenser_delay_post_entry_unclamp':     -1,
    'l2_dispenser__dispenser_pallet_lifting_1_time':        -1,
    'l2_dispenser__dispenser_entry_clamping_time':          -1,
    'l2_dispenser__dispenser_delay_post_clamping':          -1,
    'l2_dispenser__dispenser_pallet_stopper_releasing_time':-1,
    'l2_dispenser__dispenser_delay_pre_gantry_positioning': -1,
    'l2_dispenser__dispenser_gantry_positioning_time':      -1,
    'l2_dispenser__dispenser_dispensing_time':              -1,
    'l2_dispenser__dispenser_pallet_unpositioning_time':    -1,
    'l2_dispenser__dispenser_pallet_stopper_locking_time':  -1,
    'l2_dispenser__dispenser_pre_exit_unclamping':          -1,
    'l2_dispenser__dispenser_exit_unclamping':              -1,
    'l2_dispenser__dispenser_post_exit_unclamp':            -1,
    'l2_dispenser__dispenser_pallet_lowering_1':            -1,
    'l2_dispenser__dispenser_stopper_lowering':             -1,
    'l2_dispenser__dispenser_pallet_releasing':             -1,
    'l2_dispenser__dispenser_stopper_rasing':               -1,
    'l2_inspection__inspection_lifter_rising_time':         -1,
    'l2_inspection__inspection_gantry_positioning_time':    -1,
    'l2_inspection__inspection_core_vision_system_time':    -1,
    'l2_inspection__inspection_post_vision_check_delay':    -1,
    'l2_inspection__inspection_lifter_lowering_time':       -1,
    'l2_inspection__inspection_pallet_releasing_time':      -1,
    'timestamp':                                            -1,
}

END_DICT_TEMPLATE = {
    'system__cycle_time':    -1,
    'system__total_time':    -1,
    'system__rolling_uph':   -1,
    'system__ng_pallets':    -1,
    'system__yield':         -1,
    'system__total_units':   -1,
    'system__total_pallets': -1,
    'system__ok_pallets':    -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOOKUP SETS  (built once for O(1) membership tests in the hot loop)
# ─────────────────────────────────────────────────────────────────────────────

_FMT_BARE_KEYS  = set(DICT_FMT_TEMPLATE.keys())   # bare keys
_L1_FULL_KEYS   = set(DICT_L1_TEMPLATE.keys()) - {'timestamp'}
_L2_FULL_KEYS   = set(DICT_L2_TEMPLATE.keys()) - {'timestamp'}
_END_FULL_KEYS  = set(END_DICT_TEMPLATE.keys())
# system__count is special — new_metric_name IS 'system__count' (no station prefix)
_SHUTTLE_STATION_PREFIX = "shuttle_station__"

class DataHandler:
    def __init__(self, session_factory, config, target_name):
        self._sf = session_factory
        self.config = config
        self.target_name = target_name

        # config extraction
        self.required_features = config["required_features"]
        self.format = config.get("format", None)   # wide | None
        self.history_window = config["history_window"]
        self.stride = config.get("stride", 0) or 0
        self.prediction_window = config["prediction_window"]

        # internal state
        # self.df = pd.DataFrame()
        self.df = self.fetch_data()
        self.last_written_timestamps = []

    # --------------------------------------------------
    # ingest from poller
    # --------------------------------------------------
    def fetch_data(self):
        sql = text("""
            SELECT timestamp, station_name, metric_name, value
            FROM process_metrics
            ORDER BY timestamp ASC
        """)

        with self._sf() as s:
            rows = s.execute(sql).fetchall()
        
        _df =  pd.DataFrame(rows, columns=["timestamp", "station_name", "metric_name", "value"])
        _df["timestamp"] = pd.to_datetime(_df["timestamp"])

        _df['new_metric_name'] = _df['station_name'] + '__' + _df['metric_name']
        _df = _df[~_df["new_metric_name"].str.contains("exit_sink", na=False)].copy()
        _df = _df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        
        # print("===================================================================")
        # print(_df)
        # print("===================================================================")
        return self._to_wide_df(_df)

    # --------------------------------------------------
    # formatting
    # --------------------------------------------------
    # def _to_wide(self, station, metric):
    #     if station == "system":
    #         return f"system__{metric}"
    #     return f"{station}__{metric}"

    # def _format(self):
    #     if self.format == "wide":
    #         return self._to_wide_df()
    #     else:
    #         return self.df.copy()   # univariate log format

    def _bare(self, new_metric_name: str) -> str:
        """Strip 'shuttle_station__' prefix so it matches DICT_FMT_TEMPLATE keys."""
        if new_metric_name.startswith(_SHUTTLE_STATION_PREFIX):
            return new_metric_name[len(_SHUTTLE_STATION_PREFIX):]
        return new_metric_name   # e.g. 'system__count' already bare


    def _is_complete(self, d: dict) -> bool:
        """True when no value in the dict is still the sentinel -1."""
        return all(v != -1 for v in d.values())

    def _to_wide_df(self, df):
        """
        Parameters
        ----------
        df : raw log dataframe with columns
            [timestamp, station_name, metric_name, value, unit,
            state_context, new_metric_name]

        Returns
        -------
        df_l1 : one row per completed L1 cycle
        df_l2 : one row per completed L2 cycle
        Both share the column layout:
            <dict_fmt keys> + <dict_lX keys> + <end_dict keys>
        """

        # # ── pre-process ───────────────────────────────────────────────────────────
        # df['new_metric_name'] = df['station_name'] + '__' + df['metric_name']
        # df = df[~df["new_metric_name"].str.contains("exit_sink", na=False)].copy()
        # df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

        # ── active (being filled) dicts ───────────────────────────────────────────
        active_fmt = copy.deepcopy(DICT_FMT_TEMPLATE)
        active_l1  = copy.deepcopy(DICT_L1_TEMPLATE)
        active_l2  = copy.deepcopy(DICT_L2_TEMPLATE)
        active_end = copy.deepcopy(END_DICT_TEMPLATE)

        # ── completed-dict queues (FIFO) ──────────────────────────────────────────
        fmt_queue: deque[dict] = deque()
        l1_queue:  deque[dict] = deque()
        l2_queue:  deque[dict] = deque()
        end_queue: deque[dict] = deque()

        # ── output row accumulators ───────────────────────────────────────────────
        l1_rows: list[dict] = []
        l2_rows: list[dict] = []

        # ── helper: try to flush a completed triple into the output lists ─────────
        def _try_flush() -> None:
            # Keep flushing as long as all three queues have at least one entry
            while fmt_queue and end_queue and (l1_queue or l2_queue):
                # Determine line from whichever line-queue is non-empty AND whose
                # oldest entry arrived before the other (use 'timestamp' as proxy).
                # Since cycles strictly alternate and queues are FIFO, just pick
                # whichever line-queue has the oldest entry.
                if l1_queue and l2_queue:
                    # compare timestamps of oldest entries in each queue
                    ts1 = l1_queue[0].get('timestamp', pd.NaT)
                    ts2 = l2_queue[0].get('timestamp', pd.NaT)
                    use_l1 = (ts1 <= ts2)
                else:
                    use_l1 = bool(l1_queue)

                line_dict = l1_queue.popleft() if use_l1 else l2_queue.popleft()
                merged = {**fmt_queue.popleft(), **line_dict, **end_queue.popleft()}

                if use_l1:
                    l1_rows.append(merged)
                else:
                    l2_rows.append(merged)

        # ── single forward pass ───────────────────────────────────────────────────
        for _, row in df.iterrows():
            nmn   = row["new_metric_name"]   # e.g. 'l1_buffer_a__buffer_idle_time'
            val   = row["value"]
            ts    = row["timestamp"]
            bare  = self._bare(nmn)               # strips shuttle_station__ if present

            # ── (a) shuttle / system-count metrics → dict_fmt ─────────────────────
            if bare in _FMT_BARE_KEYS:
                if active_fmt[bare] == -1:          # only fill once per cycle
                    active_fmt[bare] = val
                if self._is_complete(active_fmt):
                    fmt_queue.append(active_fmt)
                    active_fmt = copy.deepcopy(DICT_FMT_TEMPLATE)
                continue

            # ── (b) end / summary metrics → end_dict ──────────────────────────────
            if nmn in _END_FULL_KEYS:
                if active_end[nmn] == -1:
                    active_end[nmn] = val
                if self._is_complete(active_end):
                    end_queue.append(active_end)
                    active_end = copy.deepcopy(END_DICT_TEMPLATE)
                    _try_flush()
                continue

            # ── (c) L1 line metrics ────────────────────────────────────────────────
            if nmn in _L1_FULL_KEYS:
                if active_l1[nmn] == -1:
                    active_l1[nmn] = val
                active_l1['timestamp'] = ts         # always update to latest ts
                if self._is_complete(active_l1):
                    l1_queue.append(active_l1)
                    active_l1 = copy.deepcopy(DICT_L1_TEMPLATE)
                    _try_flush()
                continue

            # ── (d) L2 line metrics ────────────────────────────────────────────────
            if nmn in _L2_FULL_KEYS:
                if active_l2[nmn] == -1:
                    active_l2[nmn] = val
                active_l2['timestamp'] = ts         # always update to latest ts
                if self._is_complete(active_l2):
                    l2_queue.append(active_l2)
                    active_l2 = copy.deepcopy(DICT_L2_TEMPLATE)
                    _try_flush()
                continue

        # ── build output DataFrames ───────────────────────────────────────────────
        l1_cols = (list(DICT_FMT_TEMPLATE.keys()) +
                list(DICT_L1_TEMPLATE.keys()) +
                list(END_DICT_TEMPLATE.keys()))
        l2_cols = (list(DICT_FMT_TEMPLATE.keys()) +
                list(DICT_L2_TEMPLATE.keys()) +
                list(END_DICT_TEMPLATE.keys()))

        df_l1 = pd.DataFrame(l1_rows, columns=l1_cols) if l1_rows else pd.DataFrame(columns=l1_cols)
        df_l2 = pd.DataFrame(l2_rows, columns=l2_cols) if l2_rows else pd.DataFrame(columns=l2_cols)

        # Replace any stray -1 sentinels (unfilled optional metrics) with NaN
        df_l1 = df_l1.replace(-1, pd.NA)
        df_l2 = df_l2.replace(-1, pd.NA)

        return df_l1[self.required_features]#, df_l2


    # --------------------------------------------------
    # windowing
    # --------------------------------------------------
    def fetch_next_window(self, curr_first_timestamp, for_training=False):
        # df = self._format()
        df = self.df
        # print(df)
        # print("===================================================================")
        if df.empty:
            return None

        # df = df.sort_values("timestamp").reset_index(drop=True)

        # find index
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
            y_start = end-1 # Minus coz it starts from zero index
            y_end   = y_start + self.prediction_window
            if y_end > len(df):
                return None
            y = df.iloc[y_start:y_end][[self.target_name]]
            return X, y

        return X


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import yaml
    import time

    logging.basicConfig(level=logging.INFO)

    # --------------------------------------------------
    # DB setup
    # --------------------------------------------------

    DATABASE_URL = "postgresql://postgres:myelin123@localhost:5432/glue-dispenser-db"  
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
    handler = DataHandler(
        session_factory=session_factory,
        config=test_config,
        target_name="system__cycle_time"
    )
    curr_first_timestamp = None
    while True:
        # start = time.time()
        # X = handler.fetch_next_window(curr_first_timestamp, for_training=False)
        # # print(X)
        # print("Time taken: ", time.time()-start)
        # print("===================================================================")
        # curr_first_timestamp = X.iloc[0]['timestamp']
        
        # if X is None:
        #     print("❌ Not enough data for window")
        #     break
        # else:
        #     print("✅ Window shape:", X.shape)
        #     print(X)

        # --------------------------------------------------
        # Training window test
        # --------------------------------------------------
        start = time.time()
        XY = handler.fetch_next_window(curr_first_timestamp=None, for_training=True)
        print("Time taken: ", time.time()-start)
        
        if XY is None:
            print("❌ Not enough data for training window")
        else:
            X_tr, y_tr = XY
            print("\nTraining X shape:", X_tr.shape)
            print("Training y shape:", y_tr.shape)
            print("\nX sample:")
            print(X_tr.head())
            print("\ny sample:")
            print(y_tr.head())
        print("===================================================================")
        curr_first_timestamp = XY[0].iloc[0]['timestamp']

    # print("\n✅ DataHandler verification complete")

