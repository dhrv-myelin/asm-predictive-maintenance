"""
data/data_loader.py
─────────────────────
Converts process_metrics (long format, TimescaleDB) into wide cycle-level
DataFrames ready for the Mamba model and anomaly inferencers.

Long format (what lives in the DB):
    timestamp | station_name | metric_name | value | unit | state_context

Wide format (what models consume):
    One row per manufacturing cycle, columns = wide metric names
    e.g.  system__cycle_time | shuttle_scan_success_time | l1_dispenser__... | timestamp

The pivot logic mirrors the log parser's dict-fill approach:
  - station_name + metric_name  → wide column name  (<station>__<metric>)
  - Special case: shuttle_station metrics are stored bare (no station prefix)
    in the wide format to match the Colab parser's DICT_FMT_TEMPLATE keys.
  - Rows are assembled per cycle_count (system__count acts as the cycle key).

Datasets produced:
  TrainingDataset   — X_train, y_train, X_val, y_val, feature_names, scaler
  InferenceDataset  — X (last N cycles), latest_value, timestamps
  DriftDataset      — reference_df (training window), current_df (recent window)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sqlalchemy import text
from sqlalchemy.orm import Session

from config.config_schema import MetricConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wide-column naming helpers
# (mirrors the log parser's naming so the same column names are used end-to-end)
# ---------------------------------------------------------------------------

# Shuttle station metrics are stored in the wide DF without the station prefix
# (matching DICT_FMT_TEMPLATE keys in the log parser)
_SHUTTLE_BARE_METRICS = {
    "shuttle_pre_scan_time",
    "shuttle_scan_success_time",
    "shuttle_post_scan_delay",
    "shuttle_carrier_motion_time",
    "shuttle_stopper_lowering_time",
    "shuttle_pallet_release_time",
    "shuttle_stopper_rising_time",
    "shuttle_idle_time",
}


def _wide_col(station_name: str, metric_name: str) -> str:
    """
    Build the wide-format column name for a long-format row.
    shuttle_station metrics are stored bare; everything else gets
    the <station>__<metric> prefix.
    """
    if station_name == "shuttle_station" and metric_name in _SHUTTLE_BARE_METRICS:
        return metric_name
    if station_name == "system":
        return f"system__{metric_name}"
    return f"{station_name}__{metric_name}"


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingDataset:
    X_train       : np.ndarray          # shape (N_train, seq_len, n_features)
    y_train       : np.ndarray          # shape (N_train,)
    X_val         : np.ndarray          # shape (N_val, seq_len, n_features)
    y_val         : np.ndarray          # shape (N_val,)
    feature_names : list[str]           # column names matching last axis of X
    target_col    : str
    scaler        : RobustScaler | MinMaxScaler | StandardScaler
    timestamps    : pd.DatetimeIndex    # one per cycle row (before sequencing)
    raw_df        : pd.DataFrame        # unsequenced wide df (for drift reference)


@dataclass
class InferenceDataset:
    X             : np.ndarray          # shape (1, seq_len, n_features) — last window
    feature_names : list[str]
    target_col    : str
    latest_value  : float               # most recent raw target value (unscaled)
    timestamps    : pd.DatetimeIndex    # timestamps of the seq_len cycles in X


@dataclass
class DriftDataset:
    reference_df  : pd.DataFrame        # wide df from training window
    current_df    : pd.DataFrame        # wide df from recent window
    feature_names : list[str]


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    All DB access for one station+metric is funnelled through this class.

    Parameters
    ----------
    session_factory : callable → context-manager yielding SQLAlchemy Session
    config          : validated MetricConfig for the target station+metric
    """

    def __init__(self, session_factory, config: MetricConfig) -> None:
        self._sf     = session_factory
        self._config = config

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def load_training_window(self) -> Optional[TrainingDataset]:
        """
        Pull the training lookback window from the DB, pivot to wide format,
        build sequences of shape (N, seq_len, n_features), split train/val.

        Returns None if there is insufficient data.
        """
        cfg   = self._config
        since = datetime.now(timezone.utc) - timedelta(days=cfg.data.training_lookback_days)
        wide  = self._fetch_wide(since=since)

        if wide is None or len(wide) < cfg.data.min_train_samples:
            logger.warning(
                "Insufficient training data for %s/%s: got %d rows, need %d",
                cfg.station, cfg.metric, len(wide) if wide is not None else 0,
                cfg.data.min_train_samples,
            )
            return None

        feature_cols = self._validate_features(wide)
        if feature_cols is None:
            return None

        wide_feat = wide[feature_cols].copy()
        target    = wide[cfg.target_column].values.astype(np.float32)

        # Fit scaler on training features
        scaler      = self._build_scaler()
        scaled_feat = scaler.fit_transform(wide_feat.values.astype(np.float32))

        # Scale target the same way as the target column position in features
        target_idx    = feature_cols.index(cfg.target_column)
        target_scaled = scaled_feat[:, target_idx]

        # Build sequences
        seq_len = cfg.mamba.seq_len
        X, y    = self._make_sequences(scaled_feat, target_scaled, seq_len)

        if len(X) < 4:
            logger.warning("Not enough sequences for %s/%s after windowing.", cfg.station, cfg.metric)
            return None

        # Train / val split (chronological — no shuffle)
        n_val   = max(1, int(len(X) * cfg.data.val_split))
        n_train = len(X) - n_val

        return TrainingDataset(
            X_train       = X[:n_train],
            y_train       = y[:n_train],
            X_val         = X[n_train:],
            y_val         = y[n_train:],
            feature_names = feature_cols,
            target_col    = cfg.target_column,
            scaler        = scaler,
            timestamps    = pd.DatetimeIndex(wide["timestamp"].values),
            raw_df        = wide,
        )

    def load_inference_window(
        self, scaler, feature_names: list[str]
    ) -> Optional[InferenceDataset]:
        """
        Pull the most recent N cycles (inference_lookback_cycles), apply the
        *already-fitted* scaler, return a single sequence for prediction.

        The scaler must be the one saved during training (loaded from MLflow
        artifact) to guarantee identical preprocessing.

        Returns None if insufficient data.
        """
        cfg   = self._config
        # Pull enough history to build at least one sequence
        n_cycles = max(cfg.data.inference_lookback_cycles, cfg.mamba.seq_len + 5)
        wide     = self._fetch_wide(limit_cycles=n_cycles)

        if wide is None or len(wide) < cfg.mamba.seq_len:
            logger.warning(
                "Insufficient inference data for %s/%s: got %d rows, need %d",
                cfg.station, cfg.metric,
                len(wide) if wide is not None else 0,
                cfg.mamba.seq_len,
            )
            return None

        # Align to the features the model was trained on
        missing = [f for f in feature_names if f not in wide.columns]
        if missing:
            logger.error("Inference: missing features for %s/%s: %s", cfg.station, cfg.metric, missing)
            return None

        wide_feat   = wide[feature_names].tail(cfg.mamba.seq_len + 1)
        latest_raw  = float(wide[cfg.target_column].iloc[-1])
        scaled_feat = scaler.transform(wide_feat.values.astype(np.float32))

        # Take the last seq_len rows as the input sequence
        X = scaled_feat[-cfg.mamba.seq_len:][np.newaxis, :, :]   # (1, seq_len, n_feats)

        return InferenceDataset(
            X             = X,
            feature_names = feature_names,
            target_col    = cfg.target_column,
            latest_value  = latest_raw,
            timestamps    = pd.DatetimeIndex(
                wide["timestamp"].tail(cfg.mamba.seq_len).values
            ),
        )

    def load_drift_window(self) -> Optional[DriftDataset]:
        """
        Load two windows for drift detection:
          reference : full training lookback (same as training window)
          current   : most recent inference_lookback_cycles cycles

        Returns None if either window has no data.
        """
        cfg        = self._config
        since_ref  = datetime.now(timezone.utc) - timedelta(days=cfg.data.training_lookback_days)

        ref_wide = self._fetch_wide(since=since_ref)
        cur_wide = self._fetch_wide(limit_cycles=cfg.data.inference_lookback_cycles)

        if ref_wide is None or cur_wide is None:
            return None

        feature_cols = self._validate_features(ref_wide)
        if feature_cols is None:
            return None

        return DriftDataset(
            reference_df  = ref_wide[feature_cols].dropna(),
            current_df    = cur_wide[feature_cols].dropna(),
            feature_names = feature_cols,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _fetch_wide(
        self,
        since         : Optional[datetime] = None,
        limit_cycles  : Optional[int]      = None,
    ) -> Optional[pd.DataFrame]:
        """
        Query process_metrics for all required columns, pivot long→wide,
        one row per manufacturing cycle (keyed on system__count).

        TimescaleDB optimisation: always filter by timestamp first so the
        hypertable chunk exclusion kicks in before the pivot.
        """
        cfg              = self._config
        required_stations = self._required_stations()

        # Build the time filter clause
        if since is not None:
            time_clause = f"AND timestamp >= '{since.isoformat()}'"
        elif limit_cycles is not None:
            # We don't know the exact time window for N cycles, so we pull
            # recent rows and trim after pivoting.
            # 10× margin: each cycle has many rows in long format
            time_clause = f"LIMIT {limit_cycles * 50}"
        else:
            time_clause = ""

        # Station filter for efficiency
        station_list = ", ".join(f"'{s}'" for s in required_stations)

        sql = f"""
            SELECT timestamp, station_name, metric_name, value
            FROM {cfg.db.schema_name}.{cfg.db.table}
            WHERE station_name IN ({station_list})
            {time_clause}
            ORDER BY timestamp ASC
        """

        try:
            with self._sf() as session:
                result = session.execute(text(sql))
                rows   = result.fetchall()
        except Exception as exc:
            logger.error("DB query failed for %s/%s: %s", cfg.station, cfg.metric, exc)
            return None

        if not rows:
            return None

        long_df = pd.DataFrame(rows, columns=["timestamp", "station_name", "metric_name", "value"])
        long_df["timestamp"] = pd.to_datetime(long_df["timestamp"])

        # Assign wide column names
        long_df["wide_col"] = long_df.apply(
            lambda r: _wide_col(r["station_name"], r["metric_name"]), axis=1
        )

        # Keep only columns we care about
        needed_wide_cols = set(cfg.required_features) | {cfg.target_column, "system__count"}
        long_df = long_df[long_df["wide_col"].isin(needed_wide_cols)]

        if long_df.empty:
            return None

        # Pivot: one row per cycle_count value
        # We use system__count as the cycle key — it increments by 1 each cycle.
        # Strategy: pivot by (timestamp, wide_col) then group by cycle boundaries.
        wide_df = self._pivot_to_cycles(long_df)

        if wide_df is None or len(wide_df) < 2:
            return None

        # Trim to limit_cycles if specified
        if limit_cycles is not None and len(wide_df) > limit_cycles:
            wide_df = wide_df.tail(limit_cycles).reset_index(drop=True)

        return wide_df

    def _pivot_to_cycles(self, long_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Pivot long_df into wide cycle rows.

        Approach:
          1. Sort by timestamp.
          2. Use system__count transitions to detect cycle boundaries.
          3. For each cycle, collect all wide_col → value pairs.
          4. Return DataFrame with one row per cycle.

        This mirrors the dict-fill logic of the log parser without
        requiring the full parser to be re-run at query time.
        """
        long_df = long_df.sort_values("timestamp").reset_index(drop=True)

        # Build a cycle_id column from system__count value changes
        count_rows = long_df[long_df["wide_col"] == "system__count"].copy()
        if count_rows.empty:
            # Fallback: bucket rows into pseudo-cycles by time gap
            return self._pivot_by_time_gap(long_df)

        # Map each row to the nearest preceding system__count timestamp
        count_times = count_rows["timestamp"].sort_values().values
        long_df["cycle_id"] = pd.cut(
            long_df["timestamp"].astype(np.int64),
            bins=np.append(np.int64(0), count_times.astype(np.int64)),
            labels=False,
            right=False,
        )
        long_df = long_df.dropna(subset=["cycle_id"])
        long_df["cycle_id"] = long_df["cycle_id"].astype(int)

        rows = []
        for cycle_id, grp in long_df.groupby("cycle_id", sort=True):
            row: dict = {}
            for _, r in grp.iterrows():
                col = r["wide_col"]
                if col not in row:                # first-write wins (matches parser)
                    row[col] = r["value"]
            row["timestamp"] = grp["timestamp"].max()
            rows.append(row)

        if not rows:
            return None

        return pd.DataFrame(rows)

    def _pivot_by_time_gap(self, long_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Fallback pivot when system__count is absent.
        Detects cycle boundaries by gaps > median_gap * 5.
        """
        long_df = long_df.sort_values("timestamp").reset_index(drop=True)
        diffs   = long_df["timestamp"].diff().dt.total_seconds().fillna(0)
        median  = diffs[diffs > 0].median()
        boundary = diffs > (median * 5)
        long_df["cycle_id"] = boundary.cumsum()

        rows = []
        for _, grp in long_df.groupby("cycle_id", sort=True):
            row: dict = {}
            for _, r in grp.iterrows():
                col = r["wide_col"]
                if col not in row:
                    row[col] = r["value"]
            row["timestamp"] = grp["timestamp"].max()
            rows.append(row)

        return pd.DataFrame(rows) if rows else None

    def _required_stations(self) -> set[str]:
        """
        Derive the set of DB station_name values needed to materialise
        all required_features.
        """
        stations = {"system"}  # always need system__count for cycle keying
        for col in self._config.required_features:
            if col.startswith("l1_") or col.startswith("l2_"):
                # e.g. l1_dispenser__dispenser_dispensing_time → l1_dispenser
                stations.add(col.split("__")[0])
            elif col.startswith("shuttle_"):
                stations.add("shuttle_station")
            elif col.startswith("system__"):
                stations.add("system")
        return stations

    def _validate_features(self, wide_df: pd.DataFrame) -> Optional[list[str]]:
        """
        Check that all required_features are present in wide_df.
        Returns the ordered list of feature column names, or None if
        critical columns are missing.
        """
        cfg     = self._config
        present = set(wide_df.columns)
        needed  = set(cfg.required_features)

        missing = needed - present
        if missing:
            logger.warning(
                "Missing features for %s/%s: %s. Available: %s",
                cfg.station, cfg.metric, missing, present,
            )
            # Return only what's available — partial feature sets are allowed
            available = [f for f in cfg.required_features if f in present]
            if cfg.target_column not in present:
                logger.error(
                    "Target column '%s' missing for %s/%s — cannot proceed.",
                    cfg.target_column, cfg.station, cfg.metric,
                )
                return None
            return available

        # Ensure target is first so index 0 = target in scaler
        ordered = [cfg.target_column] + [
            f for f in cfg.required_features if f != cfg.target_column
        ]
        return [f for f in ordered if f in present]

    def _build_scaler(self):
        scaler_type = self._config.data.scaler
        if scaler_type == "robust":
            return RobustScaler()
        if scaler_type == "minmax":
            return MinMaxScaler()
        return StandardScaler()

    @staticmethod
    def _make_sequences(
        X       : np.ndarray,
        y       : np.ndarray,
        seq_len : int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Slide a window of length seq_len over X to produce input sequences.
        Target y[i] = value at position i + seq_len (next-step prediction).

        Returns
        -------
        X_seq : (N - seq_len, seq_len, n_features)
        y_seq : (N - seq_len,)
        """
        X_seqs, y_seqs = [], []
        for i in range(len(X) - seq_len):
            X_seqs.append(X[i: i + seq_len])
            y_seqs.append(y[i + seq_len])
        if not X_seqs:
            return np.empty((0, seq_len, X.shape[1])), np.empty(0)
        return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)
