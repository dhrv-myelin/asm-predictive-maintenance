import sys
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

# ──────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────


def load_metric_config(config_path: str) -> dict:
    """
    Load per-metric thresholds from the JSON produced by diagnostic.py.
    Falls back to hard-coded defaults if the file is missing.
    """
    path = Path(config_path)
    if not path.exists():
        warnings.warn(
            f"Config file '{config_path}' not found — using hard-coded defaults. "
            "Run diagnostic.py first to generate per-metric thresholds.",
            UserWarning,
        )
        return _hardcoded_defaults()

    with open(path) as f:
        cfg = json.load(f)

    print(f"Loaded metric config from: {config_path}  ({len(cfg.get('metrics', {}))} metrics)")
    return cfg


def _hardcoded_defaults() -> dict:
    """Fallback config used when no JSON config file exists."""
    defaults = {
        "spike_z": 8.0,
        "step_sigma_mult": 2.0,
        "stability_ratio": 0.6,
        "abrupt_mult": 1.5,
        "drift_thresh": 1e-4,
        "variance_mult": 3.0,
        "variance_min_frac": 0.10,
        "baseline_shift_sigma": 2.5,
        "outlier_freq_mult": 2.0,
        "outlier_min_count": 3,
        "osc_type": "none",
        "osc_thresh": 0.0,
    }
    return {"metrics": {}, "osc_window": 50, "osc_step": 5, "osc_lag": 3, "defaults": defaults}


def _get(cfg: dict, metric: str, key: str):
    """Return per-metric threshold if available, else global default."""
    return cfg["metrics"].get(metric, cfg["defaults"]).get(key, cfg["defaults"][key])


# ──────────────────────────────────────────────
# Fixed params (not per-metric)
# ──────────────────────────────────────────────
STEP_WINDOW = 10
BASELINE_REF_FRAC = 0.20

ALLOWED_METRICS = {
    "entry_stopper_lowering_time",
    "entry_stopper_raising_time",
    "pallet_clamping_time",
    "pallet_lifting_time",
    "inspection_time",
    "pallet_unclamping_time",
    "pallet_lowering_time",
    "exit_stopper_lowering_time",
    "exit_stopper_raising_time",  
    "pallet_moveout_time",
    "cavity_1_dispensing_time",
    "cavity_2_dispensing_time",
    "cavity_3_dispensing_time",
    "cavity_4_dispensing_time",
    "cavity_5_dispensing_time",
    "cavity_6_dispensing_time",
}

def load_baseline(baseline_csv_path: str) -> dict:
    df = pd.read_csv(baseline_csv_path)
    required = {"metric_name", "mean", "std_dev"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Baseline CSV missing columns: {missing_cols}. Found: {list(df.columns)}"
        )
    baseline = {}
    for _, row in df.iterrows():
        name = str(row["metric_name"]).strip()
        mean = float(row["mean"])    if pd.notna(row["mean"])    else None
        std  = float(row["std_dev"]) if pd.notna(row["std_dev"]) else None
        if mean is None or std is None:
            warnings.warn(f"Null mean/std for '{name}' — skipping.", UserWarning)
            continue
        baseline[name] = (mean, max(std, 1e-6))

    for metric in sorted(ALLOWED_METRICS):
        if metric not in baseline:
            warnings.warn(
                f"No baseline entry for '{metric}' — will use local fallback.", UserWarning
            )
    return baseline


def _resolve_baseline(metric_name, day_df, global_baseline):
    if global_baseline is not None and metric_name in global_baseline:
        return global_baseline[metric_name]
    n_ref = max(int(len(day_df) * BASELINE_REF_FRAC), STEP_WINDOW * 2)
    ref   = day_df["value"].iloc[:n_ref]
    return ref.mean(), (ref.std() or 1e-6)


# ──────────────────────────────────────────────
# Data loading & feature engineering
# ──────────────────────────────────────────────


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["roll_mean"] = df["value"].rolling(10, min_periods=1).mean()
    df["roll_std"]  = df["value"].rolling(10, min_periods=1).std().fillna(0)
    df["diff"]      = df["value"].diff().fillna(0)
    return df


def _local_baseline(df: pd.DataFrame):
    """Kept for backward-compatibility (used by plot_patterns.py)."""
    n_ref = max(int(len(df) * BASELINE_REF_FRAC), STEP_WINDOW * 2)
    ref   = df["value"].iloc[:n_ref]
    return ref.mean(), (ref.std() or 1e-6)


# ──────────────────────────────────────────────
# Shared persistence gate
# ──────────────────────────────────────────────


def _find_persistent_events(
    condition: pd.Series,
    timestamps: pd.Series,
    min_readings: int,
) -> list[tuple]:
    """
    Return (start_ts, end_ts) spans where `condition` is True for at least
    `min_readings` consecutive readings.

    This is the single persistence gate shared by baseline-shift and
    variance-growth.  min_readings comes from diagnostic.py (stored in
    metric_config.json as "anomaly_min_readings") and is calibrated to
    approximately 15 minutes of data at the observed reading rate, so no
    time-based hardcoding is needed here.
    """
    events      = []
    in_run      = False
    run_start_i = None
    run_count   = 0

    for i, val in enumerate(condition):
        if val:
            if not in_run:
                in_run      = True
                run_start_i = i
                run_count   = 1
            else:
                run_count += 1
        else:
            if in_run:
                if run_count >= min_readings:
                    events.append((
                        timestamps.iloc[run_start_i],
                        timestamps.iloc[i - 1],
                    ))
                in_run    = False
                run_count = 0

    if in_run and run_count >= min_readings:
        events.append((
            timestamps.iloc[run_start_i],
            timestamps.iloc[-1],
        ))

    return events


# ══════════════════════════════════════════════
# Pattern detectors
# ══════════════════════════════════════════════


def detect_random_spikes(df, bm, bs, cfg, metric):
    """
    Single readings whose |z-score vs baseline| exceeds spike_z.

    spike_z is calibrated by diagnostic.py to the p99.9 z-score across all
    training days, so only readings far outside what baseline.csv says is
    normal are flagged.
    """
    spike_z = _get(cfg, metric, "spike_z")
    z = (df["value"] - bm).abs() / bs
    return [(t, t) for t in df[z > spike_z]["timestamp"]]


def detect_step_jumps(df, bm, bs, cfg, metric):
    """
    Abrupt, permanent level change.  The window before and after a reading
    must both be stable (std < stability_ratio * bs) but differ by more than
    step_sigma_mult * bs.  bs comes from baseline.csv.
    """
    if len(df) < 2 * STEP_WINDOW + 1:
        return []
    sigma_mult      = _get(cfg, metric, "step_sigma_mult")
    stability_ratio = _get(cfg, metric, "stability_ratio")
    abrupt_mult     = _get(cfg, metric, "abrupt_mult")

    step_indices = []
    for i in range(STEP_WINDOW, len(df) - STEP_WINDOW):
        before = df["value"].iloc[i - STEP_WINDOW : i]
        after  = df["value"].iloc[i : i + STEP_WINDOW]
        if (
            abs(after.mean() - before.mean()) > sigma_mult * bs
            and before.std() < stability_ratio * bs
            and after.std()  < stability_ratio * 1.5 * bs   # relaxed — post-step window is noisier
            # abrupt_mult check removed — the mean-difference check is sufficient
        ):
            step_indices.append(i)

    filtered, last = [], -STEP_WINDOW
    for idx in step_indices:
        if idx - last > STEP_WINDOW:
            filtered.append(idx)
            last = idx
    return [(df["timestamp"].iloc[i], df["timestamp"].iloc[i]) for i in filtered]

def detect_variance_growth(df, bm, bs, cfg, metric):
    var_mult     = _get(cfg, metric, "variance_mult")
    min_readings = int(_get(cfg, metric, "anomaly_min_readings"))
    win          = int(_get(cfg, metric, "detector_window"))
    spike_z      = _get(cfg, metric, "spike_z")

    # Spike-excluded rolling stats
    z     = (df["value"] - bm).abs() / bs
    clean = df["value"].where(z <= spike_z)

    half = max(win // 2, 5)
    roll_std  = clean.rolling(win, min_periods=half).std().ffill().fillna(bs)
    roll_mean = clean.rolling(win, min_periods=half).mean().ffill().fillna(bm)

    # Use a SHORT window to detect if the mean is actively transitioning
    short_win  = max(win // 6, 5)
    roll_mean_short = clean.rolling(short_win, min_periods=max(short_win//2,3)).mean().ffill().fillna(bm)
    roll_mean_prev  = roll_mean_short.shift(short_win)

    # How much is the short mean moving right now?
    mean_velocity = (roll_mean_short - roll_mean_prev).abs()

    threshold = var_mult * bs

    # Guard 1: mean must be near baseline (not a level shift)
    mean_near_baseline = (roll_mean - bm).abs() <= 2.0 * bs

    # Guard 2: mean must NOT be actively transitioning
    # If mean_velocity > 0.5*bs, the window is straddling a transition edge
    not_transitioning = mean_velocity <= 0.5 * bs

    condition = (roll_std > threshold) & mean_near_baseline & not_transitioning

    return _find_persistent_events(condition, df["timestamp"], min_readings)

def detect_slow_drift(df, bm, bs, cfg, metric):
    drift_thresh = _get(cfg, metric, "drift_thresh")
    win          = int(_get(cfg, metric, "detector_window"))
    min_readings = int(_get(cfg, metric, "anomaly_min_readings"))
    spike_z      = _get(cfg, metric, "spike_z")

    # Mask spikes with a tighter threshold — periodic pulses 
    # at 3σ corrupt the slope even if spike_z is 8+
    mask_z = min(spike_z, 3.0)
    z      = (df["value"] - bm).abs() / bs
    clean  = df["value"].where(z <= mask_z)   # NaN out pulses

    drifting = pd.Series(False, index=df.index)

    for i in range(win, len(df)):
        window_clean = clean.iloc[i - win: i].dropna()

        # Need at least half the window to be non-spike readings
        if len(window_clean) < win // 2:
            continue

        x = np.arange(len(window_clean)).reshape(-1, 1)
        slope = LinearRegression().fit(
            x, window_clean.values.reshape(-1, 1)
        ).coef_[0][0]

        # Normalised: total drift in σ over full window
        normalised = abs(slope) * win / bs

        # Extra guard: the window mean (spike-excluded) must 
        # itself be moving — not just jitter around baseline
        window_mean = window_clean.mean()
        mean_moving = abs(window_mean - bm) > 0.5 * bs

        if normalised > drift_thresh and mean_moving:
            drifting.iloc[i] = True

    return _find_persistent_events(drifting, df["timestamp"], min_readings)


def detect_trend_acceleration(df, bm, bs, cfg, metric):
    accel_thresh = _get(cfg, metric, "drift_thresh") * 0.01
    coeffs = np.polyfit(np.arange(len(df)), df["value"].values, 2)
    return (
        [(df["timestamp"].iloc[0], df["timestamp"].iloc[-1])]
        if abs(coeffs[0]) > accel_thresh else []
    )

def _spike_cleaned_rolling(series, bm, bs, spike_z_thresh, win):
    """
    Compute rolling mean and std with spike readings replaced by NaN
    before the rolling operation, so spikes don't poison subsequent windows.
    """
    z = (series - bm).abs() / bs
    clean = series.where(z <= spike_z_thresh)   # NaN out spikes
    
    half = max(win // 2, 3)
    roll_mean = clean.rolling(win, min_periods=half).mean()
    roll_std  = clean.rolling(win, min_periods=half).std()
    
    # Forward-fill so we don't get NaN gaps from isolated spike removals
    roll_mean = roll_mean.fillna(method='ffill').fillna(bm)
    roll_std  = roll_std.fillna(method='ffill').fillna(bs)
    
    return roll_mean, roll_std
def detect_baseline_shift(df, bm, bs, cfg, metric):
    shift_sigma       = _get(cfg, metric, "baseline_shift_sigma")
    shift_sigma_short = _get(cfg, metric, "baseline_shift_sigma_short")
    min_readings      = int(_get(cfg, metric, "anomaly_min_readings"))
    win_long          = int(_get(cfg, metric, "detector_window"))
    win_short         = int(_get(cfg, metric, "short_window"))

    win_short = max(min(win_short, win_long // 2), 5)
    half_long  = max(win_long  // 2, 5)
    half_short = max(win_short // 2, 3)

    spike_z = _get(cfg, metric, "spike_z")
    z       = (df["value"] - bm).abs() / bs
    clean   = df["value"].where(z <= spike_z)

    roll_mean_long  = clean.rolling(win_long,  min_periods=half_long ).mean().ffill().fillna(bm)
    roll_mean_short = clean.rolling(win_short, min_periods=half_short).mean().ffill().fillna(bm)
    roll_std_short  = clean.rolling(win_short, min_periods=half_short).std().ffill().fillna(bs)

    threshold_short = shift_sigma_short * bs
    threshold_long  = shift_sigma       * bs

    deviation_short = (roll_mean_short - bm).abs()
    deviation_long  = (roll_mean_long  - bm).abs()

    # Condition 1: short mean has moved far enough from baseline
    mean_shifted_short = deviation_short > threshold_short

    # Condition 2: short window is stable (not mid-transition noise)
    # 1.5x is generous — during a V-dip the short window std is elevated
    is_stable_short = roll_std_short < deviation_short * 1.5

    # Condition 3: long mean confirms same direction or is on its way
    same_direction       = ((roll_mean_short - bm) * (roll_mean_long - bm)) > 0
    long_partially_agree = deviation_long > threshold_long * 0.5
    direction_confirmed  = same_direction | long_partially_agree

    condition = mean_shifted_short & is_stable_short & direction_confirmed

    return _find_persistent_events(condition, df["timestamp"], min_readings)
def detect_increasing_outlier_frequency(df, bm, bs, cfg, metric):
    """
    Second half of the day has significantly more spikes than the first half.
    spike_z and outlier_freq_mult both come from diagnostic.py / baseline.csv.
    """
    spike_z   = _get(cfg, metric, "spike_z")
    freq_mult = _get(cfg, metric, "outlier_freq_mult")
    min_count = _get(cfg, metric, "outlier_min_count")
    z   = (df["value"] - bm).abs() / bs
    mid = len(z) // 2
    early = (z.iloc[:mid] > spike_z).sum()
    late  = (z.iloc[mid:] > spike_z).sum()
    return (
        [(df["timestamp"].iloc[mid], df["timestamp"].iloc[-1])]
        if late > freq_mult * early and late > min_count else []
    )


def detect_oscillation_loss(df, bm, bs, cfg, metric):
    osc_thresh = _get(cfg, metric, "osc_thresh")
    osc_window = cfg.get("osc_window", 50)
    osc_step   = cfg.get("osc_step", 5)
    osc_lag    = cfg.get("osc_lag", 3)

    events, in_loss, start_ts = [], False, None
    for i in range(0, len(df) - osc_window + 1, osc_step):
        window   = df["value"].iloc[i : i + osc_window]
        autocorr = window.autocorr(lag=osc_lag)
        osc_ok   = autocorr is not None and autocorr < osc_thresh
        if not osc_ok and not in_loss:
            in_loss, start_ts = True, df["timestamp"].iloc[i]
        elif osc_ok and in_loss:
            events.append((start_ts, df["timestamp"].iloc[i]))
            in_loss = False
    if in_loss:
        events.append((start_ts, df["timestamp"].iloc[-1]))
    return events


def detect_periodic_oscillation(df, bm, bs, cfg, metric):
    osc_thresh = _get(cfg, metric, "osc_thresh")
    osc_window = cfg.get("osc_window", 50)
    osc_step   = cfg.get("osc_step", 5)
    osc_lag    = cfg.get("osc_lag", 3)

    events, in_osc, start_ts = [], False, None
    for i in range(0, len(df) - osc_window + 1, osc_step):
        window   = df["value"].iloc[i : i + osc_window]
        autocorr = window.autocorr(lag=osc_lag)
        osc_ok   = autocorr is not None and abs(autocorr) > osc_thresh
        if osc_ok and not in_osc:
            in_osc, start_ts = True, df["timestamp"].iloc[i]
        elif not osc_ok and in_osc:
            events.append((start_ts, df["timestamp"].iloc[i]))
            in_osc = False
    if in_osc:
        events.append((start_ts, df["timestamp"].iloc[-1]))
    return events


# ──────────────────────────────────────────────
# Per-metric detector routing
# ──────────────────────────────────────────────

_BASE_DETECTORS = [
    ("Random spikes",                detect_random_spikes),
    ("Step jumps",                   detect_step_jumps),
    ("Slow drift",                   detect_slow_drift),
    ("Variance growth",              detect_variance_growth),
    ("Trend acceleration",           detect_trend_acceleration),
    ("Baseline shift",               detect_baseline_shift),
    ("Increasing outlier frequency", detect_increasing_outlier_frequency),
]


def get_detectors(metric, cfg):
    detectors = list(_BASE_DETECTORS)
    osc_type  = _get(cfg, metric, "osc_type")
    if osc_type == "oscillation_loss":
        detectors.append(("Oscillation loss", detect_oscillation_loss))
    elif osc_type == "periodic_oscillation":
        detectors.append(("Periodic oscillation", detect_periodic_oscillation))
    return detectors


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────


def run_pattern_pipeline(df, global_baseline=None, metric_cfg=None):
    if metric_cfg is None:
        metric_cfg = _hardcoded_defaults()

    df = df.copy()
    df["timestamp"]       = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["new_metric_name"] = df["station_name"] + "__" + df["metric_name"]
    df["date"]            = df["timestamp"].dt.floor("D")
    df = df[df["metric_name"].isin(ALLOWED_METRICS)]

    rows = []

    for combo_key in df["new_metric_name"].unique():
        station_name, metric_name = combo_key.split("__", 1)

        for _, day_df in df[df["new_metric_name"] == combo_key].groupby("date"):
            day_df = day_df.sort_values("timestamp").reset_index(drop=True)
            if len(day_df) < 2 * STEP_WINDOW + 1:
                continue

            day_df    = extract_features(day_df)
            bm, bs    = _resolve_baseline(metric_name, day_df, global_baseline)
            detectors = get_detectors(metric_name, metric_cfg)

            for pattern_name, detector_fn in detectors:
                for start_ts, end_ts in detector_fn(day_df, bm, bs, metric_cfg, metric_name):
                    rows.append({
                        "actual_timestamp":    start_ts,
                        "predicted_timestamp": end_ts,
                        "predicted_value":     1.0,
                        "station_name":        station_name,
                        "metric_name":         metric_name,
                        "model_name":          f"stats_pattern_detector::{pattern_name}",
                    })

    if not rows:
        return pd.DataFrame(columns=[
            "actual_timestamp", "predicted_timestamp", "predicted_value",
            "station_name", "metric_name", "model_name",
        ])

    result = pd.DataFrame(rows)
    result["actual_timestamp"]    = pd.to_datetime(result["actual_timestamp"],    utc=True)
    result["predicted_timestamp"] = pd.to_datetime(result["predicted_timestamp"], utc=True)
    result["predicted_value"]     = result["predicted_value"].astype(float)
    return result.reset_index(drop=True)


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────


def print_summary(out):
    if out.empty:
        print("No patterns detected.")
        return out

    out = out.copy()
    out["pattern"] = out["model_name"].str.split("::").str[1]

    print("\n" + "=" * 60)
    print("  PATTERN DETECTION SUMMARY")
    print("=" * 60)

    print("\n By Pattern Type:")
    for pattern, count in out.groupby("pattern").size().sort_values(ascending=False).items():
        print(f"   {pattern:<40s} {count:>4d} window(s)")

    print("\n By Station:")
    for station, count in out.groupby("station_name").size().sort_values(ascending=False).items():
        print(f"   {station:<40s} {count:>4d} window(s)")

    print("\n Detail (Station -> Metric -> Pattern):")
    for station, s_df in out.groupby("station_name"):
        print(f"\n   [{station}]")
        for metric, m_df in s_df.groupby("metric_name"):
            print(f"     {metric}")
            for pattern, p_df in m_df.groupby("pattern"):
                print(f"       -- {pattern:<38s} {len(p_df):>3d} window(s)")

    print("\n" + "=" * 60)
    print(
        f"  TOTAL: {len(out)} pattern window(s) across "
        f"{out['station_name'].nunique()} station(s), "
        f"{out['metric_name'].nunique()} metric(s)"
    )
    print("=" * 60)
    return out

