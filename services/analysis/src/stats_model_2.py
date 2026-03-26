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

# ──────────────────────────────────────────────
# Baseline CSV loading
# ──────────────────────────────────────────────


def load_baseline(baseline_csv_path: str) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(baseline_csv_path)
    required = {"metric_name", "mean", "std_dev"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Baseline CSV missing columns: {missing_cols}. Found: {list(df.columns)}")

    baseline: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        name = str(row["metric_name"]).strip()
        mean = float(row["mean"]) if pd.notna(row["mean"]) else None
        std = float(row["std_dev"]) if pd.notna(row["std_dev"]) else None
        if mean is None or std is None:
            warnings.warn(f"Null mean/std for '{name}' — skipping.", UserWarning)
            continue
        baseline[name] = (mean, max(std, 1e-6))

    for metric in sorted(ALLOWED_METRICS):
        if metric not in baseline:
            warnings.warn(
                f"No baseline entry for '{metric}' — will use local fallback.",
                UserWarning,
            )
    return baseline


def _resolve_baseline(
    metric_name: str, day_df: pd.DataFrame, global_baseline: dict | None
) -> tuple[float, float]:
    if global_baseline is not None and metric_name in global_baseline:
        return global_baseline[metric_name]
    n_ref = max(int(len(day_df) * BASELINE_REF_FRAC), STEP_WINDOW * 2)
    ref = day_df["value"].iloc[:n_ref]
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
    df["roll_std"] = df["value"].rolling(10, min_periods=1).std().fillna(0)
    df["diff"] = df["value"].diff().fillna(0)
    return df


def _local_baseline(df: pd.DataFrame) -> tuple[float, float]:
    """Kept for backward-compatibility (used by plot_patterns.py)."""
    n_ref = max(int(len(df) * BASELINE_REF_FRAC), STEP_WINDOW * 2)
    ref = df["value"].iloc[:n_ref]
    return ref.mean(), (ref.std() or 1e-6)


# ══════════════════════════════════════════════
# Pattern detectors  (all accept cfg + metric)
# ══════════════════════════════════════════════


def detect_random_spikes(
    df: pd.DataFrame, bm: float, bs: float, cfg: dict, metric: str
) -> list[tuple]:
    spike_z = _get(cfg, metric, "spike_z")
    z = (df["value"] - bm).abs() / bs
    return [(t, t) for t in df[z > spike_z]["timestamp"]]


def detect_step_jumps(
    df: pd.DataFrame, bm: float, bs: float, cfg: dict, metric: str
) -> list[tuple]:
    if len(df) < 2 * STEP_WINDOW + 1:
        return []
    sigma_mult = _get(cfg, metric, "step_sigma_mult")
    stability_ratio = _get(cfg, metric, "stability_ratio")
    abrupt_mult = _get(cfg, metric, "abrupt_mult")

    step_indices = []
    for i in range(STEP_WINDOW, len(df) - STEP_WINDOW):
        before = df["value"].iloc[i - STEP_WINDOW : i]
        after = df["value"].iloc[i : i + STEP_WINDOW]
        if (
            abs(after.mean() - before.mean()) > sigma_mult * bs
            and before.std() < stability_ratio * bs
            and after.std() < stability_ratio * bs
            and abs(df["value"].iloc[i] - df["value"].iloc[i - 1]) > abrupt_mult * bs
        ):
            step_indices.append(i)
    filtered, last = [], -STEP_WINDOW
    for idx in step_indices:
        if idx - last > STEP_WINDOW:
            filtered.append(idx)
            last = idx
    return [(df["timestamp"].iloc[i], df["timestamp"].iloc[i]) for i in filtered]


def detect_slow_drift(
    df: pd.DataFrame, bm: float, bs: float, cfg: dict, metric: str
) -> list[tuple]:
    drift_thresh = _get(cfg, metric, "drift_thresh")
    x = np.arange(len(df)).reshape(-1, 1)
    y = df["value"].values.reshape(-1, 1)
    slope = LinearRegression().fit(x, y).coef_[0][0]
    return (
        [(df["timestamp"].iloc[0], df["timestamp"].iloc[-1])] if abs(slope) > drift_thresh else []
    )


def detect_variance_growth(
    df: pd.DataFrame, bm: float, bs: float, cfg: dict, metric: str
) -> list[tuple]:
    if "roll_std" not in df.columns:
        return []
    var_mult = _get(cfg, metric, "variance_mult")
    min_frac = _get(cfg, metric, "variance_min_frac")
    threshold = var_mult * bs
    elevated = df["roll_std"] > threshold

    if elevated.mean() < min_frac:
        return []

    events, in_spike, start_ts = [], False, None
    for _, row in df.iterrows():
        if row["roll_std"] > threshold and not in_spike:
            in_spike, start_ts = True, row["timestamp"]
        elif row["roll_std"] <= threshold and in_spike:
            events.append((start_ts, row["timestamp"]))
            in_spike = False
    if in_spike:
        events.append((start_ts, df["timestamp"].iloc[-1]))
    return events


def detect_trend_acceleration(
    df: pd.DataFrame, bm: float, bs: float, cfg: dict, metric: str
) -> list[tuple]:
    # uses drift_thresh as proxy for acceleration sensitivity
    accel_thresh = _get(cfg, metric, "drift_thresh") * 0.01
    coeffs = np.polyfit(np.arange(len(df)), df["value"].values, 2)
    return (
        [(df["timestamp"].iloc[0], df["timestamp"].iloc[-1])]
        if abs(coeffs[0]) > accel_thresh
        else []
    )


def detect_baseline_shift(
    df: pd.DataFrame, bm: float, bs: float, cfg: dict, metric: str
) -> list[tuple]:
    if "roll_mean" not in df.columns:
        return []
    shift_sigma = _get(cfg, metric, "baseline_shift_sigma")
    if abs(df["roll_mean"].iloc[-1] - bm) > shift_sigma * bs:
        crossings = df[np.abs(df["roll_mean"] - bm) > shift_sigma * bs]
        start_ts = crossings["timestamp"].iloc[0] if len(crossings) else df["timestamp"].iloc[-1]
        return [(start_ts, df["timestamp"].iloc[-1])]
    return []


def detect_increasing_outlier_frequency(
    df: pd.DataFrame, bm: float, bs: float, cfg: dict, metric: str
) -> list[tuple]:
    spike_z = _get(cfg, metric, "spike_z")
    freq_mult = _get(cfg, metric, "outlier_freq_mult")
    min_count = _get(cfg, metric, "outlier_min_count")
    z = (df["value"] - bm).abs() / bs
    mid = len(z) // 2
    early = (z.iloc[:mid] > spike_z).sum()
    late = (z.iloc[mid:] > spike_z).sum()
    return (
        [(df["timestamp"].iloc[mid], df["timestamp"].iloc[-1])]
        if late > freq_mult * early and late > min_count
        else []
    )


def detect_oscillation_loss(
    df: pd.DataFrame, bm: float, bs: float, cfg: dict, metric: str
) -> list[tuple]:
    """Flag windows where expected oscillation is ABSENT."""
    osc_thresh = _get(cfg, metric, "osc_thresh")
    osc_window = cfg.get("osc_window", 50)
    osc_step = cfg.get("osc_step", 5)
    osc_lag = cfg.get("osc_lag", 3)

    events, in_loss, start_ts = [], False, None
    for i in range(0, len(df) - osc_window + 1, osc_step):
        window = df["value"].iloc[i : i + osc_window]
        autocorr = window.autocorr(lag=osc_lag)
        osc_ok = autocorr is not None and autocorr < osc_thresh

        if not osc_ok and not in_loss:
            in_loss, start_ts = True, df["timestamp"].iloc[i]
        elif osc_ok and in_loss:
            events.append((start_ts, df["timestamp"].iloc[i]))
            in_loss = False

    if in_loss:
        events.append((start_ts, df["timestamp"].iloc[-1]))
    return events


def detect_periodic_oscillation(
    df: pd.DataFrame, bm: float, bs: float, cfg: dict, metric: str
) -> list[tuple]:
    """Flag windows where oscillation IS present (anomalous for this metric)."""
    osc_thresh = _get(cfg, metric, "osc_thresh")
    osc_window = cfg.get("osc_window", 50)
    osc_step = cfg.get("osc_step", 5)
    osc_lag = cfg.get("osc_lag", 3)

    events, in_osc, start_ts = [], False, None
    for i in range(0, len(df) - osc_window + 1, osc_step):
        window = df["value"].iloc[i : i + osc_window]
        autocorr = window.autocorr(lag=osc_lag)
        osc_ok = autocorr is not None and abs(autocorr) > osc_thresh

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

# Base detectors run on every metric
_BASE_DETECTORS = [
    ("Random spikes", detect_random_spikes),
    ("Step jumps", detect_step_jumps),
    # ("Slow drift", detect_slow_drift),
    # ("Variance growth", detect_variance_growth),
    ("Trend acceleration", detect_trend_acceleration),
    # ("Baseline shift", detect_baseline_shift),
    ("Increasing outlier frequency", detect_increasing_outlier_frequency),
]


def get_detectors(metric: str, cfg: dict) -> list[tuple[str, callable]]:
    """Return the right detector list for this metric based on config."""
    detectors = list(_BASE_DETECTORS)
    osc_type = _get(cfg, metric, "osc_type")

    if osc_type == "oscillation_loss":
        detectors.append(("Oscillation loss", detect_oscillation_loss))
    elif osc_type == "periodic_oscillation":
        detectors.append(("Periodic oscillation", detect_periodic_oscillation))
    # osc_type == "none" → no oscillation detector added

    return detectors


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────


def run_pattern_pipeline(
    df: pd.DataFrame,
    global_baseline: dict[str, tuple[float, float]] | None = None,
    metric_cfg: dict | None = None,
) -> pd.DataFrame:
    if metric_cfg is None:
        metric_cfg = _hardcoded_defaults()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    df["new_metric_name"] = df["station_name"] + "__" + df["metric_name"]
    df["date"] = df["timestamp"].dt.floor("D")
    df = df[df["metric_name"].isin(ALLOWED_METRICS)]

    rows: list[dict] = []

    for combo_key in df["new_metric_name"].unique():
        station_name, metric_name = combo_key.split("__", 1)

        for _, day_df in df[df["new_metric_name"] == combo_key].groupby("date"):
            day_df = day_df.sort_values("timestamp").reset_index(drop=True)
            if len(day_df) < 2 * STEP_WINDOW + 1:
                continue

            day_df = extract_features(day_df)
            bm, bs = _resolve_baseline(metric_name, day_df, global_baseline)
            detectors = get_detectors(metric_name, metric_cfg)

            for pattern_name, detector_fn in detectors:
                for start_ts, end_ts in detector_fn(day_df, bm, bs, metric_cfg, metric_name):
                    rows.append(
                        {
                            "actual_timestamp": start_ts,
                            "predicted_timestamp": end_ts,
                            "predicted_value": 1.0,
                            "station_name": station_name,
                            "metric_name": metric_name,
                            "model_name": f"stats_pattern_detector::{pattern_name}",
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "actual_timestamp",
                "predicted_timestamp",
                "predicted_value",
                "station_name",
                "metric_name",
                "model_name",
            ]
        )

    result = pd.DataFrame(rows)
    result["actual_timestamp"] = pd.to_datetime(result["actual_timestamp"], utc=True)
    result["predicted_timestamp"] = pd.to_datetime(result["predicted_timestamp"], utc=True)
    result["predicted_value"] = result["predicted_value"].astype(float)
    return result.reset_index(drop=True)


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────


def print_summary(out: pd.DataFrame) -> pd.DataFrame:
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


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "train.csv"
    baseline_path = sys.argv[2] if len(sys.argv) > 2 else None
    config_path = sys.argv[3] if len(sys.argv) > 3 else "metric_config.json"

    df = load_data(csv_path)

    global_baseline = None
    if baseline_path:
        print(f"Loading baseline from: {baseline_path}")
        global_baseline = load_baseline(baseline_path)

    metric_cfg = load_metric_config(config_path)

    out = run_pattern_pipeline(df, global_baseline=global_baseline, metric_cfg=metric_cfg)
    out = print_summary(out)

    output_path = Path(csv_path).stem + "_patterns.csv"
    out.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
