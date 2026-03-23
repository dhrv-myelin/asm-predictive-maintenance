"""
diagnostic.py
─────────────
Analyses all metrics and saves per-metric thresholds to a JSON config file.
That config is then consumed by stats_model.py and plot_patterns.py.

Usage
─────
    python3 diagnostic.py <process_metrics.csv> <baseline.csv> [output_config.json]

Defaults: output -> metric_config.json
"""

import sys
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

ALLOWED_METRICS = {
    "entry_stopper_lowering_time",
    "pallet_movein_time",
    "movein_to_entry_stopper_up_delay",
    "entry_stopper_raising_time",
    "pallet_clamping_time",
    "pallet_lifting_time",
    "dispensing_time",
    "inspection_time",
    "pallet_unclamping_time",
    "pallet_lowering_time",
    "downstream_waiting_time",
    "exit_stopper_lowering_time",
    "pallet_moveout_time",
}

STEP_WINDOW       = 10
BASELINE_REF_FRAC = 0.20

# OSC sliding window params (fixed — not per-metric)
OSC_WINDOW = 50
OSC_STEP   = 5
OSC_LAG    = 3


def load_baseline(path: str) -> dict:
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        name = str(row["metric_name"]).strip()
        try:
            mean = float(row["mean"])    if pd.notna(row["mean"])    else None
            std  = float(row["std_dev"]) if pd.notna(row["std_dev"]) else None
            if mean is not None and std is not None:
                out[name] = (mean, max(std, 1e-6))
        except Exception:
            pass
    return out


def analyse_metric(metric: str, station: str, df: pd.DataFrame,
                   baseline: dict) -> dict | None:
    sub = df[
        (df["metric_name"] == metric) &
        (df["station_name"] == station)
    ].sort_values("timestamp").reset_index(drop=True)

    if len(sub) < 50:
        return None

    bm, bs = baseline.get(metric, (sub["value"].mean(), sub["value"].std() or 1e-6))

    print(f"\n{'═'*70}")
    print(f"  METRIC : {metric}  |  STATION: {station}")
    print(f"  Baseline mean={bm:.4f}  std={bs:.4f}  |  "
          f"readings={len(sub)}  days={sub['timestamp'].dt.date.nunique()}")
    print(f"{'═'*70}")

    rec = {"metric": metric, "station": station, "bm": bm, "bs": bs}

    # ── 1. SPIKE_Z ────────────────────────────────────────────────────────────
    z_abs = ((sub["value"] - bm) / bs).abs()
    print(f"\n[1] RANDOM SPIKES")
    for t in [3, 5, 8, 10, 15]:
        c = (z_abs > t).sum()
        print(f"    |z|>{t:>2}  →  {c:5d} readings  ({c/len(sub)*100:.3f}%)")
    spike_z = round(float(max(z_abs.quantile(0.999), 5.0)), 1)
    print(f"    z_max={z_abs.max():.2f}  p99={z_abs.quantile(0.99):.2f}  "
          f"p99.9={z_abs.quantile(0.999):.2f}")
    print(f"    ✓ SPIKE_Z = {spike_z}")
    rec["spike_z"] = spike_z

    # ── 2. STEP_SIGMA_MULT ────────────────────────────────────────────────────
    day_means = sub.groupby(sub["timestamp"].dt.date)["value"].mean()
    jumps     = day_means.diff().abs().dropna() / bs
    print(f"\n[2] STEP JUMPS")
    print(f"    Day-to-day jumps in σ — median={jumps.median():.2f}  "
          f"p95={jumps.quantile(0.95):.2f}  max={jumps.max():.2f}")
    step_mult = round(float(max(jumps.quantile(0.95) * 1.5, 2.0)), 1)
    print(f"    ✓ STEP_SIGMA_MULT = {step_mult}")
    rec["step_sigma_mult"] = step_mult

    # ── 3. DRIFT_THRESH ───────────────────────────────────────────────────────
    daily_slopes = []
    for _, g in sub.groupby(sub["timestamp"].dt.date):
        if len(g) < 10:
            continue
        x = np.arange(len(g)).reshape(-1, 1)
        s = LinearRegression().fit(x, g["value"].values.reshape(-1, 1)).coef_[0][0]
        daily_slopes.append(abs(s))
    daily_slopes = np.array(daily_slopes) if daily_slopes else np.array([1e-5])
    print(f"\n[3] SLOW DRIFT")
    print(f"    Within-day |slope| — median={np.median(daily_slopes):.2e}  "
          f"p75={np.percentile(daily_slopes, 75):.2e}  "
          f"p95={np.percentile(daily_slopes, 95):.2e}")
    raw_drift = float(max(np.percentile(daily_slopes, 75) * 2, 1e-5))
    exp       = int(np.floor(np.log10(raw_drift)))
    drift_thresh = round(raw_drift, -exp)
    print(f"    ✓ DRIFT_THRESH = {drift_thresh:.2e}")
    rec["drift_thresh"] = drift_thresh

    # ── 4. VARIANCE_MULT ──────────────────────────────────────────────────────
    sub["roll_std"] = sub["value"].rolling(50, min_periods=10).std()
    daily_p95       = sub.groupby(sub["timestamp"].dt.date)["roll_std"].quantile(0.95)
    normal_p95_med  = float(daily_p95.median())
    print(f"\n[4] VARIANCE GROWTH")
    print(f"    Daily roll_std(50) p95 — median={normal_p95_med:.4f}  "
          f"max={daily_p95.max():.4f}  in σ: {normal_p95_med/bs:.2f}x")
    var_mult = round(float(max(normal_p95_med * 3 / bs, 2.0)), 1)
    print(f"    ✓ VARIANCE_MULT = {var_mult}  "
          f"(threshold = {var_mult*bs:.4f}  VARIANCE_MIN_FRAC = 0.10)")
    rec["variance_mult"]     = var_mult
    rec["variance_min_frac"] = 0.10

    # ── 5. BASELINE_SHIFT_SIGMA ───────────────────────────────────────────────
    sub["roll_mean"] = sub["value"].rolling(10, min_periods=1).mean()
    end_devs    = sub.groupby(sub["timestamp"].dt.date)["roll_mean"].last()
    devs_sigma  = ((end_devs - bm) / bs).abs()
    print(f"\n[5] BASELINE SHIFT")
    print(f"    End-of-day deviation — median={devs_sigma.median():.2f}σ  "
          f"p85={devs_sigma.quantile(0.85):.2f}σ  max={devs_sigma.max():.2f}σ")
    shift_sigma = round(float(max(devs_sigma.quantile(0.85) * 1.5, 1.5)), 1)
    print(f"    ✓ BASELINE_SHIFT_SIGMA = {shift_sigma}")
    rec["baseline_shift_sigma"] = shift_sigma

    # ── 6. OUTLIER_FREQ_MULT ──────────────────────────────────────────────────
    print(f"\n[6] INCREASING OUTLIER FREQUENCY")
    daily_outliers = sub.groupby(sub["timestamp"].dt.date).apply(
        lambda g: ((g["value"] - bm).abs() / bs > spike_z).sum()
    )
    nonzero = daily_outliers[daily_outliers > 0]
    print(f"    Days with outliers: {len(nonzero)}/{len(daily_outliers)}  "
          f"max/day={daily_outliers.max()}  "
          f"median(nonzero)={nonzero.median() if len(nonzero) else 0:.1f}")
    print(f"    ✓ OUTLIER_FREQ_MULT = 2.0  OUTLIER_MIN_COUNT = 3")
    rec["outlier_freq_mult"] = 2.0
    rec["outlier_min_count"] = 3

    # ── 7. OSCILLATION ────────────────────────────────────────────────────────
    print(f"\n[7] OSCILLATION")
    lag3_vals = [
        g["value"].autocorr(lag=OSC_LAG)
        for _, g in sub.groupby(sub["timestamp"].dt.date)
        if len(g) >= 10
    ]
    lag3_vals   = [v for v in lag3_vals if v is not None]
    lag3_median = float(np.median(lag3_vals)) if lag3_vals else 0.0
    lag3_min    = float(np.min(lag3_vals))    if lag3_vals else 0.0
    lag3_max    = float(np.max(lag3_vals))    if lag3_vals else 0.0
    print(f"    lag-{OSC_LAG} autocorr — median={lag3_median:.3f}  "
          f"min={lag3_min:.3f}  max={lag3_max:.3f}")

    if lag3_median < -0.4:
        osc_type   = "oscillation_loss"
        osc_thresh = round(lag3_median * 0.75, 2)
        print(f"    → Structural NEGATIVE autocorr → detect_oscillation_loss")
        print(f"    ✓ OSC_THRESH = {osc_thresh}  (flag when autocorr weaker than this)")
    elif lag3_median > 0.4:
        osc_type   = "periodic_oscillation"
        osc_thresh = round(lag3_median * 0.6, 2)
        print(f"    → Structural POSITIVE autocorr → detect_periodic_oscillation")
        print(f"    ✓ OSC_THRESH = {osc_thresh}  (flag when autocorr exceeds this)")
    else:
        osc_type   = "none"
        osc_thresh = 0.0
        print(f"    → Near-zero autocorr → oscillation detector DISABLED")

    rec["osc_type"]    = osc_type
    rec["osc_thresh"]  = osc_thresh
    rec["lag3_median"] = lag3_median

    return rec


def build_config(results: list[dict]) -> dict:
    """Convert analysis results into a clean JSON-serialisable config."""
    metrics_cfg = {}
    for r in results:
        metrics_cfg[r["metric"]] = {
            "station":              r["station"],
            "spike_z":              r["spike_z"],
            "step_sigma_mult":      r["step_sigma_mult"],
            "drift_thresh":         r["drift_thresh"],
            "variance_mult":        r["variance_mult"],
            "variance_min_frac":    r["variance_min_frac"],
            "baseline_shift_sigma": r["baseline_shift_sigma"],
            "outlier_freq_mult":    r["outlier_freq_mult"],
            "outlier_min_count":    r["outlier_min_count"],
            "osc_type":             r["osc_type"],
            "osc_thresh":           r["osc_thresh"],
        }

    all_vals = lambda key: [r[key] for r in results]
    config = {
        "metrics":  metrics_cfg,
        "osc_window": OSC_WINDOW,
        "osc_step":   OSC_STEP,
        "osc_lag":    OSC_LAG,
        "defaults": {
            "spike_z":              round(float(np.median(all_vals("spike_z"))),             1),
            "step_sigma_mult":      round(float(np.median(all_vals("step_sigma_mult"))),     1),
            "stability_ratio":      0.6,
            "abrupt_mult":          1.5,
            "drift_thresh":         float(np.median(all_vals("drift_thresh"))),
            "variance_mult":        round(float(np.median(all_vals("variance_mult"))),       1),
            "variance_min_frac":    0.10,
            "baseline_shift_sigma": round(float(np.median(all_vals("baseline_shift_sigma"))), 1),
            "outlier_freq_mult":    2.0,
            "outlier_min_count":    3,
            "osc_type":             "none",
            "osc_thresh":           0.0,
        },
    }
    return config


def print_summary(config: dict):
    m = config["metrics"]
    d = config["defaults"]

    print(f"\n\n{'═'*70}")
    print("  CONFIG SUMMARY  →  metric_config.json")
    print(f"{'═'*70}")
    print(f"  {'metric':<45} {'spike_z':>7} {'step_σ':>6} {'var_x':>6} "
          f"{'shift_σ':>7} {'osc_type':<22} {'osc_thresh':>10}")
    print(f"  {'-'*45} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*22} {'-'*10}")
    for name, cfg in m.items():
        print(f"  {name:<45} {cfg['spike_z']:>7.1f} {cfg['step_sigma_mult']:>6.1f} "
              f"{cfg['variance_mult']:>6.1f} {cfg['baseline_shift_sigma']:>7.1f} "
              f"{cfg['osc_type']:<22} {cfg['osc_thresh']:>10.3f}")

    osc_loss = [n for n, c in m.items() if c["osc_type"] == "oscillation_loss"]
    osc_pres = [n for n, c in m.items() if c["osc_type"] == "periodic_oscillation"]
    osc_none = [n for n, c in m.items() if c["osc_type"] == "none"]

    print(f"\n  oscillation_loss  ({len(osc_loss)}): {osc_loss}")
    print(f"  periodic_osc      ({len(osc_pres)}): {osc_pres}")
    print(f"  osc disabled      ({len(osc_none)}): {osc_none}")

    print(f"\n  DEFAULTS (used when metric not in config):")
    print(f"    spike_z={d['spike_z']}  step_sigma_mult={d['step_sigma_mult']}  "
          f"variance_mult={d['variance_mult']}  baseline_shift_sigma={d['baseline_shift_sigma']}")
    print(f"{'═'*70}")


def main():
    csv_path      = sys.argv[1] if len(sys.argv) > 1 else "process_metrics.csv"
    baseline_path = sys.argv[2] if len(sys.argv) > 2 else None
    config_out    = sys.argv[3] if len(sys.argv) > 3 else "metric_config.json"

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    df = df[df["metric_name"].isin(ALLOWED_METRICS)]

    baseline = {}
    if baseline_path:
        baseline = load_baseline(baseline_path)
        print(f"Loaded {len(baseline)} baseline entries.")

    results = []
    for (station, metric), _ in df.groupby(["station_name", "metric_name"]):
        r = analyse_metric(metric, station, df, baseline)
        if r:
            results.append(r)

    config = build_config(results)
    print_summary(config)

    with open(config_out, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n  Config saved → {config_out}")


if __name__ == "__main__":
    main()