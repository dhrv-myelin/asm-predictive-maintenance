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

STEP_WINDOW       = 10
BASELINE_REF_FRAC = 0.20

OSC_WINDOW = 50
OSC_STEP   = 5
OSC_LAG    = 3

# Timing targets (seconds) — all window/persistence calibration derives from these
TARGET_DETECTOR_WINDOW_S   = 3600   # 60 minutes → rolling stat window
TARGET_ANOMALY_PERSIST_S   = 600    # 10 minutes → persistence gate
TARGET_SHORT_WINDOW_S      = 600    # 10 minutes → fast-react window for shift/drift


# ──────────────────────────────────────────────
# Baseline CSV loading
# ──────────────────────────────────────────────


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


# ──────────────────────────────────────────────
# Helper: rolling-window slope distribution
# ──────────────────────────────────────────────


def _rolling_slopes(series: pd.Series, win: int, spike_mask: pd.Series) -> np.ndarray:
    """
    Compute rolling window slopes (normalised to units of full-window drift in σ)
    over `series`, skipping spike readings (spike_mask=True → spike).

    Returns an array of |slope| * win / bs values — i.e. total drift in σ over
    one window. This is scale-independent of reading rate.
    """
    slopes = []
    clean = series.where(~spike_mask)   # NaN out spikes
    for i in range(win, len(clean)):
        window_vals = clean.iloc[i - win: i].dropna()
        if len(window_vals) < win // 2:
            continue
        x = np.arange(len(window_vals)).reshape(-1, 1)
        s = LinearRegression().fit(x, window_vals.values.reshape(-1, 1)).coef_[0][0]
        slopes.append(abs(s))
    return np.array(slopes) if slopes else np.array([0.0])


# ──────────────────────────────────────────────
# Per-metric analysis
# ──────────────────────────────────────────────


def analyse_metric(
    metric: str,
    station: str,
    df: pd.DataFrame,
    baseline: dict,
) -> Optional[dict]:

    sub = df[
        (df["metric_name"] == metric) &
        (df["station_name"] == station)
    ].sort_values("timestamp").reset_index(drop=True)

    if len(sub) < 50:
        print(f"  [SKIP] {metric} @ {station} — fewer than 50 readings.")
        return None

    bm, bs = baseline.get(metric, (sub["value"].mean(), sub["value"].std() or 1e-6))

    print(f"\n{'═'*70}")
    print(f"  METRIC : {metric}  |  STATION: {station}")
    print(f"  Baseline mean={bm:.4f}  std={bs:.4f}  |  "
          f"readings={len(sub)}  days={sub['timestamp'].dt.date.nunique()}")
    print(f"{'═'*70}")

    rec = {"metric": metric, "station": station, "bm": bm, "bs": bs}

    # ── 0. DATA DENSITY → detector_window, short_window, anomaly_min_readings ─
    #
    # All timing targets are expressed in seconds; we convert to readings via
    # the observed median reading rate (readings/sec).
    #
    # detector_window      ≈ readings in TARGET_DETECTOR_WINDOW_S  (60 min)
    #   Used as the primary rolling window for baseline-shift and variance-growth.
    #   A 60-min window ensures the rolling stat reflects a genuinely sustained
    #   signal level rather than a transient excursion or transition edge.
    #
    # short_window         ≈ readings in TARGET_SHORT_WINDOW_S  (10 min)
    #   Used by the dual-window baseline-shift detector (fast-react channel).
    #   The short window can detect a shift within ~10 min of it starting.
    #
    # anomaly_min_readings ≈ readings in TARGET_ANOMALY_PERSIST_S  (10 min)
    #   Persistence gate: anomaly condition must hold for 10 consecutive minutes
    #   of readings before an event is emitted. Eliminates transient blips.
    #   Floor = 3 readings so low-rate metrics still work.
    # ──────────────────────────────────────────────────────────────────────────
    daily_rps = []
    for _, g in sub.groupby(sub["timestamp"].dt.date):
        if len(g) < 2:
            continue
        span = (g["timestamp"].iloc[-1] - g["timestamp"].iloc[0]).total_seconds()
        if span > 0:
            daily_rps.append(len(g) / span)

    median_rps = float(np.median(daily_rps)) if daily_rps else 1.0

    detector_window      = max(int(round(median_rps * TARGET_DETECTOR_WINDOW_S)), 10)
    short_window         = max(int(round(median_rps * TARGET_SHORT_WINDOW_S)),    5)
    anomaly_min_readings = max(int(round(median_rps * TARGET_ANOMALY_PERSIST_S)), 3)

    print(f"\n[0] DATA DENSITY")
    print(f"    median readings/sec    = {median_rps:.4f}")
    print(f"    ✓ DETECTOR_WINDOW      = {detector_window} readings  "
          f"(~{TARGET_DETECTOR_WINDOW_S//60} min)")
    print(f"    ✓ SHORT_WINDOW         = {short_window} readings  "
          f"(~{TARGET_SHORT_WINDOW_S//60} min — fast-react channel)")
    print(f"    ✓ ANOMALY_MIN_READINGS = {anomaly_min_readings} readings  "
          f"(~{TARGET_ANOMALY_PERSIST_S//60} min persistence)")

    rec["detector_window"]      = detector_window
    rec["short_window"]         = short_window
    rec["anomaly_min_readings"] = anomaly_min_readings

    # ── 1. SPIKE_Z ────────────────────────────────────────────────────────────
    #
    # spike_z is set to the p99.9 z-score across all training readings, floored
    # at 5.0. Only readings far outside what baseline.csv says is normal fire.
    # ─────────────────────────────────────────────────────────────────────────
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
    #
    # Changed from single day-level regression to rolling-window slopes so the
    # threshold is calibrated to exactly the same computation as the detector.
    #
    # Normalisation: |slope| * detector_window / bs
    #   = total drift in units of σ over one full rolling window
    #   This is reading-rate-independent and directly comparable across metrics.
    #
    # Threshold = p75 of all rolling-window normalised slopes × 2.0, floored at 0.5σ.
    # The 0.6× intra-day fudge previously applied in stats_model.py is removed;
    # this calibration is already at the right time-scale.
    # ─────────────────────────────────────────────────────────────────────────
    spike_mask   = z_abs > spike_z
    roll_slopes  = _rolling_slopes(sub["value"], detector_window, spike_mask)
    # Normalise: total drift in σ over the full window
    roll_slopes_norm = roll_slopes * detector_window / bs

    print(f"\n[3] SLOW DRIFT  (rolling window = {detector_window} readings, "
          f"normalised = |slope|×win/bs in σ)")
    print(f"    p50={np.percentile(roll_slopes_norm, 50):.3f}σ  "
          f"p75={np.percentile(roll_slopes_norm, 75):.3f}σ  "
          f"p95={np.percentile(roll_slopes_norm, 95):.3f}σ  "
          f"max={roll_slopes_norm.max():.3f}σ")

    drift_thresh = round(float(max(np.percentile(roll_slopes_norm, 75) * 2.0, 0.5)), 2)
    print(f"    ✓ DRIFT_THRESH = {drift_thresh:.2f}σ  "
          f"(window-normalised, no intra-day fudge needed)")
    rec["drift_thresh"] = drift_thresh

    # ── 4. VARIANCE_MULT ──────────────────────────────────────────────────────
    #
    # Rolling std is computed using detector_window (60 min) for consistency
    # with what the variance-growth detector computes at runtime.
    #
    # var_mult is set so threshold = var_mult × bs sits above the p95 of normal
    # rolling-std values across all training days, capped at 5×, floored at 2×.
    # ─────────────────────────────────────────────────────────────────────────
    sub["roll_std_diag"] = sub["value"].rolling(
        detector_window, min_periods=max(detector_window // 2, 5)
    ).std()
    daily_p95      = sub.groupby(sub["timestamp"].dt.date)["roll_std_diag"].quantile(0.95)
    normal_p95_med = float(daily_p95.median())

    print(f"\n[4] VARIANCE GROWTH  (roll window = {detector_window} readings / ~60 min)")
    print(f"    Daily roll_std p95 — median={normal_p95_med:.6f}  "
          f"max={daily_p95.max():.6f}  in σ: {normal_p95_med/bs:.2f}x")

    var_mult = round(float(max(min(normal_p95_med * 3 / bs, 5.0), 2.0)), 1)
    print(f"    ✓ VARIANCE_MULT = {var_mult}  (threshold = {var_mult*bs:.6f})")
    rec["variance_mult"]     = var_mult
    rec["variance_min_frac"] = 0.10   # kept for legacy compatibility

    # ── 5. BASELINE_SHIFT_SIGMA ───────────────────────────────────────────────
    #
    # Two thresholds are now calibrated:
    #
    # shift_sigma (long)  — used with detector_window (60 min rolling mean).
    #   Set above the p85 of end-of-day deviations in training data, floored at 2.5σ.
    #   This is the "confirm" channel: the slow rolling mean must agree with the shift.
    #
    # shift_sigma_short   — used with short_window (10 min rolling mean).
    #   Set above the p70 of intra-day short-window mean deviations, floored at 2.0σ.
    #   This is the "fast-react" channel: fires within ~10 min of a shift starting.
    #
    # The stability gate in stats_model.py uses roll_std < deviation × 1.2
    # (relaxed from 0.9) to keep the gate open during the transition edge when
    # roll_std is temporarily elevated.
    # ─────────────────────────────────────────────────────────────────────────

    # Long-window shift sigma (60 min rolling mean → end-of-day deviation)
    sub["roll_mean_long"] = sub["value"].rolling(
        detector_window, min_periods=max(detector_window // 2, 5)
    ).mean()
    end_devs_long   = sub.groupby(sub["timestamp"].dt.date)["roll_mean_long"].last()
    devs_sigma_long = ((end_devs_long - bm) / bs).abs()

    print(f"\n[5] BASELINE SHIFT")
    print(f"    ── Long window ({detector_window} readings / ~60 min) ──────────────")
    print(f"    End-of-day deviation — "
          f"median={devs_sigma_long.median():.2f}σ  "
          f"p85={devs_sigma_long.quantile(0.85):.2f}σ  "
          f"max={devs_sigma_long.max():.2f}σ")

    shift_sigma = round(float(max(devs_sigma_long.quantile(0.85) * 1.5, 2.5)), 1)
    print(f"    ✓ BASELINE_SHIFT_SIGMA (long)  = {shift_sigma}σ  "
          f"(threshold = {shift_sigma * bs:.6f})")
    rec["baseline_shift_sigma"] = shift_sigma

    # Short-window shift sigma (10 min rolling mean → intra-day deviations)
    sub["roll_mean_short"] = sub["value"].rolling(
        short_window, min_periods=max(short_window // 2, 3)
    ).mean()
    # Collect all intra-day short-window mean deviations (not just end-of-day)
    intraday_devs_short = ((sub["roll_mean_short"] - bm) / bs).abs().dropna()

    print(f"\n    ── Short window ({short_window} readings / ~10 min) ─────────────")
    print(f"    All intra-day short-window deviations — "
          f"p50={intraday_devs_short.quantile(0.50):.2f}σ  "
          f"p70={intraday_devs_short.quantile(0.70):.2f}σ  "
          f"p85={intraday_devs_short.quantile(0.85):.2f}σ  "
          f"p95={intraday_devs_short.quantile(0.95):.2f}σ")

    # Use p70 × 1.5 so the short channel is more sensitive than the long one
    shift_sigma_short = round(float(max(intraday_devs_short.quantile(0.70) * 1.5, 2.0)), 1)
    print(f"    ✓ BASELINE_SHIFT_SIGMA (short) = {shift_sigma_short}σ  "
          f"(threshold = {shift_sigma_short * bs:.6f})")
    rec["baseline_shift_sigma_short"] = shift_sigma_short

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
        print(f"    ✓ OSC_THRESH = {osc_thresh}")
    elif lag3_median > 0.4:
        osc_type   = "periodic_oscillation"
        osc_thresh = round(lag3_median * 0.6, 2)
        print(f"    → Structural POSITIVE autocorr → detect_periodic_oscillation")
        print(f"    ✓ OSC_THRESH = {osc_thresh}")
    else:
        osc_type   = "none"
        osc_thresh = 0.0
        print(f"    → Near-zero autocorr → oscillation detector DISABLED")

    rec["osc_type"]    = osc_type
    rec["osc_thresh"]  = osc_thresh
    rec["lag3_median"] = lag3_median

    return rec


# ──────────────────────────────────────────────
# Config assembly
# ──────────────────────────────────────────────


def build_config(results: list[dict]) -> dict:
    metrics_cfg = {}
    for r in results:
        metrics_cfg[r["metric"]] = {
            "station":                    r["station"],
            "detector_window":            r["detector_window"],
            "short_window":               r["short_window"],
            "anomaly_min_readings":       r["anomaly_min_readings"],
            "spike_z":                    r["spike_z"],
            "step_sigma_mult":            r["step_sigma_mult"],
            "drift_thresh":               r["drift_thresh"],
            "variance_mult":              r["variance_mult"],
            "variance_min_frac":          r["variance_min_frac"],
            "baseline_shift_sigma":       r["baseline_shift_sigma"],
            "baseline_shift_sigma_short": r["baseline_shift_sigma_short"],
            "outlier_freq_mult":          r["outlier_freq_mult"],
            "outlier_min_count":          r["outlier_min_count"],
            "osc_type":                   r["osc_type"],
            "osc_thresh":                 r["osc_thresh"],
        }

    def all_vals(key):
        return [r[key] for r in results]

    config = {
        "metrics":    metrics_cfg,
        "osc_window": OSC_WINDOW,
        "osc_step":   OSC_STEP,
        "osc_lag":    OSC_LAG,
        # Timing targets (seconds) stored for reference / downstream validation
        "timing_targets": {
            "detector_window_s":  TARGET_DETECTOR_WINDOW_S,
            "short_window_s":     TARGET_SHORT_WINDOW_S,
            "anomaly_persist_s":  TARGET_ANOMALY_PERSIST_S,
        },
        "defaults": {
            "detector_window":            int(np.median(all_vals("detector_window"))),
            "short_window":               int(np.median(all_vals("short_window"))),
            "anomaly_min_readings":       int(np.median(all_vals("anomaly_min_readings"))),
            "spike_z":                    round(float(np.median(all_vals("spike_z"))), 1),
            "step_sigma_mult":            round(float(np.median(all_vals("step_sigma_mult"))), 1),
            "stability_ratio":            0.6,
            "abrupt_mult":                1.5,
            "drift_thresh":               round(float(np.median(all_vals("drift_thresh"))), 2),
            "variance_mult":              round(float(np.median(all_vals("variance_mult"))), 1),
            "variance_min_frac":          0.10,
            "baseline_shift_sigma":       round(float(np.median(all_vals("baseline_shift_sigma"))), 1),
            "baseline_shift_sigma_short": round(float(np.median(all_vals("baseline_shift_sigma_short"))), 1),
            "outlier_freq_mult":          2.0,
            "outlier_min_count":          3,
            "osc_type":                   "none",
            "osc_thresh":                 0.0,
        },
    }
    return config


# ──────────────────────────────────────────────
# Summary printer
# ──────────────────────────────────────────────


def print_summary(config: dict):
    m = config["metrics"]
    d = config["defaults"]

    print(f"\n\n{'═'*90}")
    print("  CONFIG SUMMARY  →  metric_config.json")
    print(f"  Timing targets: "
          f"detector_window={TARGET_DETECTOR_WINDOW_S//60} min  "
          f"short_window={TARGET_SHORT_WINDOW_S//60} min  "
          f"anomaly_persist={TARGET_ANOMALY_PERSIST_S//60} min")
    print(f"{'═'*90}")
    print(f"  {'metric':<40} {'win':>5} {'sw':>5} {'min_r':>6} {'spike_z':>7} "
          f"{'step_σ':>6} {'var_x':>6} {'shft_σ':>7} {'shft_σs':>7} {'osc_type':<22}")
    print(f"  {'-'*40} {'-'*5} {'-'*5} {'-'*6} {'-'*7} "
          f"{'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*22}")
    for name, cfg in m.items():
        print(
            f"  {name:<40} {cfg['detector_window']:>5} "
            f"{cfg['short_window']:>5} "
            f"{cfg['anomaly_min_readings']:>6} "
            f"{cfg['spike_z']:>7.1f} {cfg['step_sigma_mult']:>6.1f} "
            f"{cfg['variance_mult']:>6.1f} "
            f"{cfg['baseline_shift_sigma']:>7.1f} "
            f"{cfg['baseline_shift_sigma_short']:>7.1f} "
            f"{cfg['osc_type']:<22}"
        )

    print(f"\n  DEFAULTS:")
    print(
        f"    detector_window={d['detector_window']}  "
        f"short_window={d['short_window']}  "
        f"anomaly_min_readings={d['anomaly_min_readings']}  "
        f"spike_z={d['spike_z']}"
    )
    print(
        f"    step_sigma_mult={d['step_sigma_mult']}  "
        f"variance_mult={d['variance_mult']}  "
        f"baseline_shift_sigma={d['baseline_shift_sigma']}  "
        f"baseline_shift_sigma_short={d['baseline_shift_sigma_short']}"
    )
    print(f"    drift_thresh={d['drift_thresh']}σ  "
          f"(window-normalised, no intra-day fudge applied in stats_model.py)")
    print(f"{'═'*90}")



