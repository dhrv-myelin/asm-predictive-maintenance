import sys
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from typing import Optional
#imports for analysis_config.yaml
import yaml
from pathlib import Path
warnings.filterwarnings("ignore")

#allowed metrics set from config yaml
def _load_allowed_metrics(machine: str = "rbw_machine") -> set:
    config_path = Path(__file__).parents[1] / "config" / machine / "analysis_config.yaml"
    with open(config_path) as f:
        return set(yaml.safe_load(f)["allowed_metrics"])

ALLOWED_METRICS = _load_allowed_metrics()

STEP_WINDOW       = 10
BASELINE_REF_FRAC = 0.20

OSC_WINDOW = 50
OSC_STEP   = 5
OSC_LAG    = 3


TARGET_DETECTOR_WINDOW_S   = 3600   # 60 minutes → rolling stat window
TARGET_ANOMALY_PERSIST_S   = 600    # 10 minutes → persistence gate
TARGET_SHORT_WINDOW_S      = 600    # 10 minutes → fast-react window for shift/drift

# Hard cap on spike_z written to config.
# When training data is very clean, p99.9 z-score can be 40–50σ. A spike_z
# that high makes rolling-window masking in stats_model.py completely inert —
# clean.where(z <= spike_z) passes every reading through, so shifted values
# are never excluded and rolling means never converge to the true new level.
# 8σ covers all realistic transient spikes while leaving sustained level-shift
# readings (typically 3–6σ) visible to rolling mean/std computations.
# detect_random_spikes and detect_increasing_outlier_frequency in
# stats_model.py use spike_z as a detection threshold (not a masking one)
# and apply their own interpretation, so this cap does not affect them in
# an unintended way.
SPIKE_Z_CAP = 8.0

# Hard cap on step_sigma_mult.
# On clean training data day-to-day jumps can be near-zero, producing a
# p95 jump of <0.1σ and after ×1.5 a step_sigma_mult of 19σ+. At that level
# detect_step_jumps never fires. Cap at 4.0σ — any real step worth detecting
# will exceed this; a threshold above 4σ on a manufacturing timing metric
# means the jump is catastrophic and would be caught by other detectors first.
# STEP_SIGMA_MULT_CAP = 4.0
# STEP_SIGMA_MULT_FLOOR = 2.0
STEP_SIGMA_MULT_CAP   = 2.5   # was 4.0 — only used to calibrate step_sigma_mult
STEP_SIGMA_MULT_FLOOR = 1.5  
# Variance mult calibration bounds.
# Multiplier 2.0× (was 3.0×) keeps the threshold closer to the p95 normal
# rolling std without requiring a variance burst to be extreme to be detected.
# Cap 3.0 (was 5.0) — hitting 5.0 meant the threshold was unreachable.
VARIANCE_MULT_MULTIPLIER = 2.0
VARIANCE_MULT_CAP        = 3.0
VARIANCE_MULT_FLOOR      = 1.5

# Stability ratio calibration bounds.
# Calibrated per-metric from the median intra-day rolling std / bs.
# Floor 0.6 preserves original behaviour for stable metrics.
# Cap 2.0 allows noisy-but-healthy metrics to have a relaxed gate.
STABILITY_RATIO_MULTIPLIER = 2.0   # median_roll_std × 2.0 / bs → ratio
STABILITY_RATIO_FLOOR      = 0.6
STABILITY_RATIO_CAP        = 2.0


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

    Returns an array of |slope| values (NOT yet normalised by win/bs).
    Caller is responsible for normalisation so the same array can be reused
    for both drift_thresh and stability_ratio calibration.
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
    # at 5.0 and capped at SPIKE_Z_CAP (8.0).
    #
    # The cap is the key fix: when training data is very clean (no anomalies),
    # p99.9 can be 40–50σ. Writing that to the config makes the rolling-window
    # spike masking in stats_model.py (clean = df["value"].where(z <= spike_z))
    # completely inert — every reading passes through. Sustained level-shift
    # readings are never excluded, so the rolling mean never converges to the
    # true shifted level, and baseline_shift / variance_growth never fire.
    # ─────────────────────────────────────────────────────────────────────────
    z_abs = ((sub["value"] - bm) / bs).abs()
    print(f"\n[1] RANDOM SPIKES")
    for t in [3, 5, 8, 10, 15]:
        c = (z_abs > t).sum()
        print(f"    |z|>{t:>2}  →  {c:5d} readings  ({c/len(sub)*100:.3f}%)")
    raw_spike_z = float(max(z_abs.quantile(0.999), 5.0))
    spike_z     = round(min(raw_spike_z, SPIKE_Z_CAP), 1)
    print(f"    z_max={z_abs.max():.2f}  p99={z_abs.quantile(0.99):.2f}  "
          f"p99.9={z_abs.quantile(0.999):.2f}  (raw={raw_spike_z:.1f})")
    if raw_spike_z > SPIKE_Z_CAP:
        print(f"    ⚠ spike_z capped {raw_spike_z:.1f} → {spike_z}  "
              f"(training data is very clean; cap prevents masking inertia in detectors)")
    print(f"    ✓ SPIKE_Z = {spike_z}")
    rec["spike_z"] = spike_z

    # ── 2. STEP_SIGMA_MULT ────────────────────────────────────────────────────
    #
    # Calibrated from day-to-day mean jumps (p95 × 1.5).
    # Hard-capped at STEP_SIGMA_MULT_CAP (4.0σ).
    #
    # Key fix: on clean training data the p95 day-to-day jump can be near-zero,
    # producing a raw step_mult of 19σ+ which makes detect_step_jumps never fire.
    # 4.0σ is a reasonable upper bound: any genuine step worth detecting on a
    # manufacturing timing metric will exceed this, and a threshold above 4σ
    # means the jump is so large it would be flagged by other detectors anyway.
    # ─────────────────────────────────────────────────────────────────────────
    day_means = sub.groupby(sub["timestamp"].dt.date)["value"].mean()
    jumps     = day_means.diff().abs().dropna() / bs
    print(f"\n[2] STEP JUMPS")
    print(f"    Day-to-day jumps in σ — median={jumps.median():.2f}  "
          f"p95={jumps.quantile(0.95):.2f}  max={jumps.max():.2f}")
    raw_step_mult = float(jumps.quantile(0.95) * 1.5)
    step_mult     = round(float(max(min(raw_step_mult, STEP_SIGMA_MULT_CAP),
                                   STEP_SIGMA_MULT_FLOOR)), 1)
    if raw_step_mult > STEP_SIGMA_MULT_CAP:
        print(f"    ⚠ step_sigma_mult capped {raw_step_mult:.1f} → {step_mult}  "
              f"(training data too clean to calibrate from; using cap)")
    print(f"    ✓ STEP_SIGMA_MULT = {step_mult}")
    rec["step_sigma_mult"] = step_mult

    # ── 2b. STABILITY_RATIO ───────────────────────────────────────────────────
    #
    # NEW: calibrated per-metric instead of hardcoded at 0.6.
    #
    # The before/after stability gate in detect_step_jumps requires:
    #   before_tail.std() < stability_ratio × bs
    #   after_head.std()  < stability_ratio × bs  (× 1.5 in the detector)
    #
    # On metrics with natural jitter > 0.6×bs this gate always fails, silencing
    # the detector entirely. We calibrate stability_ratio from the median
    # rolling std of spike-excluded clean windows as a multiple of bs:
    #   stability_ratio = median_clean_roll_std × STABILITY_RATIO_MULTIPLIER / bs
    #
    # Floored at 0.6 (preserves behaviour for very stable metrics).
    # Capped at 2.0 (prevents the gate from becoming meaningless on noisy metrics).
    # ─────────────────────────────────────────────────────────────────────────
    spike_mask_sr = z_abs > spike_z
    clean_sr      = sub["value"].where(~spike_mask_sr)
    roll_std_sr   = clean_sr.rolling(
        short_window, min_periods=max(short_window // 2, 3)
    ).std().dropna()

    median_clean_roll_std = float(roll_std_sr.median()) if len(roll_std_sr) else bs * 0.6
    raw_stability_ratio   = median_clean_roll_std * STABILITY_RATIO_MULTIPLIER / bs
    stability_ratio       = round(float(
        max(min(raw_stability_ratio, STABILITY_RATIO_CAP), STABILITY_RATIO_FLOOR)
    ), 2)

    print(f"\n[2b] STABILITY RATIO")
    print(f"    median clean short-window roll_std = {median_clean_roll_std:.6f}  "
          f"({median_clean_roll_std/bs:.3f}×bs)")
    print(f"    raw ratio = {raw_stability_ratio:.3f}  "
          f"(×{STABILITY_RATIO_MULTIPLIER} / bs)")
    print(f"    ✓ STABILITY_RATIO = {stability_ratio}  "
          f"(gate: before_tail.std() < {stability_ratio * bs:.6f})")
    rec["stability_ratio"] = stability_ratio

    # ── 3. DRIFT_THRESH ───────────────────────────────────────────────────────
    #
    # Calibrated from rolling-window slopes so the threshold matches exactly
    # what detect_slow_drift computes at runtime.
    #
    # Normalisation: |slope| × detector_window / bs
    #   = total drift in units of σ over one full rolling window.
    #   Reading-rate-independent; directly comparable across metrics.
    #
    # Changed from p90×1.5 → p75×1.2:
    #   p90×1.5 overshot on metrics with occasional structured drift in training
    #   data, pushing the threshold above what anomalous drift could reach.
    #   p75×1.2 is closer to the "typical normal drift" level and gives the
    #   detector a realistic chance of firing on genuine anomalous drift.
    #   Floor remains 0.3σ.
    # ─────────────────────────────────────────────────────────────────────────
    spike_mask    = z_abs > spike_z
    roll_slopes   = _rolling_slopes(sub["value"], detector_window, spike_mask)
    roll_slopes_norm = roll_slopes * detector_window / bs

    print(f"\n[3] SLOW DRIFT  (rolling window = {detector_window} readings, "
          f"normalised = |slope|×win/bs in σ)")
    print(f"    p50={np.percentile(roll_slopes_norm, 50):.3f}σ  "
          f"p75={np.percentile(roll_slopes_norm, 75):.3f}σ  "
          f"p90={np.percentile(roll_slopes_norm, 90):.3f}σ  "
          f"p95={np.percentile(roll_slopes_norm, 95):.3f}σ  "
          f"max={roll_slopes_norm.max():.3f}σ")

    drift_thresh = round(float(max(np.percentile(roll_slopes_norm, 75) * 1.2, 0.3)), 2)
    print(f"    ✓ DRIFT_THRESH = {drift_thresh:.2f}σ  "
          f"(p75×1.2, window-normalised)")
    rec["drift_thresh"] = drift_thresh

    # ── 4. VARIANCE_MULT ──────────────────────────────────────────────────────
    #
    # Rolling std is computed using short_window (10 min) for consistency
    # with the updated variance-growth detector which uses the short window
    # as its primary signal. The long-window (60 min) std was diluting short
    # bursts of genuine variance growth.
    #
    # Changed from detector_window → short_window for the calibration roll.
    # Changed multiplier from 3.0× → 2.0× (VARIANCE_MULT_MULTIPLIER).
    # Changed cap from 5.0 → 3.0 (VARIANCE_MULT_CAP).
    # Floor unchanged at 1.5 (VARIANCE_MULT_FLOOR).
    #
    # threshold = var_mult × bs must sit above the p95 of NORMAL short-window
    # rolling-std values so it only fires on genuinely elevated variance, but
    # low enough that a real burst can cross it within its duration.
    # ─────────────────────────────────────────────────────────────────────────
    sub["roll_std_diag"] = sub["value"].rolling(
        short_window, min_periods=max(short_window // 2, 3)
    ).std()
    daily_p95      = sub.groupby(sub["timestamp"].dt.date)["roll_std_diag"].quantile(0.95)
    normal_p95_med = float(daily_p95.median())

    print(f"\n[4] VARIANCE GROWTH  (roll window = {short_window} readings / ~10 min)")
    print(f"    Daily short-window roll_std p95 — median={normal_p95_med:.6f}  "
          f"max={daily_p95.max():.6f}  in σ: {normal_p95_med/bs:.2f}x")

    var_mult = round(float(
        max(min(normal_p95_med * VARIANCE_MULT_MULTIPLIER / bs, VARIANCE_MULT_CAP),
            VARIANCE_MULT_FLOOR)
    ), 1)
    print(f"    ✓ VARIANCE_MULT = {var_mult}  "
          f"(threshold = {var_mult*bs:.6f}, "
          f"multiplier={VARIANCE_MULT_MULTIPLIER}×, cap={VARIANCE_MULT_CAP})")
    rec["variance_mult"]     = var_mult
    rec["variance_min_frac"] = 0.10   # kept for legacy compatibility

    # ── 5. BASELINE_SHIFT_SIGMA ───────────────────────────────────────────────
    #
    # Two thresholds calibrated (unchanged from previous version):
    #
    # shift_sigma (long)  — used with detector_window (60 min rolling mean).
    #   Set above the p85 of end-of-day deviations in training data, floored at 2.5σ.
    #   This is the "confirm" channel: the slow rolling mean must agree with the shift.
    #
    # shift_sigma_short   — used with short_window (10 min rolling mean).
    #   Set above the p70 of intra-day short-window mean deviations, floored at 2.0σ.
    #   This is the "fast-react" channel: fires within ~10 min of a shift starting.
    # ─────────────────────────────────────────────────────────────────────────

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

    sub["roll_mean_short"] = sub["value"].rolling(
        short_window, min_periods=max(short_window // 2, 3)
    ).mean()
    intraday_devs_short = ((sub["roll_mean_short"] - bm) / bs).abs().dropna()

    print(f"\n    ── Short window ({short_window} readings / ~10 min) ─────────────")
    print(f"    All intra-day short-window deviations — "
          f"p50={intraday_devs_short.quantile(0.50):.2f}σ  "
          f"p70={intraday_devs_short.quantile(0.70):.2f}σ  "
          f"p85={intraday_devs_short.quantile(0.85):.2f}σ  "
          f"p95={intraday_devs_short.quantile(0.95):.2f}σ")

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
            "stability_ratio":            r["stability_ratio"],
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
        # Calibration constants stored for auditability
        "calibration_constants": {
            "spike_z_cap":               SPIKE_Z_CAP,
            "step_sigma_mult_cap":        STEP_SIGMA_MULT_CAP,
            "step_sigma_mult_floor":      STEP_SIGMA_MULT_FLOOR,
            "variance_mult_multiplier":   VARIANCE_MULT_MULTIPLIER,
            "variance_mult_cap":          VARIANCE_MULT_CAP,
            "variance_mult_floor":        VARIANCE_MULT_FLOOR,
            "stability_ratio_multiplier": STABILITY_RATIO_MULTIPLIER,
            "stability_ratio_floor":      STABILITY_RATIO_FLOOR,
            "stability_ratio_cap":        STABILITY_RATIO_CAP,
        },
        "defaults": {
            "detector_window":            int(np.median(all_vals("detector_window"))),
            "short_window":               int(np.median(all_vals("short_window"))),
            "anomaly_min_readings":       int(np.median(all_vals("anomaly_min_readings"))),
            "spike_z":                    round(float(np.median(all_vals("spike_z"))), 1),
            "step_sigma_mult":            round(float(np.median(all_vals("step_sigma_mult"))), 1),
            "stability_ratio":            round(float(np.median(all_vals("stability_ratio"))), 2),
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



def print_summary(config: dict):
    m = config["metrics"]
    d = config["defaults"]

    print(f"\n\n{'═'*100}")
    print("  CONFIG SUMMARY  →  metric_config.json")
    print(f"  Timing targets: "
          f"detector_window={TARGET_DETECTOR_WINDOW_S//60} min  "
          f"short_window={TARGET_SHORT_WINDOW_S//60} min  "
          f"anomaly_persist={TARGET_ANOMALY_PERSIST_S//60} min  "
          f"spike_z_cap={SPIKE_Z_CAP}  "
          f"step_mult_cap={STEP_SIGMA_MULT_CAP}  "
          f"var_mult_cap={VARIANCE_MULT_CAP}")
    print(f"{'═'*100}")
    print(f"  {'metric':<40} {'win':>5} {'sw':>5} {'min_r':>6} {'spike_z':>7} "
          f"{'step_σ':>6} {'stab':>5} {'var_x':>6} {'shft_σ':>7} {'shft_σs':>7} {'osc_type':<22}")
    print(f"  {'-'*40} {'-'*5} {'-'*5} {'-'*6} {'-'*7} "
          f"{'-'*6} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*22}")
    for name, cfg in m.items():
        print(
            f"  {name:<40} {cfg['detector_window']:>5} "
            f"{cfg['short_window']:>5} "
            f"{cfg['anomaly_min_readings']:>6} "
            f"{cfg['spike_z']:>7.1f} "
            f"{cfg['step_sigma_mult']:>6.1f} "
            f"{cfg['stability_ratio']:>5.2f} "
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
        f"spike_z={d['spike_z']}  (capped at {SPIKE_Z_CAP})"
    )
    print(
        f"    step_sigma_mult={d['step_sigma_mult']}  (capped at {STEP_SIGMA_MULT_CAP})  "
        f"stability_ratio={d['stability_ratio']}  "
        f"variance_mult={d['variance_mult']}  (cap {VARIANCE_MULT_CAP}, mult {VARIANCE_MULT_MULTIPLIER}×)"
    )
    print(
        f"    baseline_shift_sigma={d['baseline_shift_sigma']}  "
        f"baseline_shift_sigma_short={d['baseline_shift_sigma_short']}"
    )
    print(f"    drift_thresh={d['drift_thresh']}σ  (p75×1.2, window-normalised)")
    print(f"{'═'*100}")

