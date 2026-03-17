import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter

# -----------------------------
# Config
# -----------------------------
ROLL_WIN = 20
SPIKE_Z = 3.0
ACCEL_THRESH = 1e-4
VAR_GROWTH_FACTOR = 1.8
MEAN_SHIFT_Z = 2.5
LAG_THRESH = 0.15   # seconds
DECEL_CURV_THRESH = 1e-3

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(df):
    df['roll_mean'] = df['value'].rolling(ROLL_WIN).mean()
    df['roll_std']  = df['value'].rolling(ROLL_WIN).std()
    df['diff'] = df['value'].diff()
    df['zscore'] = (df['value'] - df['value'].mean()) / df['value'].std()
    return df

# -----------------------------
# Pattern Detection
# -----------------------------
def detect_pneumatic_patterns(df, baseline_mean, baseline_std):
    events = {}

    # Baseline deviation
    df['z_base'] = (df['value'] - baseline_mean) / baseline_std

    # -----------------------------
    # Random actuation spikes
    # -----------------------------
    spikes = df[np.abs(df['z_base']) > SPIKE_Z]
    if len(spikes):
        events["Random actuation spikes"] = {
            "timestamps": list(spikes['timestamp']),
            "suggestion": "Main line pressure regulation check"
        }

    # -----------------------------
    # Actuation time creep (slow down)
    # -----------------------------
    x = np.arange(len(df)).reshape(-1,1)
    y = df['value'].values.reshape(-1,1)
    slope = LinearRegression().fit(x,y).coef_[0][0]
    if slope > 0:
        events["Actuation time creep (slow down)"] = {
            "timestamps": [df['timestamp'].iloc[-1]],
            "suggestion": "Seal lubrication check, guide/rod friction inspection"
        }

    # -----------------------------
    # Cycle time jitter (variance)
    # -----------------------------
    early = df['roll_std'].iloc[:len(df)//2].mean()
    late  = df['roll_std'].iloc[len(df)//2:].mean()
    if late > early * VAR_GROWTH_FACTOR:
        events["Cycle time jitter (variance)"] = {
            "timestamps": [df['timestamp'].iloc[len(df)//2]],
            "suggestion": "Valve restriction check, air moisture/filtration check"
        }

    # -----------------------------
    # Baseline flow elevation (idle leak)
    # -----------------------------
    if abs(df['z_base'].mean()) > MEAN_SHIFT_Z:
        events["Baseline flow elevation (idle leak)"] = {
            "timestamps": [df['timestamp'].iloc[-1]],
            "suggestion": "Internal seal (blow-by) replacement, fitting leak test"
        }

    # -----------------------------
    # End-of-stroke sensor flutter
    # -----------------------------
    smooth = savgol_filter(df['value'].values, 11, 2)
    hf_noise = np.std(df['value'].values - smooth)
    if hf_noise > baseline_std * 0.8:
        events["End-of-stroke sensor flutter"] = {
            "timestamps": list(df['timestamp'].iloc[-10:]),
            "suggestion": "Cushion adjustment, shock absorber inspection"
        }

    # -----------------------------
    # Actuation lag (delayed start)
    # -----------------------------
    if df['diff'].abs().iloc[0] < 1e-3 and df['value'].iloc[1] - df['value'].iloc[0] > LAG_THRESH:
        events["Actuation lag (delayed start)"] = {
            "timestamps": [df['timestamp'].iloc[0]],
            "suggestion": "Solenoid valve inspection, exhaust muffler cleaning"
        }

    # -----------------------------
    # Premature deceleration
    # -----------------------------
    coeffs = np.polyfit(np.arange(len(df)), df['value'].values, 2)
    curvature = coeffs[0]
    if abs(curvature) > DECEL_CURV_THRESH:
        events["Premature deceleration"] = {
            "timestamps": [df['timestamp'].iloc[-1]],
            "suggestion": "Cylinder barrel dent/damage check, mechanical binding"
        }

    return events

# -----------------------------
# Visualization
# -----------------------------
def plot_pneumatic(df, cylinder_id, baseline_mean, baseline_std, events):
    fig, ax = plt.subplots(figsize=(18, 10))

    ax.plot(df['timestamp'], df['value'], linewidth=1.5, label="Actuation Time")

    # Baseline
    ax.axhline(baseline_mean, linestyle='--', linewidth=2)
    ax.axhline(baseline_mean + 3*baseline_std, linestyle=':', linewidth=1.5)
    ax.axhline(baseline_mean - 3*baseline_std, linestyle=':', linewidth=1.5)

    style_map = {
        "Actuation time creep (slow down)": {"marker":"^","color":"blue"},
        "Cycle time jitter (variance)": {"marker":"o","color":"purple"},
        "Baseline flow elevation (idle leak)": {"marker":"D","color":"brown"},
        "End-of-stroke sensor flutter": {"marker":"P","color":"green"},
        "Actuation lag (delayed start)": {"marker":"<","color":"orange"},
        "Random actuation spikes": {"marker":"x","color":"red"},
        "Premature deceleration": {"marker":"*","color":"black"}
    }

    for pattern, info in events.items():
        style = style_map[pattern]
        ys = []
        for ts in info["timestamps"]:
            y = df[df['timestamp']==ts]['value']
            ys.append(y.values[0] if len(y) else df['value'].iloc[-1])

        ax.scatter(info["timestamps"], ys,
                   marker=style["marker"],
                   color=style["color"],
                   s=130,
                   edgecolors='black',
                   linewidths=0.4)

    ax.set_title(f"Pneumatic Cylinder Health | {cylinder_id}", fontsize=14)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Actuation Time (s)")
    ax.grid(alpha=0.3)

    # -------- Table --------
    table_data = []
    for pattern, info in events.items():
        table_data.append([pattern, len(info["timestamps"]), info["suggestion"]])

    if table_data:
        table = plt.table(
            cellText=table_data,
            colLabels=["Pattern","Events","Maintenance Action"],
            loc='bottom',
            cellLoc='center',
            colLoc='center',
            bbox=[0.0, -0.60, 1.0, 0.40]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.7)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.45)
    plt.show()

# -----------------------------
# Main Pipeline
# -----------------------------
def run_pneumatic_pipeline(csv_path, baseline_dict):
    df = pd.read_csv(csv_path)
    #df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['timestamp'] = pd.to_datetime(
    df['timestamp'],
    format='mixed',     # lets pandas auto-detect per row
    utc=True
)
    for cyl, stats in baseline_dict.items():
        cyl_df = df[df['cylinder_id'] == cyl].copy()
        if len(cyl_df) < 15:
            continue

        cyl_df = cyl_df.sort_values('timestamp')
        cyl_df = extract_features(cyl_df)

        events = detect_pneumatic_patterns(
            cyl_df,
            baseline_mean=stats['mean'],
            baseline_std=stats['std']
        )

        print(f"\n===== Cylinder: {cyl} =====")
        for p, info in events.items():
            print(f"{p} | Events: {len(info['timestamps'])}")
            print(f"Action: {info['suggestion']}\n")

        plot_pneumatic(cyl_df, cyl, stats['mean'], stats['std'], events)

if __name__ == "__main__":
    PNEUMATIC_BASELINE = {
        "cyl_A": {"mean": 1.42, "std": 0.03},
        "cyl_B": {"mean": 1.18, "std": 0.02},
    }

    run_pneumatic_pipeline("/home/varenyapathak/Downloads/4th_feb_data.csv", PNEUMATIC_BASELINE)