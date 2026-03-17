import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('4thFeb_cycle_time.csv')
df = df.iloc[1:100]
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
df = df.sort_values('timestamp').reset_index(drop=True)
values = df['value'].astype(float)

WINDOW_SIZE = 3
PRED_WINDOW = 1

# Only Zeroth Order ARIMA (d=0)
order      = (1, 0, 1)
order_name = '0th Order ARIMA (d=0)'

# SPC Control Limits
UCL  = 20.0
LCL  = 2.36
MEAN = 11.2
sigma = (UCL - MEAN) / 3   # = (20 - 11.2) / 3 = 2.933
ONE_SIGMA_UPPER = MEAN + sigma
ONE_SIGMA_LOWER = MEAN - sigma

def rolling_arima(values, window_size, pred_window, order):
    all_forecasts = []
    all_actuals   = []
    all_indices   = []

    i = 0
    while i + window_size + pred_window <= len(values):
        fit_vals    = values.iloc[i : i + window_size]
        actual_vals = values.iloc[i + window_size : i + window_size + pred_window]
        actual_idx  = list(range(i + window_size, i + window_size + pred_window))

        try:
            model    = ARIMA(fit_vals, order=order).fit()
            forecast = model.get_forecast(steps=pred_window).predicted_mean.values
        except Exception:
            forecast = np.array([np.nan] * pred_window)

        all_forecasts.extend(forecast)
        all_actuals.extend(actual_vals.values)
        all_indices.extend(actual_idx)

        i += pred_window

    return np.array(all_indices), np.array(all_forecasts), np.array(all_actuals)


print(f"Running rolling ARIMA{order} ...")
indices, forecasts, actuals = rolling_arima(values, WINDOW_SIZE, PRED_WINDOW, order)

mae  = np.mean(np.abs(forecasts - actuals))
rmse = np.sqrt(np.mean((forecasts - actuals) ** 2))
print(f"  Done — {len(indices)} forecast points | MAE: {mae:.3f} | RMSE: {rmse:.3f}")


# === PLOT ===
fig, ax = plt.subplots(figsize=(18, 7))
fig.suptitle(
    f'Model Prediction \n',fontsize=14, fontweight='bold'
)

ts = df['timestamp']

# Full series background
ax.plot(ts, values, 'b-', linewidth=1, alpha=0.3, label='Full Series')

# Actual values at forecast positions
ax.plot(ts.iloc[indices], actuals,
        'k-o', linewidth=1.5, markersize=3, label='Actual', zorder=3)

# Forecasted values
ax.plot(ts.iloc[indices], forecasts,
        'r--o', linewidth=1.5, markersize=3,
        label='Forecast', zorder=4)

# Error shading
ax.fill_between(ts.iloc[indices], actuals, forecasts,
                alpha=0.2, color='red', label='Error Region')

# ── SPC Horizontal Lines ──
ax.axhline(UCL,              color='darkred',  linewidth=1,   linestyle='-',  label=f'UCL = {UCL}')
ax.axhline(LCL,              color='darkred',  linewidth=1,   linestyle='-',  label=f'LCL = {LCL}')
ax.axhline(MEAN,             color='green',    linewidth=1,   linestyle='-',  label=f'Mean = {MEAN}')
ax.axhline(ONE_SIGMA_UPPER,  color='orange',   linewidth=1.5, linestyle='--', label=f'+1σ = {ONE_SIGMA_UPPER:.2f}')
ax.axhline(ONE_SIGMA_LOWER,  color='orange',   linewidth=1.5, linestyle='--', label=f'-1σ = {ONE_SIGMA_LOWER:.2f}')

# ── Shaded Control Bands ──
ax.axhspan(ONE_SIGMA_LOWER, ONE_SIGMA_UPPER, alpha=0.07, color='green',  label='±1σ Band')
ax.axhspan(ONE_SIGMA_UPPER, UCL,             alpha=0.05, color='yellow', label='1σ → UCL Band')
ax.axhspan(LCL,             ONE_SIGMA_LOWER, alpha=0.05, color='yellow', label='LCL → 1σ Band')

# ── Right-edge annotations for SPC lines ──
'''right_ts = ts.iloc[-1]
for y, label, color in [
    (UCL,             f'UCL={UCL}',                 'darkred'),
    (LCL,             f'LCL={LCL}',                 'darkred'),
    (MEAN,            f'Mean={MEAN}',               'green'),
    (ONE_SIGMA_UPPER, f'+1σ={ONE_SIGMA_UPPER:.2f}', 'darkorange'),
    (ONE_SIGMA_LOWER, f'-1σ={ONE_SIGMA_LOWER:.2f}', 'darkorange'),
]:
    ax.annotate(label, xy=(right_ts, y), xytext=(6, 0),
                textcoords='offset points', va='center',
                fontsize=9, color=color, fontweight='bold',
                annotation_clip=False)'''

# ── MAE / RMSE Info Box ──
ax.text(0.01, 0.97,
        f'MAE:  {mae:.2f}\nRMSE: {rmse:.2f}',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=11, fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat',
                  edgecolor='orange', alpha=0.95))

# ax.set_title(order_name, fontsize=13, fontweight='bold', pad=10)
ax.set_xlabel('Timestamp', fontsize=11)
ax.set_ylabel('Value (in seconds)', fontsize=11)
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rolling_arima_zeroth_order_spc.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary
print("\n" + "="*60)
print(f"RESULTS | Window={WINDOW_SIZE} | Pred steps={PRED_WINDOW}")
print("="*60)
print(f"{order_name:30} | MAE: {mae:.3f} | RMSE: {rmse:.3f}")
print(f"\nSPC Limits:")
print(f"  UCL          = {UCL}")
print(f"  +1σ          = {ONE_SIGMA_UPPER:.2f}")
print(f"  Mean         = {MEAN}")
print(f"  -1σ          = {ONE_SIGMA_LOWER:.2f}")
print(f"  LCL          = {LCL}")
print(f"  σ (1-sigma)  = {sigma:.3f}")