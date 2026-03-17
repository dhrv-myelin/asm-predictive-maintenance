import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('4thFeb_cycle_time.csv')
df = df.iloc[1:150]
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
df = df.sort_values('timestamp').reset_index(drop=True)
values = df['value'].astype(float)

WINDOW_SIZE = 10  # increased from 3 so d=2 has enough points
PRED_WINDOW = 1

orders      = [(1,0,1), (1,1,1), (1,2,1)]
order_names = ['0th Order (d=0)', '1st Order (d=1)', '2nd Order (d=2)']

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


results = {}
for order, order_name in zip(orders, order_names):
    print(f"Running rolling ARIMA{order} ...")
    indices, forecasts, actuals = rolling_arima(values, WINDOW_SIZE, PRED_WINDOW, order)

    residuals = actuals - forecasts
    mae  = np.nanmean(np.abs(forecasts - actuals))
    rmse = np.sqrt(np.nanmean((forecasts - actuals) ** 2))

    results[order_name] = {
        'indices':   indices,
        'forecasts': forecasts,
        'actuals':   actuals,
        'residuals': residuals,
        'order':     order,
        'mae':       mae,
        'rmse':      rmse
    }
    print(f"  Done — {len(indices)} forecast points | MAE: {mae:.3f} | RMSE: {rmse:.3f}")


# === PLOT: 3 rows x 2 cols (left=forecast, right=ACF of residuals) ===
fig, axes = plt.subplots(3, 2, figsize=(20, 15),
                         gridspec_kw={'width_ratios': [3, 1]})

fig.suptitle(
    f'ROLLING ARIMA | Window={WINDOW_SIZE} pts → Forecast {PRED_WINDOW} pts | Sliding by {PRED_WINDOW}\n'
    f'Left: Forecast vs Actual   |   Right: ACF of Forecast Residuals',
    fontsize=15, fontweight='bold', y=0.99
)

for i, (order_name, data) in enumerate(results.items()):
    ax_forecast = axes[i][0]
    ax_acf      = axes[i][1]
    p, d, q     = data['order']

    # --- LEFT: Forecast plot ---
    ax_forecast.plot(df['timestamp'], values, 'b-', linewidth=1, alpha=0.3, label='Full Series')
    ax_forecast.plot(df['timestamp'].iloc[data['indices']], data['actuals'],
                     'k-o', linewidth=1.5, markersize=3, label='Actual', zorder=3)
    ax_forecast.plot(df['timestamp'].iloc[data['indices']], data['forecasts'],
                     'r--o', linewidth=1.5, markersize=3,
                     label=f'ARIMA({p},{d},{q}) Forecast', zorder=4)
    ax_forecast.fill_between(
        df['timestamp'].iloc[data['indices']],
        data['actuals'], data['forecasts'],
        alpha=0.2, color='red', label='Error region'
    )
    ax_forecast.text(0.99, 0.97,
                     f'MAE:  {data["mae"]:.2f}\nRMSE: {data["rmse"]:.2f}',
                     transform=ax_forecast.transAxes, ha='right', va='top',
                     fontsize=11, fontfamily='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat',
                               edgecolor='orange', alpha=0.95))
    ax_forecast.set_title(f'{order_name} — Forecast vs Actual', fontsize=12, fontweight='bold')
    ax_forecast.set_xlabel('Timestamp')
    ax_forecast.set_ylabel('Value')
    ax_forecast.legend(loc='upper left', fontsize=9)
    ax_forecast.grid(True, alpha=0.3)

    # --- RIGHT: ACF of residuals ---
    residuals       = data['residuals']
    clean_residuals = residuals[~np.isnan(residuals)]
    max_lags        = min(40, len(clean_residuals) // 2 - 1)

    if len(clean_residuals) < 4 or max_lags < 1:
        ax_acf.text(0.5, 0.5,
                    'ACF unavailable\n(too many NaN residuals)\n\nTip: increase WINDOW_SIZE',
                    transform=ax_acf.transAxes, ha='center', va='center',
                    fontsize=10, color='gray',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    else:
        plot_acf(clean_residuals, ax=ax_acf, lags=max_lags, alpha=0.05,
                 color='steelblue', vlines_kwargs={'colors': 'steelblue'})
        ax_acf.axhline(0, color='black', linewidth=0.8)

    ax_acf.set_title(f'ACF of Residuals\nARIMA({p},{d},{q})', fontsize=12, fontweight='bold')
    ax_acf.set_xlabel('Lag')
    ax_acf.set_ylabel('Autocorrelation')
    ax_acf.grid(True, alpha=0.3)

plt.subplots_adjust(top=0.93, bottom=0.06, hspace=0.5, wspace=0.25)
plt.savefig('rolling_arima_with_acf.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary
print("\n" + "="*60)
print(f"RESULTS | Window={WINDOW_SIZE} | Pred steps={PRED_WINDOW}")
print("="*60)
for order_name, data in results.items():
    print(f"{order_name:22} | MAE: {data['mae']:.3f} | RMSE: {data['rmse']:.3f}")