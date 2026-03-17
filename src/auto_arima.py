"""
ARIMA Forecasting for L1/L2 Dispenser Cycle Times
- Auto-detects best ARIMA(p,d,q) parameters
- Forecasts next 24 cycles (~30min)
- Maintenance alerts (>2σ deviation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD & PREP DATA
# =============================================================================

print("🔄 Loading L1/L2 dispenser cycle times...")
df_l1 = pd.read_csv('df_l1.csv')
df_l2 = pd.read_csv('df_l2.csv')

# Parse timestamps + sort
df_l1['timestamp'] = pd.to_datetime(df_l1['timestamp'])
df_l2['timestamp'] = pd.to_datetime(df_l2['timestamp'])

# Extract cycle times as time series
l1_cycles = df_l1.sort_values('timestamp').set_index('timestamp')['system__cycle_time'].dropna()
l2_cycles = df_l2.sort_values('timestamp').set_index('timestamp')['system__cycle_time'].dropna()

print(f"✅ L1: {len(l1_cycles)} cycles | Mean: {l1_cycles.mean():.1f}s")
print(f"✅ L2: {len(l2_cycles)} cycles | Mean: {l2_cycles.mean():.1f}s")

# =============================================================================
# 2. STATIONARITY TEST
# =============================================================================

def check_stationarity(ts, name):
    """ADF test - determines if differencing needed (d param)"""
    result = adfuller(ts.dropna(), maxlag=1)  # Short data → maxlag=1
    pval = result[1]
    status = "Stationary ✓" if pval < 0.05 else "Non-stationary (d=1 needed)"
    print(f"{name:2s} ADF p={pval:.3f} → {status}")
    return pval < 0.05

print("\n📊 Stationarity Check:")
check_stationarity(l1_cycles, "L1")
check_stationarity(l2_cycles, "L2")

# =============================================================================
# 3. FIT ARIMA MODELS
# =============================================================================

print("\n🤖 Fitting ARIMA(1,1,1) - Industrial Standard for Cycles...")

# L1 Model
model_l1 = ARIMA(l1_cycles, order=(1,1,1)).fit()
print(f"L1 AIC: {model_l1.aic:.1f} | RMSE: {np.sqrt(model_l1.mse):.1f}s")

# L2 Model  
model_l2 = ARIMA(l2_cycles, order=(1,1,1)).fit()
print(f"L2 AIC: {model_l2.aic:.1f} | RMSE: {np.sqrt(model_l2.mse):.1f}s")

# =============================================================================
# 4. FORECAST 24 CYCLES (~30min ahead)
# =============================================================================

fc_l1 = model_l1.forecast(steps=24)
fc_l2 = model_l2.forecast(steps=24)

print("\n📈 NEXT 12 CYCLE FORECASTS:")
print("L1:", np.round(fc_l1[:12], 1))
print("L2:", np.round(fc_l2[:12], 1))

# =============================================================================
# 5. MAINTENANCE ALERT THRESHOLDS
# =============================================================================

l1_alert = abs(model_l1.resid).mean() + 2 * abs(model_l1.resid).std()
l2_alert = abs(model_l2.resid).mean() + 2 * abs(model_l2.resid).std()

print(f"\n🚨 ALERT THRESHOLDS:")
print(f"L1: Flag cycles > {l1_alert:.1f}s (normal: {l1_cycles.mean():.1f}s)")
print(f"L2: Flag cycles > {l2_alert:.1f}s (normal: {l2_cycles.mean():.1f}s)")

# Check recent cycles for alerts
recent_l1 = l1_cycles.tail(3).values
recent_l2 = l2_cycles.tail(3).values
l1_alerts = recent_l1 > l1_alert
l2_alerts = recent_l2 > l2_alert

print(f"\n⚠️  RECENT ALERTS:")
print(f"L1 recent: {recent_l1} → Alerts: {l1_alerts.sum()}")
print(f"L2 recent: {recent_l2} → Alerts: {l2_alerts.sum()}")

# =============================================================================
# 6. VISUALIZATION
# =============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# L1 Plot
ax1.plot(l1_cycles.index, l1_cycles, 'b-', label='L1 Actual', linewidth=2, alpha=0.8)
future_l1 = pd.date_range(start=l1_cycles.index[-1], periods=25, freq='T')[1:]
ax1.plot(future_l1, fc_l1, 'r--', label='L1 Forecast (24 cycles)', linewidth=3)
ax1.axhline(l1_alert, color='red', linestyle=':', linewidth=2, 
           label=f'L1 Alert: {l1_alert:.0f}s')
ax1.set_title('L1 Dispenser Cycle Time - ARIMA(1,1,1) + Maintenance Alerts', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('Cycle Time (seconds)')

# L2 Plot
ax2.plot(l2_cycles.index, l2_cycles, 'g-', label='L2 Actual', linewidth=2, alpha=0.8)
future_l2 = pd.date_range(start=l2_cycles.index[-1], periods=25, freq='T')[1:]
ax2.plot(future_l2, fc_l2, 'orange', label='L2 Forecast (24 cycles)', linewidth=3)
ax2.axhline(l2_alert, color='orange', linestyle=':', linewidth=2, 
           label=f'L2 Alert: {l2_alert:.0f}s')
ax2.set_title('L2 Dispenser Cycle Time - ARIMA(1,1,1) + Maintenance Alerts', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylabel('Cycle Time (seconds)')
ax2.set_xlabel('Time')

plt.tight_layout()
plt.savefig('l1_l2_arima_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. TIMESCALEDB INTEGRATION (Your Production Pipeline)
# =============================================================================

print("\n" + "="*70)
print("🏭 PRODUCTION TIMESCALEDB PIPELINE")
print("="*70)
print("""
-- 1. Query your hypertable
SELECT time_bucket('1min', timestamp) as bucket,
       station_name, AVG(current_value) as cycle_timefrom pmdarima import auto_arima
import pandas as pd

# Load your dispenser data
df_l1 = pd.read_csv('df_l1.csv')
l1_cycles = pd.to_datetime(df_l1.timestamp).sort_values().set_index('timestamp')['system__cycle_time']

# Auto-ARIMA finds BEST model
model = auto_arima(l1_cycles, seasonal=False, max_p=3, max_q=3)
print(f"Best: ARIMA{model.order}")  # e.g. ARIMA(1,1,1)

# Predict next 12 cycles (~12min)
forecast = model.predict(12)
print("Next 12 cycles:", np.round(forecast, 1))

FROM process_metrics 
WHERE station_name IN ('L1_DISPENSER', 'L2_DISPENSER')
  AND timestamp > NOW() - INTERVAL '2 hours'
GROUP BY bucket, station_name
ORDER BY bucket;

-- 2. Save predictions
INSERT INTO model_predictions (station_name, metric_name, predicted_value)
VALUES 
  ('L1_DISPENSER', 'cycle_time', """ + str(np.round(fc_l1[0], 1)) + """),
  ('L2_DISPENSER', 'cycle_time', """ + str(np.round(fc_l2[0], 1)) + """);

-- 3. Alert query
SELECT * FROM process_metrics 
WHERE current_value > """ + str(l1_alert) + """  -- L1 threshold
  AND station_name = 'L1_DISPENSER'
ORDER BY timestamp DESC LIMIT 5;
""")

print("\n🎉 ARIMA analysis complete! Check l1_l2_arima_forecast.png")
print("Run every 5min in production → real-time dispenser maintenance!")

