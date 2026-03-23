# Analysis Service

ML pipeline for predictive maintenance using XGBoost, Mamba, VAR, and Isolation Forest.

## Quick Start

### Prerequisites

1. Start infrastructure:
```bash
cd <project_root>
docker compose up -d
```

2. Set PYTHONPATH:
```bash
export PYTHONPATH=<project_root>:$PYTHONPATH
```

---

## Running Models

### 1. Stats Pattern Detection

Detects anomalies using statistical pattern recognition (spikes, oscillations, drift, etc.).

```bash
cd services/analysis
uv run python src/orchestrator.py --mode stats
```

Output: Writes to `patterns` table in database.

---

### 2. Isolation Forest (Health Score)

Trains and runs inference for health score prediction using Isolation Forest.

#### Training
```bash
uv run python src/orchestrator.py \
  --mode train \
  --start "2026-02-03 18:45:00" \
  --end "2026-02-03 19:27:35"
```

#### Live Inference
```bash
uv run python src/orchestrator.py \
  --mode infer \
  --model "ced_machine__health_score:isolation_forest"
```

#### Archive Replay
```bash
uv run python src/orchestrator.py \
  --mode backup \
  --start "2026-02-03 18:45:00" \
  --end "2026-02-03 19:27:35"
```

---

## Available Models

| Model | Type | Purpose |
|-------|------|---------|
| `isolation_forest` | sklearn | Health score / anomaly detection |
| `xgboost_forecast` | sklearn | Time series forecasting |
| `mamba` | torch | Deep learning forecasting |
| `var_forecast` | sklearn | Vector autoregression |
| `stats_pattern_detector::*` | stats | Statistical pattern detection |

---

## Configuration

Edit `config/analysis_config.yaml` to:
- Add/remove target metrics
- Configure model hyperparameters
- Adjust training parameters

---

## Database Tables

- `process_metrics` - Raw metrics from log parser
- `baseline_metrics` - Baseline statistics per metric
- `model_predictions` - Model inference results
- `patterns` - Stats pattern detection results
