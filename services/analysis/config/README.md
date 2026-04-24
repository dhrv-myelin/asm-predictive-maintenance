# Stats Pattern Detection — Multi-Machine Setup

Statistical anomaly detection pipeline for manufacturing machines using pattern recognition (spikes, step jumps, drift, variance growth, oscillations, baseline shift).

---

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

3. Seed the statistical patterns for your machine (first time only):
```bash
cd services/analysis

# rbw_machine
psql -h localhost -p 5432 -U postgres -d glue-dispenser-db \
  -f config/rbw_machine/seed_statistical_patterns.sql

# ced_machine
psql -h localhost -p 5432 -U postgres -d glue-dispenser-db \
  -f config/ced_machine/seed_statistical_patterns.sql
```

---

## Running Stats Pattern Detection

```bash
cd services/analysis

# rbw_machine
uv run python src/orchestrator.py --mode stats --machine rbw_machine

# ced_machine
uv run python src/orchestrator.py --mode stats --machine ced_machine
```

Output: Writes detected pattern windows to the `patterns` table in the database.

---

## How It Works

```
--machine rbw_machine
        ↓
config/rbw_machine/analysis_config.yaml   (allowed_metrics loaded from here)
        ↓
baseline_stats.py        →   calibrates per-metric thresholds
        ↓
stats_model_2.py         →   runs pattern detectors on process_metrics
        ↓
patterns table in DB
```

Each machine has its own config folder. The code is identical across machines — the config drives the difference.

---

## Detectable Patterns

| Pattern | Description |
|---|---|
| `Random Spikes` | Single readings far outside baseline |
| `Step Jump` | Sudden sustained level change |
| `Slow Drift` | Gradual monotonic movement away from baseline |
| `Trend Acceleration` | Rate of drift increasing over time |
| `Variance Growth` | Noise level increasing while mean stays near baseline |
| `Baseline Shift` | Rolling mean has persistently moved from baseline |
| `Increasing Outlier Frequency` | Second half of day has more outliers than first |
| `Periodic Oscillation` | Repeating structured pattern in signal |
| `Oscillation Loss` | Expected oscillation pattern has disappeared |

---

## Configuration

### Folder Structure

```
services/analysis/
  config/
    rbw_machine/
      analysis_config.yaml          ← allowed_metrics + model config
      seed_statistical_patterns.sql ← pattern → cause mapping for this machine
    ced_machine/
      analysis_config.yaml
      seed_statistical_patterns.sql
  src/
    baseline_stats.py               ← threshold calibration
    stats_model_2.py                ← pattern detectors
    orchestrator.py                 ← entry point
```

### Adding / Removing Metrics

Edit `allowed_metrics` in the machine's `analysis_config.yaml`:

```yaml
allowed_metrics:
  - clamping_time
  - pre_data_handshake_wait
  - z_axis_positioning_time
  - gantry_positioning_time
  - z_axis_homing_time
  - marking_galvo_positioning_time
```

No code changes needed — the scripts pick this up automatically on next run.

---

## Adding a New Machine

1. Create the config folder:
```
config/
  new_machine/
    analysis_config.yaml          ← copy from rbw_machine, update allowed_metrics
    seed_statistical_patterns.sql ← patterns for this machine (use ID range 2000+)
```

2. Seed the patterns (first time only):
```bash
psql -h localhost -p 5432 -U postgres -d glue-dispenser-db \
  -f config/new_machine/seed_statistical_patterns.sql
```

3. Run:
```bash
uv run python src/orchestrator.py --mode stats --machine new_machine
```

