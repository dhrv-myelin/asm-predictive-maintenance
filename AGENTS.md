# AGENTS.md

## Project Overview

Predictive maintenance system for a glue dispenser (industrial digital twin). Uses TimescaleDB for time-series storage, MLflow for model tracking, and Grafana for visualization.

## Package Manager

Uses `uv` (not pip). Always prefix commands with `uv run`.

## Required Setup

1. Set PYTHONPATH to project root before running any service:
   ```bash
   export PYTHONPATH=/path/to/asm-predictive-maintenance:$PYTHONPATH
   ```

2. Start infrastructure and run migrations:
   ```bash
   docker compose up -d
   uv run alembic upgrade head
   ```

## Running Services

Services must be run from their respective directories:

```bash
# Log Parser (from services/log_parser/)
uv run python src/main.py --mode record --input <data_log_file_path>
# Options: --no-db (output to CSV), --once (static file dump)

# Analysis Engine (from services/analysis/)
uv run python src/orchestrator.py --mode <stats|train|infer|backup>
```

## Database

- TimescaleDB on `localhost:5432`
- Database: `glue-dispenser-db`
- Credentials: `postgres:postgres`
- Migrations in `alembic/versions/`

## Linting & Formatting

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy .  # manual stage in pre-commit
```

## Architecture

```
services/
  log_parser/   - Parses logs, streams to DB
  analysis/     - ML pipeline (XGBoost, Isolation Forest, VAR, Mamba)
  grafana/      - Dashboards (localhost:3000)
shared/db/      - SQLAlchemy models, database connection
```

## Config

- Analysis models configured in `services/analysis/config/analysis_config.yaml`
- Log parser configs in `services/log_parser/config/{CED,GDM,RBW}/`
