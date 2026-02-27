# Install UV :
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
# Running Locally
## 1. Setup Alembic
In root (~/asm-predictive-maintenance/):

```
uv add alembic 
uv run alembic upgrade head 
```
# Running complete streaming Pipeline:

Pre-Requisite : Have Vector streaming logs from the windows laptop

## 1. Start Feature Aggregator:
```
docker compose up -d
```
## 2. Log Parser Service

```
cd services/log_parser
uv run python src/main.py --mode record --input <data_log_file_path>
```
Use --input data/vector_buffer.jsonl for live inference streaming from windows.
Use --no-db for output to a .csv instead of the DB. 
Use --once for a static log file dump instead of a continuously streaming one

# Using Docker

## 1. Setup Postgres, MLFlow and Grafana

```
cd <path_to_project_folder>/asm-predictive-maintenance
docker compose up --build -d
```

## 2. Set PYTHON PATH:
```
export PYTHONPATH=<path_to_project_folder>/asm-predictive-maintenance:$PYTHONPATH
```

## 3. LogParser:
```
cd services/log_parser
uv run python src/main.py --mode record --input <data_log_file_path> --once
```

## 4. Analysis Engine:
```
cd asm-predictive-maintenance/services/analysis
uv run python src/orchestrator.py
```

## 5. Grafana:
On the browser, open :
```
localhost:3000
```
