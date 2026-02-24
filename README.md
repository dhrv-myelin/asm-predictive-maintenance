# Install UV :
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Run Command 
In root (~/asm-predictive-maintenance/):

```
uv add alembic 
uv run alembic upgrade head 
```

# Starting Log Parser Service
```
cd services/log_parser
uv run python src/main.py --mode record --input data/vector_buffer.jsonl
```

Use --no-db for output to a .csv instead of the DB. 
Use --once for a static log file dump instead of a continuously streaming one

# Running complete streaming Pipeline:

Pre-Requisite : Have Vector streaming logs from the windows laptop

## 1. Start Feature Aggregator:
```
docker compose up -d
```

## 2. MLFLow Serve:
```
mlflow server --port 5000
```

## 3. LogParser:
```
cd asm-predictive-maintenance/services/log_parser
uv run python src/main.py --mode record --input data/vector_buffer.jsonl
```

## 4. Analysis Engine:
```
cd asm-predictive-maintenance/services/analysis
python src/orchestrator.py
```

## 5. Grafana:
```
cd grafana_configs_predictive/grafana_configs
docker compose up -d
```
