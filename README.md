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