# Install UV :
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Run Command 
In root (~/asm-predictive-maintenance/):
```
uv run python src/main.py --mode record --input data/vector_buffer.jsonl --patterns config/log_patterns/prod_patterns.yaml
```

Use --no-db for output to a .csv instead of the DB. 
Use --once for a static log file dump instead of a continuously streaming one