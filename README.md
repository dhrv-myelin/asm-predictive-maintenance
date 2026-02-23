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

```
uv run python src/main.py --mode record --viz --patterns config/log_patterns/test_patterns.yaml
```