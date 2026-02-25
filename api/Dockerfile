FROM python:3.11-slim

WORKDIR /app

# Install Poetry + deps
RUN pip install poetry
COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false && poetry install --no-dev

# Alembic migrations
COPY alembic ./alembic
COPY src ./src

CMD ["alembic", "upgrade", "head"]
