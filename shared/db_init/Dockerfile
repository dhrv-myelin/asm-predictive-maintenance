FROM python:3.10-slim 

WORKDIR /app
COPY shared/db_init/requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY alembic ./alembic
COPY alembic.ini ./alembic.ini
COPY shared ./shared

