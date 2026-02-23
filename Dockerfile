# Use official PostgreSQL image
FROM postgres:18

# Set environment variables from your config
ENV POSTGRES_DB=glue-dispenser-db
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=myelin123

# Expose PostgreSQL port
EXPOSE 5432
