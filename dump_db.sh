#!/bin/bash
# dump_db.sh
# This script dumps the PostgreSQL database from your Docker container to a SQL file.

# --- Configuration ---
DB_USER="airflow"
DB_NAME="airflow"
DB_PASSWORD="airflow" # Make sure this matches your docker-compose.yml
CONTAINER_NAME="stockpred-postgres"
DUMP_FILE="stock_prediction_db_dump.sql"

# --- Error Handling ---
set -e # Exit immediately if a command exits with a non-zero status

# --- Cleanup PGPASSWORD on exit ---
trap 'unset PGPASSWORD' EXIT

echo "Starting database dump from container: ${CONTAINER_NAME}..."

# 1. Stop services that might be writing to the database (optional but recommended for consistency)
echo "Stopping application services for a consistent dump (if running)..."
docker compose stop airflow-webserver airflow-scheduler frontend mlflow-server || true # Use || true to prevent script from failing if services are not running

# 2. Set PGPASSWORD environment variable for pg_dump
export PGPASSWORD="${DB_PASSWORD}"

# 3. Create the database dump file
echo "Dumping database '${DB_NAME}' as user '${DB_USER}' to '${DUMP_FILE}'..."
docker exec -t "${CONTAINER_NAME}" pg_dump -U "${DB_USER}" -d "${DB_NAME}" > "${DUMP_FILE}"

echo "Database dump completed successfully to ${DUMP_FILE}."
echo "You can now transfer this file to the destination machine."

# 4. Restart your services
echo "Restarting application services..."
docker compose up -d

echo "Script finished."