#!/bin/bash
# restore_db.sh
# This script restores the PostgreSQL database from a SQL dump file into your Docker container.

# --- Configuration ---
DB_USER="airflow"
DB_NAME="airflow"
DB_PASSWORD="airflow"
CONTAINER_NAME="stockpred-postgres"
DUMP_FILE="stock_prediction_db_dump.sql"

# --- Error Handling ---
set -e

# --- Cleanup PGPASSWORD on exit ---
trap 'unset PGPASSWORD' EXIT

echo "Starting database restore to container: ${CONTAINER_NAME}..."

# Check if dump file exists
if [ ! -f "${DUMP_FILE}" ]; then
    echo "Error: Dump file '${DUMP_FILE}' not found in the current directory."
    echo "Please ensure the dump file is present before running this script."
    exit 1
fi

# 1. Ensure PostgreSQL container is running
echo "Ensuring PostgreSQL container is running..."
docker compose up -d "${CONTAINER_NAME}"
sleep 10

# 2. Stop services that might be writing to the database
echo "Stopping application services for a clean restore..."
docker compose stop airflow-webserver airflow-scheduler frontend mlflow-server || true

# 3. Set PGPASSWORD environment variable for psql
export PGPASSWORD="${DB_PASSWORD}"

# 4. Create the database if it doesn't exist. This command will fail if it exists, which is fine.
echo "Attempting to create database '${DB_NAME}' if it doesn't exist..."
docker exec -t "${CONTAINER_NAME}" psql -U "${DB_USER}" -d postgres -c "CREATE DATABASE ${DB_NAME};" || true
# Note: Connect to the 'postgres' default database to create 'airflow'

# 5. Restore the database from the dump file
echo "Restoring database '${DB_NAME}' from '${DUMP_FILE}'..."
docker exec -i "${CONTAINER_NAME}" psql -U "${DB_USER}" -d "${DB_NAME}" < "${DUMP_FILE}"

echo "Database restore completed successfully."

# 6. Restart all services
echo "Restarting all application services..."
docker compose up -d

echo "Script finished."