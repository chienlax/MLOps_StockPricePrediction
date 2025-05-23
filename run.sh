#!/bin/bash

# run.sh
# Script to manage the Docker Compose environment for the MLOps Stock Prediction project.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Optional: Define COMPOSE_PROJECT_NAME if you want to namespace your containers/networks/volumes
# export COMPOSE_PROJECT_NAME=stockpred_mlops

# Ensure critical host directories mounted as volumes exist
echo "Ensuring host directories for Docker volumes exist..."
mkdir -p ./airflow/dags
mkdir -p ./airflow/logs
mkdir -p ./airflow/plugins
mkdir -p ./data/predictions/historical # Ensure full path for JSONs
mkdir -p ./data/raw
mkdir -p ./data/processed # If any file-based processed data is still used
mkdir -p ./data/features  # If any file-based features are still used
mkdir -p ./config
mkdir -p ./plots/training_plots # For training visualizations
echo "Host directories verified/created."

# --- Helper Functions ---
fn_print_usage() {
    echo "Usage: ./run.sh [ACTION]"
    echo "Actions:"
    echo "  up           Builds (if needed) and starts all services in detached mode, and initializes Airflow DB/user."
    echo "  start        Starts all services in detached mode (assumes images are built)."
    echo "  logs         Follows logs from all services."
    echo "  logs <service_name>"
    echo "               Follows logs for a specific service (e.g., airflow-webserver, mlflow-server)."
    echo "  stop         Stops all services."
    echo "  down         Stops and removes all services, networks. Add -v to remove volumes."
    echo "  build        Builds or rebuilds services."
    echo "  ps           Lists running containers for this project."
    echo "  init_airflow (Specific Airflow DB initialization/user creation - often run automatically by 'up')"
    echo "  clean_pyc    Removes __pycache__ and .pyc files from the src directory."
    echo ""
    echo "Default action if none specified: up"
}

fn_check_docker_compose() {
    if ! command -v docker-compose &> /dev/null
    then
        if ! command -v docker compose &> /dev/null # Check for 'docker compose' (v2)
        then
            echo "Error: docker-compose (or 'docker compose') is not installed or not in PATH."
            echo "Please install Docker and Docker Compose."
            exit 1
        else
            DOCKER_COMPOSE_CMD="docker compose"
        fi
    else
        DOCKER_COMPOSE_CMD="docker-compose"
    fi
    echo "Using Docker Compose command: $DOCKER_COMPOSE_CMD"
}

fn_set_airflow_uid() {
    # Set AIRFLOW_UID to the current user's ID to avoid permission issues with mounted volumes
    # This should be defined in your .env file, but we can set it here as a fallback if not.
    if [ -z "$AIRFLOW_UID" ]; then
        echo "AIRFLOW_UID not set in environment. Setting to current user ID: $(id -u)"
        export AIRFLOW_UID=$(id -u)
    else
        echo "AIRFLOW_UID already set in environment: $AIRFLOW_UID"
    fi
    # For Docker Compose to pick it up from this script's environment,
    # it needs to be either in the .env file or docker-compose.yml needs to reference it
    # as ${AIRFLOW_UID}. Your docker-compose.yml already does: user: "${AIRFLOW_UID:-1000}:0"
    # So, if .env doesn't define it, this export helps if this script calls docker-compose.
}

fn_clean_pyc() {
    echo "Cleaning Python cache files from ./src..."
    find ./src -type f -name "*.py[co]" -delete
    find ./src -type d -name "__pycache__" -delete
    echo "Python cache files cleaned."
}

fn_init_airflow_db_and_user() {
    echo "--- Initializing Airflow Metadata Database and creating default user ---"
    echo "This may take a moment as Airflow connects to the database..."

    # Use the webserver container to run Airflow CLI commands.
    # --rm: Remove the container after the command exits.
    # --no-deps: Don't start linked services (they should already be up).

    # 1. Initialize/Migrate Airflow's metadata database schema.
    # For Airflow 2.x, 'db migrate' is the correct command for initial setup and schema updates.
    # It will automatically create tables if they don't exist.
    echo "Running 'airflow db migrate' to initialize/update database schema..."
    $DOCKER_COMPOSE_CMD run --rm --no-deps airflow-webserver airflow db migrate

    # 2. Create the default Airflow webserver user.
    # '|| true' allows the script to continue even if the user already exists (e.g., on subsequent runs).
    echo "Creating default Airflow user 'airflow' (password: 'airflow')..."
    $DOCKER_COMPOSE_CMD run --rm --no-deps airflow-webserver airflow users create \
        --username airflow \
        --firstname airflow \
        --lastname airflow \
        --role Admin \
        --email airflow@example.com \
        --password airflow || true

    echo "Airflow metadata database initialization and user creation completed."
    echo "-------------------------------------------------------------------"
}


# --- Main Script Logic ---

# Check for Docker Compose first
fn_check_docker_compose

# Set Airflow UID
fn_set_airflow_uid

ACTION=${1:-up} # Default to 'up' if no argument is provided

case "$ACTION" in
    up)
        echo "Building and starting all services in detached mode..."
        $DOCKER_COMPOSE_CMD build
        $DOCKER_COMPOSE_CMD up -d

        # After services are up, run the Airflow DB initialization and user creation
        # We need to wait for postgres and webserver to be ready enough to run CLI commands.
        # Docker compose's depends_on with service_healthy helps, but a small sleep can ensure readiness.
        echo "Waiting a few seconds for Airflow services to become available..."
        sleep 10 # Give Airflow containers a moment to start up and connect to DB

        fn_init_airflow_db_and_user

        echo "Services starting. It might take a few minutes for Airflow and MLflow to be fully ready."
        echo "Airflow Webserver: http://localhost:8080 (default user/pass: airflow/airflow)"
        echo "MLflow UI: http://localhost:5001"
        echo "PostgreSQL (for app): localhost:5432 (if mapped, for direct access)"
        echo "Frontend API: http://localhost:8000"
        ;;
    start)
        echo "Starting all services in detached mode..."
        $DOCKER_COMPOSE_CMD up -d
        echo "Services starting."
        ;;
    logs)
        SERVICE_NAME=${2:-""} # Optional service name
        if [ -n "$SERVICE_NAME" ]; then
            echo "Following logs for service: $SERVICE_NAME..."
            $DOCKER_COMPOSE_CMD logs -f "$SERVICE_NAME"
        else
            echo "Following logs for all services..."
            $DOCKER_COMPOSE_CMD logs -f
        fi
        ;;
    stop)
        echo "Stopping all services..."
        $DOCKER_COMPOSE_CMD stop
        echo "Services stopped."
        ;;
    down)
        echo "Stopping and removing all services, networks..."
        REMOVE_VOLUMES_FLAG=""
        if [ "$2" == "-v" ]; then
            echo "Also removing volumes."
            REMOVE_VOLUMES_FLAG="-v"
        fi
        $DOCKER_COMPOSE_CMD down $REMOVE_VOLUMES_FLAG
        echo "Services and networks removed."
        if [ "$REMOVE_VOLUMES_FLAG" == "-v" ]; then
            echo "Volumes also removed."
        fi
        ;;
    build)
        echo "Building (or rebuilding) services..."
        $DOCKER_COMPOSE_CMD build
        echo "Build complete."
        ;;
    ps)
        echo "Listing running containers for this project..."
        $DOCKER_COMPOSE_CMD ps
        ;;
    init_airflow) # Now can be run manually if needed, but 'up' handles it.
        fn_init_airflow_db_and_user
        ;;
    clean_pyc)
        fn_clean_pyc
        ;;
    *)
        fn_print_usage
        exit 1
        ;;
esac

exit 0