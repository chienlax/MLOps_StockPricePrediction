# Project Setup Guide:


The environment includes:
*   **Airflow:** For orchestrating data pipelines (fetching, processing, training, prediction).
*   **MLflow:** For tracking experiments, models, and artifacts.
*   **PostgreSQL:** As the metadata database for Airflow.
*   **Custom Python Environment:** Containing all necessary libraries defined in `requirements.txt`.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

1.  **Git:** Required for cloning the repository. ([Download Git](https://git-scm.com/download/win))
2.  **Docker Desktop:** Install Docker Desktop for your operating system (Windows, macOS, Linux) and **ensure it is running** before proceeding.
    *   [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
    *   On Windows, the WSL 2 backend is recommended for better performance and compatibility. Docker Desktop will usually guide you through this setup.
3.  **Terminal/Command Line:** You'll need a terminal to run commands (e.g., Command Prompt, PowerShell, Git Bash on Windows; Terminal on macOS/Linux).

## Setup Steps

Follow these steps in your terminal:

1.  **Clone the Repository:**
    Clone the project repository from GitHub:
    ```bash
    git clone https://github.com/chienlax/MLOps_StockPricePrediction.git
    ```

2.  **Navigate to Project Directory:**
    Change into the project's root directory:
    ```bash
    cd MLOps_StockPricePrediction
    ```
    *(If you cloned it into a different location, adjust the path accordingly)*

3.  **Verify `.env` File (Important for Permissions):**
    *   The project includes a `.env.example` or the necessary environment variables might be documented in the README. A `.env` file is used by Docker Compose to set environment variables, particularly `AIRFLOW_UID` and `AIRFLOW_GID`, which help prevent file permission issues between your host machine and the Docker containers.
    *   **Check if `.env` exists:** If it doesn't, copy the example: `cp .env.example .env` (or create it manually if no example exists).
    *   **Default Values:** The typical default values are:
        ```dotenv
        AIRFLOW_UID=1000
        AIRFLOW_GID=0 # Or sometimes 1000
        MLFLOW_TRACKING_URI=http://mlflow-server:5000
        ```
    *   **Verify User/Group ID (Linux/macOS/WSL):** On Linux, macOS, or Windows using WSL, run the command `id` in your terminal. Find your `uid` (User ID) and `gid` (Group ID). If they are *not* `1000`, update the `AIRFLOW_UID` and `AIRFLOW_GID` values in the `.env` file accordingly. Setting `AIRFLOW_GID=0` often works as a fallback if group permissions are tricky.
    *   **Windows Users (without WSL ID check):** The default `AIRFLOW_UID=1000` and `AIRFLOW_GID=0` often work correctly with Docker Desktop for Windows using the WSL 2 backend. Try these first.
    *   **Security:** The `.env` file is listed in `.gitignore` and **should not be committed** to Git, especially if it contains sensitive keys later.

4.  **Build the Docker Image:**
    This command builds the custom Docker image defined in the `Dockerfile`. It installs all Python dependencies from `requirements.txt` into the Airflow environment. This might take a few minutes the first time.
    ```bash
    docker compose build
    ```
    *Note: If you pull changes later that modify `requirements.txt` or `Dockerfile`, you'll need to run this command again or use `docker compose up -d --build`.*

5.  **Initialize Airflow Database & User (First Time Only):**
    These commands need to be run once to set up the Airflow metadata database and create the initial admin user.
    *   **Start the Database:** Ensure the PostgreSQL service is running first.
        ```bash
        docker compose up -d postgres
        ```
        Wait about 10-15 seconds for the database to initialize.

    *   **Initialize the Database Schema:**
        ```bash
        docker compose run --rm airflow-scheduler airflow db init
        ```
        *(You should see output indicating tables are being created or migrated).*

    *   **Create Admin User:** Replace `your_secure_password` with a password you choose.
        ```bash
        docker compose run --rm airflow-scheduler airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password your_secure_password
        ```
        *Remember this username (`admin`) and password!*

6.  **Start All Services:**
    This command starts the Airflow webserver, scheduler, the MLflow server, and the PostgreSQL database (if not already running) in detached mode (`-d`), meaning they will run in the background.
    ```bash
    docker compose up -d
    ```

## Verification

Once the `docker compose up -d` command finishes successfully:

1.  **Check Running Containers:** Open a new terminal window and run:
    ```bash
    docker ps
    ```
    You should see containers running, including names similar to:
    *   `stockpred-postgres`
    *   `stockpred-mlflow-server`
    *   `stockpred-airflow-webserver`
    *   `stockpred-airflow-scheduler`
    *(If using CeleryExecutor, you might also see `stockpred-redis` and `stockpred-airflow-worker`)*

2.  **Access Airflow UI:**
    *   Open your web browser and navigate to: `http://localhost:8080`
    *   Log in using the username (`admin`) and password you created in Step 5.
    *   You should see the Airflow dashboard. You might need to unpause any DAGs listed (like `stock_data_dag`) by clicking the toggle switch next to their name.

3.  **Access MLflow UI:**
    *   Open your web browser and navigate to: `http://localhost:5001`
    *   You should see the MLflow Experiments dashboard. It will be empty initially.

## Common Commands

*   **Start Services:** (Run from the project root directory)
    ```bash
    docker compose up -d
    ```
*   **Stop Services:**
    ```bash
    docker compose down
    ```
*   **Stop Services and Remove Volumes:** (Use with caution: This deletes Airflow DB data and MLflow run data stored in Docker volumes)
    ```bash
    docker compose down -v
    ```
*   **Rebuild Image and Start:** (If `Dockerfile` or `requirements.txt` changed)
    ```bash
    docker compose up -d --build
    ```
*   **View Logs:** (To see output/errors from a specific service)
    ```bash
    docker compose logs airflow-scheduler
    docker compose logs airflow-webserver
    docker compose logs mlflow-server
    # Use -f to follow logs in real-time: docker compose logs -f airflow-scheduler
    ```
*   **Access Shell Inside a Container:** (Useful for debugging)
    ```bash
    # Access the scheduler container
    docker compose exec airflow-scheduler bash

    # Access the webserver container
    docker compose exec airflow-webserver bash
    ```

## Troubleshooting

*   **Permission Errors on Volumes (`dags`, `logs`, `data`):** This is often related to the `AIRFLOW_UID`/`AIRFLOW_GID` mismatch. Double-check Step 3 (Verify `.env` File) and ensure the IDs match your host user (`id` command on Linux/macOS/WSL). Rebuild the image (`docker compose build`) and restart (`docker compose down && docker compose up -d`) after correcting `.env`.
*   **Port Conflicts:** If `localhost:8080` or `localhost:5001` (or `5432`) are already in use by another application, Docker Compose will fail to start the service. Stop the other application or change the *host* port mapping in `docker-compose.yml` (e.g., change `"8080:8080"` to `"8081:8080"` to access Airflow on `localhost:8081`). Remember to commit the change if you modify `docker-compose.yml`.
*   **`docker compose build` Fails:** Check the error messages. Common issues include typos in `requirements.txt`, network problems preventing package downloads, or missing system dependencies (though the base Airflow image covers most common ones).
*   **Airflow/MLflow UI Not Loading:**
    *   Check `docker ps` to ensure the containers are running.
    *   Check the logs for errors: `docker compose logs airflow-webserver` or `docker compose logs mlflow-server`.
    *   Ensure Docker Desktop has sufficient resources (RAM/CPU) allocated in its settings.