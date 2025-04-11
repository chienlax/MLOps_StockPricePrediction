# Docker Compose Troubleshooting

This document summarizes the key issues encountered and their resolutions while setting up the Automated Stock Market Price Prediction pipeline using Docker Compose with Airflow and MLflow.

## 1. Problem: Downloaded Data Not Appearing on Host

*   **Symptom:** The Python script running inside the Airflow container (triggered by a DAG task) successfully downloaded stock data using `yfinance`, but the `.pkl` files did not appear in the `./data/raw` directory on the host machine.
*   **Root Cause:** The script was configured (via `params.yaml`) to use relative paths (e.g., `data/raw`). When executed by Airflow, the task's Current Working Directory (CWD) was a temporary directory within the container (e.g., `/tmp/...`). Therefore, the relative path resolved to `/tmp/.../data/raw`, which is *outside* the directory mounted from the host (`./data:/opt/airflow/data:rw`). Data was written to the container's ephemeral filesystem.
*   **Solution:** Modified the `output_paths` in `config/params.yaml` to use **absolute paths** within the container that point to the mounted volume, e.g., `/opt/airflow/data/raw/{ticker}_raw.pkl`.
*   **Takeaway:** Ensure file paths used within containerized tasks correctly resolve to the intended persistent volume mounts. Using absolute paths within the container context is often clearer.

## 2. Problem: Read-Only Filesystem Error Saving Hyperparameters

*   **Symptom:** The hyperparameter optimization task failed with `OSError: [Errno 30] Read-only file system: '/opt/airflow/config/best_params.json'`.
*   **Root Cause:** The volume mount for the configuration directory in `docker-compose.yml` was set to read-only: `- ./config:/opt/airflow/config:ro`.
*   **Solution:** Changed the volume mount flag to read-write (`:rw`) in `docker-compose.yml`: `- ./config:/opt/airflow/config:rw`. Recreated the containers using `docker-compose down && docker-compose up -d --build`.
*   **Takeaway:** Explicitly check volume mount permissions (`:ro` vs `:rw`) based on whether tasks running in the container need to write back to those directories on the host.

## 3. Problem: MLflow Connection Refused (ConnectionError / Errno 111)

*   **Symptom:** The model training task failed with `ConnectionRefusedError: [Errno 111]` when trying to connect to MLflow (`mlflow.set_experiment` or initial interaction). Logs showed attempts to connect to `http://localhost:5001`.
*   **Root Cause:** The script was using `localhost:5001`.
    *   `localhost` inside a container refers to the container *itself*, not other containers or the host.
    *   `5001` is the *host* port mapped to the MLflow container's internal port `5000`. Communication *between* containers must use the **service name** (defined in `docker-compose.yml`, e.g., `mlflow-server`) and the **internal port** (`5000`).
*   **Solution:** Initially attempted to fix by ensuring the Python script used `os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')` and setting the `MLFLOW_TRACKING_URI` environment variable correctly in `docker-compose.yml`.
*   **Takeaway:** Inter-container communication in Docker Compose uses service names and internal ports, not `localhost` or host-mapped ports.

## 4. Problem: MLflow Connection Refused (Override Issue)

*   **Symptom:** The `ConnectionRefusedError` persisted, and logs *still* showed the script using `http://localhost:5001` even after the `docker-compose.yml` environment variable and Python code seemed correct. Manual `curl http://mlflow-server:5000` from within the Airflow container *succeeded*, proving network connectivity was fine.
*   **Root Cause:** An environment variable definition was overriding the one set in `docker-compose.yml`. The override was located in the Airflow DAG file itself, within the `BashOperator` definition for the training task: `env={'MLFLOW_TRACKING_URI': '{{ var.value.get("mlflow_tracking_uri", "http://localhost:5001/") }}'}`. This task-specific `env` dictionary takes precedence.
*   **Solution:** Removed the `env={'MLFLOW_TRACKING_URI': ...}` parameter entirely from the `task_train_final_model` `BashOperator` definition in the DAG file. This allowed the task to inherit the correct `MLFLOW_TRACKING_URI=http://mlflow-server:5000` set globally for the container by `docker-compose.yml`. Refreshed the DAG and restarted containers.
*   **Takeaway:** Be aware of configuration precedence: task-level `env` overrides container-level `environment` (from `docker-compose.yml`), which can be overridden by values in an `.env` file or host environment variables if using `${VAR}` substitution. Simplify by removing overrides where possible.

## 5. Problem: MLflow Artifact Logging Permission Denied

*   **Symptom:** After fixing the connection URI, the training task failed later during `mlflow.pytorch.log_model` with `PermissionError: [Errno 13] Permission denied: '/mlruns'`.
*   **Root Cause:** The MLflow client library (running in the Airflow container) was incorrectly attempting to write/create directories using the absolute path `/mlruns` on its *local* filesystem, likely because the server's configuration implicitly suggested this path. The non-root user (UID 1000) running the task inside the Airflow container does not have permission to create directories at the root (`/`) level.
*   **Solution:** Modified the `mlflow-server` service's `command` in `docker-compose.yml` to *remove* the `--default-artifact-root` argument. This forces the MLflow server to advertise artifact locations using the `mlflow-artifacts:` scheme, compelling the client library to upload artifacts via the server's REST API instead of attempting direct local filesystem writes to `/mlruns`. Recreated containers.
*   **Takeaway:** When using a remote MLflow tracking server with local file-based artifact storage (`--backend-store-uri /mlruns`), avoid specifying `--default-artifact-root` on the server command. Let the server manage artifact URIs via its API to ensure clients use the correct upload mechanism, preventing client-side permission errors related to mimicking server paths.

