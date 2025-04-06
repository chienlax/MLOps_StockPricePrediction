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

# Model Team Task List: Stock Price Prediction

**Overall Goal:** Produce robust, tested, and parameterized Python scripts (`src/**/*.py`) for data processing, feature engineering, model training, evaluation, and prediction, logging experiments and artifacts effectively with MLflow. Work primarily on the `model` Git branch.

**Prerequisites:**
- [ ] Git installed.
- [ ] Docker Desktop installed and **running**.
- [ ] An IDE like VS Code installed.
- [ ] Basic understanding of Python, Pandas, ML concepts (LSTM/ARIMA), time series, and Git.

---

## 1. Setup & Branching

- [ ] **Clone Repository:**
    ```bash
    git clone <your-repository-url>
    cd MLOps_StockPricePrediction
    ```
- [ ] **Create/Switch Branch:** Create and checkout the `model` branch.
    ```bash
    # If creating new:
    git checkout -b model
    # If branch exists remotely:
    # git checkout model
    # git pull origin model
    ```
- [ ] **Launch Docker Environment:** Start services (Airflow, MLflow, DB).
    ```bash
    docker compose up -d
    ```
- [ ] **Verify Services:** Check containers are running (`docker ps`). Wait a minute for startup.

## 2. Initial Data Exploration & Acquisition (Notebook Driven)

- [ ] **Open Project:** Open the `MLOps_StockPricePrediction` folder in VS Code.
- [ ] **Use Exploration Notebook:** Open/Create `notebooks/1_data_exploration.ipynb`.
- [ ] **Fetch Initial Data:** In the notebook, use `yfinance` to fetch data for a sample ticker (e.g., 'AAPL', 5-10 years).
    ```python
    # Example imports/fetch
    import yfinance as yf
    import pandas as pd
    ticker = 'AAPL'
    data_raw = yf.download(ticker, start="2015-01-01", end="2023-12-31")
    ```
- [ ] **Explore Raw Data:** In the notebook, check `data_raw.head()`, `data_raw.info()`, visualize closing price, check for NaNs.
- [ ] **Save Raw Data:** Manually save the fetched DataFrame to `data/raw/AAPL_raw.csv`.
- [ ] **Track with DVC:** Use DVC *from your host machine's terminal* (CMD/PowerShell/Bash, **not** inside Docker) to track the raw data file.
    ```bash
    # In project root on your host machine:
    dvc add data/raw/AAPL_raw.csv
    ```
- [ ] **Commit DVC Pointer:** Add and commit the `.dvc` file and `.gitignore` changes.
    ```bash
    git add data/raw/AAPL_raw.csv.dvc data/.gitignore
    git commit -m "feat(data): Add initial raw data for AAPL"
    ```
- [ ] **Push Changes:**
    ```bash
    git push origin model
    ```

## 3. Develop Data Processing Logic (Script Focus)

- [ ] **Implement Cleaning:** Based on notebook findings, add cleaning functions (handle NaNs, types, etc.) to `src/data/make_dataset.py`.
- [ ] **Test Processing:** Import and test functions (e.g., in the notebook or a temporary script) using `data/raw/AAPL_raw.csv`.
- [ ] **Save Processed Data:** Generate and save the processed output to `data/processed/AAPL_processed.csv`.
- [ ] **Track Processed Data:** Use DVC *on your host machine* to track the processed file.
    ```bash
    # In project root on your host machine:
    dvc add data/processed/AAPL_processed.csv
    ```
- [ ] **Commit DVC Pointer:** Add and commit the `.dvc` file.
    ```bash
    git add data/processed/AAPL_processed.csv.dvc
    git commit -m "feat(data): Implement data processing logic and add processed AAPL data"
    ```
- [ ] **Push Changes:**
    ```bash
    git push origin model
    ```

## 4. Develop Feature Engineering Logic (Script Focus)

- [ ] **Implement Features:** Define feature engineering logic (lags, rolling stats, date features, **target variables** like `AdjClose_lead_N`) in `src/features/build_features.py`. **Ensure no lookahead bias in target!**
- [ ] **Implement Splitting:** Add time-series train/validation/test split logic (date-based) in the script or a utility.
- [ ] **Parameterize:** Use `config/params.yaml` for configurable parameters (lag lengths, windows, etc.). Load this config in the script.
- [ ] **Test Feature Gen:** Run the script to generate feature files (e.g., `data/features/AAPL_features_train.csv`, `_val.csv`, `_test.csv`).
- [ ] **Track Feature Data (Optional):** Decide with team if features need DVC tracking. If yes:
    ```bash
    # In project root on your host machine:
    dvc add data/features/
    git add data/features/*.dvc data/features/.gitignore
    git commit -m "feat(features): Implement feature engineering and add feature sets for AAPL"
    # If not tracking: commit only the script and config changes
    # git add src/features/build_features.py config/params.yaml
    # git commit -m "feat(features): Implement feature engineering script"
    ```
- [ ] **Push Changes:**
    ```bash
    git push origin model
    ```

## 5. Model Prototyping & MLflow Integration (Notebook Driven)

- [ ] **Use Prototyping Notebook:** Open/Create `notebooks/2_model_prototyping.ipynb`.
- [ ] **Load Feature Data:** Load the generated train/validation features (`data/features/...`).
- [ ] **Configure MLflow:** Set the MLflow tracking URI to point to the Docker service.
    ```python
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5001") # Use host port
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    ```
- [ ] **Implement Basic Model:** Define a simple LSTM or ARIMA structure in the notebook.
- [ ] **Wrap Training in MLflow:** Use `with mlflow.start_run():` to encapsulate a basic training loop.
- [ ] **Log Parameters:** Inside the run, use `mlflow.log_param()` for model type, horizon, hyperparameters.
- [ ] **Train & Evaluate:** Train the model and evaluate on the validation set.
- [ ] **Log Metrics:** Use `mlflow.log_metric()` for validation metrics (MAE, RMSE).
- [ ] **(Optional) Log Plot:** Use `mlflow.log_figure()` to save a prediction plot.
- [ ] **(Optional) Log Model:** Experiment with `mlflow.<framework>.log_model()`.
- [ ] **Verify in MLflow UI:** Run the notebook cells and check `http://localhost:5001` to see the logged run, parameters, and metrics. Experiment with changes.
- [ ] **Commit Notebook:** Commit useful notebook experiments.
    ```bash
    git add notebooks/2_model_prototyping.ipynb
    git commit -m "exp(model): Prototype LSTM with MLflow tracking"
    ```
- [ ] **Push Changes:**
    ```bash
    git push origin model
    ```

## 6. Develop Training Script (`src/models/train_model.py`)

- [ ] **Refactor Logic:** Move successful prototyping logic into `src/models/train_model.py`.
- [ ] **Load Config:** Make the script load parameters from `config/params.yaml` (model type, hyperparameters, horizons `[1, 3, 7]`).
- [ ] **Load Data:** Use functions from `src/data/make_dataset.py` and `src/features/build_features.py` to load data.
- [ ] **Set MLflow URI:** Ensure the script sets the tracking URI (can use `python-dotenv` or rely on Docker env var `MLFLOW_TRACKING_URI`).
- [ ] **Loop Through Horizons:** Structure the script to train models for each specified horizon (1d, 3d, 7d).
- [ ] **Use `mlflow.start_run()`:** Wrap training logic for each horizon.
- [ ] **Log Run Details:** Inside the run:
    - [ ] Log relevant parameters (`mlflow.log_param()`).
    - [ ] Log horizon-specific metrics (`mlflow.log_metric()` with names like `val_mae_1d`).
    - [ ] **Log Model Artifact:** Use `mlflow.<framework>.log_model()` to save the trained model.
    - [ ] Log relevant plots (`mlflow.log_figure()`).
- [ ] **Execute Training Script:** Run the script *within the Docker environment*.
    ```bash
    # Example command (run from host terminal):
    docker compose exec airflow-scheduler python /opt/airflow/src/models/train_model.py
    ```
- [ ] **Verify MLflow Runs:** Check the MLflow UI for correctly logged runs, parameters, metrics, and model artifacts for all horizons.
- [ ] **Commit Training Script:** Add and commit the script and any config changes.
    ```bash
    git add src/models/train_model.py config/params.yaml
    git commit -m "feat(model): Implement automated training script with MLflow logging for multiple horizons"
    ```
- [ ] **Push Changes:**
    ```bash
    git push origin model
    ```

## 7. Develop Prediction Script (`src/models/predict_model.py`)

- [ ] **Create Prediction Script:** Create `src/models/predict_model.py`.
- [ ] **Define Inputs:** Script should accept arguments (e.g., ticker, MLflow run ID or model URI `models:/<name>/<version>`, num recent data points). Use `argparse`.
- [ ] **Connect to MLflow:** Set the tracking URI.
- [ ] **Load Model:** Use `mlflow.<framework>.load_model(model_uri)` to load the specified model from MLflow.
- [ ] **Fetch Latest Data:** Use `yfinance` to get the most recent data needed for features.
- [ ] **Apply Transformations:** Use the *exact same* functions from `make_dataset.py` and `build_features.py` to process/feature engineer the fetched data.
- [ ] **Generate Predictions:** Use the loaded model to predict for required horizons (1d, 3d, 7d).
- [ ] **Format Output:** Return predictions in a clear format (dict, JSON).
- [ ] **Test Prediction Script:** Run the script *within Docker*, providing a valid model URI from a previous MLflow run.
    ```bash
    # Example command (run from host terminal):
    docker compose exec airflow-scheduler python /opt/airflow/src/models/predict_model.py --ticker AAPL --model-uri runs:/<some_mlflow_run_id>/model_3d
    # Or using registered model:
    # docker compose exec airflow-scheduler python /opt/airflow/src/models/predict_model.py --ticker AAPL --model-uri models:/StockPredictor_AAPL_3d/Production
    ```
- [ ] **Commit Prediction Script:** Add and commit the script.
    ```bash
    git add src/models/predict_model.py
    git commit -m "feat(model): Implement prediction script loading models from MLflow"
    ```
- [ ] **Push Changes:**
    ```bash
    git push origin model
    ```

## 8. Testing & Refinement

- [ ] **Add Unit Tests:** Implement basic unit tests using `pytest` in the `tests/` directory for critical data and feature functions.
- [ ] **Run Tests:** Execute tests (e.g., within Docker).
    ```bash
    # Example command (run from host terminal):
    # docker compose exec airflow-scheduler pytest /opt/airflow/tests/
    ```
- [ ] **Analyze MLflow Runs:** Review metrics and plots in the MLflow UI.
- [ ] **Iterate:** Adjust parameters in `config/params.yaml`, rerun training (`docker compose exec ... train_model.py`), compare runs in MLflow, and refine models/features.
- [ ] **Commit Tests & Refinements:** Commit tests and any code improvements.
    ```bash
    git add tests/ src/ config/params.yaml
    git commit -m "test(data): Add unit tests for processing"
    git commit -m "refactor(model): Improve LSTM layer config based on experiments"
    ```
- [ ] **Push Changes:**
    ```bash
    git push origin model
    ```

## 9. Handoff Preparation

- [ ] **Ensure Code Quality:** Code in `src/` is clean, commented, and follows project standards.
- [ ] **Verify MLflow Logging:** Training script logs all necessary parameters, metrics, and model artifacts consistently.
- [ ] **Confirm Prediction Script:** Prediction script works correctly with models loaded from MLflow.
- [ ] **Update `params.yaml`:** Ensure the config file reflects the best parameters found.
- [ ] **Communicate with Airflow Team:** Inform the Airflow team that the core modeling scripts on the `model` branch are ready for integration into DAGs.

---
*Remember to `git pull origin model` periodically to get updates from teammates if multiple people are working on the branch, and resolve any merge conflicts.*
*Push your changes frequently (`git push origin model`)!*

# Airflow Team Task List: Orchestrating Stock Prediction Pipeline

**Overall Goal:** Automate the execution of the Model Team's Python scripts (`src/**/*.py`) using Airflow DAGs within the Docker environment. Ensure reliable scheduling, dependency management, and correct interaction with MLflow.

**Prerequisites:**
- [ ] Git installed.
- [ ] Docker Desktop installed and **running**.
- [ ] An IDE like VS Code installed.
- [ ] Understanding of Airflow concepts (DAG, Task, Operator, Sensor, Scheduling).
- [ ] Basic Bash command knowledge.
- [ ] Access to the shared Git repository with the Model Team's completed work (scripts, config).

---

## 1. Setup & Branching

- [ ] **Clone Repository:** (If not already done)
    ```bash
    git clone <your-repository-url>
    cd MLOps_StockPricePrediction
    ```
- [ ] **Get Latest Code & Branch:** Ensure you have the latest integrated code (e.g., from `develop` or `main` which includes merged `model` branch changes) and create/checkout an `airflow` branch.
    ```bash
    # Example: Create 'airflow' branch from latest 'develop'
    git checkout develop
    git pull origin develop
    git checkout -b airflow
    # Or checkout existing airflow branch and update
    # git checkout airflow
    # git pull origin airflow
    # git merge origin/develop # If needed
    ```
- [ ] **Launch Docker Environment:** Start Airflow, MLflow, etc.
    ```bash
    docker compose up -d
    ```
- [ ] **Verify UIs:** Check Airflow (`http://localhost:8080`) and MLflow (`http://localhost:5001`) are accessible.

## 2. Understand the Scripts & Configuration

- [ ] **Review Data Script:** Analyze `src/data/make_dataset.py`. Understand its inputs/outputs and how it's triggered (arguments? sequential?).
- [ ] **Review Feature Script:** Analyze `src/features/build_features.py`. Understand inputs (processed data) and outputs (feature sets).
- [ ] **Review Training Script:** Analyze `src/models/train_model.py`. Confirm it reads `config/params.yaml` and logs correctly to MLflow (`http://mlflow-server:5000`).
- [ ] **Review Prediction Script:** Analyze `src/models/predict_model.py`. Understand how it loads models (run ID? registered name?) and generates predictions.
- [ ] **Review Config:** Examine `config/params.yaml` for tunable parameters.

## 3. Design the Airflow DAGs

- [ ] **Decide DAG Structure:** Plan the split of work into DAGs. (Recommended: `stock_data_dag.py` for fetch/process, `stock_train_predict_dag.py` for ML steps).
- [ ] **Choose Operators:** Plan to primarily use `BashOperator` to execute the Python scripts.
- [ ] **Note Paths:** Remember scripts will run inside Docker, using paths like `/opt/airflow/src/...`, `/opt/airflow/data/...`.

## 4. Implement the Data Pipeline DAG (`airflow/dags/stock_data_dag.py`)

- [ ] **Edit DAG File:** Open/Create `airflow/dags/stock_data_dag.py`.
- [ ] **Add Imports:** Include `pendulum`, `DAG`, `BashOperator`.
- [ ] **Define Default Args:** Set `owner`, `retries`, etc.
- [ ] **Instantiate DAG:** Define `dag_id='stock_data_pipeline'`, description, `schedule='0 6 * * *'` (adjust as needed), `start_date`, `catchup=False`, `tags`.
- [ ] **Define Paths:** Specify container paths for scripts, data dirs, project root.
- [ ] **Create Fetch/Process Task:** Use `BashOperator` (task_id `fetch_process_stock_data`) to run `python /opt/airflow/src/data/make_dataset.py`. Add `doc_md`.
- [ ] **(Optional) Plan DVC/Git Tasks:** *Initially comment out* `BashOperator` tasks for `dvc add` and `git commit/push`. Verify data generation manually first. Automation requires careful setup (Git credentials in container).
- [ ] **Define Task Dependencies:** Set the order (e.g., `fetch_process_data`).
- [ ] **Save DAG File:** Save the changes.
- [ ] **Test Data DAG:**
    - [ ] Check Airflow UI (`localhost:8080`) for the `stock_data_pipeline` DAG (allow time for detection).
    - [ ] Unpause the DAG (toggle switch).
    - [ ] Manually trigger the DAG (Play button).
    - [ ] Monitor the run in Graph/Grid view. Check task logs.
    - [ ] Verify data files appear in local `data/raw` and `data/processed` folders.
- [ ] **Commit Data DAG:**
    ```bash
    git add airflow/dags/stock_data_dag.py
    git commit -m "feat(airflow): Create DAG for daily data pipeline"
    git push origin airflow # Or your working branch name
    ```

## 5. Implement the Training & Prediction DAG (`airflow/dags/stock_train_predict_dag.py`)

- [ ] **Edit DAG File:** Open/Create `airflow/dags/stock_train_predict_dag.py`.
- [ ] **Add Imports:** Include `pendulum`, `DAG`, `BashOperator`, `ExternalTaskSensor`.
- [ ] **Define Default Args:** Similar to the data DAG.
- [ ] **Instantiate DAG:** Define `dag_id='stock_train_predict_pipeline'`, description, `schedule='0 8 * * 5'` (or `None`, adjust), `start_date`, `catchup=False`, `tags`.
- [ ] **Define Paths:** Specify container paths for scripts, predictions dir, project root.
- [ ] **Create Sensor Task:** Use `ExternalTaskSensor` (task_id `wait_for_stock_data_pipeline`) to wait for the `stock_data_pipeline`'s success (use correct `external_task_id` - likely `fetch_process_stock_data` initially).
- [ ] **Create Feature Task:** Use `BashOperator` (task_id `run_feature_engineering`) to run `python /opt/airflow/src/features/build_features.py`.
- [ ] **Create Training Task:** Use `BashOperator` (task_id `run_model_training`) to run `python /opt/airflow/src/models/train_model.py`. Explicitly set `env={'MLFLOW_TRACKING_URI': 'http://mlflow-server:5000'}`. Add `doc_md`.
- [ ] **Create Prediction Task:** Use `BashOperator` (task_id `generate_predictions`) to run `python /opt/airflow/src/models/predict_model.py`. Pass necessary args if needed (e.g., model URI). Set `env` for MLflow connection. Add `doc_md`. Decide on model loading strategy (latest? registered? -- discuss with Model Team).
- [ ] **(Optional) Plan DVC/Git Tasks:** *Initially comment out* tasks for DVC/Git tracking of predictions.
- [ ] **Define Task Dependencies:** Set order: `wait_for_data >> feature_engineering >> train_model >> generate_predictions`.
- [ ] **Save DAG File:** Save the changes.
- [ ] **Test Train/Predict DAG:**
    - [ ] Ensure `stock_data_pipeline` has succeeded recently (or run it).
    - [ ] Check Airflow UI for `stock_train_predict_pipeline`. Unpause it.
    - [ ] Manually trigger the DAG.
    - [ ] Monitor the run and check task logs.
    - [ ] **Verify MLflow:** Check MLflow UI (`localhost:5001`) for new runs created by `run_model_training`, including parameters, metrics, and **model artifacts**.
    - [ ] Verify output files appear in local `data/features` and `data/predictions`.
- [ ] **Commit Train/Predict DAG:**
    ```bash
    git add airflow/dags/stock_train_predict_dag.py
    git commit -m "feat(airflow): Create DAG for training and prediction pipeline"
    git push origin airflow # Or your working branch name
    ```

## 6. Refinement & Parameterization

- [ ] **Use Airflow Macros:** If scripts need dynamic arguments (like dates), investigate using macros (e.g., `{{ ds }}`) within `bash_command` strings.
- [ ] **Handle Secrets:** If API keys or other secrets are needed, explore using Airflow Connections/Variables instead of config files. Update scripts if necessary.
- [ ] **Finalize Model Loading:** Confirm the strategy for loading models in the prediction task (e.g., using a registered model alias managed via Airflow Variables or `params.yaml`) and implement it reliably.
- [ ] **Consider DVC/Git Automation:** If desired, implement and test the DVC/Git `BashOperator` tasks carefully, ensuring Git credentials are securely managed within the Airflow environment (e.g., via SSH keys mounted as secrets).

## 7. Documentation & Handoff

- [ ] **Update README:** Add a section explaining the Airflow DAGs, their schedules, purpose, and how to use the Airflow UI to monitor/trigger them.
- [ ] **Document Setup:** Note any required Airflow Variables or Connections.
- [ ] **Prepare for Merge:** Ensure the `airflow` branch is stable and ready to be merged into the main integration branch (`develop` or `main`). Communicate with the team.

---
*Remember to coordinate with the Model Team, especially regarding script arguments and the model loading strategy for predictions.*
*Pull changes from the integration branch (`develop`/`main`) frequently into your `airflow` branch (`git pull origin develop`) to stay up-to-date and resolve conflicts early.*