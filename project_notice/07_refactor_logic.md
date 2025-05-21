**Detailed Logic Flow of the Current Implementation (Phases 1 & 2):**

Two main conceptual pipelines that will eventually be orchestrated by Airflow.

**Pipeline A: The Model Training/Retraining Workflow**
*(This pipeline is run when a new model needs to be trained, either manually triggered or automatically by the monitoring process).*

1.  **Data Ingestion & Full Processing (`src/data/make_dataset.py --mode full_process`):**
    *   **Input:** Configuration from `params.yaml`.
    *   **Action 1 (Raw Data Update):** The `load_data()` function is called. It checks the `raw_stock_data` table in PostgreSQL for the latest entry for each ticker. It then fetches only *newer* data from Yahoo Finance and updates/inserts it into `raw_stock_data`.
    *   **Action 2 (Full Feature Engineering):**
        *   *All* relevant raw data is loaded from the `raw_stock_data` table.
        *   The full sequence of feature engineering steps is applied (technical indicators via `add_technical_indicators()`, preprocessing via `preprocess_data()`, alignment into 3D arrays via `align_and_process_data()`).
        *   A **new unique `DATASET_RUN_ID`** (e.g., `20231105_153000`) is generated.
    *   **Output:**
        *   The engineered features (NumPy arrays for data and targets, list of feature columns, list of tickers) are saved to the `processed_feature_data` table in PostgreSQL, tagged with the new `DATASET_RUN_ID`.
        *   This `DATASET_RUN_ID` is printed to standard output (for Airflow to capture via XComs).

2.  **Feature Scaling & Sequencing (`src/features/build_features.py`):**
    *   **Input:** `params.yaml`, and the `DATASET_RUN_ID` (passed as `--run_id` from the previous step).
    *   **Action:**
        *   Loads the specific processed dataset from `processed_feature_data` using the provided `DATASET_RUN_ID`.
        *   Creates time-series sequences (`create_sequences()`).
        *   Splits into training and testing sets.
        *   Scales the data using `MinMaxScaler` (`scale_data()`).
    *   **Output:**
        *   Scaled data arrays (`X_train_scaled`, `y_train_scaled`, etc.) are saved to the `scaled_feature_sets` table in PostgreSQL, tagged with the *same* `DATASET_RUN_ID`.
        *   The fitted scalers (`scalers_x`, `y_scalers`) are saved to the `scalers` table in PostgreSQL, also tagged with the `DATASET_RUN_ID`.

3.  **Hyperparameter Optimization (`src/models/optimize_hyperparams.py`):**
    *   **Input:** `params.yaml`, and the `DATASET_RUN_ID` (passed as `--run_id`).
    *   **Action:**
        *   Loads the specific scaled training/testing data from `scaled_feature_sets` (using `DATASET_RUN_ID`).
        *   Loads the corresponding `y_scalers` from the `scalers` table (using `DATASET_RUN_ID`).
        *   Runs an Optuna study to find the best hyperparameters for the chosen model architecture(s).
    *   **Output:**
        *   The best hyperparameters (as a JSON string) are saved to the `optimization_results` table in PostgreSQL, tagged with the `DATASET_RUN_ID`.

4.  **Final Model Training & MLflow Logging (`src/models/train_model.py`):**
    *   **Input:** `params.yaml`, and the `DATASET_RUN_ID` (passed as `--run_id`).
    *   **Action:**
        *   Loads the specific scaled training/testing data (`scaled_feature_sets`), `y_scalers` (`scalers`), and best hyperparameters (`optimization_results`), all using the `DATASET_RUN_ID`.
        *   Trains the final model.
        *   **MLflow Interaction:**
            *   An MLflow run is started under the configured experiment name.
            *   The `DATASET_RUN_ID` is logged as a parameter to this MLflow run (e.g., `mlflow.log_param("dataset_run_id", DATASET_RUN_ID)`). This creates the crucial link between the trained model and the exact version of data it used.
            *   Hyperparameters, training/evaluation metrics, and the trained PyTorch model artifact are logged to this MLflow run.
            *   The trained model is **registered** in the MLflow Model Registry (e.g., under a name like "StockPricePredictorLSTM"). Initially, it might be in a "Staging" or "None" stage.
    *   **Output:**
        *   The **MLflow Run ID of this specific training execution** is returned/printed (for Airflow).
        *   (Optional) Predictions made on its internal test set are saved to the `latest_predictions` table in PostgreSQL, tagged with this *model's MLflow Run ID*.

---

**Pipeline B: The Daily Operational Workflow**
*(This pipeline is scheduled to run daily, e.g., after market close).*

1.  **Incremental Raw Data Ingestion (Part of `src/data/make_dataset.py --mode incremental_fetch`):**
    *   **Input:** `params.yaml`.
    *   **Action:** The `load_data()` function checks `raw_stock_data` for the latest data for each ticker and fetches only newer data from Yahoo Finance, updating the `raw_stock_data` table.
    *   **Output:** Updated `raw_stock_data` table in PostgreSQL.

2.  **Get Production Model Information (Conceptual Airflow Task - preparation for next script):**
    *   **Action:** An Airflow task (e.g., a `PythonOperator`) will query the MLflow Model Registry.
    *   It finds the model currently in the "Production" stage (e.g., `models:/StockPricePredictorLSTM/Production`).
    *   It extracts two key pieces of information from this production model's metadata:
        1.  Its **MLflow Model URI** (e.g., `models:/StockPricePredictorLSTM/Production` or a specific version URI like `models:/StockPricePredictorLSTM/5`).
        2.  The **`dataset_run_id`** that was logged as a parameter when *this specific production model version was trained*. (This is why logging `dataset_run_id` in `train_model.py` was vital).
    *   **Output (via Airflow XComs):** The Production Model URI and its associated `dataset_run_id`.

3.  **Prepare Input for Daily Prediction (`src/features/prepare_prediction_input.py`):**
    *   **Input:** `params.yaml`, and the `dataset_run_id` of the *current production model* (from Airflow XComs, passed as `--production_model_training_run_id`).
    *   **Action:**
        *   Loads the `scalers_x` (feature scalers) and `feature_columns` list that were saved under the production model's `dataset_run_id` from the PostgreSQL `scalers` and `processed_feature_data` tables.
        *   Fetches the latest small window of raw data from `raw_stock_data` (using `get_latest_raw_data_window()`).
        *   Applies the *same* technical indicators (`add_technical_indicators()`).
        *   Selects the *same* `feature_columns`.
        *   Takes the most recent `sequence_length` records.
        *   Scales these records using the loaded `scalers_x`.
    *   **Output:** A NumPy array `(1, sequence_length, num_stocks, num_features)` containing the scaled input sequence is saved to a temporary file (e.g., `/tmp/prediction_input.npy`). The absolute path to this file is printed to `stdout` (for Airflow XComs).

4.  **Make Daily Predictions (`src/models/predict_model.py` - refactored):**
    *   **Input:** `params.yaml`, the path to the temporary input sequence file (from Airflow XComs, passed as `--input_sequence_path`), and the Production Model URI (from Airflow XComs, passed as `--production_model_uri`).
    *   **Action:**
        *   Loads the "Production" PyTorch model from the MLflow Model Registry using the provided URI.
        *   Extracts the `dataset_run_id` associated with this loaded production model from its MLflow parameters.
        *   Loads the corresponding `y_scalers` (target variable scalers) from the PostgreSQL `scalers` table using this `dataset_run_id`.
        *   Loads the prepared input sequence from the `.npy` file.
        *   Makes predictions.
        *   Inverse-transforms the scaled predictions using the loaded `y_scalers`.
    *   **Output:**
        *   Predictions are saved to `data/predictions/latest_predictions.json` and `data/predictions/historical/{today_date}.json` (for the API).
        *   Predictions are saved to the `latest_predictions` table in PostgreSQL, tagged with the **MLflow Run ID of the production model** that made them.

5.  **Monitor Model Performance (`src/models/monitor_performance.py`):**
    *   **Input:** `params.yaml`.
    *   **Action:**
        *   Determines the `prediction_date_to_evaluate` (e.g., yesterday, based on `evaluation_lag_days`).
        *   Fetches the predictions made *for* that `prediction_date_to_evaluate` (e.g., from the historical JSON file or by querying the DB if `latest_predictions` table stores target prediction date). It also gets the `model_mlflow_run_id` associated with these predictions.
        *   Fetches the actual closing prices for `prediction_date_to_evaluate` (and the day before for directional accuracy) from the `raw_stock_data` table.
        *   Calculates performance metrics (MAE, MAPE, directional accuracy, etc.).
        *   Saves these metrics to the `model_performance_log` table in PostgreSQL, tagged with the `model_mlflow_run_id` of the model being evaluated.
        *   Compares calculated metrics against the `performance_thresholds` defined in `params.yaml`.
    *   **Output:** Prints a specific string to `stdout` (e.g., `trigger_retraining_pipeline_task` or `no_retraining_needed_task`). Airflow's `BranchPythonOperator` will use this output to decide the next step.

6.  **Branching Logic (Conceptual Airflow Task):**
    *   If `monitor_performance.py` indicates performance is below thresholds, Airflow triggers **Pipeline A (Model Training/Retraining Workflow)**.
    *   Otherwise, the daily operational workflow ends.

---

**How the Pipelines Connect & The "Loop":**

*   The Daily Operational Workflow runs, makes predictions, and monitors.
*   If monitoring detects poor performance, it triggers the Model Training/Retraining Workflow.
*   The Retraining Workflow produces a new candidate model, logs it to MLflow, and (if a separate evaluation/promotion step is added to this workflow, which is good practice) potentially promotes this new, better model to the "Production" stage in MLflow Model Registry.
*   The *next time* the Daily Operational Workflow runs, its "Get Production Model Information" task will pick up this newly promoted "Production" model, and the cycle continues with the updated model.

This structure creates a feedback loop where the system can automatically adapt to changing market conditions or model degradation by retraining and deploying better models. The `dataset_run_id` ensures data lineage for training, and MLflow handles model lineage and deployment staging. PostgreSQL serves as the central hub for versioned data artifacts and operational logs.