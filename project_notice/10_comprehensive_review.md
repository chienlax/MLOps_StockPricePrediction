A high-level overview:

1.  **`dag_daily_stock_operations.py` (Daily DAG):**
    *   **Purpose:** Runs on a schedule (e.g., daily) to perform routine tasks:
        *   Fetch the latest incremental stock data.
        *   Get the current production model details from MLflow.
        *   Prepare input data for prediction based on the production model's characteristics.
        *   Make predictions using the production model.
        *   Monitor the performance of these predictions against actuals (from a previous day).
        *   If performance degrades below thresholds, trigger the retraining DAG.
    *   **Trigger for Retraining DAG:** Performance degradation.

2.  **`dag_stock_model_retraining.py` (Retraining DAG):**
    *   **Purpose:** Runs when triggered (manually or by the Daily DAG) to retrain the model:
        *   Process all available historical data to create a full dataset.
        *   Build features and scale them.
        *   Optimize hyperparameters for the model.
        *   Train a new candidate model using the best hyperparameters.
        *   Evaluate the candidate model against the current production model.
        *   If the candidate is better, promote it to become the new production model in MLflow.
    *   **Output:** A potentially new production model.

---

Key Scripts and Their Roles
---------------------------

*   **`src/utils/db_utils.py`**: Contains all functions for interacting with the PostgreSQL database (setup, saving/loading raw data, processed features, scaled data, scalers, predictions, performance metrics, optimization results).
*   **`src/data/make_dataset.py`**:
    *   `--mode incremental_fetch`: Fetches only new raw stock data from Yahoo Finance since the last DB entry and saves it to `raw_stock_data` table.
    *   `--mode full_process`:
        1.  Ensures raw data is up-to-date (calls `load_data` which itself can be incremental).
        2.  Loads *all* raw data from DB.
        3.  Adds technical indicators.
        4.  Preprocesses data (creates 'Target' variable).
        5.  Aligns data across tickers and converts to NumPy arrays.
        6.  Saves these (`processed_data`, `targets`, `feature_columns`, `tickers`) to `processed_feature_data` table with a unique `run_id`.
*   **`src/features/build_features.py`**:
    1.  Loads processed data (NumPy arrays, feature columns, tickers) for a given `run_id` from `processed_feature_data` table.
    2.  Splits into train/test sets.
    3.  Creates sequences (e.g., 30-day input, 1-day target).
    4.  Scales the sequenced data using `MinMaxScaler` (fits on train only).
    5.  Saves scaled data (`X_train_scaled`, `y_train_scaled`, etc.) to `scaled_feature_sets` table.
    6.  Saves the fitted scalers (`scalers_x`, `y_scalers`) to `scalers` table, linked by `run_id`.
*   **`src/features/prepare_prediction_input.py`**:
    1.  Loads `scalers_x` and `feature_columns` from DB using the `production_model_training_run_id` (the dataset `run_id` the production model was trained on).
    2.  Fetches a recent window of raw data from DB.
    3.  Applies technical indicators.
    4.  Preprocesses and selects relevant features.
    5.  Extracts the latest `sequence_length` timesteps.
    6.  Scales this sequence using the loaded `scalers_x`.
    7.  Saves the resulting NumPy array (prediction input) to a temporary `.npy` file.
*   **`src/models/optimize_hyperparams.py`**:
    1.  Loads scaled train/test data and `y_scalers` for a given `run_id` from DB.
    2.  Uses Optuna to find the best hyperparameters (batch_size, hidden_size, num_layers, dropout, learning_rate, model_type) by training and evaluating models for `n_trials`.
    3.  Saves the best hyperparameters found to `optimization_results` table, linked by `run_id`.
*   **`src/models/train_model.py`**:
    1.  Loads scaled train/test data, `y_scalers`, `tickers`, and best hyperparameters for a given `run_id` from DB.
    2.  Trains a model (type defined by best_params) using these settings.
    3.  Logs the trained model, parameters (including the `dataset_run_id` it was trained on), and evaluation metrics to MLflow.
    4.  Registers the model in MLflow Model Registry with `experiment_name` (from `params.yaml`).
    5.  **Crucially, it also sets the "Production" alias for this newly trained model if no "Production" version exists or if this one is deemed better (this internal promotion logic might be redundant if the DAG handles promotion separately, which it does).**
    6.  Saves some test predictions to the `latest_predictions` DB table (for logging/analysis of this training run).
*   **`src/models/predict_model.py`**:
    1.  Loads a specified production model from MLflow (using URI like `models:/<name>@Production`).
    2.  Retrieves the `dataset_run_id` (that the prod model was trained on) from the model's MLflow run parameters.
    3.  Loads `y_scalers` and `tickers` from DB using this `dataset_run_id`.
    4.  Loads the prepared input sequence (`.npy` file).
    5.  Makes predictions.
    6.  Inverse transforms predictions.
    7.  Saves predictions to JSON files (for API) and to the `latest_predictions` DB table (crucially, with `target_prediction_date`).
*   **`src/models/monitor_performance.py`**:
    1.  Evaluates predictions for a past date (e.g., yesterday) against actuals.
    2.  Fetches predictions for `target_prediction_date` from `latest_predictions` DB table.
    3.  Fetches actual closing prices for `target_prediction_date` (and the day before for directional accuracy) from `raw_stock_data` DB table.
    4.  Calculates MAPE, directional accuracy, etc.
    5.  Saves these metrics to `model_performance_log` DB table.
    6.  If metrics are below thresholds (from `params.yaml`), it signals to trigger retraining.
*   **`src/models/model_definitions.py`**: Defines the PyTorch `nn.Module` classes (`StockLSTM`, `StockLSTMWithAttention`, `StockLSTMWithCrossStockAttention`).
*   **`src/models/evaluate_model.py`**: Contains functions to evaluate a trained model on a test set and visualize predictions.
*   **`src/api/main.py`**: FastAPI application that serves:
    *   A dashboard (`index.html` with `static/js/app.js` and `static/css/styles.css`).
    *   `/predictions/latest_for_table`: Latest prediction for each ticker (from DB).
    *   `/tickers`: List of available tickers (from DB).
    *   `/historical_context_chart/{ticker}`: Data for plotting historical actuals alongside the latest prediction for a specific ticker (from DB).

---

Database Tables Involved
------------------------

*   **`raw_stock_data`**: Stores raw historical stock prices (Open, High, Low, Close, Volume, etc.) fetched from Yahoo Finance.
    *   *Written by:* `make_dataset.py` (both modes).
    *   *Read by:* `make_dataset.py` (full_process mode), `prepare_prediction_input.py`, `monitor_performance.py`, `api/main.py`.
*   **`processed_feature_data`**: Stores processed NumPy arrays (`processed_data`, `targets`), list of `feature_columns`, and `tickers` for a specific data processing `run_id`.
    *   *Written by:* `make_dataset.py` (full_process mode).
    *   *Read by:* `build_features.py`, `prepare_prediction_input.py` (to get `tickers`/`feature_columns` if not in scalers).
*   **`scaled_feature_sets`**: Stores scaled NumPy arrays (`X_train_scaled`, `y_train_scaled`, `X_test_scaled`, `y_test_scaled`) for a `run_id`.
    *   *Written by:* `build_features.py`.
    *   *Read by:* `optimize_hyperparams.py`, `train_model.py`.
*   **`scalers`**: Stores pickled scaler objects (`scalers_x`, `y_scalers`, `tickers`, `num_features`) for a `run_id`.
    *   *Written by:* `build_features.py`.
    *   *Read by:* `prepare_prediction_input.py`, `optimize_hyperparams.py`, `train_model.py`, `predict_model.py`.
*   **`optimization_results`**: Stores the best hyperparameters found by Optuna for a `run_id`.
    *   *Written by:* `optimize_hyperparams.py`.
    *   *Read by:* `train_model.py`.
*   **`latest_predictions`**: Stores predictions made by models. Key fields: `target_prediction_date`, `ticker`, `predicted_price`, `model_mlflow_run_id`.
    *   *Written by:* `predict_model.py` (operational daily predictions), `train_model.py` (test set predictions from a training run).
    *   *Read by:* `monitor_performance.py`, `api/main.py`.
*   **`model_performance_log`**: Stores daily performance metrics (MAPE, MAE, etc.) for each ticker, comparing predictions against actuals.
    *   *Written by:* `monitor_performance.py`.
    *   *Read by:* Potentially future analysis/dashboarding tasks (not explicitly read by current scripts for operational logic).

---

DAG 1: `daily_stock_operations_prod` (`dag_daily_stock_operations.py`)
--------------------------------------------------------------------

**General Workflow Summary:**
This DAG is the primary operational pipeline. It starts by ensuring the database schema is correct. Then, it fetches the latest stock data. With the new data and the current production model (retrieved from MLflow), it prepares the necessary input sequence for prediction. Predictions are made and stored. Finally, the model's performance (for the previous day's predictions) is monitored. If performance is poor, it triggers the model retraining DAG.

| Task ID                                 | Operator             | Python Callable / Bash Command (Script Executed)                                                                                                                                                              | Description                                                                                                                                                                                                                            | Inputs                                                                                                                                  | Outputs (XComs / DB / Files)                                                                                                                                                                                                                               | Dependencies                           |
| :-------------------------------------- | :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------- |
| `initialize_database_daily`             | `PythonOperator`     | `callable_initialize_database_daily` (calls `src.utils.db_utils.setup_database`)                                                                                                                              | Ensures all necessary database tables exist and schemas are correct.                                                                                                                                                                   | `params.yaml` (DB config).                                                                                                              | DB tables created/verified.                                                                                                                                                                                                                                | None                                   |
| `fetch_incremental_raw_data_daily`      | `BashOperator`       | `python src/data/make_dataset.py --config {CONFIG_PATH} --mode incremental_fetch`                                                                                                                               | Fetches new raw stock data for tickers since the last fetch from Yahoo Finance and saves it to the `raw_stock_data` DB table.                                                                                                            | `params.yaml` (tickers, period, interval, DB config). `raw_stock_data` DB (for last known dates).                                      | Updates `raw_stock_data` DB table.                                                                                                                                                                                                                         | `initialize_database_daily`            |
| `get_production_model_details_daily`    | `PythonOperator`     | `callable_get_production_model_details_daily`                                                                                                                                                                 | Queries MLflow for the model aliased "Production". Retrieves its loading URI, base name, version, its training MLflow run ID, and the `dataset_run_id` (param from MLflow run) used to train it.                                          | `params.yaml` (MLflow config). MLflow server.                                                                                           | XComs: `production_model_uri_for_loading`, `production_model_base_name`, `production_model_version_number`, `production_model_training_dataset_run_id`.                                                                                                  | `fetch_incremental_raw_data_daily`     |
| `prepare_daily_prediction_input`        | `BashOperator`       | `python src/features/prepare_prediction_input.py --config {CONFIG_PATH} --production_model_training_run_id "{XCOM}" --output_dir {PREDICTION_INPUT_TEMP_DIR}`                                                | Prepares the input sequence for prediction: loads scalers/features (using prod model's `dataset_run_id`), fetches recent raw data, applies TAs, scales, and saves as a `.npy` file.                                                          | `params.yaml`, XCom (`production_model_training_dataset_run_id`), `PREDICTION_INPUT_TEMP_DIR`. DB (`scalers`, `processed_feature_data`, `raw_stock_data`). | XCom (via `do_xcom_push` & script print): Path to the `.npy` input sequence file (e.g., `OUTPUT_PATH:/tmp/.../input.npy`). Creates `.npy` file. | `get_production_model_details_daily`   |
| `make_daily_prediction`                 | `BashOperator`       | `python src/models/predict_model.py --config {CONFIG_PATH} --input_sequence_path "{XCOM}" --production_model_uri_for_loading "{XCOM}" --production_model_base_name "{XCOM}" --production_model_version_number "{XCOM}"` | Loads the production model, makes predictions on the input sequence, inverse transforms, and saves predictions to JSON files and `latest_predictions` DB table (with `target_prediction_date` as current day).                         | `params.yaml`, XComs (input path, model URI, name, version). MLflow server. DB (`scalers`, `processed_feature_data`). `.npy` input file. | Updates `latest_predictions.json`, historical JSON, and `latest_predictions` DB table.                                                                                                                                                                 | `prepare_daily_prediction_input`       |
| `monitor_model_performance_daily_task`  | `BashOperator`       | `python src/models/monitor_performance.py --config {CONFIG_PATH}`                                                                                                                                             | Evaluates predictions for a past date (e.g., yesterday) against actuals. Calculates metrics (MAPE, directional accuracy), saves them to `model_performance_log` DB, and checks against thresholds to decide on retraining.                 | `params.yaml` (thresholds, lag days, DB config). DB (`latest_predictions`, `raw_stock_data`).                                           | XCom (via `do_xcom_push` & script print): String like `NEXT_TASK_ID:trigger_retraining_pipeline_task`. Updates `model_performance_log` DB.                                                                                                                    | `make_daily_prediction`                |
| `branch_based_on_performance`           | `BranchPythonOperator` | `callable_branch_on_monitoring_result_daily`                                                                                                                                                                  | Reads XCom from `monitor_model_performance_daily_task` and decides which downstream task to execute based on the `NEXT_TASK_ID` string.                                                                                                    | XCom (from `monitor_model_performance_daily_task`). `params.yaml` (task IDs for branching).                                             | Returns a task ID string for branching.                                                                                                                                                                                                                          | `monitor_model_performance_daily_task` |
| `trigger_retraining_pipeline_task`      | `TriggerDagRunOperator`| (Internal Airflow Operator)                                                                                                                                                                                   | If chosen by branching, triggers the `stock_model_retraining_prod` DAG.                                                                                                                                                                 | `params.yaml` (ID of DAG to trigger).                                                                                                   | Triggers the retraining DAG.                                                                                                                                                                                                                               | `branch_based_on_performance`          |
| `no_retraining_needed_task`             | `DummyOperator`      | (Internal Airflow Operator)                                                                                                                                                                                   | If chosen by branching, does nothing. Marks the end of this path for the DAG run.                                                                                                                                                      | None.                                                                                                                                   | None.                                                                                                                                                                                                                                                      | `branch_based_on_performance`          |

---

DAG 2: `stock_model_retraining_prod` (`dag_stock_model_retraining.py`)
---------------------------------------------------------------------

**General Workflow Summary:**
This DAG orchestrates the complete model retraining lifecycle. It begins by ensuring the database is set up. Then, it processes all historical data to create a comprehensive training dataset, generating a unique `run_id` for this dataset version. Subsequent steps (feature building, hyperparameter optimization, and candidate model training) all use this `run_id` to ensure consistency. After a candidate model is trained and logged to MLflow, it's evaluated against the current production model. If superior, the candidate is promoted to "Production" in MLflow.

| Task ID                                   | Operator             | Python Callable / Bash Command (Script Executed)                                                                                                                                  | Description                                                                                                                                                                                                                                                           | Inputs                                                                                                                                                             | Outputs (XComs / DB / Files / MLflow)                                                                                                                                                                                                | Dependencies                               |
| :---------------------------------------- | :------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------- |
| `initialize_database_for_retraining`      | `PythonOperator`     | `callable_initialize_database_for_retrain` (calls `src.utils.db_utils.setup_database`)                                                                                              | Ensures all necessary database tables exist and schemas are correct.                                                                                                                                                                                                | `params.yaml` (DB config).                                                                                                                                         | DB tables created/verified.                                                                                                                                                                                                          | None                                       |
| `process_all_data_for_retraining_task`    | `PythonOperator`     | `callable_run_make_dataset_full_retrain` (executes `src/data/make_dataset.py --mode full_process`)                                                                                   | Processes all raw data: fetches updates, adds TAs, preprocesses, creates NumPy arrays, and saves to `processed_feature_data` DB with a new `run_id`.                                                                                                            | `params.yaml`. `raw_stock_data` DB.                                                                                                                                | XCom (`new_dataset_run_id`). Updates `processed_feature_data` DB.                                                                                                                                                                    | `initialize_database_for_retraining`       |
| `build_features_for_retraining`           | `BashOperator`       | `python src/features/build_features.py --config {CONFIG_PATH} --run_id "{XCOM}"`                                                                                                     | Loads processed data (for `run_id` from XCom), creates sequences, scales them, and saves scaled data to `scaled_feature_sets` DB and scalers to `scalers` DB.                                                                                                  | `params.yaml`, XCom (`new_dataset_run_id`). `processed_feature_data` DB.                                                                                           | Updates `scaled_feature_sets` and `scalers` DB tables.                                                                                                                                                                               | `process_all_data_for_retraining_task`     |
| `optimize_hyperparams_for_retraining`     | `BashOperator`       | `python src/models/optimize_hyperparams.py --config {CONFIG_PATH} --run_id "{XCOM}"`                                                                                                | Loads scaled data and scalers (for `run_id` from XCom). Uses Optuna to find best model hyperparameters. Saves best params to `optimization_results` DB and optionally to `config/best_params.json`.                                                            | `params.yaml`, XCom (`new_dataset_run_id`). `scaled_feature_sets`, `scalers` DB.                                                                                   | Updates `optimization_results` DB and potentially `best_params.json`.                                                                                                                                                                | `build_features_for_retraining`            |
| `train_candidate_model_task`              | `PythonOperator`     | `callable_train_candidate_and_get_ids` (executes `src/models/train_model.py --config {CONFIG_PATH} --run_id "{XCOM}"`)                                                                | Loads data, scalers, best params (for `run_id` from XCom). Trains a model. Logs model, params (incl. `dataset_run_id`), metrics to MLflow. Registers model. Prints MLflow run ID of trained model.                                                            | `params.yaml`, XCom (`new_dataset_run_id`). DB (`scaled_feature_sets`, `scalers`, `optimization_results`).                                                        | XComs (`candidate_model_mlflow_run_id`, `candidate_training_dataset_run_id`). Registers model in MLflow. Updates `latest_predictions` DB (test preds).                                                                                     | `optimize_hyperparams_for_retraining`      |
| `evaluate_and_branch_on_promotion_decision` | `BranchPythonOperator` | `callable_evaluate_and_branch_promotion`                                                                                                                                            | Compares the newly trained candidate model (from XCom `candidate_model_mlflow_run_id`) against the current "Production" model in MLflow using a specified metric (from `params.yaml`). Decides whether to promote.                                              | XComs (candidate MLflow run ID, candidate dataset run ID). `params.yaml` (MLflow config, promotion logic). MLflow server.                                         | XCom (`candidate_model_version_to_promote_run_id` if promoting). Returns task ID string for branching.                                                                                                                                         | `train_candidate_model_task`               |
| `promote_candidate_to_production_task`    | `PythonOperator`     | `callable_promote_model_to_production`                                                                                                                                            | If chosen by branching, pulls the candidate's MLflow run ID (XCom). Finds its version in MLflow Registry. Sets the "Production" alias for `experiment_name` to this new model version.                                                                  | XCom (`candidate_model_version_to_promote_run_id`). `params.yaml` (MLflow config). MLflow server.                                                                | Updates MLflow Model Registry aliases.                                                                                                                                                                                               | `evaluate_and_branch_on_promotion_decision` |
| `do_not_promote_task`                     | `DummyOperator`      | (Internal Airflow Operator)                                                                                                                                                       | If chosen by branching, does nothing. The candidate model was not promoted.                                                                                                                                                                   | None.                                                                                                                                                              | None.                                                                                                                                                                                                                                | `evaluate_and_branch_on_promotion_decision` |

