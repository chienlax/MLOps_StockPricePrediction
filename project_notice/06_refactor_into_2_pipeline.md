# Plan

**Core Idea:**

1.  **`daily_stock_operations_dag.py`**: This DAG will run daily. Its primary responsibilities will be:
    *   Ingesting new raw stock data.
    *   Preparing the input needed for *daily predictions* using the current *production model's* configuration (scalers, features).
    *   Making daily predictions.
    *   Evaluating the previous day's predictions against actuals.
    *   Triggering the retraining DAG if performance degrades.

2.  **`stock_model_retraining_dag.py`**: This DAG will be triggered (by the daily DAG or manually). Its responsibilities are:
    *   Performing a full data processing and feature engineering run on *all available data* to create a fresh, versioned dataset for training.
    *   Running hyperparameter optimization.
    *   Training a new candidate model.
    *   Evaluating the candidate model against the production model.
    *   Promoting the candidate to "Production" in MLflow Model Registry if it's better.

**Refined Refactoring Plan (Two-DAG Approach):**

**Phase 1: Foundational Enhancements (DB, Config, Script Args)**

This phase is largely the same as my previous "Phase 1", as it's crucial for both DAGs.

1.  **Database Schema and Utilities (`src/utils/db_utils.py`)**:
    *   **Action**: Implement the `model_performance_log` table.
        ```sql
        CREATE TABLE IF NOT EXISTS model_performance_log (
            log_id SERIAL PRIMARY KEY,
            prediction_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            actual_price REAL,
            predicted_price REAL,
            mae REAL, rmse REAL, mape REAL,
            direction_accuracy REAL,
            model_mlflow_run_id TEXT,
            evaluation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (prediction_date, ticker, model_mlflow_run_id)
        );
        CREATE INDEX IF NOT EXISTS idx_perf_log_date_ticker ON model_performance_log (prediction_date, ticker);
        ```
    *   **Action**: Add/update these functions in `db_utils.py`:
        *   `setup_database()`: Include the new table.
        *   `save_daily_performance_metrics(db_config, date, ticker, metrics_dict, model_mlflow_run_id)`
        *   `get_recent_performance_metrics(db_config, ticker, days_lookback)`
        *   `get_latest_raw_data_window(db_config, tickers_list, window_size_days)`
        *   `get_last_data_timestamp_for_ticker(db_config, ticker)`
        *   `save_prediction()`: Ensure it can store `model_run_id`.

2.  **Configuration (`params.yaml`)**:
    *   **Action**: Add these sections:
        ```yaml
        monitoring:
          performance_thresholds:
            mape_max: 0.07        # Threshold: Max 7% MAPE
            direction_accuracy_min: 0.53 # Threshold: Min 53% directional accuracy
            # Add rmse_max, etc., if you want to monitor them too
          evaluation_lag_days: 1 # How many days to wait for actuals (e.g., T+1 for stock prices)

        airflow_dags:
          daily_operations_dag_id: "daily_stock_operations"
          retraining_pipeline_dag_id: "stock_model_retraining_pipeline"

        # Your existing database, output_paths, mlflow sections remain
        ```
    *   **Question for you**: Do these example thresholds (`mape_max: 0.07`, `direction_accuracy_min: 0.53`) seem like reasonable starting points for your stock data? We can adjust them later.

3.  **Core Python Script Argument Enhancements**:
    *   **Rationale**: To make scripts runnable via Airflow's `BashOperator` and to pass context like `run_id`.
    *   **`src/data/make_dataset.py` (`run_processing` function)**:
        *   **Action**: Modify to have two modes:
            1.  **Incremental Raw Data Fetch Mode**: Called daily. Fetches only new raw data since the last fetch for each ticker and saves it to `raw_stock_data`.
            2.  **Full Training Data Processing Mode**: Called by the retraining DAG. Processes *all* available raw data from the DB, performs feature engineering (technical indicators, alignment, etc.), and saves to `processed_feature_data` table with a **new, unique `run_id`**. This function must *return* this `run_id`.
        *   **Action**: The main `if __name__ == '__main__':` block should probably default to the "Full Training Data Processing Mode" or accept a flag like `--mode incremental_fetch` or `--mode full_process`.
    *   **`src/features/build_features.py` (`run_feature_building` function)**:
        *   **Action**: Add a required `--run_id <id>` command-line argument. This `run_id` will be used to:
            *   Load the correct `processed_data` and `targets` from `processed_feature_data` table.
            *   Save `X_train_scaled`, `y_train_scaled`, etc., to `scaled_feature_sets` table, associated with this `run_id`.
            *   Save `scalers_x`, `y_scalers` to `scalers` table, associated with this `run_id`.
    *   **`src/models/optimize_hyperparams.py` (`run_optimization` function)**:
        *   **Action**: Add a required `--run_id <id>` argument. Use it to load scaled data and scalers associated with this `run_id`. Save optimization results (best params) to `optimization_results` table, linked to this `run_id`.
    *   **`src/models/train_model.py` (`run_training` function)**:
        *   **Action**: Add a required `--run_id <id>` argument. Use it to load scaled data, scalers, and best hyperparameters (all associated with this `run_id`) from the database.

**Phase 2: New Scripts for Daily Operations**

1.  **Create `src/features/prepare_prediction_input.py`**:
    *   **Purpose**: Prepares the exact input sequence the production model expects for a new prediction.
    *   **Function**: `run_prepare_input(config_path: str, production_model_training_run_id: str)`
        *   Loads `db_config`, `sequence_length` from `params.yaml`.
        *   Loads `scalers_x` and `feature_columns` from the DB using `production_model_training_run_id` (this `run_id` is the one from when the *production model was originally trained*).
        *   Fetches the latest `sequence_length + buffer_for_ta` days of raw data from `raw_stock_data` table.
        *   Applies technical indicators.
        *   Selects features, takes the last `sequence_length` records.
        *   Scales using the loaded `scalers_x`.
        *   Saves the resulting NumPy array `(1, seq_len, num_stocks, num_features)` to a temporary path (e.g., in `/tmp/` inside the container) and prints this path. Airflow will use XComs or this path.
    *   **Action**: Implement this script with an argument parser.

2.  **Refactor `src/models/predict_model.py`**:
    *   **Purpose**: Make predictions using the production model and prepared input.
    *   **Function**: `run_daily_prediction(config_path: str, input_sequence_path: str, production_model_uri: str)`
        *   Loads `db_config`, `mlflow_experiment_name`, `mlflow_tracking_uri` from `params.yaml`.
        *   Sets MLflow tracking URI.
        *   Loads the production model from MLflow Model Registry using `production_model_uri` (e.g., `models:/Stock_Price_Prediction_LSTM_Refactored/Production`).
        *   Gets the `run_id` associated with this production model from its MLflow metadata (e.g., `mlflow.models.get_model_info(production_model_uri).run_id`). This is the `prod_model_training_run_id`.
        *   Loads `y_scalers` and `tickers` from DB using this `prod_model_training_run_id`.
        *   Loads the `input_sequence` from `input_sequence_path`.
        *   Predicts, inverse transforms.
        *   Saves to daily JSON files (as currently) AND to `latest_predictions` DB table (using `db_utils.save_prediction` with `model_run_id=prod_model_training_run_id`).
    *   **Action**: Refactor this script. Remove old file-based data loading.

3.  **Create `src/models/monitor_performance.py`**:
    *   **Purpose**: Evaluate previous day's predictions and decide on retraining.
    *   **Function**: `run_monitoring(config_path: str)`
        *   Loads `db_config`, `tickers_list`, `performance_thresholds`, `evaluation_lag_days` from `params.yaml`.
        *   Determines `prediction_date_to_evaluate = today - evaluation_lag_days`.
        *   For each ticker:
            *   Fetch its predicted price and `model_mlflow_run_id` for `prediction_date_to_evaluate` from `latest_predictions` DB table.
            *   Fetch its actual closing price for `prediction_date_to_evaluate` from `raw_stock_data` DB table.
            *   Calculate MAE, RMSE, MAPE, Directional Accuracy.
            *   Save metrics to `model_performance_log` DB table using `db_utils.save_daily_performance_metrics()`.
        *   Aggregate metrics (e.g., average MAPE).
        *   Compares against `performance_thresholds`.
        *   **Returns**: A string indicating the next task: `trigger_retraining_task_id` or `no_retraining_task_id`.
    *   **Action**: Implement this script. This will be used by a `BranchPythonOperator`.

**Phase 3: Implementing the Airflow DAGs**

1.  **Create `airflow/dags/daily_stock_operations_dag.py`**:
    *   Schedule: Daily (e.g., `0 1 * * *` - 1 AM UTC).
    *   `default_args` as per your teacher's example.
    *   Use `BashOperator` to call your Python scripts.
    *   Tasks:
        *   `task_init_db`: `PythonOperator` calling `db_utils.setup_database`.
        *   `task_fetch_incremental_raw_data`: `BashOperator` calling `python src/data/make_dataset.py --config config/params.yaml --mode incremental_fetch`.
        *   `task_get_production_model_info`: `PythonOperator`.
            *   Connects to MLflow (using `mlflow.tracking.MlflowClient`).
            *   Gets the latest version of the model in "Production" stage for `params['mlflow']['experiment_name']`.
            *   Pushes `production_model_uri` (e.g., `models:/MyModel/Production`) and `production_model_training_run_id` (the `run_id` from when this production model was trained, extracted from model version details) to XComs.
        *   `task_prepare_prediction_input`: `BashOperator` calling
            `python src/features/prepare_prediction_input.py --config config/params.yaml --production_model_training_run_id {{ ti.xcom_pull(task_ids='task_get_production_model_info', key='production_model_training_run_id') }}`.
            *   This script should print the path to the output file, which can be pulled by the next task if needed, or use a shared volume.
        *   `task_make_daily_prediction`: `BashOperator` calling
            `python src/models/predict_model.py --config config/params.yaml --input_sequence_path <path_from_previous_task_or_XCom> --production_model_uri {{ ti.xcom_pull(task_ids='task_get_production_model_info', key='production_model_uri') }}`.
        *   `task_monitor_model_performance`: `BranchPythonOperator` calling `src.models.monitor_performance.run_monitoring`.
            *   `op_kwargs={'config_path': 'config/params.yaml'}`.
        *   `task_trigger_retraining_dag`: `TriggerDagRunOperator`.
            *   `trigger_dag_id=params['airflow_dags']['retraining_pipeline_dag_id']`.
        *   `task_no_retraining_needed`: `DummyOperator`.
    *   Dependencies:
        ```
        task_init_db >> task_fetch_incremental_raw_data
        task_fetch_incremental_raw_data >> task_get_production_model_info
        task_get_production_model_info >> task_prepare_prediction_input >> task_make_daily_prediction
        task_make_daily_prediction >> task_monitor_model_performance
        task_monitor_model_performance >> [task_trigger_retraining_dag, task_no_retraining_needed]
        ```

2.  **Create `airflow/dags/stock_model_retraining_dag.py`**:
    *   Schedule: None (Triggered only).
    *   This DAG will be responsible for the "MLOps End-to-End Pipeline" part from your teacher's example, but focused on retraining your specific model.
    *   Tasks:
        *   `task_process_all_data_for_training`: `PythonOperator` calling `src.data.make_dataset.run_processing(config_path='config/params.yaml', mode='full_process')`.
            *   This Python callable should return the *new unique `current_retraining_run_id`*. This `run_id` is crucial.
            *   Pushes `current_retraining_run_id` to XCom.
        *   `task_build_features_for_training`: `BashOperator` calling
            `python src/features/build_features.py --config config/params.yaml --run_id {{ ti.xcom_pull(task_ids='task_process_all_data_for_training') }}`.
        *   `task_optimize_hyperparams`: `BashOperator` calling
            `python src/models/optimize_hyperparams.py --config config/params.yaml --run_id {{ ti.xcom_pull(task_ids='task_process_all_data_for_training') }}`.
        *   `task_train_candidate_model`: `BashOperator` calling
            `python src/models/train_model.py --config config/params.yaml --run_id {{ ti.xcom_pull(task_ids='task_process_all_data_for_training') }}`.
            *   The `train_model.py` script logs the model to MLflow and gets an MLflow `run_id`. This MLflow run ID refers to the *candidate model training run*. Let's assume `train_model.py` prints this `candidate_mlflow_run_id`.
        *   `task_evaluate_candidate_model`: `PythonOperator`.
            *   **Inputs**: `current_retraining_run_id` (from XCom, for test data), `candidate_mlflow_run_id` (from XCom, if `train_model.py` can push it, or by finding the latest run in MLflow for the experiment).
            *   Loads the test set associated with `current_retraining_run_id` from DB.
            *   Loads the candidate model: `mlflow.pytorch.load_model(f"runs:/{candidate_mlflow_run_id}/model")`.
            *   Loads the current production model from MLflow Registry.
            *   Evaluates both on the *same test set*.
            *   Compares metrics.
            *   Pushes `candidate_model_mlflow_run_id` and `is_candidate_better` (boolean) to XCom.
        *   `task_promotion_branch`: `BranchPythonOperator` based on `is_candidate_better`.
            *   Returns task ID for `task_promote_model` or `task_do_not_promote`.
        *   `task_promote_model`: `PythonOperator`.
            *   Uses `MlflowClient().transition_model_version_stage()` to promote the model version associated with `candidate_model_mlflow_run_id` to "Production". You'll need to find the model *version* from the `candidate_mlflow_run_id` as `train_model.py` likely registers it.
        *   `task_do_not_promote`: `DummyOperator`.
    *   Dependencies similar to your teacher's MLOps pipeline example.

**Key Differences from Teacher's Example & How We Adapt:**

*   **Database:** Your project uses PostgreSQL, not SQLite. `db_utils.py` handles this.
*   **Data Source:** Your raw data comes from Yahoo Finance API, not a CSV. `make_dataset.py` handles this.
*   **Model Type:** LSTM/PyTorch, not scikit-learn RandomForest. Your `model_definitions.py`, `train_model.py` handle this. MLflow works well with PyTorch.
*   **Intermediate Data:** We will heavily rely on PostgreSQL + `run_id` for versioning intermediate data (processed, scaled features, scalers, params) instead of `/tmp/` files. This is more robust for MLOps.
*   **Parameter Passing to Scripts:** We'll use `BashOperator` with CLI arguments (`--config`, `--run_id`, etc.) rather than embedding all logic within PythonOperator callables, leveraging your existing script structure. XComs will pass dynamic values like `run_id` or model URIs between tasks.

**Next Steps for You:**

1.  **Confirm Understanding:** Does this two-DAG approach and the distribution of tasks make sense for your project and align with your teacher's guidance?
2.  **Performance Thresholds:** Let me know your thoughts on the initial `mape_max` and `direction_accuracy_min`.
3.  **Start with Phase 1:**
    *   Modify `db_utils.py`.
    *   Update `params.yaml`.
    *   Start modifying your core Python scripts (`make_dataset.py`, `build_features.py`, `optimize_hyperparams.py`, `train_model.py`) to accept and use the `--run_id` argument for loading/saving versioned data from/to PostgreSQL.

# Phase 1

**Detailed Plan for Phase 1:**

**Part 1: `src/utils/db_utils.py` Modifications**

We'll modify this file to include the new table and functions.

1.  **Update `setup_database(db_config: dict) -> None`**:
    *   **Action**: Add the `CREATE TABLE` and `CREATE INDEX` statements for `model_performance_log` within this function, before the `conn.commit()`.
    *   **Code Snippet (to be inserted into `setup_database`):**
        ```python
        # Inside setup_database function, after other CREATE TABLE statements:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_performance_log (
            log_id SERIAL PRIMARY KEY,
            prediction_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            actual_price REAL,
            predicted_price REAL,
            mae REAL,
            rmse REAL,
            mape REAL,
            direction_accuracy REAL,
            model_mlflow_run_id TEXT,
            evaluation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (prediction_date, ticker, model_mlflow_run_id)
        )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_log_date_ticker ON model_performance_log (prediction_date, ticker)')
        ```

2.  **Add `save_daily_performance_metrics(...)` function**:
    *   **Action**: Create this new function.
    *   **Code Snippet (new function in `db_utils.py`):**
        ```python
        from typing import Dict # Add to imports if not already there

        def save_daily_performance_metrics(
            db_config: dict,
            prediction_date: str, # Expecting 'YYYY-MM-DD' string
            ticker: str,
            metrics_dict: Dict[str, float],
            model_mlflow_run_id: str
        ) -> bool:
            """Save daily model performance metrics to the database."""
            try:
                conn = get_db_connection(db_config)
                cursor = conn.cursor()
                
                cursor.execute('''
                INSERT INTO model_performance_log
                (prediction_date, ticker, actual_price, predicted_price, mae, rmse, mape, direction_accuracy, model_mlflow_run_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (prediction_date, ticker, model_mlflow_run_id) DO UPDATE SET
                    actual_price = EXCLUDED.actual_price,
                    predicted_price = EXCLUDED.predicted_price,
                    mae = EXCLUDED.mae,
                    rmse = EXCLUDED.rmse,
                    mape = EXCLUDED.mape,
                    direction_accuracy = EXCLUDED.direction_accuracy,
                    evaluation_timestamp = CURRENT_TIMESTAMP
                ''', (
                    prediction_date,
                    ticker,
                    metrics_dict.get('actual_price'),
                    metrics_dict.get('predicted_price'),
                    metrics_dict.get('mae'),
                    metrics_dict.get('rmse'),
                    metrics_dict.get('mape'),
                    metrics_dict.get('direction_accuracy'),
                    model_mlflow_run_id
                ))
                
                conn.commit()
                logger.info(f"Saved performance metrics for {ticker} on {prediction_date} (Model Run ID: {model_mlflow_run_id})")
                return True
            except Exception as e:
                logger.error(f"Error saving daily performance metrics for {ticker} on {prediction_date}: {e}", exc_info=True)
                # Consider re-raising or returning False based on desired error handling
                # For now, let's re-raise to make issues visible during development
                raise
            finally:
                if 'conn' in locals() and conn:
                    cursor.close()
                    conn.close()
        ```

3.  **Add `get_recent_performance_metrics(...)` function**:
    *   **Action**: Create this new function.
    *   **Code Snippet (new function in `db_utils.py`):**
        ```python
        from datetime import timedelta # Add to imports

        def get_recent_performance_metrics(db_config: dict, ticker: str, days_lookback: int) -> pd.DataFrame:
            """Retrieve recent performance metrics for a ticker."""
            try:
                conn = get_db_connection(db_config)
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days_lookback)
                
                query = """
                    SELECT prediction_date, actual_price, predicted_price, mae, rmse, mape, direction_accuracy, model_mlflow_run_id
                    FROM model_performance_log
                    WHERE ticker = %s AND prediction_date >= %s AND prediction_date <= %s
                    ORDER BY prediction_date DESC
                """
                df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
                
                logger.info(f"Retrieved {len(df)} recent performance records for {ticker} (last {days_lookback} days)")
                return df
            except Exception as e:
                logger.error(f"Error getting recent performance metrics for {ticker}: {e}", exc_info=True)
                raise
            finally:
                if 'conn' in locals() and conn:
                    conn.close()
        ```

4.  **Add `get_last_data_timestamp_for_ticker(...)` function**:
    *   **Action**: Create this new function.
    *   **Code Snippet (new function in `db_utils.py`):**
        ```python
        def get_last_data_timestamp_for_ticker(db_config: dict, ticker: str) -> Optional[datetime]:
            """Get the timestamp of the latest data point for a ticker in raw_stock_data."""
            try:
                conn = get_db_connection(db_config)
                cursor = conn.cursor()
                
                cursor.execute("SELECT MAX(date) FROM raw_stock_data WHERE ticker = %s", (ticker,))
                result = cursor.fetchone()
                
                if result and result[0]:
                    logger.debug(f"Last data timestamp for {ticker}: {result[0]}")
                    return result[0] # This will be a datetime object if 'date' column is TIMESTAMP
                else:
                    logger.info(f"No data found for {ticker}, will fetch full history.")
                    return None
            except Exception as e:
                logger.error(f"Error getting last data timestamp for {ticker}: {e}", exc_info=True)
                raise
            finally:
                if 'conn' in locals() and conn:
                    cursor.close()
                    conn.close()
        ```

5.  **Add `get_latest_raw_data_window(...)` function**:
    *   **Action**: Create this new function. This is for fetching a small window for daily predictions.
    *   **Code Snippet (new function in `db_utils.py`):**
        ```python
        def get_latest_raw_data_window(db_config: dict, tickers_list: list, window_size_days: int) -> dict:
            """
            Load the most recent 'window_size_days' of raw data for specified tickers.
            Returns a dictionary of DataFrames {ticker: df}.
            """
            try:
                conn = get_db_connection(db_config)
                ticker_data_window = {}
                
                for t in tickers_list:
                    # Fetch a bit more to ensure we have enough after potential gaps, then take tail
                    # For simplicity, let's fetch N days and sort, then take tail.
                    # A more optimized SQL might use window functions or `ORDER BY date DESC LIMIT N`.
                    query = """
                        SELECT date, open, high, low, close, volume, dividends, stock_splits 
                        FROM raw_stock_data 
                        WHERE ticker = %s 
                        ORDER BY date DESC
                        LIMIT %s 
                    """ 
                    # Fetch slightly more if TA calculations need prior data points not included in window_size_days
                    # For now, let's assume window_size_days is sufficient for TA on that window.
                    df = pd.read_sql_query(query, conn, params=(t, window_size_days), parse_dates=['date'])
                    
                    if df.empty:
                        logger.warning(f"No raw data found in DB for {t} to create latest window.")
                        continue
                    
                    # The query already sorts DESC, so we need to re-sort ASC for time series processing
                    df = df.sort_values(by='date', ascending=True).set_index('date')
                    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
                    
                    if len(df) < window_size_days:
                        logger.warning(f"Fetched {len(df)} records for {t}, less than requested window {window_size_days}.")

                    ticker_data_window[t] = df
                    logger.info(f"Loaded latest {len(df)} records for {t} for prediction input.")
                
                if not ticker_data_window:
                    logger.error("No data loaded for any ticker for the latest window.")
                    # Depending on strictness, you might raise an error here
                
                return ticker_data_window
            except Exception as e:
                logger.error(f"Error loading latest raw data window: {e}", exc_info=True)
                raise
            finally:
                if 'conn' in locals() and conn:
                    conn.close()
        ```

6.  **Modify `save_prediction(...)` function**:
    *   **Action**: Your existing `save_prediction` function already includes `model_run_id` in its parameters and the `latest_predictions` table schema also has `model_run_id`. Just double-check the `INSERT ... ON CONFLICT` statement to ensure `model_run_id = EXCLUDED.model_run_id` is present in the `DO UPDATE SET` clause.
    *   **Current `save_prediction` in your file:**
        ```python
        # ...
        cursor.execute('''
        INSERT INTO latest_predictions 
        (prediction_timestamp, ticker, predicted_price, model_run_id)
        VALUES (CURRENT_TIMESTAMP, %s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE SET
            prediction_timestamp = CURRENT_TIMESTAMP,
            predicted_price = EXCLUDED.predicted_price,
            model_run_id = EXCLUDED.model_run_id  # <--- THIS IS GOOD!
        ''', (ticker, predicted_price, model_run_id))
        # ...
        ```
    *   **No change needed here if it matches the above.**

**Part 2: `params.yaml` Modifications**

1.  **Action**: Open your `params.yaml` file.
2.  **Action**: Add the following new sections. You can place them logically, perhaps after `mlflow` or at the end.

    ```yaml
    # Add these new sections to your params.yaml

    monitoring:
      performance_thresholds:
        mape_max: 0.07        # Example: Max 7% MAPE (Adjust as needed)
        direction_accuracy_min: 0.53 # Example: Min 53% directional accuracy (Adjust as needed)
        # You can add other metrics like rmse_max if you decide to monitor them
      evaluation_lag_days: 1 # Days to wait for actuals to become available (e.g., T+1 for stock prices)

    airflow_dags:
      daily_operations_dag_id: "daily_stock_operations"
      retraining_pipeline_dag_id: "stock_model_retraining_pipeline"
    ```

**Part 3: Core Python Script Argument Enhancements**

We'll add command-line arguments and modify the main execution blocks.

1.  **`src/data/make_dataset.py`**:
    *   **Action 1**: Modify `argparse` setup.
        ```python
        # At the top of make_dataset.py, where you have argparse
        # import argparse (if not already there)
        # import sys (if not already there)
        # import os (if not already there)

        # Modify the parser setup in if __name__ == '__main__':
        # FROM:
        # parser = argparse.ArgumentParser()
        # parser.add_argument('--config', type=str, 
        #                   required=False, # Was False, let's make it consistently required or default
        #                   help='Path to the configuration file (params.yaml)')
        # ...
        # if len(sys.argv) == 1:
        #     config_path = 'params.yaml'
        # else:
        #     args = parser.parse_args()
        #     config_path = args.config

        ## START OF PROPOSED CHANGES for argparse in make_dataset.py ##
        if __name__ == '__main__':
            parser = argparse.ArgumentParser(description="Data processing script for stock prediction.")
            parser.add_argument(
                '--config',
                type=str,
                default='config/params.yaml', # Default path relative to project root
                help='Path to the configuration file (params.yaml)'
            )
            parser.add_argument(
                '--mode',
                type=str,
                choices=['incremental_fetch', 'full_process'],
                default='full_process', # Default to full processing if not specified
                help='Operation mode: "incremental_fetch" for daily raw data, "full_process" for complete training dataset generation.'
            )
            args = parser.parse_args()
            config_path = args.config
            mode = args.mode

            # ... rest of your existing __main__ block, now using 'mode' and 'config_path'
            # Example:
            # if not os.path.exists(config_path): ...
            # with open(config_path, 'r') as f: ...

            logger.info(f"Starting data processing script with config: {config_path} in mode: {mode}")
            
            # Call run_processing with the mode
            # run_processing needs to be adapted to handle the mode and return run_id for 'full_process'
            if mode == 'full_process':
                # This function will now need to return the run_id
                generated_run_id = run_processing(config_path, mode=mode)
                if generated_run_id:
                    print(f"Full processing completed. Dataset run_id: {generated_run_id}")
                else:
                    logger.error("Full processing did not return a run_id.")
                    sys.exit(1)
            else: # incremental_fetch
                run_processing(config_path, mode=mode)
            
            logger.info(f"Data processing script (mode: {mode}) finished successfully.")
        ## END OF PROPOSED CHANGES for argparse in make_dataset.py ##
        ```

    *   **Action 2**: Modify `run_processing` function signature and logic.
        ```python
        # Modify the signature of run_processing
        # FROM: def run_processing(config_path: str):
        # TO:
        def run_processing(config_path: str, mode: str = 'full_process') -> Optional[str]:
            """Main function to run data processing steps based on mode."""
            try:
                with open(config_path, 'r') as f:
                    params = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {config_path}")

                db_config = params['database']
                tickers_list = params['data_loading']['tickers']
                period = params['data_loading']['period']
                interval = params['data_loading']['interval']
                fetch_delay = params['data_loading']['fetch_delay']
                # corr_threshold = params['feature_engineering']['correlation_threshold'] # Keep if used in full_process

                if mode == 'incremental_fetch':
                    logger.info("--- Starting Incremental Raw Data Fetch ---")
                    # Simplified load_data for incremental fetch
                    # This part of load_data needs to be refactored or a new function created
                    # For now, let's assume load_data is adapted or you call a specific incremental function
                    load_data_incrementally(tickers_list, interval, fetch_delay, db_config, params) # You'll need to define/refactor this
                    logger.info("--- Finished Incremental Raw Data Fetch ---")
                    return None # No run_id for incremental raw data fetch

                elif mode == 'full_process':
                    logger.info("--- Starting Full Data Processing for Training ---")
                    
                    # 1. Ensure raw data is up-to-date (optional, could be separate Airflow task)
                    # For simplicity, let's assume load_data here also handles incremental updates if needed
                    # Or, the daily DAG already ran incremental fetch.
                    logger.info("--- Ensuring Raw Data is Up-to-Date (calling load_data) ---")
                    load_data(tickers_list, period, interval, fetch_delay, db_config) # This is your existing load_data
                    logger.info("--- Finished Ensuring Raw Data ---")

                    # 2. Load ALL data from database for processing
                    logger.info("--- Loading ALL Data from Database for Full Processing ---")
                    ticker_data = load_data_from_db(db_config, tickers_list)
                    if not ticker_data:
                        logger.error("No data loaded from DB for full processing. Exiting.")
                        sys.exit(1)
                    logger.info(f"Loaded data for {len(ticker_data)} tickers from database for full processing.")

                    # 3. Add Technical Indicators (your existing logic)
                    logger.info("--- Adding Technical Indicators ---")
                    ticker_data_with_ta = add_technical_indicators(ticker_data.copy()) # Use .copy()
                    logger.info("--- Finished Adding Technical Indicators ---")
                    
                    # 4. Filter Correlated Features (your existing logic)
                    # corr_threshold = params['feature_engineering']['correlation_threshold']
                    # logger.info("--- Filtering Correlated Features ---")
                    # ticker_data_filtered, final_feature_cols_before_target = filter_correlated_features(
                    #     ticker_data_with_ta.copy(), corr_threshold
                    # )
                    # logger.info("--- Finished Filtering Correlated Features ---")
                    # Note: The original filter_correlated_features might need adjustment if it drops 'Target'
                    # For now, let's assume it works or we adjust it later.
                    # The `align_and_process_data` will handle feature selection before creating the 3D array.

                    # 5. Align and Process Data (your existing logic)
                    logger.info("--- Aligning and Processing Data into 3D Array ---")
                    # Ensure the input to align_and_process_data is what it expects
                    # It was ticker_data_filtered in your original script after correlation filtering
                    # If you skip correlation filtering for now, pass ticker_data_with_ta
                    processed_data_arr, targets_arr, feature_columns_list, final_tickers_list = align_and_process_data(
                        ticker_data_with_ta.copy() # Or ticker_data_filtered.copy()
                    )
                    logger.info("--- Finished Aligning and Processing Data ---")

                    # 6. Save Processed Data to database with a NEW RUN_ID
                    logger.info(f"--- Saving Processed Data to database for Training ---")
                    current_run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
                    try:
                        save_processed_features_to_db(
                            db_config,
                            processed_data_arr,
                            targets_arr,
                            feature_columns_list,
                            final_tickers_list,
                            current_run_id # Pass the new run_id
                        )
                        logger.info(f"Successfully saved processed data to database with run_id: {current_run_id}")
                        return current_run_id # Return the run_id
                    except Exception as e:
                        logger.error(f"Failed to save processed data to database for run_id {current_run_id}", exc_info=True)
                        raise
                    logger.info("--- Finished Saving Processed Data for Training ---")
                else:
                    logger.error(f"Invalid mode: {mode}")
                    sys.exit(1)

            except Exception as e:
                logger.error(f"Error in run_processing (mode: {mode}): {e}", exc_info=True)
                if mode == 'full_process': return None # Ensure None is returned on error for full_process
                raise
        ```
    *   **Action 3**: Create/Refactor `load_data_incrementally` (or adapt `load_data`).
        *   This function will loop through tickers, call `get_last_data_timestamp_for_ticker`, then fetch data from yfinance using `start=` argument if a last timestamp exists, otherwise fetch full `period`. Then save using `save_to_raw_table`.
        *   **Simplified `load_data` adaptation for `make_dataset.py`:**
            ```python
            # Modify your existing load_data function in make_dataset.py
            def load_data(tickers_list, period, interval, fetch_delay, db_config):
                """Loads data for tickers. Fetches incrementally if data exists, else full period."""
                try:
                    setup_database(db_config) # Ensure tables exist
                    logger.info("Initializing/Verifying PostgreSQL database connection for raw data loading.")

                    for t in tickers_list:
                        last_db_date = get_last_data_timestamp_for_ticker(db_config, t)
                        yf_period = period # Default to full period
                        yf_start_date = None

                        if last_db_date:
                            # yfinance start is inclusive, so fetch from the day after
                            yf_start_date = (pd.to_datetime(last_db_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                            # If fetching incrementally, 'period' argument is ignored by yfinance if 'start' is present
                            logger.info(f"Incremental fetch for {t} from {yf_start_date}")
                            # We don't need to set yf_period = None, yfinance handles it.
                        else:
                            logger.info(f"Full history fetch for {t} for period {period}")
                            # yf_start_date remains None, yf_period is used

                        try:
                            logger.info(f"Downloading data for {t} (Start: {yf_start_date}, Period: {period if not yf_start_date else 'N/A'})")
                            # Pass start_date if available, otherwise yfinance uses period
                            data = yf.Ticker(t).history(start=yf_start_date, period=period if not yf_start_date else None, interval=interval)
                            
                            if data.empty:
                                if yf_start_date: # If it was an incremental fetch and no new data
                                    logger.info(f"No new data available for {t} since {last_db_date.strftime('%Y-%m-%d')}")
                                else: # If it was a full fetch and no data at all
                                    logger.warning(f"No data available for {t} for the specified period/start_date.")
                                continue
                                
                            logger.info(f"Downloaded {len(data)} records for {t}")
                            save_to_raw_table(t, data, db_config) # Your existing save function
                            time.sleep(fetch_delay)
                            
                        except Exception as e:
                            logger.error(f"Error loading/saving {t}: {e}", exc_info=True)
                            continue
                except Exception as e:
                    logger.error(f"Error in load_data function: {e}", exc_info=True)
                    raise
            ```

2.  **`src/features/build_features.py`**:
    *   **Action 1**: Modify `argparse` setup.
        ```python
        ## START OF PROPOSED CHANGES for argparse in build_features.py ##
        if __name__ == '__main__':
            parser = argparse.ArgumentParser(description="Feature building script for stock prediction.")
            parser.add_argument(
                '--config',
                type=str,
                default='config/params.yaml',
                help='Path to the configuration file (params.yaml)'
            )
            parser.add_argument(
                '--run_id',
                type=str,
                required=True, # This is crucial for linking to processed data
                help='The run_id of the processed dataset to use for feature building.'
            )
            args = parser.parse_args()
            config_path = args.config
            cli_run_id = args.run_id # Use a different variable name to avoid conflict with run_id from params

            # ... existing validation for config_path and database section ...

            logger.info(f"Starting feature building script with config: {config_path} for run_id: {cli_run_id}")
            # Pass cli_run_id to run_feature_building
            returned_run_id = run_feature_building(config_path, run_id_arg=cli_run_id)
            if returned_run_id: # If run_feature_building confirms the run_id it used
                 logger.info(f"Feature building completed successfully for run_id: {returned_run_id}")
            else:
                logger.error(f"Feature building failed for run_id: {cli_run_id}")
                sys.exit(1)
        ## END OF PROPOSED CHANGES for argparse in build_features.py ##
        ```
    *   **Action 2**: Modify `run_feature_building` function signature and logic.
        ```python
        # Modify the signature of run_feature_building
        # FROM: def run_feature_building(config_path: str):
        # TO:
        def run_feature_building(config_path: str, run_id_arg: str) -> Optional[str]:
            with open(config_path, 'r') as f:
                params = yaml.safe_load(f)
            db_config = params['database']
            logger.info(f"Using PostgreSQL database at {db_config['host']}:{db_config['port']}")
            
            # Use the passed run_id_arg
            current_run_id = run_id_arg 
            logger.info(f"Feature building for run_id: {current_run_id}")

            seq_len = params['feature_engineering']['sequence_length']
            pred_len = params['feature_engineering']['prediction_length']
            train_ratio = params.get('feature_engineering', {}).get('train_ratio', 0.8)

            logger.info(f"--- Loading Processed Data from database for run_id: {current_run_id} ---")
            # Pass current_run_id to load_processed_features_from_db
            processed_data_dict = load_processed_features_from_db(db_config, run_id=current_run_id) 
            
            if not processed_data_dict:
                logger.error(f"No processed data found in database for run_id: {current_run_id}")
                return None # Return None on failure

            # ... (rest of your existing logic for loading data from processed_data_dict) ...
            # processed_data = processed_data_dict['processed_data']
            # targets = processed_data_dict['targets']
            # ...

            # Ensure all save operations use current_run_id
            logger.info(f"--- Saving Scaled Data to database for run_id: {current_run_id} ---")
            save_scaled_features(db_config, current_run_id, 'X_train', X_train_scaled)
            save_scaled_features(db_config, current_run_id, 'y_train', y_train_scaled)
            save_scaled_features(db_config, current_run_id, 'X_test', X_test_scaled)
            save_scaled_features(db_config, current_run_id, 'y_test', y_test_scaled)
            logger.info("--- Finished Saving Scaled Data ---")

            logger.info(f"--- Saving Scalers to database for run_id: {current_run_id} ---")
            scalers_dict = {'scalers_x': scalers_x, 'y_scalers': y_scalers}
            save_scalers(db_config, current_run_id, scalers_dict)
            logger.info("--- Finished Saving Scalers ---")
            
            return current_run_id # Return the run_id it operated on
        ```

3.  **`src/models/optimize_hyperparams.py`**:
    *   **Action 1**: Modify `argparse` setup.
        ```python
        ## START OF PROPOSED CHANGES for argparse in optimize_hyperparams.py ##
        if __name__ == '__main__':
            parser = argparse.ArgumentParser(description="Hyperparameter optimization script.")
            parser.add_argument(
                '--config',
                type=str,
                default='config/params.yaml',
                help='Path to the configuration file (params.yaml)'
            )
            parser.add_argument(
                '--run_id',
                type=str,
                required=True,
                help='The run_id of the dataset (scaled features, scalers) to use for optimization.'
            )
            args = parser.parse_args()
            config_path = args.config
            cli_run_id = args.run_id

            # ... existing validation ...

            logger.info(f"Starting optimization script with config: {config_path} for run_id: {cli_run_id}")
            # Pass cli_run_id to run_optimization
            best_params, used_run_id = run_optimization(config_path, run_id_arg=cli_run_id)
            if best_params and used_run_id:
                logger.info(f"Optimization completed successfully for run_id: {used_run_id}. Best params: {best_params}")
            else:
                logger.error(f"Optimization failed for run_id: {cli_run_id}")
                sys.exit(1)
        ## END OF PROPOSED CHANGES for argparse in optimize_hyperparams.py ##
        ```
    *   **Action 2**: Modify `run_optimization` function signature and logic.
        ```python
        # Modify the signature of run_optimization
        # FROM: def run_optimization(config_path: str):
        # TO:
        def run_optimization(config_path: str, run_id_arg: str) -> tuple[Optional[dict], Optional[str]]:
            with open(config_path, 'r') as f:
                params = yaml.safe_load(f)
            db_config = params['database']
            logger.info(f"Using PostgreSQL database at {db_config['host']}:{db_config['port']}")
            
            current_run_id = run_id_arg # Use the passed run_id
            logger.info(f"Hyperparameter optimization for run_id: {current_run_id}")
            n_trials = params['optimization']['n_trials']
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            logger.info(f"--- Loading Scaled Data from database for run_id: {current_run_id} ---")
            # Pass current_run_id to load functions
            X_train_scaled = load_scaled_features(db_config, current_run_id, 'X_train')
            y_train_scaled = load_scaled_features(db_config, current_run_id, 'y_train')
            # ... load X_test, y_test similarly ...
            
            logger.info(f"--- Loading Scalers from database for run_id: {current_run_id} ---")
            scalers = load_scalers(db_config, current_run_id)
            # ... error handling if data or scalers are None ...

            # ... (rest of your Optuna study logic) ...

            logger.info(f"--- Saving Best Parameters to database for run_id: {current_run_id} ---")
            # Pass current_run_id to save_optimization_results
            save_optimization_results(db_config, current_run_id, study.best_params)
            logger.info("--- Finished Saving Best Parameters ---")
            
            # ... (optional saving to file) ...
            
            return study.best_params, current_run_id # Return best_params and the run_id used
        ```

4.  **`src/models/train_model.py`**:
    *   **Action 1**: Modify `argparse` setup.
        ```python
        ## START OF PROPOSED CHANGES for argparse in train_model.py ##
        if __name__ == '__main__':
            parser = argparse.ArgumentParser(description="Model training script.")
            parser.add_argument(
                '--config',
                type=str,
                # required=True, # Your original was True, let's keep it or provide a default
                default='config/params.yaml',
                help='Path to the configuration file (params.yaml)'
            )
            parser.add_argument(
                '--run_id',
                type=str,
                required=True,
                help='The run_id of the dataset (scaled features, scalers, params) to use for training.'
            )
            args = parser.parse_args()
            config_path = args.config
            cli_run_id = args.run_id

            # ... existing validation ...

            logger.info(f"Starting training script with config: {config_path} for run_id: {cli_run_id}")
            # Pass cli_run_id to run_training
            run_training(config_path, run_id_arg=cli_run_id)
            logger.info(f"Training completed successfully for run_id: {cli_run_id}")
        ## END OF PROPOSED CHANGES for argparse in train_model.py ##
        ```
    *   **Action 2**: Modify `run_training` function signature and logic.
        ```python
        # Modify the signature of run_training
        # FROM: def run_training(config_path: str):
        # TO:
        def run_training(config_path: str, run_id_arg: str): # No return needed if it just executes
            with open(config_path, 'r') as f:
                params = yaml.safe_load(f)
            db_config = params['database']
            
            current_run_id = run_id_arg # Use the passed run_id
            logger.info(f"Model training for run_id: {current_run_id}")
            # ... (mlflow_experiment, final_training_epochs, plot_output_dir, device setup) ...

            logger.info(f"--- Loading Scaled Data from database for run_id: {current_run_id} ---")
            # Pass current_run_id to load functions
            X_train_scaled = load_scaled_features(db_config, current_run_id, 'X_train')
            # ... load y_train, X_test, y_test similarly ...

            logger.info(f"--- Loading Scalers from database for run_id: {current_run_id} ---")
            scalers = load_scalers(db_config, current_run_id)
            # ...

            logger.info(f"--- Loading Feature Data (tickers) from database for run_id: {current_run_id} ---")
            feature_data = load_processed_features_from_db(db_config, current_run_id)
            # ...

            logger.info(f"--- Loading Best Hyperparameters from database for run_id: {current_run_id} ---")
            best_params = load_optimization_results(db_config, current_run_id)
            # ...

            # ... (rest of your MLflow setup and model training logic) ...
            # final_model, predictions, targets, final_metrics, mlflow_run_id_for_model = train_final_model(...)
            # Note: mlflow_run_id_for_model is the MLflow run ID of THIS training run.
            # The current_run_id is for the DATASET version.

            logger.info("--- Saving Predictions to Database (from this training run) ---")
            # When saving predictions from this training run (e.g., on its test set for logging),
            # the model_run_id in save_prediction should be mlflow_run_id_for_model.
            try:
                for i, ticker in enumerate(tickers): # tickers from feature_data
                    if i < len(predictions): # predictions from train_final_model
                        # This saves the *test set predictions* made during this training run
                        # For daily operational predictions, a different script will be used.
                        last_test_prediction = predictions[i][-1] # Example: last prediction on test set
                        save_prediction(db_config, ticker, float(last_test_prediction), mlflow_run_id_for_model)
                        logger.info(f"Saved test prediction for {ticker} from training run {mlflow_run_id_for_model}")
                # ...
            except Exception as e:
                logger.error(f"Error saving test predictions during training: {e}")
                # Decide if this should halt the process or just be a warning
            # ...
        ```

---

This is a comprehensive set of changes for Phase 1. Take your time implementing them file by file. Test each script individually from your command line after making changes to ensure the argument parsing and new logic paths work as expected before integrating them into Airflow.

For example, after modifying `make_dataset.py`:
`python src/data/make_dataset.py --config config/params.yaml --mode incremental_fetch`
`python src/data/make_dataset.py --config config/params.yaml --mode full_process` (and check if it prints a run_id)

Then, for `build_features.py` (assuming a `run_id` like `20230101_120000` was generated by `full_process`):
`python src/features/build_features.py --config config/params.yaml --run_id 20230101_120000`

# Phase 2:
Okay, Phase 1 is a solid achievement! Now, let's dive into Phase 2: creating the scripts that will handle the daily operational aspects of your MLOps pipeline.

I will provide detailed instructions and code for each of the three scripts. Remember, the goal is to make these scripts robust, configurable, and callable from an orchestrator like Airflow.

---

**1. Create `src/features/prepare_prediction_input.py`**

**Purpose:**
This script is responsible for taking the latest raw market data and transforming it into the precise input format expected by the *current production model*. This involves using the feature engineering logic and scalers that were associated with the production model when it was trained.

**Thinking Process & Design Choices:**

1.  **Inputs:**
    *   `--config`: Path to `params.yaml` for database config, sequence length, tickers.
    *   `--production_model_training_run_id`: This is the crucial link. It's the `run_id` (from our `processed_feature_data` table) that was used when the *current production model was trained*. This ID allows us to fetch the *exact* `feature_columns` and `scalers_x` that correspond to that model. This ID will be passed by Airflow after querying MLflow Model Registry.
2.  **Core Logic:**
    *   **Load Metadata:** Fetch `feature_columns` (from `processed_feature_data` via `load_processed_features_from_db`) and `scalers_x` (from `scalers` table via `load_scalers`) using the `production_model_training_run_id`.
    *   **Fetch Latest Raw Data:** Get a recent window of raw data. The window size needs to be at least `sequence_length` plus any lookback period required by your technical indicators (e.g., if you have a 50-day EMA, you need at least 50 prior data points). I'll use `sequence_length + 60` as a safe buffer (assuming max TA window is around 50-60 days).
    *   **Apply Technical Indicators:** Reuse the `add_technical_indicators` function from `src.data.make_dataset`. This emphasizes modularity.
    *   **Align and Select Features:**
        *   Similar to `align_and_process_data` but simplified for this specific task. We need to ensure all tickers have data for a common set of recent dates.
        *   Select only the `feature_columns` that the production model was trained on.
    *   **Prepare 3D Array:** Convert the dictionary of DataFrames into a 3D NumPy array: `(timesteps, num_stocks, num_features)`.
    *   **Extract Final Sequence:** Take the *last* `sequence_length` records from this 3D array.
    *   **Scale Sequence:** Apply the loaded `scalers_x` (feature-wise, stock-wise) to this sequence.
    *   **Reshape for Model:** Add a batch dimension: `(1, sequence_length, num_stocks, num_features)`.
3.  **Output:**
    *   The script will save the final scaled NumPy array to a temporary file (e.g., in `/tmp/` inside the container, or a shared volume if not using XComs for data passing in Airflow).
    *   It will print the *absolute path* to this file to `stdout`. Airflow's `BashOperator` can capture this output and pass it via XComs to the next task.
4.  **Modularity:** Importing `add_technical_indicators` from `make_dataset.py` is key. This means `make_dataset.py` should be structured to allow this (i.e., the function isn't only callable from its `if __name__ == '__main__':` block).

**Code for `src/features/prepare_prediction_input.py`:**

```python
# src/features/prepare_prediction_input.py
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys
import os
from datetime import datetime

# Ensure 'src' is in PYTHONPATH for imports if script is run directly
# or when Airflow executes it.
# This assumes the script is in src/features/ and db_utils is in src/utils/, make_dataset in src/data/
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from utils.db_utils import (
        load_scalers,
        load_processed_features_from_db, # To get feature_columns and tickers
        get_latest_raw_data_window
    )
    # We need add_technical_indicators from make_dataset.py
    # This requires make_dataset.py to be importable and the function to be defined at the module level.
    from data.make_dataset import add_technical_indicators, preprocess_data # Assuming preprocess_data is also needed
except ImportError as e:
    logger.error(f"Failed to import necessary modules. Ensure PYTHONPATH is set correctly or scripts are in expected locations: {e}")
    sys.exit(1)


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run_prepare_input(config_path: str, production_model_training_run_id: str, output_dir: str) -> Optional[str]:
    """
    Prepares the input sequence for daily prediction using the production model's configuration.

    Args:
        config_path (str): Path to the params.yaml file.
        production_model_training_run_id (str): The dataset run_id used to train the current production model.
        output_dir (str): Directory to save the temporary output sequence file.

    Returns:
        Optional[str]: Absolute path to the saved .npy input sequence file, or None on failure.
    """
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        db_config = params['database']
        feature_eng_params = params['feature_engineering']
        sequence_length = feature_eng_params['sequence_length']
        
        # Tickers for which to prepare input - could come from params or be derived
        # For consistency, let's assume it's the same list as used in training,
        # which can be fetched along with feature_columns.
        
        logger.info(f"Preparing prediction input using production model's training run_id: {production_model_training_run_id}")

        # 1. Load scalers_x and feature_columns associated with the production model's training
        logger.info("Loading scalers_x and feature_columns from database...")
        scalers_data = load_scalers(db_config, production_model_training_run_id)
        processed_features_meta = load_processed_features_from_db(db_config, production_model_training_run_id)

        if not scalers_data or 'scalers_x' not in scalers_data:
            logger.error(f"Could not load scalers_x for run_id: {production_model_training_run_id}")
            return None
        if not processed_features_meta or 'feature_columns' not in processed_features_meta or 'tickers' not in processed_features_meta:
            logger.error(f"Could not load feature_columns or tickers for run_id: {production_model_training_run_id}")
            return None

        scalers_x = scalers_data['scalers_x'] # List of lists of MinMaxScaler objects
        feature_columns_prod_model = processed_features_meta['feature_columns']
        tickers_prod_model = processed_features_meta['tickers']
        num_stocks_prod_model = len(tickers_prod_model)
        num_features_prod_model = len(feature_columns_prod_model)

        if len(scalers_x) != num_stocks_prod_model or (num_stocks_prod_model > 0 and len(scalers_x[0]) != num_features_prod_model):
            logger.error("Mismatch in dimensions of loaded scalers_x vs feature_columns/tickers.")
            logger.error(f"Num stocks from tickers: {num_stocks_prod_model}, Num stocks in scalers_x: {len(scalers_x)}")
            if num_stocks_prod_model > 0:
                 logger.error(f"Num features from columns: {num_features_prod_model}, Num features in scalers_x[0]: {len(scalers_x[0]) if scalers_x else 'N/A'}")
            return None


        # 2. Fetch latest raw data window
        # Buffer for TA indicators (e.g., if max window for an indicator is 50, need 50 prior points)
        # This should ideally be configurable or derived from feature_engineering params.
        raw_data_lookback_days = sequence_length + 60 # Heuristic buffer
        logger.info(f"Fetching latest {raw_data_lookback_days} days of raw data for tickers: {tickers_prod_model}...")
        latest_raw_data_dict = get_latest_raw_data_window(db_config, tickers_prod_model, raw_data_lookback_days)

        if not latest_raw_data_dict or all(df.empty for df in latest_raw_data_dict.values()):
            logger.error("Failed to fetch any latest raw data.")
            return None

        # Filter out tickers for which no data was fetched
        valid_tickers_fetched = [t for t, df in latest_raw_data_dict.items() if not df.empty]
        if not valid_tickers_fetched:
            logger.error("No valid raw data fetched for any target ticker.")
            return None
        
        # Ensure the order of tickers for processing matches tickers_prod_model for consistency with scalers
        # Reconstruct latest_raw_data_dict to maintain order and include only relevant tickers
        ordered_raw_data_dict = {}
        for ticker in tickers_prod_model:
            if ticker in latest_raw_data_dict and not latest_raw_data_dict[ticker].empty:
                ordered_raw_data_dict[ticker] = latest_raw_data_dict[ticker]
            else:
                logger.error(f"Raw data for ticker {ticker} (expected by production model) is missing or empty. Cannot proceed.")
                return None # Or handle by predicting for subset, but for now, require all.
        
        current_num_stocks = len(ordered_raw_data_dict)
        if current_num_stocks != num_stocks_prod_model:
             logger.error(f"Data fetched for {current_num_stocks} stocks, but production model expects {num_stocks_prod_model}. Mismatch.")
             return None


        # 3. Apply technical indicators
        logger.info("Applying technical indicators to the latest raw data...")
        data_with_ta = add_technical_indicators(ordered_raw_data_dict.copy()) # .copy() is important

        # 4. Preprocess (e.g., handle NaNs from TAs, but 'Target' is not created here)
        # The preprocess_data from make_dataset creates 'Target'. We don't need target for prediction input.
        # We just need to ensure features are clean.
        data_preprocessed_for_pred = {}
        for ticker, df in data_with_ta.items():
            # Simplified preprocessing for prediction input: fill NaNs
            df_filled = df.ffill().bfill()
            # Ensure all feature_columns_prod_model are present
            missing_cols = [col for col in feature_columns_prod_model if col not in df_filled.columns]
            if missing_cols:
                logger.error(f"Ticker {ticker}: Missing expected feature columns after TA: {missing_cols}")
                return None
            data_preprocessed_for_pred[ticker] = df_filled[feature_columns_prod_model] # Select only relevant features

        # 5. Align data across tickers for the feature columns
        # Create a common index from the preprocessed data
        common_idx = pd.Index([])
        for df in data_preprocessed_for_pred.values():
            common_idx = common_idx.union(df.index)
        common_idx = common_idx.sort_values()

        if common_idx.empty:
            logger.error("Common index is empty after preprocessing for prediction. Cannot create input.")
            return None

        # Align each ticker's DataFrame to the common index
        aligned_feature_data_list = []
        for ticker in tickers_prod_model: # Iterate in the order of production model's tickers
            df = data_preprocessed_for_pred[ticker]
            df_aligned = df.reindex(common_idx).ffill().bfill() # Reindex and fill
            aligned_feature_data_list.append(df_aligned.values) # Get NumPy array of features

        # Stack into (timesteps, num_stocks, num_features)
        # np.stack creates a new axis, np.array directly if list elements are already correct shape
        # aligned_feature_data_list contains arrays of shape (timesteps, num_features)
        # We want to stack them along a new "stocks" axis.
        if not aligned_feature_data_list:
            logger.error("No data to stack after alignment.")
            return None
            
        # Transpose each (timesteps, num_features) to (num_features, timesteps) then stack, then transpose back
        # Or, more simply, create the final array and fill it
        final_feature_array = np.zeros((len(common_idx), num_stocks_prod_model, num_features_prod_model))
        for i in range(num_stocks_prod_model):
            final_feature_array[:, i, :] = aligned_feature_data_list[i]
        
        # 6. Extract the last 'sequence_length' timesteps
        if final_feature_array.shape[0] < sequence_length:
            logger.error(f"Not enough timesteps ({final_feature_array.shape[0]}) after processing to form a sequence of length {sequence_length}.")
            return None
        
        input_sequence_raw = final_feature_array[-sequence_length:, :, :] # (seq_len, num_stocks, num_features)
        logger.info(f"Raw input sequence for prediction shape: {input_sequence_raw.shape}")

        # 7. Scale the sequence using loaded scalers_x
        input_sequence_scaled = np.zeros_like(input_sequence_raw)
        for stock_idx in range(num_stocks_prod_model):
            for feature_idx in range(num_features_prod_model):
                scaler = scalers_x[stock_idx][feature_idx]
                feature_slice = input_sequence_raw[:, stock_idx, feature_idx].reshape(-1, 1)
                input_sequence_scaled[:, stock_idx, feature_idx] = scaler.transform(feature_slice).flatten()
        
        logger.info(f"Scaled input sequence shape: {input_sequence_scaled.shape}")

        # 8. Reshape for model input (add batch dimension)
        # Model expects (batch_size, seq_len, num_stocks, num_features)
        # Or (batch_size, seq_len, num_stocks * num_features) depending on model_definitions.py
        # Assuming your StockLSTM and variants take (batch, seq_len, num_stocks, num_features)
        # and then reshape internally if needed.
        # Let's check the model_definitions.py:
        # StockLSTM reshapes x.reshape(batch_size, seq_len, n_stocks * n_features)
        # StockLSTMWithAttention also reshapes to n_stocks * n_features
        # StockLSTMWithCrossStockAttention takes (batch, seq, n_stocks, n_features)
        # For safety, let's provide (1, seq_len, num_stocks, num_features)
        # The model training script prepares X_train_scaled with shape (samples, seq_len, num_stocks, num_features)
        # So, for prediction, we need (1, seq_len, num_stocks, num_features)
        
        model_input_sequence = np.expand_dims(input_sequence_scaled, axis=0) # (1, seq_len, num_stocks, num_features)
        logger.info(f"Final model input sequence shape: {model_input_sequence.shape}")

        # 9. Save to a temporary file and print path
        # Ensure output_dir exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"prediction_input_sequence_{timestamp}.npy"
        output_file_path = Path(output_dir) / output_filename
        
        np.save(output_file_path, model_input_sequence)
        logger.info(f"Saved prepared input sequence to: {output_file_path.resolve()}")
        
        return str(output_file_path.resolve())

    except Exception as e:
        logger.error(f"Error in run_prepare_input: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare input sequence for daily stock prediction.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/params.yaml',
        help='Path to the configuration file (e.g., config/params.yaml)'
    )
    parser.add_argument(
        '--production_model_training_run_id',
        type=str,
        required=True,
        help="The dataset run_id used when the current production model was trained (for loading correct scalers/features)."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/tmp/stock_prediction_inputs', # Default temporary directory
        help="Directory to save the output .npy file."
    )
    args = parser.parse_args()

    config_path_resolved = Path(args.config).resolve()
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)
    
    logger.info(f"Starting input preparation with config: {config_path_resolved}, prod model training run_id: {args.production_model_training_run_id}")

    output_file = run_prepare_input(str(config_path_resolved), args.production_model_training_run_id, args.output_dir)

    if output_file:
        logger.info(f"Successfully prepared input sequence: {output_file}")
        print(f"OUTPUT_PATH:{output_file}") # For Airflow XCom capture
    else:
        logger.error("Failed to prepare input sequence.")
        sys.exit(1)
```

**Key considerations for `prepare_prediction_input.py`:**

*   **Importing `add_technical_indicators`**:
    *   Ensure `src.data.make_dataset.py` defines `add_technical_indicators` at the module level so it can be imported.
    *   The `sys.path.append` line is a common way to handle imports when running scripts directly within a project structure. Airflow might require `PYTHONPATH` to be set correctly in its environment.
*   **`preprocess_data` for Prediction**: The `preprocess_data` function from `make_dataset.py` creates a 'Target' column. For prediction input, we don't have a target yet. So, the preprocessing here should focus on cleaning features (handling NaNs from TAs) and selecting the correct `feature_columns_prod_model`. I've adapted this in the code above.
*   **Buffer for TA**: The `raw_data_lookback_days = sequence_length + 60` is a heuristic. You might want to make the `60` (buffer) configurable or derive it from the maximum window size used in your `add_technical_indicators` function.
*   **Output Path**: The script prints `OUTPUT_PATH:<absolute_path_to_file>`. This specific format can be parsed by Airflow if you use `BashOperator` and need to pass the path via XComs.

---

**2. Refactor `src/models/predict_model.py`**

**Purpose:**
This script will now load the production model from MLflow Model Registry, consume the input sequence prepared by the previous script, make predictions, and save them appropriately.

**Thinking Process & Design Choices:**

1.  **Inputs:**
    *   `--config`: Path to `params.yaml`.
    *   `--input_sequence_path`: Absolute path to the `.npy` file generated by `prepare_prediction_input.py`.
    *   `--production_model_uri`: The MLflow Model Registry URI (e.g., `models:/MyStockPredictor/Production`). Airflow will provide this.
2.  **Core Logic:**
    *   **Load Model:** Use `mlflow.pytorch.load_model(model_uri=production_model_uri)`.
    *   **Get Associated Training `run_id`:** Extract the `run_id` from the loaded model's metadata (`mlflow.models.get_model_info(production_model_uri).run_id`). This `run_id` is the one that was used to *train this specific production model version*.
    *   **Load `y_scalers` and `tickers`:** Use this extracted `prod_model_training_run_id` to fetch the corresponding `y_scalers` (and `tickers` list if not part of `scalers_dict`) from the database. This ensures predictions are inverse-transformed correctly.
    *   **Load Input Sequence:** `np.load(input_sequence_path)`.
    *   **Predict & Inverse Transform.**
    *   **Save Outputs:**
        *   Update `data/predictions/latest_predictions.json`.
        *   Create `data/predictions/historical/{today_date}.json`.
        *   Save to `latest_predictions` table in PostgreSQL using `db_utils.save_prediction()`, passing the `prod_model_training_run_id` as the `model_run_id` argument.
3.  **Removal of Old Logic:** The script will no longer load test data or scalers from fixed file paths like `split_data_path` or `scalers_path`.

**Code for `src/models/predict_model.py` (Refactored):**

```python
# src/models/predict_model.py
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
# import joblib # No longer loading scalers via joblib from file
import torch
from datetime import date, datetime # Ensure datetime is imported
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient # For getting model info
import sys
import os

# Ensure 'src' is in PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from utils.db_utils import (
        load_scalers,
        save_prediction,
        load_processed_features_from_db # Fallback for tickers if not in scalers_dict
    )
    # Assuming model_definitions are not directly needed here as we load a full mlflow.pytorch model
    # If you were loading only state_dict, you'd need them.
except ImportError as e:
    logger.error(f"Failed to import db_utils: {e}") # Define logger first
    sys.exit(1)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run_daily_prediction(config_path: str, input_sequence_path: str, production_model_uri: str) -> bool:
    """
    Loads the production model, makes predictions on the prepared input sequence, and saves results.

    Args:
        config_path (str): Path to params.yaml.
        input_sequence_path (str): Path to the .npy file containing the prepared input sequence.
        production_model_uri (str): MLflow Model Registry URI for the production model.

    Returns:
        bool: True if prediction and saving were successful, False otherwise.
    """
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        db_config = cfg['database']
        mlflow_cfg = cfg['mlflow']
        
        # Output paths for JSON predictions (should be configurable, ensure they are writable)
        # These paths are relative to where the script is run or need to be absolute for containers
        # For Airflow, these might point to a shared volume /opt/airflow/data/predictions
        # Defaulting to paths that might work if script is run from project root.
        # In Airflow, you'd likely use absolute paths like /opt/airflow/data/predictions
        predictions_base_dir_str = cfg.get('output_paths', {}).get('predictions_dir', 'data/predictions')
        predictions_base_dir = Path(predictions_base_dir_str) # Convert to Path object
        
        # Ensure MLflow tracking URI is set
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
        if not mlflow_tracking_uri:
            logger.error("MLFLOW_TRACKING_URI is not set. Cannot connect to MLflow.")
            return False
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"Using MLflow Tracking URI: {mlflow_tracking_uri}")

        # 1. Load the production model from MLflow Model Registry
        logger.info(f"Loading production model from URI: {production_model_uri}")
        try:
            model = mlflow.pytorch.load_model(model_uri=production_model_uri)
            model.eval() # Set to evaluation mode
        except Exception as e_load_model:
            logger.error(f"Failed to load model from {production_model_uri}: {e_load_model}", exc_info=True)
            return False
        
        # 2. Get the training run_id associated with this production model
        # This run_id is for the DATASET the model was trained on.
        client = MlflowClient()
        try:
            # The model_uri for get_model_version_by_alias is like "models:/MyModelName" and alias "Production"
            # Or, if production_model_uri is already specific like "models:/MyModelName/Production"
            # we can parse it or use get_model_version
            model_name_from_uri = production_model_uri.split('/')[1] # "models:/<model_name>/Production"
            
            # Find the version associated with the "Production" alias/stage
            # This is a bit more robust than just getting run_id from model_info if URI is generic
            latest_prod_versions = client.get_latest_versions(name=model_name_from_uri, stages=["Production"])
            if not latest_prod_versions:
                logger.error(f"No version of model '{model_name_from_uri}' found in 'Production' stage.")
                # Fallback: try to get run_id directly if URI is specific to a version
                try:
                    model_version_details = client.get_model_version(name=model_name_from_uri, version=production_model_uri.split('/')[-1])
                    prod_model_mlflow_run_id = model_version_details.run_id
                except Exception:
                    logger.error(f"Could not determine MLflow run_id for production model {production_model_uri}")
                    return False
            else:
                # Assuming only one version is in "Production" for a given model name.
                # If multiple, this might need refinement (e.g., take highest version number).
                prod_model_mlflow_run_id = latest_prod_versions[0].run_id
            
            logger.info(f"Production model was trained with MLflow Run ID: {prod_model_mlflow_run_id}")
            
            # Now, we need the DATASET run_id. This should have been logged as a param to the MLflow run.
            mlflow_run_details = client.get_run(prod_model_mlflow_run_id)
            prod_model_dataset_run_id = mlflow_run_details.data.params.get("dataset_run_id")
            if not prod_model_dataset_run_id:
                logger.error(f"Parameter 'dataset_run_id' not found in MLflow Run {prod_model_mlflow_run_id}. Cannot load correct scalers.")
                return False
            logger.info(f"Production model's dataset_run_id (for scalers/features): {prod_model_dataset_run_id}")

        except Exception as e_mlflow_meta:
            logger.error(f"Error fetching metadata for production model {production_model_uri}: {e_mlflow_meta}", exc_info=True)
            return False

        # 3. Load y_scalers and tickers using the prod_model_dataset_run_id
        logger.info(f"Loading y_scalers and tickers using dataset_run_id: {prod_model_dataset_run_id}")
        scalers_dict = load_scalers(db_config, prod_model_dataset_run_id)
        if not scalers_dict or 'y_scalers' not in scalers_dict:
            logger.error(f"Failed to load y_scalers for dataset_run_id: {prod_model_dataset_run_id}")
            return False
        y_scalers = scalers_dict['y_scalers']
        
        tickers = scalers_dict.get('tickers')
        if not tickers: # Fallback
            processed_features_meta = load_processed_features_from_db(db_config, prod_model_dataset_run_id)
            if not processed_features_meta or 'tickers' not in processed_features_meta:
                logger.error(f"Failed to load tickers for dataset_run_id: {prod_model_dataset_run_id}")
                return False
            tickers = processed_features_meta['tickers']
        
        if len(y_scalers) != len(tickers):
            logger.error(f"Mismatch between number of y_scalers ({len(y_scalers)}) and tickers ({len(tickers)}).")
            return False
        logger.info(f"Loaded {len(y_scalers)} y_scalers for tickers: {tickers}")

        # 4. Load the input sequence
        logger.info(f"Loading input sequence from: {input_sequence_path}")
        if not Path(input_sequence_path).exists():
            logger.error(f"Input sequence file not found: {input_sequence_path}")
            return False
        input_sequence_np = np.load(input_sequence_path) # Shape: (1, seq_len, num_stocks, num_features)
        
        # Verify num_stocks in input sequence matches expected
        if input_sequence_np.shape[2] != len(tickers):
            logger.error(f"Input sequence has {input_sequence_np.shape[2]} stocks, but model/scalers expect {len(tickers)}.")
            return False

        input_tensor = torch.tensor(input_sequence_np, dtype=torch.float32)
        # Determine device (though model loaded from MLflow should handle its own device placement if saved correctly)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)
        # input_tensor = input_tensor.to(device)


        # 5. Make predictions
        logger.info("Making predictions...")
        with torch.no_grad():
            predictions_scaled_tensor = model(input_tensor) # Expected shape: (1, pred_len, num_stocks)
        predictions_scaled_np = predictions_scaled_tensor.cpu().numpy()
        # Assuming pred_len is 1, so shape is (1, 1, num_stocks)
        
        # 6. Inverse transform predictions
        logger.info("Inverse transforming predictions...")
        predicted_prices_final = {}
        # predictions_scaled_np[0, 0, stock_idx] gives the scaled prediction for that stock
        for stock_idx, ticker_name in enumerate(tickers):
            scaled_pred_value = predictions_scaled_np[0, 0, stock_idx]
            actual_pred_price = y_scalers[stock_idx].inverse_transform(np.array([[scaled_pred_value]]))[0][0]
            predicted_prices_final[ticker_name] = float(actual_pred_price)
        
        logger.info(f"Final predicted prices: {predicted_prices_final}")

        # 7. Save predictions to JSON files (for API/dashboard)
        predictions_base_dir.mkdir(parents=True, exist_ok=True)
        historical_dir = predictions_base_dir / 'historical'
        historical_dir.mkdir(parents=True, exist_ok=True)

        today_iso = date.today().isoformat()
        payload = {"date": today_iso, "predictions": predicted_prices_final, "model_mlflow_run_id": prod_model_mlflow_run_id}

        latest_file_path = predictions_base_dir / 'latest_predictions.json'
        with open(latest_file_path, 'w') as f:
            json.dump(payload, f, indent=4)
        logger.info(f"Saved latest predictions to {latest_file_path}")

        historical_file_path = historical_dir / f"{today_iso}.json"
        with open(historical_file_path, 'w') as f:
            json.dump(payload, f, indent=4)
        logger.info(f"Saved historical predictions to {historical_file_path}")

        # 8. Save predictions to PostgreSQL database
        logger.info("Saving predictions to PostgreSQL database...")
        for ticker_name, predicted_price_val in predicted_prices_final.items():
            save_prediction(db_config, ticker_name, predicted_price_val, prod_model_mlflow_run_id)
        logger.info("Successfully saved predictions to database.")
        
        return True

    except Exception as e:
        logger.error(f"Error in run_daily_prediction: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make daily stock predictions using the production model.")
    parser.add_argument(
        '--config', type=str, default='config/params.yaml',
        help='Path to the configuration file (e.g., config/params.yaml)'
    )
    parser.add_argument(
        '--input_sequence_path', type=str, required=True,
        help='Path to the .npy file containing the prepared input sequence for prediction.'
    )
    parser.add_argument(
        '--production_model_uri', type=str, required=True,
        help='MLflow Model Registry URI for the production model (e.g., "models:/MyModel/Production").'
    )
    args = parser.parse_args()

    config_path_resolved = Path(args.config).resolve()
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)
    
    input_sequence_path_resolved = Path(args.input_sequence_path).resolve()
    if not input_sequence_path_resolved.exists():
        logger.error(f"Input sequence file not found: {input_sequence_path_resolved}")
        sys.exit(1)

    logger.info(f"Starting daily prediction with config: {config_path_resolved}, input: {input_sequence_path_resolved}, model: {args.production_model_uri}")

    success = run_daily_prediction(str(config_path_resolved), str(input_sequence_path_resolved), args.production_model_uri)

    if success:
        logger.info("Daily prediction process completed successfully.")
    else:
        logger.error("Daily prediction process failed.")
        sys.exit(1)
```

---

**3. Create `src/models/monitor_performance.py`**

**Purpose:**
This script evaluates the previous day's predictions against actual market outcomes, logs these performance metrics, and determines if model retraining should be triggered based on predefined thresholds.

**Thinking Process & Design Choices:**

1.  **Inputs:**
    *   `--config`: Path to `params.yaml`.
2.  **Core Logic:**
    *   **Load Configs:** `db_config`, `tickers_list`, `performance_thresholds`, `evaluation_lag_days`.
    *   **Determine Evaluation Date:** `prediction_date_to_evaluate = today - evaluation_lag_days`. This is the date for which we have predictions and should now have actuals.
    *   **Fetch Predictions:** For each ticker, get the prediction made for `prediction_date_to_evaluate` from the `latest_predictions` DB table. This also gives us the `model_mlflow_run_id` of the model that made that prediction.
    *   **Fetch Actuals:** For each ticker, get the actual closing price for `prediction_date_to_evaluate` from the `raw_stock_data` DB table. Also fetch the actual price for `prediction_date_to_evaluate - 1 day` for directional accuracy.
    *   **Calculate Metrics:** MAE, RMSE, MAPE, Directional Accuracy.
    *   **Save Metrics:** Store these calculated metrics in the `model_performance_log` DB table using `db_utils.save_daily_performance_metrics()`.
    *   **Check Thresholds:** Aggregate performance (e.g., average MAPE across all tickers, or check if *any* ticker breaches its individual threshold if you define per-ticker thresholds). Compare against `performance_thresholds` from `params.yaml`.
3.  **Output for Airflow Branching:**
    *   The script will print a specific string (e.g., `trigger_retraining_pipeline` or `no_retraining_needed`) to `stdout`. Airflow's `BranchPythonOperator` will use this string to decide which downstream task to execute.
4.  **Directional Accuracy Detail:**
    *   `Predicted Direction`: `sign(predicted_price_on_eval_date - actual_price_on_eval_date_minus_1)`.
    *   `Actual Direction`: `sign(actual_price_on_eval_date - actual_price_on_eval_date_minus_1)`.
    *   Accuracy is when these signs match.

**Code for `src/models/monitor_performance.py`:**

```python
# src/models/monitor_performance.py
import argparse
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import date, timedelta, datetime # Ensure all are imported
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Ensure 'src' is in PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from utils.db_utils import (
        get_latest_predictions, # To get predictions for the evaluation date
        save_daily_performance_metrics,
        load_data_from_db # Can be used to get specific actual prices
    )
except ImportError as e:
    logger.error(f"Failed to import db_utils: {e}") # Define logger first
    sys.exit(1)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def calculate_directional_accuracy(predicted_today: float, actual_today: float, actual_yesterday: float) -> Optional[float]:
    """Calculates directional accuracy. Returns 1.0 if direction matches, 0.0 otherwise, None if undefined."""
    if pd.isna(predicted_today) or pd.isna(actual_today) or pd.isna(actual_yesterday):
        return None

    pred_change = predicted_today - actual_yesterday
    actual_change = actual_today - actual_yesterday

    # Avoid division by zero or issues if prices are identical
    if pred_change == 0 and actual_change == 0: # No change predicted, no change happened
        return 1.0 
    if pred_change == 0 or actual_change == 0: # One changed, other didn't - ambiguous, count as mismatch or neutral
        return 0.0 # Or handle as 0.5 if preferred for neutrality

    return 1.0 if np.sign(pred_change) == np.sign(actual_change) else 0.0


def run_monitoring(config_path: str) -> str:
    """
    Evaluates previous day's predictions, logs performance, and decides if retraining is needed.

    Args:
        config_path (str): Path to params.yaml.

    Returns:
        str: Task ID for Airflow branching (e.g., 'trigger_retraining_task' or 'no_retraining_task').
    """
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        db_config = params['database']
        # Tickers to monitor - should align with what's being predicted
        # This could come from params['data_loading']['tickers'] or dynamically from latest_predictions table
        tickers_to_monitor = params['data_loading']['tickers'] 
        
        monitoring_cfg = params['monitoring']
        thresholds = monitoring_cfg['performance_thresholds']
        evaluation_lag_days = monitoring_cfg.get('evaluation_lag_days', 1)

        airflow_dag_cfg = params.get('airflow_dags', {})
        retraining_task_id = airflow_dag_cfg.get('retraining_trigger_task_id', 'trigger_retraining_pipeline_task') # Task ID in daily DAG
        no_retraining_task_id = airflow_dag_cfg.get('no_retraining_task_id', 'no_retraining_needed_task')


        prediction_date_to_evaluate = date.today() - timedelta(days=evaluation_lag_days)
        prediction_date_str = prediction_date_to_evaluate.isoformat()
        
        # For directional accuracy, we need actuals from one day prior to prediction_date_to_evaluate
        actuals_needed_end_date = prediction_date_to_evaluate
        actuals_needed_start_date = prediction_date_to_evaluate - timedelta(days=1) # Need at least two days of actuals

        logger.info(f"Monitoring performance for predictions made for date: {prediction_date_str}")

        # 1. Fetch predictions made for the evaluation_date
        # get_latest_predictions returns a dict {ticker: {'timestamp': ..., 'predicted_price': ..., 'model_run_id': ...}}
        # We need to filter this for the specific prediction_date_to_evaluate if it stores more than one day.
        # For now, assuming get_latest_predictions gives the most recent, which should be for 'today's prediction'
        # So, if evaluation_lag_days=1, we need predictions made 'yesterday' for 'yesterday'.
        # Let's adjust: fetch predictions that were *recorded* around prediction_date_to_evaluate
        # This part needs careful alignment with how `latest_predictions` table is populated.
        # Assuming `latest_predictions` stores the prediction for `prediction_date_to_evaluate` made on that day or day before.
        
        # Simpler: Assume `latest_predictions` table has one row per ticker, updated daily.
        # The `prediction_timestamp` in that table tells us when the prediction was stored.
        # The `payload["date"]` in predict_model.py tells us *for which date* the prediction is.
        # We need to query `latest_predictions` where the *prediction target date* (not timestamp) matches.
        # This requires `latest_predictions` to store the target date of the prediction.
        # Let's assume `db_utils.get_latest_predictions()` is adapted or we query directly.
        # For now, let's assume `get_latest_predictions` gives us the predictions we need to evaluate.
        # A robust way: query historical JSONs or adapt DB to store target_prediction_date.
        
        # Let's refine: We need to fetch predictions that were *targeted* for `prediction_date_to_evaluate`.
        # The `latest_predictions.json` and historical files store this.
        # The `latest_predictions` DB table should ideally store `target_prediction_date`.
        # If `latest_predictions` DB table's `prediction_timestamp` is the *time of prediction*,
        # and we assume predictions are made for `date.today()`, then for `evaluation_lag_days=1`,
        # we look for predictions made `evaluation_lag_days` ago.
        # This is getting complex. Let's simplify:
        # Assume the `latest_predictions.json` file (or DB equivalent) correctly stores predictions
        # with a "date" field indicating the date the prediction is FOR.

        # We will load the historical JSON for the prediction_date_to_evaluate
        # This is simpler than complex DB queries for now.
        predictions_base_dir_str = params.get('output_paths', {}).get('predictions_dir', 'data/predictions')
        historical_pred_file = Path(predictions_base_dir_str) / 'historical' / f"{prediction_date_str}.json"
        
        predictions_to_evaluate = {}
        model_mlflow_run_id_for_these_preds = None
        if historical_pred_file.exists():
            with open(historical_pred_file, 'r') as f:
                pred_data = json.load(f)
            if pred_data.get("date") == prediction_date_str:
                predictions_to_evaluate = pred_data.get("predictions", {})
                model_mlflow_run_id_for_these_preds = pred_data.get("model_mlflow_run_id")
            else:
                logger.warning(f"Date mismatch in {historical_pred_file}. Expected {prediction_date_str}, got {pred_data.get('date')}")
        else:
            logger.warning(f"Prediction file not found for {prediction_date_str} at {historical_pred_file}")

        if not predictions_to_evaluate or not model_mlflow_run_id_for_these_preds:
            logger.warning(f"No predictions found or model_run_id missing for {prediction_date_str}. Skipping performance monitoring for this date.")
            return no_retraining_task_id # Default to no retraining if no data to evaluate

        # 2. Fetch actual prices for evaluation_date and (evaluation_date - 1 day)
        logger.info(f"Fetching actual prices for {tickers_to_monitor} around {prediction_date_str}...")
        # Use load_data_from_db and filter for specific dates
        # This is inefficient if called repeatedly. A dedicated function in db_utils would be better.
        # `get_actual_prices_for_dates(db_config, tickers, date_list)`
        
        # For simplicity, let's assume we can query raw_stock_data directly here for the two dates.
        actual_prices_on_eval_date = {}
        actual_prices_on_prev_date = {}
        
        conn = None # Ensure conn is defined for finally block
        try:
            conn = db_utils.get_db_connection(db_config) # Assuming db_utils is imported
            cursor = conn.cursor()
            for ticker in tickers_to_monitor:
                # Actual for evaluation date
                cursor.execute(
                    "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
                    (ticker, prediction_date_to_evaluate)
                )
                result = cursor.fetchone()
                if result: actual_prices_on_eval_date[ticker] = result[0]

                # Actual for previous day (for directional accuracy)
                prev_business_day = prediction_date_to_evaluate - timedelta(days=1) # Simplistic, doesn't handle weekends/holidays well
                # A robust solution would use pandas market holiday calendars or fetch a small window and pick latest before eval_date
                cursor.execute(
                    "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
                    (ticker, prev_business_day) 
                )
                result_prev = cursor.fetchone()
                if result_prev: actual_prices_on_prev_date[ticker] = result_prev[0]
        finally:
            if conn:
                cursor.close()
                conn.close()

        all_metrics = []
        retrain_triggered_by_metric = False

        for ticker in tickers_to_monitor:
            predicted_price = predictions_to_evaluate.get(ticker)
            actual_price = actual_prices_on_eval_date.get(ticker)
            actual_prev_price = actual_prices_on_prev_date.get(ticker)

            if predicted_price is None or actual_price is None:
                logger.warning(f"Missing predicted or actual price for {ticker} on {prediction_date_str}. Skipping metrics calculation for it.")
                continue

            metrics = {'ticker': ticker, 'prediction_date': prediction_date_str, 'predicted_price': predicted_price, 'actual_price': actual_price}
            metrics['mae'] = mean_absolute_error([actual_price], [predicted_price])
            metrics['rmse'] = np.sqrt(mean_squared_error([actual_price], [predicted_price]))
            if actual_price != 0:
                metrics['mape'] = mean_absolute_percentage_error([actual_price], [predicted_price])
            else:
                metrics['mape'] = np.nan # Avoid division by zero

            metrics['direction_accuracy'] = calculate_directional_accuracy(predicted_price, actual_price, actual_prev_price) if actual_prev_price is not None else np.nan
            
            logger.info(f"Metrics for {ticker} on {prediction_date_str}: MAPE={metrics['mape']:.4f}, DirAcc={metrics['direction_accuracy']}")
            all_metrics.append(metrics)

            # Save individual ticker metrics to DB
            save_daily_performance_metrics(db_config, prediction_date_str, ticker, metrics, model_mlflow_run_id_for_these_preds)

            # Check thresholds for this ticker
            if not pd.isna(metrics['mape']) and metrics['mape'] > thresholds.get('mape_max', float('inf')):
                logger.warning(f"TRIGGER: {ticker} MAPE ({metrics['mape']:.4f}) exceeded threshold ({thresholds.get('mape_max')}).")
                retrain_triggered_by_metric = True
            if not pd.isna(metrics['direction_accuracy']) and metrics['direction_accuracy'] < thresholds.get('direction_accuracy_min', float('-inf')):
                logger.warning(f"TRIGGER: {ticker} Directional Accuracy ({metrics['direction_accuracy']:.4f}) below threshold ({thresholds.get('direction_accuracy_min')}).")
                retrain_triggered_by_metric = True
            # Add checks for other metrics (RMSE, MAE) if defined in thresholds

        if not all_metrics:
            logger.warning("No metrics were calculated for any ticker. Defaulting to no retraining.")
            return no_retraining_task_id

        # Optional: Aggregate metrics (e.g., average MAPE across all monitored tickers)
        # avg_mape = np.nanmean([m['mape'] for m in all_metrics if 'mape' in m])
        # logger.info(f"Average MAPE across all tickers: {avg_mape:.4f}")
        # if avg_mape > thresholds.get('avg_mape_max', float('inf')): # Example for aggregated threshold
        #     logger.warning(f"TRIGGER: Average MAPE ({avg_mape:.4f}) exceeded threshold.")
        #     retrain_triggered_by_metric = True
            
        if retrain_triggered_by_metric:
            logger.info("Performance thresholds breached. Triggering retraining pipeline.")
            return retraining_task_id
        else:
            logger.info("Model performance is within acceptable thresholds. No retraining triggered.")
            return no_retraining_task_id

    except Exception as e:
        logger.error(f"Error in run_monitoring: {e}", exc_info=True)
        # Default to no retraining on error to avoid unintended retraining loops
        return params.get('airflow_dags', {}).get('no_retraining_task_id', 'no_retraining_needed_task_on_error')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Monitor model performance and decide on retraining.")
    parser.add_argument(
        '--config', type=str, default='config/params.yaml',
        help='Path to the configuration file (e.g., config/params.yaml)'
    )
    args = parser.parse_args()

    config_path_resolved = Path(args.config).resolve()
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)

    logger.info(f"Starting model performance monitoring with config: {config_path_resolved}")
    
    next_task_id = run_monitoring(str(config_path_resolved))
    
    logger.info(f"Monitoring complete. Next Airflow task to run: {next_task_id}")
    print(f"NEXT_TASK_ID:{next_task_id}") # For Airflow BranchPythonOperator
```

**Key Considerations for `monitor_performance.py`:**

*   **Fetching Predictions for Evaluation:** I've opted to load predictions from the historical JSON files. This is simpler for now than assuming a complex query on the `latest_predictions` DB table if it only stores the *absolute latest* prediction. If your `latest_predictions` table is designed to hold predictions for specific target dates, you can adapt to query that.
*   **Fetching Actual Prices:** The current DB query for actual prices is basic. For robustness against non-trading days when calculating `actual_prev_price`, you might need a more sophisticated way to get the "last available closing price before `prediction_date_to_evaluate`". Pandas `bdate_range` or querying a small window and taking the last could work.
*   **Thresholds:** The logic checks if *any* ticker breaches an individual metric threshold. You could also implement logic for aggregated metrics (e.g., average MAPE across all stocks).
*   **Airflow Task IDs:** The script returns task IDs defined in `params.yaml` for branching.

---

**Next Steps After Implementing Phase 2 Scripts:**

1.  **Thoroughly Test Each Script Individually:**
    *   **`prepare_prediction_input.py`**:
        *   Manually find a `production_model_training_run_id` from your DB (a `run_id` in `processed_feature_data` that also has entries in `scalers`).
        *   Run: `python src/features/prepare_prediction_input.py --config config/params.yaml --production_model_training_run_id <YOUR_ID> --output_dir /tmp/pred_inputs`
        *   Check if the `.npy` file is created in `/tmp/pred_inputs` and inspect its shape.
    *   **`predict_model.py`**:
        *   You'll need an MLflow model registered and in "Production" (or specify a direct run URI for testing). For initial testing, you can use a `runs:/<mlflow_run_id_of_a_trained_model>/model` URI.
        *   Use the `.npy` file from the previous step.
        *   Run: `python src/models/predict_model.py --config config/params.yaml --input_sequence_path /tmp/pred_inputs/prediction_input_sequence_....npy --production_model_uri models:/YourModelName/Production` (or the `runs:/...` URI).
        *   Check `data/predictions/` for JSON files and the `latest_predictions` DB table.
    *   **`monitor_performance.py`**:
        *   Ensure you have some predictions in `data/predictions/historical/` for a past date for which you also have actuals in `raw_stock_data`.
        *   Run: `python src/models/monitor_performance.py --config config/params.yaml`
        *   Check logs, the `model_performance_log` DB table, and the printed `NEXT_TASK_ID`.
2.  **Refine and Debug:** Expect to iterate and debug. Pay close attention to paths, database interactions, and data shapes.