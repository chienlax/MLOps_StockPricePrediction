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

Let me know when you're ready to move to the next part or if you have questions about these steps!