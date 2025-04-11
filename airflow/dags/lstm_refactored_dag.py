# airflow/dags/lstm_refactored_dag.py
from __future__ import annotations

import pendulum
from pathlib import Path

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# Define project paths used within the container
# These should align with your docker-compose volume mounts and params.yaml
PROJECT_ROOT = Path('/opt/airflow')
CONFIG_PATH = PROJECT_ROOT / 'config/params.yaml'
MAKE_DATASET_SCRIPT = PROJECT_ROOT / 'src/data/make_dataset.py'
BUILD_FEATURES_SCRIPT = PROJECT_ROOT / 'src/features/build_features.py'
OPTIMIZE_SCRIPT = PROJECT_ROOT / 'src/models/optimize_hyperparams.py'
TRAIN_SCRIPT = PROJECT_ROOT / 'src/models/train_model.py'

# Define default arguments
default_args = {
    'owner': 'airflow_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=2), # Shorter delay for quicker retries if needed
}

with DAG (
    dag_id='lstm_stock_prediction_refactored',
    default_args=default_args,
    description='Refactored pipeline for LSTM stock prediction',
    schedule='0 8 * * 1', # Weekly Monday 8 AM UTC
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=['ml', 'lstm', 'refactored', 'mlflow', 'pytorch', 'optuna'],
) as dag:

    task_process_data = BashOperator(
        task_id='process_data',
        bash_command=f'python {MAKE_DATASET_SCRIPT} --config {CONFIG_PATH}',
        doc_md="Runs data loading, feature addition, filtering, and processing. Saves intermediate data.",
    )

    task_build_features = BashOperator(
        task_id='build_features',
        bash_command=f'python {BUILD_FEATURES_SCRIPT} --config {CONFIG_PATH}',
        doc_md="Loads processed data, creates sequences, splits train/test, scales data, and saves features and scalers.",
    )

    task_optimize_hyperparams = BashOperator(
        task_id='optimize_hyperparams',
        bash_command=f'python {OPTIMIZE_SCRIPT} --config {CONFIG_PATH}',
        doc_md="Loads features, runs Optuna hyperparameter optimization, and saves the best parameters to a JSON file.",
    )

    task_train_final_model = BashOperator(
        task_id='train_final_model',
        bash_command=f'python {TRAIN_SCRIPT} --config {CONFIG_PATH}',
        # Ensure MLflow URI is available in the environment (should be from docker-compose)
        # env={'MLFLOW_TRACKING_URI': '{{ var.value.get("mlflow_tracking_uri", "http://localhost:5001/") }}'}, --> This is the problem!!! Remove it
        doc_md="Loads features, scalers, and best params. Trains the final model, evaluates, visualizes, and logs results (params, metrics, model artifact) to MLflow.",
    )

    # Define task dependencies
    task_process_data >> task_build_features >> task_optimize_hyperparams >> task_train_final_model