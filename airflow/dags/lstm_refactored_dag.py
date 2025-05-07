# airflow/dags/lstm_refactored_dag.py
from __future__ import annotations

import pendulum
from pathlib import Path
import os

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from src.utils.db_utils import setup_database

# Define project paths used within the container
# These should align with your docker-compose volume mounts and params.yaml
PROJECT_ROOT = Path('/opt/airflow')
CONFIG_PATH = PROJECT_ROOT / 'config/params.yaml'
MAKE_DATASET_SCRIPT = PROJECT_ROOT / 'src/data/make_dataset.py'
BUILD_FEATURES_SCRIPT = PROJECT_ROOT / 'src/features/build_features.py'
OPTIMIZE_SCRIPT = PROJECT_ROOT / 'src/models/optimize_hyperparams.py'
TRAIN_SCRIPT = PROJECT_ROOT / 'src/models/train_model.py'

# Function to initialize the database
def initialize_database(**kwargs):
    """Initialize the PostgreSQL database with all required tables."""
    import yaml
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    db_config = config['database']
    setup_database(db_config)
    return "Database initialized"

# Define default arguments
default_args = {
    'owner': 'airflow_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=2),  # Shorter delay for quicker retries if needed
}

with DAG(
    dag_id='lstm_stock_prediction_db',
    default_args=default_args,
    description='Database-backed pipeline for LSTM stock prediction',
    schedule='0 8 * * *',  # Weekly Monday 8 AM UTC
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=['ml', 'lstm', 'database', 'mlflow', 'pytorch', 'optuna'],
) as dag:
    
    # Initialize database task
    task_init_db = PythonOperator(
        task_id='initialize_database',
        python_callable=initialize_database,
        doc_md="Initializes the PostgreSQL database with all required tables for the pipeline.",
    )
    
    # Update command to include DB_PATH as environment variable
    task_process_data = BashOperator(
        task_id='process_data',
        bash_command=f'python {MAKE_DATASET_SCRIPT} --config {CONFIG_PATH}',
        doc_md="Runs data loading, feature addition, filtering, and processing. Saves data to PostgreSQL.",
    )
    
    task_build_features = BashOperator(
        task_id='build_features',
        bash_command=f'python {BUILD_FEATURES_SCRIPT} --config {CONFIG_PATH}',
        doc_md="Loads processed data from PostgreSQL, creates sequences, splits train/test, scales data.",
    )
    
    task_optimize_hyperparams = BashOperator(
        task_id='optimize_hyperparams',
        bash_command=f'python {OPTIMIZE_SCRIPT} --config {CONFIG_PATH}',
        doc_md="Loads features from PostgreSQL, runs Optuna optimization, saves best parameters.",
    )
    
    task_train_final_model = BashOperator(
        task_id='train_final_model',
        bash_command=f'python {TRAIN_SCRIPT} --config {CONFIG_PATH}',
        env={
            'MLFLOW_TRACKING_URI': os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        },
        doc_md="Loads features, scalers, and best params from PostgreSQL. Trains model and logs to MLflow.",
    )
    
    # Define task dependencies with database initialization first
    task_init_db >> task_process_data >> task_build_features >> task_optimize_hyperparams >> task_train_final_model
    