# airflow/dags/lstm_refactored_dag.py
from __future__ import annotations

import pendulum
from pathlib import Path
import os
import subprocess 
import logging

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator # We'll still use it for some tasks
from airflow.operators.python import PythonOperator

# Setup logging for Airflow context
log = logging.getLogger(__name__)

# Define project paths used within the container
PROJECT_ROOT = Path('/opt/airflow') # This is the root *inside* the Airflow worker container
CONFIG_PATH = PROJECT_ROOT / 'config/params.yaml'
# Python scripts are in /opt/airflow/src/... inside the container due to volume mounts
MAKE_DATASET_SCRIPT = PROJECT_ROOT / 'src/data/make_dataset.py'
BUILD_FEATURES_SCRIPT = PROJECT_ROOT / 'src/features/build_features.py'
OPTIMIZE_SCRIPT = PROJECT_ROOT / 'src/models/optimize_hyperparams.py'
TRAIN_SCRIPT = PROJECT_ROOT / 'src/models/train_model.py'

# Function to initialize the database
def callable_initialize_database(**kwargs):
    """Python callable to initialize the PostgreSQL database."""
    import yaml
    # Since this runs in Airflow, direct import of your project's utils should work
    # if /opt/airflow (PROJECT_ROOT) is effectively in PYTHONPATH or structure allows.
    # To be safe, ensure PYTHONPATH includes /opt/airflow or use relative imports carefully.
    try:
        from src.utils.db_utils import setup_database
    except ImportError:
        # Adjust sys.path if running in an environment where src isn't directly visible
        # This might happen depending on how Airflow's Python environment is set up
        # and how modules are discovered.
        import sys
        sys.path.insert(0, str(PROJECT_ROOT)) # Add project root to path
        from src.utils.db_utils import setup_database


    log.info(f"Attempting to load config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    db_config = config['database']
    log.info(f"DATABASE CONFIG LOADED for initialization: {db_config.get('host')}:{db_config.get('port')}/{db_config.get('dbname')}")
    setup_database(db_config)
    log.info("Database initialization callable finished.")
    return "Database initialized"

# Python callable for task_process_data
def callable_run_make_dataset_full_process(**kwargs):
    """
    Calls make_dataset.py with --mode full_process and pushes the run_id to XCom.
    """
    script_path = str(MAKE_DATASET_SCRIPT)
    config_file_path = str(CONFIG_PATH)
    command = [
        "python", script_path,
        "--config", config_file_path,
        "--mode", "full_process",
    ]
    
    log.info(f"Executing command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    log.info("make_dataset.py STDOUT:")
    log.info(stdout)
    
    if process.returncode != 0:
        log.error("make_dataset.py STDERR:")
        log.error(stderr)
        raise Exception(f"make_dataset.py failed with return code {process.returncode}. Error: {stderr}")

    # Extract RUN_ID from stdout
    # Assuming make_dataset.py prints "RUN_ID:<actual_id>" as the last relevant line or clearly.
    run_id_line = [line for line in stdout.splitlines() if line.startswith("RUN_ID:")]
    if not run_id_line:
        log.error("Could not find RUN_ID: in the output of make_dataset.py.")
        log.error(f"Full stdout for make_dataset.py:\n{stdout}")
        raise Exception("make_dataset.py did not output RUN_ID.")
    
    # Get the last RUN_ID line in case there are multiple debug prints
    actual_run_id = run_id_line[-1].split("RUN_ID:")[1].strip()
    log.info(f"Extracted DATASET_RUN_ID: {actual_run_id}")
    
    # Push to XCom. The key will be 'return_value' by default if not specified.
    # Or explicitly:
    kwargs['ti'].xcom_push(key='dataset_run_id', value=actual_run_id)
    return actual_run_id # Also returned by the PythonOperator

# Define default arguments
default_args = {
    'owner': 'airflow_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': pendulum.duration(seconds=15), # Increased slightly for retries
}

with DAG(
    dag_id='temp_phase1_retraining_pipeline', # Renamed for clarity during this testing phase
    default_args=default_args,
    description='(Temporary) Pipeline to test Phase 1 refactored scripts for LSTM stock prediction. Uses XComs for run_id.',
    schedule=None,  # Manually triggered for testing
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=['ml', 'lstm', 'database', 'mlflow', 'pytorch', 'optuna', 'phase1_test'],
) as dag:
    
    task_init_db = PythonOperator(
        task_id='initialize_database',
        python_callable=callable_initialize_database,
    )
    
    task_process_data_full = PythonOperator(
        task_id='process_data_full_and_get_run_id',
        python_callable=callable_run_make_dataset_full_process,
        # provide_context=True, # already default for PythonOperator
        doc_md="Runs make_dataset.py in 'full_process' mode. Captures and pushes 'dataset_run_id' to XComs.",
    )
    
    # Subsequent tasks use BashOperator and pull 'dataset_run_id' from XComs
    # The key 'dataset_run_id' was explicitly set in task_process_data_full
    dataset_run_id_xcom = "{{ ti.xcom_pull(task_ids='process_data_full_and_get_run_id', key='dataset_run_id') }}"

    task_build_features = BashOperator(
        task_id='build_features',
        bash_command=f"python {BUILD_FEATURES_SCRIPT} --config {CONFIG_PATH} --run_id {dataset_run_id_xcom}",
        doc_md="Builds features using the dataset_run_id from the previous step.",
    )
    
    task_optimize_hyperparams = BashOperator(
        task_id='optimize_hyperparams',
        bash_command=f"python {OPTIMIZE_SCRIPT} --config {CONFIG_PATH} --run_id {dataset_run_id_xcom}",
        doc_md="Optimizes hyperparameters using the dataset_run_id.",
    )
    
    task_train_final_model = BashOperator(
        task_id='train_final_model',
        bash_command=f"python {TRAIN_SCRIPT} --config {CONFIG_PATH} --run_id {dataset_run_id_xcom}",
        env={
            'MLFLOW_TRACKING_URI': os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000'),
            # Set PYTHONPATH if necessary for your scripts to find modules within /opt/airflow/src
            'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"
        },
        doc_md="Trains the final model using the dataset_run_id and logs to MLflow.",
    )
    
    # Define task dependencies
    task_init_db >> task_process_data_full
    task_process_data_full >> task_build_features
    task_build_features >> task_optimize_hyperparams
    task_optimize_hyperparams >> task_train_final_model