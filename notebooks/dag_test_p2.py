# airflow/dags/temp_phase2_daily_ops_test_dag.py
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import mlflow  # For MlflowClient interaction
import pendulum
import yaml
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

# --- Define Constants and Paths ---
PROJECT_ROOT = Path('/opt/airflow')
CONFIG_PATH = PROJECT_ROOT / 'config/params.yaml'

# Script paths
MAKE_DATASET_SCRIPT = PROJECT_ROOT / 'src/data/make_dataset.py'
PREPARE_INPUT_SCRIPT = PROJECT_ROOT / 'src/features/prepare_prediction_input.py'
PREDICT_MODEL_SCRIPT = PROJECT_ROOT / 'src/models/predict_model.py'
MONITOR_PERFORMANCE_SCRIPT = PROJECT_ROOT / 'src/models/monitor_performance.py'
PREDICTION_INPUT_TEMP_DIR = "/tmp/daily_airflow_pred_inputs"

# --- Python Callable for Getting Production Model Info ---
def callable_get_production_model_details(**kwargs):
    """
    Queries MLflow Model Registry for the model version pointed to by the "Production" alias.
    Extracts its URI for loading, base name, version number, and the dataset_run_id 
    it was trained with. Pushes these to XComs.
    """
    ti = kwargs['ti']
    log.info("Starting callable_get_production_model_details...")

    with open(CONFIG_PATH, 'r') as f:
        params = yaml.safe_load(f)
    
    mlflow_cfg = params.get('mlflow', {})
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
    # Use experiment_name as the registered_model_name by convention from train_model.py
    base_registered_model_name = mlflow_cfg.get('experiment_name') 

    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI is not configured in params.yaml or environment.")
    if not base_registered_model_name:
        raise ValueError("mlflow.experiment_name (used as registered_model_name) not configured in params.yaml.")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    
    production_alias = "Production" # The alias we are looking for

    try:
        log.info(f"Querying MLflow for model '{base_registered_model_name}' with alias '{production_alias}'...")
        model_version_obj = client.get_model_version_by_alias(
            name=base_registered_model_name, 
            alias=production_alias
        )
        
        production_model_uri_for_loading = f"models:/{base_registered_model_name}@{production_alias}"
        production_version_number_str = model_version_obj.version
        prod_model_origin_mlflow_run_id = model_version_obj.run_id # MLflow Run ID of the training job
        
        log.info(f"Found Production Model: URI='{production_model_uri_for_loading}', Version='{production_version_number_str}', Training MLflow Run ID='{prod_model_origin_mlflow_run_id}'")

        mlflow_run_details = client.get_run(prod_model_origin_mlflow_run_id)
        prod_model_training_dataset_run_id = mlflow_run_details.data.params.get("dataset_run_id")
        
        if not prod_model_training_dataset_run_id:
            raise ValueError(f"Parameter 'dataset_run_id' not found in MLflow Run '{prod_model_origin_mlflow_run_id}' for the production model. This is required to load correct scalers/features.")
        
        log.info(f"Production model's training dataset_run_id (for scalers/features): {prod_model_training_dataset_run_id}")

    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) and f"alias '{production_alias}'" in str(e):
            log.error(f"CRITICAL: Alias '{production_alias}' does not exist for model '{base_registered_model_name}'. Please ensure a model version is aliased as '{production_alias}' in MLflow Model Registry.")
            raise # Re-raise to fail the task
        else:
            log.error(f"Error querying MLflow for production model using alias '{production_alias}': {e}", exc_info=True)
            raise
    except Exception as e:
        log.error(f"Unexpected error in callable_get_production_model_details: {e}", exc_info=True)
        raise

    ti.xcom_push(key='production_model_uri_for_loading', value=production_model_uri_for_loading)
    ti.xcom_push(key='production_model_base_name', value=base_registered_model_name)
    ti.xcom_push(key='production_model_version_number', value=production_version_number_str)
    ti.xcom_push(key='production_model_training_dataset_run_id', value=prod_model_training_dataset_run_id)
    
    log.info("Successfully pushed production model details to XComs.")
    return "Production model details fetched and pushed to XComs."


# --- Default Arguments for the DAG ---
default_args = {
    'owner': 'airflow_test_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': pendulum.duration(seconds=30),
}

# --- DAG Definition ---
with DAG(
    dag_id='temp_phase2_daily_operations_test_v2', # New version ID for clarity
    default_args=default_args,
    description='(Temporary V2) Tests Phase 2 daily operational scripts with clearer XCom handling.',
    schedule=None, # Manually triggerable for testing
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=['ml', 'daily_ops_test', 'phase2_test_v2'],
) as dag:

    # Common environment variables for BashOperator tasks
    common_env_vars = {
        'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}",
        'MLFLOW_TRACKING_URI': os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
    }
    # Load MLFLOW_TRACKING_URI from params.yaml if needed for common_env_vars
    try:
        with open(CONFIG_PATH, 'r') as f:
            params_for_env = yaml.safe_load(f)
        common_env_vars['MLFLOW_TRACKING_URI'] = os.environ.get(
            'MLFLOW_TRACKING_URI', 
            params_for_env.get('mlflow', {}).get('tracking_uri', 'http://mlflow-server:5000')
        )
    except Exception as e_params_load:
        log.warning(f"Could not load MLFLOW_TRACKING_URI from params.yaml for common_env_vars, using default/env: {e_params_load}")


    task_fetch_incremental_data = BashOperator(
        task_id='fetch_incremental_raw_data',
        bash_command=f"python {MAKE_DATASET_SCRIPT} --config {CONFIG_PATH} --mode incremental_fetch",
        env=common_env_vars,
        doc_md="Fetches the latest raw stock data incrementally."
    )

    task_get_production_model_details = PythonOperator(
        task_id='get_production_model_details', # Renamed for clarity
        python_callable=callable_get_production_model_details,
        doc_md="Queries MLflow for the Production model's URI, base name, version, and its training dataset_run_id."
    )

    # --- XCom Pull Definitions (clearer variable names) ---
    # Dataset run_id associated with the production model (for preparing input features & scalers)
    xcom_prod_model_training_dataset_run_id = "{{ ti.xcom_pull(task_ids='get_production_model_details', key='production_model_training_dataset_run_id') }}"
    
    # For loading the model itself
    xcom_prod_model_uri_for_loading = "{{ ti.xcom_pull(task_ids='get_production_model_details', key='production_model_uri_for_loading') }}"
    
    # For getting metadata about the specific production model version (e.g., its original training run_id)
    xcom_prod_model_base_name = "{{ ti.xcom_pull(task_ids='get_production_model_details', key='production_model_base_name') }}"
    xcom_prod_model_version_number = "{{ ti.xcom_pull(task_ids='get_production_model_details', key='production_model_version_number') }}"


    task_prepare_prediction_input = BashOperator(
        task_id='prepare_daily_prediction_input',
        bash_command=(
            f"mkdir -p {PREDICTION_INPUT_TEMP_DIR} && " # Ensure output directory exists
            f"python {PREPARE_INPUT_SCRIPT} "
            f"--config {CONFIG_PATH} "
            f"--production_model_training_run_id \"{xcom_prod_model_training_dataset_run_id}\" "
            f"--output_dir {PREDICTION_INPUT_TEMP_DIR}"
        ),
        env=common_env_vars,
        do_xcom_push=True, # Captures stdout (expected: "OUTPUT_PATH:<filepath>")
        doc_md="Prepares the scaled input sequence for daily prediction."
    )

    # NOTE on XCom for file path:
    # The following relies on prepare_prediction_input.py printing *only* "OUTPUT_PATH:<filepath>"
    # or that being the last relevant line. For robust production, consider making
    # task_prepare_prediction_input a PythonOperator that explicitly pushes the path.
    xcom_input_sequence_file_path = "{{ ti.xcom_pull(task_ids='prepare_daily_prediction_input').split('OUTPUT_PATH:')[1].strip() }}"

    task_make_daily_prediction = BashOperator(
        task_id='make_daily_prediction',
        bash_command=(
            f"python {PREDICT_MODEL_SCRIPT} "
            f"--config {CONFIG_PATH} "
            f"--input_sequence_path \"{xcom_input_sequence_file_path}\" "
            f"--production_model_uri_for_loading \"{xcom_prod_model_uri_for_loading}\" "
            f"--production_model_base_name \"{xcom_prod_model_base_name}\" "
            f"--production_model_version_number \"{xcom_prod_model_version_number}\""
        ),
        env=common_env_vars, # Includes MLFLOW_TRACKING_URI
        doc_md="Makes daily predictions using the production model and prepared input."
    )

    task_monitor_model_performance = BashOperator(
        task_id='monitor_model_performance', # Renamed for consistency
        bash_command=f"python {MONITOR_PERFORMANCE_SCRIPT} --config {CONFIG_PATH}",
        env=common_env_vars,
        do_xcom_push=True, # Captures stdout (expected: "NEXT_TASK_ID:<decision_task_id>")
        doc_md="Evaluates previous day's predictions and checks performance thresholds."
    )

    # --- Define Task Dependencies ---
    task_fetch_incremental_data >> task_get_production_model_details
    task_get_production_model_details >> task_prepare_prediction_input
    task_prepare_prediction_input >> task_make_daily_prediction
    task_make_daily_prediction >> task_monitor_model_performance