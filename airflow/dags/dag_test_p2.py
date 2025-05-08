# airflow/dags/temp_phase2_daily_ops_test_dag.py
from __future__ import annotations

import pendulum
from pathlib import Path
import os
import subprocess
import logging
import json
import mlflow 

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

# --- Define Paths (same as your other DAG) ---
PROJECT_ROOT = Path('/opt/airflow')
CONFIG_PATH = PROJECT_ROOT / 'config/params.yaml'
# Phase 1 script
MAKE_DATASET_SCRIPT = PROJECT_ROOT / 'src/data/make_dataset.py'
# Phase 2 scripts
PREPARE_INPUT_SCRIPT = PROJECT_ROOT / 'src/features/prepare_prediction_input.py'
PREDICT_MODEL_SCRIPT = PROJECT_ROOT / 'src/models/predict_model.py'
MONITOR_PERFORMANCE_SCRIPT = PROJECT_ROOT / 'src/models/monitor_performance.py'

# --- Python Callables for PythonOperator Tasks ---
def callable_get_production_model_info(**kwargs):
    """
    Queries MLflow Model Registry for the "Production" model and its associated
    training dataset_run_id. Pushes these to XComs.
    """
    import yaml # For loading params.yaml inside the callable

    ti = kwargs['ti']
    conf_path = str(CONFIG_PATH)
    with open(conf_path, 'r') as f:
        params = yaml.safe_load(f)
    
    mlflow_cfg = params['mlflow']
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
    # The experiment_name is used as the registered_model_name in train_model.py
    # If you used a different registered_model_name, update this.
    registered_model_name = mlflow_cfg.get('experiment_name', "Stock_Price_Prediction_LSTM_Refactored") 

    if not mlflow_tracking_uri:
        log.error("MLFLOW_TRACKING_URI not set in callable_get_production_model_info.")
        raise ValueError("MLFLOW_TRACKING_URI not set.")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    
    production_model_uri = None
    prod_model_dataset_run_id = None
    prod_model_mlflow_run_id = None # The MLflow run ID of the training job for the prod model

    try:
        latest_prod_versions = client.get_latest_versions(name=registered_model_name, stages=["Production"])
        if not latest_prod_versions:
            log.warning(f"No model named '{registered_model_name}' found in 'Production' stage.")
            # For testing, we might want to allow using a 'Staging' model or a specific run if no Prod exists
            # For now, let's be strict. In a real scenario, you might have a fallback.
            raise ValueError(f"No Production model found for '{registered_model_name}'. Please ensure a model is promoted.")
        
        # Assuming one version in Production, or taking the latest if multiple (by version number)
        production_version_obj = sorted(latest_prod_versions, key=lambda v: int(v.version), reverse=True)[0]
        production_model_uri = f"models:/{registered_model_name}/{production_version_obj.stage}" # or /version
        prod_model_mlflow_run_id = production_version_obj.run_id
        
        log.info(f"Found Production Model: URI='{production_model_uri}', Version='{production_version_obj.version}', MLflow Run ID='{prod_model_mlflow_run_id}'")

        # Fetch the MLflow run details to get the 'dataset_run_id' parameter
        mlflow_run_details = client.get_run(prod_model_mlflow_run_id)
        prod_model_dataset_run_id = mlflow_run_details.data.params.get("dataset_run_id")
        
        if not prod_model_dataset_run_id:
            log.error(f"Parameter 'dataset_run_id' not found in MLflow Run {prod_model_mlflow_run_id} for the production model.")
            raise ValueError(f"'dataset_run_id' param missing from production model's MLflow run.")
        
        log.info(f"Production model's associated training dataset_run_id: {prod_model_dataset_run_id}")

    except Exception as e:
        log.error(f"Error querying MLflow for production model: {e}", exc_info=True)
        raise # Fail the task if we can't get this critical info

    ti.xcom_push(key='production_model_uri', value=production_model_uri)
    ti.xcom_push(key='production_model_training_dataset_run_id', value=prod_model_dataset_run_id)
    
    return {"production_model_uri": production_model_uri, "dataset_run_id": prod_model_dataset_run_id}


# --- Default Arguments ---
default_args = {
    'owner': 'airflow_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': pendulum.duration(seconds=30),
}

# --- DAG Definition ---
with DAG(
    dag_id='temp_phase2_daily_operations_test',
    default_args=default_args,
    description='(Temporary) Tests Phase 2 daily operational scripts (prepare_input, predict, monitor).',
    schedule=None, # Manually triggerable for testing
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=['ml', 'daily_ops_test', 'phase2_test'],
) as dag:

    task_fetch_incremental_data = BashOperator(
        task_id='fetch_incremental_raw_data',
        bash_command=(
            f"python {MAKE_DATASET_SCRIPT} "
            f"--config {CONFIG_PATH} "
            f"--mode incremental_fetch"
        ),
        env={'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"},
        doc_md="Fetches the latest raw stock data incrementally."
    )

    task_get_prod_model_info = PythonOperator(
        task_id='get_production_model_info',
        python_callable=callable_get_production_model_info,
        doc_md="Queries MLflow for the Production model URI and its training dataset_run_id."
    )

    # XCom pull definitions
    prod_model_uri_xcom = "{{ ti.xcom_pull(task_ids='get_production_model_info', key='production_model_uri') }}"
    prod_model_dataset_run_id_xcom = "{{ ti.xcom_pull(task_ids='get_production_model_info', key='production_model_training_dataset_run_id') }}"
    
    # Path where prepare_prediction_input.py will save its .npy file
    # This should be a path accessible by subsequent tasks within the Airflow worker environment
    # Using /tmp/ is common for temporary inter-task files if not using larger XComs backend.
    prediction_input_output_dir = "/tmp/daily_prediction_inputs" # Ensure this dir is creatable/writable by airflow user

    task_prepare_prediction_input = BashOperator(
        task_id='prepare_daily_prediction_input',
        bash_command=(
            f"python {PREPARE_INPUT_SCRIPT} "
            f"--config {CONFIG_PATH} "
            f"--production_model_training_run_id \"{prod_model_dataset_run_id_xcom}\" " # Quote XCom value
            f"--output_dir {prediction_input_output_dir}"
        ),
        env={'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"},
        # Assuming prepare_prediction_input.py prints "OUTPUT_PATH:<filepath>"
        # The default BashOperator XCom push will capture the last line of stdout.
        # If it prints other things after, this might be fragile.
        # For robustness, change PREPARE_INPUT_SCRIPT to a PythonOperator to explicitly push path.
        # For now, let's rely on prepare_prediction_input.py printing only the path at the end.
        do_xcom_push=True, # To capture the output path
        doc_md="Prepares the scaled input sequence for daily prediction using production model's scalers."
    )

    # Extract the file path from the XCom pushed by task_prepare_prediction_input
    # Assuming the script prints "OUTPUT_PATH:/path/to/file.npy"
    # We need to parse this. A PythonOperator would be cleaner for this.
    # Let's try a simple Jinja manipulation for BashOperator, but it's tricky.
    # A safer bet is to have prepare_input.py just print the path, and use return_value.
    # If `do_xcom_push=True`, the full stdout (or last line) is pushed with key `return_value`.
    # Assuming prepare_prediction_input.py prints "OUTPUT_PATH:/path/to/file.npy"
    # and the BashOperator captures it. We'd then need to parse it in the next task's command.
    # This is getting complex for BashOperator.

    # Let's simplify: Assume prepare_prediction_input.py writes to a *fixed, known temporary filename*
    # that predict_model.py can also know. This avoids complex XCom parsing in Bash.
    # Or, make task_prepare_prediction_input a PythonOperator.
    # For now, let's stick to the "print OUTPUT_PATH:" and assume predict_model.py can take it.
    # The previous `BashOperator` for `task_prepare_prediction_input` has `do_xcom_push=True`.
    # So, `ti.xcom_pull(task_ids='prepare_daily_prediction_input')` will get its stdout.
    # We need to extract the path from "OUTPUT_PATH:/actual/path.npy".

    # Using a PythonOperator for predict_model would make XCom handling easier.
    # Let's try BashOperator first and see if we can make it work with Jinja.
    # `prepare_prediction_input.py` prints `OUTPUT_PATH:<filepath>`
    # The `return_value` XCom from `task_prepare_prediction_input` will be this string.
    input_sequence_file_path_xcom = "{{ ti.xcom_pull(task_ids='prepare_daily_prediction_input').split('OUTPUT_PATH:')[1] }}"


    task_make_daily_prediction = BashOperator(
        task_id='make_daily_prediction',
        bash_command=(
            f"python {PREDICT_MODEL_SCRIPT} "
            f"--config {CONFIG_PATH} "
            f"--input_sequence_path \"{input_sequence_file_path_xcom}\" " # Quote XCom value
            f"--production_model_uri \"{prod_model_uri_xcom}\""  # Quote XCom value
        ),
        env={
            'MLFLOW_TRACKING_URI': os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000'),
            'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"
        },
        doc_md="Makes daily predictions using the production model and prepared input."
    )

    task_monitor_model_performance = BashOperator(
        task_id='monitor_model_performance',
        bash_command=(
            f"python {MONITOR_PERFORMANCE_SCRIPT} "
            f"--config {CONFIG_PATH}"
        ),
        env={'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"},
        # This script prints "NEXT_TASK_ID:<task_id_for_branching>"
        # We are not branching in this test DAG, but we can check its output.
        do_xcom_push=True,
        doc_md="Evaluates previous day's predictions and checks performance thresholds."
    )

    # --- Define Task Dependencies ---
    task_fetch_incremental_data >> task_get_prod_model_info
    task_get_prod_model_info >> task_prepare_prediction_input
    task_prepare_prediction_input >> task_make_daily_prediction
    task_make_daily_prediction >> task_monitor_model_performance