# airflow/dags/daily_stock_operations_prod_dag.py
from __future__ import annotations

import pendulum
from pathlib import Path
import os
import logging
import yaml 

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy import DummyOperator
import mlflow # For MlflowClient
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

# --- Define Constants and Paths ---
PROJECT_ROOT = Path('/opt/airflow')
CONFIG_PATH = PROJECT_ROOT / 'config/params.yaml'

# Load params once for DAG definition (e.g., for DAG ID, task IDs)
try:
    with open(CONFIG_PATH, 'r') as f:
        params_glob = yaml.safe_load(f)
except Exception as e:
    log.error(f"CRITICAL: Could not load params.yaml at {CONFIG_PATH} for DAG definition. Error: {e}")
    # Fallback or raise error to prevent DAG from loading incorrectly
    params_glob = {
        'airflow_dags': {
            'daily_operations_dag_id': 'daily_stock_operations_fallback_id',
            'retraining_pipeline_dag_id': 'stock_model_retraining_fallback_id',
            'retraining_trigger_task_id': 'trigger_retraining_pipeline_task',
            'no_retraining_task_id': 'no_retraining_needed_task'
        },
        'mlflow': {'tracking_uri': 'http://mlflow-server:5000'} # Ensure a default
    }


# Script paths
MAKE_DATASET_SCRIPT = PROJECT_ROOT / 'src/data/make_dataset.py'
PREPARE_INPUT_SCRIPT = PROJECT_ROOT / 'src/features/prepare_prediction_input.py'
PREDICT_MODEL_SCRIPT = PROJECT_ROOT / 'src/models/predict_model.py'
MONITOR_PERFORMANCE_SCRIPT = PROJECT_ROOT / 'src/models/monitor_performance.py'

PREDICTION_INPUT_TEMP_DIR = "/tmp/prod_daily_pred_inputs" # Unique temp dir for this DAG

# --- Python Callables ---
def callable_initialize_database_daily(**kwargs):
    import yaml # Load fresh params inside callable
    try:
        from src.utils.db_utils import setup_database
    except ImportError:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.db_utils import setup_database
    
    log.info(f"Daily Ops DAG: Loading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    db_config = config['database']
    setup_database(db_config)
    log.info("Daily Ops DAG: Database initialization complete.")
    return "DB Initialized for Daily Ops"

# ------------------------------------------------------

def callable_get_production_model_details_daily(**kwargs):
    ti = kwargs['ti']
    log.info("Daily Ops DAG: Starting callable_get_production_model_details...")
    with open(CONFIG_PATH, 'r') as f:
        params = yaml.safe_load(f)
    
    mlflow_cfg = params.get('mlflow', {})
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
    base_registered_model_name = mlflow_cfg.get('experiment_name') 

    if not mlflow_tracking_uri or not base_registered_model_name:
        raise ValueError("MLflow tracking URI or base_registered_model_name not configured.")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    production_alias = "Production"

    try:
        log.info(f"Querying MLflow for model '{base_registered_model_name}' with alias '{production_alias}'...")
        model_version_obj = client.get_model_version_by_alias(name=base_registered_model_name, alias=production_alias)
        
        production_model_uri_for_loading = f"models:/{base_registered_model_name}@{production_alias}"
        production_version_number_str = model_version_obj.version
        prod_model_origin_mlflow_run_id = model_version_obj.run_id
        
        log.info(f"Daily Ops: Using Prod Model: URI='{production_model_uri_for_loading}', Ver='{production_version_number_str}', Training MLflow Run='{prod_model_origin_mlflow_run_id}'")

        mlflow_run_details = client.get_run(prod_model_origin_mlflow_run_id)
        prod_model_training_dataset_run_id = mlflow_run_details.data.params.get("dataset_run_id")
        
        if not prod_model_training_dataset_run_id:
            raise ValueError(f"'dataset_run_id' param missing from Prod Model's (MLflow Run {prod_model_origin_mlflow_run_id}) params.")
        log.info(f"Daily Ops: Prod model's training dataset_run_id: {prod_model_training_dataset_run_id}")

    # ... (Error handling) ...
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) and (f"alias '{production_alias}'" in str(e) or f"model '{base_registered_model_name}'" in str(e)):
            log.error(f"CRITICAL: Alias '{production_alias}' or model '{base_registered_model_name}' does not exist.")
            raise ValueError(f"Production alias/model not found. Please ensure '{base_registered_model_name}' has a version aliased as '{production_alias}'.")
        else:
            log.error(f"Error querying MLflow for production model: {e}", exc_info=True)
            raise
    except Exception as e:
        log.error(f"Unexpected error in callable_get_production_model_details: {e}", exc_info=True)
        raise

    ti.xcom_push(key='production_model_uri_for_loading', value=production_model_uri_for_loading)
    ti.xcom_push(key='production_model_base_name', value=base_registered_model_name)
    ti.xcom_push(key='production_model_version_number', value=production_version_number_str)
    ti.xcom_push(key='production_model_training_dataset_run_id', value=prod_model_training_dataset_run_id)
    return "Production model details fetched."

# ------------------------------------------------------

def callable_branch_on_monitoring_result_daily(**kwargs): # Renamed
    ti = kwargs['ti']
    monitoring_output = ti.xcom_pull(task_ids='monitor_model_performance_daily_task', key='return_value') # Use correct task_id
    log.info(f"Daily Ops DAG: Monitoring script output for branching: {monitoring_output}")

    # Load task IDs from params for branching
    with open(CONFIG_PATH, 'r') as f:
        params = yaml.safe_load(f)
    airflow_dag_cfg = params.get('airflow_dags', {})
    retraining_branch_task_id = airflow_dag_cfg.get('retraining_trigger_task_id', 'trigger_retraining_pipeline_task')
    no_retraining_branch_task_id = airflow_dag_cfg.get('no_retraining_task_id', 'no_retraining_needed_task')

    if monitoring_output and f"NEXT_TASK_ID:{retraining_branch_task_id}" in monitoring_output:
        log.info(f"Branching decision: proceed to task_id '{retraining_branch_task_id}'")
        return retraining_branch_task_id
    elif monitoring_output and f"NEXT_TASK_ID:{no_retraining_branch_task_id}" in monitoring_output:
        log.info(f"Branching decision: proceed to task_id '{no_retraining_branch_task_id}'")
        return no_retraining_branch_task_id
    else:
        log.warning(f"Could not determine next task from monitoring output: '{monitoring_output}'. Defaulting to '{no_retraining_branch_task_id}'.")
        return no_retraining_branch_task_id

# --- Default Arguments ---
default_args_daily = {
    'owner': 'airflow_mlops_prod_team',
    'depends_on_past': False,
    'email_on_failure': True, 
    'email': [params_glob.get('alert_email', 'alert@example.com')], # Get from params or default
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': pendulum.duration(minutes=5),
}

# --- DAG Definition ---
with DAG(
    dag_id=params_glob['airflow_dags']['daily_operations_dag_id'],
    default_args=default_args_daily,
    description='Daily stock data ingestion, prediction, and performance monitoring with retraining trigger.',
    schedule=params_glob.get('schedules',{}).get('daily_dag', '0 2 * * 1-5'), # 2 AM UTC Weekdays (configurable)
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=['mlops_prod', 'daily_stock_pipeline'],
) as dag_daily:

    common_env_vars_daily = {
        'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}",
        'MLFLOW_TRACKING_URI': os.environ.get('MLFLOW_TRACKING_URI', params_glob['mlflow'].get('tracking_uri'))
    }

    task_init_db = PythonOperator(
        task_id='initialize_database_daily', # Renamed
        python_callable=callable_initialize_database_daily,
    )

    task_fetch_data = BashOperator(
        task_id='fetch_incremental_raw_data_daily',
        bash_command=f"python {MAKE_DATASET_SCRIPT} --config {CONFIG_PATH} --mode incremental_fetch",
        env=common_env_vars_daily,
    )

    task_get_prod_model = PythonOperator(
        task_id='get_production_model_details_daily', # Renamed
        python_callable=callable_get_production_model_details_daily,
    )

    xcom_prod_training_dataset_run_id = "{{ ti.xcom_pull(task_ids='get_production_model_details_daily', key='production_model_training_dataset_run_id') }}"
    xcom_prod_model_uri_for_loading = "{{ ti.xcom_pull(task_ids='get_production_model_details_daily', key='production_model_uri_for_loading') }}"
    xcom_prod_model_base_name = "{{ ti.xcom_pull(task_ids='get_production_model_details_daily', key='production_model_base_name') }}"
    xcom_prod_model_version_number = "{{ ti.xcom_pull(task_ids='get_production_model_details_daily', key='production_model_version_number') }}"

    task_prepare_input = BashOperator(
        task_id='prepare_daily_prediction_input',
        bash_command=(
            f"mkdir -p {PREDICTION_INPUT_TEMP_DIR} && "
            f"python {PREPARE_INPUT_SCRIPT} "
            f"--config {CONFIG_PATH} "
            f"--production_model_training_run_id \"{xcom_prod_training_dataset_run_id}\" "
            f"--output_dir {PREDICTION_INPUT_TEMP_DIR}"
        ),
        env=common_env_vars_daily,
        do_xcom_push=True, 
    )
    
    xcom_input_seq_file_path = "{{ ti.xcom_pull(task_ids='prepare_daily_prediction_input').split('OUTPUT_PATH:')[1].strip() }}"

    task_make_prediction = BashOperator(
        task_id='make_daily_prediction',
        bash_command=(
            f"python {PREDICT_MODEL_SCRIPT} "
            f"--config {CONFIG_PATH} "
            f"--input_sequence_path \"{xcom_input_seq_file_path}\" "
            f"--production_model_uri_for_loading \"{xcom_prod_model_uri_for_loading}\" "
            f"--production_model_base_name \"{xcom_prod_model_base_name}\" "
            f"--production_model_version_number \"{xcom_prod_model_version_number}\""
        ),
        env=common_env_vars_daily,
    )

    task_monitor_performance = BashOperator(
        task_id='monitor_model_performance_daily_task', # Consistent ID for branching callable
        bash_command=f"python {MONITOR_PERFORMANCE_SCRIPT} --config {CONFIG_PATH}",
        env=common_env_vars_daily,
        do_xcom_push=True, 
    )

    branch_on_monitor = BranchPythonOperator(
        task_id='branch_based_on_performance', # Renamed
        python_callable=callable_branch_on_monitoring_result_daily,
    )

    retraining_trigger_id = params_glob['airflow_dags']['retraining_trigger_task_id']
    retraining_dag_to_trigger = params_glob['airflow_dags']['retraining_pipeline_dag_id']
    no_retraining_id = params_glob['airflow_dags']['no_retraining_task_id']

    task_trigger_retraining_dag = TriggerDagRunOperator(
        task_id=retraining_trigger_id, 
        trigger_dag_id=retraining_dag_to_trigger,
        wait_for_completion=False, 
    )

    task_no_retraining_needed = DummyOperator(
        task_id=no_retraining_id,
    )

    # Define Dependencies
    task_init_db >> task_fetch_data >> task_get_prod_model
    task_get_prod_model >> task_prepare_input >> task_make_prediction
    task_make_prediction >> task_monitor_performance
    task_monitor_performance >> branch_on_monitor
    branch_on_monitor >> [task_trigger_retraining_dag, task_no_retraining_needed]