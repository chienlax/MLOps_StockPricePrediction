# airflow/dags/daily_stock_operations_dag.py
from __future__ import annotations

import pendulum
from pathlib import Path
import os
import subprocess
import logging
import yaml # For loading params directly in Python callables if needed

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.dummy import DummyOperator # For branching
import mlflow
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

# --- Define Paths ---
PROJECT_ROOT = Path('/opt/airflow')
CONFIG_PATH = PROJECT_ROOT / 'config/params.yaml'
MAKE_DATASET_SCRIPT = PROJECT_ROOT / 'src/data/make_dataset.py'
PREPARE_INPUT_SCRIPT = PROJECT_ROOT / 'src/features/prepare_prediction_input.py'
PREDICT_MODEL_SCRIPT = PROJECT_ROOT / 'src/models/predict_model.py'
MONITOR_PERFORMANCE_SCRIPT = PROJECT_ROOT / 'src/models/monitor_performance.py'

# --- Python Callables ---
def callable_initialize_database(**kwargs):
    import yaml
    try:
        from src.utils.db_utils import setup_database
    except ImportError:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.db_utils import setup_database
    
    log.info(f"Daily Ops: Loading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    db_config = config['database']
    setup_database(db_config)
    log.info("Daily Ops: Database initialization complete.")
    return "DB Initialized"

def callable_get_production_model_info_for_daily(**kwargs):
    ti = kwargs['ti']
    with open(CONFIG_PATH, 'r') as f:
        params = yaml.safe_load(f)
    
    mlflow_cfg = params['mlflow']
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
    registered_model_name = mlflow_cfg.get('experiment_name') # Assuming this is also the registered model name

    if not mlflow_tracking_uri or not registered_model_name:
        raise ValueError("MLflow tracking URI or registered_model_name not configured in params.yaml.")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    
    production_model_uri = None
    prod_model_dataset_run_id = None

    latest_prod_versions = client.get_latest_versions(name=registered_model_name, stages=["Production"])
    if not latest_prod_versions:
        log.error(f"CRITICAL: No model '{registered_model_name}' found in 'Production' stage. Daily prediction cannot proceed.")
        # In a real system, this might trigger an alert or a specific fallback.
        # For now, we'll let it fail the task.
        raise ValueError(f"No Production model for '{registered_model_name}'.")
        
    production_version_obj = sorted(latest_prod_versions, key=lambda v: int(v.version), reverse=True)[0]
    # Construct URI that points to the specific version in production for clarity
    production_model_uri = f"models:/{registered_model_name}/{production_version_obj.version}"
    prod_model_mlflow_run_id = production_version_obj.run_id
    
    mlflow_run_details = client.get_run(prod_model_mlflow_run_id)
    prod_model_dataset_run_id = mlflow_run_details.data.params.get("dataset_run_id")
    
    if not prod_model_dataset_run_id:
        raise ValueError(f"'dataset_run_id' param missing from Production model's (MLflow Run {prod_model_mlflow_run_id}) training parameters.")
    
    log.info(f"Daily Ops: Using Production Model URI: {production_model_uri}")
    log.info(f"Daily Ops: Production model's training dataset_run_id: {prod_model_dataset_run_id}")

    ti.xcom_push(key='production_model_uri', value=production_model_uri)
    ti.xcom_push(key='production_model_training_dataset_run_id', value=prod_model_dataset_run_id)

# Python callable for branching based on monitoring output
def callable_branch_on_monitoring_result(**kwargs):
    ti = kwargs['ti']
    # monitor_performance.py prints "NEXT_TASK_ID:<task_id_for_branching>"
    # This is captured by BashOperator's default XCom push (key='return_value')
    monitoring_output = ti.xcom_pull(task_ids='monitor_model_performance_task', key='return_value')
    log.info(f"Monitoring script output for branching: {monitoring_output}")

    if monitoring_output and "NEXT_TASK_ID:" in monitoring_output:
        decision_task_id = monitoring_output.split("NEXT_TASK_ID:")[1].strip()
        log.info(f"Branching decision: proceed to task_id '{decision_task_id}'")
        return decision_task_id # This must match the task_id of one of the downstream tasks
    else:
        log.warning("Could not determine next task from monitoring output. Defaulting to no retraining.")
        # Fallback to the task_id for "no retraining"
        with open(CONFIG_PATH, 'r') as f:
            params = yaml.safe_load(f)
        return params.get('airflow_dags', {}).get('no_retraining_task_id', 'no_retraining_needed_task')


# --- Default Arguments ---
default_args = {
    'owner': 'airflow_mlops_team',
    'depends_on_past': False,
    'email_on_failure': True, # Enable email on failure for production
    'email': ['your_email@example.com'], # Add your email
    'email_on_retry': False,
    'retries': 2, # Increased retries for daily ops
    'retry_delay': pendulum.duration(minutes=5),
}

# --- DAG Definition ---
with DAG(
    # dag_id from params.yaml or hardcoded
    dag_id=yaml.safe_load(open(CONFIG_PATH))['airflow_dags']['daily_operations_dag_id'],
    default_args=default_args,
    description='Daily stock data ingestion, prediction, and performance monitoring.',
    schedule='0 1 * * 1-5',  # Example: 1 AM UTC on weekdays (adjust to your market)
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False, # Recommended to be False for most production DAGs
    tags=['mlops', 'daily_operations', 'stock_prediction'],
) as dag:

    task_init_db_daily = PythonOperator(
        task_id='initialize_database_daily_check',
        python_callable=callable_initialize_database,
    )

    task_fetch_incremental_data_daily = BashOperator(
        task_id='fetch_incremental_raw_data_daily',
        bash_command=f"python {MAKE_DATASET_SCRIPT} --config {CONFIG_PATH} --mode incremental_fetch",
        env={'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"},
    )

    task_get_prod_model_info_daily = PythonOperator(
        task_id='get_production_model_info_daily',
        python_callable=callable_get_production_model_info_for_daily,
    )

    prod_model_uri_xcom = "{{ ti.xcom_pull(task_ids='get_production_model_info_daily', key='production_model_uri') }}"
    prod_model_dataset_run_id_xcom = "{{ ti.xcom_pull(task_ids='get_production_model_info_daily', key='production_model_training_dataset_run_id') }}"
    prediction_input_output_dir = "/tmp/daily_airflow_pred_inputs" # Ensure Airflow worker can write here

    task_prepare_input_daily = BashOperator(
        task_id='prepare_daily_prediction_input',
        bash_command=(
            f"mkdir -p {prediction_input_output_dir} && " # Ensure dir exists
            f"python {PREPARE_INPUT_SCRIPT} "
            f"--config {CONFIG_PATH} "
            f"--production_model_training_run_id \"{prod_model_dataset_run_id_xcom}\" "
            f"--output_dir {prediction_input_output_dir}"
        ),
        env={'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"},
        do_xcom_push=True, # To capture "OUTPUT_PATH:<filepath>"
    )
    
    # Parse the output path for the next task
    input_sequence_file_path_xcom = "{{ ti.xcom_pull(task_ids='prepare_daily_prediction_input').split('OUTPUT_PATH:')[1].strip() }}"


    task_make_prediction_daily = BashOperator(
        task_id='make_daily_prediction',
        bash_command=(
            f"python {PREDICT_MODEL_SCRIPT} "
            f"--config {CONFIG_PATH} "
            f"--input_sequence_path \"{input_sequence_file_path_xcom}\" "
            f"--production_model_uri \"{prod_model_uri_xcom}\""
        ),
        env={
            'MLFLOW_TRACKING_URI': os.environ.get('MLFLOW_TRACKING_URI', yaml.safe_load(open(CONFIG_PATH))['mlflow'].get('tracking_uri')),
            'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"
        },
    )

    # This task needs to run *after* the predictions for evaluation_date are made AND actuals are available.
    # The schedule of this DAG and evaluation_lag_days handle this timing.
    task_monitor_performance_daily = BashOperator(
        task_id='monitor_model_performance_task', # This task_id is used in callable_branch_on_monitoring_result
        bash_command=f"python {MONITOR_PERFORMANCE_SCRIPT} --config {CONFIG_PATH}",
        env={'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"},
        do_xcom_push=True, # To capture "NEXT_TASK_ID:<decision>"
    )

    # Branching logic
    branch_op = BranchPythonOperator(
        task_id='branch_on_monitoring_decision',
        python_callable=callable_branch_on_monitoring_result,
    )

    # Trigger Retraining DAG task
    # The task_id here should match one of the outputs from callable_branch_on_monitoring_result
    # and what monitor_performance.py prints.
    # E.g., if monitor_performance.py prints NEXT_TASK_ID:trigger_retraining_pipeline_task
    # params.yaml:
    #   airflow_dags:
    #     retraining_trigger_task_id: "trigger_retraining_pipeline_task"
    #     retraining_pipeline_dag_id: "stock_model_retraining_pipeline"
    
    with open(CONFIG_PATH, 'r') as f:
        params_cfg = yaml.safe_load(f)
    
    trigger_retraining_task_id_from_cfg = params_cfg['airflow_dags'].get('retraining_trigger_task_id', 'trigger_retraining_pipeline_task')
    retraining_dag_id_from_cfg = params_cfg['airflow_dags']['retraining_pipeline_dag_id']
    no_retraining_task_id_from_cfg = params_cfg['airflow_dags'].get('no_retraining_task_id', 'no_retraining_needed_task')


    task_trigger_retraining = TriggerDagRunOperator(
        task_id=trigger_retraining_task_id_from_cfg, # Must match one of the outputs of branch_op
        trigger_dag_id=retraining_dag_id_from_cfg,
        # conf={'external_trigger': True, 'triggered_by_dag_id': dag.dag_id}, # Optional: pass config to triggered DAG
        wait_for_completion=False, # Run retraining DAG asynchronously
    )

    task_no_retraining = DummyOperator(
        task_id=no_retraining_task_id_from_cfg, # Must match one of the outputs of branch_op
    )

    # Define Task Dependencies for Daily DAG
    task_init_db_daily >> task_fetch_incremental_data_daily >> task_get_prod_model_info_daily
    task_get_prod_model_info_daily >> task_prepare_input_daily
    task_prepare_input_daily >> task_make_prediction_daily
    task_make_prediction_daily >> task_monitor_performance_daily
    task_monitor_performance_daily >> branch_op
    branch_op >> [task_trigger_retraining, task_no_retraining]