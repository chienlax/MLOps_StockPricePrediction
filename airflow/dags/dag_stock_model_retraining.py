# airflow/dags/stock_model_retraining_prod_dag.py
from __future__ import annotations

import pendulum
from pathlib import Path
import os
import subprocess
import logging
import yaml
import json 

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
import mlflow
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

# --- Define Paths & Load Params ---
PROJECT_ROOT = Path('/opt/airflow')
CONFIG_PATH = PROJECT_ROOT / 'config/params.yaml'

try:
    with open(CONFIG_PATH, 'r') as f:
        params_glob_retrain = yaml.safe_load(f) # Use different name for params
except Exception as e:
    log.error(f"CRITICAL: Could not load params.yaml at {CONFIG_PATH} for Retraining DAG. Error: {e}")
    params_glob_retrain = {
        'airflow_dags': {'retraining_pipeline_dag_id': 'stock_model_retraining_fallback_id'},
        'mlflow': {'tracking_uri': 'http://mlflow-server:5000', 'experiment_name': 'Fallback_Experiment'},
        'model_promotion': {} # Default empty dict
    }

MAKE_DATASET_SCRIPT = PROJECT_ROOT / 'src/data/make_dataset.py'
BUILD_FEATURES_SCRIPT = PROJECT_ROOT / 'src/features/build_features.py'
OPTIMIZE_SCRIPT = PROJECT_ROOT / 'src/models/optimize_hyperparams.py'
TRAIN_SCRIPT = PROJECT_ROOT / 'src/models/train_model.py'

# --- Python Callables ---
def callable_initialize_database_for_retrain(**kwargs):
    import yaml
    try:
        from src.utils.db_utils import setup_database
    except ImportError:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.utils.db_utils import setup_database
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    setup_database(config['database'])
    return "DB Initialized for Retraining DAG"

def callable_run_make_dataset_full_retrain(**kwargs): # Renamed
    ti = kwargs['ti']
    script_path = str(MAKE_DATASET_SCRIPT)
    config_file_path = str(CONFIG_PATH)
    command = ["python", script_path, "--config", config_file_path, "--mode", "full_process"]
    log.info(f"Retraining DAG: Executing make_dataset: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    log.info(f"Retraining DAG: make_dataset.py STDOUT:\n{stdout}")
    if process.returncode != 0:
        log.error(f"Retraining DAG: make_dataset.py STDERR:\n{stderr}")
        raise Exception(f"Retraining DAG: make_dataset.py failed. Error: {stderr}")
    run_id_line = [line for line in stdout.splitlines() if line.startswith("RUN_ID:")]
    if not run_id_line:
        raise Exception("Retraining DAG: make_dataset.py did not output RUN_ID.")
    new_dataset_run_id = run_id_line[-1].split("RUN_ID:")[1].strip()
    ti.xcom_push(key='new_dataset_run_id', value=new_dataset_run_id) # Key for XCom
    log.info(f"Retraining DAG: New dataset_run_id: {new_dataset_run_id}")
    return new_dataset_run_id

def callable_train_candidate_and_get_ids(**kwargs): # Renamed
    ti = kwargs['ti']
    new_dataset_run_id = ti.xcom_pull(task_ids='process_all_data_for_retraining_task', key='new_dataset_run_id') # Use correct task_id
    if not new_dataset_run_id:
        raise ValueError("Could not pull new_dataset_run_id from XComs for training.")

    with open(CONFIG_PATH, 'r') as f:
        params = yaml.safe_load(f)
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', params['mlflow'].get('tracking_uri'))
    
    command = ["python", str(TRAIN_SCRIPT), "--config", str(CONFIG_PATH), "--run_id", new_dataset_run_id]
    env = os.environ.copy()
    env['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    env['PYTHONPATH'] = f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{env.get('PYTHONPATH', '')}"

    log.info(f"Retraining DAG: Executing training: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    stdout, stderr = process.communicate()
    log.info(f"Retraining DAG: train_model.py STDOUT:\n{stdout}")
    if process.returncode != 0:
        log.error(f"Retraining DAG: train_model.py STDERR:\n{stderr}")
        raise Exception(f"Retraining DAG: train_model.py failed. Error: {stderr}")

    mlflow_run_id_line = [line for line in stdout.splitlines() if line.startswith("TRAINING_SUCCESS_MLFLOW_RUN_ID:")]
    if not mlflow_run_id_line:
        raise Exception("Retraining DAG: train_model.py did not output TRAINING_SUCCESS_MLFLOW_RUN_ID.")
    
    candidate_model_mlflow_run_id = mlflow_run_id_line[-1].split("TRAINING_SUCCESS_MLFLOW_RUN_ID:")[1].strip()
    log.info(f"Retraining DAG: Candidate model trained. MLflow Run ID: {candidate_model_mlflow_run_id}")
    
    ti.xcom_push(key='candidate_model_mlflow_run_id', value=candidate_model_mlflow_run_id)
    ti.xcom_push(key='candidate_training_dataset_run_id', value=new_dataset_run_id) # For evaluation context
    return candidate_model_mlflow_run_id

def callable_evaluate_and_branch_promotion(**kwargs): # Renamed
    ti = kwargs['ti']
    candidate_mlflow_run_id = ti.xcom_pull(task_ids='train_candidate_model_task', key='candidate_model_mlflow_run_id')
    candidate_dataset_run_id = ti.xcom_pull(task_ids='train_candidate_model_task', key='candidate_training_dataset_run_id')

    if not candidate_mlflow_run_id or not candidate_dataset_run_id:
        log.error("Missing candidate_mlflow_run_id or candidate_dataset_run_id for evaluation.")
        return params_glob_retrain['airflow_dags'].get('no_promotion_task_id', 'do_not_promote_task') # Default branch

    with open(CONFIG_PATH, 'r') as f:
        params = yaml.safe_load(f)
    
    mlflow_cfg = params['mlflow']
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
    registered_model_name = mlflow_cfg.get('experiment_name')
    
    promotion_cfg = params.get('model_promotion', {})
    eval_metric_name = promotion_cfg.get('metric', 'final_avg_mape_test') # Metric from train_model.py's test eval
    higher_is_better = promotion_cfg.get('higher_is_better', False) # For MAPE, lower is better

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    candidate_run_details = client.get_run(candidate_mlflow_run_id)
    candidate_metric_value = candidate_run_details.data.metrics.get(eval_metric_name)
    
    if candidate_metric_value is None:
        log.warning(f"Metric '{eval_metric_name}' not found for candidate model {candidate_mlflow_run_id}. Cannot compare.")
        return params_glob_retrain['airflow_dags'].get('no_promotion_task_id', 'do_not_promote_task')
    log.info(f"Candidate model ({candidate_mlflow_run_id}) evaluation metric ({eval_metric_name}): {candidate_metric_value}")

    prod_model_metric_value = None
    current_prod_alias = "Production"
    try:
        prod_version_obj = client.get_model_version_by_alias(name=registered_model_name, alias=current_prod_alias)
        prod_mlflow_run_id = prod_version_obj.run_id
        prod_run_details = client.get_run(prod_mlflow_run_id)
        prod_model_metric_value = prod_run_details.data.metrics.get(eval_metric_name)
        log.info(f"Current Production model ({prod_mlflow_run_id}) metric ({eval_metric_name}): {prod_model_metric_value}")
    except mlflow.exceptions.MlflowException as e: # Handles case where alias or model doesn't exist
        log.info(f"No current Production model or its metric found to compare against ({e}). Candidate will be promoted if valid.")
        prod_model_metric_value = None # Ensure it's None

    is_candidate_better = False
    if prod_model_metric_value is not None:
        if higher_is_better: is_candidate_better = candidate_metric_value > prod_model_metric_value
        else: is_candidate_better = candidate_metric_value < prod_model_metric_value
    elif candidate_metric_value is not None : # No production model to compare, promote if candidate is valid
        is_candidate_better = True

    if is_candidate_better:
        log.info("Candidate model is better or no production model to compare. Proceeding with promotion logic.")
        ti.xcom_push(key='candidate_model_version_to_promote_run_id', value=candidate_mlflow_run_id) # Pass run_id for promotion task
        return params_glob_retrain['airflow_dags'].get('promotion_task_id', 'promote_candidate_to_production_task')
    else:
        log.info("Candidate model is not better. Not promoting.")
        return params_glob_retrain['airflow_dags'].get('no_promotion_task_id', 'do_not_promote_task')

def callable_promote_model_to_production(**kwargs):
    ti = kwargs['ti']
    candidate_mlflow_run_id = ti.xcom_pull(task_ids='evaluate_and_branch_on_promotion_decision', key='candidate_model_version_to_promote_run_id')
    if not candidate_mlflow_run_id:
        log.error("No candidate_model_mlflow_run_id received for promotion. Skipping.")
        return

    with open(CONFIG_PATH, 'r') as f:
        params = yaml.safe_load(f)
    mlflow_cfg = params['mlflow']
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
    registered_model_name = mlflow_cfg.get('experiment_name')
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    production_alias_name = "Production"

    versions = client.search_model_versions(f"run_id='{candidate_mlflow_run_id}'")
    if not versions:
        log.error(f"No model version found in registry for run_id {candidate_mlflow_run_id}. Cannot promote.")
        return
    
    candidate_version_to_promote = sorted(versions, key=lambda v: int(v.version), reverse=True)[0].version
    log.info(f"Promoting model '{registered_model_name}' version {candidate_version_to_promote} (from run {candidate_mlflow_run_id}) to alias '{production_alias_name}'.")
    
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=production_alias_name,
        version=candidate_version_to_promote
    )
    log.info(f"Successfully set alias '{production_alias_name}' to version {candidate_version_to_promote} of model '{registered_model_name}'.")


# --- Default Arguments ---
default_args_retrain = {
    'owner': 'airflow_mlops_prod_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': [params_glob_retrain.get('alert_email', 'alert@example.com')],
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=2),
}

# --- DAG Definition ---
with DAG(
    dag_id=params_glob_retrain['airflow_dags']['retraining_pipeline_dag_id'],
    default_args=default_args_retrain,
    description='Stock model retraining pipeline: data processing, optimization, training, evaluation, and promotion.',
    schedule=None, 
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=['mlops_prod', 'retraining_pipeline'],
) as dag_retrain:

    common_env_vars_retrain = {
        'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}",
        'MLFLOW_TRACKING_URI': os.environ.get('MLFLOW_TRACKING_URI', params_glob_retrain['mlflow'].get('tracking_uri'))
    }

    task_init_db = PythonOperator(
        task_id='initialize_database_for_retraining',
        python_callable=callable_initialize_database_for_retrain,
    )

    task_process_data = PythonOperator(
        task_id='process_all_data_for_retraining_task', # Consistent ID
        python_callable=callable_run_make_dataset_full_retrain,
    )

    xcom_new_dataset_run_id = "{{ ti.xcom_pull(task_ids='process_all_data_for_retraining_task', key='new_dataset_run_id') }}"

    task_build_features = BashOperator(
        task_id='build_features_for_retraining',
        bash_command=f"python {BUILD_FEATURES_SCRIPT} --config {CONFIG_PATH} --run_id \"{xcom_new_dataset_run_id}\"",
        env=common_env_vars_retrain,
    )

    task_optimize_params = BashOperator(
        task_id='optimize_hyperparams_for_retraining',
        bash_command=f"python {OPTIMIZE_SCRIPT} --config {CONFIG_PATH} --run_id \"{xcom_new_dataset_run_id}\"",
        env=common_env_vars_retrain,
    )

    task_train_candidate = PythonOperator( # Changed to PythonOperator
        task_id='train_candidate_model_task', # Consistent ID
        python_callable=callable_train_candidate_and_get_ids,
    )

    branch_on_evaluation = BranchPythonOperator(
        task_id='evaluate_and_branch_on_promotion_decision', # Consistent ID
        python_callable=callable_evaluate_and_branch_promotion,
    )

    # Define task IDs for branching outcomes based on params.yaml or defaults
    promotion_task_id_from_cfg = params_glob_retrain['airflow_dags'].get('promotion_task_id', 'promote_candidate_to_production_task')
    no_promotion_task_id_from_cfg = params_glob_retrain['airflow_dags'].get('no_promotion_task_id', 'do_not_promote_task')


    task_promote_model = PythonOperator( # Changed to PythonOperator
        task_id=promotion_task_id_from_cfg,
        python_callable=callable_promote_model_to_production,
    )

    task_do_not_promote_model = DummyOperator(
        task_id=no_promotion_task_id_from_cfg,
    )

    # Dependencies
    task_init_db >> task_process_data >> task_build_features >> task_optimize_params
    task_optimize_params >> task_train_candidate >> branch_on_evaluation
    branch_on_evaluation >> [task_promote_model, task_do_not_promote_model]