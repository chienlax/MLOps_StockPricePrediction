# airflow/dags/stock_model_retraining_dag.py
from __future__ import annotations

import pendulum
from pathlib import Path
import os
import subprocess
import logging
import yaml
import json # For loading params in callables

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
import mlflow
from mlflow.tracking import MlflowClient
# Assuming your evaluate_model can be imported or its logic included in a callable
# For simplicity, a dedicated callable might be better here.

log = logging.getLogger(__name__)

# --- Define Paths ---
PROJECT_ROOT = Path('/opt/airflow')
CONFIG_PATH = PROJECT_ROOT / 'config/params.yaml'
MAKE_DATASET_SCRIPT = PROJECT_ROOT / 'src/data/make_dataset.py'
BUILD_FEATURES_SCRIPT = PROJECT_ROOT / 'src/features/build_features.py'
OPTIMIZE_SCRIPT = PROJECT_ROOT / 'src/models/optimize_hyperparams.py'
TRAIN_SCRIPT = PROJECT_ROOT / 'src/models/train_model.py'

# --- Python Callables ---
def callable_initialize_database_retrain(**kwargs): # Renamed to avoid conflict if in same file
    # Same as in daily_stock_operations_dag.py
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
    return "DB Initialized for Retraining"

def callable_run_make_dataset_full_process_retrain(**kwargs):
    # Same as in temp_phase1_retraining_pipeline
    script_path = str(MAKE_DATASET_SCRIPT)
    config_file_path = str(CONFIG_PATH)
    command = ["python", script_path, "--config", config_file_path, "--mode", "full_process"]
    log.info(f"Retrain: Executing command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    log.info(f"Retrain: make_dataset.py STDOUT:\n{stdout}")
    if process.returncode != 0:
        log.error(f"Retrain: make_dataset.py STDERR:\n{stderr}")
        raise Exception(f"Retrain: make_dataset.py failed. Error: {stderr}")
    run_id_line = [line for line in stdout.splitlines() if line.startswith("RUN_ID:")]
    if not run_id_line:
        raise Exception("Retrain: make_dataset.py did not output RUN_ID.")
    actual_run_id = run_id_line[-1].split("RUN_ID:")[1].strip()
    kwargs['ti'].xcom_push(key='new_dataset_run_id', value=actual_run_id)
    return actual_run_id

def callable_train_candidate_model_and_get_mlflow_id(**kwargs):
    ti = kwargs['ti']
    new_dataset_run_id = ti.xcom_pull(task_ids='process_all_data_for_retraining', key='new_dataset_run_id')
    if not new_dataset_run_id:
        raise ValueError("Could not pull new_dataset_run_id from XComs for training.")

    script_path = str(TRAIN_SCRIPT)
    config_file_path = str(CONFIG_PATH)
    
    # MLflow tracking URI from params or environment
    with open(CONFIG_PATH, 'r') as f:
        params = yaml.safe_load(f)
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', params['mlflow'].get('tracking_uri'))
    
    command = [
        "python", script_path,
        "--config", config_file_path,
        "--run_id", new_dataset_run_id # This is the dataset_run_id
    ]
    env = os.environ.copy()
    env['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri
    env['PYTHONPATH'] = f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{env.get('PYTHONPATH', '')}"

    log.info(f"Retrain: Executing training command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    stdout, stderr = process.communicate()
    log.info(f"Retrain: train_model.py STDOUT:\n{stdout}")

    if process.returncode != 0:
        log.error(f"Retrain: train_model.py STDERR:\n{stderr}")
        raise Exception(f"Retrain: train_model.py failed. Error: {stderr}")

    # train_model.py should print "TRAINING_SUCCESS_MLFLOW_RUN_ID:<actual_mlflow_run_id>"
    mlflow_run_id_line = [line for line in stdout.splitlines() if line.startswith("TRAINING_SUCCESS_MLFLOW_RUN_ID:")]
    if not mlflow_run_id_line:
        raise Exception("Retrain: train_model.py did not output TRAINING_SUCCESS_MLFLOW_RUN_ID.")
    
    candidate_model_mlflow_run_id = mlflow_run_id_line[-1].split("TRAINING_SUCCESS_MLFLOW_RUN_ID:")[1].strip()
    log.info(f"Retrain: Candidate model trained. MLflow Run ID: {candidate_model_mlflow_run_id}")
    
    ti.xcom_push(key='candidate_model_mlflow_run_id', value=candidate_model_mlflow_run_id)
    # Also push the dataset_run_id it was trained on, for the evaluation task
    ti.xcom_push(key='candidate_training_dataset_run_id', value=new_dataset_run_id)
    return candidate_model_mlflow_run_id


def callable_evaluate_and_promote_candidate(**kwargs):
    ti = kwargs['ti']
    candidate_mlflow_run_id = ti.xcom_pull(task_ids='train_candidate_model_task', key='candidate_model_mlflow_run_id')
    # This is the dataset_run_id the CANDIDATE was trained on, used to load ITS test set
    candidate_dataset_run_id = ti.xcom_pull(task_ids='train_candidate_model_task', key='candidate_training_dataset_run_id')

    if not candidate_mlflow_run_id or not candidate_dataset_run_id:
        log.error("Missing candidate_mlflow_run_id or candidate_dataset_run_id from XComs.")
        return "do_not_promote_candidate_task" # Default branch

    with open(CONFIG_PATH, 'r') as f:
        params = yaml.safe_load(f)
    
    db_config = params['database']
    mlflow_cfg = params['mlflow']
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
    registered_model_name = mlflow_cfg.get('experiment_name')
    # Define a promotion threshold, e.g., candidate's MAPE must be X% better or some absolute value.
    # For simplicity, let's say candidate is promoted if its test MAPE is lower than production's last known MAPE.
    # This is a simplified comparison; a proper A/B test or shadow deployment is more robust.
    promotion_decision_metric = params.get('model_promotion', {}).get('metric', 'avg_mape_test') # from train_model eval output
    promotion_higher_is_better = params.get('model_promotion', {}).get('higher_is_better', False) # for MAPE, lower is better

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    # --- 1. Evaluate Candidate Model ---
    candidate_model_uri = f"runs:/{candidate_mlflow_run_id}/model" # Path to model artifact
    log.info(f"Evaluating candidate model from URI: {candidate_model_uri}")
    
    # Load candidate model's test data (X_test_scaled, y_test_scaled, y_scalers for this candidate_dataset_run_id)
    # This requires importing or defining evaluation logic here.
    # For brevity, assume we can get the key metric (e.g., avg_mape_test) from the candidate's MLflow run.
    candidate_run_details = client.get_run(candidate_mlflow_run_id)
    candidate_metric_value = candidate_run_details.data.metrics.get(f"final_{promotion_decision_metric}") # final_avg_mape_test
    
    if candidate_metric_value is None:
        log.warning(f"Metric '{promotion_decision_metric}' not found for candidate model {candidate_mlflow_run_id}. Cannot compare.")
        return "do_not_promote_candidate_task"
    log.info(f"Candidate model ({candidate_mlflow_run_id}) metric ({promotion_decision_metric}): {candidate_metric_value}")

    # --- 2. Get Performance of Current Production Model (Simplified) ---
    # Ideally, you'd re-evaluate production model on the SAME new test set if possible,
    # or use its last reported test metric, or average recent monitored performance.
    # For this example, let's fetch the metric from the MLflow run of the current Production model.
    prod_model_metric_value = None
    latest_prod_versions = client.get_latest_versions(name=registered_model_name, stages=["Production"])
    if latest_prod_versions:
        prod_version_obj = sorted(latest_prod_versions, key=lambda v: int(v.version), reverse=True)[0]
        prod_mlflow_run_id = prod_version_obj.run_id
        prod_run_details = client.get_run(prod_mlflow_run_id)
        prod_model_metric_value = prod_run_details.data.metrics.get(f"final_{promotion_decision_metric}")
        log.info(f"Current Production model ({prod_mlflow_run_id}) metric ({promotion_decision_metric}): {prod_model_metric_value}")
    else:
        log.info("No current Production model to compare against. Candidate will be promoted by default if it has a metric.")
        # If no production model, promote if candidate is valid
        if candidate_metric_value is not None:
             is_candidate_better = True # First model
        else:
            return "do_not_promote_candidate_task"


    # --- 3. Compare and Decide ---
    if prod_model_metric_value is not None: # If there's a prod model to compare to
        if promotion_higher_is_better:
            is_candidate_better = candidate_metric_value > prod_model_metric_value
        else: # Lower is better (e.g., for MAPE, RMSE)
            is_candidate_better = candidate_metric_value < prod_model_metric_value
    # If prod_model_metric_value is None (no prod model was found with the metric), is_candidate_better is already True if candidate_metric_value exists.

    if is_candidate_better:
        log.info("Candidate model is better than current production (or no production model exists). Promoting.")
        # Find the model version associated with the candidate_mlflow_run_id
        # The train_model.py script registers the model. We need its version.
        # This is a bit tricky as log_model doesn't directly return the version if it's new.
        # We can search for the version linked to candidate_mlflow_run_id.
        versions = client.search_model_versions(f"run_id='{candidate_mlflow_run_id}'")
        if not versions:
            log.error(f"No model version found in registry for run_id {candidate_mlflow_run_id}. Cannot promote.")
            return "do_not_promote_candidate_task"
        
        candidate_version = versions[0].version # Assuming first one is the one we want
        log.info(f"Transitioning model '{registered_model_name}' version {candidate_version} (run_id {candidate_mlflow_run_id}) to Production.")
        
        # Archive existing Production versions first (optional but good practice)
        for old_prod_version in latest_prod_versions:
            if old_prod_version.version != candidate_version : # Don't archive itself if already there
                log.info(f"Archiving old Production version: {old_prod_version.version}")
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=old_prod_version.version,
                    stage="Archived"
                )
        
        client.transition_model_version_stage(
            name=registered_model_name,
            version=candidate_version,
            stage="Production",
            archive_existing_versions=False # We did it manually above
        )
        log.info(f"Model version {candidate_version} successfully promoted to Production.")
        return "promotion_complete_task"
    else:
        log.info("Candidate model is not better than current production. Not promoting.")
        return "do_not_promote_candidate_task"

# --- Default Arguments ---
default_args_retrain = { # Use a different name to avoid clashes if in same Python interpreter context
    'owner': 'airflow_mlops_team',
    'depends_on_past': False, # Retraining can run independently of past retraining success
    'email_on_failure': True,
    'email': ['your_email@example.com'],
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': pendulum.duration(minutes=2),
}

# --- DAG Definition ---
with DAG(
    dag_id=yaml.safe_load(open(CONFIG_PATH))['airflow_dags']['retraining_pipeline_dag_id'],
    default_args=default_args_retrain,
    description='Stock model retraining pipeline: processes data, optimizes, trains, evaluates, and promotes.',
    schedule=None, # Triggered by daily_ops_dag or manually
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    tags=['mlops', 'retraining', 'stock_prediction'],
) as dag_retrain: # Use a different variable name for the DAG object

    task_init_db_retrain = PythonOperator(
        task_id='initialize_database_for_retraining',
        python_callable=callable_initialize_database_retrain,
    )

    task_process_all_data_retrain = PythonOperator(
        task_id='process_all_data_for_retraining',
        python_callable=callable_run_make_dataset_full_process_retrain,
    )

    new_dataset_run_id_xcom = "{{ ti.xcom_pull(task_ids='process_all_data_for_retraining', key='new_dataset_run_id') }}"

    task_build_features_retrain = BashOperator(
        task_id='build_features_for_retraining',
        bash_command=f"python {BUILD_FEATURES_SCRIPT} --config {CONFIG_PATH} --run_id \"{new_dataset_run_id_xcom}\"",
        env={'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"},
    )

    task_optimize_hyperparams_retrain = BashOperator(
        task_id='optimize_hyperparams_for_retraining',
        bash_command=f"python {OPTIMIZE_SCRIPT} --config {CONFIG_PATH} --run_id \"{new_dataset_run_id_xcom}\"",
        env={'PYTHONPATH': f"{PROJECT_ROOT}:{PROJECT_ROOT}/src:{os.environ.get('PYTHONPATH', '')}"},
    )

    # This task now also needs to push the MLflow run ID of the trained candidate model
    task_train_candidate_model = PythonOperator(
        task_id='train_candidate_model_task',
        python_callable=callable_train_candidate_model_and_get_mlflow_id,
    )

    evaluate_and_promote_branch_op = BranchPythonOperator(
        task_id='evaluate_and_promote_candidate_branch',
        python_callable=callable_evaluate_and_promote_candidate,
    )

    # Task IDs for promotion branch outcomes
    # These must match the strings returned by callable_evaluate_and_promote_candidate
    promotion_done_task_id = "promotion_complete_task" 
    no_promotion_task_id = "do_not_promote_candidate_task"

    task_promotion_complete = DummyOperator(
        task_id=promotion_done_task_id,
    )
    task_do_not_promote = DummyOperator(
        task_id=no_promotion_task_id,
    )

    # Define Task Dependencies for Retraining DAG
    task_init_db_retrain >> task_process_all_data_retrain
    task_process_all_data_retrain >> task_build_features_retrain
    task_build_features_retrain >> task_optimize_hyperparams_retrain
    task_optimize_hyperparams_retrain >> task_train_candidate_model
    task_train_candidate_model >> evaluate_and_promote_branch_op
    evaluate_and_promote_branch_op >> [task_promotion_complete, task_do_not_promote]