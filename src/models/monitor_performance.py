# src/models/monitor_performance.py
import argparse
import yaml
import numpy as np
import pandas as pd
import logging
import sys
import os
import json
from typing import Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import date, timedelta, datetime # Ensure all are imported
from pathlib import Path
from src.utils import db_utils

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.db_utils import (
    get_latest_predictions,
    save_daily_performance_metrics,
    load_data_from_db ,
    get_prediction_for_date_ticker,
    get_db_connection
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# -------------------------------------------------------------------

def calculate_directional_accuracy(predicted_today: float, actual_today: float, actual_yesterday: float) -> Optional[float]:
    """Calculates directional accuracy. Returns 1.0 if direction matches, 0.0 otherwise, None if undefined."""
    if pd.isna(predicted_today) or pd.isna(actual_today) or pd.isna(actual_yesterday):
        return None

    pred_change = predicted_today - actual_yesterday
    actual_change = actual_today - actual_yesterday

    # Avoid division by zero or issues if prices are identical
    if pred_change == 0 and actual_change == 0: # No change predicted, no change happened
        return 1.0 
    if pred_change == 0 or actual_change == 0: # One changed, other didn't - ambiguous, count as mismatch or neutral
        return 0.0 # Or handle as 0.5 if preferred for neutrality

    return 1.0 if np.sign(pred_change) == np.sign(actual_change) else 0.0

# -------------------------------------------------------------------

def run_monitoring(config_path: str) -> str:
    """
    Evaluates previous day's predictions, logs performance, and decides if retraining is needed.

    Args:
        config_path (str): Path to params.yaml.

    Returns:
        str: Task ID for Airflow branching (e.g., 'trigger_retraining_task' or 'no_retraining_task').
    """
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        db_config = params['database']
        # Tickers to monitor - should align with what's being predicted
        tickers_to_monitor = params['data_loading']['tickers'] 
        
        monitoring_cfg = params['monitoring']
        thresholds = monitoring_cfg['performance_thresholds']
        evaluation_lag_days = monitoring_cfg.get('evaluation_lag_days', 1)

        airflow_dag_cfg = params.get('airflow_dags', {})
        retraining_task_id = airflow_dag_cfg.get('retraining_trigger_task_id', 'trigger_retraining_pipeline_task') # Task ID in daily DAG
        no_retraining_task_id = airflow_dag_cfg.get('no_retraining_task_id', 'no_retraining_needed_task')

        prediction_date_to_evaluate = date.today() - timedelta(days=evaluation_lag_days)
        prediction_date_str = prediction_date_to_evaluate.isoformat()
        
        # For directional accuracy, we need actuals from one day prior to prediction_date_to_evaluate
        actuals_needed_end_date = prediction_date_to_evaluate
        actuals_needed_start_date = prediction_date_to_evaluate - timedelta(days=1) # Need at least two days of actuals

        logger.info(f"Monitoring performance for predictions made for date: {prediction_date_str}")

        # 1. Fetch predictions made for the evaluation_date
        predictions_data_from_db = {} # To store {ticker: {'predicted_price': val, 'model_mlflow_run_id': id}}
        unique_model_run_ids_found = set()

        for ticker in tickers_to_monitor:
            pred_info = get_prediction_for_date_ticker(db_config, prediction_date_str, ticker)
            if pred_info:
                predictions_data_from_db[ticker] = pred_info
                if pred_info.get('model_mlflow_run_id'):
                    unique_model_run_ids_found.add(pred_info['model_mlflow_run_id'])
            else:
                logger.warning(f"No prediction found in database for {ticker} on target date {prediction_date_str}.")
        
        if not predictions_data_from_db:
            logger.warning(f"No predictions retrieved from database for any ticker on target date {prediction_date_str}. Skipping performance monitoring.")
            return no_retraining_task_id

        # Determine the model_mlflow_run_id to use for logging in model_performance_log.
        # If multiple models were used for different tickers (not typical for this setup but possible),
        # this logic might need refinement. For now, assume one dominant model_run_id or take the first one.
        if len(unique_model_run_ids_found) > 1:
            logger.warning(f"Multiple model_mlflow_run_ids found for predictions on {prediction_date_str}: {unique_model_run_ids_found}. Using the first one for overall log.")
        
        model_mlflow_run_id_for_these_preds = list(unique_model_run_ids_found)[0] if unique_model_run_ids_found else None
        
        if not model_mlflow_run_id_for_these_preds:
            logger.warning(f"Could not determine a model_mlflow_run_id for predictions on {prediction_date_str}. Performance log will be incomplete.")
        # --- END MODIFIED ---

        # Fetch actual prices 
        actual_prices_on_eval_date = {}
        actual_prices_on_prev_date = {}
        
        conn_actuals = None
        try:
            conn_actuals = get_db_connection(db_config)
            cursor_actuals = conn_actuals.cursor()
            for ticker in tickers_to_monitor:
                # Actual for evaluation date
                cursor_actuals.execute(
                    "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
                    (ticker, prediction_date_to_evaluate) # Use date object here
                )
                result = cursor_actuals.fetchone()
                if result: actual_prices_on_eval_date[ticker] = result[0]

                prev_business_day = prediction_date_to_evaluate - timedelta(days=1) 
                cursor_actuals.execute(
                    "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
                    (ticker, prev_business_day) 
                )
                result_prev = cursor_actuals.fetchone()
                if result_prev: actual_prices_on_prev_date[ticker] = result_prev[0]
        finally:
            if conn_actuals:
                if 'cursor_actuals' in locals() and cursor_actuals:
                    cursor_actuals.close()
                conn_actuals.close()

        all_metrics_calculated = []
        retrain_triggered_by_metric = False

        for ticker in tickers_to_monitor:
            pred_info = predictions_data_from_db.get(ticker)
            if not pred_info:
                continue # Already logged warning if prediction was missing

            predicted_price = pred_info.get('predicted_price')
            actual_price = actual_prices_on_eval_date.get(ticker)
            actual_prev_price = actual_prices_on_prev_date.get(ticker)

            if predicted_price is None or actual_price is None:
                logger.warning(f"Missing predicted or actual price for {ticker} on {prediction_date_str} for metric calculation. Predicted: {predicted_price}, Actual: {actual_price}")
                continue

            metrics = {'ticker': ticker, 'prediction_date': prediction_date_str, 
                       'predicted_price': predicted_price, 'actual_price': actual_price}
            metrics['mae'] = mean_absolute_error([actual_price], [predicted_price])
            metrics['rmse'] = np.sqrt(mean_squared_error([actual_price], [predicted_price]))
            if actual_price != 0:
                metrics['mape'] = mean_absolute_percentage_error([actual_price], [predicted_price])
            else:
                metrics['mape'] = np.nan 

            metrics['direction_accuracy'] = calculate_directional_accuracy(predicted_price, actual_price, actual_prev_price) if actual_prev_price is not None else np.nan
            
            logger.info(f"Metrics for {ticker} on {prediction_date_str}: MAPE={metrics['mape']:.4f}, DirAcc={metrics['direction_accuracy'] if not pd.isna(metrics['direction_accuracy']) else 'N/A'}")
            all_metrics_calculated.append(metrics)

            # Use model_mlflow_run_id_for_these_preds which should be common for predictions made for this date
            # If a per-ticker model_run_id was stored and fetched, you could use pred_info['model_mlflow_run_id']
            model_run_id_to_log = model_mlflow_run_id_for_these_preds if model_mlflow_run_id_for_these_preds else pred_info.get('model_mlflow_run_id')

            save_daily_performance_metrics(db_config, prediction_date_str, ticker, metrics, model_run_id_to_log)

            # Check thresholds
            if not pd.isna(metrics['mape']) and metrics['mape'] > thresholds.get('mape_max', float('inf')):
                logger.warning(f"TRIGGER: {ticker} MAPE ({metrics['mape']:.4f}) exceeded threshold ({thresholds.get('mape_max')}).")
                retrain_triggered_by_metric = True
            if not pd.isna(metrics['direction_accuracy']) and metrics['direction_accuracy'] < thresholds.get('direction_accuracy_min', float('-inf')):
                logger.warning(f"TRIGGER: {ticker} Directional Accuracy ({metrics['direction_accuracy']:.4f}) below threshold ({thresholds.get('direction_accuracy_min')}).")
                retrain_triggered_by_metric = True

        if not all_metrics_calculated:
            logger.warning("No metrics were calculated for any ticker. Defaulting to no retraining.")
            return no_retraining_task_id
            
        if retrain_triggered_by_metric:
            logger.info("Performance thresholds breached. Triggering retraining pipeline.")
            return retraining_task_id
        else:
            logger.info("Model performance is within acceptable thresholds. No retraining triggered.")
            return no_retraining_task_id

    except Exception as e:
        logger.error(f"Error in run_monitoring: {e}", exc_info=True)
        return params.get('airflow_dags', {}).get('no_retraining_task_id', 'no_retraining_needed_task_on_error') # Fallback on error

# -------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Monitor model performance and decide on retraining.")
    parser.add_argument(
        '--config', type=str, default='config/params.yaml',
        help='Path to the configuration file (e.g., config/params.yaml)'
    )
    args = parser.parse_args()

    config_path_resolved = Path(args.config).resolve()
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)

    logger.info(f"Starting model performance monitoring with config: {config_path_resolved}")
    
    next_task_id = run_monitoring(str(config_path_resolved))
    
    logger.info(f"Monitoring complete. Next Airflow task to run: {next_task_id}")
    print(f"NEXT_TASK_ID:{next_task_id}") # For Airflow BranchPythonOperator   