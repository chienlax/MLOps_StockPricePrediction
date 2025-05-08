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
    load_data_from_db 
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
        # This could come from params['data_loading']['tickers'] or dynamically from latest_predictions table
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
        # get_latest_predictions returns a dict {ticker: {'timestamp': ..., 'predicted_price': ..., 'model_run_id': ...}}
        # We need to filter this for the specific prediction_date_to_evaluate if it stores more than one day.
        # For now, assuming get_latest_predictions gives the most recent, which should be for 'today's prediction'
        # So, if evaluation_lag_days=1, we need predictions made 'yesterday' for 'yesterday'.
        # Let's adjust: fetch predictions that were *recorded* around prediction_date_to_evaluate
        # This part needs careful alignment with how `latest_predictions` table is populated.
        # Assuming `latest_predictions` stores the prediction for `prediction_date_to_evaluate` made on that day or day before.
        
        # Simpler: Assume `latest_predictions` table has one row per ticker, updated daily.
        # The `prediction_timestamp` in that table tells us when the prediction was stored.
        # The `payload["date"]` in predict_model.py tells us *for which date* the prediction is.
        # We need to query `latest_predictions` where the *prediction target date* (not timestamp) matches.
        # This requires `latest_predictions` to store the target date of the prediction.
        # Let's assume `db_utils.get_latest_predictions()` is adapted or we query directly.
        # For now, let's assume `get_latest_predictions` gives us the predictions we need to evaluate.
        # A robust way: query historical JSONs or adapt DB to store target_prediction_date.
        
        # Let's refine: We need to fetch predictions that were *targeted* for `prediction_date_to_evaluate`.
        # The `latest_predictions.json` and historical files store this.
        # The `latest_predictions` DB table should ideally store `target_prediction_date`.
        # If `latest_predictions` DB table's `prediction_timestamp` is the *time of prediction*,
        # and we assume predictions are made for `date.today()`, then for `evaluation_lag_days=1`,
        # we look for predictions made `evaluation_lag_days` ago.
        # This is getting complex. Let's simplify:
        # Assume the `latest_predictions.json` file (or DB equivalent) correctly stores predictions
        # with a "date" field indicating the date the prediction is FOR.

        # We will load the historical JSON for the prediction_date_to_evaluate
        # This is simpler than complex DB queries for now.
        predictions_base_dir_str = params.get('output_paths', {}).get('predictions_dir', 'data/predictions')
        historical_pred_file = Path(predictions_base_dir_str) / 'historical' / f"{prediction_date_str}.json"
        
        predictions_to_evaluate = {}
        model_mlflow_run_id_for_these_preds = None
        if historical_pred_file.exists():
            with open(historical_pred_file, 'r') as f:
                pred_data = json.load(f)
            if pred_data.get("date") == prediction_date_str:
                predictions_to_evaluate = pred_data.get("predictions", {})
                model_mlflow_run_id_for_these_preds = pred_data.get("model_mlflow_run_id")
            else:
                logger.warning(f"Date mismatch in {historical_pred_file}. Expected {prediction_date_str}, got {pred_data.get('date')}")
        else:
            logger.warning(f"Prediction file not found for {prediction_date_str} at {historical_pred_file}")

        if not predictions_to_evaluate or not model_mlflow_run_id_for_these_preds:
            logger.warning(f"No predictions found or model_run_id missing for {prediction_date_str}. Skipping performance monitoring for this date.")
            return no_retraining_task_id # Default to no retraining if no data to evaluate

        # 2. Fetch actual prices for evaluation_date and (evaluation_date - 1 day)
        logger.info(f"Fetching actual prices for {tickers_to_monitor} around {prediction_date_str}...")
        # Use load_data_from_db and filter for specific dates
        # This is inefficient if called repeatedly. A dedicated function in db_utils would be better.
        # `get_actual_prices_for_dates(db_config, tickers, date_list)`
        
        # For simplicity, let's assume we can query raw_stock_data directly here for the two dates.
        actual_prices_on_eval_date = {}
        actual_prices_on_prev_date = {}
        
        conn = None # Ensure conn is defined for finally block
        try:
            conn = db_utils.get_db_connection(db_config) # Assuming db_utils is imported
            cursor = conn.cursor()
            for ticker in tickers_to_monitor:
                # Actual for evaluation date
                cursor.execute(
                    "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
                    (ticker, prediction_date_to_evaluate)
                )
                result = cursor.fetchone()
                if result: actual_prices_on_eval_date[ticker] = result[0]

                # Actual for previous day (for directional accuracy)
                prev_business_day = prediction_date_to_evaluate - timedelta(days=1) # Simplistic, doesn't handle weekends/holidays well
                # A robust solution would use pandas market holiday calendars or fetch a small window and pick latest before eval_date
                cursor.execute(
                    "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
                    (ticker, prev_business_day) 
                )
                result_prev = cursor.fetchone()
                if result_prev: actual_prices_on_prev_date[ticker] = result_prev[0]
        finally:
            if conn:
                cursor.close()
                conn.close()

        all_metrics = []
        retrain_triggered_by_metric = False

        for ticker in tickers_to_monitor:
            predicted_price = predictions_to_evaluate.get(ticker)
            actual_price = actual_prices_on_eval_date.get(ticker)
            actual_prev_price = actual_prices_on_prev_date.get(ticker)

            if predicted_price is None or actual_price is None:
                logger.warning(f"Missing predicted or actual price for {ticker} on {prediction_date_str}. Skipping metrics calculation for it.")
                continue

            metrics = {'ticker': ticker, 'prediction_date': prediction_date_str, 'predicted_price': predicted_price, 'actual_price': actual_price}
            metrics['mae'] = mean_absolute_error([actual_price], [predicted_price])
            metrics['rmse'] = np.sqrt(mean_squared_error([actual_price], [predicted_price]))
            if actual_price != 0:
                metrics['mape'] = mean_absolute_percentage_error([actual_price], [predicted_price])
            else:
                metrics['mape'] = np.nan # Avoid division by zero

            metrics['direction_accuracy'] = calculate_directional_accuracy(predicted_price, actual_price, actual_prev_price) if actual_prev_price is not None else np.nan
            
            logger.info(f"Metrics for {ticker} on {prediction_date_str}: MAPE={metrics['mape']:.4f}, DirAcc={metrics['direction_accuracy']}")
            all_metrics.append(metrics)

            # Save individual ticker metrics to DB
            save_daily_performance_metrics(db_config, prediction_date_str, ticker, metrics, model_mlflow_run_id_for_these_preds)

            # Check thresholds for this ticker
            if not pd.isna(metrics['mape']) and metrics['mape'] > thresholds.get('mape_max', float('inf')):
                logger.warning(f"TRIGGER: {ticker} MAPE ({metrics['mape']:.4f}) exceeded threshold ({thresholds.get('mape_max')}).")
                retrain_triggered_by_metric = True
            if not pd.isna(metrics['direction_accuracy']) and metrics['direction_accuracy'] < thresholds.get('direction_accuracy_min', float('-inf')):
                logger.warning(f"TRIGGER: {ticker} Directional Accuracy ({metrics['direction_accuracy']:.4f}) below threshold ({thresholds.get('direction_accuracy_min')}).")
                retrain_triggered_by_metric = True
            # Add checks for other metrics (RMSE, MAE) if defined in thresholds

        if not all_metrics:
            logger.warning("No metrics were calculated for any ticker. Defaulting to no retraining.")
            return no_retraining_task_id

        # Optional: Aggregate metrics (e.g., average MAPE across all monitored tickers)
        # avg_mape = np.nanmean([m['mape'] for m in all_metrics if 'mape' in m])
        # logger.info(f"Average MAPE across all tickers: {avg_mape:.4f}")
        # if avg_mape > thresholds.get('avg_mape_max', float('inf')): # Example for aggregated threshold
        #     logger.warning(f"TRIGGER: Average MAPE ({avg_mape:.4f}) exceeded threshold.")
        #     retrain_triggered_by_metric = True
            
        if retrain_triggered_by_metric:
            logger.info("Performance thresholds breached. Triggering retraining pipeline.")
            return retraining_task_id
        else:
            logger.info("Model performance is within acceptable thresholds. No retraining triggered.")
            return no_retraining_task_id

    except Exception as e:
        logger.error(f"Error in run_monitoring: {e}", exc_info=True)
        # Default to no retraining on error to avoid unintended retraining loops
        return params.get('airflow_dags', {}).get('no_retraining_task_id', 'no_retraining_needed_task_on_error')

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