"""
Monitor stock price prediction model performance and trigger retraining when needed.

This module evaluates the performance of previous day's stock price predictions
by comparing against actual values, calculates key metrics like MAPE and directional
accuracy, and determines if model retraining is necessary based on configurable thresholds.
"""

# Standard library imports
import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Third-party imports
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

# Set up import path for local modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Local imports
from utils.db_utils import (
    get_db_connection,
    get_prediction_for_date_ticker,
    save_daily_performance_metrics,
)

# Set up logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# -------------------------------------------------------------------


def calculate_directional_accuracy(
    predicted_today: float, actual_today: float, actual_yesterday: float
) -> Optional[float]:
    """
    Calculate if the predicted price direction matches actual price direction.
    
    Args:
        predicted_today: Predicted stock price for today
        actual_today: Actual stock price for today
        actual_yesterday: Actual stock price for yesterday
        
    Returns:
        1.0 if direction matches, 0.0 if direction differs, None if undefined
    """
    if pd.isna(predicted_today) or pd.isna(actual_today) or pd.isna(actual_yesterday):
        return None

    pred_change = predicted_today - actual_yesterday
    actual_change = actual_today - actual_yesterday

    if pred_change == 0 and actual_change == 0:
        return 1.0
    if np.sign(pred_change) == np.sign(actual_change):
        return 1.0
    else:
        return 0.0


# -------------------------------------------------------------------


def run_monitoring(config_path: str) -> str:
    """
    Evaluate previous day's predictions, log performance metrics, and decide if retraining is needed.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ID of the next Airflow task to run (retraining or no-retraining)
    """
    params: Dict = {}  # Initialize params
    # Define default fallback task_id outside try block
    default_fallback_task_id = 'no_retraining_needed_task_on_error'
    
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        db_config = params['database']
        tickers_to_monitor = params['data_loading']['tickers']
        
        monitoring_cfg = params['monitoring']
        thresholds = monitoring_cfg['performance_thresholds']
        evaluation_lag_days = monitoring_cfg.get('evaluation_lag_days', 1)

        airflow_dag_cfg = params.get('airflow_dags', {})
        retraining_task_id = airflow_dag_cfg.get(
            'retraining_trigger_task_id', 'trigger_retraining_pipeline_task'
        )
        no_retraining_task_id = airflow_dag_cfg.get(
            'no_retraining_task_id', 'no_retraining_needed_task'
        )
        # Use the value from params if available, otherwise use the default
        error_fallback_task_id = airflow_dag_cfg.get(
            'no_retraining_needed_task_on_error', default_fallback_task_id
        )

        prediction_date_to_evaluate = date.today() - timedelta(days=evaluation_lag_days)
        prediction_date_str = prediction_date_to_evaluate.isoformat()
        
        logger.info(
            f"Monitoring performance for predictions made for date: {prediction_date_str}"
        )

        predictions_data_from_db = {}
        unique_model_run_ids_found = set()

        for ticker in tickers_to_monitor:
            pred_info = get_prediction_for_date_ticker(
                db_config, prediction_date_str, ticker
            )
            if pred_info:
                predictions_data_from_db[ticker] = pred_info
                if pred_info.get('model_mlflow_run_id'):
                    unique_model_run_ids_found.add(pred_info['model_mlflow_run_id'])
            else:
                logger.warning(
                    f"No prediction found in database for {ticker} on target date "
                    f"{prediction_date_str}."
                )
        
        if not predictions_data_from_db:
            logger.warning(
                f"No predictions retrieved from database for any ticker on target date "
                f"{prediction_date_str}. Skipping performance monitoring."
            )
            return no_retraining_task_id

        if len(unique_model_run_ids_found) > 1:
            logger.warning(
                f"Multiple model_mlflow_run_ids found for predictions on "
                f"{prediction_date_str}: {unique_model_run_ids_found}. "
                f"Using the first one for overall log."
            )
        
        model_mlflow_run_id_for_these_preds = (
            list(unique_model_run_ids_found)[0] if unique_model_run_ids_found else None
        )
        
        if not model_mlflow_run_id_for_these_preds:
            logger.warning(
                f"Could not determine a model_mlflow_run_id for predictions on "
                f"{prediction_date_str}. Performance log will be incomplete."
            )

        actual_prices_on_eval_date = {}
        actual_prices_on_prev_date = {}
        
        conn_actuals = None
        try:
            conn_actuals = get_db_connection(db_config)
            cursor_actuals = conn_actuals.cursor()  # type: ignore
            for ticker in tickers_to_monitor:
                cursor_actuals.execute(
                    "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
                    (ticker, prediction_date_to_evaluate)
                )
                result = cursor_actuals.fetchone()
                if result:
                    actual_prices_on_eval_date[ticker] = result[0]

                prev_business_day = prediction_date_to_evaluate - timedelta(days=1)
                cursor_actuals.execute(
                    "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
                    (ticker, prev_business_day)
                )
                result_prev = cursor_actuals.fetchone()
                if result_prev:
                    actual_prices_on_prev_date[ticker] = result_prev[0]
        finally:
            if conn_actuals:
                if 'cursor_actuals' in locals() and cursor_actuals:  # type: ignore
                    cursor_actuals.close()
                conn_actuals.close()

        all_metrics_calculated = []
        retrain_triggered_by_metric = False

        for ticker in tickers_to_monitor:
            pred_info = predictions_data_from_db.get(ticker)
            if not pred_info:
                continue

            predicted_price = pred_info.get('predicted_price')
            actual_price = actual_prices_on_eval_date.get(ticker)
            actual_prev_price = actual_prices_on_prev_date.get(ticker)

            if predicted_price is None or actual_price is None:
                logger.warning(
                    f"Missing predicted or actual price for {ticker} on "
                    f"{prediction_date_str} for metric calculation. "
                    f"Predicted: {predicted_price}, Actual: {actual_price}"
                )
                continue

            metrics = {
                'ticker': ticker,
                'prediction_date': prediction_date_str,
                'predicted_price': predicted_price,
                'actual_price': actual_price
            }
            metrics['mae'] = mean_absolute_error([actual_price], [predicted_price])
            metrics['rmse'] = np.sqrt(mean_squared_error([actual_price], [predicted_price]))
            if actual_price != 0:
                metrics['mape'] = mean_absolute_percentage_error(
                    [actual_price], [predicted_price]
                )
            else:
                metrics['mape'] = np.nan

            metrics['direction_accuracy'] = (
                calculate_directional_accuracy(
                    predicted_price, actual_price, actual_prev_price
                )
                if actual_prev_price is not None else np.nan
            )
            
            # Format metrics for logging
            mape_str = f"{metrics['mape']:.4f}" if not pd.isna(metrics['mape']) else 'NaN'
            dir_acc_str = (
                f"{metrics['direction_accuracy']:.2f}" 
                if not pd.isna(metrics['direction_accuracy']) else 'N/A'
            )
            logger.info(
                f"Metrics for {ticker} on {prediction_date_str}: "
                f"MAPE={mape_str}, DirAcc={dir_acc_str}"
            )
            
            all_metrics_calculated.append(metrics)

            model_run_id_to_log = (
                model_mlflow_run_id_for_these_preds 
                if model_mlflow_run_id_for_these_preds else pred_info.get('model_mlflow_run_id')
            )

            save_daily_performance_metrics(
                db_config, prediction_date_str, ticker, metrics, model_run_id_to_log
            )

            # Check if metrics exceed thresholds for retraining
            if not pd.isna(metrics['mape']) and metrics['mape'] > thresholds.get('mape_max', float('inf')):
                logger.warning(
                    f"TRIGGER: {ticker} MAPE ({metrics['mape']:.4f}) exceeded threshold "
                    f"({thresholds.get('mape_max')})."
                )
                retrain_triggered_by_metric = True
                
            if (not pd.isna(metrics['direction_accuracy']) and 
                    metrics['direction_accuracy'] < thresholds.get('direction_accuracy_min', float('-inf'))):
                logger.warning(
                    f"TRIGGER: {ticker} Directional Accuracy ({metrics['direction_accuracy']:.2f}) "
                    f"below threshold ({thresholds.get('direction_accuracy_min')})."
                )
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
        # Use error_fallback_task_id defined earlier
        if (params and 'airflow_dags' in params and 
                'no_retraining_task_id' in params['airflow_dags']):  # Check if params loaded
            return params.get('airflow_dags', {}).get(
                'no_retraining_needed_task_on_error', default_fallback_task_id
            )
        return default_fallback_task_id


# -------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Monitor model performance and decide on retraining."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/params.yaml',
        help='Path to the configuration file (e.g., config/params.yaml)'
    )
    args = parser.parse_args()

    config_path_resolved = Path(args.config).resolve()
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)

    logger.info(f"Starting model performance monitoring with config: {config_path_resolved}")

    next_task_id_val = run_monitoring(str(config_path_resolved))

    logger.info(f"Monitoring complete. Next Airflow task to run: {next_task_id_val}")
    print(f"NEXT_TASK_ID:{next_task_id_val}")  # For Airflow BranchPythonOperator