# tests/models/test_monitor_performance.py
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import pandas as pd
import yaml
from datetime import date, timedelta, datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error # For spec or direct use

# Add src to sys.path
PROJECT_ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT_FOR_TESTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models import monitor_performance # Import the module
from utils import db_utils # Need to mock functions from here

# --- Fixtures ---
@pytest.fixture
def mock_db_config_mon():
    return {'dbname': 'test_mon_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_monitoring_params_cfg():
    return {
        'evaluation_lag_days': 1,
        'performance_thresholds': {
            'mape_max': 0.10, # 10%
            'direction_accuracy_min': 0.60 # 60%
        }
    }

@pytest.fixture
def mock_airflow_dag_cfg():
    return {
        'retraining_trigger_task_id': 'trigger_retraining_task',
        'no_retraining_task_id': 'no_retraining_task'
    }

@pytest.fixture
def mock_params_config_mon(mock_db_config_mon, mock_monitoring_params_cfg, mock_airflow_dag_cfg):
    return {
        'database': mock_db_config_mon,
        'data_loading': {'tickers': ['TICKA', 'TICKB']}, # Tickers to monitor
        'monitoring': mock_monitoring_params_cfg,
        'airflow_dags': mock_airflow_dag_cfg
    }

# --- Tests for calculate_directional_accuracy ---
class TestCalculateDirectionalAccuracy:
    @pytest.mark.parametrize("predicted_today, actual_today, actual_yesterday, expected_accuracy", [
        (110, 105, 100, 1.0), # Both up
        (90, 95, 100, 1.0),   # Both down
        (100, 100, 100, 1.0), # Both flat
        (110, 95, 100, 0.0),  # Predicted up, actual down
        (90, 105, 100, 0.0),  # Predicted down, actual up
        (110, 100, 100, 0.0), # Predicted up, actual flat
        (100, 105, 100, 0.0), # Predicted flat, actual up
        (np.nan, 105, 100, None), # NaN predicted
        (110, np.nan, 100, None), # NaN actual today
        (110, 105, np.nan, None), # NaN actual yesterday
        (110, 105, 105, 0.0), # Predicted up, actual flat (from yesterday's close)
        (105, 105, 110, 0.0), # Predicted flat, actual down (from yesterday's close)
    ])
    def test_calculate_directional_accuracy(self, predicted_today, actual_today, actual_yesterday, expected_accuracy):
        result = monitor_performance.calculate_directional_accuracy(predicted_today, actual_today, actual_yesterday)
        
        if expected_accuracy is None:
            assert result is None
        else:
            assert result == expected_accuracy

# --- Tests for run_monitoring ---
class TestRunMonitoring:
    @patch('models.monitor_performance.yaml.safe_load')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.get_db_connection') # To mock fetching actuals
    @patch('models.monitor_performance.save_daily_performance_metrics')
    @patch('models.monitor_performance.mean_absolute_error')
    @patch('models.monitor_performance.mean_squared_error')
    @patch('models.monitor_performance.mean_absolute_percentage_error')
    @patch('models.monitor_performance.calculate_directional_accuracy')
    @patch('models.monitor_performance.date') # Mock date.today()
    def test_run_monitoring_retrain_triggered(self, mock_date, mock_calc_dir_acc, mock_mape, mock_mse, mock_mae,
                                              mock_save_metrics, mock_get_db_conn, mock_get_pred, mock_yaml_load,
                                              mock_params_config_mon, mock_db_config_mon, mock_airflow_dag_cfg, caplog):
        
        mock_yaml_load.return_value = mock_params_config_mon
        
        # Mock date.today() to control the evaluation date
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today
        eval_date_str = (fixed_today - timedelta(days=mock_params_config_mon['monitoring']['evaluation_lag_days'])).isoformat()

        # Mock predictions fetched from DB
        mock_get_pred.side_effect = lambda db_cfg, date_str, ticker: {
            'predicted_price': 105.0 if ticker == 'TICKA' else 205.0,
            'model_mlflow_run_id': 'mock_run_id_123'
        } if ticker in mock_params_config_mon['data_loading']['tickers'] else None

        # Mock fetching actuals from DB (using a mock cursor)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Simulate cursor results for actual prices (evaluation date and prev day)
        # TICKA: Actual 100 on eval date, 90 prev day (Up prediction 105 vs Actual 100 is wrong direction)
        # TICKB: Actual 200 on eval date, 190 prev day (Up prediction 205 vs Actual 200 is wrong direction)
        mock_cursor.fetchone.side_effect = [
            (100.0,), # TICKA actual on eval date
            (90.0,),  # TICKA actual prev day
            (200.0,), # TICKB actual on eval date
            (190.0,)  # TICKB actual prev day
        ]

        # Mock metric calculations to trigger retraining
        mock_mae.return_value = 5.0 # dummy
        mock_mse.return_value = 25.0 # dummy
        mock_mape.return_value = 0.15 # MAPE > 0.10 threshold
        mock_calc_dir_acc.return_value = 0.5 # Dir Acc < 0.60 threshold

        # Call the function
        result_task_id = monitor_performance.run_monitoring("dummy_config.yaml")

        # Assertions
        mock_yaml_load.assert_called_once()
        mock_get_pred.assert_any_call(mock_db_config_mon, eval_date_str, 'TICKA')
        mock_get_pred.assert_any_call(mock_db_config_mon, eval_date_str, 'TICKB')
        
        mock_get_db_conn.assert_called_once_with(mock_db_config_mon)
        # Check cursor execute calls for actuals
        mock_cursor.execute.assert_any_call(
            "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
            ('TICKA', date(2023, 1, 9)) # Eval date
        )
        mock_cursor.execute.assert_any_call(
            "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
            ('TICKA', date(2023, 1, 8)) # Prev day
        )
        mock_cursor.execute.assert_any_call(
            "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
            ('TICKB', date(2023, 1, 9)) # Eval date
        )
        mock_cursor.execute.assert_any_call(
            "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
            ('TICKB', date(2023, 1, 8)) # Prev day
        )

        # Check metric calculation calls (called for each ticker)
        assert mock_mae.call_count == 2
        assert mock_mape.call_count == 2
        assert mock_calc_dir_acc.call_count == 2

        # Check save_daily_performance_metrics calls (called for each ticker)
        assert mock_save_metrics.call_count == 2
        mock_save_metrics.assert_any_call(
            mock_db_config_mon, eval_date_str, 'TICKA', pytest.ANY, 'mock_run_id_123'
        )
        mock_save_metrics.assert_any_call(
            mock_db_config_mon, eval_date_str, 'TICKB', pytest.ANY, 'mock_run_id_123'
        )

        assert result_task_id == mock_airflow_dag_cfg['retraining_trigger_task_id']
        assert "Performance thresholds breached. Triggering retraining pipeline." in caplog.text


    @patch('models.monitor_performance.yaml.safe_load')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.get_db_connection')
    @patch('models.monitor_performance.save_daily_performance_metrics')
    @patch('models.monitor_performance.mean_absolute_percentage_error')
    @patch('models.monitor_performance.calculate_directional_accuracy')
    @patch('models.monitor_performance.date')
    def test_run_monitoring_no_retrain_needed(self, mock_date, mock_calc_dir_acc, mock_mape,
                                              mock_save_metrics, mock_get_db_conn, mock_get_pred, mock_yaml_load,
                                              mock_params_config_mon, mock_db_config_mon, mock_airflow_dag_cfg, caplog):
        mock_yaml_load.return_value = mock_params_config_mon
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today
        eval_date_str = (fixed_today - timedelta(days=mock_params_config_mon['monitoring']['evaluation_lag_days'])).isoformat()

        # Mock predictions
        mock_get_pred.side_effect = lambda db_cfg, date_str, ticker: {
            'predicted_price': 105.0 if ticker == 'TICKA' else 205.0,
            'model_mlflow_run_id': 'mock_run_id_123'
        } if ticker in mock_params_config_mon['data_loading']['tickers'] else None

        # Mock actuals (same as before)
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            (100.0,), (90.0,), (200.0,), (190.0,)
        ]

        # Mock metric calculations to NOT trigger retraining
        mock_mape.return_value = 0.08 # MAPE < 0.10 threshold
        mock_calc_dir_acc.return_value = 0.70 # Dir Acc > 0.60 threshold

        # Call the function
        result_task_id = monitor_performance.run_monitoring("dummy_config.yaml")

        # Assertions (similar calls as before, but check return value and log)
        mock_save_metrics.assert_called() # Ensure metrics were saved
        assert result_task_id == mock_airflow_dag_cfg['no_retraining_task_id']
        assert "Model performance is within acceptable thresholds. No retraining triggered." in caplog.text


    @patch('models.monitor_performance.yaml.safe_load')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.date')
    def test_run_monitoring_no_predictions_found(self, mock_date, mock_get_pred, mock_yaml_load,
                                                 mock_params_config_mon, mock_airflow_dag_cfg, caplog):
        mock_yaml_load.return_value = mock_params_config_mon
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today

        mock_get_pred.return_value = None # No predictions found for any ticker

        # Call the function
        result_task_id = monitor_performance.run_monitoring("dummy_config.yaml")

        # Assertions
        assert result_task_id == mock_airflow_dag_cfg['no_retraining_task_id']
        assert "No predictions retrieved from database for any ticker" in caplog.text
        # Ensure no attempts to fetch actuals or save metrics
        with patch('models.monitor_performance.get_db_connection') as mock_get_db_conn, \
             patch('models.monitor_performance.save_daily_performance_metrics') as mock_save_metrics:
             # Re-run to check mocks
             monitor_performance.run_monitoring("dummy_config.yaml")
             mock_get_db_conn.assert_not_called()
             mock_save_metrics.assert_not_called()

    @patch('models.monitor_performance.yaml.safe_load')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.get_db_connection')
    @patch('models.monitor_performance.save_daily_performance_metrics')
    @patch('models.monitor_performance.mean_absolute_percentage_error')
    @patch('models.monitor_performance.calculate_directional_accuracy')
    @patch('models.monitor_performance.date')
    def test_run_monitoring_missing_actuals_for_ticker(self, mock_date, mock_calc_dir_acc, mock_mape,
                                                       mock_save_metrics, mock_get_db_conn, mock_get_pred, mock_yaml_load,
                                                       mock_params_config_mon, mock_db_config_mon, mock_airflow_dag_cfg, caplog):
        mock_yaml_load.return_value = mock_params_config_mon
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today
        eval_date_str = (fixed_today - timedelta(days=mock_params_config_mon['monitoring']['evaluation_lag_days'])).isoformat()

        # Mock predictions for both tickers
        mock_get_pred.side_effect = lambda db_cfg, date_str, ticker: {
            'predicted_price': 105.0 if ticker == 'TICKA' else 205.0,
            'model_mlflow_run_id': 'mock_run_id_123'
        } if ticker in mock_params_config_mon['data_loading']['tickers'] else None

        # Mock actuals - simulate missing actual for TICKB on eval date
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            (100.0,), # TICKA actual on eval date
            (90.0,),  # TICKA actual prev day
            (None,),  # TICKB actual on eval date - MISSING
            (190.0,)  # TICKB actual prev day
        ]

        # Mock metric calculations (will only be called for TICKA)
        mock_mape.return_value = 0.08
        mock_calc_dir_acc.return_value = 0.70

        # Call the function
        result_task_id = monitor_performance.run_monitoring("dummy_config.yaml")

        # Assertions
        assert "Missing predicted or actual price for TICKB" in caplog.text
        mock_save_metrics.assert_called_once() # Only called for TICKA
        mock_save_metrics.assert_any_call(
             mock_db_config_mon, eval_date_str, 'TICKA', pytest.ANY, 'mock_run_id_123'
        )
        # Branching decision is based on metrics calculated *if any*.
        # Since TICKA metrics were good, it should not trigger retraining.
        assert result_task_id == mock_airflow_dag_cfg['no_retraining_task_id']

    @patch('models.monitor_performance.yaml.safe_load')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.get_db_connection')
    @patch('models.monitor_performance.save_daily_performance_metrics')
    @patch('models.monitor_performance.mean_absolute_percentage_error')
    @patch('models.monitor_performance.calculate_directional_accuracy')
    @patch('models.monitor_performance.date')
    def test_run_monitoring_metric_is_nan(self, mock_date, mock_calc_dir_acc, mock_mape,
                                          mock_save_metrics, mock_get_db_conn, mock_get_pred, mock_yaml_load,
                                          mock_params_config_mon, mock_db_config_mon, mock_airflow_dag_cfg, caplog):
        mock_yaml_load.return_value = mock_params_config_mon
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today
        eval_date_str = (fixed_today - timedelta(days=mock_params_config_mon['monitoring']['evaluation_lag_days'])).isoformat()

        # Mock predictions
        mock_get_pred.side_effect = lambda db_cfg, date_str, ticker: {
            'predicted_price': 105.0, 'model_mlflow_run_id': 'mock_run_id_123'
        } if ticker == 'TICKA' else None

        # Mock actuals
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            (0.0,), # TICKA actual on eval date is 0 (causes MAPE NaN)
            (90.0,), # TICKA actual prev day
        ]

        # Mock metric calculations - MAPE will be NaN, Dir Acc will be calculated
        mock_mape.return_value = np.nan
        mock_calc_dir_acc.return_value = 0.70 # Good Dir Acc

        # Call the function
        result_task_id = monitor_performance.run_monitoring("dummy_config.yaml")

        # Assertions
        assert "MAPE=nan" in caplog.text # Check log reflects NaN
        mock_save_metrics.assert_called_once() # Metrics saved, including NaN MAPE
        
        # Branching decision: MAPE is NaN (doesn't trigger), Dir Acc is good (doesn't trigger)
        assert result_task_id == mock_airflow_dag_cfg['no_retraining_task_id']

    @patch('models.monitor_performance.yaml.safe_load')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.get_db_connection')
    @patch('models.monitor_performance.save_daily_performance_metrics')
    @patch('models.monitor_performance.mean_absolute_percentage_error')
    @patch('models.monitor_performance.calculate_directional_accuracy')
    @patch('models.monitor_performance.date')
    def test_run_monitoring_error_fallback(self, mock_date, mock_calc_dir_acc, mock_mape,
                                           mock_save_metrics, mock_get_db_conn, mock_get_pred, mock_yaml_load,
                                           mock_params_config_mon, caplog):
        mock_yaml_load.return_value = mock_params_config_mon
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today

        # Simulate an error during processing (e.g., DB error fetching predictions)
        mock_get_pred.side_effect = Exception("Simulated DB Error")

        # Call the function - expect it to catch and return fallback task ID
        result_task_id = monitor_performance.run_monitoring("dummy_config.yaml")

        # Assertions
        assert "Error in run_monitoring" in caplog.text
        # Check for the fallback task ID defined in the function's error handling
        assert result_task_id == 'no_retraining_needed_task_on_error' # Hardcoded fallback in the script