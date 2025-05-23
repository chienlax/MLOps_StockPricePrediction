# tests/models/test_monitor_performance.py
import io
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import (  # For spec or direct use
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

# Add src to sys.path
PROJECT_ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT_FOR_TESTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models import monitor_performance  # Import the module


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
        'no_retraining_task_id': 'no_retraining_task',
        'no_retraining_needed_task_on_error': 'no_retraining_needed_task_on_error' # Ensure this exists
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
        (110, 105, 100, 1.0), # Pred Up, Actual Up
        (90, 95, 100, 1.0),   # Pred Down, Actual Down
        (100, 100, 100, 1.0), # Pred Flat, Actual Flat
        (110, 95, 100, 0.0),  # Pred Up, Actual Down
        (90, 105, 100, 0.0),  # Pred Down, Actual Up
        (110, 100, 100, 0.0), # Pred Up, Actual Flat
        (100, 105, 100, 0.0), # Pred Flat, Actual Up
        (np.nan, 105, 100, None),
        (110, np.nan, 100, None),
        (110, 105, np.nan, None),
        (110, 105, 105, 0.0), 
        (105, 105, 110, 1.0), 
    ])
    def test_calculate_directional_accuracy(self, predicted_today, actual_today, actual_yesterday, expected_accuracy):
        result = monitor_performance.calculate_directional_accuracy(predicted_today, actual_today, actual_yesterday)
        
        if expected_accuracy is None:
            assert result is None, f"Expected None, got {result} for inputs {predicted_today}, {actual_today}, {actual_yesterday}"
        else:
            assert result == pytest.approx(expected_accuracy), f"Expected {expected_accuracy}, got {result} for inputs {predicted_today}, {actual_today}, {actual_yesterday}"

# --- Tests for run_monitoring ---
class TestRunMonitoring:
    @patch('models.monitor_performance.date')
    @patch('models.monitor_performance.calculate_directional_accuracy')
    @patch('models.monitor_performance.mean_absolute_percentage_error')
    @patch('models.monitor_performance.mean_squared_error')
    @patch('models.monitor_performance.mean_absolute_error')
    @patch('models.monitor_performance.save_daily_performance_metrics')
    @patch('models.monitor_performance.get_db_connection')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.open') 
    @patch('yaml.safe_load')
    def test_run_monitoring_retrain_triggered(self, mock_yaml_load, mock_open, 
                                              mock_get_pred, mock_get_db_conn, mock_save_metrics, 
                                              mock_mae, mock_mse, mock_mape, mock_calc_dir_acc, 
                                              mock_date,
                                              mock_params_config_mon, mock_db_config_mon, mock_airflow_dag_cfg, caplog):
        
        dummy_config_path = "dummy_config.yaml" 
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_load.return_value = mock_params_config_mon
        
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today
        eval_date_obj = (fixed_today - timedelta(days=mock_params_config_mon['monitoring']['evaluation_lag_days']))
        eval_date_str = eval_date_obj.isoformat()

        mock_get_pred.side_effect = lambda db_cfg, date_str, ticker: {
            'predicted_price': 105.0 if ticker == 'TICKA' else 205.0,
            'model_mlflow_run_id': 'mock_run_id_123'
        } if ticker in mock_params_config_mon['data_loading']['tickers'] else None

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            (100.0,), (90.0,), (200.0,), (190.0,)
        ]

        mock_mae.return_value = 5.0 
        mock_mse.return_value = 25.0 
        mock_mape.return_value = 0.15 
        mock_calc_dir_acc.return_value = 0.5 

        result_task_id = monitor_performance.run_monitoring(dummy_config_path)

        mock_open.assert_called_once_with(dummy_config_path, 'r')
        mock_yaml_load.assert_called_once_with(mock_open.return_value.__enter__.return_value)
        mock_get_pred.assert_any_call(mock_db_config_mon, eval_date_str, 'TICKA')
        mock_get_db_conn.assert_called_once_with(mock_db_config_mon)
        
        mock_cursor.execute.assert_any_call(
            "SELECT close FROM raw_stock_data WHERE ticker = %s AND date::date = %s",
            ('TICKA', eval_date_obj) 
        )
        
        assert mock_mape.call_count == 2 
        assert mock_calc_dir_acc.call_count == 2 
        assert mock_save_metrics.call_count == 2 
        
        mock_save_metrics.assert_any_call(
            mock_db_config_mon, eval_date_str, 'TICKA', ANY, 'mock_run_id_123' # --- MODIFIED ---
        )
        mock_save_metrics.assert_any_call(
            mock_db_config_mon, eval_date_str, 'TICKB', ANY, 'mock_run_id_123' # --- MODIFIED ---
        )
        
        assert result_task_id == mock_airflow_dag_cfg['retraining_trigger_task_id']
        assert "Performance thresholds breached. Triggering retraining pipeline." in caplog.text


    @patch('models.monitor_performance.date')
    @patch('models.monitor_performance.calculate_directional_accuracy')
    @patch('models.monitor_performance.mean_absolute_percentage_error')
    @patch('models.monitor_performance.mean_squared_error', return_value=0.01) 
    @patch('models.monitor_performance.mean_absolute_error', return_value=0.1)  
    @patch('models.monitor_performance.save_daily_performance_metrics')
    @patch('models.monitor_performance.get_db_connection')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.open') 
    @patch('yaml.safe_load')
    def test_run_monitoring_no_retrain_needed(self, mock_yaml_load, mock_open, 
                                              mock_get_pred, mock_get_db_conn, mock_save_metrics,
                                              mock_mae, mock_mse, 
                                              mock_mape, mock_calc_dir_acc, mock_date,
                                              mock_params_config_mon, mock_db_config_mon, mock_airflow_dag_cfg, caplog):
        dummy_config_path = "dummy_config.yaml"
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_load.return_value = mock_params_config_mon
        
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today

        mock_get_pred.side_effect = lambda db_cfg, date_str, ticker: {
            'predicted_price': 105.0 if ticker == 'TICKA' else 205.0,
            'model_mlflow_run_id': 'mock_run_id_123'
        } if ticker in mock_params_config_mon['data_loading']['tickers'] else None

        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [(100.0,), (90.0,), (200.0,), (190.0,)]

        mock_mape.return_value = 0.08 
        mock_calc_dir_acc.return_value = 0.70 

        result_task_id = monitor_performance.run_monitoring(dummy_config_path)

        mock_save_metrics.assert_called() 
        assert result_task_id == mock_airflow_dag_cfg['no_retraining_task_id']
        assert "Model performance is within acceptable thresholds. No retraining triggered." in caplog.text


    @patch('models.monitor_performance.date')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.open') 
    @patch('yaml.safe_load')
    def test_run_monitoring_no_predictions_found(self, mock_yaml_load, mock_open, 
                                                 mock_get_pred, mock_date,
                                                 mock_params_config_mon, mock_airflow_dag_cfg, caplog):
        dummy_config_path = "dummy_config.yaml"
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_load.return_value = mock_params_config_mon
        
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today
        mock_get_pred.return_value = None 

        with patch('models.monitor_performance.get_db_connection') as mock_get_db, \
             patch('models.monitor_performance.save_daily_performance_metrics') as mock_save:
            result_task_id = monitor_performance.run_monitoring(dummy_config_path)
            mock_get_db.assert_not_called()
            mock_save.assert_not_called()

        assert result_task_id == mock_airflow_dag_cfg['no_retraining_task_id']
        assert "No predictions retrieved from database for any ticker" in caplog.text
        

    @patch('models.monitor_performance.date')
    @patch('models.monitor_performance.calculate_directional_accuracy')
    @patch('models.monitor_performance.mean_absolute_percentage_error')
    @patch('models.monitor_performance.mean_squared_error', return_value=0.01) 
    @patch('models.monitor_performance.mean_absolute_error', return_value=0.1)  
    @patch('models.monitor_performance.save_daily_performance_metrics')
    @patch('models.monitor_performance.get_db_connection')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.open') 
    @patch('yaml.safe_load')
    def test_run_monitoring_missing_actuals_for_ticker(self, mock_yaml_load, mock_open, 
                                                       mock_get_pred, mock_get_db_conn, mock_save_metrics,
                                                       mock_mae, mock_mse, 
                                                       mock_mape, mock_calc_dir_acc, mock_date,
                                                       mock_params_config_mon, mock_db_config_mon, mock_airflow_dag_cfg, caplog):
        dummy_config_path = "dummy_config.yaml"
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_load.return_value = mock_params_config_mon
        
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today
        eval_date_str = (fixed_today - timedelta(days=mock_params_config_mon['monitoring']['evaluation_lag_days'])).isoformat()

        mock_get_pred.side_effect = lambda db_cfg, date_str, ticker: {
            'predicted_price': 105.0 if ticker == 'TICKA' else 205.0,
            'model_mlflow_run_id': 'mock_run_id_123'
        } if ticker in mock_params_config_mon['data_loading']['tickers'] else None

        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            (100.0,), (90.0,), (None,), (190.0,) 
        ]

        mock_mape.return_value = 0.08
        mock_calc_dir_acc.return_value = 0.70

        result_task_id = monitor_performance.run_monitoring(dummy_config_path)

        assert "Missing predicted or actual price for TICKB" in caplog.text
        mock_save_metrics.assert_called_once() 
        mock_save_metrics.assert_any_call(
             mock_db_config_mon, eval_date_str, 'TICKA', ANY, 'mock_run_id_123' # --- MODIFIED ---
        )
        assert result_task_id == mock_airflow_dag_cfg['no_retraining_task_id']

    @patch('models.monitor_performance.date')
    @patch('models.monitor_performance.calculate_directional_accuracy')
    @patch('models.monitor_performance.mean_absolute_percentage_error')
    @patch('models.monitor_performance.mean_squared_error', return_value=0.01) 
    @patch('models.monitor_performance.mean_absolute_error', return_value=0.1)  
    @patch('models.monitor_performance.save_daily_performance_metrics')
    @patch('models.monitor_performance.get_db_connection')
    @patch('models.monitor_performance.get_prediction_for_date_ticker')
    @patch('models.monitor_performance.open') 
    @patch('yaml.safe_load')
    def test_run_monitoring_metric_is_nan(self, mock_yaml_load, mock_open, 
                                          mock_get_pred, mock_get_db_conn, mock_save_metrics,
                                          mock_mae, mock_mse, 
                                          mock_mape, mock_calc_dir_acc, mock_date,
                                          mock_params_config_mon, mock_db_config_mon, mock_airflow_dag_cfg, caplog):
        dummy_config_path = "dummy_config.yaml"
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_load.return_value = mock_params_config_mon
        
        fixed_today = date(2023, 1, 10)
        mock_date.today.return_value = fixed_today

        mock_get_pred.side_effect = lambda db_cfg, date_str, ticker: {
            'predicted_price': 105.0, 'model_mlflow_run_id': 'mock_run_id_123'
        } 
        
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_conn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.side_effect = [
            (0.0,),  
            (90.0,), 
            (200.0,),
            (190.0,) 
        ]
        
        def mape_side_effect(actual, predicted):
            if actual[0] == 0.0: return np.nan 
            return 0.08 
        mock_mape.side_effect = mape_side_effect
        mock_calc_dir_acc.return_value = 0.70 

        result_task_id = monitor_performance.run_monitoring(dummy_config_path)
        
        assert "MAPE=NaN" in caplog.text 
        assert "MAPE=0.0800" in caplog.text 
        mock_save_metrics.assert_called() 
        assert mock_save_metrics.call_count == 2 
        assert result_task_id == mock_airflow_dag_cfg['no_retraining_task_id']

    @patch('models.monitor_performance.date') 
    @patch('models.monitor_performance.get_prediction_for_date_ticker', side_effect=Exception("Simulated DB Error"))
    @patch('models.monitor_performance.open') 
    @patch('yaml.safe_load')
    def test_run_monitoring_error_fallback(self, mock_yaml_load, mock_open, 
                                           mock_get_pred_with_error, mock_date_with_error, 
                                           mock_params_config_mon, caplog):
        dummy_config_path = "dummy_config.yaml"
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_load.return_value = mock_params_config_mon
        
        fixed_today = date(2023, 1, 10)
        mock_date_with_error.today.return_value = fixed_today

        result_task_id = monitor_performance.run_monitoring(dummy_config_path)

        assert "Error in run_monitoring: Simulated DB Error" in caplog.text
        expected_fallback_id = mock_params_config_mon.get('airflow_dags', {}).get('no_retraining_needed_task_on_error', 'no_retraining_needed_task_on_error')
        assert result_task_id == expected_fallback_id