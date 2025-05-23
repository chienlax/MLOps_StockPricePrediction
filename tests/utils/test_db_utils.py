# tests/utils/test_examples.py (or test_db_utils.py)
import unittest
from unittest.mock import patch, MagicMock, ANY, call
import psycopg2
import psycopg2.extras # For DictCursor
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, date, timedelta
from decimal import Decimal # Import Decimal
from pathlib import Path
import sys

# Ensure src is in a discoverable path for tests
PROJECT_ROOT = Path(__file__).resolve().parents[2] 
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import db_utils

# Sample DB config for tests
SAMPLE_DB_CONFIG = {
    'dbname': 'test_db', 'user': 'test_user', 'password': 'test_password',
    'host': 'localhost', 'port': '5432'
}

# Simple pickleable class for scaler tests
class SimpleScaler:
    def __init__(self, min_val, scale_val):
        self.min_ = np.array([min_val])
        self.scale_ = np.array([scale_val])

    def __eq__(self, other): 
        if not isinstance(other, SimpleScaler):
            return False
        return np.array_equal(self.min_, other.min_) and \
               np.array_equal(self.scale_, other.scale_)

class TestDBUtils(unittest.TestCase):

    @patch('utils.db_utils.psycopg2.connect')
    def test_get_db_connection_success(self, mock_connect):
        mock_conn_instance = MagicMock()
        mock_connect.return_value = mock_conn_instance
        conn = db_utils.get_db_connection(SAMPLE_DB_CONFIG)
        mock_connect.assert_called_once_with(
            dbname=SAMPLE_DB_CONFIG['dbname'], user=SAMPLE_DB_CONFIG['user'],
            password=SAMPLE_DB_CONFIG['password'], host=SAMPLE_DB_CONFIG['host'],
            port=SAMPLE_DB_CONFIG['port']
        )
        self.assertEqual(conn, mock_conn_instance)

    @patch('utils.db_utils.psycopg2.connect')
    def test_get_db_connection_failure(self, mock_connect):
        mock_connect.side_effect = psycopg2.OperationalError("Connection failed")
        with self.assertRaises(psycopg2.OperationalError):
            db_utils.get_db_connection(SAMPLE_DB_CONFIG)

    @patch('utils.db_utils.get_db_connection')
    def test_setup_database(self, mock_get_db_connection):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        db_utils.setup_database(SAMPLE_DB_CONFIG)
        executed_sql = [c[0][0].strip() for c in mock_cursor.execute.call_args_list]
        # Check for presence of key table/index creation statements
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS raw_stock_data" in s for s in executed_sql))
        self.assertTrue(any("CREATE INDEX IF NOT EXISTS idx_raw_ticker_date ON raw_stock_data (ticker, date)" in s for s in executed_sql))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS processed_feature_data" in s for s in executed_sql))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS scaled_feature_sets" in s for s in executed_sql))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS scalers" in s for s in executed_sql))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS latest_predictions" in s for s in executed_sql))
        self.assertTrue(any("CREATE INDEX IF NOT EXISTS idx_latest_predictions_ticker_target_date" in s for s in executed_sql))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS optimization_results" in s for s in executed_sql))
        self.assertTrue(any("CREATE TABLE IF NOT EXISTS model_performance_log" in s for s in executed_sql))
        self.assertTrue(any("CREATE INDEX IF NOT EXISTS idx_perf_log_date_ticker ON model_performance_log (prediction_date, ticker)" in s for s in executed_sql))
        
        mock_conn.commit.assert_called_once()
        self.assertEqual(mock_cursor.close.call_count, 2)
        self.assertEqual(mock_conn.close.call_count, 2)

    @patch('utils.db_utils.get_db_connection')
    def test_save_to_raw_table_success(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        data = {
            'Open': [np.float64(100.0), 101.0, pd.NA], 'High': [102.0, 103.0, 104.0],
            'Low': [99.0, 100.0, 100.5], 'Close': [101.5, 102.5, 103.5],
            'Volume': [np.int64(100000), 120000, np.int32(130000)],
            'Dividends': [0.0, np.float32(0.5), 0.0], 'Stock Splits': [0.0, 0.0, 2.0]
        }
        dates = [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02'), pd.Timestamp('2023-01-03')]
        df = pd.DataFrame(data, index=pd.Index(dates, name='date'))
        ticker = 'TESTRAW'; rows_affected = db_utils.save_to_raw_table(ticker, df, SAMPLE_DB_CONFIG)
        self.assertEqual(rows_affected, 3)
        mock_cursor.executemany.assert_called_once()
        args_list = mock_cursor.executemany.call_args[0][1]
        self.assertEqual(args_list[0][0], ticker); self.assertIsInstance(args_list[0][2], float)
        self.assertIsNone(args_list[2][2]); mock_conn.commit.assert_called_once()

    @patch('utils.db_utils.get_db_connection')
    def test_save_to_raw_table_empty_df(self, mock_get_db_connection):
        df = pd.DataFrame()
        rows_affected = db_utils.save_to_raw_table('TESTEMPTY', df, SAMPLE_DB_CONFIG)
        self.assertEqual(rows_affected, 0)
        mock_get_db_connection.return_value.cursor.return_value.executemany.assert_not_called()

    @patch('utils.db_utils.get_db_connection')
    def test_save_to_raw_table_db_error(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        mock_cursor.executemany.side_effect = psycopg2.Error("DB write error")
        data = {'Open': [100.0], 'High': [101.0], 'Low': [99.0], 'Close': [100.5],
                'Volume': [10000.0], 'Dividends': [0.0], 'Stock Splits': [0.0]}
        df = pd.DataFrame(data, index=[pd.Timestamp('2023-01-01')])
        with self.assertRaises(psycopg2.Error):
            db_utils.save_to_raw_table('TESTFAIL', df, SAMPLE_DB_CONFIG)
        mock_conn.rollback.assert_called_once()

    @patch('utils.db_utils.get_db_connection')
    def test_check_ticker_exists(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (1,); self.assertTrue(db_utils.check_ticker_exists('AAPLCHK', SAMPLE_DB_CONFIG))
        mock_cursor.fetchone.return_value = (0,); self.assertFalse(db_utils.check_ticker_exists('MSFTCHK', SAMPLE_DB_CONFIG))

    @patch('utils.db_utils.pd.read_sql_query')
    @patch('utils.db_utils.get_db_connection')
    def test_load_data_from_db(self, mock_get_db_connection, mock_read_sql):
        mock_conn = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        
        df_data_aapload_sql = {
            'date': [datetime(2023,1,1), datetime(2023,1,2)], 'open': [100.0, 101.0],
            'high': [102.0, 103.0], 'low': [99.0, 100.0], 'close': [101.5, 102.5],
            'volume': [100000.0, 120000.0], 'dividends': [0.0, 0.0], 'stock_splits': [0.0, 0.0]
        }
        mock_df_aapload_sql = pd.DataFrame(df_data_aapload_sql)

        df_data_msftload_sql = { 
            'date': [datetime(2023,1,3), datetime(2023,1,4)], 'open': [200.0, 201.0],
            'high': [202.0, 203.0], 'low': [199.0, 200.0], 'close': [201.5, 202.5],
            'volume': [200000.0, 220000.0], 'dividends': [0.1, 0.1], 'stock_splits': [0.0, 0.0]
        }
        mock_df_msftload_sql = pd.DataFrame(df_data_msftload_sql)

        # CORRECTED side_effect function signature
        def read_sql_side_effect_for_load_data(sql, con, params=None, parse_dates=None, **kwargs):
            # 'sql' is the query string
            # 'con' is the connection object
            # 'params' is the tuple, e.g., ('AAPLLOAD',)
            # 'parse_dates' is the list, e.g., ['date']
            # '**kwargs' collects any other keyword arguments
            
            ticker_param = params[0] # The ticker 't'
            if ticker_param == 'AAPLLOAD':
                return mock_df_aapload_sql.copy()
            elif ticker_param == 'MSFTLOAD':
                return mock_df_msftload_sql.copy()
            else:
                return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits'])

        mock_read_sql.side_effect = read_sql_side_effect_for_load_data

        tickers = ['AAPLLOAD', 'MSFTLOAD']
        result_dict = db_utils.load_data_from_db(SAMPLE_DB_CONFIG, tickers)

        self.assertEqual(mock_read_sql.call_count, 2)
        self.assertIn('AAPLLOAD', result_dict); self.assertIn('MSFTLOAD', result_dict)
        
        expected_df_aapload_processed = mock_df_aapload_sql.set_index('date')
        expected_df_aapload_processed.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        pd.testing.assert_frame_equal(result_dict['AAPLLOAD'], expected_df_aapload_processed)

        expected_df_msftload_processed = mock_df_msftload_sql.set_index('date')
        expected_df_msftload_processed.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        pd.testing.assert_frame_equal(result_dict['MSFTLOAD'], expected_df_msftload_processed)
        
        mock_conn.close.assert_called_once()

    @patch('utils.db_utils.pd.read_sql_query')
    @patch('utils.db_utils.get_db_connection')
    def test_load_data_from_db_empty_for_one_ticker(self, mock_get_db_connection, mock_read_sql):
        mock_conn = MagicMock()
        mock_get_db_connection.return_value = mock_conn
        aapl_data_from_sql = {
            'date': [datetime(2023,1,1)], 'open': [149.0], 'high': [151.0], 'low': [148.0],
            'close': [150.0], 'volume': [1e6], 'dividends': [0.0], 'stock_splits': [0.0]
        }
        aapl_df_from_sql = pd.DataFrame(aapl_data_from_sql)
        msft_df_empty_from_sql = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits'])
        
        # CORRECTED side_effect function signature
        def side_effect_read_sql(sql, con, params=None, parse_dates=None, **kwargs):
            ticker_param = params[0] 
            if ticker_param == 'AAPLSIDE': return aapl_df_from_sql.copy()
            elif ticker_param == 'MSFTSIDE': return msft_df_empty_from_sql.copy()
            return pd.DataFrame(columns=msft_df_empty_from_sql.columns)
        mock_read_sql.side_effect = side_effect_read_sql
        
        tickers = ['AAPLSIDE', 'MSFTSIDE']
        result_dict = db_utils.load_data_from_db(SAMPLE_DB_CONFIG, tickers)
        
        self.assertIn('AAPLSIDE', result_dict); self.assertNotIn('MSFTSIDE', result_dict)
        self.assertEqual(len(result_dict['AAPLSIDE']), 1)
        expected_aapl_processed = aapl_df_from_sql.set_index('date')
        expected_aapl_processed.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        pd.testing.assert_frame_equal(result_dict['AAPLSIDE'], expected_aapl_processed)

    @patch('utils.db_utils.get_db_connection')
    def test_save_and_load_processed_features(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        run_id = "test_run_proc_123"; data_np = np.array([[[1.1, 2.2], [3.3, 4.4]]])
        targets_np = np.array([[5.5, 6.6]]); features = ['feat1_proc', 'feat2_proc']
        tickers_list = ['T1PROC', 'T2PROC']
        db_utils.save_processed_features_to_db(SAMPLE_DB_CONFIG, data_np, targets_np, features, tickers_list, run_id)
        args_tuple = mock_cursor.execute.call_args[0][1]
        np.testing.assert_array_equal(pickle.loads(args_tuple[1]), data_np) 
        np.testing.assert_array_equal(pickle.loads(args_tuple[2]), targets_np)
        mock_cursor.fetchone.return_value = (run_id, pickle.dumps(data_np), pickle.dumps(targets_np), json.dumps(features), json.dumps(tickers_list))
        loaded_data = db_utils.load_processed_features_from_db(SAMPLE_DB_CONFIG, run_id)
        self.assertIsNotNone(loaded_data); np.testing.assert_array_equal(loaded_data['processed_data'], data_np)

    @patch('utils.db_utils.get_db_connection')
    def test_load_processed_features_not_found(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        loaded_data = db_utils.load_processed_features_from_db(SAMPLE_DB_CONFIG, "non_existent_proc_run")
        self.assertIsNone(loaded_data)

    @patch('utils.db_utils.get_db_connection')
    def test_save_and_load_scaled_features(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        run_id = "scaled_run_feat_456"; set_name = "X_train_scaled"; data_np = np.array([[0.111, 0.222]])
        db_utils.save_scaled_features(SAMPLE_DB_CONFIG, run_id, set_name, data_np)
        mock_cursor.fetchone.return_value = (pickle.dumps(data_np),)
        loaded_data = db_utils.load_scaled_features(SAMPLE_DB_CONFIG, run_id, set_name)
        np.testing.assert_array_equal(loaded_data, data_np)

    @patch('utils.db_utils.get_db_connection')
    def test_save_and_load_scalers(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        run_id = "scaler_run_data_789"; simple_scaler_x = SimpleScaler(0.1, 0.9)
        scalers_data = {'scalers_x': [[simple_scaler_x]], 'tickers': ['TICKA_SCL']}
        db_utils.save_scalers(SAMPLE_DB_CONFIG, run_id, scalers_data)
        mock_cursor.fetchone.return_value = (pickle.dumps(scalers_data),)
        loaded_scalers_from_db = db_utils.load_scalers(SAMPLE_DB_CONFIG, run_id)
        self.assertEqual(loaded_scalers_from_db['scalers_x'][0][0], simple_scaler_x)

    @patch('utils.db_utils.get_db_connection')
    def test_save_and_load_optimization_results(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        run_id = "opt_run_res_000"; best_params_data = {'lr_opt': 0.00112}
        db_utils.save_optimization_results(SAMPLE_DB_CONFIG, run_id, best_params_data)
        mock_cursor.fetchone.return_value = (json.dumps(best_params_data),)
        loaded_params = db_utils.load_optimization_results(SAMPLE_DB_CONFIG, run_id)
        self.assertEqual(loaded_params, best_params_data)

    @patch('utils.db_utils.get_db_connection')
    def test_save_prediction(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        ticker = "PREDSAVE"; price = 123.4567; model_id = "model_v1_save"; target_date = "2023-10-26"
        db_utils.save_prediction(SAMPLE_DB_CONFIG, ticker, price, model_id, target_date)
        expected_params = (target_date, ticker, price, model_id)
        mock_cursor.execute.assert_called_once(); call_args = mock_cursor.execute.call_args[0]
        self.assertEqual(call_args[1], expected_params)

    @patch('utils.db_utils.get_db_connection')
    def test_get_latest_prediction_for_all_tickers(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock(spec=psycopg2.extensions.cursor)
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        mock_row1 = {'ticker': 'AAPLALL', 'target_prediction_date': date(2023,10,25), 
                     'predicted_price': Decimal('150.751'), 'model_mlflow_run_id': 'run1all'}
        mock_row2 = {'ticker': 'MSFTALL', 'target_prediction_date': date(2023,10,25), 
                     'predicted_price': Decimal('300.502'), 'model_mlflow_run_id': 'run1all'}
        mock_cursor.fetchall.return_value = [mock_row1, mock_row2]
        result = db_utils.get_latest_prediction_for_all_tickers(SAMPLE_DB_CONFIG)
        
        # Expect Decimal as db_utils.get_latest_prediction_for_all_tickers does not convert (as per constraint)
        expected = [
            {'ticker': 'AAPLALL', 'date': '2023-10-25', 'predicted_price': Decimal('150.751'), 'model_mlflow_run_id': 'run1all'},
            {'ticker': 'MSFTALL', 'date': '2023-10-25', 'predicted_price': Decimal('300.502'), 'model_mlflow_run_id': 'run1all'}
        ]
        self.assertEqual(result, expected)

    @patch('utils.db_utils.get_db_connection')
    def test_get_latest_target_date_prediction_for_ticker(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock(spec=psycopg2.extensions.cursor)
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        
        mock_row_decimal = {'target_prediction_date': date(2023,10,25), 
                            'predicted_price': Decimal('180.2031'), 
                            'model_mlflow_run_id': 'run_xyz_target_decimal'}
        mock_cursor.fetchone.return_value = mock_row_decimal
        result_decimal = db_utils.get_latest_target_date_prediction_for_ticker(SAMPLE_DB_CONFIG, 'XYZTARGETDECIMAL')
        expected_decimal = {'target_prediction_date': '2023-10-25', 
                            'predicted_price': 180.2031, # This function IS converting Decimal to float
                            'model_mlflow_run_id': 'run_xyz_target_decimal'}
        self.assertEqual(result_decimal, expected_decimal)

        mock_row_float = {'target_prediction_date': date(2023,10,26), 
                          'predicted_price': 190.554, 
                          'model_mlflow_run_id': 'run_abc_target_float'}
        mock_cursor.fetchone.return_value = mock_row_float
        result_float = db_utils.get_latest_target_date_prediction_for_ticker(SAMPLE_DB_CONFIG, 'ABCTARGETFLOAT')
        expected_float = {'target_prediction_date': '2023-10-26', 
                          'predicted_price': 190.554, 
                          'model_mlflow_run_id': 'run_abc_target_float'}
        self.assertEqual(result_float, expected_float)

        mock_row_none_price = {'target_prediction_date': date(2023,10,27), 
                               'predicted_price': None, 
                               'model_mlflow_run_id': 'run_def_target_none'}
        mock_cursor.fetchone.return_value = mock_row_none_price
        result_none_price = db_utils.get_latest_target_date_prediction_for_ticker(SAMPLE_DB_CONFIG, 'DEFTARGETNONE')
        expected_none_price = {'target_prediction_date': '2023-10-27', 
                               'predicted_price': None, 
                               'model_mlflow_run_id': 'run_def_target_none'}
        self.assertEqual(result_none_price, expected_none_price)
        
        mock_cursor.fetchone.return_value = None
        result_not_found = db_utils.get_latest_target_date_prediction_for_ticker(SAMPLE_DB_CONFIG, 'NOTFOUNDTICKER')
        self.assertIsNone(result_not_found)

    @patch('utils.db_utils.pd.read_sql_query')
    @patch('utils.db_utils.get_db_connection')
    def test_get_raw_stock_data_for_period(self, mock_get_db_connection, mock_read_sql):
        mock_conn = MagicMock(); mock_get_db_connection.return_value = mock_conn
        sample_data_from_sql = {'date': [datetime(2023,1,2), datetime(2023,1,1)], 'close': [101.1, 100.1]}
        mock_df_from_sql = pd.DataFrame(sample_data_from_sql)
        mock_read_sql.return_value = mock_df_from_sql.copy()
        end_dt = date(2023,1,2); num_d = 2
        result_df = db_utils.get_raw_stock_data_for_period(SAMPLE_DB_CONFIG, 'TESTPERIOD', end_dt, num_d)
        expected_df_data = {'date': pd.to_datetime(['2023-01-01', '2023-01-02']), 'close': [100.1, 101.1]}
        expected_df = pd.DataFrame(expected_df_data)
        pd.testing.assert_frame_equal(result_df, expected_df)

    @patch('utils.db_utils.get_db_connection')
    def test_get_all_distinct_tickers_from_predictions(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [('AAPLDISTX',), ('MSFTDISTX',)]
        result = db_utils.get_all_distinct_tickers_from_predictions(SAMPLE_DB_CONFIG)
        self.assertEqual(result, ['AAPLDISTX', 'MSFTDISTX'])

    @patch('utils.db_utils.get_db_connection')
    def test_save_daily_performance_metrics(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        p_date = "2023-10-25"; ticker_sym = "PERFSAVEX"; model_id = "perf_model_v1_savex"
        metrics = {'actual_price': 100.11, 'predicted_price': 102.22, 'mae': 2.11, 'rmse': 2.11, 'mape': 0.0211, 'direction_accuracy': 1.0}
        db_utils.save_daily_performance_metrics(SAMPLE_DB_CONFIG, p_date, ticker_sym, metrics, model_id)
        expected_params = (p_date, ticker_sym, metrics['actual_price'], metrics['predicted_price'],
                           metrics.get('mae'), metrics.get('rmse'), metrics.get('mape'),
                           metrics['direction_accuracy'], model_id)
        mock_cursor.execute.assert_called_once_with(ANY, expected_params)

    @patch('utils.db_utils.pd.read_sql_query')
    @patch('utils.db_utils.get_db_connection')
    def test_get_recent_performance_metrics(self, mock_get_db_connection, mock_read_sql):
        mock_conn = MagicMock(); mock_get_db_connection.return_value = mock_conn
        sample_perf_data = {'prediction_date': [date(2023,10,24)], 'mape': [0.0512]}
        mock_perf_df = pd.DataFrame(sample_perf_data)
        mock_read_sql.return_value = mock_perf_df.copy()
        days_lookback = 7; expected_end_date = datetime.now().date()
        expected_start_date = expected_end_date - timedelta(days=days_lookback)
        result_df = db_utils.get_recent_performance_metrics(SAMPLE_DB_CONFIG, 'TESTRECENTX', days_lookback)
        args, kwargs = mock_read_sql.call_args
        self.assertEqual(kwargs['params'], ('TESTRECENTX', expected_start_date, expected_end_date))
        pd.testing.assert_frame_equal(result_df, mock_perf_df)

    @patch('utils.db_utils.get_db_connection')
    def test_get_last_data_timestamp_for_ticker(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock()
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        dt_obj = datetime(2023, 10, 25, 16, 0, 0)
        mock_cursor.fetchone.return_value = (dt_obj,); self.assertEqual(db_utils.get_last_data_timestamp_for_ticker(SAMPLE_DB_CONFIG, 'AAPLLASTX'), dt_obj)
        mock_cursor.fetchone.return_value = (None,); self.assertIsNone(db_utils.get_last_data_timestamp_for_ticker(SAMPLE_DB_CONFIG, 'NEWTLASTX'))

    @patch('utils.db_utils.pd.read_sql_query')
    @patch('utils.db_utils.get_db_connection')
    def test_get_latest_raw_data_window(self, mock_get_db_connection, mock_read_sql):
        mock_conn = MagicMock(); mock_get_db_connection.return_value = mock_conn
        raw_db_data_full = {'date': [datetime(2023,1,3), datetime(2023,1,2), datetime(2023,1,1)],
                            'open': [102.1,101.1,100.1], 'high': [103.1,102.1,101.1], 'low': [101.1,100.1,99.1],
                            'close': [102.51,101.51,100.51], 'volume': [12001.0,11001.0,10001.0],
                            'dividends': [0.0,0.0,0.0], 'stock_splits': [0.0,0.0,0.0]}
        full_mock_df_from_sql = pd.DataFrame(raw_db_data_full)
        tickers = ['WINRAWTESTX']; window_size = 2 
        limited_mock_df_from_sql = full_mock_df_from_sql.head(window_size).copy()
        mock_read_sql.return_value = limited_mock_df_from_sql
        result_dict = db_utils.get_latest_raw_data_window(SAMPLE_DB_CONFIG, tickers, window_size)
        df_win = result_dict['WINRAWTESTX']
        expected_data = {'Open': [101.1, 102.1], 'High': [102.1, 103.1], 'Low': [100.1, 101.1],
                         'Close': [101.51, 102.51], 'Volume': [11001.0, 12001.0],
                         'Dividends': [0.0,0.0], 'Stock Splits': [0.0,0.0]}
        expected_dates = pd.to_datetime([datetime(2023,1,2), datetime(2023,1,3)])
        expected_df = pd.DataFrame(expected_data, index=pd.Index(expected_dates, name='date'))
        pd.testing.assert_frame_equal(df_win, expected_df)

    @patch('utils.db_utils.get_db_connection')
    def test_get_prediction_for_date_ticker(self, mock_get_db_connection):
        mock_conn = MagicMock(); mock_cursor = MagicMock(spec=psycopg2.extensions.cursor)
        mock_get_db_connection.return_value = mock_conn; mock_conn.cursor.return_value = mock_cursor
        target_date = "2023-11-01"; ticker_symbol = "XYZPREDDT"
        mock_db_row = {'predicted_price': Decimal('200.505'), 'model_mlflow_run_id': 'model_run_abc_preddt'}
        mock_cursor.fetchone.return_value = mock_db_row
        result = db_utils.get_prediction_for_date_ticker(SAMPLE_DB_CONFIG, target_date, ticker_symbol)
        
        # Expect Decimal as db_utils.get_prediction_for_date_ticker does not convert (as per constraint)
        expected = {'predicted_price': Decimal('200.505'), 'model_mlflow_run_id': 'model_run_abc_preddt'}
        self.assertEqual(result, expected)

        mock_cursor.fetchone.return_value = None
        result_none = db_utils.get_prediction_for_date_ticker(SAMPLE_DB_CONFIG, target_date, "NONEPREDDT")
        self.assertIsNone(result_none)

    @patch('utils.db_utils.pd.read_sql_query')
    @patch('utils.db_utils.get_db_connection')
    def test_get_predictions_for_ticker_in_daterange(self, mock_get_db_connection, mock_read_sql):
        mock_conn = MagicMock(); mock_get_db_connection.return_value = mock_conn
        ticker = "RANGERX"; start_date = "2023-11-01"; end_date = "2023-11-03"
        sample_data = {'target_prediction_date': [datetime(2023,11,1), datetime(2023,11,2)], 'predicted_price': [100.11, 102.22]}
        mock_df_from_sql = pd.DataFrame(sample_data); mock_read_sql.return_value = mock_df_from_sql.copy()
        result_df = db_utils.get_predictions_for_ticker_in_daterange(SAMPLE_DB_CONFIG, ticker, start_date, end_date)
        expected_df = pd.DataFrame({'target_prediction_date': pd.to_datetime([datetime(2023,11,1), datetime(2023,11,2)]),
                                    'predicted_price': [100.11, 102.22]})
        pd.testing.assert_frame_equal(result_df, expected_df)

if __name__ == '__main__':
    unittest.main()