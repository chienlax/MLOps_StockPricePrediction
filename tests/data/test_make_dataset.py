# tests/data/test_make_dataset.py
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import yaml
import io

PROJECT_ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT_FOR_TESTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data import make_dataset # Import the module to be tested

# --- Fixtures ---

@pytest.fixture
def mock_db_config():
    return {
        'dbname': 'test_db',
        'user': 'test_user',
        'password': 'test_password',
        'host': 'localhost',
        'port': '5432'
    }

@pytest.fixture
def mock_data_loading_params():
    return {
        'tickers': ['TICKA', 'TICKB'],
        'period': '1y',
        'interval': '1d',
        'fetch_delay': 0
    }

@pytest.fixture
def mock_feature_eng_params():
    return {
        'correlation_threshold': 0.95
    }

@pytest.fixture
def mock_params_config(mock_db_config, mock_data_loading_params, mock_feature_eng_params):
    return {
        'database': mock_db_config,
        'data_loading': mock_data_loading_params,
        'feature_engineering': mock_feature_eng_params
    }

@pytest.fixture
def sample_raw_stock_data_df():
    num_rows = 250
    dates = pd.to_datetime([date(2023, 1, 1) + timedelta(days=i) for i in range(num_rows)])
    data = {
        'Open': [100 + i*0.1 for i in range(num_rows)],
        'High': [105 + i*0.1 for i in range(num_rows)],
        'Low': [99 + i*0.1 for i in range(num_rows)],
        'Close': [101 + i*0.1 for i in range(num_rows)],
        'Volume': [1000 + i*10 for i in range(num_rows)],
        'Dividends': [0] * num_rows, # Ensure these columns exist
        'Stock Splits': [0] * num_rows # Ensure these columns exist
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_raw_stock_data_with_nans_df():
    num_rows = 250
    dates = pd.to_datetime([date(2023, 1, 1) + timedelta(days=i) for i in range(num_rows)])
    data = {
        'Open': [100 + i*0.1 for i in range(num_rows)],
        'High': [105 + i*0.1 for i in range(num_rows)],
        'Low': [99 + i*0.1 for i in range(num_rows)],
        'Close': [101 + i*0.1 for i in range(num_rows)],
        'Volume': [1000 + i*10 for i in range(num_rows)],
        'Dividends': [0] * num_rows, # Ensure these columns exist
        'Stock Splits': [0] * num_rows # Ensure these columns exist
    }
    df = pd.DataFrame(data, index=dates)
    # Introduce NaNs strategically, ensuring not too many at the start for TA calcs
    df.loc[df.index[int(num_rows*0.1)], 'Open'] = np.nan # Add a NaN after some initial rows
    df.loc[df.index[int(num_rows*0.2)], 'High'] = np.nan
    df.loc[df.index[int(num_rows*0.3)], 'Low'] = np.nan
    df.loc[df.index[num_rows-1], 'Close'] = np.nan # NaN at the very end for Close
    return df


# --- Helper to mock yfinance ---
def get_mock_yfinance_ticker(data_df_to_return, raise_exception=None):
    mock_ticker_instance = MagicMock()
    if raise_exception:
        mock_ticker_instance.history.side_effect = raise_exception
    else:
        mock_ticker_instance.history.return_value = data_df_to_return
    return MagicMock(return_value=mock_ticker_instance)

# --- Tests for individual functions ---

class TestLoadData:
    @patch('data.make_dataset.yf.Ticker')
    @patch('data.make_dataset.get_last_data_timestamp_for_ticker')
    @patch('data.make_dataset.save_to_raw_table')
    @patch('data.make_dataset.time.sleep') # Mock time.sleep
    def test_load_data_new_ticker(self, mock_sleep, mock_save_raw, mock_get_last_ts, mock_yf_ticker,
                                  mock_db_config, mock_data_loading_params, sample_raw_stock_data_df):
        mock_get_last_ts.return_value = None
        mock_yf_ticker_class = get_mock_yfinance_ticker(sample_raw_stock_data_df)
        mock_yf_ticker.side_effect = mock_yf_ticker_class # yf.Ticker('TICKA') will use this

        make_dataset.load_data(['TICKA'], mock_data_loading_params['period'],
                               mock_data_loading_params['interval'],
                               mock_data_loading_params['fetch_delay'], mock_db_config)

        mock_get_last_ts.assert_called_once_with(mock_db_config, 'TICKA')
        mock_yf_ticker_class.return_value.history.assert_called_once_with(
            period=mock_data_loading_params['period'],
            start=None,
            interval=mock_data_loading_params['interval']
        )
        mock_save_raw.assert_called_once()
        args, _ = mock_save_raw.call_args
        assert args[0] == 'TICKA'
        pd.testing.assert_frame_equal(args[1], sample_raw_stock_data_df)
        assert args[2] == mock_db_config
        mock_sleep.assert_not_called() # fetch_delay is 0

    @patch('data.make_dataset.yf.Ticker')
    @patch('data.make_dataset.get_last_data_timestamp_for_ticker')
    @patch('data.make_dataset.save_to_raw_table')
    def test_load_data_incremental_fetch(self, mock_save_raw, mock_get_last_ts, mock_yf_ticker,
                                         mock_db_config, mock_data_loading_params, sample_raw_stock_data_df):
        last_known_date = datetime(2023, 1, 5, 0, 0) # pd.Timestamp might be more accurate
        mock_get_last_ts.return_value = last_known_date

        mock_yf_ticker_class = get_mock_yfinance_ticker(sample_raw_stock_data_df)
        mock_yf_ticker.side_effect = mock_yf_ticker_class

        expected_start_date_str = (last_known_date + timedelta(days=1)).strftime('%Y-%m-%d')

        make_dataset.load_data(['TICKA'], mock_data_loading_params['period'],
                               mock_data_loading_params['interval'],
                               mock_data_loading_params['fetch_delay'], mock_db_config)

        mock_get_last_ts.assert_called_once_with(mock_db_config, 'TICKA')
        mock_yf_ticker_class.return_value.history.assert_called_once_with(
            period=None, # Period is ignored if start is provided
            start=expected_start_date_str,
            interval=mock_data_loading_params['interval']
        )
        mock_save_raw.assert_called_once()

    @patch('data.make_dataset.yf.Ticker')
    @patch('data.make_dataset.get_last_data_timestamp_for_ticker')
    @patch('data.make_dataset.save_to_raw_table')
    def test_load_data_incremental_fetch_no_new_data(self, mock_save_raw, mock_get_last_ts, mock_yf_ticker,
                                                     mock_db_config, mock_data_loading_params):
        last_known_date = datetime(2023, 1, 5, 0, 0)
        mock_get_last_ts.return_value = last_known_date

        empty_df = pd.DataFrame()
        mock_yf_ticker_class = get_mock_yfinance_ticker(empty_df)
        mock_yf_ticker.side_effect = mock_yf_ticker_class

        make_dataset.load_data(['TICKA'], mock_data_loading_params['period'],
                               mock_data_loading_params['interval'],
                               mock_data_loading_params['fetch_delay'], mock_db_config)

        mock_save_raw.assert_not_called() # No data to save

    @patch('data.make_dataset.yf.Ticker')
    @patch('data.make_dataset.get_last_data_timestamp_for_ticker')
    @patch('data.make_dataset.save_to_raw_table')
    def test_load_data_future_start_date(self, mock_save_raw, mock_get_last_ts, mock_yf_ticker,
                                         mock_db_config, mock_data_loading_params, caplog):
        # Make last_known_date yesterday, so start_date is today (assuming test runs before market open or on weekend)
        # Or more robustly, a date far in the future
        last_known_date = datetime.now() + timedelta(days=5) # Ensures start date is in future
        mock_get_last_ts.return_value = last_known_date

        mock_yf_ticker_class = get_mock_yfinance_ticker(pd.DataFrame()) # Should not be called
        mock_yf_ticker.side_effect = mock_yf_ticker_class

        make_dataset.load_data(['TICKA'], mock_data_loading_params['period'],
                               mock_data_loading_params['interval'],
                               mock_data_loading_params['fetch_delay'], mock_db_config)

        mock_yf_ticker_class.return_value.history.assert_not_called()
        mock_save_raw.assert_not_called()
        assert "is in the future. No data to fetch." in caplog.text

    @patch('data.make_dataset.yf.Ticker')
    @patch('data.make_dataset.get_last_data_timestamp_for_ticker')
    @patch('data.make_dataset.save_to_raw_table')
    def test_load_data_yfinance_exception(self, mock_save_raw, mock_get_last_ts, mock_yf_ticker,
                                          mock_db_config, mock_data_loading_params, caplog):
        mock_get_last_ts.return_value = None # New ticker

        mock_yf_ticker_class = get_mock_yfinance_ticker(None, raise_exception=ValueError("API Error"))
        mock_yf_ticker.side_effect = mock_yf_ticker_class

        make_dataset.load_data(['TICKA'], mock_data_loading_params['period'],
                               mock_data_loading_params['interval'],
                               mock_data_loading_params['fetch_delay'], mock_db_config)

        mock_save_raw.assert_not_called()
        assert "Error processing ticker TICKA: API Error" in caplog.text


class TestAddTechnicalIndicators:
    def test_adds_columns(self, sample_raw_stock_data_df):
        ticker_data = {'TICKA': sample_raw_stock_data_df.copy()} # Use copy
        result_data = make_dataset.add_technical_indicators(ticker_data)
        df_result = result_data['TICKA']

        expected_new_cols = [
            'EMA_50', 'EMA_200', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'ATR', 'OBV',
            'MFI', 'ADX', 'SMA_50', 'VWAP', 'PSAR', 'Stochastic_K', 'Stochastic_D',
            'CCI', 'Williams_R', 'Donchian_High', 'Donchian_Low', 'Keltner_High',
            'Keltner_Low', 'Log_Return', 'Price_Rate_Of_Change', 'Volume_SMA',
            'Chaikin_Money_Flow', 'Force_Index', 'DI_Positive', 'DI_Negative'
        ]
        for col in expected_new_cols:
            assert col in df_result.columns
        # Check if some values are populated (e.g., RSI for later rows)
        # With 250 rows, the last row should have most indicators calculated
        assert not df_result['RSI'].iloc[-1:].isnull().all()
        assert not df_result['ATR'].iloc[-1:].isnull().all()
        assert not df_result['ADX'].iloc[-1:].isnull().all() # Check ADX specifically


class TestPreprocessData:
    def test_creates_target_and_drops_last_row(self, sample_raw_stock_data_df):
        df_copy = sample_raw_stock_data_df.copy()
        original_len = len(df_copy)
        processed_df = make_dataset.preprocess_data(df_copy)

        assert 'Target' in processed_df.columns
        assert len(processed_df) == original_len - 1
        # Target for first row should be Close of second original row
        pd.testing.assert_series_equal(
            processed_df['Target'].reset_index(drop=True),
            sample_raw_stock_data_df['Close'].iloc[1:original_len].reset_index(drop=True),
            check_dtype=False,
            check_names=False # --- MODIFIED: Added check_names=False ---
        )

    def test_handles_nans(self, sample_raw_stock_data_with_nans_df):
        df_copy = sample_raw_stock_data_with_nans_df.copy()
        processed_df = make_dataset.preprocess_data(df_copy)

        assert not processed_df.drop(columns=['Target'], errors='ignore').isnull().values.any()
        assert not processed_df['Target'].isnull().values.any()


class TestAlignAndProcessData:
    def test_correct_shape_and_type(self):
        # Create two simple DataFrames for two tickers
        dates1 = pd.to_datetime([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)])
        # Ensure 'Dividends' and 'Stock Splits' are present as they are in sample_raw_stock_data_df
        data1 = {'Close': [10, 11, 12], 'Volume': [100, 110, 120], 'Dividends': [0, 0, 0], 'Stock Splits': [0, 0, 0]}
        df1 = pd.DataFrame(data1, index=dates1)

        dates2 = pd.to_datetime([date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)])
        # Ensure 'Dividends' and 'Stock Splits' are present
        data2 = {'Close': [20, 21, 22], 'Volume': [200, 210, 220], 'Dividends': [0, 0, 0], 'Stock Splits': [0, 0, 0]}
        df2 = pd.DataFrame(data2, index=dates2)

        processed_data_np, targets_np, feature_cols, tickers_list = \
            make_dataset.align_and_process_data({'TICKA': df1, 'TICKB': df2}) # Pass original dfs

        assert isinstance(processed_data_np, np.ndarray)
        assert isinstance(targets_np, np.ndarray)

        # Expected timesteps: 3 (2023-01-01, 2023-01-02, 2023-01-03)
        # Expected features: 4 (Close, Volume, Dividends, Stock Splits)
        assert processed_data_np.shape == (3, 2, 4) # (timesteps, stocks, features)
        assert targets_np.shape == (3, 2)          # (timesteps, stocks)

        assert sorted(feature_cols) == sorted(['Close', 'Volume', 'Dividends', 'Stock Splits']) # Sort for consistent comparison
        assert sorted(tickers_list) == sorted(['TICKA', 'TICKB']) # Order can vary
        assert not np.isnan(processed_data_np).any()
        assert not np.isnan(targets_np).any()


    def test_align_and_process_data_with_nans_before_cleaning(self):
        # DF1: Valid data only for 2023-01-01. (Needs 2 days for preprocess to give 1 row).
        dates1_raw = pd.to_datetime([date(2023, 1, 1), date(2023, 1, 2)])
        data1_raw = {'Close': [10, 11], 'Volume': [100, 110], 'Dividends': [0, 0], 'Stock Splits': [0, 0]}
        df1_processed = make_dataset.preprocess_data(pd.DataFrame(data1_raw, index=dates1_raw))
        # df1_processed now has index [2023-01-01], with Target 11.

        # DF2: Valid data only for 2023-01-03. (Needs 2 days for preprocess to give 1 row).
        dates2_raw = pd.to_datetime([date(2023, 1, 3), date(2023, 1, 4)])
        data2_raw = {'Close': [20, 21], 'Volume': [200, 210], 'Dividends': [0, 0], 'Stock Splits': [0, 0]}
        df2_processed = make_dataset.preprocess_data(pd.DataFrame(data2_raw, index=dates2_raw))
        # df2_processed now has index [2023-01-03], with Target 21.

        ticker_data_preprocessed = {'TICKA': df1_processed, 'TICKB': df2_processed}

        # Expected behavior within align_and_process_data:
        # 1. `all_indices` will be `[2023-01-01, 2023-01-03]`
        # 2. Reindexing TICKA to `all_indices` will introduce NaNs for 2023-01-03.
        # 3. Reindexing TICKB to `all_indices` will introduce NaNs for 2023-01-01.
        # 4. The internal `ffill().bfill()` inside the loop for `processed_data` and `targets` will NOT fill
        #    these `NaN`s because entire rows for a given ticker are missing in the `all_indices` range.
        # 5. `nan_mask` will detect `NaN`s for both 2023-01-01 (due to TICKB) and 2023-01-03 (due to TICKA).
        # 6. Both timepoints will be dropped, resulting in empty output arrays.

        processed_data_np, targets_np, feature_cols, tickers_list = \
            make_dataset.align_and_process_data(ticker_data_preprocessed)

        assert isinstance(processed_data_np, np.ndarray)
        assert isinstance(targets_np, np.ndarray)
        
    
        assert processed_data_np.shape == (0, 2, 4) # (timesteps, stocks, features)
        assert targets_np.shape == (0, 2)           # (timesteps, stocks)
        
        # Feature columns and tickers list should still be derived correctly even if data is empty
        assert sorted(feature_cols) == sorted(['Close', 'Volume', 'Dividends', 'Stock Splits'])
        assert sorted(tickers_list) == sorted(['TICKA', 'TICKB'])
        assert processed_data_np.size == 0 # Ensure it's truly empty
        assert targets_np.size == 0


class TestRunProcessing:
    @patch('data.make_dataset.setup_database')
    @patch('data.make_dataset.load_data')
    @patch('builtins.open') 
    @patch('yaml.safe_load')
    def test_run_processing_incremental_fetch_mode(self, mock_yaml_safe_load, mock_open, mock_load_data, mock_setup_db,
                                                   mock_params_config, mock_db_config, mock_data_loading_params,
                                                   tmp_path):
        # Configure mocks for config file reading
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase) # Simulates file object
        mock_yaml_safe_load.return_value = mock_params_config

        config_file_path = str(tmp_path / "params.yaml") # Just a dummy path string
        
        # Pass the string path to run_processing
        result = make_dataset.run_processing(config_file_path, mode='incremental_fetch')

        mock_open.assert_called_once_with(config_file_path, 'r')
        mock_yaml_safe_load.assert_called_once_with(mock_open.return_value.__enter__.return_value)
        
        mock_setup_db.assert_called_once_with(mock_db_config)
        mock_load_data.assert_called_once_with(
            mock_data_loading_params['tickers'],
            mock_data_loading_params['period'],
            mock_data_loading_params['interval'],
            mock_data_loading_params['fetch_delay'],
            mock_db_config
        )
        assert result is None

    @patch('data.make_dataset.setup_database')
    @patch('data.make_dataset.load_data') # For initial raw data update
    @patch('data.make_dataset.load_data_from_db')
    @patch('data.make_dataset.add_technical_indicators')
    @patch('data.make_dataset.preprocess_data')
    @patch('data.make_dataset.align_and_process_data')
    @patch('data.make_dataset.save_processed_features_to_db')
    @patch('data.make_dataset.datetime') # To mock datetime.now() for run_id
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_run_processing_full_process_mode_success(self, mock_yaml_safe_load, mock_open,
                                                      mock_datetime, mock_save_processed, mock_align,
                                                      mock_preprocess, mock_add_ta, mock_load_from_db,
                                                      mock_initial_load_data, mock_setup_db,
                                                      mock_params_config, sample_raw_stock_data_df, tmp_path):
        config_file = tmp_path / "params.yaml" # Still use Path for the dummy name

        # Configure mocks for config file reading
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config

        # Mock return values for chained calls
        mock_load_from_db.return_value = {'TICKA': sample_raw_stock_data_df.copy()}
        
        # Simulate add_technical_indicators modifying the df
        def add_ta_side_effect(data_dict):
            for ticker, df_in in data_dict.items():
                df_out = df_in.copy()
                # Ensure a few TA columns are present after mock_add_ta
                # Based on the TA functions, original 7 cols + 28 new TA cols = 35 columns
                # We need to explicitly add these to the mock_align.return_value feature_cols list as well.
                # Here, we'll just add a few to demonstrate, as the full list is long.
                for col in ['ATR', 'RSI', 'MACD', 'EMA_50', 'SMA_50']: # Add some common TA columns
                    if col not in df_out.columns:
                        df_out[col] = np.random.rand(len(df_in))
                data_dict[ticker] = df_out
            return data_dict
        mock_add_ta.side_effect = add_ta_side_effect

        # preprocess_data returns a single df
        preprocessed_df_sample = sample_raw_stock_data_df.iloc[:-1].copy()
        preprocessed_df_sample['Target'] = sample_raw_stock_data_df['Close'].iloc[1:].values
        mock_preprocess.return_value = preprocessed_df_sample

        # align_and_process_data returns multiple values
        num_timesteps_after_align = len(preprocessed_df_sample) # Assuming single ticker, no alignment loss
        # The number of features should match the expected output of add_technical_indicators
        # which is 7 original + 28 new = 35. Let's use this actual number.
        expected_feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', # Original
            'EMA_50', 'EMA_200', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'ATR', 'OBV',
            'MFI', 'ADX', 'SMA_50', 'VWAP', 'PSAR', 'Stochastic_K', 'Stochastic_D',
            'CCI', 'Williams_R', 'Donchian_High', 'Donchian_Low', 'Keltner_High',
            'Keltner_Low', 'Log_Return', 'Price_Rate_Of_Change', 'Volume_SMA',
            'Chaikin_Money_Flow', 'Force_Index', 'DI_Positive', 'DI_Negative'
        ]
        num_features_align = len(expected_feature_cols)

        mock_align.return_value = (
            np.random.rand(num_timesteps_after_align, 1, num_features_align),
            np.random.rand(num_timesteps_after_align, 1),
            expected_feature_cols,
            ['TICKA']
        )
        
        fixed_now = datetime(2023, 1, 10, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now
        expected_run_id = fixed_now.strftime('%Y%m%d_%H%M%S')

        # Pass the string path to run_processing
        result_run_id = make_dataset.run_processing(str(config_file), mode='full_process')

        mock_open.assert_called_once_with(str(config_file), 'r') # Assert open was called
        mock_yaml_safe_load.assert_called_once_with(mock_open.return_value.__enter__.return_value) # Assert yaml.safe_load was called

        mock_setup_db.assert_called_once()
        mock_initial_load_data.assert_called_once()
        mock_load_from_db.assert_called_once()
        mock_add_ta.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_align.assert_called_once()
        mock_save_processed.assert_called_once()
        
        assert result_run_id == expected_run_id

    @patch('data.make_dataset.setup_database')
    @patch('data.make_dataset.load_data')
    @patch('data.make_dataset.load_data_from_db')
    @patch('builtins.open') # --- ADDED: Patch builtins.open ---
    @patch('yaml.safe_load') # --- ADDED: Patch yaml.safe_load ---
    def test_run_processing_full_process_no_raw_data(self, mock_yaml_safe_load, mock_open, # Args added
                                                     mock_load_from_db, mock_initial_load_data,
                                                     mock_setup_db, mock_params_config, tmp_path, caplog):
        config_file = tmp_path / "params.yaml"

        # Configure mocks for config file reading
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config

        mock_load_from_db.return_value = {} # No data loaded

        result = make_dataset.run_processing(str(config_file), mode='full_process')
        
        assert result is None
        assert "No raw data found in the database for any ticker after load_data" in caplog.text

    @patch('data.make_dataset.setup_database')
    @patch('data.make_dataset.load_data')
    @patch('data.make_dataset.load_data_from_db')
    @patch('data.make_dataset.add_technical_indicators')
    @patch('data.make_dataset.preprocess_data')
    @patch('builtins.open')
    @patch('yaml.safe_load')
    def test_run_processing_full_process_no_data_after_preprocessing(
        self, mock_yaml_safe_load, mock_open, # Args added
        mock_preprocess, mock_add_ta, mock_load_from_db, mock_initial_load_data,
        mock_setup_db, mock_params_config, sample_raw_stock_data_df, tmp_path, caplog
    ):
        config_file = tmp_path / "params.yaml"

        # Configure mocks for config file reading
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config

        mock_load_from_db.return_value = {'TICKA': sample_raw_stock_data_df.copy()}
        mock_add_ta.side_effect = lambda x: x # Pass through
        mock_preprocess.return_value = pd.DataFrame() # Preprocessing results in empty DF

        result = make_dataset.run_processing(str(config_file), mode='full_process')
        
        assert result is None
        assert "No data available after preprocessing all tickers" in caplog.text

    @patch('data.make_dataset.setup_database')
    @patch('data.make_dataset.load_data')
    @patch('data.make_dataset.load_data_from_db')
    @patch('data.make_dataset.add_technical_indicators')
    @patch('data.make_dataset.preprocess_data')
    @patch('data.make_dataset.align_and_process_data')
    @patch('builtins.open') 
    @patch('yaml.safe_load')
    def test_run_processing_full_process_align_failure(
        self, mock_yaml_safe_load, mock_open, # Args added
        mock_align, mock_preprocess, mock_add_ta, mock_load_from_db, mock_initial_load_data,
        mock_setup_db, mock_params_config, sample_raw_stock_data_df, tmp_path, caplog
    ):
        config_file = tmp_path / "params.yaml"

        # Configure mocks for config file reading
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config

        mock_load_from_db.return_value = {'TICKA': sample_raw_stock_data_df.copy()}
        mock_add_ta.side_effect = lambda x: x # Pass through
        preprocessed_df_sample = sample_raw_stock_data_df.iloc[:-1].copy()
        preprocessed_df_sample['Target'] = sample_raw_stock_data_df['Close'].iloc[1:].values
        mock_preprocess.return_value = preprocessed_df_sample
        mock_align.return_value = (None, None, None, None) # Alignment fails

        result = make_dataset.run_processing(str(config_file), mode='full_process')
        
        assert result is None
        assert "Failed to align and process data into numpy arrays" in caplog.text