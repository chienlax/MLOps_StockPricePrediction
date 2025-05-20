# tests/data/test_make_dataset.py
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import yaml

# Add src to sys.path to allow direct import of make_dataset
# This might be handled by your pytest configuration (e.g., conftest.py or pytest.ini)
# For robustness, especially if running tests individually:
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
        'fetch_delay': 0 # No delay for tests
    }

@pytest.fixture
def mock_feature_eng_params():
    return {
        'correlation_threshold': 0.95 # Though not used by make_dataset's core logic
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
    dates = pd.to_datetime([date(2023, 1, 1) + timedelta(days=i) for i in range(5)])
    data = {
        'Open': [100, 101, 102, 103, 104],
        'High': [105, 106, 107, 108, 109],
        'Low': [99, 100, 101, 102, 103],
        'Close': [101, 102, 103, 104, 105],
        'Volume': [1000, 1100, 1200, 1300, 1400],
        'Dividends': [0, 0, 0, 0, 0],
        'Stock Splits': [0, 0, 0, 0, 0]
    }
    return pd.DataFrame(data, index=dates)

@pytest.fixture
def sample_raw_stock_data_with_nans_df():
    dates = pd.to_datetime([date(2023, 1, 1) + timedelta(days=i) for i in range(5)])
    data = {
        'Open': [100, np.nan, 102, 103, 104],
        'High': [105, 106, np.nan, 108, 109],
        'Low': [99, 100, 101, np.nan, 103],
        'Close': [101, 102, 103, 104, np.nan], # NaN at the end for Close
        'Volume': [1000, 1100, 1200, 1300, 1400],
        'Dividends': [0, 0, 0, 0, 0],
        'Stock Splits': [0, 0, 0, 0, 0]
    }
    return pd.DataFrame(data, index=dates)


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
        # This depends on window sizes, so pick a row that should have a value
        assert not df_result['RSI'].iloc[-1:].isnull().all() # Last row's RSI should not be NaN if enough data


class TestPreprocessData:
    def test_creates_target_and_drops_last_row(self, sample_raw_stock_data_df):
        df_copy = sample_raw_stock_data_df.copy()
        original_len = len(df_copy)
        processed_df = make_dataset.preprocess_data(df_copy)

        assert 'Target' in processed_df.columns
        assert len(processed_df) == original_len - 1
        # Target for first row should be Close of second original row
        pd.testing.assert_series_equal(
            processed_df['Target'],
            sample_raw_stock_data_df['Close'].iloc[1:].reset_index(drop=True),
            check_dtype=False # Allow float vs int if applicable
        )

    def test_handles_nans(self, sample_raw_stock_data_with_nans_df):
        df_copy = sample_raw_stock_data_with_nans_df.copy()
        processed_df = make_dataset.preprocess_data(df_copy)
        
        # After ffill, bfill, and dropping last row (for Target NaN), no NaNs should remain in features
        # except potentially in 'Target' if the original 'Close' had NaNs that affected shifted values.
        # The preprocess_data itself drops rows where Target is NaN.
        assert not processed_df.drop(columns=['Target'], errors='ignore').isnull().values.any()
        assert not processed_df['Target'].isnull().values.any() # Target column itself should also be clean


class TestAlignAndProcessData:
    def test_correct_shape_and_type(self):
        # Create two simple DataFrames for two tickers
        dates1 = pd.to_datetime([date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)])
        data1 = {'Close': [10, 11, 12], 'Volume': [100, 110, 120], 'Target': [11, 12, 13]}
        df1 = pd.DataFrame(data1, index=dates1)

        dates2 = pd.to_datetime([date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)])
        data2 = {'Close': [20, 21, 22], 'Volume': [200, 210, 220], 'Target': [21, 22, 23]}
        df2 = pd.DataFrame(data2, index=dates2)

        ticker_data = {'TICKA': df1, 'TICKB': df2}
        
        # Expected common index: 2023-01-02, 2023-01-03 (where both have non-NaN targets after alignment)
        # df1 target for 2023-01-03 is 13. df2 target for 2023-01-03 is 22.
        # After alignment and ffill/bfill, and then NaN removal based on target:
        # Common dates with valid targets for both would be 2023-01-02, 2023-01-03
        # Expected number of timesteps = 2 (2023-01-02, 2023-01-03)
        # Expected num_stocks = 2
        # Expected num_features = 2 (Close, Volume)

        processed_data_np, targets_np, feature_cols, tickers_list = \
            make_dataset.align_and_process_data(ticker_data)

        assert isinstance(processed_data_np, np.ndarray)
        assert isinstance(targets_np, np.ndarray)
        
        # Check shapes based on the logic of align_and_process_data
        # It aligns, then ffill/bfill, then removes rows where any target or feature is NaN.
        # For df1, index [0,1,2], target [11,12,13]
        # For df2, index [1,2,3], target [21,22,23]
        # Common index after reindex: [0,1,2,3]
        # TICKA aligned: Close [10,11,12,12], Volume [100,110,120,120], Target [11,12,13,13] (bfill for last)
        # TICKB aligned: Close [20,20,21,22], Volume [200,200,210,220], Target [21,21,22,23] (ffill for first)
        # After nan_mask (no nans in this example after ffill/bfill)
        # Expected timesteps = 4
        assert processed_data_np.shape == (4, 2, 2) # (timesteps, stocks, features)
        assert targets_np.shape == (4, 2)          # (timesteps, stocks)
        
        assert feature_cols == ['Close', 'Volume']
        assert tickers_list == ['TICKA', 'TICKB']
        assert not np.isnan(processed_data_np).any()
        assert not np.isnan(targets_np).any()

    def test_align_and_process_data_with_nans_before_cleaning(self):
        dates1 = pd.to_datetime([date(2023, 1, 1), date(2023, 1, 2)])
        data1 = {'Close': [10, np.nan], 'Volume': [100, 110], 'Target': [np.nan, 12]} # NaN in feature and target
        df1 = pd.DataFrame(data1, index=dates1)

        dates2 = pd.to_datetime([date(2023, 1, 1), date(2023, 1, 2)])
        data2 = {'Close': [20, 21], 'Volume': [np.nan, 210], 'Target': [21, np.nan]} # NaN in feature and target
        df2 = pd.DataFrame(data2, index=dates2)
        ticker_data = {'TICKA': df1, 'TICKB': df2}

        # Timestep 2023-01-01: TICKA has NaN Target, TICKB has NaN Volume.
        # Timestep 2023-01-02: TICKA has NaN Close, TICKB has NaN Target.
        # The nan_mask `np.isnan(processed_data).any(axis=(1, 2)) | np.isnan(targets).any(axis=1)`
        # will remove any timestep where *any* stock has a NaN feature or *any* stock has a NaN target.
        # After ffill/bfill within align_and_process_data (before final NaN mask):
        # TICKA: Close [10, 10], Volume [100, 110], Target [12, 12] (ffill/bfill applied to df, then to aligned_data[ticker])
        # TICKB: Close [20, 21], Volume [210, 210], Target [21, 21]
        # So, after ffill/bfill, there should be no NaNs.
        processed_data_np, targets_np, _, _ = \
            make_dataset.align_and_process_data(ticker_data)
        
        assert not np.isnan(processed_data_np).any()
        assert not np.isnan(targets_np).any()
        assert processed_data_np.shape[0] > 0 # Should have some data left


class TestRunProcessing:
    @patch('data.make_dataset.setup_database')
    @patch('data.make_dataset.load_data')
    @patch('builtins.open', new_callable=yaml.safe_load) # Mock open for reading params
    def test_run_processing_incremental_fetch_mode(self, mock_open_yaml, mock_load_data, mock_setup_db,
                                                   mock_params_config, mock_db_config, mock_data_loading_params,
                                                   tmp_path):
        # Create a dummy config file for the mock_open to "read"
        config_file = tmp_path / "params.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config, f)
        
        # Configure mock_open_yaml to simulate reading this file
        # This is tricky because open is called with 'r', then yaml.safe_load is called on the file object.
        # A simpler way is to patch yaml.safe_load directly if open is too complex.
        with patch('yaml.safe_load', return_value=mock_params_config) as mock_yaml_load:
            result = make_dataset.run_processing(str(config_file), mode='incremental_fetch')

        mock_yaml_load.assert_called_once() # Check that config was loaded
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
    def test_run_processing_full_process_mode_success(self, mock_datetime, mock_save_processed, mock_align,
                                                      mock_preprocess, mock_add_ta, mock_load_from_db,
                                                      mock_initial_load_data, mock_setup_db,
                                                      mock_params_config, sample_raw_stock_data_df, tmp_path):
        config_file = tmp_path / "params.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config, f)

        # Mock return values for chained calls
        mock_load_from_db.return_value = {'TICKA': sample_raw_stock_data_df.copy()}
        mock_add_ta.side_effect = lambda x: x # Assume it modifies in place or returns modified
        
        # preprocess_data returns a single df, so we need to mock its behavior if called in a loop
        preprocessed_df_sample = sample_raw_stock_data_df.iloc[:-1].copy() # Simulate target drop
        preprocessed_df_sample['Target'] = sample_raw_stock_data_df['Close'].iloc[1:].values
        mock_preprocess.return_value = preprocessed_df_sample

        # align_and_process_data returns multiple values
        mock_align.return_value = (
            np.random.rand(10, 1, 5), # processed_data_np (timesteps, stocks, features)
            np.random.rand(10, 1),    # targets_np (timesteps, stocks)
            ['feat1', 'feat2'],       # feature_columns_list
            ['TICKA']                 # final_tickers_list
        )
        
        fixed_now = datetime(2023, 1, 10, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now
        expected_run_id = fixed_now.strftime('%Y%m%d_%H%M%S')
        mock_save_processed.return_value = expected_run_id # save_processed_features_to_db returns run_id

        with patch('yaml.safe_load', return_value=mock_params_config):
            result_run_id = make_dataset.run_processing(str(config_file), mode='full_process')

        mock_setup_db.assert_called_once()
        mock_initial_load_data.assert_called_once() # For ensuring raw data is up-to-date
        mock_load_from_db.assert_called_once()
        mock_add_ta.assert_called_once()
        mock_preprocess.assert_called_once() # Called for 'TICKA'
        mock_align.assert_called_once()
        mock_save_processed.assert_called_once()
        
        assert result_run_id == expected_run_id

    @patch('data.make_dataset.setup_database')
    @patch('data.make_dataset.load_data')
    @patch('data.make_dataset.load_data_from_db')
    def test_run_processing_full_process_no_raw_data(self, mock_load_from_db, mock_initial_load_data,
                                                     mock_setup_db, mock_params_config, tmp_path, caplog):
        config_file = tmp_path / "params.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config, f)

        mock_load_from_db.return_value = {} # No data loaded

        with patch('yaml.safe_load', return_value=mock_params_config):
            result = make_dataset.run_processing(str(config_file), mode='full_process')
        
        assert result is None
        assert "No raw data found in the database" in caplog.text

    @patch('data.make_dataset.setup_database')
    @patch('data.make_dataset.load_data')
    @patch('data.make_dataset.load_data_from_db')
    @patch('data.make_dataset.add_technical_indicators')
    @patch('data.make_dataset.preprocess_data')
    def test_run_processing_full_process_no_data_after_preprocessing(
        self, mock_preprocess, mock_add_ta, mock_load_from_db, mock_initial_load_data,
        mock_setup_db, mock_params_config, sample_raw_stock_data_df, tmp_path, caplog
    ):
        config_file = tmp_path / "params.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config, f)

        mock_load_from_db.return_value = {'TICKA': sample_raw_stock_data_df.copy()}
        mock_add_ta.side_effect = lambda x: x
        mock_preprocess.return_value = pd.DataFrame() # Preprocessing results in empty DF

        with patch('yaml.safe_load', return_value=mock_params_config):
            result = make_dataset.run_processing(str(config_file), mode='full_process')
        
        assert result is None
        assert "No data available after preprocessing all tickers" in caplog.text

    @patch('data.make_dataset.setup_database')
    @patch('data.make_dataset.load_data')
    @patch('data.make_dataset.load_data_from_db')
    @patch('data.make_dataset.add_technical_indicators')
    @patch('data.make_dataset.preprocess_data')
    @patch('data.make_dataset.align_and_process_data')
    def test_run_processing_full_process_align_failure(
        self, mock_align, mock_preprocess, mock_add_ta, mock_load_from_db, mock_initial_load_data,
        mock_setup_db, mock_params_config, sample_raw_stock_data_df, tmp_path, caplog
    ):
        config_file = tmp_path / "params.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config, f)

        mock_load_from_db.return_value = {'TICKA': sample_raw_stock_data_df.copy()}
        mock_add_ta.side_effect = lambda x: x
        preprocessed_df_sample = sample_raw_stock_data_df.iloc[:-1].copy()
        preprocessed_df_sample['Target'] = sample_raw_stock_data_df['Close'].iloc[1:].values
        mock_preprocess.return_value = preprocessed_df_sample
        mock_align.return_value = (None, None, None, None) # Alignment fails

        with patch('yaml.safe_load', return_value=mock_params_config):
            result = make_dataset.run_processing(str(config_file), mode='full_process')
        
        assert result is None
        assert "Failed to align and process data into numpy arrays" in caplog.text