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
    # --- MODIFIED: Increased number of rows to 20 ---
    num_rows = 20
    dates = pd.to_datetime([date(2023, 1, 1) + timedelta(days=i) for i in range(num_rows)])
    data = {
        'Open': [100 + i*0.1 for i in range(num_rows)],
        'High': [105 + i*0.1 for i in range(num_rows)],
        'Low': [99 + i*0.1 for i in range(num_rows)],
        'Close': [101 + i*0.1 for i in range(num_rows)],
        'Volume': [1000 + i*10 for i in range(num_rows)],
        'Dividends': [0] * num_rows,
        'Stock Splits': [0] * num_rows
    }
    return pd.DataFrame(data, index=dates)
    # --- END MODIFICATION ---

@pytest.fixture
def sample_raw_stock_data_with_nans_df():
    # --- MODIFIED: Increased number of rows to 20 and adjusted NaN placement ---
    num_rows = 20
    dates = pd.to_datetime([date(2023, 1, 1) + timedelta(days=i) for i in range(num_rows)])
    data = {
        'Open': [100 + i*0.1 for i in range(num_rows)],
        'High': [105 + i*0.1 for i in range(num_rows)],
        'Low': [99 + i*0.1 for i in range(num_rows)],
        'Close': [101 + i*0.1 for i in range(num_rows)],
        'Volume': [1000 + i*10 for i in range(num_rows)],
        'Dividends': [0] * num_rows,
        'Stock Splits': [0] * num_rows
    }
    df = pd.DataFrame(data, index=dates)
    # Introduce NaNs strategically, ensuring not too many at the start for TA calcs
    df.loc[df.index[3], 'Open'] = np.nan
    df.loc[df.index[5], 'High'] = np.nan
    df.loc[df.index[7], 'Low'] = np.nan
    df.loc[df.index[num_rows-1], 'Close'] = np.nan # NaN at the end for Close
    return df
    # --- END MODIFICATION ---


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
        # With 20 rows, the last row should have most indicators calculated
        assert not df_result['RSI'].iloc[-1:].isnull().all()
        assert not df_result['ATR'].iloc[-1:].isnull().all() # Check ATR specifically


class TestPreprocessData:
    def test_creates_target_and_drops_last_row(self, sample_raw_stock_data_df):
        df_copy = sample_raw_stock_data_df.copy()
        original_len = len(df_copy)
        processed_df = make_dataset.preprocess_data(df_copy)

        assert 'Target' in processed_df.columns
        assert len(processed_df) == original_len - 1
        # Target for first row should be Close of second original row
        pd.testing.assert_series_equal(
            processed_df['Target'].reset_index(drop=True), # <-- MODIFIED HERE
            sample_raw_stock_data_df['Close'].iloc[1:original_len].reset_index(drop=True),
            check_dtype=False
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
        data1 = {'Close': [10, 11, 12], 'Volume': [100, 110, 120], 'Target': [11, 12, 13]}
        df1 = pd.DataFrame(data1, index=dates1)

        dates2 = pd.to_datetime([date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)])
        data2 = {'Close': [20, 21, 22], 'Volume': [200, 210, 220], 'Target': [21, 22, 23]}
        df2 = pd.DataFrame(data2, index=dates2)

        ticker_data_processed = { # Simulate output of preprocess_data
            'TICKA': make_dataset.preprocess_data(df1.copy()), # df1 will be length 2 after preprocess
            'TICKB': make_dataset.preprocess_data(df2.copy())  # df2 will be length 2 after preprocess
        }
        # TICKA (processed): index [2023-01-01, 2023-01-02], Target [11, 12]
        # TICKB (processed): index [2023-01-02, 2023-01-03], Target [21, 22]

        # Align and process will use these preprocessed dfs.
        # Common dates after reindex before NaN drop: 2023-01-01, 2023-01-02, 2023-01-03
        # TICKA aligned (features Close, Volume):
        #   2023-01-01: [10, 100], Target: 11
        #   2023-01-02: [11, 110], Target: 12
        #   2023-01-03: [11, 110], Target: 12 (bfill)
        # TICKB aligned (features Close, Volume):
        #   2023-01-01: [20, 200], Target: 21 (ffill)
        #   2023-01-02: [20, 200], Target: 21
        #   2023-01-03: [21, 210], Target: 22

        # The function align_and_process_data itself calls preprocess_data internally.
        # So, we should pass the original dfs and let the function handle preprocessing.
        # The internal preprocess will drop last row.
        # df1 becomes len 2, df2 becomes len 2.
        # TICKA processed: idx 0,1 (2023-01-01, 2023-01-02); Target for idx 0 is 11, for idx 1 is 12
        # TICKB processed: idx 0,1 (2023-01-02, 2023-01-03); Target for idx 0 is 21, for idx 1 is 22

        # Re-aligning the test's expectation of align_and_process_data based on its internal calls:
        # 1. preprocess_data is called for each ticker DF.
        #    df1_p: index [2023-01-01, 2023-01-02], features [C,V], Target [11,12]
        #    df2_p: index [2023-01-02, 2023-01-03], features [C,V], Target [21,22]
        # 2. Common index of df1_p and df2_p: [2023-01-01, 2023-01-02, 2023-01-03]
        # 3. Reindex and ffill/bfill
        #    TICKA_aligned:
        #      01-01: C:10, V:100, T:11
        #      01-02: C:11, V:110, T:12
        #      01-03: C:11, V:110, T:12 (bfill from 01-02)
        #    TICKB_aligned:
        #      01-01: C:20, V:200, T:21 (ffill from 01-02 for C,V; but Target from its own data. T is NaN here for 01-01)
        #             Actually, Target for TICKB on 01-01 would be NaN because df2_p doesn't have 01-01.
        #             Let's trace align_and_process_data:
        #             `ticker_data = {t: preprocess_data(df) for t, df in ticker_data.items()}`
        #             `all_indices = set().union(*[ticker_data[d].index for d in ticker_data])` -> {01-01, 01-02, 01-03}
        #             `aligned_data[t] = ticker_data[t].reindex(index=all_indices).sort_index()`
        #             TICKA_reindexed:
        #               01-01: C:10, V:100, T:11
        #               01-02: C:11, V:110, T:12
        #               01-03: C:NaN,V:NaN, T:NaN
        #             TICKB_reindexed:
        #               01-01: C:NaN,V:NaN, T:NaN
        #               01-02: C:20, V:200, T:21
        #               01-03: C:21, V:210, T:22
        #             Then: `df = aligned_data[ticker][feature_columns + ['Target']].ffill().bfill()`
        #             TICKA_ff_bf:
        #               01-01: C:10, V:100, T:11
        #               01-02: C:11, V:110, T:12
        #               01-03: C:11, V:110, T:12
        #             TICKB_ff_bf:
        #               01-01: C:20, V:200, T:21 (ffill from 01-02)
        #               01-02: C:20, V:200, T:21
        #               01-03: C:21, V:210, T:22
        #             NaN mask applied to `processed_data` (from features) and `targets` (from Target).
        #             In this case, after ffill/bfill, no NaNs. So all 3 timesteps remain.
        #             So expected timesteps = 3.

        processed_data_np, targets_np, feature_cols, tickers_list = \
            make_dataset.align_and_process_data({'TICKA': df1, 'TICKB': df2}) # Pass original dfs

        assert isinstance(processed_data_np, np.ndarray)
        assert isinstance(targets_np, np.ndarray)

        assert processed_data_np.shape == (3, 2, 2) # (timesteps, stocks, features)
        assert targets_np.shape == (3, 2)          # (timesteps, stocks)

        assert feature_cols == ['Close', 'Volume'] # Assuming these are common after preprocessing
        assert sorted(tickers_list) == sorted(['TICKA', 'TICKB']) # Order can vary
        assert not np.isnan(processed_data_np).any()
        assert not np.isnan(targets_np).any()


    def test_align_and_process_data_with_nans_before_cleaning(self):
        dates1 = pd.to_datetime([date(2023, 1, 1), date(2023, 1, 2), date(2023,1,3)]) # 3 days for Target creation
        data1 = {'Close': [10, np.nan, 12], 'Volume': [100, 110, 120]} # No Target here
        df1 = pd.DataFrame(data1, index=dates1)

        dates2 = pd.to_datetime([date(2023, 1, 1), date(2023, 1, 2), date(2023,1,3)])
        data2 = {'Close': [20, 21, 22], 'Volume': [np.nan, 210, 220]}
        df2 = pd.DataFrame(data2, index=dates2)
        ticker_data_raw = {'TICKA': df1, 'TICKB': df2}

        # Let align_and_process_data handle the preprocessing.
        # After preprocess_data:
        # df1_p: index [01-01, 01-02], Close [10, NaN], Volume [100,110], Target [NaN, 12] -> ffill/bfill -> Close [10,10], Vol [100,110], Tgt [12,12]
        #        -> drops NaN from original target means df1_p has 01-02 only: C:NaN, V:110, T:12 -> ffill/bfill -> C:12, V:110, T:12
        #        Let's re-trace preprocess_data:
        #        df1: C:[10,NaN,12], V:[100,110,120]
        #        ffill.bfill -> C:[10,10,12], V:[100,110,120]
        #        Target = Close.shift(-1) -> T:[10,12,NaN]
        #        dropna -> df1_p: index [01-01, 01-02], C:[10,10], V:[100,110], T:[10,12]
        # df2_p: index [01-01, 01-02], Close [20, 21], Volume [NaN,210], Target [21, 22] -> ffill/bfill -> Vol [210,210]
        #        df2: C:[20,21,22], V:[NaN,210,220]
        #        ffill.bfill -> C:[20,21,22], V:[210,210,220]
        #        Target = Close.shift(-1) -> T:[21,22,NaN]
        #        dropna -> df2_p: index [01-01, 01-02], C:[20,21], V:[210,210], T:[21,22]
        # Common indices: [01-01, 01-02]. Both have 2 rows.
        # After reindex and ffill/bfill in align_and_process_data, no NaNs should remain.
        # Shape should be (2, 2, 2 features)

        processed_data_np, targets_np, _, _ = \
            make_dataset.align_and_process_data(ticker_data_raw)

        assert not np.isnan(processed_data_np).any()
        assert not np.isnan(targets_np).any()
        assert processed_data_np.shape == (2, 2, 2) # (timesteps, stocks, features)
        assert targets_np.shape == (2, 2)          # (timesteps, stocks)


class TestRunProcessing:
    @patch('data.make_dataset.setup_database')
    @patch('data.make_dataset.load_data')
    # Removed mock for builtins.open for yaml, will patch yaml.safe_load directly
    def test_run_processing_incremental_fetch_mode(self, mock_load_data, mock_setup_db,
                                                   mock_params_config, mock_db_config, mock_data_loading_params,
                                                   tmp_path):
        config_file = tmp_path / "params.yaml"
        with open(config_file, 'w') as f: # Create the dummy file
            yaml.dump(mock_params_config, f)

        with patch('yaml.safe_load', return_value=mock_params_config) as mock_yaml_load:
            # Pass the string path to run_processing
            result = make_dataset.run_processing(str(config_file), mode='incremental_fetch')

        # Assert that yaml.safe_load was called with a file object (which is a MagicMock in testing)
        # The actual file object is created internally by `with open(config_path, 'r') as f:`, so we check if safe_load was called.
        mock_yaml_load.assert_called_once()
        assert isinstance(mock_yaml_load.call_args[0][0], MagicMock) # Check it was called with a file-like object from the 'open'
        
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
        mock_load_from_db.return_value = {'TICKA': sample_raw_stock_data_df.copy()} # Now sample_raw_stock_data_df has 20 rows
        
        # Simulate add_technical_indicators modifying the df
        def add_ta_side_effect(data_dict):
            for ticker, df_in in data_dict.items():
                # Just ensure it doesn't error and returns something like a df
                df_out = df_in.copy()
                df_out['ATR'] = np.random.rand(len(df_in)) # Dummy TA
                data_dict[ticker] = df_out
            return data_dict
        mock_add_ta.side_effect = add_ta_side_effect

        # preprocess_data returns a single df
        # Original sample_raw_stock_data_df has 20 rows. After preprocess, it will have 19.
        preprocessed_df_sample = sample_raw_stock_data_df.iloc[:-1].copy()
        preprocessed_df_sample['Target'] = sample_raw_stock_data_df['Close'].iloc[1:].values
        mock_preprocess.return_value = preprocessed_df_sample

        # align_and_process_data returns multiple values
        # Based on 19 rows from preprocess, 1 stock, e.g., 5 features
        num_timesteps_after_align = 19 # Assuming no data loss in alignment for single stock
        num_features_align = 5 # Example
        mock_align.return_value = (
            np.random.rand(num_timesteps_after_align, 1, num_features_align),
            np.random.rand(num_timesteps_after_align, 1),
            [f'feat{i}' for i in range(num_features_align)],
            ['TICKA']
        )

        fixed_now = datetime(2023, 1, 10, 12, 0, 0)
        mock_datetime.now.return_value = fixed_now
        expected_run_id = fixed_now.strftime('%Y%m%d_%H%M%S')
        # save_processed_features_to_db is called with current_run_id, doesn't return it in this mock setup
        # The function run_processing returns current_run_id

        with patch('yaml.safe_load', return_value=mock_params_config):
            result_run_id = make_dataset.run_processing(str(config_file), mode='full_process')

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
    def test_run_processing_full_process_no_raw_data(self, mock_load_from_db, mock_initial_load_data,
                                                     mock_setup_db, mock_params_config, tmp_path, caplog):
        config_file = tmp_path / "params.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config, f)

        mock_load_from_db.return_value = {} # No data loaded

        with patch('yaml.safe_load', return_value=mock_params_config):
            result = make_dataset.run_processing(str(config_file), mode='full_process')

        assert result is None
        assert "No raw data found in the database for any ticker after load_data" in caplog.text # Adjusted log message

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
        mock_add_ta.side_effect = lambda x: x # Pass through
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
        mock_add_ta.side_effect = lambda x: x # Pass through
        preprocessed_df_sample = sample_raw_stock_data_df.iloc[:-1].copy()
        preprocessed_df_sample['Target'] = sample_raw_stock_data_df['Close'].iloc[1:].values
        mock_preprocess.return_value = preprocessed_df_sample
        mock_align.return_value = (None, None, None, None) # Alignment fails

        with patch('yaml.safe_load', return_value=mock_params_config):
            result = make_dataset.run_processing(str(config_file), mode='full_process')

        assert result is None
        assert "Failed to align and process data into numpy arrays" in caplog.text