# tests/features/test_prepare_prediction_input.py
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler # For type hinting and understanding
from datetime import datetime, timedelta, date # For sample data
import io # Needed for mocking file objects

# Add src to sys.path
PROJECT_ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT_FOR_TESTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from features import prepare_prediction_input # Import the module


# --- Fixtures ---

@pytest.fixture
def mock_db_config_ppi(): # Suffix ppi for prepare_prediction_input
    return {
        'dbname': 'test_db_ppi',
        'user': 'test_user_ppi',
        'password': 'test_password',
        'host': 'localhost',
        'port': '5432'
    }

@pytest.fixture
def mock_params_config_ppi(mock_db_config_ppi):
    return {
        'database': mock_db_config_ppi,
        'feature_engineering': {
            'sequence_length': 5, # Example value
        },
        'data_loading': {
            'tickers': ['TICKA', 'TICKB'] # Used to determine expected tickers in output
        },
        'output_paths': {
            'predictions_dir': '/tmp/test_predictions' # Dummy path, will be mocked
        }
    }

@pytest.fixture
def sample_scalers_data_ppi():
    # scalers_x: list of lists of scalers [stock_idx][feature_idx]
    # Assuming 2 stocks, 3 features (Open, High, Close)
    scalers_x = [
        [MagicMock(spec=MinMaxScaler) for _ in range(3)], # Scalers for TICKA's features
        [MagicMock(spec=MinMaxScaler) for _ in range(3)]  # Scalers for TICKB's features
    ]
    # Set return values for transform if needed by the actual logic
    for stock_scalers in scalers_x:
        for scaler in stock_scalers:
            scaler.transform.side_effect = lambda x: x # Identity transform for tests
    return {'scalers_x': scalers_x}

@pytest.fixture
def sample_processed_features_meta_ppi():
    return {
        'feature_columns': ['Open', 'High', 'Close'], # Example features from a processed data run
        'tickers': ['TICKA', 'TICKB'] # Tickers used in that run
    }

@pytest.fixture
def sample_latest_raw_data_dict_ppi():
    num_days = 250 # Increased to ensure enough data for TA and sequence_length if needed
    dates = pd.to_datetime([date(2023, 1, 1) + timedelta(days=i) for i in range(num_days)])
    
    data_template = {
        'Open': np.random.rand(num_days) * 100 + 100,
        'High': np.random.rand(num_days) * 100 + 105,
        'Low': np.random.rand(num_days) * 100 + 95,
        'Close': np.random.rand(num_days) * 100 + 101,
        'Volume': np.random.randint(1000, 3000, num_days),
        'Dividends': np.zeros(num_days),
        'Stock Splits': np.zeros(num_days),
    }

    return {
        'TICKA': pd.DataFrame(data_template, index=dates),
        'TICKB': pd.DataFrame(data_template, index=dates), # Same data for simplicity
    }


# --- Tests for run_prepare_input ---
class TestRunPrepareInput:
    @patch('features.prepare_prediction_input.np.save')
    @patch('features.prepare_prediction_input.Path.mkdir')
    @patch('features.prepare_prediction_input.add_technical_indicators') # Mocking this from make_dataset
    @patch('features.prepare_prediction_input.get_latest_raw_data_window')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.open') # Patching module's open
    @patch('yaml.safe_load')
    def test_run_prepare_input_success(self, mock_yaml_safe_load, mock_open,
                                       mock_load_scalers, mock_load_proc_meta, mock_get_latest_raw,
                                       mock_add_ta, mock_mkdir, mock_np_save,
                                       mock_params_config_ppi, sample_scalers_data_ppi,
                                       sample_processed_features_meta_ppi,
                                       sample_latest_raw_data_dict_ppi, tmp_path):
        prod_model_run_id = "prod_model_train_run_001"
        output_dir = str(tmp_path / "pred_inputs")
        dummy_config_path = str(tmp_path / "dummy_config.yaml")

        # Configure mock_open and mock_yaml_safe_load
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_ppi

        mock_load_scalers.return_value = sample_scalers_data_ppi
        mock_load_proc_meta.return_value = sample_processed_features_meta_ppi
        mock_get_latest_raw.return_value = sample_latest_raw_data_dict_ppi

        # Mock add_technical_indicators to return the input data but with expected feature columns
        def mock_add_ta_side_effect(data_dict_copy):
            output_dict = {}
            for ticker, df_raw in data_dict_copy.items():
                df_with_ta_mock = df_raw.copy()
                # Ensure all columns from feature_columns_prod_model are present
                for col in sample_processed_features_meta_ppi['feature_columns']:
                    if col not in df_with_ta_mock:
                        df_with_ta_mock[col] = np.random.rand(len(df_raw)) # Add dummy data
                output_dict[ticker] = df_with_ta_mock
            return output_dict
        mock_add_ta.side_effect = mock_add_ta_side_effect

        result_path_str = prepare_prediction_input.run_prepare_input(
            dummy_config_path, prod_model_run_id, output_dir
        )

        # Assertions for mocking
        mock_open.assert_called_once_with(dummy_config_path, 'r')
        mock_yaml_safe_load.assert_called_once_with(mock_open.return_value.__enter__.return_value)
        mock_load_scalers.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_load_proc_meta.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_get_latest_raw.assert_called_once() # Args can be checked more deeply if needed
        mock_add_ta.assert_called_once()
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_np_save.assert_called_once()

        assert result_path_str is not None
        # --- MODIFIED: Removed Path.exists() check as np.save is mocked ---
        # assert Path(result_path_str).exists() 
        # Instead, check if the path looks correct:
        assert "prediction_input_sequence" in result_path_str
        assert ".npy" in result_path_str
        assert output_dir in result_path_str

    @patch('features.prepare_prediction_input.get_latest_raw_data_window') 
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.open') # Patching module's open
    @patch('yaml.safe_load')
    def test_run_prepare_input_load_scalers_fails(self, mock_yaml_safe_load, mock_open,
                                                  mock_load_scalers, mock_load_proc_meta, mock_get_latest_raw,
                                                  mock_params_config_ppi, tmp_path, caplog):
        prod_model_run_id = "run_fail_scalers"
        output_dir = str(tmp_path)
        dummy_config_path = str(tmp_path / "dummy_config.yaml")

        # Configure mock_open and mock_yaml_safe_load
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_ppi
        
        # Configure the mocks for this specific test
        mock_load_scalers.return_value = None # This is the condition being tested
        mock_load_proc_meta.return_value = {'feature_columns': ['Open'], 'tickers': ['TICKA']} # Provide valid data
        mock_get_latest_raw.return_value = {} # Doesn't matter, won't be reached if load_scalers fails

        result = prepare_prediction_input.run_prepare_input(
            dummy_config_path, prod_model_run_id, output_dir
        )
        assert result is None
        # Check log message
        assert f"Could not load scalers_x for run_id: {prod_model_run_id}" in caplog.text
        # Ensure only the first DB utility was called before failing
        mock_load_scalers.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        # --- MODIFIED: mock_load_proc_meta should have been called before the check ---
        mock_load_proc_meta.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_get_latest_raw.assert_not_called() # Should not be called


    @patch('features.prepare_prediction_input.get_latest_raw_data_window') 
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.open') # Patching module's open
    @patch('yaml.safe_load')
    def test_run_prepare_input_load_meta_fails(self, mock_yaml_safe_load, mock_open,
                                               mock_load_scalers, mock_load_proc_meta, mock_get_latest_raw,
                                               mock_params_config_ppi, sample_scalers_data_ppi,
                                               tmp_path, caplog):
        prod_model_run_id = "run_fail_meta"
        output_dir = str(tmp_path)
        dummy_config_path = str(tmp_path / "dummy_config.yaml")

        # Configure mock_open and mock_yaml_safe_load
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_ppi

        # Configure the mocks for this specific test
        mock_load_scalers.return_value = sample_scalers_data_ppi # Provide valid data
        mock_load_proc_meta.return_value = None # This is the condition being tested
        mock_get_latest_raw.return_value = {} # Doesn't matter, won't be reached

        result = prepare_prediction_input.run_prepare_input(
            dummy_config_path, prod_model_run_id, output_dir
        )
        assert result is None
        assert f"Could not load feature_columns or tickers for run_id: {prod_model_run_id}" in caplog.text
        # Ensure only the first two DB utilities were called before failing
        mock_load_scalers.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_load_proc_meta.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_get_latest_raw.assert_not_called()


    @patch('features.prepare_prediction_input.get_latest_raw_data_window')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.open') # Patching module's open
    @patch('yaml.safe_load')
    def test_run_prepare_input_scaler_dimension_mismatch(self, mock_yaml_safe_load, mock_open,
                                                         mock_load_scalers, mock_load_proc_meta, mock_get_latest_raw,
                                                         mock_params_config_ppi, tmp_path, caplog):
        prod_model_run_id = "run_fail_dim_mismatch"
        output_dir = str(tmp_path)
        dummy_config_path = str(tmp_path / "dummy_config.yaml")

        # Configure mock_open and mock_yaml_safe_load
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_ppi

        # Scalers for 1 stock, 2 features
        mismatched_scalers = {'scalers_x': [[MagicMock(spec=MinMaxScaler)]*2]}
        # Meta for 2 stocks, 3 features
        mismatched_meta = {'feature_columns': ['f1','f2','f3'], 'tickers': ['T1','T2']}

        # Configure the mocks for this specific test
        mock_load_scalers.return_value = mismatched_scalers
        mock_load_proc_meta.return_value = mismatched_meta
        mock_get_latest_raw.return_value = {} # Doesn't matter, won't be reached

        result = prepare_prediction_input.run_prepare_input(
            dummy_config_path, prod_model_run_id, output_dir
        )
        assert result is None
        assert "Mismatch in dimensions of loaded scalers_x" in caplog.text
        # Assertions to ensure calls up to the point of failure
        mock_load_scalers.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_load_proc_meta.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_get_latest_raw.assert_not_called()


    @patch('features.prepare_prediction_input.add_technical_indicators')
    @patch('features.prepare_prediction_input.get_latest_raw_data_window')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.open') # Patching module's open
    @patch('yaml.safe_load')
    def test_run_prepare_input_get_raw_data_fails(self, mock_yaml_safe_load, mock_open,
                                                  mock_load_scalers, mock_load_proc_meta, mock_get_latest_raw, mock_add_ta,
                                                  mock_params_config_ppi, sample_scalers_data_ppi,
                                                  sample_processed_features_meta_ppi,
                                                  tmp_path, caplog):
        prod_model_run_id = "run_fail_raw"
        output_dir = str(tmp_path)
        dummy_config_path = str(tmp_path / "dummy_config.yaml")

        # Configure mock_open and mock_yaml_safe_load
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_ppi

        # Configure the mocks for this specific test
        mock_load_scalers.return_value = sample_scalers_data_ppi
        mock_load_proc_meta.return_value = sample_processed_features_meta_ppi
        mock_get_latest_raw.return_value = {} # This is the condition being tested (no data fetched)
        mock_add_ta.return_value = {} # Will not be called if get_latest_raw_data_window fails

        result = prepare_prediction_input.run_prepare_input(
            dummy_config_path, prod_model_run_id, output_dir
        )
        assert result is None
        assert "Failed to fetch any latest raw data" in caplog.text
        # Assertions to ensure calls up to the point of failure
        mock_load_scalers.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_load_proc_meta.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_get_latest_raw.assert_called_once()
        mock_add_ta.assert_not_called()


    @patch('features.prepare_prediction_input.add_technical_indicators')
    @patch('features.prepare_prediction_input.get_latest_raw_data_window')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.open') # Patching module's open
    @patch('yaml.safe_load')
    def test_run_prepare_input_missing_ticker_data(self, mock_yaml_safe_load, mock_open,
                                                   mock_load_scalers, mock_load_proc_meta, mock_get_latest_raw, mock_add_ta,
                                                   mock_params_config_ppi, sample_scalers_data_ppi,
                                                   sample_processed_features_meta_ppi,
                                                   sample_latest_raw_data_dict_ppi, tmp_path, caplog):
        prod_model_run_id = "run_fail_missing_ticker"
        output_dir = str(tmp_path)
        dummy_config_path = str(tmp_path / "dummy_config.yaml")

        # Configure mock_open and mock_yaml_safe_load
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_ppi

        mock_load_scalers.return_value = sample_scalers_data_ppi
        mock_load_proc_meta.return_value = sample_processed_features_meta_ppi # Expects TICKA, TICKB

        # Simulate raw data fetched only for TICKA
        raw_data_missing_one_ticker = {'TICKA': sample_latest_raw_data_dict_ppi['TICKA']}
        mock_get_latest_raw.return_value = raw_data_missing_one_ticker
        
        # Configure add_ta if it were called
        mock_add_ta.return_value = {} # Default if not called

        result = prepare_prediction_input.run_prepare_input(
            dummy_config_path, prod_model_run_id, output_dir
        )
        assert result is None
        assert "Raw data for ticker TICKB (expected by production model) is missing or empty" in caplog.text
        # Assertions to ensure calls up to the point of failure
        mock_load_scalers.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_load_proc_meta.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_get_latest_raw.assert_called_once()
        # --- MODIFIED: add_technical_indicators should NOT be called ---
        mock_add_ta.assert_not_called()


    @patch('features.prepare_prediction_input.add_technical_indicators')
    @patch('features.prepare_prediction_input.get_latest_raw_data_window')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.open') # Patching module's open
    @patch('yaml.safe_load')
    def test_run_prepare_input_not_enough_timesteps(self, mock_yaml_safe_load, mock_open,
                                                    mock_load_scalers, mock_load_proc_meta, mock_get_latest_raw,
                                                    mock_add_ta,
                                                    mock_params_config_ppi, sample_scalers_data_ppi,
                                                    sample_processed_features_meta_ppi, tmp_path, caplog):
        prod_model_run_id = "run_fail_short_seq"
        output_dir = str(tmp_path)
        dummy_config_path = str(tmp_path / "dummy_config.yaml")

        # Configure mock_open and mock_yaml_safe_load
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_ppi

        # Config sequence_length is 5
        # Provide raw data that results in fewer than 5 timesteps after TA and alignment
        short_raw_data = {}
        # Changed to 250 days to ensure enough for `MAX_TA_WINDOW_REQUIRED` from `run_prepare_input`
        # Then, slice this down for the actual short sequence test
        full_dates = pd.to_datetime([pd.Timestamp('2023-01-01') + pd.Timedelta(days=i) for i in range(250)])
        
        # Now create a short window for the test case itself
        dates_short_for_test = full_dates[-3:] # Only 3 days for sequence test

        for ticker in sample_processed_features_meta_ppi['tickers']:
            short_raw_data[ticker] = pd.DataFrame({
                col: np.random.rand(len(dates_short_for_test)) for col in sample_processed_features_meta_ppi['feature_columns']
            }, index=dates_short_for_test)
            # Add other necessary columns for add_technical_indicators if not in feature_columns
            if 'Close' not in short_raw_data[ticker]: short_raw_data[ticker]['Close'] = np.random.rand(len(dates_short_for_test))
            if 'High' not in short_raw_data[ticker]: short_raw_data[ticker]['High'] = np.random.rand(len(dates_short_for_test))
            if 'Low' not in short_raw_data[ticker]: short_raw_data[ticker]['Low'] = np.random.rand(len(dates_short_for_test))
            if 'Volume' not in short_raw_data[ticker]: short_raw_data[ticker]['Volume'] = np.random.rand(len(dates_short_for_test))
            # Added Dividends and Stock Splits for consistency with data pipeline
            if 'Dividends' not in short_raw_data[ticker]: short_raw_data[ticker]['Dividends'] = np.zeros(len(dates_short_for_test))
            if 'Stock Splits' not in short_raw_data[ticker]: short_raw_data[ticker]['Stock Splits'] = np.zeros(len(dates_short_for_test))


        mock_load_scalers.return_value = sample_scalers_data_ppi
        mock_load_proc_meta.return_value = sample_processed_features_meta_ppi
        mock_get_latest_raw.return_value = short_raw_data
        mock_add_ta.side_effect = lambda x: x # Pass through for this test

        result = prepare_prediction_input.run_prepare_input(
            dummy_config_path, prod_model_run_id, output_dir
        )
        assert result is None
        assert "Not enough timesteps (3) after processing to form a sequence of length 5" in caplog.text