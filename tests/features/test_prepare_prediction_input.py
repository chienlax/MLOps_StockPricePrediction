# tests/features/test_prepare_prediction_input.py
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import MinMaxScaler # For type hinting

# Add src to sys.path
PROJECT_ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT_FOR_TESTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from features import prepare_prediction_input # Import the module
# Also need to mock functions from make_dataset if they are called
# from data import make_dataset # This would be needed if we weren't mocking its functions

# --- Fixtures ---

@pytest.fixture
def mock_db_config_ppi(): # Suffix ppi for prepare_prediction_input
    return {
        'dbname': 'test_db_ppi',
        'user': 'test_user',
        'password': 'test_password',
        'host': 'localhost',
        'port': '5432'
    }

@pytest.fixture
def mock_feature_eng_params_ppi():
    return {'sequence_length': 5}

@pytest.fixture
def mock_params_config_ppi(mock_db_config_ppi, mock_feature_eng_params_ppi):
    return {
        'database': mock_db_config_ppi,
        'feature_engineering': mock_feature_eng_params_ppi,
        'data_loading': {'tickers': ['TICKA', 'TICKB']} # Example, might be derived from loaded features
    }

@pytest.fixture
def sample_scalers_data_ppi():
    # Mock scalers_x: list of lists of scalers
    # For 2 stocks, 3 features
    scalers_x = [
        [MagicMock(spec=MinMaxScaler) for _ in range(3)], # Stock 1
        [MagicMock(spec=MinMaxScaler) for _ in range(3)]  # Stock 2
    ]
    # Mock transform to return identity for simplicity
    for stock_scalers in scalers_x:
        for scaler in stock_scalers:
            scaler.transform.side_effect = lambda x: x # input shape (N,1), output (N,1)
    return {'scalers_x': scalers_x}

@pytest.fixture
def sample_processed_features_meta_ppi():
    return {
        'feature_columns': ['Open', 'High', 'Close'],
        'tickers': ['TICKA', 'TICKB']
    }

@pytest.fixture
def sample_latest_raw_data_dict_ppi():
    # For 2 tickers, sequence_length 5 + buffer for TA (e.g., 200)
    # Let's say we need 20 timesteps of raw data for each
    dates = pd.to_datetime([pd.Timestamp('2023-01-01') + pd.Timedelta(days=i) for i in range(20)])
    df_ticka = pd.DataFrame({
        'Open': np.random.rand(20) + 100,
        'High': np.random.rand(20) + 105,
        'Low': np.random.rand(20) + 95,
        'Close': np.random.rand(20) + 100,
        'Volume': np.random.randint(1000, 2000, 20)
    }, index=dates)
    df_tickb = pd.DataFrame({
        'Open': np.random.rand(20) + 200,
        'High': np.random.rand(20) + 205,
        'Low': np.random.rand(20) + 195,
        'Close': np.random.rand(20) + 200,
        'Volume': np.random.randint(2000, 3000, 20)
    }, index=dates)
    return {'TICKA': df_ticka, 'TICKB': df_tickb}


# --- Tests for run_prepare_input ---

class TestRunPrepareInput:
    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.get_latest_raw_data_window')
    @patch('features.prepare_prediction_input.add_technical_indicators') # Mocking this from make_dataset
    @patch('features.prepare_prediction_input.Path.mkdir')
    @patch('features.prepare_prediction_input.np.save')
    def test_run_prepare_input_success(self, mock_np_save, mock_mkdir, mock_add_ta,
                                       mock_get_latest_raw, mock_load_proc_meta, mock_load_scalers,
                                       mock_params_config_ppi, sample_scalers_data_ppi,
                                       sample_processed_features_meta_ppi,
                                       sample_latest_raw_data_dict_ppi, tmp_path):
        prod_model_run_id = "prod_model_train_run_001"
        output_dir = str(tmp_path / "pred_inputs")

        mock_load_scalers.return_value = sample_scalers_data_ppi
        mock_load_proc_meta.return_value = sample_processed_features_meta_ppi
        mock_get_latest_raw.return_value = sample_latest_raw_data_dict_ppi

        # Mock add_technical_indicators to return the input data but with expected feature columns
        # It should add TA columns, and then the selection `df_filled[feature_columns_prod_model]` happens.
        # So, ensure the mocked df_with_ta has the necessary columns.
        def mock_add_ta_side_effect(data_dict_copy):
            # Simulate adding TA columns and ensuring original columns are present
            # For this test, we just need to ensure the columns expected by `feature_columns_prod_model` exist.
            # The actual TA values don't matter for this unit test's scope.
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
        
        with patch('yaml.safe_load', return_value=mock_params_config_ppi):
            result_path_str = prepare_prediction_input.run_prepare_input(
                "dummy_config.yaml", prod_model_run_id, output_dir
            )

        assert result_path_str is not None
        result_path = Path(result_path_str)
        assert result_path.parent == Path(output_dir).resolve()
        assert result_path.name.startswith("prediction_input_sequence_")
        assert result_path.suffix == ".npy"

        mock_load_scalers.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_load_proc_meta.assert_called_once_with(mock_params_config_ppi['database'], prod_model_run_id)
        mock_get_latest_raw.assert_called_once()
        mock_add_ta.assert_called_once()
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_np_save.assert_called_once()
        
        # Check shape of saved array: (1, seq_len, num_stocks, num_features)
        saved_array_arg = mock_np_save.call_args[0][1]
        seq_len = mock_params_config_ppi['feature_engineering']['sequence_length']
        num_stocks = len(sample_processed_features_meta_ppi['tickers'])
        num_features = len(sample_processed_features_meta_ppi['feature_columns'])
        assert saved_array_arg.shape == (1, seq_len, num_stocks, num_features)

    @patch('features.prepare_prediction_input.load_scalers')
    def test_run_prepare_input_load_scalers_fails(self, mock_load_scalers,
                                                  mock_params_config_ppi, tmp_path, caplog):
        prod_model_run_id = "run_fail_scalers"
        output_dir = str(tmp_path)
        mock_load_scalers.return_value = None

        with patch('yaml.safe_load', return_value=mock_params_config_ppi):
            result = prepare_prediction_input.run_prepare_input(
                "dummy_config.yaml", prod_model_run_id, output_dir
            )
        assert result is None
        assert f"Could not load scalers_x for run_id: {prod_model_run_id}" in caplog.text

    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    def test_run_prepare_input_load_meta_fails(self, mock_load_proc_meta, mock_load_scalers,
                                               mock_params_config_ppi, sample_scalers_data_ppi,
                                               tmp_path, caplog):
        prod_model_run_id = "run_fail_meta"
        output_dir = str(tmp_path)
        mock_load_scalers.return_value = sample_scalers_data_ppi
        mock_load_proc_meta.return_value = None

        with patch('yaml.safe_load', return_value=mock_params_config_ppi):
            result = prepare_prediction_input.run_prepare_input(
                "dummy_config.yaml", prod_model_run_id, output_dir
            )
        assert result is None
        assert f"Could not load feature_columns or tickers for run_id: {prod_model_run_id}" in caplog.text

    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    def test_run_prepare_input_scaler_dimension_mismatch(self, mock_load_proc_meta, mock_load_scalers,
                                                         mock_params_config_ppi, tmp_path, caplog):
        prod_model_run_id = "run_fail_dim_mismatch"
        output_dir = str(tmp_path)
        
        # Scalers for 1 stock, 2 features
        mismatched_scalers = {'scalers_x': [[MagicMock(spec=MinMaxScaler)]*2]}
        # Meta for 2 stocks, 3 features
        mismatched_meta = {'feature_columns': ['f1','f2','f3'], 'tickers': ['T1','T2']}

        mock_load_scalers.return_value = mismatched_scalers
        mock_load_proc_meta.return_value = mismatched_meta

        with patch('yaml.safe_load', return_value=mock_params_config_ppi):
            result = prepare_prediction_input.run_prepare_input(
                "dummy_config.yaml", prod_model_run_id, output_dir
            )
        assert result is None
        assert "Mismatch in dimensions of loaded scalers_x" in caplog.text


    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.get_latest_raw_data_window')
    def test_run_prepare_input_get_raw_data_fails(self, mock_get_latest_raw, mock_load_proc_meta,
                                                  mock_load_scalers, mock_params_config_ppi,
                                                  sample_scalers_data_ppi, sample_processed_features_meta_ppi,
                                                  tmp_path, caplog):
        prod_model_run_id = "run_fail_raw"
        output_dir = str(tmp_path)
        mock_load_scalers.return_value = sample_scalers_data_ppi
        mock_load_proc_meta.return_value = sample_processed_features_meta_ppi
        mock_get_latest_raw.return_value = {} # No data fetched

        with patch('yaml.safe_load', return_value=mock_params_config_ppi):
            result = prepare_prediction_input.run_prepare_input(
                "dummy_config.yaml", prod_model_run_id, output_dir
            )
        assert result is None
        assert "Failed to fetch any latest raw data" in caplog.text

    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.get_latest_raw_data_window')
    def test_run_prepare_input_missing_ticker_data(self, mock_get_latest_raw, mock_load_proc_meta,
                                                   mock_load_scalers, mock_params_config_ppi,
                                                   sample_scalers_data_ppi, sample_processed_features_meta_ppi,
                                                   sample_latest_raw_data_dict_ppi, tmp_path, caplog):
        prod_model_run_id = "run_fail_missing_ticker"
        output_dir = str(tmp_path)
        mock_load_scalers.return_value = sample_scalers_data_ppi
        mock_load_proc_meta.return_value = sample_processed_features_meta_ppi # Expects TICKA, TICKB
        
        # Simulate raw data fetched only for TICKA
        raw_data_missing_one_ticker = {'TICKA': sample_latest_raw_data_dict_ppi['TICKA']}
        mock_get_latest_raw.return_value = raw_data_missing_one_ticker

        with patch('yaml.safe_load', return_value=mock_params_config_ppi):
            result = prepare_prediction_input.run_prepare_input(
                "dummy_config.yaml", prod_model_run_id, output_dir
            )
        assert result is None
        assert "Raw data for ticker TICKB (expected by production model) is missing or empty" in caplog.text


    @patch('features.prepare_prediction_input.load_scalers')
    @patch('features.prepare_prediction_input.load_processed_features_from_db')
    @patch('features.prepare_prediction_input.get_latest_raw_data_window')
    @patch('features.prepare_prediction_input.add_technical_indicators')
    def test_run_prepare_input_not_enough_timesteps(self, mock_add_ta, mock_get_latest_raw,
                                                    mock_load_proc_meta, mock_load_scalers,
                                                    mock_params_config_ppi, sample_scalers_data_ppi,
                                                    sample_processed_features_meta_ppi, tmp_path, caplog):
        prod_model_run_id = "run_fail_short_seq"
        output_dir = str(tmp_path)
        
        # Config sequence_length is 5
        # Provide raw data that results in fewer than 5 timesteps after TA and alignment
        short_raw_data = {}
        dates_short = pd.to_datetime([pd.Timestamp('2023-01-01') + pd.Timedelta(days=i) for i in range(3)]) # Only 3 days
        for ticker in sample_processed_features_meta_ppi['tickers']:
            short_raw_data[ticker] = pd.DataFrame({
                col: np.random.rand(3) for col in sample_processed_features_meta_ppi['feature_columns']
            }, index=dates_short)
            # Add other necessary columns for add_technical_indicators if not in feature_columns
            if 'Close' not in short_raw_data[ticker]: short_raw_data[ticker]['Close'] = np.random.rand(3)
            if 'High' not in short_raw_data[ticker]: short_raw_data[ticker]['High'] = np.random.rand(3)
            if 'Low' not in short_raw_data[ticker]: short_raw_data[ticker]['Low'] = np.random.rand(3)
            if 'Volume' not in short_raw_data[ticker]: short_raw_data[ticker]['Volume'] = np.random.rand(3)


        mock_load_scalers.return_value = sample_scalers_data_ppi
        mock_load_proc_meta.return_value = sample_processed_features_meta_ppi
        mock_get_latest_raw.return_value = short_raw_data
        mock_add_ta.side_effect = lambda x: x # Pass through for this test

        with patch('yaml.safe_load', return_value=mock_params_config_ppi):
            result = prepare_prediction_input.run_prepare_input(
                "dummy_config.yaml", prod_model_run_id, output_dir
            )
        assert result is None
        assert "Not enough timesteps (3) after processing to form a sequence of length 5" in caplog.text