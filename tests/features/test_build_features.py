# tests/features/test_build_features.py
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import pandas as pd # Though not directly used by build_features, good for sample data context
import yaml
from sklearn.preprocessing import MinMaxScaler # For type hinting and understanding

# Add src to sys.path
PROJECT_ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT_FOR_TESTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from features import build_features # Import the module

# --- Fixtures ---

@pytest.fixture
def mock_db_config():
    return {
        'dbname': 'test_db_features',
        'user': 'test_user',
        'password': 'test_password',
        'host': 'localhost',
        'port': '5432'
    }

@pytest.fixture
def mock_feature_eng_params_bf(): # Suffix bf for build_features
    return {
        'sequence_length': 5,
        'prediction_length': 1,
        'train_ratio': 0.8
    }

@pytest.fixture
def mock_params_config_bf(mock_db_config, mock_feature_eng_params_bf):
    return {
        'database': mock_db_config,
        'feature_engineering': mock_feature_eng_params_bf
    }

@pytest.fixture
def sample_processed_data_dict_bf():
    # (timesteps, stocks, features)
    processed_data = np.arange(100 * 2 * 3).reshape(100, 2, 3).astype(float)
    # (timesteps, stocks)
    targets = np.arange(100 * 2).reshape(100, 2).astype(float)
    return {
        'processed_data': processed_data,
        'targets': targets,
        'feature_columns': ['feat1', 'feat2', 'feat3'],
        'tickers': ['TICKA', 'TICKB']
    }

@pytest.fixture
def sample_scalers_dict_bf():
    # Mock scalers
    mock_scaler_x1_f1 = MagicMock(spec=MinMaxScaler)
    mock_scaler_x1_f2 = MagicMock(spec=MinMaxScaler)
    mock_scaler_x2_f1 = MagicMock(spec=MinMaxScaler)
    mock_scaler_x2_f2 = MagicMock(spec=MinMaxScaler)
    
    mock_scaler_y1 = MagicMock(spec=MinMaxScaler)
    mock_scaler_y2 = MagicMock(spec=MinMaxScaler)

    return {
        'scalers_x': [[mock_scaler_x1_f1, mock_scaler_x1_f2], [mock_scaler_x2_f1, mock_scaler_x2_f2]],
        'y_scalers': [mock_scaler_y1, mock_scaler_y2],
        'tickers': ['TICKA', 'TICKB'],
        'num_features': 2 # Assuming 2 features for this mock scaler dict
    }


# --- Tests for individual functions ---

class TestCreateSequences:
    def test_correct_sequence_creation(self):
        # data_x: (timesteps, stocks, features)
        data_x = np.arange(10 * 2 * 3).reshape(10, 2, 3)
        # data_y: (timesteps, stocks)
        data_y = np.arange(10 * 2).reshape(10, 2)
        seq_len = 3
        pred_len = 1
        
        # Expected number of sequences: len(data_x) - seq_len - pred_len + 1 = 10 - 3 - 1 + 1 = 7
        expected_num_sequences = 7

        sequences_x, sequences_y = build_features.create_sequences(data_x, data_y, seq_len, pred_len)

        assert sequences_x.shape == (expected_num_sequences, seq_len, 2, 3)
        assert sequences_y.shape == (expected_num_sequences, pred_len, 2) # y is (samples, pred_len, num_stocks)

        # Check content of first sequence
        np.testing.assert_array_equal(sequences_x[0], data_x[0:3])
        np.testing.assert_array_equal(sequences_y[0], data_y[3:4].reshape(pred_len, 2)) # Target is data_y[i+seq_len : i+seq_len+pred_len]

    def test_insufficient_data_for_sequences(self):
        data_x = np.arange(3 * 2 * 3).reshape(3, 2, 3) # Only 3 timesteps
        data_y = np.arange(3 * 2).reshape(3, 2)
        seq_len = 5 # Requires 5 timesteps
        pred_len = 1

        sequences_x, sequences_y = build_features.create_sequences(data_x, data_y, seq_len, pred_len)

        assert sequences_x.shape == (0, seq_len, 2, 3) # Expect empty array with correct other dims
        assert sequences_y.shape == (0, pred_len, 2)


class TestScaleData:
    @patch('features.build_features.MinMaxScaler') # Mock the class
    def test_scale_data_logic(self, mock_min_max_scaler_class):
        # X_train: (samples, seq_len, num_stocks, num_features)
        X_train = np.random.rand(50, 5, 2, 3) # 50 samples, 5 seq_len, 2 stocks, 3 features
        X_test = np.random.rand(20, 5, 2, 3)
        # y_train: (samples, pred_len, num_stocks)
        y_train = np.random.rand(50, 1, 2) # pred_len = 1
        y_test = np.random.rand(20, 1, 2)
        num_features = 3
        num_stocks = 2

        # Create mock instances for each scaler that would be created
        mock_scalers_x_instances = [
            [MagicMock(spec=MinMaxScaler) for _ in range(num_features)] for _ in range(num_stocks)
        ]
        mock_y_scalers_instances = [MagicMock(spec=MinMaxScaler) for _ in range(num_stocks)]

        # Configure the mock class to return our instances in order
        all_mock_scaler_instances = []
        for stock_scalers in mock_scalers_x_instances:
            all_mock_scaler_instances.extend(stock_scalers)
        all_mock_scaler_instances.extend(mock_y_scalers_instances)
        
        mock_min_max_scaler_class.side_effect = all_mock_scaler_instances
        
        # Mock transform to return identity for simplicity (we're testing calls, not scaling values)
        for stock_scalers in mock_scalers_x_instances:
            for scaler in stock_scalers:
                scaler.transform.side_effect = lambda x: x 
        for scaler in mock_y_scalers_instances:
            scaler.transform.side_effect = lambda x: x

        X_train_s, X_test_s, y_train_s, y_test_s, scalers_x_out, y_scalers_out = \
            build_features.scale_data(X_train, X_test, y_train, y_test, num_features, num_stocks)

        assert mock_min_max_scaler_class.call_count == (num_stocks * num_features) + num_stocks

        # Verify fit calls for X scalers (only on train data)
        for stock_idx in range(num_stocks):
            for feature_idx in range(num_features):
                scaler_mock = mock_scalers_x_instances[stock_idx][feature_idx]
                scaler_mock.fit.assert_called_once()
                # Check that transform was called for both train and test
                assert scaler_mock.transform.call_count == 2
        
        # Verify fit calls for y scalers (only on train data)
        for stock_idx in range(num_stocks):
            scaler_mock = mock_y_scalers_instances[stock_idx]
            scaler_mock.fit.assert_called_once()
            assert scaler_mock.transform.call_count == 2

        assert X_train_s.shape == X_train.shape
        assert X_test_s.shape == X_test.shape
        assert y_train_s.shape == y_train.shape
        assert y_test_s.shape == y_test.shape
        assert len(scalers_x_out) == num_stocks
        assert len(scalers_x_out[0]) == num_features
        assert len(y_scalers_out) == num_stocks

        # Check that the returned scalers are the ones we mocked
        for r_idx, row in enumerate(scalers_x_out):
            for c_idx, scaler in enumerate(row):
                assert scaler is mock_scalers_x_instances[r_idx][c_idx]
        for idx, scaler in enumerate(y_scalers_out):
            assert scaler is mock_y_scalers_instances[idx]


class TestRunFeatureBuilding:
    @patch('features.build_features.load_processed_features_from_db')
    @patch('features.build_features.create_sequences')
    @patch('features.build_features.scale_data')
    @patch('features.build_features.save_scaled_features')
    @patch('features.build_features.save_scalers')
    def test_run_feature_building_success(self, mock_save_scalers, mock_save_scaled_features,
                                          mock_scale_data, mock_create_sequences,
                                          mock_load_processed, mock_params_config_bf,
                                          sample_processed_data_dict_bf, tmp_path):
        run_id = "test_run_123"
        config_file = tmp_path / "params_bf.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config_bf, f)

        mock_load_processed.return_value = sample_processed_data_dict_bf
        
        # Mock create_sequences return
        # X_train: (samples, seq_len, num_stocks, num_features)
        # y_train: (samples, pred_len, num_stocks)
        seq_len = mock_params_config_bf['feature_engineering']['sequence_length']
        pred_len = mock_params_config_bf['feature_engineering']['prediction_length']
        num_stocks = len(sample_processed_data_dict_bf['tickers'])
        num_features = len(sample_processed_data_dict_bf['feature_columns'])
        
        # Based on train_ratio = 0.8, 100 timesteps -> 80 train, 20 test
        # Train sequences: 80 - 5 - 1 + 1 = 75
        # Test sequences: 20 - 5 - 1 + 1 = 15
        X_train_seq = np.random.rand(75, seq_len, num_stocks, num_features)
        y_train_seq = np.random.rand(75, pred_len, num_stocks)
        X_test_seq = np.random.rand(15, seq_len, num_stocks, num_features)
        y_test_seq = np.random.rand(15, pred_len, num_stocks)
        mock_create_sequences.side_effect = [
            (X_train_seq, y_train_seq), # For train split
            (X_test_seq, y_test_seq)    # For test split
        ]

        # Mock scale_data return
        mock_scalers_x = [[MagicMock(spec=MinMaxScaler)]*num_features]*num_stocks
        mock_y_scalers = [MagicMock(spec=MinMaxScaler)]*num_stocks
        mock_scale_data.return_value = (
            X_train_seq, X_test_seq, y_train_seq, y_test_seq, # Scaled data (mocked as same for simplicity)
            mock_scalers_x, mock_y_scalers
        )

        with patch('yaml.safe_load', return_value=mock_params_config_bf):
            result_run_id = build_features.run_feature_building(str(config_file), run_id)

        assert result_run_id == run_id
        mock_load_processed.assert_called_once_with(mock_params_config_bf['database'], run_id=run_id)
        assert mock_create_sequences.call_count == 2 # Once for train, once for test
        mock_scale_data.assert_called_once()
        
        expected_save_scaled_calls = [
            call(mock_params_config_bf['database'], run_id, 'X_train', X_train_seq),
            call(mock_params_config_bf['database'], run_id, 'y_train', y_train_seq),
            call(mock_params_config_bf['database'], run_id, 'X_test', X_test_seq),
            call(mock_params_config_bf['database'], run_id, 'y_test', y_test_seq),
        ]
        mock_save_scaled_features.assert_has_calls(expected_save_scaled_calls, any_order=False)
        
        expected_scalers_dict_to_save = {
            'scalers_x': mock_scalers_x, 
            'y_scalers': mock_y_scalers, 
            'tickers': sample_processed_data_dict_bf['tickers'], 
            'num_features': num_features
        }
        # For dict comparison, it's safer to check args if the dict is complex
        args_call_save_scalers, _ = mock_save_scalers.call_args
        assert args_call_save_scalers[0] == mock_params_config_bf['database']
        assert args_call_save_scalers[1] == run_id
        assert args_call_save_scalers[2] == expected_scalers_dict_to_save


    @patch('features.build_features.load_processed_features_from_db')
    def test_run_feature_building_load_processed_fails(self, mock_load_processed,
                                                       mock_params_config_bf, tmp_path, caplog):
        run_id = "test_run_fail1"
        config_file = tmp_path / "params_bf.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config_bf, f)
        
        mock_load_processed.return_value = None # Simulate failure

        with patch('yaml.safe_load', return_value=mock_params_config_bf):
            result = build_features.run_feature_building(str(config_file), run_id)
        
        assert result is None
        assert f"No processed data found in database for run_id: {run_id}" in caplog.text

    @patch('features.build_features.load_processed_features_from_db')
    def test_run_feature_building_empty_processed_data(self, mock_load_processed,
                                                       mock_params_config_bf, tmp_path, caplog):
        run_id = "test_run_fail2"
        config_file = tmp_path / "params_bf.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config_bf, f)
        
        empty_data_dict = {
            'processed_data': np.array([]), 'targets': np.array([]),
            'feature_columns': [], 'tickers': []
        }
        mock_load_processed.return_value = empty_data_dict

        with patch('yaml.safe_load', return_value=mock_params_config_bf):
            result = build_features.run_feature_building(str(config_file), run_id)
        
        assert result is None
        assert f"Loaded processed data or targets are empty/None for run_id: {run_id}" in caplog.text

    @patch('features.build_features.load_processed_features_from_db')
    def test_run_feature_building_train_size_too_small(self, mock_load_processed,
                                                       mock_params_config_bf, tmp_path, caplog):
        run_id = "test_run_fail3"
        config_file = tmp_path / "params_bf.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config_bf, f)

        # Data too short for sequence_length + prediction_length
        short_data_dict = {
            'processed_data': np.random.rand(5, 2, 3), # Only 5 timesteps
            'targets': np.random.rand(5, 2),
            'feature_columns': ['f1','f2','f3'], 'tickers': ['T1','T2']
        }
        mock_load_processed.return_value = short_data_dict
        # seq_len=5, pred_len=1. train_ratio=0.8. 5*0.8 = 4 train samples.
        # Need seq_len + pred_len = 6 for one sequence. 4 < 6.

        with patch('yaml.safe_load', return_value=mock_params_config_bf):
            result = build_features.run_feature_building(str(config_file), run_id)
        
        assert result is None
        assert "Train size (4) is too small to create any sequences" in caplog.text


    @patch('features.build_features.load_processed_features_from_db')
    @patch('features.build_features.create_sequences')
    def test_run_feature_building_empty_train_sequences(self, mock_create_sequences, mock_load_processed,
                                                        mock_params_config_bf, sample_processed_data_dict_bf,
                                                        tmp_path, caplog):
        run_id = "test_run_fail4"
        config_file = tmp_path / "params_bf.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(mock_params_config_bf, f)

        mock_load_processed.return_value = sample_processed_data_dict_bf
        # Mock create_sequences to return empty for training
        mock_create_sequences.return_value = (np.array([]), np.array([]))

        with patch('yaml.safe_load', return_value=mock_params_config_bf):
            result = build_features.run_feature_building(str(config_file), run_id)
        
        assert result is None
        assert "Failed to create training sequences" in caplog.text