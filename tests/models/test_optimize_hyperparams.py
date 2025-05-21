# tests/models/test_optimize_hyperparams.py
import pytest
from unittest.mock import patch, MagicMock, call, ANY
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np
import yaml
import json
from pathlib import Path
import optuna
import io

from models.model_definitions import StockLSTM, StockLSTMWithAttention, StockLSTMWithCrossStockAttention
from models.optimize_hyperparams import objective, run_optimization

# --- Fixtures ---
@pytest.fixture
def mock_db_config_opt():
    return {'dbname': 'test_opt_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_opt_params_config():
    return {
        'n_trials': 1,
        'epochs': 1,
        'patience': 2,
        'timeout_seconds': None
    }

@pytest.fixture
def mock_params_config_opt(mock_db_config_opt, mock_opt_params_config):
    return {
        'database': mock_db_config_opt,
        'optimization': mock_opt_params_config,
        'output_paths': {'best_params_path': 'config/best_params_test.json'}
    }

@pytest.fixture
def sample_scaled_data_opt():
    return {
        'X_train': np.random.rand(32, 5, 2, 3).astype(np.float32),
        'y_train': np.random.rand(32, 1, 2).astype(np.float32),
        'X_test': np.random.rand(10, 5, 2, 3).astype(np.float32),
        'y_test': np.random.rand(10, 1, 2).astype(np.float32),
    }

@pytest.fixture
def mock_y_scalers_opt():
    s1, s2 = MagicMock(spec=MinMaxScaler), MagicMock(spec=MinMaxScaler)
    s1.inverse_transform.side_effect = lambda x: x
    s2.inverse_transform.side_effect = lambda x: x
    return [s1, s2]

@pytest.fixture
def mock_trial():
    trial = MagicMock(spec=optuna.trial.Trial)
    trial.suggest_int.side_effect = [32, 64, 2]  # batch_size, hidden_size, num_layers
    trial.suggest_float.side_effect = [0.3, 0.001]  # dropout_rate, learning_rate
    trial.suggest_categorical.return_value = 'lstm'
    return trial

@pytest.fixture
def device_eval():
    return torch.device("cpu")

# --- Tests for objective function (Optuna trial) ---
class TestObjectiveFunction:
    @patch('models.optimize_hyperparams.StockLSTMWithCrossStockAttention')
    @patch('models.optimize_hyperparams.StockLSTMWithAttention')
    @patch('models.optimize_hyperparams.StockLSTM')
    @patch('models.optimize_hyperparams.evaluate_model')
    # --- MODIFIED: Correct patch target for DataLoader ---
    @patch('models.optimize_hyperparams.DataLoader')
    def test_objective_calls_and_returns_loss(self, mock_data_loader_class, mock_evaluate_model,
                                              mock_stock_lstm_class, mock_stock_lstm_att_class, mock_stock_lstm_cross_class,
                                              mock_trial, sample_scaled_data_opt,
                                              mock_y_scalers_opt, device_eval, mock_opt_params_config):

        class SimpleDummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Ensure input_features_flat matches X_train's feature dimensions flattened
                # X_train: (batch, seq_len, num_stocks, num_features) -> (32, 5, 2, 3)
                # input for linear: seq_len * num_stocks * num_features = 5 * 2 * 3 = 30
                # output for linear: pred_len * num_stocks = 1 * 2 = 2 (assuming pred_len=1)
                self.linear = nn.Linear(5 * 2 * 3, 1 * 2)
            def forward(self, x): # x: (batch, seq, stocks, features)
                batch_size = x.shape[0]
                x_flat = x.reshape(batch_size, -1) # Use reshape for contiguity if needed, or view
                out_flat = self.linear(x_flat)
                return out_flat.view(batch_size, 1, 2) # Assuming pred_len=1, num_stocks=2

        actual_model_instance = SimpleDummyModel().to(device_eval)
        mock_stock_lstm_class.return_value = actual_model_instance

        X_train_tensor = torch.tensor(sample_scaled_data_opt['X_train'], device=device_eval)
        y_train_tensor = torch.tensor(sample_scaled_data_opt['y_train'], device=device_eval)
        X_test_tensor = torch.tensor(sample_scaled_data_opt['X_test'], device=device_eval)
        y_test_tensor = torch.tensor(sample_scaled_data_opt['y_test'], device=device_eval)

        mock_train_loader_instance = iter([(X_train_tensor, y_train_tensor)])
        mock_test_loader_instance = iter([(X_test_tensor, y_test_tensor)])

        # When DataLoader class is called, return our mock instances in order
        mock_data_loader_class.side_effect = [
            mock_train_loader_instance, # For train_loader
            mock_test_loader_instance   # For test_loader
        ]
        
        mock_evaluate_model.return_value = (None, None, {'test_loss': 0.5, 'avg_mape': 0.1})

        loss = objective(
            mock_trial,
            sample_scaled_data_opt['X_train'], sample_scaled_data_opt['y_train'],
            sample_scaled_data_opt['X_test'], sample_scaled_data_opt['y_test'],
            num_features=3, num_stocks=2, y_scalers=mock_y_scalers_opt,
            device=device_eval, optimization_epochs=mock_opt_params_config['epochs'],
            patience=mock_opt_params_config['patience']
        )

        mock_trial.suggest_int.assert_any_call('batch_size', 16, 128, step=16)
        mock_trial.suggest_categorical.assert_called_once_with('model_type', ['lstm', 'lstm_attention', 'lstm_cross_attention'])
        
        mock_stock_lstm_class.assert_called_once()
        
        # DataLoader should be called twice: once for training, once for testing
        assert mock_data_loader_class.call_count == 2
        
        # Check calls to DataLoader instantiation
        # First call is test_loader, second is train_loader based on code structure in `objective` (if X_test.size > 0)
        # or first is train_loader, second is test_loader if conditions for test_loader creation are met first.
        # Let's check the code in `objective`: test_loader is created first if X_test has data.
        expected_calls_to_dataloader = []
        # Test DataLoader call args
        # The TensorDataset is created inline, so we use ANY for that part
        if sample_scaled_data_opt['X_test'].size > 0: # Based on fixture data
             expected_calls_to_dataloader.append(call(ANY, batch_size=32, shuffle=False)) # test_loader
        else: # If test data was empty, the dummy loader would be created.
             expected_calls_to_dataloader.append(call(ANY, batch_size=32, shuffle=False)) # dummy_test_loader

        expected_calls_to_dataloader.append(call(ANY, batch_size=32, shuffle=True))  # train_loader
        
        mock_data_loader_class.assert_has_calls(expected_calls_to_dataloader, any_order=False)


        mock_evaluate_model.assert_called()
        assert loss == pytest.approx(0.5)

# --- Tests for run_optimization ---
class TestRunOptimization:
    @patch('models.optimize_hyperparams.json.dump')
    # --- MODIFIED: Removed redundant Path.mkdir patch ---
    # @patch('models.optimize_hyperparams.Path.mkdir')
    @patch('models.optimize_hyperparams.save_optimization_results')
    @patch('models.optimize_hyperparams.optuna.create_study')
    @patch('models.optimize_hyperparams.load_scalers')
    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.open')
    @patch('yaml.safe_load')
    def test_run_optimization_success(self, mock_yaml_safe_load, mock_open,
                                      mock_load_scaled, mock_load_scalers,
                                      mock_create_study, mock_save_opt_results,
                                      # mock_path_mkdir_constructor, # Removed
                                      mock_json_dump,
                                      mock_params_config_opt, sample_scaled_data_opt,
                                      mock_y_scalers_opt, tmp_path):
        run_id = "opt_run_001"
        dummy_config_path = str(tmp_path / "params_opt.yaml")

        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt

        mock_load_scaled.side_effect = lambda db_cfg, r_id, set_name: sample_scaled_data_opt.get(set_name)
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_opt, 'tickers': ['T1','T2'], 'num_features':3}

        mock_study_instance = MagicMock(spec=optuna.study.Study)
        mock_study_instance.best_params = {'lr': 0.001, 'hidden_size': 128}
        mock_study_instance.best_value = 0.05
        mock_study_instance.optimize = MagicMock()
        mock_create_study.return_value = mock_study_instance

        # --- MODIFIED: Define mock_path_instance_for_json and its parent before the 'with' block ---
        mock_path_instance_for_json = MagicMock(spec=Path)
        mock_path_parent_instance_for_json = MagicMock(spec=Path)
        mock_path_instance_for_json.parent = mock_path_parent_instance_for_json
        # mock_path_parent_instance_for_json.mkdir is already a MagicMock by default

        with patch('models.optimize_hyperparams.Path') as mock_path_class_overall:
            def path_side_effect(path_arg):
                # Path(path_arg) here will use pathlib.Path from the test file's global scope
                if str(path_arg) == mock_params_config_opt['output_paths']['best_params_path']:
                    return mock_path_instance_for_json
                return Path(path_arg) # Uses real pathlib.Path for other instantiations
            mock_path_class_overall.side_effect = path_side_effect

            best_params_out, run_id_out = run_optimization(dummy_config_path, run_id)

            # Assert that mkdir was called on the parent of the json path
            mock_path_parent_instance_for_json.mkdir.assert_called_once_with(parents=True, exist_ok=True)

        mock_open.assert_any_call(dummy_config_path, 'r')
        # --- MODIFIED: Assert open with the mock_path_instance_for_json ---
        mock_open.assert_any_call(mock_path_instance_for_json, 'w')

        mock_yaml_safe_load.assert_called_once_with(ANY) # Called with the first open's file handle

        mock_load_scaled.assert_any_call(mock_params_config_opt['database'], run_id, 'X_train')
        mock_load_scalers.assert_called_once_with(mock_params_config_opt['database'], run_id)
        mock_create_study.assert_called_once_with(direction="minimize", pruner=ANY)
        mock_study_instance.optimize.assert_called_once()
        mock_save_opt_results.assert_called_once_with(
            mock_params_config_opt['database'], run_id, mock_study_instance.best_params
        )

        mock_json_dump.assert_called_once_with(mock_study_instance.best_params, ANY, indent=4)

        assert best_params_out == mock_study_instance.best_params
        assert run_id_out == run_id

    @patch('models.optimize_hyperparams.load_scalers')
    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.open')
    @patch('yaml.safe_load')
    def test_run_optimization_load_scaled_fails(self, mock_yaml_safe_load, mock_open,
                                                mock_load_scaled, mock_load_scalers_used, # Renamed fixture to avoid confusion
                                                mock_params_config_opt, tmp_path, caplog):
        run_id = "opt_fail_load_scaled"
        dummy_config_path = str(tmp_path / "cfg.yaml")
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt

        mock_load_scaled.return_value = None
        # --- MODIFIED: Initialize params dict as it's done in run_optimization ---
        # This helps if the error occurs before params['database'] is accessed for load_scaled_features
        mock_params_config_opt_copy = mock_params_config_opt.copy() # Avoid modifying fixture for other tests

        best_params_out, run_id_out = run_optimization(dummy_config_path, run_id)

        assert best_params_out is None
        assert run_id_out is None
        assert "Failed to load scaled training data (X_train or y_train)" in caplog.text
        # X_train, y_train, X_test, y_test are attempted to be loaded. If X_train is None, it fails.
        # The exact number of calls can vary depending on where the None is returned.
        # If X_train is None, it might be 1 call. If y_train is None, 2 calls.
        # The code checks `if X_train_scaled is None or y_train_scaled is None:`
        # So, it will always call for X_train and y_train.
        assert mock_load_scaled.call_count >= 2 # At least for X_train and y_train
        mock_load_scalers_used.assert_not_called()


    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.load_scalers')
    @patch('models.optimize_hyperparams.open')
    @patch('yaml.safe_load')
    def test_run_optimization_load_scalers_fails(self, mock_yaml_safe_load, mock_open,
                                                 mock_load_scalers, mock_load_scaled,
                                                 mock_params_config_opt, sample_scaled_data_opt,
                                                 tmp_path, caplog):
        run_id = "opt_fail_load_scalers"
        dummy_config_path = str(tmp_path / "cfg.yaml")
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt

        mock_load_scaled.side_effect = lambda db_cfg, r_id, set_name: sample_scaled_data_opt.get(set_name)
        mock_load_scalers.return_value = None

        best_params_out, run_id_out = run_optimization(dummy_config_path, run_id)

        assert best_params_out is None
        assert run_id_out is None
        assert "Failed to load scalers or 'y_scalers' not found" in caplog.text
        assert mock_load_scaled.call_count == 4 # X_train, y_train, X_test, y_test
        mock_load_scalers.assert_called_once()


    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.load_scalers')
    @patch('models.optimize_hyperparams.open')
    @patch('yaml.safe_load')
    def test_run_optimization_x_train_bad_dims(self, mock_yaml_safe_load, mock_open,
                                               mock_load_scalers, mock_load_scaled,
                                               mock_params_config_opt, mock_y_scalers_opt,
                                               tmp_path, caplog):
        run_id = "opt_fail_dims"
        dummy_config_path = str(tmp_path / "cfg.yaml")
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt

        bad_X_train = np.random.rand(20, 5, 2).astype(np.float32) # 3D instead of 4D
        def load_scaled_side_effect_dims(db_cfg, r_id, set_name):
            if set_name == 'X_train': return bad_X_train
            if set_name == 'y_train': return np.random.rand(20,1,2).astype(np.float32)
            if set_name == 'X_test': return np.random.rand(10,5,2,3).astype(np.float32)
            if set_name == 'y_test': return np.random.rand(10,1,2).astype(np.float32)
            return None
        mock_load_scaled.side_effect = load_scaled_side_effect_dims

        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_opt, 'tickers':['T1','T2'], 'num_features':2}

        best_params_out, run_id_out = run_optimization(dummy_config_path, run_id)

        assert best_params_out is None
        assert run_id_out is None
        assert "X_train_scaled has unexpected dimensions" in caplog.text
        # It loads X_train, y_train, X_test, y_test, then scalers, then checks X_train.ndim
        assert mock_load_scaled.call_count == 4
        mock_load_scalers.assert_called_once()