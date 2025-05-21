# tests/models/test_optimize_hyperparams.py
import pytest
from unittest.mock import patch, MagicMock, call
import torch
import torch.nn as nn
import numpy as np
import yaml
import json
from pathlib import Path
import optuna # For spec and trial object

from models.optimize_hyperparams import objective, run_optimization
# We might need to mock model_definitions and evaluate_model if testing objective deeply
# from models.model_definitions import StockLSTM # Example
# from models.evaluate_model import evaluate_model # Example

# --- Fixtures ---
@pytest.fixture
def mock_db_config_opt():
    return {'dbname': 'test_opt_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_opt_params_config():
    return {
        'n_trials': 2, # Small number for tests
        'epochs': 3,   # Epochs per Optuna trial
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
    # X: (samples, seq_len, num_stocks, num_features)
    # y: (samples, pred_len, num_stocks)
    return {
        'X_train': np.random.rand(20, 5, 2, 3).astype(np.float32),
        'y_train': np.random.rand(20, 1, 2).astype(np.float32),
        'X_test': np.random.rand(10, 5, 2, 3).astype(np.float32),
        'y_test': np.random.rand(10, 1, 2).astype(np.float32),
    }

@pytest.fixture
def mock_y_scalers_opt():
    # Mock 2 scalers for 2 stocks
    s1, s2 = MagicMock(), MagicMock()
    s1.inverse_transform.side_effect = lambda x: x
    s2.inverse_transform.side_effect = lambda x: x
    return [s1, s2]

@pytest.fixture
def mock_trial():
    trial = MagicMock(spec=optuna.trial.Trial)
    trial.suggest_int.side_effect = [32, 64, 2] # batch_size, hidden_size, num_layers
    trial.suggest_float.side_effect = [0.3, 0.001] # dropout_rate, learning_rate
    trial.suggest_categorical.return_value = 'lstm' # model_type
    return trial

# --- Tests for objective function (Optuna trial) ---
# COMPLEXITY NOTICE: Testing the 'objective' function thoroughly is hard because it
# involves a mini training loop. We'll mock key parts of that loop.
class TestObjectiveFunction:
    @patch('models.optimize_hyperparams.StockLSTM') # Mock the actual model class
    @patch('models.optimize_hyperparams.StockLSTMWithAttention')
    @patch('models.optimize_hyperparams.StockLSTMWithCrossStockAttention')
    @patch('models.optimize_hyperparams.evaluate_model')
    @patch('models.optimize_hyperparams.torch.utils.data.DataLoader') # Mock DataLoader
    def test_objective_calls_and_returns_loss(self, mock_data_loader, mock_evaluate_model,
                                              mock_lstm_cross, mock_lstm_att, mock_lstm_basic,
                                              mock_trial, sample_scaled_data_opt,
                                              mock_y_scalers_opt, device_eval, mock_opt_params_config):
        
        # Setup mocks for the training loop components
        mock_model_instance = MagicMock(spec=nn.Module)
        mock_model_instance.to.return_value = mock_model_instance # for .to(device)
        # Simulate model output for loss calculation
        # Model output: (batch_size, 1, num_stocks)
        # Target shape: (batch_size, 1, num_stocks)
        # For batch_size 32 (from mock_trial), 2 stocks
        mock_model_instance.return_value = torch.rand(32, 1, 2) 
        mock_lstm_basic.return_value = mock_model_instance # If 'lstm' is chosen

        # Mock DataLoader to return a single batch
        dummy_X_batch = torch.rand(32, 5, 2, 3)
        dummy_y_batch = torch.rand(32, 1, 2)
        mock_data_loader.return_value = [(dummy_X_batch, dummy_y_batch)] # Iterable

        # Mock evaluate_model to return a dummy metrics dict
        mock_evaluate_model.return_value = (None, None, {'test_loss': 0.5, 'avg_mape': 0.1})

        # Call the objective function
        loss = objective(
            mock_trial,
            sample_scaled_data_opt['X_train'], sample_scaled_data_opt['y_train'],
            sample_scaled_data_opt['X_test'], sample_scaled_data_opt['y_test'],
            num_features=3, num_stocks=2, y_scalers=mock_y_scalers_opt,
            device=device_eval, optimization_epochs=mock_opt_params_config['epochs'],
            patience=mock_opt_params_config['patience']
        )

        # Assertions
        mock_trial.suggest_int.assert_any_call('batch_size', 16, 128, step=16)
        mock_trial.suggest_categorical.assert_called_once_with('model_type', ['lstm', 'lstm_attention', 'lstm_cross_attention'])
        
        mock_lstm_basic.assert_called_once() # Since 'lstm' was returned by suggest_categorical
        mock_model_instance.to.assert_called_with(device_eval)
        
        assert mock_data_loader.call_count == 2 # Once for train, once for test
        
        # Check that the model's forward pass (mock_model_instance()) was called during training
        # This depends on batch_size and epochs. Here, 1 batch per epoch, 3 epochs.
        assert mock_model_instance.call_count >= mock_opt_params_config['epochs'] # At least once per training epoch

        mock_evaluate_model.assert_called() # Should be called at least once per epoch
        
        assert loss == 0.5 # Matches the mocked 'test_loss' from evaluate_model

# --- Tests for run_optimization ---
class TestRunOptimization:
    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.load_scalers')
    @patch('models.optimize_hyperparams.save_optimization_results')
    @patch('models.optimize_hyperparams.optuna.create_study')
    @patch('models.optimize_hyperparams.json.dump') # For saving best_params.json
    @patch('models.optimize_hyperparams.Path.mkdir') # For output_paths
    def test_run_optimization_success(self, mock_path_mkdir, mock_json_dump, mock_create_study,
                                      mock_save_opt_results, mock_load_scalers, mock_load_scaled,
                                      mock_params_config_opt, sample_scaled_data_opt,
                                      mock_y_scalers_opt, tmp_path):
        run_id = "opt_run_001"
        config_file = tmp_path / "params_opt.yaml" # Not actually read due to yaml.safe_load mock
        
        # Mock DB load calls
        mock_load_scaled.side_effect = lambda db_cfg, r_id, set_name: sample_scaled_data_opt[set_name]
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_opt, 'tickers': ['T1','T2'], 'num_features':3}

        # Mock Optuna study
        mock_study_instance = MagicMock()
        mock_study_instance.best_params = {'lr': 0.001, 'hidden_size': 128}
        mock_study_instance.best_value = 0.05
        mock_study_instance.optimize = MagicMock()
        mock_create_study.return_value = mock_study_instance

        # Mock config loading
        with patch('yaml.safe_load', return_value=mock_params_config_opt):
            best_params_out, run_id_out = run_optimization(str(config_file), run_id)

        mock_load_scaled.assert_any_call(mock_params_config_opt['database'], run_id, 'X_train')
        mock_load_scalers.assert_called_once_with(mock_params_config_opt['database'], run_id)
        
        mock_create_study.assert_called_once_with(direction="minimize", pruner=pytest.ANY) # ANY for pruner instance
        mock_study_instance.optimize.assert_called_once() # Check that optimize was called
        
        mock_save_opt_results.assert_called_once_with(
            mock_params_config_opt['database'], run_id, mock_study_instance.best_params
        )
        
        # Check saving to file
        expected_json_path = Path(mock_params_config_opt['output_paths']['best_params_path'])
        # mock_path_mkdir.assert_called_with(parents=True, exist_ok=True) # This is for expected_json_path.parent
        # Check that json.dump was called (the file object 'f' will be a MagicMock)
        mock_json_dump.assert_called_once_with(mock_study_instance.best_params, pytest.ANY, indent=4)


        assert best_params_out == mock_study_instance.best_params
        assert run_id_out == run_id

    @patch('models.optimize_hyperparams.load_scaled_features')
    def test_run_optimization_load_scaled_fails(self, mock_load_scaled, mock_params_config_opt, tmp_path, caplog):
        run_id = "opt_fail_load_scaled"
        mock_load_scaled.return_value = None # Simulate failure

        with patch('yaml.safe_load', return_value=mock_params_config_opt):
            best_params_out, run_id_out = run_optimization(str(tmp_path / "cfg.yaml"), run_id)
        
        assert best_params_out is None
        assert run_id_out is None
        assert "Failed to load scaled training data" in caplog.text

    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.load_scalers')
    def test_run_optimization_load_scalers_fails(self, mock_load_scalers, mock_load_scaled,
                                                 mock_params_config_opt, sample_scaled_data_opt,
                                                 tmp_path, caplog):
        run_id = "opt_fail_load_scalers"
        mock_load_scaled.side_effect = lambda db_cfg, r_id, set_name: sample_scaled_data_opt[set_name]
        mock_load_scalers.return_value = None # Simulate failure

        with patch('yaml.safe_load', return_value=mock_params_config_opt):
            best_params_out, run_id_out = run_optimization(str(tmp_path / "cfg.yaml"), run_id)

        assert best_params_out is None
        assert run_id_out is None
        assert "Failed to load scalers or 'y_scalers' not found" in caplog.text

    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.load_scalers')
    def test_run_optimization_x_train_bad_dims(self, mock_load_scalers, mock_load_scaled,
                                               mock_params_config_opt, mock_y_scalers_opt,
                                               tmp_path, caplog):
        run_id = "opt_fail_dims"
        # X_train with bad dimensions (e.g., 3D instead of 4D)
        bad_X_train = np.random.rand(20, 5, 2).astype(np.float32)
        mock_load_scaled.side_effect = lambda db_cfg, r_id, set_name: \
            bad_X_train if set_name == 'X_train' else np.random.rand(20,1,2).astype(np.float32)
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_opt}

        with patch('yaml.safe_load', return_value=mock_params_config_opt):
            best_params_out, run_id_out = run_optimization(str(tmp_path / "cfg.yaml"), run_id)
        
        assert best_params_out is None
        assert run_id_out is None
        assert "X_train_scaled has unexpected dimensions" in caplog.text