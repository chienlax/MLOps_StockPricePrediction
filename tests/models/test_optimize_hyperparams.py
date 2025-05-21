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

# Import the module to be tested
from models import optimize_hyperparams

# --- Fixtures (Simplified where possible or kept if essential for structure) ---
@pytest.fixture
def mock_db_config_opt():
    return {'dbname': 'test_opt_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_opt_params_config_simple(): # Simplified
    return {
        'n_trials': 1,
        'epochs': 1,
        'patience': 1,
        'timeout_seconds': None
    }

@pytest.fixture
def mock_params_config_opt_simple(mock_db_config_opt, mock_opt_params_config_simple): # Simplified
    return {
        'database': mock_db_config_opt,
        'optimization': mock_opt_params_config_simple,
        'output_paths': {'best_params_path': 'config/best_params_test_simple.json'}
    }

@pytest.fixture
def sample_scaled_data_opt_simple(): # Simplified, focusing on shapes
    return {
        'X_train': np.random.rand(8, 5, 2, 3).astype(np.float32), # Small dataset
        'y_train': np.random.rand(8, 1, 2).astype(np.float32),
        'X_test': np.random.rand(4, 5, 2, 3).astype(np.float32),
        'y_test': np.random.rand(4, 1, 2).astype(np.float32),
    }

@pytest.fixture
def mock_y_scalers_opt_simple():
    # Only one scaler needed if num_stocks is 1, or a list if multiple
    mock_scaler = MagicMock(spec=MinMaxScaler)
    mock_scaler.inverse_transform.side_effect = lambda x: x
    return [mock_scaler, mock_scaler] # Assuming 2 stocks from sample_scaled_data_opt_simple

@pytest.fixture
def mock_trial_simple():
    trial = MagicMock(spec=optuna.trial.Trial)
    trial.suggest_int.return_value = 16 # batch_size, hidden_size, num_layers - return simple values
    trial.suggest_float.return_value = 0.1 # dropout_rate, learning_rate
    trial.suggest_categorical.return_value = 'lstm'
    return trial

@pytest.fixture
def device_eval_simple():
    return torch.device("cpu")

# --- Simplified Tests for objective function ---
class TestObjectiveFunctionSimple:
    @patch('models.optimize_hyperparams.evaluate_model')
    @patch('models.optimize_hyperparams.torch.utils.data.DataLoader')
    @patch('models.optimize_hyperparams.StockLSTM') # Assuming 'lstm' is chosen by mock_trial
    @patch('models.optimize_hyperparams.torch.optim.Adam')
    def test_objective_runs_and_returns_loss(
        self, mock_adam_class, mock_stock_lstm_class, mock_data_loader_class, mock_evaluate_model,
        mock_trial_simple, sample_scaled_data_opt_simple,
        mock_y_scalers_opt_simple, device_eval_simple, mock_opt_params_config_simple
    ):
        # Mock model instance
        mock_model_inst = MagicMock(spec=nn.Module)
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(1))] # Dummy parameter
        # Model output should be compatible with criterion and target
        # Batch size from trial is 16, but sample_scaled_data_opt_simple has 8 train samples
        # DataLoader with batch_size=16 and 8 samples will yield one batch of 8.
        mock_model_inst.return_value = torch.rand(8, 1, 2) # (batch_size_actual, pred_len, num_stocks)
        mock_stock_lstm_class.return_value = mock_model_inst

        # Mock DataLoader instances
        mock_train_loader_inst = MagicMock()
        mock_train_loader_inst.__iter__.return_value = iter([(
            torch.rand(8, 5, 2, 3), torch.rand(8, 1, 2) # One batch of actual size
        )])
        type(mock_train_loader_inst).__len__ = MagicMock(return_value=1) # For avg loss calculation

        mock_test_loader_inst = MagicMock()
        mock_test_loader_inst.__iter__.return_value = iter([(
            torch.rand(4, 5, 2, 3), torch.rand(4, 1, 2) # One batch of actual size
        )])
        type(mock_test_loader_inst).__len__ = MagicMock(return_value=1)

        mock_data_loader_class.side_effect = [mock_train_loader_inst, mock_test_loader_inst]

        # Mock evaluate_model to return a loss
        mock_evaluate_model.return_value = (None, None, {'test_loss': 0.75})

        # Mock optimizer
        mock_optimizer_inst = MagicMock()
        mock_adam_class.return_value = mock_optimizer_inst

        loss = optimize_hyperparams.objective(
            mock_trial_simple,
            sample_scaled_data_opt_simple['X_train'], sample_scaled_data_opt_simple['y_train'],
            sample_scaled_data_opt_simple['X_test'], sample_scaled_data_opt_simple['y_test'],
            num_features=3, num_stocks=2, y_scalers=mock_y_scalers_opt_simple,
            device=device_eval_simple,
            optimization_epochs=mock_opt_params_config_simple['epochs'],
            patience=mock_opt_params_config_simple['patience']
        )

        mock_stock_lstm_class.assert_called_once()
        assert mock_data_loader_class.call_count == 2
        mock_adam_class.assert_called_once()
        mock_optimizer_inst.step.assert_called() # Check training step happened
        mock_evaluate_model.assert_called_once()
        assert isinstance(loss, float)
        assert loss == 0.75

# --- Simplified Tests for run_optimization ---
class TestRunOptimizationSimple:
    @patch('models.optimize_hyperparams.json.dump')
    @patch('models.optimize_hyperparams.Path') # Mock the Path class itself
    @patch('models.optimize_hyperparams.save_optimization_results')
    @patch('models.optimize_hyperparams.optuna.create_study')
    @patch('models.optimize_hyperparams.load_scalers')
    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.open')
    @patch('yaml.safe_load')
    def test_run_optimization_flow(
        self, mock_yaml_safe_load, mock_open,
        mock_load_scaled, mock_load_scalers,
        mock_create_study, mock_save_opt_results,
        mock_path_class, mock_json_dump, # mock_path_class replaces mock_path_mkdir
        mock_params_config_opt_simple, sample_scaled_data_opt_simple,
        mock_y_scalers_opt_simple, tmp_path # tmp_path is still useful for dummy paths
    ):
        run_id = "opt_run_simple_001"
        dummy_config_path = str(tmp_path / "params_opt_simple.yaml")

        # Configure file/config mocks
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt_simple

        # Configure DB load mocks
        mock_load_scaled.side_effect = lambda db, rid, name: sample_scaled_data_opt_simple.get(name)
        mock_load_scalers.return_value = {
            'y_scalers': mock_y_scalers_opt_simple,
            'tickers': ['S1', 'S2'],
            'num_features': 3
        }

        # Configure Optuna mocks
        mock_study_inst = MagicMock(spec=optuna.study.Study)
        mock_study_inst.best_params = {'param': 'value'}
        mock_study_inst.best_value = 0.1
        mock_study_inst.optimize = MagicMock()
        mock_create_study.return_value = mock_study_inst
        
        # Configure Path class mock for creating output directory and file path
        # Path("...").parent.mkdir(...) and Path("...") for file opening
        mock_file_path_instance = MagicMock(spec=Path)
        mock_file_path_instance.parent.mkdir = MagicMock() # Mock the mkdir call on parent
        
        # When Path is called with the output json path, return our specific mock instance
        # For other calls to Path (if any), return a generic Path mock or real Path.
        def path_constructor_side_effect(path_arg_str):
            if path_arg_str == mock_params_config_opt_simple['output_paths']['best_params_path']:
                return mock_file_path_instance
            return Path(path_arg_str) # Fallback to real Path for other uses, if any.
        mock_path_class.side_effect = path_constructor_side_effect
        
        # Call the function
        best_params, out_run_id = optimize_hyperparams.run_optimization(dummy_config_path, run_id)

        # Basic assertions
        mock_open.assert_any_call(dummy_config_path, 'r') # Config read
        mock_yaml_safe_load.assert_called_once()
        mock_load_scaled.assert_called()
        mock_load_scalers.assert_called_once()
        mock_create_study.assert_called_once()
        mock_study_inst.optimize.assert_called_once()
        mock_save_opt_results.assert_called_once()
        
        # Check if directory creation for JSON was attempted on the correct parent
        mock_file_path_instance.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        # Check if json file was attempted to be opened for writing
        mock_open.assert_any_call(mock_file_path_instance, 'w')
        mock_json_dump.assert_called_once()

        assert best_params == {'param': 'value'}
        assert out_run_id == run_id

    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.open')
    @patch('yaml.safe_load')
    def test_run_optimization_load_data_fails(
        self, mock_yaml_safe_load, mock_open, mock_load_scaled,
        mock_params_config_opt_simple, tmp_path, caplog
    ):
        run_id = "opt_run_fail_load"
        dummy_config_path = str(tmp_path / "cfg_fail.yaml")
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt_simple

        mock_load_scaled.return_value = None # Simulate failure

        best_params, out_run_id = optimize_hyperparams.run_optimization(dummy_config_path, run_id)

        assert best_params is None
        assert out_run_id is None
        assert "Failed to load scaled training data" in caplog.text