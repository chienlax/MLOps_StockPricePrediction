# tests/models/test_optimize_hyperparams.py
import pytest
from unittest.mock import patch, MagicMock, call, ANY
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim # Import optim
import numpy as np
import yaml
import json
from pathlib import Path
import optuna
import io

from models import optimize_hyperparams
from models.model_definitions import StockLSTM # Assuming this is the one suggested by mock_trial


# --- Fixtures ---
@pytest.fixture
def mock_db_config_opt():
    return {'dbname': 'test_opt_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_opt_params_config_simple():
    return {
        'n_trials': 1,
        'epochs': 1,
        'patience': 1,
        'timeout_seconds': None
    }

@pytest.fixture
def mock_params_config_opt_simple(mock_db_config_opt, mock_opt_params_config_simple):
    return {
        'database': mock_db_config_opt,
        'optimization': mock_opt_params_config_simple,
        'output_paths': {'best_params_path': 'config/best_params_test_simple.json'}
    }

@pytest.fixture
def sample_scaled_data_opt_simple():
    return {
        'X_train': np.random.rand(16, 5, 2, 3).astype(np.float32),
        'y_train': np.random.rand(16, 1, 2).astype(np.float32),
        'X_test': np.random.rand(4, 5, 2, 3).astype(np.float32), # Test can be smaller
        'y_test': np.random.rand(4, 1, 2).astype(np.float32),
    }

@pytest.fixture
def mock_y_scalers_opt_simple():
    mock_scaler = MagicMock(spec=MinMaxScaler)
    mock_scaler.inverse_transform.side_effect = lambda x: x
    return [mock_scaler, mock_scaler] # Assuming 2 stocks

@pytest.fixture
def mock_trial_simple():
    trial = MagicMock(spec=optuna.trial.Trial)
    trial.suggest_int.side_effect = [
        16, # batch_size (must match data or be a divisor)
        64, # hidden_size
        1   # num_layers
    ]
    trial.suggest_float.side_effect = [
        0.1,  # dropout_rate
        0.001 # learning_rate
    ]
    trial.suggest_categorical.return_value = 'lstm' # Ensure this matches a patched model class
    return trial

@pytest.fixture
def device_eval_simple():
    return torch.device("cpu")

class TestObjectiveFunctionSimple:
    @patch('models.optimize_hyperparams.evaluate_model')
    @patch('models.optimize_hyperparams.torch.utils.data.DataLoader')
    # Patch the specific model class that will be instantiated based on mock_trial_simple
    @patch('models.optimize_hyperparams.StockLSTM')
    @patch('models.optimize_hyperparams.torch.optim.Adam') # Keep Adam mock
    def test_objective_runs_and_returns_loss(
        self, mock_adam_class, mock_stock_lstm_class, # Order matches patches from bottom up
        mock_data_loader_class, mock_evaluate_model,
        mock_trial_simple, sample_scaled_data_opt_simple,
        mock_y_scalers_opt_simple, device_eval_simple, mock_opt_params_config_simple
    ):
        # 1. Configure mock_trial (already done by fixture)

        # 2. Mock the Model Instantiation (e.g., StockLSTM)
        #    The instance needs a parameters() method that returns an iterable of nn.Parameter
        #    and its forward pass should produce a tensor connected to these parameters.

        # --- Use a simple, real nn.Module for the mocked model instance ---
        class DummyTrainableModel(nn.Module):
            def __init__(self, input_size, output_size):
                super().__init__()
                # This parameter will have requires_grad=True by default
                self.fc = nn.Linear(input_size, output_size)

            def forward(self, x):
                # x shape: (batch, seq_len, num_stocks, num_features)
                batch_size = x.shape[0]
                # Flatten appropriate dimensions for the linear layer
                # Example: flatten seq_len, num_stocks, num_features
                x_flat = x.view(batch_size, -1) 
                out_flat = self.fc(x_flat)
                # Reshape to (batch_size, pred_len=1, num_stocks=2)
                return out_flat.view(batch_size, 1, 2)

        # Determine input/output sizes for DummyTrainableModel based on data
        # X_train shape: (samples, seq_len, num_stocks, num_features) -> (16, 5, 2, 3)
        # y_train shape: (samples, pred_len, num_stocks) -> (16, 1, 2)
        input_dim_flat = 5 * 2 * 3 # seq_len * num_stocks_data * num_features_data
        output_dim_flat = 1 * 2    # pred_len * num_stocks_data

        actual_model_instance = DummyTrainableModel(input_dim_flat, output_dim_flat).to(device_eval_simple)
        
        # Configure the mocked StockLSTM class to return our actual_model_instance
        mock_stock_lstm_class.return_value = actual_model_instance

        # 3. Mock DataLoader instantiation and iteration
        X_train_tensor = torch.tensor(sample_scaled_data_opt_simple['X_train'], device=device_eval_simple)
        y_train_tensor = torch.tensor(sample_scaled_data_opt_simple['y_train'], device=device_eval_simple)
        X_test_tensor = torch.tensor(sample_scaled_data_opt_simple['X_test'], device=device_eval_simple)
        y_test_tensor = torch.tensor(sample_scaled_data_opt_simple['y_test'], device=device_eval_simple)

        mock_train_loader_inst = MagicMock()
        # DataLoader with batch_size=16 and 16 samples will yield one batch of 16.
        mock_train_loader_inst.__iter__.return_value = iter([(X_train_tensor, y_train_tensor)])
        type(mock_train_loader_inst).__len__ = MagicMock(return_value=1)

        mock_test_loader_inst = MagicMock()
        # DataLoader with batch_size=16 and 4 test samples will yield one batch of 4.
        mock_test_loader_inst.__iter__.return_value = iter([(X_test_tensor, y_test_tensor)])
        type(mock_test_loader_inst).__len__ = MagicMock(return_value=1)

        mock_data_loader_class.side_effect = [mock_train_loader_inst, mock_test_loader_inst]
        
        # 4. Mock evaluate_model
        mock_evaluate_model.return_value = (None, None, {'test_loss': 0.75})

        # 5. Mock Adam optimizer
        mock_optimizer_inst = MagicMock(spec=torch.optim.Adam)
        mock_adam_class.return_value = mock_optimizer_inst

        # Call the objective function
        loss = optimize_hyperparams.objective(
            mock_trial_simple,
            sample_scaled_data_opt_simple['X_train'], sample_scaled_data_opt_simple['y_train'],
            sample_scaled_data_opt_simple['X_test'], sample_scaled_data_opt_simple['y_test'],
            num_features=3, num_stocks=2, y_scalers=mock_y_scalers_opt_simple,
            device=device_eval_simple,
            optimization_epochs=mock_opt_params_config_simple['epochs'], # Should be 1
            patience=mock_opt_params_config_simple['patience']
        )

        # Assertions
        mock_stock_lstm_class.assert_called_once()
        assert mock_data_loader_class.call_count == 2
        mock_adam_class.assert_called_once_with(actual_model_instance.parameters(), lr=0.001)
        mock_optimizer_inst.step.assert_called_once() 
        mock_evaluate_model.assert_called_once()
        assert loss == pytest.approx(0.75)

# --- Tests for run_optimization ---
class TestRunOptimizationSimple:
    @patch('models.optimize_hyperparams.json.dump')
    @patch('models.optimize_hyperparams.Path') 
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
        mock_path_class, mock_json_dump,
        mock_params_config_opt_simple, sample_scaled_data_opt_simple,
        mock_y_scalers_opt_simple, tmp_path
    ):
        run_id = "opt_run_simple_001"
        dummy_config_path = str(tmp_path / "params_opt_simple.yaml")

        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt_simple

        mock_load_scaled.side_effect = lambda db, rid, name: sample_scaled_data_opt_simple.get(name)
        mock_load_scalers.return_value = {
            'y_scalers': mock_y_scalers_opt_simple,
            'tickers': ['S1', 'S2'],
            'num_features': 3
        }

        mock_study_inst = MagicMock(spec=optuna.study.Study)
        mock_study_inst.best_params = {'param': 'value'}
        mock_study_inst.best_value = 0.1
        mock_study_inst.optimize = MagicMock() # This will be called with the objective function
        mock_create_study.return_value = mock_study_inst
        
        # Configure how Path objects are mocked
        mock_json_file_path_instance = MagicMock(spec=Path)
        mock_json_file_path_instance.parent.mkdir = MagicMock()

        def path_constructor_side_effect(path_arg_str):
            if str(path_arg_str) == mock_params_config_opt_simple['output_paths']['best_params_path']:
                return mock_json_file_path_instance
            return Path(path_arg_str) # Fallback for other Path uses
        mock_path_class.side_effect = path_constructor_side_effect
        
        best_params, out_run_id = optimize_hyperparams.run_optimization(dummy_config_path, run_id)

        mock_open.assert_any_call(dummy_config_path, 'r') 
        mock_yaml_safe_load.assert_called_once_with(ANY) 
        
        assert mock_load_scaled.call_count == 4 # X_train, y_train, X_test, y_test
        mock_load_scalers.assert_called_once()
        mock_create_study.assert_called_once()
        mock_study_inst.optimize.assert_called_once()
        mock_save_opt_results.assert_called_once()
        
        mock_json_file_path_instance.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_open.assert_any_call(mock_json_file_path_instance, 'w')
        mock_json_dump.assert_called_once()

        assert best_params == {'param': 'value'}
        assert out_run_id == run_id

    @patch('models.optimize_hyperparams.load_scalers') 
    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.open')
    @patch('yaml.safe_load')
    def test_run_optimization_load_data_fails(
        self, mock_yaml_safe_load, mock_open, mock_load_scaled,
        mock_load_scalers_not_used, # Renamed as it won't be used in this path
        mock_params_config_opt_simple, tmp_path, caplog
    ):
        run_id = "opt_run_fail_load"
        dummy_config_path = str(tmp_path / "cfg_fail.yaml")
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt_simple

        mock_load_scaled.return_value = None 

        best_params, out_run_id = optimize_hyperparams.run_optimization(dummy_config_path, run_id)

        assert best_params is None
        assert out_run_id is None
        assert "Failed to load scaled training data" in caplog.text
        assert mock_load_scaled.call_count == 4
        mock_load_scalers_not_used.assert_not_called()