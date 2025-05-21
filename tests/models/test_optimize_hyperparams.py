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
        'n_trials': 1, # Reduced for faster test, was 2
        'epochs': 1,   # Reduced for faster test, was 3
        'patience': 2, # Keep patience
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
    trial.suggest_int.side_effect = [32, 64, 2] 
    trial.suggest_float.side_effect = [0.3, 0.001] 
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
    @patch('models.optimize_hyperparams.torch.utils.data.DataLoader') 
    def test_objective_calls_and_returns_loss(self, mock_data_loader_class, mock_evaluate_model,
                                              mock_stock_lstm_class, mock_stock_lstm_att_class, mock_stock_lstm_cross_class,
                                              mock_trial, sample_scaled_data_opt,
                                              mock_y_scalers_opt, device_eval, mock_opt_params_config):

        # --- MODIFIED: Mock the model instance correctly for backward() ---
        # Create a real, simple model instance for the test to ensure .parameters() works
        # and its output can be part of a computation graph.
        class SimpleDummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3*5*2, 1*2) # Input_features_flat, Output_features_flat
            def forward(self, x): # x: (batch, seq, stocks, features)
                batch_size = x.shape[0]
                # A dummy operation that uses a parameter and produces the right output shape
                # This is just to ensure the output has a grad_fn.
                # Flatten all but batch dim for linear layer, then reshape to (batch, pred_len, stocks)
                x_flat = x.view(batch_size, -1)
                out_flat = self.linear(x_flat)
                return out_flat.view(batch_size, 1, 2) # Assuming pred_len=1, num_stocks=2

        actual_model_instance = SimpleDummyModel().to(device_eval)
        
        # Configure the mocked model CLASS (e.g., StockLSTM) to return our actual_model_instance
        mock_stock_lstm_class.return_value = actual_model_instance
        # --- END MODIFIED ---

        # DataLoader setup: Objective creates its own DataLoaders.
        # We mock the DataLoader class to control what its instances yield.
        # Batch size is 32 from mock_trial. X_train has 32 samples.
        X_train_tensor = torch.tensor(sample_scaled_data_opt['X_train'], device=device_eval)
        y_train_tensor = torch.tensor(sample_scaled_data_opt['y_train'], device=device_eval)
        X_test_tensor = torch.tensor(sample_scaled_data_opt['X_test'], device=device_eval)
        y_test_tensor = torch.tensor(sample_scaled_data_opt['y_test'], device=device_eval)

        # Mock what DataLoader instances will do when iterated
        mock_train_loader_instance = iter([(X_train_tensor, y_train_tensor)]) # Single batch for train
        mock_test_loader_instance = iter([(X_test_tensor, y_test_tensor)])   # Single batch for test

        mock_data_loader_class.side_effect = [
            mock_train_loader_instance, 
            mock_test_loader_instance
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
        
        # Check that the correct model class was instantiated
        mock_stock_lstm_class.assert_called_once() 
        
        assert mock_data_loader_class.call_count == 2 
        
        # Check that actual_model_instance's forward pass was called.
        # Its forward method is not directly a MagicMock, but part of a real nn.Module.
        # We can check if evaluate_model was called, which implies model was run.
        mock_evaluate_model.assert_called() 
        
        assert loss == pytest.approx(0.5) # Use approx for float comparison

# --- Tests for run_optimization ---
class TestRunOptimization:
    @patch('models.optimize_hyperparams.json.dump') 
    @patch('models.optimize_hyperparams.Path.mkdir') 
    @patch('models.optimize_hyperparams.save_optimization_results')
    @patch('models.optimize_hyperparams.optuna.create_study')
    @patch('models.optimize_hyperparams.load_scalers')
    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.open') 
    @patch('yaml.safe_load')
    def test_run_optimization_success(self, mock_yaml_safe_load, mock_open, 
                                      mock_load_scaled, mock_load_scalers, 
                                      mock_create_study, mock_save_opt_results, 
                                      mock_path_mkdir_constructor, mock_json_dump, # path_mkdir is for Path(...).mkdir
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

        # Mock the Path(...).parent.mkdir call
        # mock_path_mkdir_constructor refers to the `Path` class constructor when `Path(output_paths_config['best_params_path'])` is called.
        # We need to mock the `mkdir` method of the `Path` *instance*.
        mock_parent_path_instance = MagicMock(spec=Path)
        mock_parent_path_instance.mkdir = MagicMock() # This is what `mock_path_mkdir` will now effectively be

        # This is a bit tricky. We need to mock `Path(...).parent` to return our mock_parent_path_instance.
        # The patch for `models.optimize_hyperparams.Path.mkdir` means `Path.mkdir` itself is a mock.
        # So when `best_params_output_path.parent.mkdir` is called, `best_params_output_path.parent` is a real Path,
        # and its `.mkdir` is our `mock_path_mkdir_constructor`.

        with patch('models.optimize_hyperparams.Path') as mock_path_class_overall:
            # Configure the Path class mock for the specific instantiation
            mock_path_instance_for_json = MagicMock(spec=Path)
            mock_path_parent_instance_for_json = MagicMock(spec=Path)
            mock_path_instance_for_json.parent = mock_path_parent_instance_for_json
            
            # If Path() is called with the json path, return our configured instance
            def path_side_effect(path_arg):
                if str(path_arg) == mock_params_config_opt['output_paths']['best_params_path']:
                    return mock_path_instance_for_json
                return Path(path_arg) # Default to real Path for other uses
            mock_path_class_overall.side_effect = path_side_effect

            best_params_out, run_id_out = run_optimization(dummy_config_path, run_id)

            # Assert that mkdir was called on the parent of the json path
            mock_path_parent_instance_for_json.mkdir.assert_called_once_with(parents=True, exist_ok=True)


        mock_open.assert_any_call(dummy_config_path, 'r') 
        mock_open.assert_any_call(Path(mock_params_config_opt['output_paths']['best_params_path']), 'w')
        
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
                                                mock_load_scaled, mock_load_scalers_used, 
                                                mock_params_config_opt, tmp_path, caplog):
        run_id = "opt_fail_load_scaled"
        dummy_config_path = str(tmp_path / "cfg.yaml")
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt
        
        mock_load_scaled.return_value = None # All calls to load_scaled_features return None
        
        best_params_out, run_id_out = run_optimization(dummy_config_path, run_id)
        
        assert best_params_out is None
        assert run_id_out is None
        assert "Failed to load scaled training data (X_train or y_train)" in caplog.text
        # --- MODIFIED: load_scaled_features is called 4 times before the check ---
        assert mock_load_scaled.call_count == 4 
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
        assert mock_load_scaled.call_count == 4 
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
            # Provide valid data for others so execution proceeds to dimension check
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
        assert mock_load_scaled.call_count >= 1 # Called for X_train at least
        mock_load_scalers.assert_called_once()