# tests/models/test_optimize_hyperparams.py
import pytest
from unittest.mock import patch, MagicMock, call, ANY
import torch
import torch.nn as nn
import numpy as np
import yaml
import json
from pathlib import Path
import optuna 
from sklearn.preprocessing import MinMaxScaler
import io

from models.optimize_hyperparams import objective, run_optimization


# --- Fixtures ---
@pytest.fixture
def mock_db_config_opt():
    return {'dbname': 'test_opt_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_opt_params_config():
    return {
        'n_trials': 2, 
        'epochs': 3,   
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
        'X_train': np.random.rand(20, 5, 2, 3).astype(np.float32),
        'y_train': np.random.rand(20, 1, 2).astype(np.float32),
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
    @patch('models.optimize_hyperparams.StockLSTM') 
    @patch('models.optimize_hyperparams.StockLSTMWithAttention')
    @patch('models.optimize_hyperparams.StockLSTMWithCrossStockAttention')
    @patch('models.optimize_hyperparams.evaluate_model')
    @patch('models.optimize_hyperparams.torch.utils.data.DataLoader') 
    def test_objective_calls_and_returns_loss(self, mock_data_loader, mock_evaluate_model,
                                              mock_lstm_cross, mock_lstm_att, mock_lstm_basic,
                                              mock_trial, sample_scaled_data_opt,
                                              mock_y_scalers_opt, device_eval, mock_opt_params_config):

        mock_model_instance = MagicMock(spec=nn.Module)
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.return_value = torch.rand(32, 1, 2) 
        # --- MODIFIED: Configure parameters() for the optimizer ---
        mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.randn(10,10))] 
        mock_lstm_basic.return_value = mock_model_instance

        dummy_X_batch = torch.rand(32, 5, 2, 3)
        dummy_y_batch = torch.rand(32, 1, 2)
        mock_data_loader.return_value = [(dummy_X_batch, dummy_y_batch)] 

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
        mock_lstm_basic.assert_called_once() 
        mock_model_instance.to.assert_called_with(device_eval)
        assert mock_data_loader.call_count == 2 
        assert mock_model_instance.call_count >= mock_opt_params_config['epochs']
        mock_evaluate_model.assert_called() 
        assert loss == 0.5 

# --- Tests for run_optimization ---
class TestRunOptimization:
    @patch('models.optimize_hyperparams.json.dump') 
    @patch('models.optimize_hyperparams.Path.mkdir') 
    @patch('models.optimize_hyperparams.save_optimization_results')
    @patch('models.optimize_hyperparams.optuna.create_study')
    @patch('models.optimize_hyperparams.load_scalers')
    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.open') # --- MODIFIED: Patch module's open ---
    @patch('yaml.safe_load')
    def test_run_optimization_success(self, mock_yaml_safe_load, mock_open, # Args order
                                      mock_load_scaled, mock_load_scalers, 
                                      mock_create_study, mock_save_opt_results, 
                                      mock_path_mkdir, mock_json_dump, # Reversed order of mkdir and json_dump
                                      mock_params_config_opt, sample_scaled_data_opt,
                                      mock_y_scalers_opt, tmp_path):
        run_id = "opt_run_001"
        dummy_config_path = str(tmp_path / "params_opt.yaml") 

        # --- MODIFIED: Configure open and yaml mocks ---
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt
        
        mock_load_scaled.side_effect = lambda db_cfg, r_id, set_name: sample_scaled_data_opt[set_name]
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_opt, 'tickers': ['T1','T2'], 'num_features':3}

        mock_study_instance = MagicMock(spec=optuna.study.Study) # Use spec for better mocking
        mock_study_instance.best_params = {'lr': 0.001, 'hidden_size': 128}
        mock_study_instance.best_value = 0.05
        mock_study_instance.optimize = MagicMock()
        mock_create_study.return_value = mock_study_instance

        best_params_out, run_id_out = run_optimization(dummy_config_path, run_id)

        mock_open.assert_called_once_with(dummy_config_path, 'r')
        mock_yaml_safe_load.assert_called_once_with(mock_open.return_value.__enter__.return_value)
        mock_load_scaled.assert_any_call(mock_params_config_opt['database'], run_id, 'X_train')
        mock_load_scalers.assert_called_once_with(mock_params_config_opt['database'], run_id)
        mock_create_study.assert_called_once_with(direction="minimize", pruner=ANY) 
        mock_study_instance.optimize.assert_called_once() 
        mock_save_opt_results.assert_called_once_with(
            mock_params_config_opt['database'], run_id, mock_study_instance.best_params
        )
        
        # Assert that parent directory of best_params_path was created
        expected_json_parent_dir = Path(mock_params_config_opt['output_paths']['best_params_path']).parent
        mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True) # Checks if mkdir on parent was called
        # Assert json.dump was called (mock_open for json is separate from config open)
        # We need to patch the open used by json.dump if we want to check its file handle.
        # For simplicity, we check it was called with the right data.
        # The file handle `f` in `with open(...) as f: json.dump` will be a mock from a different patch
        # if we were to patch 'models.optimize_hyperparams.open' globally for json.dump too.
        # To keep it simpler: assume json.dump's open works or patch it specifically for that context if needed.
        # Here we assert it was called with best_params and ANY for the file object.
        mock_json_dump.assert_called_once_with(mock_study_instance.best_params, ANY, indent=4)

        assert best_params_out == mock_study_instance.best_params
        assert run_id_out == run_id

    @patch('models.optimize_hyperparams.load_scalers') # To ensure it's mocked if called
    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.open') # --- MODIFIED ---
    @patch('yaml.safe_load')
    def test_run_optimization_load_scaled_fails(self, mock_yaml_safe_load, mock_open, # Args order
                                                mock_load_scaled, mock_load_scalers_not_used, # Renamed
                                                mock_params_config_opt, tmp_path, caplog):
        run_id = "opt_fail_load_scaled"
        dummy_config_path = str(tmp_path / "cfg.yaml")
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt
        
        mock_load_scaled.return_value = None 
        # mock_load_scalers_not_used doesn't need specific config as it shouldn't be reached.

        best_params_out, run_id_out = run_optimization(dummy_config_path, run_id)
        
        assert best_params_out is None
        assert run_id_out is None
        assert "Failed to load scaled training data" in caplog.text
        mock_load_scaled.assert_called_once() # Ensure it was called
        mock_load_scalers_not_used.assert_not_called() # Should not be called

    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.load_scalers')
    @patch('models.optimize_hyperparams.open') # --- MODIFIED ---
    @patch('yaml.safe_load')
    def test_run_optimization_load_scalers_fails(self, mock_yaml_safe_load, mock_open, # Args order
                                                 mock_load_scalers, mock_load_scaled,
                                                 mock_params_config_opt, sample_scaled_data_opt,
                                                 tmp_path, caplog):
        run_id = "opt_fail_load_scalers"
        dummy_config_path = str(tmp_path / "cfg.yaml")
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt

        mock_load_scaled.side_effect = lambda db_cfg, r_id, set_name: sample_scaled_data_opt[set_name]
        mock_load_scalers.return_value = None 

        best_params_out, run_id_out = run_optimization(dummy_config_path, run_id)

        assert best_params_out is None
        assert run_id_out is None
        assert "Failed to load scalers or 'y_scalers' not found" in caplog.text
        mock_load_scaled.assert_called() # Called multiple times
        mock_load_scalers.assert_called_once()


    @patch('models.optimize_hyperparams.load_scaled_features')
    @patch('models.optimize_hyperparams.load_scalers')
    @patch('models.optimize_hyperparams.open') # --- MODIFIED ---
    @patch('yaml.safe_load')
    def test_run_optimization_x_train_bad_dims(self, mock_yaml_safe_load, mock_open, # Args order
                                               mock_load_scalers, mock_load_scaled,
                                               mock_params_config_opt, mock_y_scalers_opt,
                                               tmp_path, caplog):
        run_id = "opt_fail_dims"
        dummy_config_path = str(tmp_path / "cfg.yaml")
        mock_open.return_value.__enter__.return_value = MagicMock(spec=io.TextIOBase)
        mock_yaml_safe_load.return_value = mock_params_config_opt

        bad_X_train = np.random.rand(20, 5, 2).astype(np.float32) # 3D instead of 4D
        mock_load_scaled.side_effect = lambda db_cfg, r_id, set_name: \
            bad_X_train if set_name == 'X_train' else np.random.rand(20,1,2).astype(np.float32)
        
        # Configure load_scalers to return valid data so the flow reaches the dimension check
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_opt, 'tickers':['T1','T2'], 'num_features':2}


        best_params_out, run_id_out = run_optimization(dummy_config_path, run_id)
        
        assert best_params_out is None
        assert run_id_out is None
        assert "X_train_scaled has unexpected dimensions" in caplog.text
        mock_load_scaled.assert_called()
        mock_load_scalers.assert_called_once() # Should be called before the dim check