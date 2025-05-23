# tests/models/test_predict_model.py
import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import (  # Added mock_open for completeness if needed
    ANY,
    MagicMock,
    call,
    mock_open,
    patch,
)

import mlflow
import numpy as np
import pytest
import torch
import yaml
from mlflow.entities import Run as MlflowRun
from mlflow.entities.model_registry import ModelVersion as MlflowModelVersion
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT_FOR_TESTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models import predict_model


# Placed at the module level for clarity
def mock_path_factory_universal(path_arg_str_or_mock):
    current_path_str = str(path_arg_str_or_mock)
    path_mock = MagicMock(spec=Path)
    
    # Basic Path behavior setup
    path_mock.__str__ = MagicMock(return_value=current_path_str)
    path_mock.__fspath__ = MagicMock(return_value=current_path_str)
    path_mock.resolve.return_value = path_mock
    path_mock.exists.return_value = True
    path_mock.is_file.return_value = True
    path_mock.is_dir.return_value = True
    
    # Make mkdir actually create the directory
    def mkdir_side_effect(*args, **kwargs):
        # Actually create the directory when mkdir is called
        real_path = Path(current_path_str)
        real_path.mkdir(*args, **kwargs)
    
    path_mock.mkdir = MagicMock(side_effect=mkdir_side_effect)
    
    # Parent handling (same as before)
    mock_parent = MagicMock(spec=Path)
    if Path(current_path_str) == Path(current_path_str).parent:
        parent_str = current_path_str
    else:
        parent_str = str(Path(current_path_str).parent)
    
    mock_parent.__str__ = MagicMock(return_value=parent_str)
    mock_parent.__fspath__ = MagicMock(return_value=parent_str)
    
    # Also make parent's mkdir create the directory
    mock_parent.mkdir = MagicMock(side_effect=lambda *args, **kwargs: Path(parent_str).mkdir(*args, **kwargs))
    mock_parent.exists.return_value = True
    path_mock.parent = mock_parent
    
    # Path division handling (same as before)
    def truediv_handler(segment_to_join):
        base_path_for_join = str(path_mock) 
        new_combined_path_str = str(Path(base_path_for_join) / segment_to_join)
        return mock_path_factory_universal(new_combined_path_str)

    path_mock.__truediv__ = MagicMock(side_effect=truediv_handler)
    
    return path_mock


@pytest.fixture
def mock_db_config_pred():
    return {'dbname': 'test_pred_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_mlflow_config_pred():
    return {'experiment_name': 'TestPredictionExperiment', 
            'tracking_uri': 'mock_mlflow_uri_pred', 
            'model_name': 'MyModel'}

@pytest.fixture
def mock_output_paths_pred(tmp_path):
    # This fixture provides a string path, which is correct for params config
    return {'predictions_dir': str(tmp_path / 'test_predictions')}

@pytest.fixture
def mock_params_config_pred(mock_db_config_pred, mock_mlflow_config_pred, mock_output_paths_pred):
    return {
        'database': mock_db_config_pred,
        'mlflow': mock_mlflow_config_pred,
        'output_paths': mock_output_paths_pred # This will contain the real tmp_path string
    }

@pytest.fixture
def sample_input_sequence_np():
    return np.random.rand(1, 5, 2, 3).astype(np.float32)

@pytest.fixture
def mock_y_scalers_pred():
    scaler1 = MagicMock(spec=MinMaxScaler)
    scaler2 = MagicMock(spec=MinMaxScaler)
    scaler1.inverse_transform.side_effect = lambda x: x * 100
    scaler2.inverse_transform.side_effect = lambda x: x * 200
    return [scaler1, scaler2]

@pytest.fixture
def mock_tickers_pred():
    return ['TICKA', 'TICKB']

@pytest.fixture
def mock_prod_model_train_dataset_run_id():
    return "prod_data_run_456"

@pytest.fixture
def mock_mlflow_client_pred(mock_prod_model_train_dataset_run_id):
    client = MagicMock(spec=MlflowClient)
    mock_model_version_details = MagicMock(spec=MlflowModelVersion)
    mock_model_version_details.run_id = "prod_model_mlflow_run_789"
    client.get_model_version.return_value = mock_model_version_details

    mock_mlflow_run_details = MagicMock(spec=MlflowRun)
    mock_mlflow_run_details.data.params = {"dataset_run_id": mock_prod_model_train_dataset_run_id}
    client.get_run.return_value = mock_mlflow_run_details
    return client

class TestRunDailyPrediction:
    # Decorators are applied from bottom up. Arguments are passed in reverse order of decorators.
    @patch('models.predict_model.date')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.Path') 
    @patch('models.predict_model.torch')
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.load_processed_features_from_db') # Not used in success, but kept for consistency if logic changes
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.yaml.safe_load')
    def test_run_daily_prediction_success(self, mock_yaml_safe_load, mock_mlflow_module, MockMlflowClientClass,
                                          mock_load_scalers, mock_load_proc_meta, mock_np_load, mock_torch, 
                                          MockPath, mock_json_dump, mock_save_pred, mock_date,
                                          mock_params_config_pred, sample_input_sequence_np, # Fixtures
                                          mock_y_scalers_pred, mock_tickers_pred,
                                          mock_prod_model_train_dataset_run_id,
                                          mock_mlflow_client_pred, tmp_path): # Fixtures

        MockPath.side_effect = mock_path_factory_universal
    
        config_file = tmp_path / "params_pred.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        input_seq_path = tmp_path / "input_seq.npy"
        np.save(input_seq_path, sample_input_sequence_np)

        prod_model_uri = f"models:/{mock_params_config_pred['mlflow']['model_name']}@Production"
        prod_model_name = mock_params_config_pred['mlflow']['model_name']
        prod_model_version = "Production"

        mock_yaml_safe_load.return_value = mock_params_config_pred
        MockMlflowClientClass.return_value = mock_mlflow_client_pred

        real_cpu_device = torch.device('cpu')
        mock_torch.device.return_value = real_cpu_device

        # Create model instance with properly tracked .to() method
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.eval = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32)
        mock_model_instance.to.side_effect = lambda device: mock_model_instance
        mock_mlflow_module.pytorch.load_model.return_value = mock_model_instance

        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        mock_np_load.return_value = sample_input_sequence_np

        real_cpu_device = torch.device('cpu')
        mock_torch.device.return_value = real_cpu_device
        mock_torch.float32 = torch.float32 
        mock_torch.tensor.side_effect = lambda data, dtype: torch.tensor(data, dtype=dtype) 

        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.isnan.return_value.any.return_value = False

        fixed_today = date(2023, 1, 11)
        mock_date.today.return_value = fixed_today

        success = predict_model.run_daily_prediction(
            str(config_file), str(input_seq_path), prod_model_uri, prod_model_name, prod_model_version
        )

        assert success is True
        mock_mlflow_module.pytorch.load_model.assert_called_once_with(model_uri=prod_model_uri)
        
        
        call_args_list = mock_torch.tensor.call_args_list
        assert len(call_args_list) > 0 # Ensure it was called
        first_call_args, first_call_kwargs = call_args_list[0]
        assert np.array_equal(first_call_args[0], sample_input_sequence_np)
        assert first_call_kwargs['dtype'] == torch.float32

        assert mock_model_instance.eval.called
        
        mock_np_load.assert_called_once()
        loaded_path_arg = mock_np_load.call_args[0][0]
        assert str(loaded_path_arg) == str(input_seq_path)

        mock_save_pred.assert_called() # Check it was called
        expected_predictions_dir_str = mock_params_config_pred['output_paths']['predictions_dir']

        assert mock_json_dump.call_count == 2
        # Check first call (latest_predictions.json)
        first_call_data = mock_json_dump.call_args_list[0][0][0]
        assert 'predictions' in first_call_data
        assert 'TICKA' in first_call_data['predictions']
        assert 'TICKB' in first_call_data['predictions']
        assert first_call_data['predictions']['TICKA'] == 50.0 
        assert first_call_data['predictions']['TICKB'] == pytest.approx(120.00000762939453)

        # Check second call (historical/date.json)
        second_call_data = mock_json_dump.call_args_list[1][0][0]
        assert first_call_data == second_call_data  # Both should have identical data

    @patch('models.predict_model.Path') 
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.yaml.safe_load')
    def test_run_daily_prediction_load_model_fails(self, mock_yaml_safe_load, mock_mlflow_module, MockPath,
                                                   mock_params_config_pred, tmp_path, caplog):
        MockPath.side_effect = mock_path_factory_universal # Use the robust factory
        # MockPath.return_value.exists.return_value = True # Factory handles this
        
        config_file = tmp_path/"cfg_lm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.side_effect = Exception("Model load error")

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Failed to load model from uri" in caplog.text


    @patch('models.predict_model.Path')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.yaml.safe_load')
    def test_run_daily_prediction_get_mlflow_metadata_fails(self, mock_yaml_safe_load, mock_mlflow_module, 
                                                            MockMlflowClientClass, MockPath, 
                                                            mock_params_config_pred, tmp_path, caplog):
        MockPath.side_effect = mock_path_factory_universal
        # MockPath.return_value.exists.return_value = True # Factory handles this
        
        config_file = tmp_path/"cfg_meta_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock() # Model loads fine
        MockMlflowClientClass.return_value.get_model_version.side_effect = Exception("MLflow metadata error")

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Error fetching metadata for production model" in caplog.text


    @patch('models.predict_model.Path')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.yaml.safe_load')
    def test_run_daily_prediction_load_scalers_fails(self, mock_yaml_safe_load, mock_mlflow_module, 
                                                     MockMlflowClientClass, mock_load_scalers, MockPath,
                                                     mock_params_config_pred, mock_mlflow_client_pred, 
                                                     tmp_path, caplog): 
        MockPath.side_effect = mock_path_factory_universal
        # MockPath.return_value.exists.return_value = True # Factory handles this
        
        config_file = tmp_path/"cfg_ls_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock() # Model loads
        MockMlflowClientClass.return_value = mock_mlflow_client_pred # MLflow client works
        mock_load_scalers.return_value = None # Scalers fail to load

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        expected_dataset_run_id = mock_mlflow_client_pred.get_run.return_value.data.params["dataset_run_id"]
        assert f"Failed to load y_scalers for dataset_run_id: {expected_dataset_run_id}" in caplog.text


    @patch('models.predict_model.Path')
    @patch('models.predict_model.load_processed_features_from_db') # Mock this
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.yaml.safe_load')
    def test_run_daily_prediction_load_tickers_fails(self, mock_yaml_safe_load, mock_mlflow_module,
                                                     MockMlflowClientClass, mock_load_scalers, 
                                                     mock_load_proc_meta, MockPath, # mock_load_proc_meta is here
                                                     mock_params_config_pred, mock_mlflow_client_pred, 
                                                     tmp_path, caplog):
        MockPath.side_effect = mock_path_factory_universal
        # MockPath.return_value.exists.return_value = True # Factory handles this

        config_file = tmp_path/"cfg_lt_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock() # Model loads
        MockMlflowClientClass.return_value = mock_mlflow_client_pred # Client works
        # Scalers load, but tickers are missing in the returned dict from load_scalers
        mock_load_scalers.return_value = {'y_scalers': [MagicMock()]} 
        # OR tickers are present but load_processed_features_from_db (which also gives tickers) fails
        # The SUT tries load_scalers first for tickers. If not found, then load_processed_features_from_db.
        # Let's assume tickers are not in the dict from load_scalers, then load_processed_features_from_db is called and fails.
        mock_load_proc_meta.return_value = None # This is what gives {'tickers': ...} if load_scalers doesn't.

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        expected_dataset_run_id = mock_mlflow_client_pred.get_run.return_value.data.params["dataset_run_id"]
        # The error message comes from the SUT after trying load_scalers then load_processed_features_from_db
        assert f"Failed to load tickers for dataset_run_id: {expected_dataset_run_id}" in caplog.text


    @patch('models.predict_model.Path') 
    @patch('models.predict_model.np.load') 
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.yaml.safe_load')
    def test_run_daily_prediction_input_file_missing(self, mock_yaml_safe_load, mock_mlflow_module,
                                                     MockMlflowClientClass, mock_load_scalers, 
                                                     mock_np_load, MockPath,
                                                     mock_params_config_pred, mock_mlflow_client_pred, 
                                                     mock_y_scalers_pred, mock_tickers_pred, # Fixtures
                                                     tmp_path, caplog):
        
        input_seq_path_str = str(tmp_path / "non_existent_input.npy") # Real path string
        
        # Special side effect for Path to make one specific file not exist
        def path_side_effect_input_missing(p_arg):
            path_obj = mock_path_factory_universal(p_arg) # Get a standard mock path
            if str(p_arg) == input_seq_path_str:
                path_obj.exists.return_value = False # Override for this specific path
            # else: path_obj.exists.return_value = True # Default from factory is True
            return path_obj
        MockPath.side_effect = path_side_effect_input_missing

        config_file = tmp_path/"cfg_ifm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        MockMlflowClientClass.return_value = mock_mlflow_client_pred
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        
        success = predict_model.run_daily_prediction(
            str(config_file), input_seq_path_str, "uri", "name", "ver"
        )
        assert success is False
        assert f"Input sequence file not found: {input_seq_path_str}" in caplog.text
        mock_np_load.assert_not_called()

    # Patch order: date, save_pred, json.dump, Path, torch, np.load, load_scalers, MlflowClient, mlflow, yaml.safe_load
    # Args reverse: yaml, mlflow, MlflowClient, load_scalers, np.load, torch, Path, json.dump, save_pred, date
    @patch('models.predict_model.date')                      # mock_date_idm
    @patch('models.predict_model.save_prediction')           # mock_save_pred_idm (not used)
    @patch('models.predict_model.json.dump')                 # mock_json_dump_idm (not used)
    @patch('models.predict_model.Path')                      # MockPath_idm
    @patch('models.predict_model.torch')                     # mock_torch_idm
    @patch('models.predict_model.np.load')                   # mock_np_load_idm
    @patch('models.predict_model.load_scalers')              # mock_load_scalers_idm
    @patch('models.predict_model.MlflowClient')              # MockMlflowClientClass_idm
    @patch('models.predict_model.mlflow')                    # mock_mlflow_module_idm
    @patch('models.predict_model.yaml.safe_load')            # mock_yaml_safe_load_idm
    def test_run_daily_prediction_input_dimension_mismatch(self, 
                                                           mock_yaml_safe_load_idm, mock_mlflow_module_idm,
                                                           MockMlflowClientClass_idm, mock_load_scalers_idm, 
                                                           mock_np_load_idm, mock_torch_idm, MockPath_idm,
                                                           mock_json_dump_idm, mock_save_pred_idm, mock_date_idm, # Mocks from patches
                                                           mock_params_config_pred, # Fixture
                                                           mock_mlflow_client_pred, mock_y_scalers_pred, 
                                                           mock_tickers_pred, # Fixtures
                                                           tmp_path, caplog): # Pytest fixtures
        MockPath_idm.side_effect = mock_path_factory_universal

        # mock_params_config_pred IS A FIXTURE, not a mock from a patch.
        # This was the source of the yaml.dump error previously.
        config_file = tmp_path/"cfg_idm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred)) # Should work now
        
        input_seq_path_for_dim_mismatch = tmp_path/"input_dim_mismatch.npy"
        mismatched_input_seq = np.random.rand(1, 5, 3, 3).astype(np.float32) # 3 stocks
        np.save(input_seq_path_for_dim_mismatch, mismatched_input_seq)

        mock_yaml_safe_load_idm.return_value = mock_params_config_pred
        mock_mlflow_module_idm.pytorch.load_model.return_value = MagicMock() # Model loads
        MockMlflowClientClass_idm.return_value = mock_mlflow_client_pred # Client works
        # mock_tickers_pred has 2 tickers. The input has 3.
        mock_load_scalers_idm.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred} 
        
        mock_np_load_idm.return_value = mismatched_input_seq # Load the mismatched sequence
        
        mock_torch_idm.device.return_value = torch.device('cpu')
        mock_torch_idm.float32 = torch.float32
        mock_torch_idm.tensor.side_effect = lambda data, dtype: torch.tensor(data, dtype=dtype)
        mock_model_instance = mock_mlflow_module_idm.pytorch.load_model.return_value
        mock_model_instance.to.return_value = mock_model_instance
        mock_torch_idm.no_grad.return_value.__enter__.return_value = None

        success = predict_model.run_daily_prediction(
            str(config_file), str(input_seq_path_for_dim_mismatch), "uri", "name", "ver"
        )
        assert success is False
        assert "Input sequence has 3 stocks, model/scalers expect 2." in caplog.text

    @patch('models.predict_model.date')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.Path') 
    @patch('models.predict_model.torch')
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.yaml.safe_load')
    def test_run_daily_prediction_save_prediction_fails(self, mock_yaml_safe_load, mock_mlflow_module, 
                                                        MockMlflowClientClass, mock_load_scalers,
                                                        mock_np_load, mock_torch, MockPath,
                                                        mock_json_dump, mock_save_pred, mock_date,
                                                        mock_params_config_pred, sample_input_sequence_np, # Fixtures
                                                        mock_y_scalers_pred, mock_tickers_pred,
                                                        mock_mlflow_client_pred, tmp_path, caplog): # Fixtures
        MockPath.side_effect = mock_path_factory_universal
    
        # Pre-create the predictions directory to avoid file not found error
        predictions_dir = Path(mock_params_config_pred['output_paths']['predictions_dir'])
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = tmp_path/"cfg_sp_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        input_seq_path_for_save_fail = tmp_path/"input_save_fail.npy"
        np.save(input_seq_path_for_save_fail, sample_input_sequence_np)

        mock_yaml_safe_load.return_value = mock_params_config_pred
        MockMlflowClientClass.return_value = mock_mlflow_client_pred
        
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.eval = MagicMock()
        # Configure what the model returns when called directly
        mock_model_instance.return_value = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32)
        # Configure mlflow to return this mock model
        mock_mlflow_module.pytorch.load_model.return_value = mock_model_instance
        # Configure .to() to return the same instance for chaining
        mock_model_instance.to.return_value = mock_model_instance
        
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        mock_np_load.return_value = sample_input_sequence_np
        
        mock_torch.device.return_value = torch.device('cpu')
        mock_torch.float32 = torch.float32
        mock_torch.tensor.side_effect = lambda data, dtype: torch.tensor(data, dtype=dtype)
        
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.isnan.return_value.any.return_value = False
        
        fixed_today = date(2023, 1, 11)
        mock_date.today.return_value = fixed_today

        # This is the key for this test: save_prediction itself throws an error
        mock_save_pred.side_effect = Exception("DB Save Error for TICKA")

        # Execute test
        success = predict_model.run_daily_prediction(
            str(config_file), str(input_seq_path_for_save_fail), "uri", "name", "ver"
        )
        
        assert success is False
        assert "DB Save Error for TICKA" in caplog.text