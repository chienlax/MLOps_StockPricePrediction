# tests/models/test_predict_model.py
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call, ANY
import numpy as np
import yaml
import json
import torch
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run as MlflowRun
from mlflow.entities.model_registry import ModelVersion as MlflowModelVersion
from sklearn.preprocessing import MinMaxScaler
from datetime import date

PROJECT_ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT_FOR_TESTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models import predict_model

# Universal Path mock factory
def mock_path_factory_universal(path_arg_str_or_mock):
    # Convert argument to a string
    current_path_str = str(path_arg_str_or_mock)

    path_mock = MagicMock(spec=Path)
    
    # Configure methods that return string representations or Path-like objects
    path_mock.__str__ = MagicMock(return_value=current_path_str)
    path_mock.__fspath__ = MagicMock(return_value=current_path_str)
    path_mock.resolve.return_value = path_mock

    # Configure file/directory status methods - default behavior
    path_mock.exists.return_value = True
    path_mock.is_file.return_value = True
    path_mock.is_dir.return_value = True
    
    path_mock.mkdir = MagicMock(return_value=None)

    # Configure parent mock
    if current_path_str == str(Path(current_path_str).parent):
        mock_parent = path_mock  # Parent of root is root
    else:
        parent_str = str(Path(current_path_str).parent)
        mock_parent = MagicMock(spec=Path)
        mock_parent.__str__ = MagicMock(return_value=parent_str)
        mock_parent.__fspath__ = MagicMock(return_value=parent_str)
        mock_parent.mkdir = MagicMock(return_value=None)
        mock_parent.exists.return_value = True
    path_mock.parent = mock_parent
    
    # Configure the __truediv__ method (for the / operator)
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
    return {'predictions_dir': str(tmp_path / 'test_predictions')}

@pytest.fixture
def mock_params_config_pred(mock_db_config_pred, mock_mlflow_config_pred, mock_output_paths_pred):
    return {
        'database': mock_db_config_pred,
        'mlflow': mock_mlflow_config_pred,
        'output_paths': mock_output_paths_pred
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
    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db')
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.torch')
    @patch('models.predict_model.Path') 
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.date')
    def test_run_daily_prediction_success(self, mock_date, mock_save_pred, mock_json_dump, MockPath,
                                          mock_torch, mock_np_load, mock_load_proc_meta, mock_load_scalers,
                                          MockMlflowClientClass, mock_mlflow_module,
                                          mock_yaml_safe_load,
                                          mock_params_config_pred, sample_input_sequence_np,
                                          mock_y_scalers_pred, mock_tickers_pred,
                                          mock_prod_model_train_dataset_run_id,
                                          mock_mlflow_client_pred, tmp_path):

        # Use the universal path factory instead of custom side effect
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

        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.eval = MagicMock()
        raw_model_output = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32)
        mock_model_instance.return_value = raw_model_output
        mock_mlflow_module.pytorch.load_model.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance 

        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        mock_np_load.return_value = sample_input_sequence_np

        real_cpu_device = torch.device('cpu')
        mock_torch.device.return_value = real_cpu_device

        # Ensure that mock_torch.float32 returns the actual torch.float32 type
        mock_torch.float32 = torch.float32
        mock_torch.tensor.side_effect = lambda data, dtype: torch.tensor(data, dtype=dtype)

        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.isnan.return_value.any.return_value = False

        fixed_today = date(2023, 1, 11)
        mock_date.today.return_value = fixed_today
        today_iso = fixed_today.isoformat()

        success = predict_model.run_daily_prediction(
            str(config_file), str(input_seq_path), prod_model_uri, prod_model_name, prod_model_version
        )

        assert success is True
        mock_mlflow_module.pytorch.load_model.assert_called_once_with(model_uri=prod_model_uri)
        
        mock_torch.tensor.assert_called_once_with(sample_input_sequence_np, dtype=torch.float32)
        mock_model_instance.to.assert_called_with(real_cpu_device)
  
        mock_np_load.assert_called_once()
        assert mock_np_load.call_args[0][0].__str__() == str(input_seq_path)


    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.Path') 
    def test_run_daily_prediction_load_model_fails(self, MockPath, mock_mlflow_module, mock_yaml_safe_load,
                                                   mock_params_config_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_lm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        MockPath.return_value.exists.return_value = True 

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.side_effect = Exception("Model load error")

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Failed to load model from uri" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.Path')
    def test_run_daily_prediction_get_mlflow_metadata_fails(self, MockPath, MockMlflowClientClass, mock_mlflow_module,
                                                            mock_yaml_safe_load, mock_params_config_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_meta_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        MockPath.return_value.exists.return_value = True

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        MockMlflowClientClass.return_value.get_model_version.side_effect = Exception("MLflow metadata error")

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Error fetching metadata for production model" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.Path')
    def test_run_daily_prediction_load_scalers_fails(self, MockPath, mock_load_scalers, MockMlflowClientClass,
                                                     mock_mlflow_module, mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, tmp_path, caplog): 
        config_file = tmp_path/"cfg_ls_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        MockPath.return_value.exists.return_value = True
        
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        MockMlflowClientClass.return_value = mock_mlflow_client_pred 
        mock_load_scalers.return_value = None

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        expected_dataset_run_id = mock_mlflow_client_pred.get_run.return_value.data.params["dataset_run_id"]
        assert f"Failed to load y_scalers for dataset_run_id: {expected_dataset_run_id}" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db')
    @patch('models.predict_model.Path')
    def test_run_daily_prediction_load_tickers_fails(self, MockPath, mock_load_proc_meta, mock_load_scalers,
                                                     MockMlflowClientClass, mock_mlflow_module,
                                                     mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_lt_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        MockPath.return_value.exists.return_value = True

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        MockMlflowClientClass.return_value = mock_mlflow_client_pred
        mock_load_scalers.return_value = {'y_scalers': [MagicMock()]} 
        mock_load_proc_meta.return_value = None

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        expected_dataset_run_id = mock_mlflow_client_pred.get_run.return_value.data.params["dataset_run_id"]
        assert f"Failed to load tickers for dataset_run_id: {expected_dataset_run_id}" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.np.load') 
    @patch('models.predict_model.Path') 
    def test_run_daily_prediction_input_file_missing(self, MockPath, mock_np_load, mock_load_scalers,
                                               MockMlflowClientClass, mock_mlflow_module,
                                               mock_yaml_safe_load, mock_params_config_pred,
                                               mock_mlflow_client_pred, mock_y_scalers_pred, mock_tickers_pred,
                                               tmp_path, caplog):
        config_file = tmp_path/"cfg_ifm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        input_seq_path_str = str(tmp_path / "non_existent_input.npy")
        
        # Custom side effect to handle the non-existent input file case
        def custom_path_factory_for_missing_file(path_arg_str_or_mock):
            path_mock = mock_path_factory_universal(path_arg_str_or_mock)
            if str(path_mock) == input_seq_path_str:
                path_mock.exists.return_value = False
            return path_mock
        
        MockPath.side_effect = custom_path_factory_for_missing_file

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

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.torch')
    @patch('models.predict_model.Path') 
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.date')
    def test_run_daily_prediction_input_dimension_mismatch(self, MockPath, mock_torch, mock_np_load,
                                                           mock_load_scalers, MockMlflowClientClass, mock_mlflow_module,
                                                           mock_yaml_safe_load, mock_params_config_pred,
                                                           mock_mlflow_client_pred, mock_y_scalers_pred, mock_tickers_pred,
                                                           tmp_path, caplog):
        config_file = tmp_path/"cfg_idm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        input_seq_path_for_dim_mismatch = tmp_path/"input_dim_mismatch.npy"
        mismatched_input_seq = np.random.rand(1, 5, 3, 3).astype(np.float32)
        np.save(input_seq_path_for_dim_mismatch, mismatched_input_seq)

        def path_side_effect_idm(p_arg):
            path_obj = MagicMock(spec=Path); path_obj.__str__.return_value = str(p_arg)
            path_obj.exists.return_value = True; path_obj.resolve.return_value = path_obj
            return path_obj
        MockPath.side_effect = path_side_effect_idm

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        MockMlflowClientClass.return_value = mock_mlflow_client_pred
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred} 
        
        mock_np_load.return_value = mismatched_input_seq
        mock_torch.device.return_value = torch.device('cpu')
        mock_torch.tensor.side_effect = lambda data, dtype: torch.tensor(data, dtype=dtype) # Use real
        mock_model_instance = mock_mlflow_module.pytorch.load_model.return_value
        mock_model_instance.to.return_value = mock_model_instance


        mock_torch.no_grad.return_value.__enter__.return_value = None

        success = predict_model.run_daily_prediction(
            str(config_file), str(input_seq_path_for_dim_mismatch), "uri", "name", "ver"
        )
        assert success is False
        assert "Input sequence has 3 stocks, model/scalers expect 2." in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient') 
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.torch')
    @patch('models.predict_model.Path') 
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.date')
    def test_run_daily_prediction_save_prediction_fails(self, mock_date, mock_save_pred, mock_json_dump, MockPath,
                                                        mock_torch, mock_np_load, mock_load_scalers,
                                                        MockMlflowClientClass, mock_mlflow_module,
                                                        mock_yaml_safe_load,
                                                        mock_params_config_pred, sample_input_sequence_np,
                                                        mock_y_scalers_pred, mock_tickers_pred,
                                                        mock_prod_model_train_dataset_run_id,
                                                        mock_mlflow_client_pred, tmp_path, caplog):
        # Use the universal path factory
        MockPath.side_effect = mock_path_factory_universal

        config_file = tmp_path/"cfg_sp_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        input_seq_path_for_save_fail = tmp_path/"input_save_fail.npy"
        np.save(input_seq_path_for_save_fail, sample_input_sequence_np)

        mock_yaml_safe_load.return_value = mock_params_config_pred
        MockMlflowClientClass.return_value = mock_mlflow_client_pred
        
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.eval = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32)
        mock_model_instance.to.return_value = mock_model_instance
        mock_mlflow_module.pytorch.load_model.return_value = mock_model_instance
        
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        mock_np_load.return_value = sample_input_sequence_np
        
        mock_torch.device.return_value = torch.device('cpu')
        mock_torch.float32 = torch.float32
        mock_torch.tensor.side_effect = lambda data, dtype: torch.tensor(data, dtype=dtype)
        
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.isnan.return_value.any.return_value = False
        
        fixed_today = date(2023, 1, 11)
        mock_date.today.return_value = fixed_today

        mock_save_pred.side_effect = Exception("DB Save Error for TICKA")

        success = predict_model.run_daily_prediction(
            str(config_file), str(input_seq_path_for_save_fail), "uri", "name", "ver"
        )
        assert success is False
        assert "Error saving prediction for TICKA" in caplog.text 
        assert "Error in run_daily_prediction: Error saving prediction for TICKA" in caplog.text