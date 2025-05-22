# tests/models/test_predict_model.py
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call, ANY # Import ANY
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

@pytest.fixture
def mock_db_config_pred():
    return {'dbname': 'test_pred_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_mlflow_config_pred():
    return {'experiment_name': 'TestPredictionExperiment', 
            'tracking_uri': 'mock_mlflow_uri_pred', 
            'model_name': 'MyModel'} # Added model_name based on usage

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
    return np.random.rand(1, 5, 2, 3).astype(np.float32) # (batch, seq, stocks, features)

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
    @patch('models.predict_model.Path') # Mock Path class for mkdir
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.date')
    def test_run_daily_prediction_success(self, mock_date, mock_save_pred, mock_json_dump, MockPath, # Changed mock_path_mkdir
                                          mock_torch, mock_np_load, mock_load_proc_meta, mock_load_scalers,
                                          mock_mlflow_client_class, mock_mlflow_module,
                                          mock_yaml_safe_load,
                                          mock_params_config_pred, sample_input_sequence_np,
                                          mock_y_scalers_pred, mock_tickers_pred,
                                          mock_prod_model_train_dataset_run_id,
                                          mock_mlflow_client_pred, tmp_path):

        config_file = tmp_path / "params_pred.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        input_seq_path = tmp_path / "input_seq.npy"
        # *** ADDED: Create the dummy input sequence file ***
        np.save(input_seq_path, sample_input_sequence_np)

        prod_model_uri = f"models:/{mock_params_config_pred['mlflow']['model_name']}@Production"
        prod_model_name = mock_params_config_pred['mlflow']['model_name']
        prod_model_version = "Production"

        # Configure Path mock
        mock_path_instance = MockPath.return_value
        mock_path_instance.exists.return_value = True # For input_sequence_path.exists()
        mock_path_instance.parent.mkdir = MagicMock() # For json_file_path.parent.mkdir()
        mock_path_instance.mkdir = MagicMock() # For predictions_output_dir.mkdir() and historical_dir.mkdir()
        
        # Make Path act like a constructor but also allow attribute access like .parent
        def path_side_effect(path_arg):
            path_obj = MagicMock(spec=Path) # Create a new mock for each Path() call
            path_obj.mkdir = MagicMock()
            path_obj.parent = MagicMock(spec=Path)
            path_obj.parent.mkdir = MagicMock()
            path_obj.exists.return_value = True # Default to exists = True for simplicity here
            path_obj.resolve.return_value = path_obj # for .resolve()
            path_obj.__str__.return_value = str(path_arg) # Make it stringable
            if str(path_arg) == str(input_seq_path):
                 path_obj.exists.return_value = True # Ensure our target input file exists
            return path_obj
        MockPath.side_effect = path_side_effect


        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred

        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.eval = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32)
        mock_mlflow_module.pytorch.load_model.return_value = mock_model_instance

        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        mock_np_load.return_value = sample_input_sequence_np

        mock_torch.tensor.side_effect = lambda x, dtype, device=None: torch.from_numpy(x).to(dtype=dtype).to(device) if isinstance(x, np.ndarray) else torch.tensor(x, dtype=dtype, device=device)
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.isnan.return_value.any.return_value = False
        mock_torch.device.return_value = torch.device('cpu') # Mock device creation

        fixed_today = date(2023, 1, 11)
        mock_date.today.return_value = fixed_today
        today_iso = fixed_today.isoformat()

        success = predict_model.run_daily_prediction(
            str(config_file), str(input_seq_path), prod_model_uri, prod_model_name, prod_model_version
        )

        assert success is True
        mock_yaml_safe_load.assert_called_once()
        mock_mlflow_module.set_tracking_uri.assert_called_once_with(mock_params_config_pred['mlflow']['tracking_uri'])
        mock_mlflow_module.pytorch.load_model.assert_called_once_with(model_uri=prod_model_uri)
        
        model_mlflow_run_id = mock_mlflow_client_pred.get_model_version.return_value.run_id
        mock_load_scalers.assert_called_once_with(mock_params_config_pred['database'], mock_prod_model_train_dataset_run_id)
        
        mock_np_load.assert_called_once_with(MockPath(input_seq_path)) # np.load is called with the Path object
        
        mock_model_instance.assert_called_once_with(ANY) # Input tensor

        # Check mkdir calls on the mocked Path instances
        # Assert that Path(...).mkdir was called. The number of calls depends on logic.
        # Example: one for predictions_dir, one for historical_dir
        assert mock_path_instance.mkdir.call_count >= 1 # Check if any mkdir was called on Path instances
        assert mock_path_instance.parent.mkdir.call_count >= 1 # For parent of JSON files
        
        assert mock_json_dump.call_count == 2
        assert mock_save_pred.call_count == len(mock_tickers_pred)


    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.Path') # Mock Path
    def test_run_daily_prediction_load_model_fails(self, MockPath, mock_mlflow_module, mock_yaml_safe_load,
                                                   mock_params_config_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_lm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        MockPath.return_value.exists.return_value = True # Ensure config file "exists" for open

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
    @patch('models.predict_model.Path') # Mock Path
    def test_run_daily_prediction_get_mlflow_metadata_fails(self, MockPath, mock_mlflow_client_class, mock_mlflow_module,
                                                            mock_yaml_safe_load, mock_params_config_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_meta_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        MockPath.return_value.exists.return_value = True

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value.get_model_version.side_effect = Exception("MLflow metadata error")

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Error fetching metadata for production model" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.Path') # Mock Path
    def test_run_daily_prediction_load_scalers_fails(self, MockPath, mock_load_scalers, mock_mlflow_client_class,
                                                     mock_mlflow_module, mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_ls_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        MockPath.return_value.exists.return_value = True
        
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
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
    @patch('models.predict_model.Path') # Mock Path
    def test_run_daily_prediction_load_tickers_fails(self, MockPath, mock_load_proc_meta, mock_load_scalers,
                                                     mock_mlflow_client_class, mock_mlflow_module,
                                                     mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_lt_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        MockPath.return_value.exists.return_value = True

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
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
    @patch('models.predict_model.np.load') # np.load directly mocked
    @patch('models.predict_model.Path') # Mock Path class
    def test_run_daily_prediction_input_file_missing(self, MockPath, mock_np_load, mock_load_scalers,
                                                     mock_mlflow_client_class, mock_mlflow_module,
                                                     mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, mock_y_scalers_pred, mock_tickers_pred,
                                                     tmp_path, caplog):
        config_file = tmp_path/"cfg_ifm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        # Simulate Path(input_seq_path).exists() -> False
        input_seq_path_str = str(tmp_path / "non_existent_input.npy")
        
        def path_side_effect_ifm(p_arg):
            path_obj = MagicMock(spec=Path)
            path_obj.__str__.return_value = str(p_arg)
            path_obj.resolve.return_value = path_obj
            if str(p_arg) == input_seq_path_str:
                path_obj.exists.return_value = False # Key: input file does not exist
            else:
                path_obj.exists.return_value = True # Other paths (like config) exist
            return path_obj
        MockPath.side_effect = path_side_effect_ifm

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        
        # np.load won't be called if path.exists() is false.
        # mock_np_load.side_effect = FileNotFoundError("Simulated file not found for np.load") 

        success = predict_model.run_daily_prediction(
            str(config_file), input_seq_path_str, "uri", "name", "ver"
        )
        assert success is False
        # Path.resolve() is called inside the SUT, so use the string path.
        assert f"Input sequence file not found: {input_seq_path_str}" in caplog.text


    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.torch')
    @patch('models.predict_model.Path') # Mock Path
    def test_run_daily_prediction_input_dimension_mismatch(self, MockPath, mock_torch, mock_np_load,
                                                           mock_load_scalers, mock_mlflow_client_class, mock_mlflow_module,
                                                           mock_yaml_safe_load, mock_params_config_pred,
                                                           mock_mlflow_client_pred, mock_y_scalers_pred, mock_tickers_pred,
                                                           tmp_path, caplog):
        config_file = tmp_path/"cfg_idm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        input_seq_path_for_dim_mismatch = tmp_path/"input_dim_mismatch.npy"
        # Input sequence with 3 stocks, while mock_tickers_pred implies 2
        mismatched_input_seq = np.random.rand(1, 5, 3, 3).astype(np.float32)
        np.save(input_seq_path_for_dim_mismatch, mismatched_input_seq) # Create the file

        # Configure Path mock
        def path_side_effect_idm(path_arg):
            path_obj = MagicMock(spec=Path)
            path_obj.__str__.return_value = str(path_arg)
            path_obj.resolve.return_value = path_obj
            path_obj.exists.return_value = True # All paths exist for this test
            return path_obj
        MockPath.side_effect = path_side_effect_idm

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred} 
        
        mock_np_load.return_value = mismatched_input_seq # np.load will return this
        mock_torch.tensor.side_effect = lambda x, dtype, device=None: torch.from_numpy(x).to(dtype=dtype).to(device) if isinstance(x, np.ndarray) else torch.tensor(x, dtype=dtype, device=device)
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.device.return_value = torch.device('cpu')


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
    @patch('models.predict_model.Path') # Mock Path class
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.date')
    def test_run_daily_prediction_save_prediction_fails(self, mock_date, mock_save_pred, mock_json_dump, MockPath,
                                                        mock_torch, mock_np_load, mock_load_scalers,
                                                        mock_mlflow_client_class, mock_mlflow_module,
                                                        mock_yaml_safe_load,
                                                        mock_params_config_pred, sample_input_sequence_np,
                                                        mock_y_scalers_pred, mock_tickers_pred,
                                                        mock_prod_model_train_dataset_run_id,
                                                        mock_mlflow_client_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_sp_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        input_seq_path_for_save_fail = tmp_path/"input_save_fail.npy"
        np.save(input_seq_path_for_save_fail, sample_input_sequence_np) # Create file

        # Configure Path mock
        def path_side_effect_spf(path_arg):
            path_obj = MagicMock(spec=Path)
            path_obj.__str__.return_value = str(path_arg)
            path_obj.mkdir = MagicMock()
            path_obj.parent = MagicMock(spec=Path)
            path_obj.parent.mkdir = MagicMock()
            path_obj.exists.return_value = True
            path_obj.resolve.return_value = path_obj
            return path_obj
        MockPath.side_effect = path_side_effect_spf

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.eval = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32)
        mock_mlflow_module.pytorch.load_model.return_value = mock_model_instance
        
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        mock_np_load.return_value = sample_input_sequence_np # np.load returns this
        mock_torch.tensor.side_effect = lambda x, dtype, device=None: torch.from_numpy(x).to(dtype=dtype).to(device) if isinstance(x, np.ndarray) else torch.tensor(x, dtype=dtype, device=device)
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.isnan.return_value.any.return_value = False
        mock_torch.device.return_value = torch.device('cpu')
        
        fixed_today = date(2023, 1, 11)
        mock_date.today.return_value = fixed_today

        mock_save_pred.side_effect = Exception("DB Save Error for TICKA")

        success = predict_model.run_daily_prediction(
            str(config_file), str(input_seq_path_for_save_fail), "uri", "name", "ver"
        )
        assert success is False
        assert "Error saving prediction for TICKA" in caplog.text
        assert "Error in run_daily_prediction: Error saving prediction for TICKA" in caplog.text