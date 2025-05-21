# tests/models/test_predict_model.py
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
# import pandas as pd # Not strictly needed here
import yaml
import json
import torch
import mlflow # For spec
from mlflow.tracking import MlflowClient # For spec
from mlflow.entities import Run as MlflowRun # For spec for run details
from mlflow.entities.model_registry import ModelVersion as MlflowModelVersion # For spec
from sklearn.preprocessing import MinMaxScaler # For spec
from datetime import date

# Add src to sys.path - Adjust if your structure is different or use PYTHONPATH
PROJECT_ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT_FOR_TESTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Absolute import after path modification
from models import predict_model
# from utils import db_utils # Not directly used if sub-functions are mocked in predict_model

# --- Fixtures ---
@pytest.fixture
def mock_db_config_pred():
    return {'dbname': 'test_pred_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_mlflow_config_pred():
    return {'experiment_name': 'TestPredictionExperiment', 'tracking_uri': 'mock_mlflow_uri_pred', 'model_name': 'MyModel'}

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
    return "prod_data_run_456" # This is the run_id of the data used to TRAIN the prod model

@pytest.fixture
def mock_mlflow_client_pred(mock_prod_model_train_dataset_run_id): # Pass fixture
    client = MagicMock(spec=MlflowClient)
    
    mock_model_version_details = MagicMock(spec=MlflowModelVersion)
    mock_model_version_details.run_id = "prod_model_mlflow_run_789" # MLflow run_id of the TRAINED MODEL
    client.get_model_version.return_value = mock_model_version_details

    mock_mlflow_run_details = MagicMock(spec=MlflowRun)
    # This is the dataset_run_id logged when the production model was trained
    mock_mlflow_run_details.data.params = {"dataset_run_id": mock_prod_model_train_dataset_run_id}
    client.get_run.return_value = mock_mlflow_run_details
    return client

# --- Tests for run_daily_prediction ---
class TestRunDailyPrediction:
    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db')
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.torch')
    @patch('models.predict_model.Path.mkdir') # Mocking Path(...).mkdir()
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.date')
    def test_run_daily_prediction_success(self, mock_date, mock_save_pred, mock_json_dump, mock_path_mkdir,
                                          mock_torch, mock_np_load, mock_load_proc_meta, mock_load_scalers,
                                          mock_mlflow_client_class, mock_mlflow_module,
                                          mock_yaml_safe_load,
                                          mock_params_config_pred, sample_input_sequence_np,
                                          mock_y_scalers_pred, mock_tickers_pred,
                                          mock_prod_model_train_dataset_run_id, # from fixture
                                          mock_mlflow_client_pred, tmp_path): # from fixture

        config_file = tmp_path / "params_pred.yaml"
        # *** ADDED: Create the dummy config file ***
        config_file.write_text(yaml.dump(mock_params_config_pred))
        
        input_seq_path = tmp_path / "input_seq.npy"
        # These are part of the function arguments, not from config for this test setup
        prod_model_uri = f"models:/{mock_params_config_pred['mlflow']['model_name']}@Production"
        prod_model_name = mock_params_config_pred['mlflow']['model_name']
        prod_model_version = "Production" # Using alias

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred

        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.eval = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32) # Model output
        mock_mlflow_module.pytorch.load_model.return_value = mock_model_instance

        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        # mock_load_proc_meta not called if tickers are in scalers_dict
        mock_np_load.return_value = sample_input_sequence_np

        # Ensure torch.tensor inside SUT receives the numpy array and converts it
        mock_torch.tensor.side_effect = lambda x, dtype: torch.from_numpy(x).to(dtype=dtype) if isinstance(x, np.ndarray) else torch.tensor(x, dtype=dtype)
        mock_torch.no_grad.return_value.__enter__.return_value = None # For with torch.no_grad()
        mock_torch.isnan.return_value.any.return_value = False # For input validation

        fixed_today = date(2023, 1, 11)
        mock_date.today.return_value = fixed_today
        today_iso = fixed_today.isoformat()
        
        # Path(...).mkdir() is mocked by mock_path_mkdir.
        # If Path(predictions_output_dir) is used, its .mkdir will be mock_path_mkdir.
        # If Path(json_file_path).parent.mkdir is used, that needs more specific mocking if Path itself is mocked.
        # Here, predict_model.py uses predictions_output_dir.mkdir and json_file_path.parent.mkdir.
        # We'll rely on the single mock_path_mkdir to catch both if they are Path instances.

        success = predict_model.run_daily_prediction(
            str(config_file), str(input_seq_path), prod_model_uri, prod_model_name, prod_model_version
        )

        assert success is True
        mock_yaml_safe_load.assert_called_once() # With the file handle from open(config_file)
        mock_mlflow_module.set_tracking_uri.assert_called_once_with(mock_params_config_pred['mlflow']['tracking_uri'])
        mock_mlflow_module.pytorch.load_model.assert_called_once_with(model_uri=prod_model_uri)
        mock_model_instance.eval.assert_called_once()

        mock_mlflow_client_class.assert_called_once()
        mock_mlflow_client_pred.get_model_version.assert_called_once_with(name=prod_model_name, version=prod_model_version)
        # run_id of the model obtained from get_model_version
        model_mlflow_run_id = mock_mlflow_client_pred.get_model_version.return_value.run_id
        mock_mlflow_client_pred.get_run.assert_called_once_with(model_mlflow_run_id)

        # dataset_run_id is retrieved from the TRAINED model's MLflow run parameters
        mock_load_scalers.assert_called_once_with(mock_params_config_pred['database'], mock_prod_model_train_dataset_run_id)
        mock_load_proc_meta.assert_not_called() # Because tickers were found in scalers_dict

        mock_np_load.assert_called_once_with(str(input_seq_path))
        # torch.tensor is called for the input sequence
        mock_torch.tensor.assert_called_with(sample_input_sequence_np, dtype=torch.float32)
        mock_torch.no_grad.assert_called_once()

        mock_model_instance.assert_called_once_with(mock_torch.tensor.return_value.to(predict_model.DEVICE)) # DEVICE is global in predict_model

        assert mock_y_scalers_pred[0].inverse_transform.call_count == 1
        assert mock_y_scalers_pred[1].inverse_transform.call_count == 1

        # Check mkdir calls (for predictions_dir and its subdirectories like historical)
        # The exact calls depend on the internal logic of predict_model.py for path creation
        # Example: predictions_output_dir.mkdir and historical_dir.mkdir
        assert mock_path_mkdir.call_count >= 1 # At least predictions_output_dir should be created

        assert mock_json_dump.call_count == 2 # For latest_predictions.json and historical file

        assert mock_save_pred.call_count == len(mock_tickers_pred)
        # Expected unscaled prediction values: 0.5 * 100 = 50, 0.6 * 200 = 120
        mock_save_pred.assert_any_call(
            mock_params_config_pred['database'], 'TICKA', 50.0, model_mlflow_run_id, today_iso
        )
        mock_save_pred.assert_any_call(
            mock_params_config_pred['database'], 'TICKB', 120.0, model_mlflow_run_id, today_iso
        )

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    def test_run_daily_prediction_load_model_fails(self, mock_mlflow_module, mock_yaml_safe_load,
                                                   mock_params_config_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_lm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred)) # Create file

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
    def test_run_daily_prediction_get_mlflow_metadata_fails(self, mock_mlflow_client_class, mock_mlflow_module,
                                                            mock_yaml_safe_load, mock_params_config_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_meta_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred)) # Create file

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock() # Model loads fine
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
    def test_run_daily_prediction_load_scalers_fails(self, mock_load_scalers, mock_mlflow_client_class,
                                                     mock_mlflow_module, mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, # Use the configured client fixture
                                                     tmp_path, caplog):
        config_file = tmp_path/"cfg_ls_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred)) # Create file
        
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred # Simulate successful client init
        mock_load_scalers.return_value = None # Simulate failure to load scalers

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        # The dataset_run_id for scalers comes from MLflow metadata, which is mocked by mock_mlflow_client_pred
        expected_dataset_run_id = mock_mlflow_client_pred.get_run.return_value.data.params["dataset_run_id"]
        assert f"Failed to load y_scalers for dataset_run_id: {expected_dataset_run_id}" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db')
    def test_run_daily_prediction_load_tickers_fails(self, mock_load_proc_meta, mock_load_scalers,
                                                     mock_mlflow_client_class, mock_mlflow_module,
                                                     mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, tmp_path, caplog):
        config_file = tmp_path/"cfg_lt_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred)) # Create file

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        # Scalers load, but 'tickers' key is missing
        mock_load_scalers.return_value = {'y_scalers': [MagicMock()]} 
        mock_load_proc_meta.return_value = None # Fallback to load_processed_features_from_db also fails

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
    @patch('models.predict_model.load_processed_features_from_db') # Keep for consistency
    @patch('models.predict_model.np.load')
    def test_run_daily_prediction_input_file_missing(self, mock_np_load, mock_load_proc_meta, mock_load_scalers,
                                                     mock_mlflow_client_class, mock_mlflow_module,
                                                     mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, mock_y_scalers_pred, mock_tickers_pred,
                                                     tmp_path, caplog):
        config_file = tmp_path/"cfg_ifm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred)) # Create file

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        
        input_seq_path = tmp_path / "non_existent_input.npy" # This file won't be created
        mock_np_load.side_effect = FileNotFoundError("Simulated file not found for np.load")

        success = predict_model.run_daily_prediction(
            str(config_file), str(input_seq_path), "uri", "name", "ver"
        )
        assert success is False
        assert f"Input sequence file not found: {input_seq_path.resolve()}" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db') # Keep for consistency
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.torch')
    def test_run_daily_prediction_input_dimension_mismatch(self, mock_torch, mock_np_load, mock_load_proc_meta,
                                                           mock_load_scalers, mock_mlflow_client_class, mock_mlflow_module,
                                                           mock_yaml_safe_load, mock_params_config_pred,
                                                           mock_mlflow_client_pred, mock_y_scalers_pred, mock_tickers_pred,
                                                           tmp_path, caplog):
        config_file = tmp_path/"cfg_idm_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred)) # Create file

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        # mock_tickers_pred has 2 tickers, model expects 2 stocks
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred} 
        
        # Input sequence with 3 stocks, while mock_tickers_pred implies 2
        mismatched_input_seq = np.random.rand(1, 5, 3, 3).astype(np.float32) 
        mock_np_load.return_value = mismatched_input_seq
        mock_torch.tensor.return_value = torch.tensor(mismatched_input_seq, dtype=torch.float32)
        mock_torch.no_grad.return_value.__enter__.return_value = None

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        # The number of stocks from input_sequence (3) vs scalers/tickers (2)
        assert "Input sequence has 3 stocks, model/scalers expect 2." in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db')
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.torch')
    @patch('models.predict_model.Path.mkdir') # Keep Path.mkdir mocked
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.date')
    def test_run_daily_prediction_save_prediction_fails(self, mock_date, mock_save_pred, mock_json_dump, mock_path_mkdir,
                                                        mock_torch, mock_np_load, mock_load_proc_meta, mock_load_scalers,
                                                        mock_mlflow_client_class, mock_mlflow_module,
                                                        mock_yaml_safe_load,
                                                        mock_params_config_pred, sample_input_sequence_np,
                                                        mock_y_scalers_pred, mock_tickers_pred,
                                                        mock_prod_model_train_dataset_run_id, # Fixture
                                                        mock_mlflow_client_pred, tmp_path, caplog): # Fixture
        config_file = tmp_path/"cfg_sp_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_pred)) # Create file

        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.eval = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32)
        mock_mlflow_module.pytorch.load_model.return_value = mock_model_instance
        
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        mock_np_load.return_value = sample_input_sequence_np
        mock_torch.tensor.side_effect = lambda x, dtype: torch.from_numpy(x).to(dtype=dtype) if isinstance(x, np.ndarray) else torch.tensor(x, dtype=dtype)
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.isnan.return_value.any.return_value = False
        fixed_today = date(2023, 1, 11)
        mock_date.today.return_value = fixed_today

        # Simulate failure during the first call to save_prediction
        mock_save_pred.side_effect = Exception("DB Save Error for TICKA")

        success = predict_model.run_daily_prediction(
            str(config_file), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Error saving prediction for TICKA" in caplog.text # Specific error before outer catch
        assert "Error in run_daily_prediction: Error saving prediction for TICKA" in caplog.text # Message from outer catch