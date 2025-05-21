# tests/models/test_predict_model.py
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import pandas as pd # Not strictly needed, but useful for context
import yaml
import json
import torch
import mlflow # For spec
from mlflow.tracking import MlflowClient # For spec
from sklearn.preprocessing import MinMaxScaler # For spec
from datetime import date

# Add src to sys.path
PROJECT_ROOT_FOR_TESTS = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT_FOR_TESTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from models import predict_model # Import the module
from utils import db_utils # Need to mock functions from here

# --- Fixtures ---
@pytest.fixture
def mock_db_config_pred():
    return {'dbname': 'test_pred_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_mlflow_config_pred():
    return {'experiment_name': 'TestPredictionExperiment', 'tracking_uri': 'mock_mlflow_uri_pred'}

@pytest.fixture
def mock_output_paths_pred(tmp_path): # Use tmp_path for dynamic output paths
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
    # Shape: (1, seq_len, num_stocks, num_features)
    return np.random.rand(1, 5, 2, 3).astype(np.float32)

@pytest.fixture
def mock_y_scalers_pred():
    # Mock 2 scalers for 2 stocks
    scaler1 = MagicMock(spec=MinMaxScaler)
    scaler2 = MagicMock(spec=MinMaxScaler)
    # Mock inverse_transform to return predictable values
    scaler1.inverse_transform.side_effect = lambda x: x * 100 # Simulate unscaling
    scaler2.inverse_transform.side_effect = lambda x: x * 200 # Simulate unscaling
    return [scaler1, scaler2]

@pytest.fixture
def mock_tickers_pred():
    return ['TICKA', 'TICKB']

@pytest.fixture
def mock_prod_model_train_dataset_run_id():
    return "prod_data_run_456"

@pytest.fixture
def mock_mlflow_client_pred():
    client = MagicMock(spec=MlflowClient)
    # Mock get_model_version to return a dummy object with run_id
    mock_model_version_details = MagicMock()
    mock_model_version_details.run_id = "prod_model_mlflow_run_789"
    client.get_model_version.return_value = mock_model_version_details

    # Mock get_run to return a dummy object with params
    mock_mlflow_run_details = MagicMock()
    mock_mlflow_run_details.data.params = {"dataset_run_id": "prod_data_run_456"}
    client.get_run.return_value = mock_mlflow_run_details

    return client

# --- Tests for run_daily_prediction ---
class TestRunDailyPrediction:
    @patch('models.predict_model.yaml.safe_load') # Patch yaml.safe_load
    @patch('models.predict_model.mlflow') # Mock the entire mlflow module
    @patch('models.predict_model.MlflowClient') # Mock the MlflowClient class
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db') # For fallback tickers
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.torch') # Mock torch module
    @patch('models.predict_model.Path.mkdir')
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.date') # Mock date.today()
    def test_run_daily_prediction_success(self, mock_date, mock_save_pred, mock_json_dump, mock_mkdir,
                                          mock_torch, mock_np_load, mock_load_proc_meta, mock_load_scalers,
                                          mock_mlflow_client_class, mock_mlflow_module,
                                          mock_yaml_safe_load, # <--- Corrected: use the patched name
                                          mock_params_config_pred, sample_input_sequence_np,
                                          mock_y_scalers_pred, mock_tickers_pred,
                                          mock_prod_model_train_dataset_run_id,
                                          mock_mlflow_client_pred, tmp_path):

        config_file = tmp_path / "params_pred.yaml" # Not actually read
        input_seq_path = tmp_path / "input_seq.npy"
        prod_model_uri = "models:/MyModel@Production"
        prod_model_name = "MyModel"
        prod_model_version = "1"

        mock_yaml_safe_load.return_value = mock_params_config_pred # <--- Use the patched mock
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred

        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.eval = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32)
        mock_mlflow_module.pytorch.load_model.return_value = mock_model_instance

        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        mock_load_proc_meta.return_value = {'tickers': mock_tickers_pred}
        mock_np_load.return_value = sample_input_sequence_np

        mock_torch.tensor.return_value = torch.tensor(sample_input_sequence_np, dtype=torch.float32)
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.isnan.return_value.any.return_value = False

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
        mock_model_instance.eval.assert_called_once()

        mock_mlflow_client_class.assert_called_once()
        mock_mlflow_client_pred.get_model_version.assert_called_once_with(name=prod_model_name, version=prod_model_version)
        mock_mlflow_client_pred.get_run.assert_called_once_with("prod_model_mlflow_run_789")

        mock_load_scalers.assert_called_once_with(mock_params_config_pred['database'], mock_prod_model_train_dataset_run_id)
        mock_load_proc_meta.assert_not_called()

        mock_np_load.assert_called_once_with(str(input_seq_path))
        mock_torch.tensor.assert_called_once()
        mock_torch.no_grad.assert_called_once()

        mock_model_instance.assert_called_once()

        assert mock_y_scalers_pred[0].inverse_transform.call_count == 1
        assert mock_y_scalers_pred[1].inverse_transform.call_count == 1

        assert mock_mkdir.call_count == 2
        assert mock_json_dump.call_count == 2

        assert mock_save_pred.call_count == len(mock_tickers_pred)
        mock_save_pred.assert_any_call(
            mock_params_config_pred['database'], 'TICKA', pytest.approx(0.5 * 100), 'prod_model_mlflow_run_789', today_iso
        )
        mock_save_pred.assert_any_call(
            mock_params_config_pred['database'], 'TICKB', pytest.approx(0.6 * 200), 'prod_model_mlflow_run_789', today_iso
        )

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    def test_run_daily_prediction_load_model_fails(self, mock_mlflow_module, mock_yaml_safe_load,
                                                   mock_params_config_pred, tmp_path, caplog):
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.side_effect = Exception("Model load error")

        success = predict_model.run_daily_prediction(
            str(tmp_path/"cfg.yaml"), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Failed to load model from uri" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    def test_run_daily_prediction_get_mlflow_metadata_fails(self, mock_mlflow_client_class, mock_mlflow_module,
                                                            mock_yaml_safe_load, mock_params_config_pred, tmp_path, caplog):
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value.get_model_version.side_effect = Exception("MLflow metadata error")

        success = predict_model.run_daily_prediction(
            str(tmp_path/"cfg.yaml"), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Error fetching metadata for production model" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    def test_run_daily_prediction_load_scalers_fails(self, mock_load_scalers, mock_mlflow_client_class,
                                                     mock_mlflow_module, mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, tmp_path, caplog):
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        mock_load_scalers.return_value = None

        success = predict_model.run_daily_prediction(
            str(tmp_path/"cfg.yaml"), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Failed to load y_scalers for dataset_run_id" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db')
    def test_run_daily_prediction_load_tickers_fails(self, mock_load_proc_meta, mock_load_scalers,
                                                     mock_mlflow_client_class, mock_mlflow_module,
                                                     mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, tmp_path, caplog):
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        mock_load_scalers.return_value = {'y_scalers': [MagicMock()]}
        mock_load_proc_meta.return_value = None

        success = predict_model.run_daily_prediction(
            str(tmp_path/"cfg.yaml"), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Failed to load tickers for dataset_run_id" in caplog.text


    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db')
    @patch('models.predict_model.np.load')
    def test_run_daily_prediction_input_file_missing(self, mock_np_load, mock_load_proc_meta, mock_load_scalers,
                                                     mock_mlflow_client_class, mock_mlflow_module,
                                                     mock_yaml_safe_load, mock_params_config_pred,
                                                     mock_mlflow_client_pred, mock_y_scalers_pred, mock_tickers_pred,
                                                     tmp_path, caplog):
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        
        input_seq_path = tmp_path / "non_existent_input.npy"
        mock_np_load.side_effect = FileNotFoundError

        success = predict_model.run_daily_prediction(
            str(tmp_path/"cfg.yaml"), str(input_seq_path), "uri", "name", "ver"
        )
        assert success is False
        assert f"Input sequence file not found: {input_seq_path.resolve()}" in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db')
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.torch')
    def test_run_daily_prediction_input_dimension_mismatch(self, mock_torch, mock_np_load, mock_load_proc_meta,
                                                           mock_load_scalers, mock_mlflow_client_class, mock_mlflow_module,
                                                           mock_yaml_safe_load, mock_params_config_pred,
                                                           mock_mlflow_client_pred, mock_y_scalers_pred, mock_tickers_pred,
                                                           tmp_path, caplog):
        mock_yaml_safe_load.return_value = mock_params_config_pred
        mock_mlflow_module.pytorch.load_model.return_value = MagicMock()
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        
        mismatched_input_seq = np.random.rand(1, 5, 3, 3).astype(np.float32)
        mock_np_load.return_value = mismatched_input_seq
        mock_torch.tensor.return_value = torch.tensor(mismatched_input_seq, dtype=torch.float32)
        mock_torch.no_grad.return_value.__enter__.return_value = None

        success = predict_model.run_daily_prediction(
            str(tmp_path/"cfg.yaml"), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        assert "Input sequence has 3 stocks, model/scalers expect 2." in caplog.text

    @patch('models.predict_model.yaml.safe_load')
    @patch('models.predict_model.mlflow')
    @patch('models.predict_model.MlflowClient')
    @patch('models.predict_model.load_scalers')
    @patch('models.predict_model.load_processed_features_from_db')
    @patch('models.predict_model.np.load')
    @patch('models.predict_model.torch')
    @patch('models.predict_model.Path.mkdir')
    @patch('models.predict_model.json.dump')
    @patch('models.predict_model.save_prediction')
    @patch('models.predict_model.date')
    def test_run_daily_prediction_save_prediction_fails(self, mock_date, mock_save_pred, mock_json_dump, mock_mkdir,
                                                        mock_torch, mock_np_load, mock_load_proc_meta, mock_load_scalers,
                                                        mock_mlflow_client_class, mock_mlflow_module,
                                                        mock_yaml_safe_load, # <--- Corrected
                                                        mock_params_config_pred, sample_input_sequence_np,
                                                        mock_y_scalers_pred, mock_tickers_pred,
                                                        mock_prod_model_train_dataset_run_id,
                                                        mock_mlflow_client_pred, tmp_path, caplog):
        mock_yaml_safe_load.return_value = mock_params_config_pred # <--- Use the patched mock
        mock_mlflow_client_class.return_value = mock_mlflow_client_pred
        mock_model_instance = MagicMock(spec=torch.nn.Module)
        mock_model_instance.eval = MagicMock()
        mock_model_instance.return_value = torch.tensor([[[0.5, 0.6]]], dtype=torch.float32)
        mock_mlflow_module.pytorch.load_model.return_value = mock_model_instance
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_pred, 'tickers': mock_tickers_pred}
        mock_np_load.return_value = sample_input_sequence_np
        mock_torch.tensor.return_value = torch.tensor(sample_input_sequence_np, dtype=torch.float32)
        mock_torch.no_grad.return_value.__enter__.return_value = None
        mock_torch.isnan.return_value.any.return_value = False
        mock_date.today.return_value = date(2023, 1, 11)

        mock_save_pred.side_effect = [Exception("DB Save Error for TICKA"), MagicMock()]

        success = predict_model.run_daily_prediction(
            str(tmp_path/"cfg.yaml"), str(tmp_path/"input.npy"), "uri", "name", "ver"
        )
        assert success is False
        # The log "Error saving prediction for TICKA" is tricky to assert because the exception
        # is caught by the outer try-except in run_daily_prediction.
        # The primary log will be "Error in run_daily_prediction"
        assert "DB Save Error for TICKA" in caplog.text # Original error message
        assert "Error in run_daily_prediction: DB Save Error for TICKA" in caplog.text # Message from outer catch