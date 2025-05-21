# tests/models/test_train_model.py
import pytest
from unittest.mock import patch, MagicMock, call
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import mlflow # For spec
from mlflow.entities import Run as MlflowRun
from mlflow.entities.model_registry import ModelVersion as MlflowModelVersion 
from mlflow.tracking import MlflowClient

from models.train_model import train_final_model, run_training

# --- Fixtures ---
@pytest.fixture
def mock_db_config_train():
    return {'dbname': 'test_train_db', 'user': 'u', 'password': 'p', 'host': 'h', 'port': '5432'}

@pytest.fixture
def mock_mlflow_config_train():
    return {'experiment_name': 'TestExperiment', 'tracking_uri': 'mock_mlflow_uri'}

@pytest.fixture
def mock_training_params_cfg_train():
    return {'epochs': 2}

@pytest.fixture
def mock_params_config_train(mock_db_config_train, mock_mlflow_config_train, mock_training_params_cfg_train, tmp_path):
    return {
        'database': mock_db_config_train,
        'mlflow': mock_mlflow_config_train,
        'training': mock_training_params_cfg_train,
        'output_paths': {'training_plots_dir': str(tmp_path / 'plots/test_training_plots')}
    }

@pytest.fixture
def sample_best_hyperparams_train():
    return {
        'batch_size': 32, 'hidden_size': 64, 'num_layers': 1,
        'dropout_rate': 0.1, 'learning_rate': 0.01, 'model_type': 'lstm'
    }

@pytest.fixture
def sample_scaled_data_train(): # Return numpy arrays
    return {
        'X_train': np.random.rand(20, 5, 2, 3).astype(np.float32),
        'y_train': np.random.rand(20, 1, 2).astype(np.float32),
        'X_test': np.random.rand(10, 5, 2, 3).astype(np.float32),
        'y_test': np.random.rand(10, 1, 2).astype(np.float32),
    }

@pytest.fixture
def mock_y_scalers_train():
    s1, s2 = MagicMock(), MagicMock()
    s1.inverse_transform.side_effect = lambda x: x
    s2.inverse_transform.side_effect = lambda x: x
    return [s1, s2]

@pytest.fixture
def mock_tickers_train():
    return ['TICKA', 'TICKB']

@pytest.fixture
def device_eval():
    return torch.device('cpu')


# --- Tests for train_final_model ---
class TestTrainFinalModel:
    @patch('models.train_model.StockLSTM')
    @patch('models.train_model.StockLSTMWithAttention')
    @patch('models.train_model.StockLSTMWithCrossStockAttention')
    @patch('models.train_model.DataLoader')
    @patch('models.train_model.evaluate_model')
    @patch('models.train_model.visualize_predictions')
    @patch('models.train_model.mlflow')
    def test_train_final_model_flow_and_mlflow_calls(
            self, mock_mlflow, mock_viz_preds, mock_eval_model, mock_dataloader,
            mock_lstm_cross, mock_lstm_att, mock_lstm_basic,
            sample_best_hyperparams_train, sample_scaled_data_train,
            mock_y_scalers_train, mock_tickers_train, device_eval,
            mock_training_params_cfg_train, mock_mlflow_config_train, tmp_path):

        dataset_run_id = "data_run_789"
        plot_output_dir = tmp_path / "train_plots"

        mock_model_inst = MagicMock(spec=nn.Module)
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.state_dict.return_value = {'param': torch.tensor(1.0)}
        mock_model_inst.load_state_dict = MagicMock()
        mock_model_inst.return_value = torch.rand(sample_best_hyperparams_train['batch_size'], 1, 2)
        mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(1))] # For optimizer
        
        # Determine which mock model class should be used based on model_type
        model_type = sample_best_hyperparams_train.get('model_type', 'lstm')
        if model_type == 'lstm':
            mock_lstm_basic.return_value = mock_model_inst
        elif model_type == 'lstm_attention':
            mock_lstm_att.return_value = mock_model_inst
        elif model_type == 'lstm_cross_attention':
            mock_lstm_cross.return_value = mock_model_inst


        dummy_X_batch = torch.rand(sample_best_hyperparams_train['batch_size'], 5, 2, 3)
        dummy_y_batch = torch.rand(sample_best_hyperparams_train['batch_size'], 1, 2)
        
        mock_train_loader_obj = MagicMock()
        mock_train_loader_obj.__iter__.return_value = iter([(dummy_X_batch, dummy_y_batch)])
        mock_train_loader_obj.__len__.return_value = 1
        
        mock_test_loader_obj = MagicMock()
        mock_test_loader_obj.__iter__.return_value = iter([(dummy_X_batch, dummy_y_batch)])
        mock_test_loader_obj.__len__.return_value = 1

        def dataloader_side_effect(dataset, batch_size, shuffle):
            if shuffle: return mock_train_loader_obj
            return mock_test_loader_obj
        mock_dataloader.side_effect = dataloader_side_effect

        mock_eval_model.return_value = (
            np.random.rand(10,1,2), # preds
            np.random.rand(10,1,2), # targets
            {'test_loss': 0.1, 'avg_mape': 0.05, 'avg_mse': 0.01, 'avg_direction_accuracy': 0.8}
        )

        mock_mlflow_run = MagicMock(spec=MlflowRun)
        mock_mlflow_run.info.run_id = "mock_mlflow_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_mlflow_run

        y_test_np = sample_scaled_data_train['y_test']

        model_obj, preds_test, metrics_test, mlflow_run_id_out = train_final_model(
            dataset_run_id, sample_best_hyperparams_train,
            sample_scaled_data_train['X_train'], sample_scaled_data_train['y_train'],
            sample_scaled_data_train['X_test'], y_test_np,
            num_features=3, num_stocks=2, y_scalers=mock_y_scalers_train,
            tickers=mock_tickers_train, device=device_eval,
            training_epochs=mock_training_params_cfg_train['epochs'],
            plot_output_dir=plot_output_dir,
            mlflow_experiment_name=mock_mlflow_config_train['experiment_name']
        )
        
        mock_mlflow.set_experiment.assert_called_once_with(mock_mlflow_config_train['experiment_name'])
        mock_mlflow.start_run.assert_called_once()
        
        mock_mlflow.log_param.assert_any_call("dataset_run_id", dataset_run_id)
        mock_mlflow.log_params.assert_called_once_with(sample_best_hyperparams_train)
        assert mock_mlflow.log_metric.call_count >= mock_training_params_cfg_train['epochs']
        
        mock_eval_model.assert_called()
        mock_viz_preds.assert_called_once_with(
            pytest.ANY, y_test_np, mock_y_scalers_train,
            mock_tickers_train, plot_output_dir, num_points=20
        )
        
        mock_mlflow.pytorch.log_model.assert_called_once_with(
            pytorch_model=mock_model_inst,
            artifact_path="model",
            registered_model_name=mock_mlflow_config_train['experiment_name']
        )
        
        assert model_obj is mock_model_inst
        assert mlflow_run_id_out == "mock_mlflow_run_123"

    @patch('models.train_model.StockLSTM')
    @patch('models.train_model.StockLSTMWithAttention') # Add other model types if needed
    @patch('models.train_model.StockLSTMWithCrossStockAttention')
    @patch('models.train_model.DataLoader')
    @patch('models.train_model.mlflow')
    def test_train_final_model_no_test_set(self, mock_mlflow, mock_dataloader, 
                                           mock_lstm_cross, mock_lstm_att, mock_lstm_basic, # Mocks for all model types
                                           sample_best_hyperparams_train, sample_scaled_data_train,
                                           mock_y_scalers_train, mock_tickers_train, device_eval,
                                           tmp_path, caplog):
        mock_model_inst = MagicMock(spec=nn.Module)
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.return_value = torch.rand(32,1,2)
        mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]

        model_type = sample_best_hyperparams_train.get('model_type', 'lstm')
        if model_type == 'lstm':
            mock_lstm_basic.return_value = mock_model_inst
        elif model_type == 'lstm_attention':
            mock_lstm_att.return_value = mock_model_inst
        elif model_type == 'lstm_cross_attention':
            mock_lstm_cross.return_value = mock_model_inst

        mock_train_loader_obj = MagicMock()
        mock_train_loader_obj.__iter__.return_value = iter([(torch.rand(32,5,2,3), torch.rand(32,1,2))])
        mock_train_loader_obj.__len__.return_value = 1
        mock_dataloader.return_value = mock_train_loader_obj

        mock_mlflow_run = MagicMock(spec=MlflowRun)
        mock_mlflow_run.info.run_id = "mfr_no_test"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_mlflow_run

        with patch('models.train_model.evaluate_model') as mock_eval, \
             patch('models.train_model.visualize_predictions') as mock_viz:
            train_final_model(
                "data_run_no_test", sample_best_hyperparams_train,
                sample_scaled_data_train['X_train'], sample_scaled_data_train['y_train'],
                X_test=None, y_test=None,
                num_features=3, num_stocks=2, y_scalers=mock_y_scalers_train,
                tickers=mock_tickers_train, device=device_eval,
                training_epochs=1, plot_output_dir=tmp_path,
                mlflow_experiment_name="TestExp"
            )
            mock_eval.assert_not_called()
            mock_viz.assert_not_called()
        assert "Test data is empty or None." in caplog.text
        assert "No test set evaluation performed" in caplog.text # Check actual log message


# --- Tests for run_training ---
class TestRunTraining:
    @patch('models.train_model.yaml.safe_load')
    @patch('models.train_model.load_scaled_features')
    @patch('models.train_model.load_scalers')
    @patch('models.train_model.load_processed_features_from_db')
    @patch('models.train_model.load_optimization_results')
    @patch('models.train_model.train_final_model')
    @patch('models.train_model.mlflow')
    @patch('models.train_model.save_prediction')
    def test_run_training_success(self, mock_save_pred, mock_mlflow_module, mock_train_final,
                                  mock_load_opt_res, mock_load_proc_meta, mock_load_scalers, mock_load_scaled,
                                  mock_yaml_safe_load,
                                  mock_params_config_train, sample_scaled_data_train,
                                  mock_y_scalers_train, mock_tickers_train,
                                  sample_best_hyperparams_train, tmp_path):
        dataset_run_id = "final_train_data_run_001"
        config_file = tmp_path / "params_train.yaml"
        config_file.write_text(yaml.dump(mock_params_config_train)) # Create file

        mock_yaml_safe_load.return_value = mock_params_config_train
        mock_load_scaled.side_effect = lambda db, drun_id, set_name: sample_scaled_data_train.get(set_name)
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_train, 'tickers': mock_tickers_train}
        mock_load_opt_res.return_value = sample_best_hyperparams_train
        mock_load_proc_meta.return_value = {'tickers': mock_tickers_train}

        trained_model_mock = MagicMock()
        test_predictions_np = np.random.rand(10,1,2)
        mock_train_final.return_value = (
            trained_model_mock,
            test_predictions_np, # test_predictions_np
            {'avg_mape': 0.1},      # final_test_metrics
            "mlflow_run_for_trained_model_xyz" # mlflow_model_run_id
        )
        
        mock_mlflow_client_inst = MagicMock(spec=MlflowClient)
        mock_mlflow_module.tracking.MlflowClient.return_value = mock_mlflow_client_inst
        
        mock_model_version = MagicMock(spec=MlflowModelVersion)
        mock_model_version.version = "3"
        mock_mlflow_client_inst.search_model_versions.return_value = [mock_model_version]
        # mock_mlflow_client_inst.get_latest_versions.return_value = [] # If needed for archiving logic

        result_mlflow_run_id = run_training(str(config_file), dataset_run_id)

        assert result_mlflow_run_id == "mlflow_run_for_trained_model_xyz"
        mock_mlflow_module.set_tracking_uri.assert_called_once_with(mock_params_config_train['mlflow']['tracking_uri'])
        mock_load_scaled.assert_any_call(mock_params_config_train['database'], dataset_run_id, 'X_train')
        mock_load_scalers.assert_called_once_with(mock_params_config_train['database'], dataset_run_id)
        mock_load_opt_res.assert_called_once_with(mock_params_config_train['database'], dataset_run_id)
        
        mock_train_final.assert_called_once()
        args_train_final, _ = mock_train_final.call_args
        assert args_train_final[0] == dataset_run_id
        assert args_train_final[1] == sample_best_hyperparams_train
        
        mock_mlflow_client_inst.search_model_versions.assert_called_once_with(f"run_id='{result_mlflow_run_id}'")
        mock_mlflow_client_inst.set_registered_model_alias.assert_called_once_with(
            name=mock_params_config_train['mlflow']['experiment_name'],
            alias="Production",
            version="3"
        )
        
        if test_predictions_np is not None and test_predictions_np.size > 0:
            mock_save_pred.assert_called()
        else:
            mock_save_pred.assert_not_called()

    @patch('models.train_model.yaml.safe_load')
    @patch('models.train_model.load_scaled_features')
    @patch('models.train_model.train_final_model') 
    def test_run_training_load_scaled_fails(self, mock_train_final_unused, mock_load_scaled, mock_yaml_safe_load,
                                            mock_params_config_train, tmp_path, caplog):
        dataset_run_id = "train_fail_ls"
        config_file = tmp_path/"cfg_ls.yaml"
        config_file.write_text(yaml.dump(mock_params_config_train))

        mock_yaml_safe_load.return_value = mock_params_config_train
        mock_load_scaled.return_value = None
        result = run_training(str(config_file), dataset_run_id)
        assert result is None
        assert f"Failed to load scaled training data for dataset_run_id: {dataset_run_id}" in caplog.text

    @patch('models.train_model.yaml.safe_load')
    @patch('models.train_model.load_scaled_features')
    @patch('models.train_model.load_scalers')
    @patch('models.train_model.train_final_model')
    def test_run_training_load_scalers_fails(self, mock_train_final_unused, mock_load_scalers, mock_load_scaled,
                                             mock_yaml_safe_load, mock_params_config_train,
                                             sample_scaled_data_train, tmp_path, caplog):
        dataset_run_id = "train_fail_lsc"
        config_file = tmp_path/"cfg_lsc.yaml"
        config_file.write_text(yaml.dump(mock_params_config_train))

        mock_yaml_safe_load.return_value = mock_params_config_train
        mock_load_scaled.side_effect = lambda db, dr_id, set_name: sample_scaled_data_train.get(set_name)
        mock_load_scalers.return_value = None
        result = run_training(str(config_file), dataset_run_id)
        assert result is None
        assert f"Failed to load scalers or 'y_scalers' not found for dataset_run_id: {dataset_run_id}" in caplog.text

    @patch('models.train_model.yaml.safe_load')
    @patch('models.train_model.load_scaled_features')
    @patch('models.train_model.load_scalers')
    @patch('models.train_model.load_optimization_results')
    @patch('models.train_model.train_final_model')
    def test_run_training_load_opt_res_fails(self, mock_train_final_unused, mock_load_opt_res, mock_load_scalers, mock_load_scaled,
                                             mock_yaml_safe_load, mock_params_config_train,
                                             sample_scaled_data_train, mock_y_scalers_train,
                                             mock_tickers_train, tmp_path, caplog):
        dataset_run_id = "train_fail_lor"
        config_file = tmp_path/"cfg_lor.yaml"
        config_file.write_text(yaml.dump(mock_params_config_train))

        mock_yaml_safe_load.return_value = mock_params_config_train
        mock_load_scaled.side_effect = lambda db, dr_id, set_name: sample_scaled_data_train.get(set_name)
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_train, 'tickers': mock_tickers_train}
        mock_load_opt_res.return_value = None
        result = run_training(str(config_file), dataset_run_id)
        assert result is None
        assert f"Failed to load best hyperparameters for dataset_run_id: {dataset_run_id}" in caplog.text

    @patch('models.train_model.yaml.safe_load')
    @patch('models.train_model.load_scaled_features')
    @patch('models.train_model.load_scalers')
    @patch('models.train_model.load_optimization_results')
    @patch('models.train_model.train_final_model')
    @patch('models.train_model.mlflow')
    def test_run_training_promotion_fails_no_version(self, mock_mlflow_module, mock_train_final,
                                                     mock_load_opt_res, mock_load_scalers, mock_load_scaled,
                                                     mock_yaml_safe_load,
                                                     mock_params_config_train, sample_scaled_data_train,
                                                     mock_y_scalers_train, mock_tickers_train,
                                                     sample_best_hyperparams_train, tmp_path, caplog):
        dataset_run_id = "train_promo_fail"
        config_file = tmp_path / "params_train_promo_fail.yaml"
        config_file.write_text(yaml.dump(mock_params_config_train))

        mock_yaml_safe_load.return_value = mock_params_config_train
        mock_load_scaled.side_effect = lambda db, dr_id, set_name: sample_scaled_data_train.get(set_name)
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_train, 'tickers': mock_tickers_train}
        mock_load_opt_res.return_value = sample_best_hyperparams_train
        mock_train_final.return_value = (MagicMock(), None, {}, "mlflow_run_promo_fail") # Model, no preds, no metrics, run_id
        
        mock_mlflow_client_inst = MagicMock(spec=MlflowClient)
        mock_mlflow_module.tracking.MlflowClient.return_value = mock_mlflow_client_inst
        mock_mlflow_client_inst.search_model_versions.return_value = [] # Simulate no version found

        result_mlflow_run_id = run_training(str(config_file), dataset_run_id)
        
        assert result_mlflow_run_id == "mlflow_run_promo_fail" # Training itself succeeds
        assert "No model version found in registry for MLflow run_id mlflow_run_promo_fail" in caplog.text
        mock_mlflow_client_inst.set_registered_model_alias.assert_not_called()