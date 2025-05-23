# tests/models/test_train_model.py
import pytest
from unittest.mock import patch, MagicMock, call, ANY # Use unittest.mock.ANY
import torch
import torch.nn as nn
import numpy as np
import yaml
from pathlib import Path
import mlflow 
from mlflow.entities import Run as MlflowRun
from mlflow.entities.model_registry import ModelVersion as MlflowModelVersion
from mlflow.tracking import MlflowClient # Import MlflowClient for spec

from models.train_model import train_final_model, run_training

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
def sample_scaled_data_train():
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
        mock_model_inst.return_value = torch.rand(sample_best_hyperparams_train['batch_size'], 1, 2, requires_grad=True)
        mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        
        model_type = sample_best_hyperparams_train.get('model_type', 'lstm')
        if model_type == 'lstm': mock_lstm_basic.return_value = mock_model_inst
        elif model_type == 'lstm_attention': mock_lstm_att.return_value = mock_model_inst
        else: mock_lstm_cross.return_value = mock_model_inst

        dummy_X_batch = torch.rand(sample_best_hyperparams_train['batch_size'], 5, 2, 3, device=device_eval)
        dummy_y_batch = torch.rand(sample_best_hyperparams_train['batch_size'], 1, 2, device=device_eval)
        
        mock_train_loader_obj = MagicMock()
        mock_train_loader_obj.__iter__.return_value = iter([(dummy_X_batch, dummy_y_batch)])
        mock_train_loader_obj.__len__.return_value = 1
        
        mock_test_loader_obj = MagicMock()
        mock_test_loader_obj.__iter__.return_value = iter([(dummy_X_batch, dummy_y_batch)]) # Can reuse dummy data
        mock_test_loader_obj.__len__.return_value = 1

        def dataloader_side_effect(dataset, batch_size, shuffle):
            # Based on shuffle arg, return the appropriate mock loader
            # train_loader has shuffle=True, test_loader has shuffle=False
            if shuffle: return mock_train_loader_obj
            return mock_test_loader_obj
        mock_dataloader.side_effect = dataloader_side_effect

        # Mock return for evaluate_model
        # preds_np, targets_np, metrics_dict
        mock_eval_model.return_value = (
            np.random.rand(10,1,2), 
            sample_scaled_data_train['y_test'], # Ensure targets_np matches y_test for visualize
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
        
        mock_viz_preds.assert_called_once_with(
            ANY, # This will be the predictions_on_test from evaluate_model
            y_test_np, # This is the y_test passed to train_final_model
            mock_y_scalers_train,
            mock_tickers_train, plot_output_dir, num_points=20
        )
    
    @patch('models.train_model.StockLSTM')
    @patch('models.train_model.StockLSTMWithAttention')
    @patch('models.train_model.StockLSTMWithCrossStockAttention')
    @patch('models.train_model.DataLoader')
    @patch('models.train_model.mlflow')
    def test_train_final_model_no_test_set(self, mock_mlflow, mock_dataloader, 
                                           mock_lstm_cross, mock_lstm_att, mock_lstm_basic,
                                           sample_best_hyperparams_train, sample_scaled_data_train,
                                           mock_y_scalers_train, mock_tickers_train, device_eval,
                                           tmp_path, caplog):
        mock_model_inst = MagicMock(spec=nn.Module)
        mock_model_inst.to.return_value = mock_model_inst
        mock_model_inst.return_value = torch.rand(32,1,2, requires_grad=True)
        mock_model_inst.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]

        model_type = sample_best_hyperparams_train.get('model_type', 'lstm')
        if model_type == 'lstm': mock_lstm_basic.return_value = mock_model_inst
        elif model_type == 'lstm_attention': mock_lstm_att.return_value = mock_model_inst
        else: mock_lstm_cross.return_value = mock_model_inst

        mock_train_loader_obj = MagicMock()
        dummy_train_X = torch.rand(32,5,2,3, device=device_eval)
        dummy_train_y = torch.rand(32,1,2, device=device_eval)
        mock_train_loader_obj.__iter__.return_value = iter([(dummy_train_X, dummy_train_y)])
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
        assert "No test set evaluation performed" in caplog.text


class TestRunTraining:
    # Patch MlflowClient where it's looked up in models.train_model
    @patch('models.train_model.MlflowClient') 
    @patch('models.train_model.yaml.safe_load')
    @patch('models.train_model.load_scaled_features')
    @patch('models.train_model.load_scalers')
    @patch('models.train_model.load_processed_features_from_db')
    @patch('models.train_model.load_optimization_results')
    @patch('models.train_model.train_final_model')
    @patch('models.train_model.mlflow') # For mlflow.set_tracking_uri
    @patch('models.train_model.save_prediction')
    def test_run_training_success(self, mock_save_pred, mock_mlflow_module, mock_train_final,
                                  mock_load_opt_res, mock_load_proc_meta, mock_load_scalers, mock_load_scaled,
                                  mock_yaml_safe_load, MockMlflowClientClass, # Patched MlflowClient class
                                  mock_params_config_train, sample_scaled_data_train,
                                  mock_y_scalers_train, mock_tickers_train, device_eval,
                                  sample_best_hyperparams_train, tmp_path):
        dataset_run_id = "final_train_data_run_001"
        config_file = tmp_path / "params_train.yaml"
        config_file.write_text(yaml.dump(mock_params_config_train))

        mock_yaml_safe_load.return_value = mock_params_config_train
        mock_load_scaled.side_effect = lambda db, drun_id, set_name: sample_scaled_data_train.get(set_name)
        mock_load_scalers.return_value = {'y_scalers': mock_y_scalers_train, 'tickers': mock_tickers_train}
        mock_load_opt_res.return_value = sample_best_hyperparams_train
        mock_load_proc_meta.return_value = {'tickers': mock_tickers_train}

        trained_model_mock = MagicMock(spec=nn.Module)
        test_predictions_np = np.random.rand(10,1,2)
        final_metrics = {'avg_mape': 0.1}
        expected_mlflow_run_id = "mlflow_run_for_trained_model_xyz"
        mock_train_final.return_value = (
            trained_model_mock, test_predictions_np, final_metrics, expected_mlflow_run_id
        )
        
        # This is the mock for the INSTANCE of MlflowClient
        mock_mlflow_client_instance = MagicMock(spec=MlflowClient)
        MockMlflowClientClass.return_value = mock_mlflow_client_instance # MlflowClient() in SUT returns this
        
        mock_model_version = MagicMock(spec=MlflowModelVersion)
        mock_model_version.version = "3"
        # search_model_versions is called on the INSTANCE
        mock_mlflow_client_instance.search_model_versions.return_value = [mock_model_version]
        
        result_mlflow_run_id = run_training(str(config_file), dataset_run_id)

        assert result_mlflow_run_id == expected_mlflow_run_id
        
        mock_train_final.assert_called_once_with(
            dataset_run_id=dataset_run_id,
            best_params=sample_best_hyperparams_train,
            X_train=sample_scaled_data_train['X_train'],
            y_train=sample_scaled_data_train['y_train'],
            X_test=sample_scaled_data_train['X_test'],
            y_test=sample_scaled_data_train['y_test'],
            num_features=sample_scaled_data_train['X_train'].shape[3],
            num_stocks=sample_scaled_data_train['X_train'].shape[2],
            y_scalers=mock_y_scalers_train,
            tickers=mock_tickers_train,
            device=ANY, 
            training_epochs=mock_params_config_train['training']['epochs'],
            plot_output_dir=Path(mock_params_config_train['output_paths']['training_plots_dir']),
            mlflow_experiment_name=mock_params_config_train['mlflow']['experiment_name']
        )
        
        # Assert that search_model_versions was called on our instance
        mock_mlflow_client_instance.search_model_versions.assert_called_once_with(f"run_id='{expected_mlflow_run_id}'")
        mock_mlflow_client_instance.set_registered_model_alias.assert_called_once_with(
            name=mock_params_config_train['mlflow']['experiment_name'],
            alias="Production",
            version="3"
        )

    @patch('models.train_model.MlflowClient') # Patch correct location
    @patch('models.train_model.yaml.safe_load')
    @patch('models.train_model.load_scaled_features')
    @patch('models.train_model.train_final_model') 
    def test_run_training_load_scaled_fails(self, mock_train_final_unused, mock_load_scaled, 
                                            mock_yaml_safe_load, MockMlflowClientClass_unused, # Add unused mock client
                                            mock_params_config_train, tmp_path, caplog):
        dataset_run_id = "train_fail_ls"
        config_file = tmp_path/"cfg_ls.yaml"
        config_file.write_text(yaml.dump(mock_params_config_train))

        mock_yaml_safe_load.return_value = mock_params_config_train
        mock_load_scaled.return_value = None
        result = run_training(str(config_file), dataset_run_id)
        assert result is None
        assert f"Failed to load scaled training data for dataset_run_id: {dataset_run_id}" in caplog.text

    @patch('models.train_model.MlflowClient')
    @patch('models.train_model.yaml.safe_load')
    @patch('models.train_model.load_scaled_features')
    @patch('models.train_model.load_scalers')
    @patch('models.train_model.train_final_model')
    def test_run_training_load_scalers_fails(self, mock_train_final_unused, mock_load_scalers, mock_load_scaled,
                                             mock_yaml_safe_load, MockMlflowClientClass_unused,
                                             mock_params_config_train,
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

    @patch('models.train_model.MlflowClient')
    @patch('models.train_model.yaml.safe_load')
    @patch('models.train_model.load_scaled_features')
    @patch('models.train_model.load_scalers')
    @patch('models.train_model.load_optimization_results')
    @patch('models.train_model.train_final_model')
    def test_run_training_load_opt_res_fails(self, mock_train_final_unused, mock_load_opt_res, mock_load_scalers, mock_load_scaled,
                                             mock_yaml_safe_load, MockMlflowClientClass_unused, 
                                             mock_params_config_train,
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

    @patch('models.train_model.MlflowClient') # Target where MlflowClient is defined for models.train_model
    @patch('models.train_model.yaml.safe_load')
    @patch('models.train_model.load_scaled_features')
    @patch('models.train_model.load_scalers')
    @patch('models.train_model.load_optimization_results')
    @patch('models.train_model.train_final_model')
    @patch('models.train_model.mlflow') # For mlflow.set_tracking_uri
    def test_run_training_promotion_fails_no_version(self, mock_mlflow_module, mock_train_final,
                                                     mock_load_opt_res, mock_load_scalers, mock_load_scaled,
                                                     mock_yaml_safe_load, MockMlflowClientClass, # Patched class
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
        mock_train_final.return_value = (MagicMock(), None, {}, "mlflow_run_promo_fail")
        
        mock_mlflow_client_instance = MagicMock(spec=MlflowClient) # Instance mock
        MockMlflowClientClass.return_value = mock_mlflow_client_instance # Class returns our instance
        mock_mlflow_client_instance.search_model_versions.return_value = [] # search_model_versions on instance returns []

        result_mlflow_run_id = run_training(str(config_file), dataset_run_id)
        
        assert result_mlflow_run_id == "mlflow_run_promo_fail"
        assert "No model version found in registry for MLflow run_id mlflow_run_promo_fail" in caplog.text
        mock_mlflow_client_instance.set_registered_model_alias.assert_not_called()