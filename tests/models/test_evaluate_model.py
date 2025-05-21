# tests/models/test_evaluate_model.py
import pytest
from unittest.mock import patch, MagicMock, call
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler # For spec

from models.evaluate_model import evaluate_model, visualize_predictions
import logging # --- ADDED: To check log messages ---

# --- Fixtures ---
@pytest.fixture
def mock_model_eval():
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock() 
    model.return_value = torch.tensor([[[0.5, 0.6]], [[0.7, 0.8]]], dtype=torch.float32)
    return model

@pytest.fixture
def mock_criterion_eval():
    criterion = MagicMock(spec=nn.MSELoss)
    criterion.return_value = torch.tensor(0.1)
    return criterion

@pytest.fixture
def mock_y_scalers_eval():
    # Mock 2 scalers for 2 stocks
    scaler1 = MagicMock(spec=MinMaxScaler)
    scaler2 = MagicMock(spec=MinMaxScaler)
    # Mock inverse_transform to return predictable values or identity
    scaler1.inverse_transform.side_effect = lambda x: x * 10 # Example scaling
    scaler2.inverse_transform.side_effect = lambda x: x * 20 # Example scaling
    return [scaler1, scaler2]

@pytest.fixture
def mock_test_loader_eval():
    # Simulate a DataLoader yielding one batch
    # X_batch shape: (batch_size, seq_len, num_stocks, num_features)
    X_batch = torch.randn(2, 5, 2, 3) # Batch size 2, seq_len 5, 2 stocks, 3 features
    # y_batch shape: (batch_size, pred_len (1), num_stocks)
    y_batch = torch.tensor([[[1.0, 1.1]], [[1.2, 1.3]]], dtype=torch.float32)
    return [(X_batch, y_batch)] # Iterable with one batch

@pytest.fixture
def device_eval():
    return torch.device('cpu') # Use CPU for tests

# --- Tests for evaluate_model ---
class TestEvaluateModel:
    def test_evaluate_model_calls_and_metrics_structure(self, mock_model_eval, mock_test_loader_eval,
                                                        mock_criterion_eval, mock_y_scalers_eval, device_eval):
        predictions_np, targets_np, metrics = evaluate_model(
            mock_model_eval, mock_test_loader_eval, mock_criterion_eval,
            mock_y_scalers_eval, device_eval
        )

        mock_model_eval.eval.assert_called_once() # Check model set to eval mode
        
        # Check model was called with data from loader
        call_args_list = mock_model_eval.call_args_list
        assert len(call_args_list) == 1 # Called once for the single batch
        torch.testing.assert_close(call_args_list[0][0][0], mock_test_loader_eval[0][0].to(device_eval))


        mock_criterion_eval.assert_called_once() # Criterion called

        # Check scalers' inverse_transform called for each stock
        assert mock_y_scalers_eval[0].inverse_transform.call_count > 0
        assert mock_y_scalers_eval[1].inverse_transform.call_count > 0
        
        # Check output shapes
        # predictions_np from model output: (batch_size, pred_len, num_stocks)
        assert predictions_np.shape == (2, 1, 2)
        # targets_np from loader: (batch_size, pred_len, num_stocks)
        assert targets_np.shape == (2, 1, 2)

        # Check metrics dictionary structure
        expected_metrics_keys = [
            'test_loss', 'mse_per_stock', 'mape_per_stock',
            'direction_accuracy', 'avg_mse', 'avg_mape', 'avg_direction_accuracy'
        ]
        for key in expected_metrics_keys:
            assert key in metrics
        
        assert isinstance(metrics['test_loss'], float)
        assert len(metrics['mse_per_stock']) == len(mock_y_scalers_eval)
        assert len(metrics['mape_per_stock']) == len(mock_y_scalers_eval)
        # Direction accuracy might be NaN if diff results in zero length array (e.g. 1 data point)
        # For this test, with 2 samples in batch, diff will produce 1 element.
        assert len(metrics['direction_accuracy']) == len(mock_y_scalers_eval)


# --- Tests for visualize_predictions ---
class TestVisualizePredictions:
    @patch('models.evaluate_model.plt') # Mock the entire plt module
    @patch('models.evaluate_model.mlflow') # Mock mlflow
    @patch('models.evaluate_model.Path.mkdir')
    def test_visualize_predictions_calls(self, mock_mkdir, mock_mlflow, mock_plt,
                                         mock_y_scalers_eval, tmp_path):
        num_stocks = len(mock_y_scalers_eval)
        tickers = [f"TICK{i}" for i in range(num_stocks)]
        # Predictions/targets shape: (num_samples, pred_len, num_stocks)
        # Let's use num_points for num_samples to match the slicing in the function
        num_points_to_plot = 5
        predictions = np.random.rand(num_points_to_plot, 1, num_stocks)
        targets = np.random.rand(num_points_to_plot, 1, num_stocks)
        output_plot_dir = tmp_path / "plots"

        visualize_predictions(predictions, targets, mock_y_scalers_eval, tickers, str(output_plot_dir), num_points=num_points_to_plot)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        assert mock_plt.figure.call_count == num_stocks
        assert mock_plt.plot.call_count == num_stocks * 2 # Actual and Predicted for each stock
        assert mock_plt.title.call_count == num_stocks
        assert mock_plt.xlabel.call_count == num_stocks
        assert mock_plt.ylabel.call_count == num_stocks
        assert mock_plt.legend.call_count == num_stocks
        assert mock_plt.grid.call_count == num_stocks
        assert mock_plt.savefig.call_count == num_stocks
        assert mock_plt.close.call_count == num_stocks
        assert mock_mlflow.log_artifact.call_count == num_stocks

        # Check if savefig and log_artifact were called with correct paths
        for i, ticker in enumerate(tickers):
            expected_plot_path = output_plot_dir / f'prediction_{ticker}.png'
            # Check savefig call
            assert call(expected_plot_path) in mock_plt.savefig.call_args_list
            # Check mlflow log_artifact call
            assert call(str(expected_plot_path)) in mock_mlflow.log_artifact.call_args_list

    @patch('models.evaluate_model.plt')
    @patch('models.evaluate_model.mlflow')
    @patch('models.evaluate_model.Path.mkdir')
    def test_visualize_predictions_mlflow_exception(self, mock_mkdir, mock_mlflow, mock_plt,
                                                    mock_y_scalers_eval, tmp_path, caplog):
        num_stocks = 1
        tickers = ["TICKA"]
        predictions = np.random.rand(5, 1, num_stocks)
        targets = np.random.rand(5, 1, num_stocks)
        output_plot_dir = tmp_path / "plots_err"

        mock_mlflow.log_artifact.side_effect = Exception("MLflow down")

        # --- MODIFIED: Capture logs at WARNING level for the test ---
        with caplog.at_level(logging.WARNING):
            visualize_predictions(predictions, targets, [mock_y_scalers_eval[0]], tickers, str(output_plot_dir))
        # --- END MODIFIED ---
        
        assert "Could not log artifact" in caplog.text # --- MODIFIED: Check specific substring ---
        assert "MLflow down" in caplog.text
        mock_plt.savefig.assert_called_once() # Still tries to save