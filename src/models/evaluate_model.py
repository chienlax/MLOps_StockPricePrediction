# src/models/evaluate_model.py
"""Module for evaluating and visualizing stock price prediction models."""

# Standard library imports
import logging
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def evaluate_model(model, test_loader, criterion, y_scalers, device):
    """
    Evaluate the model and return predictions and metrics.
    
    Args:
        model: The neural network model to evaluate
        test_loader: DataLoader containing test data
        criterion: Loss function
        y_scalers: List of scalers for inverse transforming predictions
        device: Computation device (CPU/GPU)
        
    Returns:
        Tuple of (predictions, targets, metrics)
    """
    model.eval()
    total_test_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_sequences, batch_targets in test_loader:
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)

            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_targets)
            total_test_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())

    # Handle empty loader
    avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0

    # Combine batches
    # Handle case where test_loader was empty
    if not all_predictions or not all_targets:
        # Return empty arrays with appropriate dimensions if possible
        num_stocks = len(y_scalers)
        # Assuming pred_len is 1
        predictions = np.empty((0, 1, num_stocks))
        targets = np.empty((0, 1, num_stocks))
        metrics = {
            'test_loss': avg_test_loss,
            'mse_per_stock': [np.nan] * num_stocks,
            'mape_per_stock': [np.nan] * num_stocks,
            'direction_accuracy': [np.nan] * num_stocks,
            'avg_mse': np.nan,
            'avg_mape': np.nan,
            'avg_direction_accuracy': np.nan
        }
        logger.warning(
            "Test loader was empty or produced no data. Returning NaN metrics."
        )
        return predictions, targets, metrics
        
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)

    # Calculate additional metrics
    metrics = {
        'test_loss': avg_test_loss,
        'mse_per_stock': [],
        'mape_per_stock': [],
        'direction_accuracy': []
    }

    # Transform back to original scale and calculate metrics per stock
    num_stocks = predictions.shape[2]

    for stock_idx in range(num_stocks):
        # Inverse transform
        pred_stock = y_scalers[stock_idx].inverse_transform(
            predictions[:, 0, stock_idx].reshape(-1, 1)
        )
        true_stock = y_scalers[stock_idx].inverse_transform(
            targets[:, 0, stock_idx].reshape(-1, 1)
        )

        # Calculate MSE and MAPE
        mse = mean_squared_error(true_stock, pred_stock)
        try:
            mape = mean_absolute_percentage_error(true_stock, pred_stock)
        # Can happen if true_stock contains zeros
        except ValueError:
            mape = np.nan
            logger.warning(
                f"MAPE calculation for stock {stock_idx} resulted in NaN "
                f"(possibly due to zeros in true values)."
            )

        # Calculate direction accuracy
        # Need at least 2 points to calculate diff
        if len(true_stock) > 1:
            pred_direction = np.diff(pred_stock.flatten())
            true_direction = np.diff(true_stock.flatten())
            # Handle cases where pred_direction or true_direction might be all zeros
            if pred_direction.size > 0 and true_direction.size > 0:
                # More robust: checks if signs are same
                direction_accuracy = np.mean(
                    np.sign(pred_direction) == np.sign(true_direction)
                )
            else:
                direction_accuracy = np.nan
        else:
            direction_accuracy = np.nan
        
        metrics['mse_per_stock'].append(mse)
        metrics['mape_per_stock'].append(mape)
        metrics['direction_accuracy'].append(direction_accuracy)

    # Use nanmean to handle NaN values
    metrics['avg_mse'] = np.nanmean(metrics['mse_per_stock'])
    metrics['avg_mape'] = np.nanmean(metrics['mape_per_stock'])
    metrics['avg_direction_accuracy'] = np.nanmean(metrics['direction_accuracy'])

    return predictions, targets, metrics


def visualize_predictions(
    predictions, targets, y_scalers, tickers, output_dir, num_points=20
):
    """
    Visualize predictions vs actual values for each stock.
    
    Args:
        predictions: Model predictions
        targets: Actual target values
        y_scalers: List of scalers for inverse transforming predictions
        tickers: List of stock ticker symbols
        output_dir: Directory to save visualization plots
        num_points: Number of data points to visualize (default: 20)
    """
    num_stocks = len(tickers)
    # Use Path object directly
    output_dir_path = Path(output_dir)
    # Ensure dir exists
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if predictions.shape[0] < num_points:
        logger.warning(
            f"Number of available predictions ({predictions.shape[0]}) is less than "
            f"num_points ({num_points}). Plotting all available points."
        )
        num_points_to_plot = predictions.shape[0]
    else:
        num_points_to_plot = num_points
        
    if num_points_to_plot == 0:
        logger.warning("No data points to plot in visualize_predictions.")
        return

    for stock_idx in range(num_stocks):
        # Inverse transform
        pred_stock = y_scalers[stock_idx].inverse_transform(
            predictions[-num_points_to_plot:, 0, stock_idx].reshape(-1, 1)
        )
        true_stock = y_scalers[stock_idx].inverse_transform(
            targets[-num_points_to_plot:, 0, stock_idx].reshape(-1, 1)
        )

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(true_stock, label=f'Actual Price ({tickers[stock_idx]})', color='blue')
        plt.plot(pred_stock, label=f'Predicted Price ({tickers[stock_idx]})', color='red')
        plt.title(
            f'Actual vs. Predicted Price for {tickers[stock_idx]} '
            f'(Last {num_points_to_plot} Timesteps)'
        )
        plt.xlabel('Timestep')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Save figure for MLflow
        plt_path = output_dir_path / f'prediction_{tickers[stock_idx]}.png'
        # Pass Path object directly
        plt.savefig(plt_path)
        try:
            # Ensure mlflow.log_artifact gets a string path if it requires it
            # Log artifact to active MLflow run
            mlflow.log_artifact(str(plt_path))
        except Exception as e:
            logger.warning(
                f"Could not log artifact {plt_path} to MLflow. Error: {e}"
            )
            logger.warning(
                "Ensure an MLflow run is active when calling visualize_predictions."
            )
        plt.close()