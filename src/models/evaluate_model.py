# src/models/evaluate_model.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import mlflow
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
if not logger.handlers: # Ensure handlers are not added multiple times if module is reloaded
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # Or your desired default level

def evaluate_model(model, test_loader, criterion, y_scalers, device):
    """Evaluate the model and return predictions and metrics"""
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

    avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0 # Handle empty loader

    # Combine batches
    if not all_predictions or not all_targets: # Handle case where test_loader was empty
        # Return empty arrays with appropriate dimensions if possible, or handle as error
        num_stocks = len(y_scalers)
        predictions = np.empty((0, 1, num_stocks)) # Assuming pred_len is 1
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
        logger.warning("Test loader was empty or produced no data. Returning NaN metrics.")
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
        pred_stock = y_scalers[stock_idx].inverse_transform(predictions[:, 0, stock_idx].reshape(-1, 1))
        true_stock = y_scalers[stock_idx].inverse_transform(targets[:, 0, stock_idx].reshape(-1, 1))

        # Calculate MSE and MAPE
        mse = mean_squared_error(true_stock, pred_stock)
        try:
            mape = mean_absolute_percentage_error(true_stock, pred_stock)
        except ValueError: # Can happen if true_stock contains zeros
            mape = np.nan 
            logger.warning(f"MAPE calculation for stock {stock_idx} resulted in NaN (possibly due to zeros in true values).")


        # Calculate direction accuracy
        if len(true_stock) > 1: # Need at least 2 points to calculate diff
            pred_direction = np.diff(pred_stock.flatten())
            true_direction = np.diff(true_stock.flatten())
            # Handle cases where pred_direction or true_direction might be all zeros (flat line)
            if pred_direction.size > 0 and true_direction.size > 0:
                 direction_accuracy = np.mean(np.sign(pred_direction) == np.sign(true_direction)) # More robust: checks if signs are same
            else:
                direction_accuracy = np.nan
        else:
            direction_accuracy = np.nan
        
        metrics['mse_per_stock'].append(mse)
        metrics['mape_per_stock'].append(mape)
        metrics['direction_accuracy'].append(direction_accuracy)

    metrics['avg_mse'] = np.nanmean(metrics['mse_per_stock']) # Use nanmean
    metrics['avg_mape'] = np.nanmean(metrics['mape_per_stock'])
    metrics['avg_direction_accuracy'] = np.nanmean(metrics['direction_accuracy'])

    return predictions, targets, metrics

def visualize_predictions(predictions, targets, y_scalers, tickers, output_dir, num_points=20):
    """Visualize predictions vs actual values for each stock"""
    num_stocks = len(tickers)
    output_dir_path = Path(output_dir) # Use Path object directly
    output_dir_path.mkdir(parents=True, exist_ok=True) # Ensure dir exists

    if predictions.shape[0] < num_points:
        logger.warning(f"Number of available predictions ({predictions.shape[0]}) is less than num_points ({num_points}). Plotting all available points.")
        num_points_to_plot = predictions.shape[0]
    else:
        num_points_to_plot = num_points
        
    if num_points_to_plot == 0:
        logger.warning("No data points to plot in visualize_predictions.")
        return

    for stock_idx in range(num_stocks):
        # Inverse transform
        pred_stock = y_scalers[stock_idx].inverse_transform(predictions[-num_points_to_plot:, 0, stock_idx].reshape(-1, 1))
        true_stock = y_scalers[stock_idx].inverse_transform(targets[-num_points_to_plot:, 0, stock_idx].reshape(-1, 1))

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(true_stock, label=f'Actual Price ({tickers[stock_idx]})', color='blue')
        plt.plot(pred_stock, label=f'Predicted Price ({tickers[stock_idx]})', color='red')
        plt.title(f'Actual vs. Predicted Price for {tickers[stock_idx]} (Last {num_points_to_plot} Timesteps)')
        plt.xlabel('Timestep')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Save figure for MLflow
        plt_path = output_dir_path / f'prediction_{tickers[stock_idx]}.png'
        plt.savefig(plt_path) # Pass Path object directly
        try:
            # Ensure mlflow.log_artifact gets a string path if it requires it
            mlflow.log_artifact(str(plt_path)) # Log artifact to active MLflow run
        except Exception as e:
            logger.warning(f"Could not log artifact {plt_path} to MLflow. Error: {e}")
            logger.warning("Ensure an MLflow run is active when calling visualize_predictions.")
        plt.close()