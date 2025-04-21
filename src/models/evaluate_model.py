# src/models/evaluate_model.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import mlflow
import torch # Needed for no_grad
from pathlib import Path

#%%
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
    
    avg_test_loss = total_test_loss / len(test_loader)
    
    # Combine batches
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
        mape = mean_absolute_percentage_error(true_stock, pred_stock)
        
        # Calculate direction accuracy
        pred_direction = np.diff(pred_stock.flatten())
        true_direction = np.diff(true_stock.flatten())
        direction_accuracy = np.mean((pred_direction * true_direction) > 0)
        
        metrics['mse_per_stock'].append(mse)
        metrics['mape_per_stock'].append(mape)
        metrics['direction_accuracy'].append(direction_accuracy)
    
    metrics['avg_mse'] = np.mean(metrics['mse_per_stock'])
    metrics['avg_mape'] = np.mean(metrics['mape_per_stock'])
    metrics['avg_direction_accuracy'] = np.mean(metrics['direction_accuracy'])
    
    return predictions, targets, metrics

#%%
# --- Modify visualize_predictions to accept an output dir ---
def visualize_predictions(predictions, targets, y_scalers, tickers, output_dir, num_points=20):
    """Visualize predictions vs actual values for each stock"""
    num_stocks = len(tickers)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists

    for stock_idx in range(num_stocks):
        # Inverse transform
        pred_stock = y_scalers[stock_idx].inverse_transform(predictions[-num_points:, 0, stock_idx].reshape(-1, 1))
        true_stock = y_scalers[stock_idx].inverse_transform(targets[-num_points:, 0, stock_idx].reshape(-1, 1))
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(true_stock, label=f'Actual Price ({tickers[stock_idx]})', color='blue')
        plt.plot(pred_stock, label=f'Predicted Price ({tickers[stock_idx]})', color='red')
        plt.title(f'Actual vs. Predicted Price for {tickers[stock_idx]} (Last {num_points} Timesteps)')
        plt.xlabel('Timestep')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)

        # Save figure for MLflow
        plt_path = output_dir / f'prediction_{tickers[stock_idx]}.png'
        plt.savefig(plt_path)
        try:
            mlflow.log_artifact(str(plt_path)) # Log artifact to active MLflow run
        except Exception as e:
            print(f"Warning: Could not log artifact {plt_path} to MLflow. Error: {e}")
            print("Ensure an MLflow run is active when calling visualize_predictions.")
        plt.close()