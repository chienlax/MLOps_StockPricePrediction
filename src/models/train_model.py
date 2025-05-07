import argparse
import json
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from pathlib import Path
import os
import sys
import logging

from model_definitions import StockLSTM, StockLSTMWithAttention, StockLSTMWithCrossStockAttention
from evaluate_model import evaluate_model, visualize_predictions
# Import the db_utils file
from src.utils.db_utils import (
    get_db_connection,
    load_scaled_features,
    load_scalers,
    load_processed_features_from_db,
    load_optimization_results,
    save_prediction
)

# Setup logging
logger = logging.getLogger(__name__)
# Basic logging setup if not configured globally
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def train_final_model(best_params, X_train, y_train, X_test, y_test, num_features, num_stocks, y_scalers, tickers, device, training_epochs, plot_output_dir):
    """Train the final model with the best hyperparameters and log to MLflow"""
    # Extract parameters
    batch_size = best_params['batch_size']
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    model_type = best_params['model_type']
    
    # Prepare data loaders
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with best parameters and type
    if model_type == 'lstm':
        model = StockLSTM(
            num_stocks=num_stocks, 
            num_features=num_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout_rate=dropout_rate
        ).to(device)
    elif model_type == 'lstm_attention':
        model = StockLSTMWithAttention(
            num_stocks=num_stocks, 
            num_features=num_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout_rate=dropout_rate
        ).to(device)
    elif model_type == 'lstm_cross_attention':
        model = StockLSTMWithCrossStockAttention(
            num_stocks=num_stocks, 
            num_features=num_features, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            dropout_rate=dropout_rate
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # MLflow tracking for final model
    # Note: Experiment is set outside, run is started here
    with mlflow.start_run(run_name="final_model_training"): # Use run name from config?
        print("--- MLflow Run Started for Final Training ---")
        # Log best parameters from Optuna
        mlflow.log_params(best_params)
        mlflow.log_param('num_stocks', num_stocks)
        mlflow.log_param('num_features', num_features)
        mlflow.log_param('tickers', str(tickers))
        mlflow.log_param('final_training_epochs', training_epochs)

        # Full training loop
        num_epochs = training_epochs # Use the parameter
        best_model = None
        best_test_loss = float('inf')
    
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            
            for batch_sequences, batch_targets in train_loader:
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Evaluate on test set
            predictions, targets, metrics = evaluate_model(model, test_loader, criterion, y_scalers, device)
            
            # Log metrics
            mlflow.log_metric('train_loss', avg_train_loss, step=epoch)
            mlflow.log_metric('test_loss', metrics['test_loss'], step=epoch)
            mlflow.log_metric('avg_mse', metrics['avg_mse'], step=epoch)
            mlflow.log_metric('avg_mape', metrics['avg_mape'], step=epoch)
            mlflow.log_metric('avg_direction_accuracy', metrics['avg_direction_accuracy'], step=epoch)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {metrics["test_loss"]:.4f}')
            
            # Save best model
            if metrics['test_loss'] < best_test_loss:
                best_test_loss = metrics['test_loss']
                best_model = model.state_dict().copy()

        # Load best model state dict found during training
        if best_model is not None:
            model.load_state_dict(best_model)
            print("Loaded best model state_dict from training run.")
        else:
            print("Warning: No best model state_dict saved during training.")

        # Final evaluation on test set
        predictions, targets, final_metrics = evaluate_model(model, test_loader, criterion, y_scalers, device)

        # Log final metrics
        mlflow.log_metrics({
            'final_test_loss': final_metrics['test_loss'],
            'final_avg_mse': final_metrics['avg_mse'],
            'final_avg_mape': final_metrics['avg_mape'],
            'final_avg_direction_accuracy': final_metrics['avg_direction_accuracy']
        })
        print("Final model metrics logged to MLflow.")

        # Log model artifact
        print("Logging PyTorch model to MLflow...")
        mlflow.pytorch.log_model(model, "model") # Logs the model structure and state_dict
        print("Model logged.")

        # Visualize predictions and log plots
        print("Generating prediction visualizations...")
        # We need the original tickers list here
        visualize_predictions(predictions, targets, y_scalers, tickers, plot_output_dir)
        print("Visualizations generated and logged.")

        print("--- MLflow Run Finished ---")
        
        # Get the MLflow run ID to use it as a reference
        mlflow_run_id = mlflow.active_run().info.run_id

    return model, predictions, targets, final_metrics, mlflow_run_id


def run_training(config_path: str):
    """Run model training using PostgreSQL database."""
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)

    # Load database configuration
    db_config = params['database']
    run_id = params.get('run_id')  # Get run_id if specified
    mlflow_experiment = params['mlflow']['experiment_name']
    final_training_epochs = params['training']['epochs']
    plot_output_dir = Path("./plots")
    plot_output_dir.mkdir(exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. Database connection verification
    logger.info(f"--- Verifying PostgreSQL connection at {db_config['host']}:{db_config['port']} ---")
    try:
        conn = get_db_connection(db_config)
        conn.close()
        logger.info("--- Database connection verified ---")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

    # 2. Load Scaled Data from database
    logger.info(f"--- Loading Scaled Data from database for run_id: {run_id or 'latest'} ---")
    X_train_scaled = load_scaled_features(db_config, run_id, 'X_train')
    y_train_scaled = load_scaled_features(db_config, run_id, 'y_train')
    X_test_scaled = load_scaled_features(db_config, run_id, 'X_test')
    y_test_scaled = load_scaled_features(db_config, run_id, 'y_test')
    
    if None in (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
        logger.error("Failed to load necessary scaled features from database.")
        raise ValueError("Failed to load scaled features")
    logger.info("--- Finished Loading Scaled Data ---")

    # 3. Load Scalers from database
    logger.info(f"--- Loading Scalers from database for run_id: {run_id or 'latest'} ---")
    scalers = load_scalers(db_config, run_id)
    if scalers is None:
        logger.error("Failed to load scalers from database.")
        raise ValueError("Failed to load scalers")
    y_scalers = scalers['y_scalers']
    logger.info("--- Finished Loading Scalers ---")

    # 4. Load processed feature data to get tickers
    logger.info(f"--- Loading Feature Data from database for run_id: {run_id or 'latest'} ---")
    feature_data = load_processed_features_from_db(db_config, run_id)
    if feature_data is None:
        logger.error("Failed to load processed feature data from database.")
        raise ValueError("Failed to load processed feature data")
    tickers = feature_data['tickers']
    logger.info(f"Tickers: {tickers}")
    logger.info("--- Finished Loading Feature Data ---")

    # 5. Load Best Hyperparameters from database
    logger.info(f"--- Loading Best Hyperparameters from database for run_id: {run_id or 'latest'} ---")
    best_params = load_optimization_results(db_config, run_id)
    if best_params is None:
        logger.error("Failed to load best hyperparameters from database.")
        raise ValueError("Failed to load best hyperparameters")
    logger.info(f"Best parameters loaded: {best_params}")
    logger.info("--- Finished Loading Best Hyperparameters ---")

    num_stocks = y_train_scaled.shape[2]
    num_features = X_train_scaled.shape[3]

    # 6. Set up MLflow
    print(f"--- Setting MLflow Experiment: {mlflow_experiment} ---")
    # Tracking URI should be set by Airflow/Docker environment variable
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
    
    print(f"Using MLflow Tracking URI: {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    # 7. Train Final Model
    print("--- Starting Final Model Training ---")
    final_model, predictions, targets, final_metrics, mlflow_run_id = train_final_model(
        best_params, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        num_features, num_stocks, y_scalers, tickers, device, final_training_epochs, plot_output_dir
    )
    print("--- Finished Final Model Training ---")

    print("\nFinal Training Summary:")
    print("========================")
    for metric_name, value in final_metrics.items():
         if isinstance(value, (int, float)):
             print(f"{metric_name}: {value:.4f}")
         # else: print(f"{metric_name}: {value}") # Print lists if needed
    print("========================")
    
    # 8. Save predictions to database
    logger.info("--- Saving Predictions to Database ---")
    try:
        for i, ticker in enumerate(tickers):
            if i < len(predictions):
                last_prediction = predictions[i][-1]
                save_prediction(db_config, ticker, float(last_prediction), mlflow_run_id)
                logger.info(f"Saved prediction for {ticker}: {float(last_prediction):.4f}")
            else:
                logger.warning(f"No prediction data found for ticker {ticker}")
                
        logger.info(f"Successfully saved predictions for {len(tickers)} tickers")
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        raise

    logger.info("--- Training Complete ---")


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, 
                          required=True, 
                          help='Path to the configuration file (params.yaml)')
        args = parser.parse_args()

        # Verify config file exists
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)

        # Verify database configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if 'database' not in config:
                logger.error("Database configuration missing from params.yaml")
                sys.exit(1)
            required_db_fields = ['dbname', 'user', 'password', 'host', 'port']
            missing_fields = [field for field in required_db_fields 
                            if field not in config['database']]
            if missing_fields:
                logger.error(f"Missing required database fields: {missing_fields}")
                sys.exit(1)

        run_training(args.config)
    except Exception as e:
        logger.error(f"Error in training process: {e}", exc_info=True)
        sys.exit(1)