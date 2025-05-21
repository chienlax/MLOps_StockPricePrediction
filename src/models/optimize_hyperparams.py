# src/models/optimize_hyperparams.py
import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import optuna
import logging
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

try:
    from .model_definitions import StockLSTM, StockLSTMWithAttention, StockLSTMWithCrossStockAttention
    from .evaluate_model import evaluate_model
except ImportError: # Fallback for direct execution if src/models is not a package in PYTHONPATH
    from model_definitions import StockLSTM, StockLSTMWithAttention, StockLSTMWithCrossStockAttention
    from evaluate_model import evaluate_model

try:
    from src.utils.db_utils import (
        get_db_connection,
        load_scaled_features,
        load_scalers,
        save_optimization_results
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1])) # Add 'src' to path
    from utils.db_utils import (
        load_scaled_features,
        load_scalers,
        save_optimization_results
    )

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ------------------------------------------------------------------

def objective(trial, X_train, y_train, X_test, y_test, 
              num_features, num_stocks, y_scalers, 
              device, optimization_epochs, patience):
    """Optuna objective function for hyperparameter optimization"""
    
    # Define hyperparameters to tune
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # Model selection as a hyperparameter
    model_type = trial.suggest_categorical('model_type', ['lstm', 'lstm_attention', 'lstm_cross_attention'])
    
    # Create PyTorch datasets and loaders
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model based on type
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
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    num_epochs = optimization_epochs
    best_val_loss = float('inf')
    patience_counter = 0
    patience = patience 
    
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
        _, _, metrics = evaluate_model(model, test_loader, criterion, y_scalers, device)
        
        # Early stopping check
        if metrics['test_loss'] < best_val_loss:
            best_val_loss = metrics['test_loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    return best_val_loss

# ------------------------------------------------------------------

def run_optimization(config_path: str, run_id_arg: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Run hyperparameter optimization using data from a specific run_id.
    Args:
        config_path (str): Path to the params.yaml configuration file.
        run_id_arg (str): The specific run_id for the scaled dataset to use.
    Returns:
        tuple[Optional[dict], Optional[str]]: (best_params, run_id_arg) if successful, (None, None) otherwise.
    """
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        # Load database configuration
        db_config = params['database']
        logger.info(f"Using PostgreSQL database at {db_config['host']}:{db_config['port']}")
        
        current_run_id = run_id_arg
        logger.info(f"Targeting dataset with run_id: {current_run_id} for hyperparameter optimization.")

        opt_params = params['optimization']
        n_trials = opt_params['n_trials']
        optimization_epochs = opt_params.get('epochs', 20) # Epochs per Optuna trial
        patience = opt_params.get('patience', 5) # Early stopping patience per trial

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # 1. Load Scaled Data from database using current_run_id
        logger.info(f"--- Loading Scaled Data from database for run_id: {current_run_id} ---")
        X_train_scaled = load_scaled_features(db_config, current_run_id, 'X_train')
        y_train_scaled = load_scaled_features(db_config, current_run_id, 'y_train')
        X_test_scaled = load_scaled_features(db_config, current_run_id, 'X_test')
        y_test_scaled = load_scaled_features(db_config, current_run_id, 'y_test')
        
        if X_train_scaled is None or y_train_scaled is None:
            logger.error(f"Failed to load scaled training data (X_train or y_train) for run_id: {current_run_id}.")
            return None, None
        if X_test_scaled is None or y_test_scaled is None:
            logger.warning(f"Scaled test data (X_test or y_test) not found or empty for run_id: {current_run_id}. Optimization will proceed without it if objective allows.")
            # Ensure objective function can handle empty X_test/y_test or use a validation split from train
            # For now, assuming objective needs X_test, y_test. If they are critical, make this an error.
            if X_test_scaled is None or X_test_scaled.size == 0 : X_test_scaled = np.array([]) # Ensure they are empty arrays if None
            if y_test_scaled is None or y_test_scaled.size == 0 : y_test_scaled = np.array([])


        logger.info(f"X_train_scaled shape: {X_train_scaled.shape}, y_train_scaled shape: {y_train_scaled.shape}")
        logger.info(f"X_test_scaled shape: {X_test_scaled.shape if X_test_scaled.size > 0 else 'Empty'}, y_test_scaled shape: {y_test_scaled.shape if y_test_scaled.size > 0 else 'Empty'}")
        logger.info("--- Finished Loading Scaled Data ---")

        # 2. Load Scalers from database using current_run_id
        logger.info(f"--- Loading Scalers from database for run_id: {current_run_id} ---")
        scalers_dict = load_scalers(db_config, current_run_id)
        if scalers_dict is None or 'y_scalers' not in scalers_dict:
            logger.error(f"Failed to load scalers or 'y_scalers' not found for run_id: {current_run_id}.")
            return None, None
        y_scalers = scalers_dict['y_scalers']
        # tickers = scalers_dict.get('tickers', []) # If needed by objective or evaluation
        # num_features_from_scaler = scalers_dict.get('num_features') # If needed
        logger.info("--- Finished Loading Scalers ---")

        # Determine num_stocks and num_features from the loaded data
        if X_train_scaled.ndim < 4 : # Expected (samples, seq_len, stocks, features)
            logger.error(f"X_train_scaled has unexpected dimensions: {X_train_scaled.ndim}. Expected 4.")
            return None, None
        num_stocks = X_train_scaled.shape[2]
        num_features = X_train_scaled.shape[3]

        # 3. Run Optuna Study
        logger.info(f"--- Starting Optuna Optimization ({n_trials} trials) ---")
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(
            trial, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
            num_features, num_stocks, y_scalers, device, optimization_epochs, patience
        ), n_trials=n_trials, timeout=params['optimization'].get('timeout_seconds', None)) # Optional timeout
        logger.info("--- Finished Optuna Optimization ---")

        best_params_found = study.best_params
        logger.info(f"Best trial value (loss): {study.best_value}")
        logger.info(f"Best parameters: {best_params_found}")

        # 4. Save Best Parameters to database, associated with current_run_id
        logger.info(f"--- Saving Best Parameters to database for run_id: {current_run_id} ---")
        save_optimization_results(db_config, current_run_id, best_params_found)
        logger.info("--- Finished Saving Best Parameters to database ---")
        
        # Save to local file if specified in params.yaml (optional)
        output_paths_config = params.get('output_paths', {})
        if 'best_params_path' in output_paths_config:
            # This path should be absolute or relative to where Airflow runs the script
            # For Docker, this path needs to be accessible within the container.
            # Example: /opt/airflow/config/best_params.json (if config is mounted at /opt/airflow/config)
            best_params_file_path_str = output_paths_config['best_params_path']
            # Ensure the path is treated as being inside the container if run by Airflow
            # If running locally, it's just a local path.
            # For now, assume it's a path accessible by the script.
            best_params_output_path = Path(best_params_file_path_str)
            try:
                best_params_output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(best_params_output_path, 'w') as f:
                    json.dump(best_params_found, f, indent=4)
                logger.info(f"--- Also saved Best Parameters to file: {best_params_output_path} ---")
            except Exception as e_file:
                logger.warning(f"Could not save best_params to file {best_params_output_path}: {e_file}")

        return best_params_found, current_run_id
    except Exception as e:
        logger.error(f"Error in run_optimization for run_id {run_id_arg}: {e}", exc_info=True)
        return None, None

# ------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter optimization script for stock prediction.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/params.yaml',
        help='Path to the configuration file (e.g., config/params.yaml)'
    )
    parser.add_argument(
        '--run_id',
        type=str,
        required=True,
        help='The run_id of the dataset (scaled features, scalers from DB) to use for optimization.'
    )
    args = parser.parse_args()
    config_path_arg = args.config
    cli_run_id_arg = args.run_id

    config_path_resolved = Path(config_path_arg)
    if not config_path_resolved.is_absolute(): # Simplified for brevity
        config_path_resolved = (Path.cwd() / config_path_resolved).resolve()
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)

    logger.info(f"Starting optimization script with resolved config: {config_path_resolved} for run_id: {cli_run_id_arg}")

    try:
        with open(config_path_resolved, 'r') as f:
            config = yaml.safe_load(f)
            if 'database' not in config: # Basic validation
                logger.error("Database configuration missing from params.yaml")
                sys.exit(1)

        best_params, used_run_id = run_optimization(str(config_path_resolved), run_id_arg=cli_run_id_arg)
        
        if best_params and used_run_id:
            logger.info(f"Optimization completed successfully for run_id: {used_run_id}.")
            logger.info(f"Best parameters found: {best_params}")
            print(f"OPTIMIZATION_SUCCESS_RUN_ID:{used_run_id}") # For Airflow or capture
        else:
            logger.error(f"Optimization failed for run_id: {cli_run_id_arg}. Check logs.")
            sys.exit(1)

    except yaml.YAMLError as e_yaml:
        logger.error(f"Error parsing configuration file {config_path_resolved}: {e_yaml}", exc_info=True)
        sys.exit(1)
    except Exception as e_main:
        logger.error(f"Fatal error in optimization script for run_id {cli_run_id_arg}: {e_main}", exc_info=True)
        sys.exit(1)
