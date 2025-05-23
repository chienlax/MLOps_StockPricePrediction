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
        load_scaled_features,
        load_scalers,
        save_optimization_results
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1])) # Add 'src' to path
    from utils.db_utils import (
        get_db_connection,
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
    
    # --- ADDED: Handle potentially empty X_test, y_test more gracefully ---
    if X_test.size > 0 and y_test.size > 0 :
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        # Create an empty loader or handle evaluation differently if test set is empty
        logger.warning("Test data is empty for this Optuna trial. Using a dummy empty loader.")
        # Create a dummy empty TensorDataset and DataLoader
        empty_X = torch.empty(0, X_train_tensor.shape[1], X_train_tensor.shape[2], X_train_tensor.shape[3], dtype=torch.float32)
        empty_y = torch.empty(0, y_train_tensor.shape[1], y_train_tensor.shape[2], dtype=torch.float32)
        test_dataset = torch.utils.data.TensorDataset(empty_X, empty_y)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # --- END ADDED ---

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
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
    else: # Should not happen with categorical suggestion
        raise ValueError(f"Unknown model_type: {model_type}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    num_epochs = optimization_epochs # Use the passed argument
    best_val_loss = float('inf')
    patience_counter = 0

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
        
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        # Evaluate on test set only if test_loader has data
        if len(test_loader.dataset) > 0: # type: ignore
            _, _, metrics = evaluate_model(model, test_loader, criterion, y_scalers, device)
            current_val_loss = metrics['test_loss']
        else: # If no test data, we can't truly validate. Could use train loss or a fixed high loss.
            current_val_loss = avg_train_loss # Or some other strategy if test data is optional
            logger.warning("No test data available for Optuna trial evaluation, using average training loss for this epoch.")

        # Early stopping check
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} for trial {trial.number}")
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
    params: dict = {} # --- MODIFIED: Initialize params ---
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
        patience_optuna = opt_params.get('patience', 5) # Early stopping patience per trial (renamed to avoid conflict with objective's patience)

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
        
        # --- MODIFIED: Ensure X_test_scaled and y_test_scaled are handled if None ---
        if X_test_scaled is None:
            logger.warning(f"X_test_scaled is None for run_id: {current_run_id}. Creating empty array.")
            # Create empty array with correct number of dimensions based on X_train_scaled
            # (0 samples, seq_len, num_stocks, num_features)
            X_test_scaled = np.empty((0, X_train_scaled.shape[1], X_train_scaled.shape[2], X_train_scaled.shape[3]), dtype=X_train_scaled.dtype)
        if y_test_scaled is None:
            logger.warning(f"y_test_scaled is None for run_id: {current_run_id}. Creating empty array.")
            # (0 samples, pred_len, num_stocks)
            y_test_scaled = np.empty((0, y_train_scaled.shape[1], y_train_scaled.shape[2]), dtype=y_train_scaled.dtype)


        logger.info(f"X_train_scaled shape: {X_train_scaled.shape}, y_train_scaled shape: {y_train_scaled.shape}")
        logger.info(f"X_test_scaled shape: {X_test_scaled.shape}, y_test_scaled shape: {y_test_scaled.shape}") # Updated logging for potentially empty arrays
        logger.info("--- Finished Loading Scaled Data ---")

        # 2. Load Scalers from database using current_run_id
        logger.info(f"--- Loading Scalers from database for run_id: {current_run_id} ---")
        scalers_dict = load_scalers(db_config, current_run_id)
        if scalers_dict is None or 'y_scalers' not in scalers_dict:
            logger.error(f"Failed to load scalers or 'y_scalers' not found for run_id: {current_run_id}.")
            return None, None
        y_scalers = scalers_dict['y_scalers']
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
            num_features, num_stocks, y_scalers, device, optimization_epochs, patience_optuna
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
            best_params_file_path_str = output_paths_config['best_params_path']
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
    if not config_path_resolved.is_absolute():
        config_path_resolved = (Path.cwd() / config_path_resolved).resolve()
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)

    logger.info(f"Starting optimization script with resolved config: {config_path_resolved} for run_id: {cli_run_id_arg}")

    try:
        with open(config_path_resolved, 'r') as f:
            config = yaml.safe_load(f)
            if 'database' not in config: 
                logger.error("Database configuration missing from params.yaml")
                sys.exit(1)

        best_params, used_run_id = run_optimization(str(config_path_resolved), run_id_arg=cli_run_id_arg)
        
        if best_params and used_run_id:
            logger.info(f"Optimization completed successfully for run_id: {used_run_id}.")
            logger.info(f"Best parameters found: {best_params}")
            print(f"OPTIMIZATION_SUCCESS_RUN_ID:{used_run_id}") 
        else:
            logger.error(f"Optimization failed for run_id: {cli_run_id_arg}. Check logs.")
            sys.exit(1)

    except yaml.YAMLError as e_yaml:
        logger.error(f"Error parsing configuration file {config_path_resolved}: {e_yaml}", exc_info=True)
        sys.exit(1)
    except Exception as e_main:
        logger.error(f"Fatal error in optimization script for run_id {cli_run_id_arg}: {e_main}", exc_info=True)
        sys.exit(1)