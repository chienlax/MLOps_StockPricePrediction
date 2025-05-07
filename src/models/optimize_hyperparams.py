# src/models/optimize_hyperparams.py
import argparse
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sys
import optuna
from pathlib import Path
import logging
from datetime import datetime
from model_definitions import StockLSTM, StockLSTMWithAttention, StockLSTMWithCrossStockAttention
from evaluate_model import evaluate_model  # We need evaluation during objective calculation
from src.utils.db_utils import (
    get_db_connection,
    load_scaled_features,
    load_scalers,
    save_optimization_results
)

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def objective(trial, X_train, y_train, X_test, y_test, num_features, num_stocks, y_scalers, device):
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
    num_epochs = 20  # Limited epochs for hyperparameter search
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5  # Early stopping patience
    
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


def run_optimization(config_path: str):
    """Run hyperparameter optimization using PostgreSQL database."""
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)

    # Load database configuration
    db_config = params['database']
    logger.info(f"Using PostgreSQL database at {db_config['host']}:{db_config['port']}")
    
    # Get run_id from params
    run_id = params.get('run_id')
    if not run_id:
        logger.warning("No run_id provided in params, will attempt to use most recent data")
        run_id = None

    n_trials = params['optimization']['n_trials']

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 1. Load Scaled Data from database
    logger.info(f"--- Loading Scaled Data from database ---")
    X_train_scaled = load_scaled_features(db_config, run_id, 'X_train')
    y_train_scaled = load_scaled_features(db_config, run_id, 'y_train')
    X_test_scaled = load_scaled_features(db_config, run_id, 'X_test')
    y_test_scaled = load_scaled_features(db_config, run_id, 'y_test')
    
    if any(data is None for data in [X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled]):
        logger.error("Failed to load scaled data from database")
        raise ValueError("Failed to load scaled data from database")
        
    logger.info(f"X_train_scaled shape: {X_train_scaled.shape}")
    logger.info(f"y_train_scaled shape: {y_train_scaled.shape}")
    logger.info("--- Finished Loading Scaled Data ---")

    # 2. Load Scalers from database
    logger.info(f"--- Loading Scalers from database ---")
    scalers = load_scalers(db_config, run_id)
    if scalers is None:
        logger.error("Failed to load scalers from database")
        raise ValueError("Failed to load scalers from database")
        
    y_scalers = scalers['y_scalers']
    logger.info("--- Finished Loading Scalers ---")


    num_stocks = y_train_scaled.shape[2]
    num_features = X_train_scaled.shape[3]  # Note the index change

    # 3. Run Optuna Study
    logger.info(f"--- Starting Optuna Optimization ({n_trials} trials) ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(
        trial, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        num_features, num_stocks, y_scalers, device
    ), n_trials=n_trials)
    logger.info("--- Finished Optuna Optimization ---")

    # 4. Save Best Parameters to database
    logger.info(f"Best trial value (loss): {study.best_value}")
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"--- Saving Best Parameters to database ---")
    save_optimization_results(db_config, run_id, study.best_params)
    logger.info("--- Finished Saving Best Parameters ---")
    
    # Save to file if specified
    if 'output_paths' in params and 'best_params_path' in params['output_paths']:
        best_params_output_path = Path(params['output_paths']['best_params_path'])
        best_params_output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"--- Also saving Best Parameters to file: {best_params_output_path} ---")
        with open(best_params_output_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
    
    return study.best_params, run_id



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

        run_optimization(args.config)
        logger.info("Optimization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in optimization process: {e}", exc_info=True)
        sys.exit(1)