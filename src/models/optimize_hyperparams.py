# src/models/optimize_hyperparams.py
import argparse
import json
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from pathlib import Path
from model_definitions import StockLSTM, StockLSTMWithAttention, StockLSTMWithCrossStockAttention
from evaluate_model import evaluate_model # We need evaluation during objective calculation

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
    
    # Removed mlflow.start_run block
    
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
        
        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Test Loss: {metrics['test_loss']:.6f}")
        
        # Early stopping check
        if metrics['test_loss'] < best_val_loss:
            best_val_loss = metrics['test_loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return best_val_loss


def run_optimization(config_path: str):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)

    # Load paths and parameters
    split_data_path = Path(params['output_paths']['split_data_path'])
    scalers_path = Path(params['output_paths']['scalers_path'])
    best_params_output_path = Path(params['output_paths']['best_params_path'])
    best_params_output_path.parent.mkdir(parents=True, exist_ok=True)

    n_trials = params['optimization']['n_trials']

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Scaled Data
    print(f"--- Loading Scaled Data from {split_data_path} ---")
    split_data = np.load(split_data_path)
    X_train_scaled = split_data['X_train_scaled']
    y_train_scaled = split_data['y_train_scaled']
    X_test_scaled = split_data['X_test_scaled']
    y_test_scaled = split_data['y_test_scaled']
    print("--- Finished Loading Scaled Data ---")

    # 2. Load Scalers
    print(f"--- Loading Scalers from {scalers_path} ---")
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    # scalers_x = scalers['scalers_x'] # Not needed directly for objective
    y_scalers = scalers['y_scalers']
    print("--- Finished Loading Scalers ---")

    num_stocks = y_train_scaled.shape[2]
    num_features = X_train_scaled.shape[3] # Note the index change

    # 3. Run Optuna Study
    print(f"--- Starting Optuna Optimization ({n_trials} trials) ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(
        trial, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        num_features, num_stocks, y_scalers, device
    ), n_trials=n_trials)
    print("--- Finished Optuna Optimization ---")

    # 4. Save Best Parameters
    print(f"Best trial value (loss): {study.best_value}")
    print(f"Best parameters: {study.best_params}")
    print(f"--- Saving Best Parameters to {best_params_output_path} ---")
    with open(best_params_output_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print("--- Finished Saving Best Parameters ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (params.yaml)')
    args = parser.parse_args()
    run_optimization(args.config)