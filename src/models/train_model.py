import argparse
import json
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import os
import sys
import logging
from pathlib import Path
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from model_definitions import StockLSTM, StockLSTMWithAttention, StockLSTMWithCrossStockAttention
from evaluate_model import evaluate_model, visualize_predictions
from src.utils.db_utils import (
    get_db_connection,
    load_scaled_features,
    load_scalers,
    load_processed_features_from_db,
    load_optimization_results,
    save_prediction
)

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --------------------------------------------------------------

def train_final_model(
    dataset_run_id: str, # Added for logging purposes
    best_params: dict, 
    X_train, y_train, X_test, y_test, 
    num_features, num_stocks, 
    y_scalers, tickers, 
    device, training_epochs, plot_output_dir,
    mlflow_experiment_name: str # Pass experiment name
    ):
    """Train the final model with the best hyperparameters and log to MLflow"""
    batch_size = best_params['batch_size']
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    model_type = best_params.get('model_type', 'lstm') # Default to 'lstm' if not in best_params
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Handle potentially empty test set
    test_loader = None
    if X_test is not None and X_test.size > 0 and y_test is not None and y_test.size > 0:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        logger.warning("Test data is empty or None. Model will be trained but not evaluated on a test set during this training run.")

    if model_type == 'lstm':
        model = StockLSTM(num_stocks=num_stocks, num_features=num_features, hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate).to(device)
    elif model_type == 'lstm_attention':
        model = StockLSTMWithAttention(num_stocks=num_stocks, num_features=num_features, hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate).to(device)
    elif model_type == 'lstm_cross_attention':
        model = StockLSTMWithCrossStockAttention(num_stocks=num_stocks, num_features=num_features, hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # MLflow: Experiment should be set before starting the run
    mlflow.set_experiment(mlflow_experiment_name)
    
    # More descriptive run name
    mlflow_run_name = f"final_train_{model_type}_{dataset_run_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    with mlflow.start_run(run_name=mlflow_run_name) as run: # Use the specific run name
        mlflow_model_run_id = run.info.run_id # Get the MLflow run ID for this training
        logger.info(f"--- MLflow Run Started for Final Training (MLflow Run ID: {mlflow_model_run_id}) ---")
        
        mlflow.log_param("dataset_run_id", dataset_run_id) # Log the data version used
        mlflow.log_params(best_params)
        mlflow.log_param('num_stocks', num_stocks)
        mlflow.log_param('num_features', num_features)
        mlflow.log_param('tickers_used_for_training', str(tickers))
        mlflow.log_param('final_training_epochs_config', training_epochs)

        best_model_state = None
        best_test_loss = float('inf')
        actual_epochs_trained = 0
    
        for epoch in range(training_epochs):
            actual_epochs_trained += 1
            model.train()
            total_train_loss = 0
            for batch_sequences, batch_targets in train_loader:
                batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(device)
                optimizer.zero_grad()
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(train_loader)
            mlflow.log_metric('train_loss', avg_train_loss, step=epoch)
            
            current_test_loss = float('nan') # Default if no test set
            if test_loader:
                # Evaluate on test set
                # evaluate_model returns: predictions_np, targets_np, metrics_dict
                _, _, eval_metrics = evaluate_model(model, test_loader, criterion, y_scalers, device)
                current_test_loss = eval_metrics['test_loss']
                mlflow.log_metric('test_loss', current_test_loss, step=epoch)
                mlflow.log_metric('avg_mse_test', eval_metrics['avg_mse'], step=epoch)
                mlflow.log_metric('avg_mape_test', eval_metrics['avg_mape'], step=epoch)
                mlflow.log_metric('avg_direction_accuracy_test', eval_metrics['avg_direction_accuracy'], step=epoch)
                
                if current_test_loss < best_test_loss:
                    best_test_loss = current_test_loss
                    best_model_state = model.state_dict().copy() # Save the best model state
                    mlflow.set_tag("best_epoch", epoch + 1)
            else: # No test loader, save model from last epoch or based on train loss (less ideal)
                if epoch == training_epochs -1 : # Save last epoch if no test set
                    best_model_state = model.state_dict().copy()

            logger.info(f'Epoch [{epoch+1}/{training_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {current_test_loss:.4f}')
        
        mlflow.log_param('actual_epochs_trained', actual_epochs_trained)

        if best_model_state:
            model.load_state_dict(best_model_state)
            logger.info("Loaded best model state_dict from training run based on test loss (or last epoch if no test set).")
        else: # Should not happen if training_epochs > 0
            logger.warning("No best model state_dict saved. Using model from last iteration.")

        final_metrics_on_test = {}
        predictions_on_test = None # For saving to DB
        if test_loader:
            predictions_on_test, _, final_metrics_on_test = evaluate_model(model, test_loader, criterion, y_scalers, device)
            mlflow.log_metrics({
                'final_test_loss': final_metrics_on_test['test_loss'],
                'final_avg_mse_test': final_metrics_on_test['avg_mse'],
                'final_avg_mape_test': final_metrics_on_test['avg_mape'],
                'final_avg_direction_accuracy_test': final_metrics_on_test['avg_direction_accuracy']
            })
            logger.info("Final model metrics on test set logged to MLflow.")
            visualize_predictions(predictions_on_test, y_test, y_scalers, tickers, plot_output_dir, num_points=20) # y_test is from outer scope
            logger.info("Visualizations generated and logged to MLflow artifacts.")
        else:
            logger.info("No test set evaluation performed in this training run.")


        logger.info("Logging PyTorch model to MLflow...")
        # Log model with a registered model name for easier versioning and deployment
        registered_model_name = mlflow_experiment_name # Or a more specific name like "StockPredictorLSTM"
        mlflow.pytorch.log_model(
            pytorch_model=model, 
            artifact_path="model", # Standard path within the run
            registered_model_name=registered_model_name # This will create the model in registry if not exists
        )
        logger.info(f"Model logged to MLflow with run_id {mlflow_model_run_id} and registered as '{registered_model_name}'.")
        logger.info("--- MLflow Run Finished ---")
        
    # Return the trained model, its predictions on test set (if any), final metrics, and its MLflow run ID
    return model, predictions_on_test, final_metrics_on_test, mlflow_model_run_id

# -------------------------------------------------------------

def run_training(config_path: str, run_id_arg: str) -> Optional[str]:

    """
    Run model training using data and parameters associated with a specific run_id.
    Args:
        config_path (str): Path to the params.yaml configuration file.
        run_id_arg (str): The specific run_id for the dataset (scaled features, scalers, best_params).
    Returns:
        Optional[str]: The MLflow run ID of the trained model if successful, None otherwise.
    """

    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        db_config = params['database']
        
        current_dataset_run_id = run_id_arg # This is the run_id for the DATA/FEATURES
        logger.info(f"Model training using dataset from run_id: {current_dataset_run_id}")

        mlflow_config = params['mlflow']
        mlflow_experiment_name = mlflow_config['experiment_name']
        # MLFLOW_TRACKING_URI should be set as an environment variable or in params
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_config.get('tracking_uri', 'http://mlflow-server:5000'))
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        training_params_cfg = params['training']
        final_training_epochs = training_params_cfg['epochs']
        
        # Define plot output directory relative to script or an absolute path for container
        # Example: plots will be saved in <project_root>/plots/
        # Ensure this path is writable by the process running the script.
        plot_output_dir = Path(params.get('output_paths', {}).get('training_plots_dir', 'plots/training_plots'))
        plot_output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # 1. Load Scaled Data from database using current_dataset_run_id
        logger.info(f"--- Loading Scaled Data from database for dataset_run_id: {current_dataset_run_id} ---")
        X_train_scaled = load_scaled_features(db_config, current_dataset_run_id, 'X_train')
        y_train_scaled = load_scaled_features(db_config, current_dataset_run_id, 'y_train')
        X_test_scaled = load_scaled_features(db_config, current_dataset_run_id, 'X_test') # Might be None/empty
        y_test_scaled = load_scaled_features(db_config, current_dataset_run_id, 'y_test') # Might be None/empty
        
        if X_train_scaled is None or y_train_scaled is None:
            logger.error(f"Failed to load scaled training data for dataset_run_id: {current_dataset_run_id}.")
            return None
        logger.info("--- Finished Loading Scaled Data ---")

        # 2. Load Scalers from database using current_dataset_run_id
        logger.info(f"--- Loading Scalers from database for dataset_run_id: {current_dataset_run_id} ---")
        scalers_dict = load_scalers(db_config, current_dataset_run_id)
        if scalers_dict is None or 'y_scalers' not in scalers_dict:
            logger.error(f"Failed to load scalers or 'y_scalers' not found for dataset_run_id: {current_dataset_run_id}.")
            return None
        y_scalers = scalers_dict['y_scalers']
        tickers = scalers_dict.get('tickers', []) # Get tickers from scalers_dict
        if not tickers: # Fallback to processed_feature_data if not in scalers
            logger.warning("Tickers not found in scalers_dict, attempting to load from processed_feature_data.")
            feature_data = load_processed_features_from_db(db_config, current_dataset_run_id)
            if feature_data and 'tickers' in feature_data:
                tickers = feature_data['tickers']
            else:
                logger.error(f"Failed to load tickers for dataset_run_id: {current_dataset_run_id}.")
                return None
        logger.info(f"Using tickers: {tickers}")
        logger.info("--- Finished Loading Scalers ---")

        # 3. Load Best Hyperparameters from database using current_dataset_run_id
        logger.info(f"--- Loading Best Hyperparameters from database for dataset_run_id: {current_dataset_run_id} ---")
        best_hyperparams = load_optimization_results(db_config, current_dataset_run_id)
        if best_hyperparams is None:
            logger.error(f"Failed to load best hyperparameters for dataset_run_id: {current_dataset_run_id}.")
            # Decide: proceed with default params or fail? For now, fail.
            return None
        logger.info(f"Best hyperparameters loaded: {best_hyperparams}")
        logger.info("--- Finished Loading Best Hyperparameters ---")

        if X_train_scaled.ndim < 4:
             logger.error(f"X_train_scaled has unexpected dimensions: {X_train_scaled.ndim}. Expected 4.")
             return None
        num_stocks = X_train_scaled.shape[2]
        num_features = X_train_scaled.shape[3]

        # 4. Train Final Model
        logger.info("--- Starting Final Model Training ---")
        final_model_obj, test_predictions_np, final_test_metrics, mlflow_model_run_id = train_final_model(
            dataset_run_id=current_dataset_run_id, # Pass the dataset_run_id for logging
            best_params=best_hyperparams, 
            X_train=X_train_scaled, y_train=y_train_scaled, 
            X_test=X_test_scaled, y_test=y_test_scaled, # Pass y_test_scaled here
            num_features=num_features, num_stocks=num_stocks, 
            y_scalers=y_scalers, tickers=tickers, 
            device=device, training_epochs=final_training_epochs, 
            plot_output_dir=plot_output_dir,
            mlflow_experiment_name=mlflow_experiment_name
        )
        logger.info(f"--- Finished Final Model Training. MLflow Run ID for trained model: {mlflow_model_run_id} ---")

        if final_test_metrics: # If test evaluation was done
            logger.info("\nFinal Training Summary (on test set):")
            logger.info("========================")
            for metric_name, value in final_test_metrics.items():
                if isinstance(value, (int, float)): logger.info(f"{metric_name}: {value:.4f}")
            logger.info("========================")
        
        # 5. Save test set predictions (if made) to database
        # These are predictions made by THIS trained model on ITS test set.
        # Not to be confused with daily operational predictions.
        if test_predictions_np is not None and test_predictions_np.size > 0:
            logger.info(f"--- Saving Test Set Predictions from this Training Run (Model MLflow Run ID: {mlflow_model_run_id}) to Database ---")
            try:
                # test_predictions_np shape: (num_test_samples, pred_len, num_stocks)
                # We usually save the last prediction for each stock if pred_len > 1, or the only one if pred_len=1
                for stock_idx, ticker_name in enumerate(tickers):
                    # Assuming pred_len is 1, so test_predictions_np[:, 0, stock_idx]
                    # If you want to save all test predictions, the DB schema/logic would need to change.
                    # For now, let's save the *last* prediction from the test set for each stock as an example.
                    if test_predictions_np.shape[0] > 0 : # If there are any test samples
                        last_pred_on_test_for_ticker_scaled = test_predictions_np[-1, 0, stock_idx]
                        
                        # Inverse transform this single prediction
                        # Reshape for scaler: needs to be 2D array-like for inverse_transform
                        last_pred_on_test_for_ticker_inv = y_scalers[stock_idx].inverse_transform(
                            np.array([[last_pred_on_test_for_ticker_scaled]])
                        )[0][0]

                        # The save_prediction function expects a single predicted price.
                        # The 'prediction_timestamp' in the DB table is auto-set to CURRENT_TIMESTAMP.
                        # This is okay for logging test results, but for actual forecasts, the date matters.
                        # For now, this is just logging a sample prediction from the test set.
                        save_prediction(db_config, ticker_name, float(last_pred_on_test_for_ticker_inv), mlflow_model_run_id)
                        logger.info(f"Saved last test set prediction for {ticker_name}: {float(last_pred_on_test_for_ticker_inv):.4f}")
                logger.info(f"Successfully saved sample test set predictions for {len(tickers)} tickers.")
            except Exception as e_save:
                logger.error(f"Error saving test set predictions from training run: {e_save}", exc_info=True)
        else:
            logger.info("No test set predictions to save from this training run.")

        logger.info(f"--- Training process for dataset_run_id {current_dataset_run_id} complete. Trained model MLflow Run ID: {mlflow_model_run_id} ---")
        return mlflow_model_run_id # Return the MLflow run ID of the trained model
    except Exception as e:
        logger.error(f"Error in run_training for dataset_run_id {run_id_arg}: {e}", exc_info=True)
        return None

# -------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model training script for stock prediction.")
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
        help='The run_id of the dataset (scaled features, scalers, best_params from DB) to use for training.'
    )
    args = parser.parse_args()
    config_path_arg = args.config
    cli_run_id_arg = args.run_id

    config_path_resolved = Path(config_path_arg)
    # ... (same config path resolution logic as in build_features.py) ...
    if not config_path_resolved.is_absolute(): # Simplified for brevity
        config_path_resolved = (Path.cwd() / config_path_resolved).resolve()
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)

    logger.info(f"Starting training script with resolved config: {config_path_resolved} for dataset_run_id: {cli_run_id_arg}")

    try:
        with open(config_path_resolved, 'r') as f:
            config = yaml.safe_load(f)
            if 'database' not in config or 'mlflow' not in config: # Basic validation
                logger.error("Database or MLflow configuration missing from params.yaml")
                sys.exit(1)

        trained_model_mlflow_run_id = run_training(str(config_path_resolved), run_id_arg=cli_run_id_arg)
        
        if trained_model_mlflow_run_id:
            logger.info(f"Training completed successfully for dataset_run_id: {cli_run_id_arg}.")
            logger.info(f"MLflow Run ID of the trained model: {trained_model_mlflow_run_id}")
            print(f"TRAINING_SUCCESS_MLFLOW_RUN_ID:{trained_model_mlflow_run_id}") # For Airflow or capture
        else:
            logger.error(f"Training failed for dataset_run_id: {cli_run_id_arg}. Check logs.")
            sys.exit(1)

    except yaml.YAMLError as e_yaml:
        logger.error(f"Error parsing configuration file {config_path_resolved}: {e_yaml}", exc_info=True)
        sys.exit(1)
    except Exception as e_main:
        logger.error(f"Fatal error in training script for dataset_run_id {cli_run_id_arg}: {e_main}", exc_info=True)
        sys.exit(1)
