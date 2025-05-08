import argparse
import json
import pickle
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
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

# -------------------------------------------------------------

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

    # with open(config_path, 'r') as f:
    #     params = yaml.safe_load(f)

    # # Load database configuration
    # db_config = params['database']
    # run_id = params.get('run_id')  # Get run_id if specified
    # mlflow_experiment = params['mlflow']['experiment_name']
    # final_training_epochs = params['training']['epochs']
    # plot_output_dir = Path("./plots")
    # plot_output_dir.mkdir(exist_ok=True)

    # # Device setup
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logger.info(f"Using device: {device}")

    # # 1. Database connection verification
    # logger.info(f"--- Verifying PostgreSQL connection at {db_config['host']}:{db_config['port']} ---")
    # try:
    #     conn = get_db_connection(db_config)
    #     conn.close()
    #     logger.info("--- Database connection verified ---")
    # except Exception as e:
    #     logger.error(f"Database connection failed: {e}")
    #     raise

    # # 2. Load Scaled Data from database
    # logger.info(f"--- Loading Scaled Data from database for run_id: {run_id or 'latest'} ---")
    # X_train_scaled = load_scaled_features(db_config, run_id, 'X_train')
    # y_train_scaled = load_scaled_features(db_config, run_id, 'y_train')
    # X_test_scaled = load_scaled_features(db_config, run_id, 'X_test')
    # y_test_scaled = load_scaled_features(db_config, run_id, 'y_test')
    
    # if None in (X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled):
    #     logger.error("Failed to load necessary scaled features from database.")
    #     raise ValueError("Failed to load scaled features")
    # logger.info("--- Finished Loading Scaled Data ---")

    # # 3. Load Scalers from database
    # logger.info(f"--- Loading Scalers from database for run_id: {run_id or 'latest'} ---")
    # scalers = load_scalers(db_config, run_id)
    # if scalers is None:
    #     logger.error("Failed to load scalers from database.")
    #     raise ValueError("Failed to load scalers")
    # y_scalers = scalers['y_scalers']
    # logger.info("--- Finished Loading Scalers ---")

    # # 4. Load processed feature data to get tickers
    # logger.info(f"--- Loading Feature Data from database for run_id: {run_id or 'latest'} ---")
    # feature_data = load_processed_features_from_db(db_config, run_id)
    # if feature_data is None:
    #     logger.error("Failed to load processed feature data from database.")
    #     raise ValueError("Failed to load processed feature data")
    # tickers = feature_data['tickers']
    # logger.info(f"Tickers: {tickers}")
    # logger.info("--- Finished Loading Feature Data ---")

    # # 5. Load Best Hyperparameters from database
    # logger.info(f"--- Loading Best Hyperparameters from database for run_id: {run_id or 'latest'} ---")
    # best_params = load_optimization_results(db_config, run_id)
    # if best_params is None:
    #     logger.error("Failed to load best hyperparameters from database.")
    #     raise ValueError("Failed to load best hyperparameters")
    # logger.info(f"Best parameters loaded: {best_params}")
    # logger.info("--- Finished Loading Best Hyperparameters ---")

    # num_stocks = y_train_scaled.shape[2]
    # num_features = X_train_scaled.shape[3]

    # # 6. Set up MLflow
    # print(f"--- Setting MLflow Experiment: {mlflow_experiment} ---")
    # # Tracking URI should be set by Airflow/Docker environment variable
    # mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow-server:5000')
    
    # print(f"Using MLflow Tracking URI: {mlflow_tracking_uri}")
    # mlflow.set_tracking_uri(mlflow_tracking_uri)
    # mlflow.set_experiment(mlflow_experiment)

    # # 7. Train Final Model
    # print("--- Starting Final Model Training ---")
    # final_model, predictions, targets, final_metrics, mlflow_run_id = train_final_model(
    #     best_params, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
    #     num_features, num_stocks, y_scalers, tickers, device, final_training_epochs, plot_output_dir
    # )
    # print("--- Finished Final Model Training ---")

    # print("\nFinal Training Summary:")
    # print("========================")
    # for metric_name, value in final_metrics.items():
    #      if isinstance(value, (int, float)):
    #          print(f"{metric_name}: {value:.4f}")
    #      # else: print(f"{metric_name}: {value}") # Print lists if needed
    # print("========================")
    
    # # 8. Save predictions to database
    # logger.info("--- Saving Predictions to Database ---")
    # try:
    #     for i, ticker in enumerate(tickers):
    #         if i < len(predictions):
    #             last_prediction = predictions[i][-1]
    #             save_prediction(db_config, ticker, float(last_prediction), mlflow_run_id)
    #             logger.info(f"Saved prediction for {ticker}: {float(last_prediction):.4f}")
    #         else:
    #             logger.warning(f"No prediction data found for ticker {ticker}")
                
    #     logger.info(f"Successfully saved predictions for {len(tickers)} tickers")
    # except Exception as e:
    #     logger.error(f"Error saving predictions: {e}")
    #     raise

    # logger.info("--- Training Complete ---")

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
    # try:
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--config', type=str, 
    #                       required=True, 
    #                       help='Path to the configuration file (params.yaml)')
    #     args = parser.parse_args()

    #     # Verify config file exists
    #     if not os.path.exists(args.config):
    #         logger.error(f"Configuration file not found: {args.config}")
    #         sys.exit(1)

    #     # Verify database configuration
    #     with open(args.config, 'r') as f:
    #         config = yaml.safe_load(f)
    #         if 'database' not in config:
    #             logger.error("Database configuration missing from params.yaml")
    #             sys.exit(1)
    #         required_db_fields = ['dbname', 'user', 'password', 'host', 'port']
    #         missing_fields = [field for field in required_db_fields 
    #                         if field not in config['database']]
    #         if missing_fields:
    #             logger.error(f"Missing required database fields: {missing_fields}")
    #             sys.exit(1)

    #     run_training(args.config)
    # except Exception as e:
    #     logger.error(f"Error in training process: {e}", exc_info=True)
    #     sys.exit(1)

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
