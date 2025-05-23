# src/models/train_model.py
"""Module for training deep learning models for stock price prediction."""

# Standard library imports
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Third-party imports
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader, TensorDataset

# Ensure correct relative imports for package structure
try:
    from .model_definitions import (
        StockLSTM,
        StockLSTMWithAttention,
        StockLSTMWithCrossStockAttention
    )
    from .evaluate_model import evaluate_model, visualize_predictions
except ImportError:
    from model_definitions import (
        StockLSTM,
        StockLSTMWithAttention,
        StockLSTMWithCrossStockAttention
    )
    from evaluate_model import evaluate_model, visualize_predictions

try:
    from src.utils.db_utils import (
        get_db_connection,
        load_scaled_features,
        load_scalers,
        load_processed_features_from_db,
        load_optimization_results,
        save_prediction
    )
except ImportError:
    # Adjust path if running as a script and src is not in PYTHONPATH
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils.db_utils import (
        get_db_connection,
        load_scaled_features,
        load_scalers,
        load_processed_features_from_db,
        load_optimization_results,
        save_prediction
    )


# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:  # Check if handlers are already added
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)  # Changed to INFO for less verbose default logging
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Set logger level


# --------------------------------------------------------------

def train_final_model(
    dataset_run_id: str,
    best_params: dict,
    X_train: np.ndarray, y_train: np.ndarray,  # Expect NumPy arrays
    X_test: Optional[np.ndarray], y_test: Optional[np.ndarray],  # Expect NumPy arrays or None
    num_features: int, num_stocks: int,
    y_scalers: list, tickers: list,
    device: torch.device, training_epochs: int, plot_output_dir: Path,
    mlflow_experiment_name: str
) -> Tuple[nn.Module, Optional[np.ndarray], Dict[str, Any], str]:
    """
    Train the final model with the best hyperparameters and log to MLflow.
    
    Args:
        dataset_run_id: Dataset run ID used for training
        best_params: Best hyperparameters from optimization
        X_train: Training feature data (NumPy array)
        y_train: Training target data (NumPy array)
        X_test: Testing feature data (NumPy array or None)
        y_test: Testing target data (NumPy array or None)
        num_features: Number of features per stock
        num_stocks: Number of stocks
        y_scalers: List of scalers for targets
        tickers: List of stock tickers
        device: PyTorch device (CPU/GPU)
        training_epochs: Maximum number of epochs for training
        plot_output_dir: Directory for saving plots
        mlflow_experiment_name: Name of MLflow experiment
        
    Returns:
        Tuple with trained model, test predictions, evaluation metrics, and MLflow run ID
    """
    batch_size = best_params['batch_size']
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    model_type = best_params.get('model_type', 'lstm')

    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = None
    # Check if X_test and y_test (NumPy arrays) have data
    if (X_test is not None and X_test.shape[0] > 0 and
            y_test is not None and y_test.shape[0] > 0):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        logger.warning(
            "Test data is empty or None. Model will be trained but not "
            "evaluated on a test set during this training run."
        )

    # Initialize model based on model type
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

    mlflow.set_experiment(mlflow_experiment_name)
    mlflow_run_name = (
        f"final_train_{model_type}_{dataset_run_id}_"
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )

    with mlflow.start_run(run_name=mlflow_run_name) as run:
        mlflow_model_run_id = run.info.run_id
        logger.info(
            f"--- MLflow Run Started for Final Training "
            f"(MLflow Run ID: {mlflow_model_run_id}) ---"
        )

        mlflow.log_param("dataset_run_id", dataset_run_id)
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
                # Move data to device if not already done by DataLoader
                batch_sequences = batch_sequences.to(device)
                batch_targets = batch_targets.to(device)

                optimizer.zero_grad()
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = (
                total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            )
            mlflow.log_metric('train_loss', avg_train_loss, step=epoch)

            current_test_loss = float('nan')
            if test_loader:
                _, _, eval_metrics = evaluate_model(
                    model, test_loader, criterion, y_scalers, device
                )
                current_test_loss = eval_metrics['test_loss']
                mlflow.log_metric('test_loss', current_test_loss, step=epoch)
                mlflow.log_metric('avg_mse_test', eval_metrics['avg_mse'], step=epoch)
                mlflow.log_metric('avg_mape_test', eval_metrics['avg_mape'], step=epoch)
                mlflow.log_metric(
                    'avg_direction_accuracy_test',
                    eval_metrics['avg_direction_accuracy'],
                    step=epoch
                )

                if current_test_loss < best_test_loss:
                    best_test_loss = current_test_loss
                    best_model_state = model.state_dict().copy()
                    mlflow.set_tag("best_epoch", epoch + 1)
            elif epoch == training_epochs - 1:  # If no test loader, save last epoch
                best_model_state = model.state_dict().copy()

            logger.info(
                f'Epoch [{epoch+1}/{training_epochs}], '
                f'Train Loss: {avg_train_loss:.4f}, '
                f'Test Loss: {current_test_loss:.4f}'
            )

        mlflow.log_param('actual_epochs_trained', actual_epochs_trained)

        if best_model_state:
            model.load_state_dict(best_model_state)
            logger.info("Loaded best model state_dict from training run.")
        else:
            logger.warning(
                "No best model state_dict saved. Using model from last iteration."
            )

        final_metrics_on_test: Dict[str, Any] = {}
        predictions_on_test: Optional[np.ndarray] = None
        
        if test_loader and y_test is not None:  # y_test is the original NumPy array
            predictions_on_test, _, final_metrics_on_test = evaluate_model(
                model, test_loader, criterion, y_scalers, device
            )
            mlflow.log_metrics({
                'final_test_loss': final_metrics_on_test.get(
                    'test_loss', float('nan')
                ),
                'final_avg_mse_test': final_metrics_on_test.get(
                    'avg_mse', float('nan')
                ),
                'final_avg_mape_test': final_metrics_on_test.get(
                    'avg_mape', float('nan')
                ),
                'final_avg_direction_accuracy_test': final_metrics_on_test.get(
                    'avg_direction_accuracy', float('nan')
                )
            })
            logger.info("Final model metrics on test set logged to MLflow.")
            
            if predictions_on_test is not None and y_test is not None:
                visualize_predictions(
                    predictions_on_test, y_test, y_scalers, tickers,
                    plot_output_dir, num_points=20
                )
                logger.info("Visualizations generated and logged to MLflow artifacts.")
        else:
            logger.info(
                "No test set evaluation performed or y_test was None for visualization."
            )

        logger.info("Logging PyTorch model to MLflow...")
        registered_model_name = mlflow_experiment_name
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )
        logger.info(
            f"Model logged to MLflow with run_id {mlflow_model_run_id} "
            f"and registered as '{registered_model_name}'."
        )
        logger.info("--- MLflow Run Finished ---")

    return model, predictions_on_test, final_metrics_on_test, mlflow_model_run_id


# -------------------------------------------------------------

def run_training(config_path: str, run_id_arg: str) -> Optional[str]:
    """
    Run model training using data and parameters associated with a specific run_id.
    
    Args:
        config_path: Path to the configuration file
        run_id_arg: Run ID for the dataset to use
        
    Returns:
        MLflow run ID of the trained model, or None if training failed
    """
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        db_config = params['database']
        current_dataset_run_id = run_id_arg
        logger.info(f"Model training using dataset from run_id: {current_dataset_run_id}")

        mlflow_config = params['mlflow']
        mlflow_experiment_name = mlflow_config['experiment_name']
        mlflow_tracking_uri = os.environ.get(
            'MLFLOW_TRACKING_URI', 
            mlflow_config.get('tracking_uri', 'http://mlflow-server:5000')
        )
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        training_params_cfg = params['training']
        final_training_epochs = training_params_cfg['epochs']

        plot_output_dir_str = params.get('output_paths', {}).get(
            'training_plots_dir', 'plots/training_plots'
        )
        plot_output_dir = Path(plot_output_dir_str)
        plot_output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        logger.info(
            f"--- Loading Scaled Data from database for dataset_run_id: "
            f"{current_dataset_run_id} ---"
        )
        X_train_scaled = load_scaled_features(db_config, current_dataset_run_id, 'X_train')
        y_train_scaled = load_scaled_features(db_config, current_dataset_run_id, 'y_train')
        X_test_scaled = load_scaled_features(db_config, current_dataset_run_id, 'X_test')
        y_test_scaled = load_scaled_features(db_config, current_dataset_run_id, 'y_test')

        if X_train_scaled is None or y_train_scaled is None:
            logger.error(
                f"Failed to load scaled training data for dataset_run_id: "
                f"{current_dataset_run_id}."
            )
            return None
        logger.info("--- Finished Loading Scaled Data ---")

        logger.info(
            f"--- Loading Scalers from database for dataset_run_id: "
            f"{current_dataset_run_id} ---"
        )
        scalers_dict = load_scalers(db_config, current_dataset_run_id)
        if scalers_dict is None or 'y_scalers' not in scalers_dict:
            logger.error(
                f"Failed to load scalers or 'y_scalers' not found for "
                f"dataset_run_id: {current_dataset_run_id}."
            )
            return None
        y_scalers = scalers_dict['y_scalers']
        tickers = scalers_dict.get('tickers', [])
        
        if not tickers:
            logger.warning(
                "Tickers not found in scalers_dict, attempting to load from "
                "processed_feature_data."
            )
            feature_data_info = load_processed_features_from_db(
                db_config, current_dataset_run_id
            )
            if feature_data_info and 'tickers' in feature_data_info:
                tickers = feature_data_info['tickers']
            else:
                logger.error(
                    f"Failed to load tickers for dataset_run_id: "
                    f"{current_dataset_run_id}."
                )
                return None
        logger.info(f"Using tickers: {tickers}")
        logger.info("--- Finished Loading Scalers ---")

        logger.info(
            f"--- Loading Best Hyperparameters from database for dataset_run_id: "
            f"{current_dataset_run_id} ---"
        )
        best_hyperparams = load_optimization_results(db_config, current_dataset_run_id)
        if best_hyperparams is None:
            logger.error(
                f"Failed to load best hyperparameters for dataset_run_id: "
                f"{current_dataset_run_id}."
            )
            return None
        logger.info(f"Best hyperparameters loaded: {best_hyperparams}")
        logger.info("--- Finished Loading Best Hyperparameters ---")

        if X_train_scaled.ndim < 4:
            logger.error(
                f"X_train_scaled has unexpected dimensions: {X_train_scaled.ndim}. "
                f"Expected 4."
            )
            return None
            
        num_stocks = X_train_scaled.shape[2]
        num_features = X_train_scaled.shape[3]

        logger.info("--- Starting Final Model Training ---")
        final_model_obj, test_predictions_np, final_test_metrics, mlflow_model_run_id = train_final_model(
            dataset_run_id=current_dataset_run_id,
            best_params=best_hyperparams,
            X_train=X_train_scaled, y_train=y_train_scaled,
            X_test=X_test_scaled, y_test=y_test_scaled,
            num_features=num_features, num_stocks=num_stocks,
            y_scalers=y_scalers, tickers=tickers,
            device=device, training_epochs=final_training_epochs,
            plot_output_dir=plot_output_dir,
            mlflow_experiment_name=mlflow_experiment_name
        )
        logger.info(
            f"--- Finished Final Model Training. MLflow Run ID for trained "
            f"model: {mlflow_model_run_id} ---"
        )

        logger.info(
            f"Attempting to promote model version associated with MLflow run: "
            f"{mlflow_model_run_id}"
        )
        try:
            client = MlflowClient()
            registered_model_name = mlflow_experiment_name  # Consistent with registration

            versions = client.search_model_versions(f"run_id='{mlflow_model_run_id}'")
            if not versions:
                logger.error(
                    f"No model version found in registry for MLflow run_id "
                    f"{mlflow_model_run_id}. Cannot promote automatically."
                )
            else:
                latest_version_for_run = sorted(
                    versions, key=lambda v: int(v.version), reverse=True
                )[0]
                new_model_version = latest_version_for_run.version
                logger.info(
                    f"Found new model version: {new_model_version} for registered "
                    f"model '{registered_model_name}' from run_id {mlflow_model_run_id}."
                )

                latest_prod_versions = client.get_latest_versions(
                    name=registered_model_name, stages=["Production"]
                )
                for old_prod_version in latest_prod_versions:
                    if old_prod_version.version != new_model_version:
                        logger.info(
                            f"Archiving existing Production version: "
                            f"{old_prod_version.version} of model '{registered_model_name}'."
                        )
                        client.transition_model_version_stage(
                            name=registered_model_name,
                            version=old_prod_version.version,
                            stage="Archived",
                            archive_existing_versions=False
                        )
                
                production_alias_name = "Production"
                logger.info(
                    f"Setting alias '{production_alias_name}' for model "
                    f"'{registered_model_name}' version {new_model_version}."
                )
                try:
                    client.set_registered_model_alias(
                        name=registered_model_name,
                        alias=production_alias_name,
                        version=new_model_version
                    )
                    logger.info(
                        f"Successfully set alias '{production_alias_name}' to version "
                        f"{new_model_version} of model '{registered_model_name}'."
                    )
                except Exception as e_alias:
                    logger.error(
                        f"Failed to set alias '{production_alias_name}' for model "
                        f"'{registered_model_name}' version {new_model_version}: {e_alias}",
                        exc_info=True
                    )

        except Exception as e_promote:
            logger.error(
                f"Failed to promote model version for MLflow run_id "
                f"{mlflow_model_run_id}: {e_promote}",
                exc_info=True
            )

        if final_test_metrics:
            logger.info("\nFinal Training Summary (on test set):")
            logger.info("========================")
            for metric_name, value in final_test_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"{metric_name}: {value:.4f}")
            logger.info("========================")
        
        if test_predictions_np is not None and test_predictions_np.size > 0:
            logger.info(
                f"--- Saving Test Set Predictions from this Training Run "
                f"(Model MLflow Run ID: {mlflow_model_run_id}) to Database ---"
            )
            try:
                for stock_idx, ticker_name in enumerate(tickers):
                    if test_predictions_np.shape[0] > 0:
                        # Ensure pred_len is handled; assuming pred_len = 1 for this save
                        pred_idx_to_save = 0  # Save the first prediction if pred_len > 1
                        
                        last_pred_on_test_for_ticker_scaled = test_predictions_np[
                            -1, pred_idx_to_save, stock_idx
                        ]
                        last_pred_on_test_for_ticker_inv = y_scalers[stock_idx].inverse_transform(
                            np.array([[last_pred_on_test_for_ticker_scaled]])
                        )[0][0]
                        save_prediction(
                            db_config,
                            ticker_name,
                            float(last_pred_on_test_for_ticker_inv),
                            mlflow_model_run_id
                        )
                        logger.info(
                            f"Saved last test set prediction for {ticker_name}: "
                            f"{float(last_pred_on_test_for_ticker_inv):.4f}"
                        )
                logger.info(
                    f"Successfully saved sample test set predictions for "
                    f"{len(tickers)} tickers."
                )
            except Exception as e_save:
                logger.error(
                    f"Error saving test set predictions from training run: {e_save}",
                    exc_info=True
                )
        else:
            logger.info("No test set predictions to save from this training run.")

        logger.info(
            f"--- Training process for dataset_run_id {current_dataset_run_id} "
            f"complete. Trained model MLflow Run ID: {mlflow_model_run_id} ---"
        )
        return mlflow_model_run_id
        
    except FileNotFoundError as e_fnf:  # More specific exception handling
        logger.error(
            f"Configuration file not found at {config_path} for run_training, "
            f"dataset_run_id {run_id_arg}: {e_fnf}",
            exc_info=True
        )
        return None
    except Exception as e:
        logger.error(
            f"Error in run_training for dataset_run_id {run_id_arg}: {e}",
            exc_info=True
        )
        return None


# -------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Model training script for stock prediction."
    )
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
        help='The run_id of the dataset (scaled features, scalers, best_params from DB) '
             'to use for training.'
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

    logger.info(
        f"Starting training script with resolved config: {config_path_resolved} "
        f"for dataset_run_id: {cli_run_id_arg}"
    )

    try:
        # No need to open config here, run_training will do it.
        trained_model_mlflow_run_id = run_training(
            str(config_path_resolved), run_id_arg=cli_run_id_arg
        )
        
        if trained_model_mlflow_run_id:
            logger.info(
                f"Training completed successfully for dataset_run_id: {cli_run_id_arg}."
            )
            logger.info(
                f"MLflow Run ID of the trained model: {trained_model_mlflow_run_id}"
            )
            print(f"TRAINING_SUCCESS_MLFLOW_RUN_ID:{trained_model_mlflow_run_id}")
        else:
            logger.error(
                f"Training failed for dataset_run_id: {cli_run_id_arg}. Check logs."
            )
            sys.exit(1)

    except Exception as e_main:  # Catching other potential errors during setup or teardown
        logger.error(
            f"Fatal error in training script for dataset_run_id {cli_run_id_arg}: "
            f"{e_main}",
            exc_info=True
        )
        sys.exit(1)