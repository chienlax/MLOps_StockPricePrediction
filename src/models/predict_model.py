#!/usr/bin/env python
import argparse
import json
import yaml
import numpy as np
import joblib
import torch
import mlflow
import mlflow.pytorch
import sys
import os
import logging
import subprocess
from datetime import date
from mlflow.tracking import MlflowClient
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


from utils.db_utils import (
    load_scalers,
    save_prediction,
    load_processed_features_from_db # Fallback for tickers if not in scalers_dict
)
# Assuming model_definitions are not directly needed here as we load a full mlflow.pytorch model
# If you were loading only state_dict, you'd need them.

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Generate latest stock price predictions and save daily historical JSON files"
#     )
#     parser.add_argument(
#         "--config", required=True,
#         help="Path to params.yaml"
#     )
#     return parser.parse_args()


# def load_model_from_registry(experiment_name: str):
#     """
#     Load the PyTorch model from MLflow Registry (Production),
#     fallback to most recent run if unavailable.
#     """
#     try:
#         return mlflow.pytorch.load_model(f"models:/{experiment_name}/Production")
#     except Exception:
#         client = MlflowClient()
#         exp = client.get_experiment_by_name(experiment_name)
#         if exp is None:
#             raise ValueError(f"Experiment not found: {experiment_name}")
#         runs = client.search_runs([exp.experiment_id], order_by=["attribute.start_time DESC"], max_results=1)
#         if not runs:
#             raise ValueError(f"No runs found in experiment {experiment_name}")
#         run_id = runs[0].info.run_id
#         return mlflow.pytorch.load_model(f"runs:/{run_id}/model")


# def main():
#     args = parse_args()

#     # Load config
#     cfg_path = Path(args.config)
#     with open(cfg_path, 'r') as f:
#         cfg = yaml.safe_load(f)

#     # Set paths from config
#     split_data_path = Path(cfg['output_paths']['split_data_path'])
#     scalers_path = Path(cfg['output_paths']['scalers_path'])
#     processed_data_path = Path(cfg['output_paths']['processed_data_path'])
#     predictions_dir = Path(cfg['output_paths'].get('predictions_dir', '/opt/airflow/data/predictions'))
#     experiment_name = cfg['mlflow']['experiment_name']

#     # Load latest test sequence
#     if not split_data_path.exists():
#         raise FileNotFoundError(f"Split data not found: {split_data_path}")
#     data = np.load(split_data_path, allow_pickle=True)
#     if 'X_test_scaled' in data:
#         X_test = data['X_test_scaled']
#     elif 'x_test' in data:
#         X_test = data['x_test']
#     else:
#         raise KeyError("X_test_scaled or x_test not found in split data file")
#     X_latest = X_test[-1:]
#     X_tensor = torch.tensor(X_latest, dtype=torch.float32)

#     # Load scalers
#     if not scalers_path.exists():
#         raise FileNotFoundError(f"Scalers not found: {scalers_path}")
#     scaler_dict = joblib.load(scalers_path)
#     y_scalers = scaler_dict.get('y_scalers')
#     if y_scalers is None:
#         raise KeyError("y_scalers key not found in scalers file")

#     # Load model
#     mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
#     model = load_model_from_registry(experiment_name)
#     model.eval()

#     # Predict
#     with torch.no_grad():
#         preds = model(X_tensor)
#         preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else np.array(preds)

#     # Load tickers list
#     proc_data = np.load(processed_data_path, allow_pickle=True)
#     tickers = proc_data['tickers'].tolist()

#     # Inverse transform predictions
#     results = {}
#     for idx, ticker in enumerate(tickers):
#         try:
#             # Handle 3D output (batch, horizon, features)
#             val_scaled = float(preds_np[0, 0, idx])
#         except Exception:
#             # 2D output (batch, features)
#             val_scaled = float(preds_np[0, idx])
#         scaler = y_scalers[idx]
#         try:
#             val = scaler.inverse_transform([[val_scaled]])[0][0]
#         except Exception:
#             val = val_scaled
#         results[ticker] = val

#     # Prepare output directories
#     predictions_dir.mkdir(parents=True, exist_ok=True)
#     hist_dir = predictions_dir / 'historical'
#     hist_dir.mkdir(parents=True, exist_ok=True)

#     # Prepare payload
#     today = date.today().isoformat()
#     payload = {"date": today, "predictions": results}

#     # Write latest
#     latest_file = predictions_dir / 'latest_predictions.json'
#     with open(latest_file, 'w') as f:
#         json.dump(payload, f, indent=2)
#     print(f"Saved latest predictions to {latest_file}")

#     # Write daily historical file
#     hist_file = hist_dir / f"{today}.json"
#     with open(hist_file, 'w') as f:
#         json.dump(payload, f, indent=2)
#     print(f"Saved historical predictions to {hist_file}")

# if __name__ == '__main__':
#     main()

# def run_daily_prediction(config_path: str, input_sequence_path: str, production_model_uri: str) -> bool:
#     """
#     Loads the production model, makes predictions on the prepared input sequence, and saves results.

#     Args:
#         config_path (str): Path to params.yaml.
#         input_sequence_path (str): Path to the .npy file containing the prepared input sequence.
#         production_model_uri (str): MLflow Model Registry URI for the production model.

#     Returns:
#         bool: True if prediction and saving were successful, False otherwise.
#     """
#     try:
#         with open(config_path, 'r') as f:
#             cfg = yaml.safe_load(f)

#         db_config = cfg['database']
#         mlflow_cfg = cfg['mlflow']
        
#         # Output paths for JSON predictions (should be configurable, ensure they are writable)
#         # These paths are relative to where the script is run or need to be absolute for containers
#         # For Airflow, these might point to a shared volume /opt/airflow/data/predictions
#         # Defaulting to paths that might work if script is run from project root.
#         # In Airflow, you'd likely use absolute paths like /opt/airflow/data/predictions
#         predictions_base_dir_str = cfg.get('output_paths', {}).get('predictions_dir', 'data/predictions')
#         predictions_base_dir = Path(predictions_base_dir_str) # Convert to Path object
        
#         # Ensure MLflow tracking URI is set
#         mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
#         if not mlflow_tracking_uri:
#             logger.error("MLFLOW_TRACKING_URI is not set. Cannot connect to MLflow.")
#             return False
#         mlflow.set_tracking_uri(mlflow_tracking_uri)
#         logger.info(f"Using MLflow Tracking URI: {mlflow_tracking_uri}")

#         # 1. Load the production model from MLflow Model Registry
#         logger.info(f"Loading production model from URI: {production_model_uri}")
#         try:
#             model = mlflow.pytorch.load_model(model_uri=production_model_uri)
#             model.eval() # Set to evaluation mode
#         except Exception as e_load_model:
#             logger.error(f"Failed to load model from {production_model_uri}: {e_load_model}", exc_info=True)
#             return False
        
#         # 2. Get the training run_id associated with this production model
#         # This run_id is for the DATASET the model was trained on.
#         client = MlflowClient()
#         try:
#             # The model_uri for get_model_version_by_alias is like "models:/MyModelName" and alias "Production"
#             # Or, if production_model_uri is already specific like "models:/MyModelName/Production"
#             # we can parse it or use get_model_version
#             model_name_from_uri = production_model_uri.split('/')[1] # "models:/<model_name>/Production"
            
#             # Find the version associated with the "Production" alias/stage
#             # This is a bit more robust than just getting run_id from model_info if URI is generic
#             latest_prod_versions = client.get_latest_versions(name=model_name_from_uri, stages=["Production"])
#             if not latest_prod_versions:
#                 logger.error(f"No version of model '{model_name_from_uri}' found in 'Production' stage.")
#                 # Fallback: try to get run_id directly if URI is specific to a version
#                 try:
#                     model_version_details = client.get_model_version(name=model_name_from_uri, version=production_model_uri.split('/')[-1])
#                     prod_model_mlflow_run_id = model_version_details.run_id
#                 except Exception:
#                     logger.error(f"Could not determine MLflow run_id for production model {production_model_uri}")
#                     return False
#             else:
#                 # Assuming only one version is in "Production" for a given model name.
#                 # If multiple, this might need refinement (e.g., take highest version number).
#                 prod_model_mlflow_run_id = latest_prod_versions[0].run_id
            
#             logger.info(f"Production model was trained with MLflow Run ID: {prod_model_mlflow_run_id}")
            
#             # Now, we need the DATASET run_id. This should have been logged as a param to the MLflow run.
#             mlflow_run_details = client.get_run(prod_model_mlflow_run_id)
#             prod_model_dataset_run_id = mlflow_run_details.data.params.get("dataset_run_id")
#             if not prod_model_dataset_run_id:
#                 logger.error(f"Parameter 'dataset_run_id' not found in MLflow Run {prod_model_mlflow_run_id}. Cannot load correct scalers.")
#                 return False
#             logger.info(f"Production model's dataset_run_id (for scalers/features): {prod_model_dataset_run_id}")

#         except Exception as e_mlflow_meta:
#             logger.error(f"Error fetching metadata for production model {production_model_uri}: {e_mlflow_meta}", exc_info=True)
#             return False

#         # 3. Load y_scalers and tickers using the prod_model_dataset_run_id
#         logger.info(f"Loading y_scalers and tickers using dataset_run_id: {prod_model_dataset_run_id}")
#         scalers_dict = load_scalers(db_config, prod_model_dataset_run_id)
#         if not scalers_dict or 'y_scalers' not in scalers_dict:
#             logger.error(f"Failed to load y_scalers for dataset_run_id: {prod_model_dataset_run_id}")
#             return False
#         y_scalers = scalers_dict['y_scalers']
        
#         tickers = scalers_dict.get('tickers')
#         if not tickers: # Fallback
#             processed_features_meta = load_processed_features_from_db(db_config, prod_model_dataset_run_id)
#             if not processed_features_meta or 'tickers' not in processed_features_meta:
#                 logger.error(f"Failed to load tickers for dataset_run_id: {prod_model_dataset_run_id}")
#                 return False
#             tickers = processed_features_meta['tickers']
        
#         if len(y_scalers) != len(tickers):
#             logger.error(f"Mismatch between number of y_scalers ({len(y_scalers)}) and tickers ({len(tickers)}).")
#             return False
#         logger.info(f"Loaded {len(y_scalers)} y_scalers for tickers: {tickers}")

#         # 4. Load the input sequence
#         logger.info(f"Loading input sequence from: {input_sequence_path}")
#         if not Path(input_sequence_path).exists():
#             logger.error(f"Input sequence file not found: {input_sequence_path}")
#             return False
#         input_sequence_np = np.load(input_sequence_path) # Shape: (1, seq_len, num_stocks, num_features)
        
#         # Verify num_stocks in input sequence matches expected
#         if input_sequence_np.shape[2] != len(tickers):
#             logger.error(f"Input sequence has {input_sequence_np.shape[2]} stocks, but model/scalers expect {len(tickers)}.")
#             return False

#         input_tensor = torch.tensor(input_sequence_np, dtype=torch.float32)
#         # Determine device (though model loaded from MLflow should handle its own device placement if saved correctly)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model.to(device)
#         input_tensor = input_tensor.to(device)


#         # 5. Make predictions
#         logger.info("Making predictions...")
#         with torch.no_grad():
#             predictions_scaled_tensor = model(input_tensor) # Expected shape: (1, pred_len, num_stocks)
#         predictions_scaled_np = predictions_scaled_tensor.cpu().numpy()
#         # Assuming pred_len is 1, so shape is (1, 1, num_stocks)
        
#         # 6. Inverse transform predictions
#         logger.info("Inverse transforming predictions...")
#         predicted_prices_final = {}
#         # predictions_scaled_np[0, 0, stock_idx] gives the scaled prediction for that stock
#         for stock_idx, ticker_name in enumerate(tickers):
#             scaled_pred_value = predictions_scaled_np[0, 0, stock_idx]
#             actual_pred_price = y_scalers[stock_idx].inverse_transform(np.array([[scaled_pred_value]]))[0][0]
#             predicted_prices_final[ticker_name] = float(actual_pred_price)
        
#         logger.info(f"Final predicted prices: {predicted_prices_final}")

#         # 7. Save predictions to JSON files (for API/dashboard)
#         predictions_base_dir.mkdir(parents=True, exist_ok=True)
#         historical_dir = predictions_base_dir / 'historical'
#         historical_dir.mkdir(parents=True, exist_ok=True)

#         today_iso = date.today().isoformat()
#         payload = {"date": today_iso, "predictions": predicted_prices_final, "model_mlflow_run_id": prod_model_mlflow_run_id}

#         latest_file_path = predictions_base_dir / 'latest_predictions.json'
#         with open(latest_file_path, 'w') as f:
#             json.dump(payload, f, indent=4)
#         logger.info(f"Saved latest predictions to {latest_file_path}")

#         historical_file_path = historical_dir / f"{today_iso}.json"
#         with open(historical_file_path, 'w') as f:
#             json.dump(payload, f, indent=4)
#         logger.info(f"Saved historical predictions to {historical_file_path}")

#         # 8. Save predictions to PostgreSQL database
#         logger.info("Saving predictions to PostgreSQL database...")
#         for ticker_name, predicted_price_val in predicted_prices_final.items():
#             save_prediction(db_config, ticker_name, predicted_price_val, prod_model_mlflow_run_id)
#         logger.info("Successfully saved predictions to database.")
        
#         return True

#     except Exception as e:
#         logger.error(f"Error in run_daily_prediction: {e}", exc_info=True)
#         return False
    

# def run_daily_prediction(config_path: str, input_sequence_path: str, production_model_uri: str) -> bool:
#     """
#     Loads the production model, makes predictions on the prepared input sequence, and saves results.
#     Args:
#         config_path (str): Path to params.yaml.
#         input_sequence_path (str): Path to the .npy file containing the prepared input sequence.
#         production_model_uri (str): MLflow Model Registry URI (e.g., "models:/MyModel@Production" or "models:/MyModel/VersionNum").
#     Returns:
#         bool: True if prediction and saving were successful, False otherwise.
#     """
#     try:
#         with open(config_path, 'r') as f:
#             cfg = yaml.safe_load(f)

#         db_config = cfg['database']
#         mlflow_cfg = cfg['mlflow']
#         predictions_base_dir_str = cfg.get('output_paths', {}).get('predictions_dir', 'data/predictions')
#         predictions_base_dir = Path(predictions_base_dir_str)
        
#         mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
#         if not mlflow_tracking_uri:
#             logger.error("MLFLOW_TRACKING_URI is not set.")
#             return False
#         mlflow.set_tracking_uri(mlflow_tracking_uri)
#         logger.info(f"Using MLflow Tracking URI: {mlflow_tracking_uri}")

#         # 1. Load the production model from MLflow Model Registry using the provided URI
#         logger.info(f"Loading production model from URI: {production_model_uri}")
#         try:
#             model_pytorch = mlflow.pytorch.load_model(model_uri=production_model_uri)
#             model_pytorch.eval()
#         except Exception as e_load_model:
#             logger.error(f"Failed to load model from {production_model_uri}: {e_load_model}", exc_info=True)
#             return False
        
#         # 2. Get the training run_id and dataset_run_id associated with this production model version
#         client = MlflowClient()
#         prod_model_dataset_run_id = None
#         prod_model_origin_mlflow_run_id = None # The MLflow run ID of the training job for the loaded prod model

#         try:
#             # Parse the model URI to get name and alias/version
#             # URI formats: models:/<name>/<version>, models:/<name>@<alias>
#             parsed_uri = mlflow.utils.uri.parse_model_uri(production_model_uri)
#             registered_model_name = parsed_uri.name # This is the BASE model name

#             if parsed_uri.version:
#                 model_version_details = client.get_model_version(name=registered_model_name, version=parsed_uri.version)
#             elif parsed_uri.alias: # MLflow > 2.9 uses alias rather than stage in URI
#                 # In MLflow 2.x, 'stage' in model URI might be treated as an alias.
#                 # The callable_get_production_model_info already uses get_model_version_by_alias if URI contains '@'
#                 # So, if production_model_uri is models:/MyModel@Production, parsed_uri.alias will be "Production"
#                 model_version_details = client.get_model_version_by_alias(name=registered_model_name, alias=parsed_uri.alias)
#             else: # Should not happen if URI is from our get_production_model_info task
#                 logger.error(f"Could not determine version or alias from production_model_uri: {production_model_uri}")
#                 return False

#             prod_model_origin_mlflow_run_id = model_version_details.run_id
#             logger.info(f"Production model (Version: {model_version_details.version}, Name: {registered_model_name}) was trained with MLflow Run ID: {prod_model_origin_mlflow_run_id}")
            
#             mlflow_run_details = client.get_run(prod_model_origin_mlflow_run_id)
#             prod_model_dataset_run_id = mlflow_run_details.data.params.get("dataset_run_id")
            
#             if not prod_model_dataset_run_id:
#                 logger.error(f"Parameter 'dataset_run_id' not found in MLflow Run {prod_model_origin_mlflow_run_id}. Cannot load correct scalers.")
#                 return False
#             logger.info(f"Production model's dataset_run_id (for scalers/features): {prod_model_dataset_run_id}")

#         except Exception as e_mlflow_meta:
#             logger.error(f"Error fetching metadata for production model {production_model_uri}: {e_mlflow_meta}", exc_info=True)
#             return False

#         # 3. Load y_scalers and tickers using the prod_model_dataset_run_id
#         logger.info(f"Loading y_scalers and tickers using dataset_run_id: {prod_model_dataset_run_id}")
#         scalers_dict = load_scalers(db_config, prod_model_dataset_run_id)
#         if not scalers_dict or 'y_scalers' not in scalers_dict:
#             logger.error(f"Failed to load y_scalers for dataset_run_id: {prod_model_dataset_run_id}")
#             return False
#         y_scalers = scalers_dict['y_scalers']
        
#         tickers = scalers_dict.get('tickers')
#         if not tickers: 
#             processed_features_meta = load_processed_features_from_db(db_config, prod_model_dataset_run_id)
#             if not processed_features_meta or 'tickers' not in processed_features_meta:
#                 logger.error(f"Failed to load tickers for dataset_run_id: {prod_model_dataset_run_id}")
#                 return False
#             tickers = processed_features_meta['tickers']
        
#         if len(y_scalers) != len(tickers):
#             logger.error(f"Mismatch between number of y_scalers ({len(y_scalers)}) and tickers ({len(tickers)}).")
#             return False
#         logger.info(f"Loaded {len(y_scalers)} y_scalers for tickers: {tickers}")


#         # 4. Load the input sequence
#         logger.info(f"Loading input sequence from: {input_sequence_path}")
#         if not Path(input_sequence_path).exists():
#             logger.error(f"Input sequence file not found: {input_sequence_path}")
#             return False
#         input_sequence_np = np.load(input_sequence_path) 
        
#         if input_sequence_np.shape[2] != len(tickers):
#             logger.error(f"Input sequence has {input_sequence_np.shape[2]} stocks, but model/scalers expect {len(tickers)}.")
#             return False
#         input_tensor = torch.tensor(input_sequence_np, dtype=torch.float32)


#         # 5. Make predictions
#         logger.info("Making predictions...")
#         with torch.no_grad():
#             predictions_scaled_tensor = model_pytorch(input_tensor)
#         predictions_scaled_np = predictions_scaled_tensor.cpu().numpy()
        
#         # 6. Inverse transform predictions
#         logger.info("Inverse transforming predictions...")
#         predicted_prices_final = {}
#         for stock_idx, ticker_name in enumerate(tickers):
#             scaled_pred_value = predictions_scaled_np[0, 0, stock_idx] # Assuming pred_len=1
#             actual_pred_price = y_scalers[stock_idx].inverse_transform(np.array([[scaled_pred_value]]))[0][0]
#             predicted_prices_final[ticker_name] = float(actual_pred_price)
#         logger.info(f"Final predicted prices: {predicted_prices_final}")

#         # 7. Save predictions to JSON files
#         predictions_base_dir.mkdir(parents=True, exist_ok=True)
#         historical_dir = predictions_base_dir / 'historical'
#         historical_dir.mkdir(parents=True, exist_ok=True)
#         today_iso = date.today().isoformat()
#         # Use prod_model_origin_mlflow_run_id (the run_id of the model that made the prediction)
#         payload = {"date": today_iso, "predictions": predicted_prices_final, "model_mlflow_run_id": prod_model_origin_mlflow_run_id}
#         latest_file_path = predictions_base_dir / 'latest_predictions.json'
#         with open(latest_file_path, 'w') as f:
#             json.dump(payload, f, indent=4)
#         logger.info(f"Saved latest predictions to {latest_file_path}")
#         historical_file_path = historical_dir / f"{today_iso}.json"
#         with open(historical_file_path, 'w') as f:
#             json.dump(payload, f, indent=4)
#         logger.info(f"Saved historical predictions to {historical_file_path}")

#         # 8. Save predictions to PostgreSQL database
#         logger.info("Saving predictions to PostgreSQL database...")
#         for ticker_name, predicted_price_val in predicted_prices_final.items():
#             # Use prod_model_origin_mlflow_run_id here
#             save_prediction(db_config, ticker_name, predicted_price_val, prod_model_origin_mlflow_run_id)
#         logger.info("Successfully saved predictions to database.")
        
#         return True

#     except Exception as e:
#         logger.error(f"Error in run_daily_prediction: {e}", exc_info=True)
#         return False


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Make daily stock predictions using the production model.")
#     parser.add_argument(
#         '--config', type=str, default='config/params.yaml',
#         help='Path to the configuration file (e.g., config/params.yaml)'
#     )
#     parser.add_argument(
#         '--input_sequence_path', type=str, required=True,
#         help='Path to the .npy file containing the prepared input sequence for prediction.'
#     )
#     parser.add_argument(
#         '--production_model_uri', type=str, required=True,
#         help='MLflow Model Registry URI for the production model (e.g., "models:/MyModel/Production").'
#     )
#     args = parser.parse_args()

#     config_path_resolved = Path(args.config).resolve()
#     if not config_path_resolved.exists():
#         logger.error(f"Configuration file not found: {config_path_resolved}")
#         sys.exit(1)
    
#     input_sequence_path_resolved = Path(args.input_sequence_path).resolve()
#     if not input_sequence_path_resolved.exists():
#         logger.error(f"Input sequence file not found: {input_sequence_path_resolved}")
#         sys.exit(1)

#     logger.info(f"Starting daily prediction with config: {config_path_resolved}, input: {input_sequence_path_resolved}, model: {args.production_model_uri}")

#     success = run_daily_prediction(str(config_path_resolved), str(input_sequence_path_resolved), args.production_model_uri)

#     if success:
#         logger.info("Daily prediction process completed successfully.")
#     else:
#         logger.error("Daily prediction process failed.")
#         sys.exit(1)

def run_daily_prediction(
    config_path: str, 
    input_sequence_path: str, 
    production_model_uri_for_loading: str, # URI with alias or version for loading
    production_model_base_name: str,       # Base registered model name
    production_model_version_number: str   # Specific version number of the prod model
) -> bool:
    """
    Loads the production model, makes predictions on the prepared input sequence, and saves results.
    """
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        db_config = cfg['database']
        mlflow_cfg = cfg['mlflow']
        predictions_base_dir_str = cfg.get('output_paths', {}).get('predictions_dir', 'data/predictions')
        predictions_base_dir = Path(predictions_base_dir_str)
        
        mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri'))
        if not mlflow_tracking_uri:
            logger.error("MLFLOW_TRACKING_URI is not set.")
            return False
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"Using MLflow Tracking URI: {mlflow_tracking_uri}")

        # 1. Load the production model from MLflow Model Registry using the provided URI
        logger.info(f"Loading production model from URI: {production_model_uri_for_loading}")
        try:
            model_pytorch = mlflow.pytorch.load_model(model_uri=production_model_uri_for_loading)
            model_pytorch.eval()
        except Exception as e_load_model:
            logger.error(f"Failed to load model from {production_model_uri_for_loading}: {e_load_model}", exc_info=True)
            return False
        
        # 2. Get the training run_id and dataset_run_id associated with this production model version
        client = MlflowClient()
        prod_model_dataset_run_id = None
        prod_model_origin_mlflow_run_id = None 

        try:
            # We now have the base name and version directly
            model_version_details = client.get_model_version(
                name=production_model_base_name, 
                version=production_model_version_number
            )
            
            prod_model_origin_mlflow_run_id = model_version_details.run_id
            logger.info(f"Production model (Version: {production_model_version_number}, Name: {production_model_base_name}) was trained with MLflow Run ID: {prod_model_origin_mlflow_run_id}")
            
            mlflow_run_details = client.get_run(prod_model_origin_mlflow_run_id)
            prod_model_dataset_run_id = mlflow_run_details.data.params.get("dataset_run_id")
            
            if not prod_model_dataset_run_id:
                logger.error(f"Parameter 'dataset_run_id' not found in MLflow Run {prod_model_origin_mlflow_run_id}. Cannot load correct scalers.")
                return False
            logger.info(f"Production model's dataset_run_id (for scalers/features): {prod_model_dataset_run_id}")

        except Exception as e_mlflow_meta:
            logger.error(f"Error fetching metadata for production model {production_model_base_name} version {production_model_version_number}: {e_mlflow_meta}", exc_info=True)
            return False

        # Ensure these parts are correct:
        # 3. Load y_scalers and tickers using the prod_model_dataset_run_id
        logger.info(f"Loading y_scalers and tickers using dataset_run_id: {prod_model_dataset_run_id}")
        scalers_dict = load_scalers(db_config, prod_model_dataset_run_id)
        if not scalers_dict or 'y_scalers' not in scalers_dict:
            logger.error(f"Failed to load y_scalers for dataset_run_id: {prod_model_dataset_run_id}")
            return False
        y_scalers = scalers_dict['y_scalers']
        tickers = scalers_dict.get('tickers')
        if not tickers: 
            processed_features_meta = load_processed_features_from_db(db_config, prod_model_dataset_run_id)
            if not processed_features_meta or 'tickers' not in processed_features_meta:
                logger.error(f"Failed to load tickers for dataset_run_id: {prod_model_dataset_run_id}")
                return False
            tickers = processed_features_meta['tickers']
        if len(y_scalers) != len(tickers):
            logger.error(f"Mismatch between number of y_scalers ({len(y_scalers)}) and tickers ({len(tickers)}).")
            return False
        logger.info(f"Loaded {len(y_scalers)} y_scalers for tickers: {tickers}")

        # 4. Load the input sequence
        logger.info(f"Loading input sequence from: {input_sequence_path}")
        if not Path(input_sequence_path).exists():
            logger.error(f"Input sequence file not found: {input_sequence_path}")
            return False
        input_sequence_np = np.load(input_sequence_path) 
        if input_sequence_np.shape[2] != len(tickers): # Check num_stocks
            logger.error(f"Input sequence has {input_sequence_np.shape[2]} stocks, model/scalers expect {len(tickers)}.")
            return False
        input_tensor = torch.tensor(input_sequence_np, dtype=torch.float32)

        # 5. Make predictions
        logger.info("Making predictions...")
        with torch.no_grad():
            predictions_scaled_tensor = model_pytorch(input_tensor)
        predictions_scaled_np = predictions_scaled_tensor.cpu().numpy()
        
        # 6. Inverse transform predictions
        logger.info("Inverse transforming predictions...")
        predicted_prices_final = {}
        for stock_idx, ticker_name in enumerate(tickers):
            scaled_pred_value = predictions_scaled_np[0, 0, stock_idx] 
            actual_pred_price = y_scalers[stock_idx].inverse_transform(np.array([[scaled_pred_value]]))[0][0]
            predicted_prices_final[ticker_name] = float(actual_pred_price)
        logger.info(f"Final predicted prices: {predicted_prices_final}")

        today_iso = date.today().isoformat() # This is the date the prediction is FOR

        # 7. Save predictions to JSON files (KEEPING THIS FOR API for now)
        predictions_base_dir.mkdir(parents=True, exist_ok=True)
        historical_dir = predictions_base_dir / 'historical'
        historical_dir.mkdir(parents=True, exist_ok=True)
        payload_json = {"date": today_iso, "predictions": predicted_prices_final, "model_mlflow_run_id": prod_model_origin_mlflow_run_id}
        latest_file_path = predictions_base_dir / 'latest_predictions.json'
        with open(latest_file_path, 'w+') as f:
            json.dump(payload_json, f, indent=4)
        logger.info(f"Saved latest predictions to JSON: {latest_file_path}")
        historical_file_path = historical_dir / f"{today_iso}.json" # The historical file is for today_iso
        with open(historical_file_path, 'w') as f:
            json.dump(payload_json, f, indent=4)
        logger.info(f"Saved historical predictions to JSON: {historical_file_path}")

        # 8. Save predictions to PostgreSQL database
        logger.info("Saving predictions to PostgreSQL database (with target_prediction_date)...")
        for ticker_name, predicted_price_val in predicted_prices_final.items():
            save_prediction(
                db_config, 
                ticker_name, 
                predicted_price_val, 
                prod_model_origin_mlflow_run_id, # model_mlflow_run_id
                today_iso  # target_prediction_date_str: prediction is for "today"
            )
        logger.info("Successfully saved predictions to database.")

        
        return True

    except Exception as e:
        logger.error(f"Error in run_daily_prediction: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make daily stock predictions using the production model.")
    parser.add_argument('--config', type=str, default='config/params.yaml', help='Path to params.yaml')
    parser.add_argument('--input_sequence_path', type=str, required=True, help='Path to the .npy input sequence.')
    
    # These will now be pulled from XComs by the DAG task
    parser.add_argument('--production_model_uri_for_loading', type=str, required=True, help='MLflow Model URI for loading (e.g., models:/MyModel@Alias or models:/MyModel/Version).')
    parser.add_argument('--production_model_base_name', type=str, required=True, help='Base registered model name.')
    parser.add_argument('--production_model_version_number', type=str, required=True, help='Specific version number of the production model.')
    
    args = parser.parse_args()

    # ... (config path resolution logic as before) ...
    config_path_resolved = Path(args.config).resolve() # Simplified for brevity
    input_sequence_path_resolved = Path(args.input_sequence_path).resolve()

    logger.info(f"Starting daily prediction: config={config_path_resolved}, input={input_sequence_path_resolved}, model_uri_load={args.production_model_uri_for_loading}, model_name={args.production_model_base_name}, model_version={args.production_model_version_number}")

    success = run_daily_prediction(
        str(config_path_resolved), 
        str(input_sequence_path_resolved), 
        args.production_model_uri_for_loading,
        args.production_model_base_name,
        args.production_model_version_number
    )

    if success:
        logger.info("Daily prediction process completed successfully.")
    else:
        logger.error("Daily prediction process failed.")
        sys.exit(1)
