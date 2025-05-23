#!/usr/bin/env python
"""Module for making predictions using trained stock price prediction models."""

# Standard library imports
import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

# Third-party imports
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from mlflow.tracking import MlflowClient

# Set up import path for local modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Local imports
from utils.db_utils import (
    load_processed_features_from_db,
    load_scalers,
    save_prediction,
)

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def run_daily_prediction(
    config_path: str,
    input_sequence_path: str,
    production_model_uri_for_loading: str,
    production_model_base_name: str,
    production_model_version_number: str
) -> bool:
    """
    Load the production model, make predictions on the prepared input sequence, and save results.

    Args:
        config_path: Path to configuration file
        input_sequence_path: Path to the prepared input sequence numpy file
        production_model_uri_for_loading: MLflow URI to load the model from
        production_model_base_name: Base name of the registered model
        production_model_version_number: Version number of the production model

    Returns:
        bool: True if prediction process succeeds, False otherwise
    """
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        db_config = cfg['database']
        mlflow_cfg = cfg['mlflow']
        predictions_base_dir_str = cfg.get('output_paths', {}).get(
            'predictions_dir', 'data/predictions'
        )
        predictions_base_dir = Path(predictions_base_dir_str)
        
        mlflow_tracking_uri = os.environ.get(
            'MLFLOW_TRACKING_URI', mlflow_cfg.get('tracking_uri')
        )
        if not mlflow_tracking_uri:
            logger.error("MLFLOW_TRACKING_URI is not set.")
            return False
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"Using MLflow Tracking URI: {mlflow_tracking_uri}")

        # 1. Load the production model from MLflow Model Registry using the provided URI
        logger.info(f"Loading production model from URI: {production_model_uri_for_loading}")
        try:
            model_pytorch = mlflow.pytorch.load_model(
                model_uri=production_model_uri_for_loading
            )
            model_pytorch.eval()
        except Exception as e_load_model:
            logger.error(
                f"Failed to load model from {production_model_uri_for_loading}: "
                f"{e_load_model}",
                exc_info=True
            )
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
            logger.info(
                f"Production model (Version: {production_model_version_number}, "
                f"Name: {production_model_base_name}) was trained with MLflow Run ID: "
                f"{prod_model_origin_mlflow_run_id}"
            )
            
            mlflow_run_details = client.get_run(prod_model_origin_mlflow_run_id)
            prod_model_dataset_run_id = mlflow_run_details.data.params.get("dataset_run_id")
            
            if not prod_model_dataset_run_id:
                logger.error(
                    f"Parameter 'dataset_run_id' not found in MLflow Run "
                    f"{prod_model_origin_mlflow_run_id}. Cannot load correct scalers."
                )
                return False
            logger.info(
                f"Production model's dataset_run_id (for scalers/features): "
                f"{prod_model_dataset_run_id}"
            )

        except Exception as e_mlflow_meta:
            logger.error(
                f"Error fetching metadata for production model {production_model_base_name} "
                f"version {production_model_version_number}: {e_mlflow_meta}",
                exc_info=True
            )
            return False

        # 3. Load y_scalers and tickers using the prod_model_dataset_run_id
        logger.info(
            f"Loading y_scalers and tickers using dataset_run_id: "
            f"{prod_model_dataset_run_id}"
        )
        scalers_dict = load_scalers(db_config, prod_model_dataset_run_id)
        if not scalers_dict or 'y_scalers' not in scalers_dict:
            logger.error(
                f"Failed to load y_scalers for dataset_run_id: {prod_model_dataset_run_id}"
            )
            return False
        y_scalers = scalers_dict['y_scalers']
        tickers = scalers_dict.get('tickers')
        if not tickers:
            processed_features_meta = load_processed_features_from_db(
                db_config, prod_model_dataset_run_id
            )
            if not processed_features_meta or 'tickers' not in processed_features_meta:
                logger.error(
                    f"Failed to load tickers for dataset_run_id: {prod_model_dataset_run_id}"
                )
                return False
            tickers = processed_features_meta['tickers']
        if len(y_scalers) != len(tickers):
            logger.error(
                f"Mismatch between number of y_scalers ({len(y_scalers)}) and "
                f"tickers ({len(tickers)})."
            )
            return False
        logger.info(f"Loaded {len(y_scalers)} y_scalers for tickers: {tickers}")

        # 4. Load the input sequence
        logger.info(f"Loading input sequence from: {input_sequence_path}")
        if not Path(input_sequence_path).exists():
            logger.error(f"Input sequence file not found: {input_sequence_path}")
            return False
        input_sequence_np = np.load(input_sequence_path)
        if input_sequence_np.shape[2] != len(tickers):  # Check num_stocks
            logger.error(
                f"Input sequence has {input_sequence_np.shape[2]} stocks, "
                f"model/scalers expect {len(tickers)}."
            )
            return False
        input_tensor = torch.tensor(input_sequence_np, dtype=torch.float32)

        # 5. Make predictions
        logger.info("Making predictions...")
        logger.info(f"Shape of input_tensor to model: {input_tensor.shape}")
        # Check if input_tensor has NaNs
        if torch.isnan(input_tensor).any():
            logger.warning(
                f"!!! Input tensor CONTAINS NaNs before model inference: {input_tensor}"
            )
        else:
            logger.info(
                f"Input tensor does NOT contain NaNs. Min: {input_tensor.min()}, "
                f"Max: {input_tensor.max()}"
            )

        with torch.no_grad():
            predictions_scaled_tensor = model_pytorch(input_tensor)

        # Check model's direct output tensor
        if torch.isnan(predictions_scaled_tensor).any():
            logger.warning(
                f"!!! Scaled prediction tensor from model CONTAINS NaNs: "
                f"{predictions_scaled_tensor}"
            )
        else:
            logger.info(
                f"Scaled prediction tensor from model does NOT contain NaNs. "
                f"Min: {predictions_scaled_tensor.min()}, "
                f"Max: {predictions_scaled_tensor.max()}"
            )
        logger.info(f"Shape of predictions_scaled_tensor: {predictions_scaled_tensor.shape}")

        predictions_scaled_np = predictions_scaled_tensor.cpu().numpy()
        
        if np.isnan(predictions_scaled_np).any():
            logger.warning(
                f"!!! Scaled prediction numpy array CONTAINS NaNs: {predictions_scaled_np}"
            )

        # 6. Inverse transform predictions
        logger.info("Inverse transforming predictions...")
        predicted_prices_final = {}
        for stock_idx, ticker_name in enumerate(tickers):
            scaled_pred_value = predictions_scaled_np[0, 0, stock_idx]
            
            logger.info(
                f"Ticker: {ticker_name}, Scaled Predicted Value before inverse_transform: "
                f"{scaled_pred_value}"
            )
            if np.isnan(scaled_pred_value):
                logger.warning(
                    f"!!! Scaled_pred_value for {ticker_name} is NaN BEFORE inverse transform."
                )

            actual_pred_price = y_scalers[stock_idx].inverse_transform(
                np.array([[scaled_pred_value]])
            )[0][0]

            logger.info(
                f"Ticker: {ticker_name}, Actual Predicted Price after inverse_transform: "
                f"{actual_pred_price}"
            )
            if np.isnan(actual_pred_price):
                logger.warning(
                    f"!!! Actual_pred_price for {ticker_name} is NaN AFTER inverse "
                    f"transform. Scaled input was: {scaled_pred_value}"
                )

            predicted_prices_final[ticker_name] = float(actual_pred_price)
        logger.info(f"Final predicted prices: {predicted_prices_final}")

        today_iso = date.today().isoformat()

        # 7. Save predictions to JSON files (KEEPING THIS FOR API for now)
        predictions_base_dir.mkdir(parents=True, exist_ok=True)
        historical_dir = predictions_base_dir / 'historical'
        historical_dir.mkdir(parents=True, exist_ok=True)
        payload_json = {
            "date": today_iso,
            "predictions": predicted_prices_final,
            "model_mlflow_run_id": prod_model_origin_mlflow_run_id
        }
        latest_file_path = predictions_base_dir / 'latest_predictions.json'
        with open(latest_file_path, 'w+') as f:
            json.dump(payload_json, f, indent=4)
        logger.info(f"Saved latest predictions to JSON: {latest_file_path}")
        historical_file_path = historical_dir / f"{today_iso}.json"
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
                prod_model_origin_mlflow_run_id,  # model_mlflow_run_id
                today_iso  # target_prediction_date_str: prediction is for "today"
            )
        logger.info("Successfully saved predictions to database.")

        return True

    except Exception as e:
        logger.error(f"Error in run_daily_prediction: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Make daily stock predictions using the production model."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/params.yaml',
        help='Path to params.yaml'
    )
    parser.add_argument(
        '--input_sequence_path',
        type=str,
        required=True,
        help='Path to the .npy input sequence.'
    )
    
    # These will now be pulled from XComs by the DAG task
    parser.add_argument(
        '--production_model_uri_for_loading',
        type=str,
        required=True,
        help='MLflow Model URI for loading (e.g., models:/MyModel@Alias or models:/MyModel/Version).'
    )
    parser.add_argument(
        '--production_model_base_name',
        type=str,
        required=True,
        help='Base registered model name.'
    )
    parser.add_argument(
        '--production_model_version_number',
        type=str,
        required=True,
        help='Specific version number of the production model.'
    )
    
    args = parser.parse_args()

    config_path_resolved = Path(args.config).resolve()  # Simplified for brevity
    input_sequence_path_resolved = Path(args.input_sequence_path).resolve()

    logger.info(
        f"Starting daily prediction: config={config_path_resolved}, "
        f"input={input_sequence_path_resolved}, "
        f"model_uri_load={args.production_model_uri_for_loading}, "
        f"model_name={args.production_model_base_name}, "
        f"model_version={args.production_model_version_number}"
    )

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
