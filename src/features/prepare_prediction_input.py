# src/features/prepare_prediction_input.py
"""Module for preparing prediction input sequences for stock price prediction."""

# Standard library imports
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Third-party imports
import numpy as np
import pandas as pd
import yaml

# Set up import path for local modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Local imports
from data.make_dataset import add_technical_indicators, preprocess_data
from utils.db_utils import (
    get_latest_raw_data_window,
    load_processed_features_from_db,
    load_scalers,
)


# Set up logger
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


def run_prepare_input(
    config_path: str,
    production_model_training_run_id: str,
    output_dir: str
) -> Optional[str]:
    """
    Prepare input sequence for daily prediction using the production model's configuration.

    Args:
        config_path: Path to the params.yaml file.
        production_model_training_run_id: The dataset run_id used to train the current
                                          production model.
        output_dir: Directory to save the temporary output sequence file.

    Returns:
        Optional[str]: Absolute path to the saved .npy input sequence file, or None on failure.
    """
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        db_config = params['database']
        feature_eng_params = params['feature_engineering']
        sequence_length = feature_eng_params['sequence_length']
        
        logger.info(
            f"Preparing prediction input using production model's "
            f"training run_id: {production_model_training_run_id}"
        )

        # 1. Load scalers_x and feature_columns associated with the production model's training
        logger.info("Loading scalers_x and feature_columns from database...")
        scalers_data = load_scalers(db_config, production_model_training_run_id)
        processed_features_meta = load_processed_features_from_db(
            db_config, production_model_training_run_id
        )

        if not scalers_data or 'scalers_x' not in scalers_data:
            logger.error(
                f"Could not load scalers_x for run_id: {production_model_training_run_id}"
            )
            return None
        
        if not processed_features_meta or 'feature_columns' not in processed_features_meta or \
           'tickers' not in processed_features_meta:
            logger.error(
                f"Could not load feature_columns or tickers for run_id: "
                f"{production_model_training_run_id}"
            )
            return None

        # List of lists of MinMaxScaler objects
        scalers_x = scalers_data['scalers_x']
        feature_columns_prod_model = processed_features_meta['feature_columns']
        tickers_prod_model = processed_features_meta['tickers']
        num_stocks_prod_model = len(tickers_prod_model)
        num_features_prod_model = len(feature_columns_prod_model)

        if len(scalers_x) != num_stocks_prod_model or \
           (num_stocks_prod_model > 0 and len(scalers_x[0]) != num_features_prod_model):
            logger.error(
                "Mismatch in dimensions of loaded scalers_x vs feature_columns/tickers."
            )
            logger.error(
                f"Num stocks from tickers: {num_stocks_prod_model}, "
                f"Num stocks in scalers_x: {len(scalers_x)}"
            )
            if num_stocks_prod_model > 0:
                logger.error(
                    f"Num features from columns: {num_features_prod_model}, "
                    f"Num features in scalers_x[0]: "
                    f"{len(scalers_x[0]) if scalers_x else 'N/A'}"
                )
            return None

        # 2. Fetch latest raw data window
        # Buffer for TA indicators (e.g., if max window for an indicator is 50, need 50 prior points)
        MAX_TA_WINDOW_REQUIRED = 200
        # +15 for buffer
        raw_data_lookback_days = sequence_length + MAX_TA_WINDOW_REQUIRED + 15
        logger.info(
            f"Fetching latest {raw_data_lookback_days} days of raw data "
            f"for tickers: {tickers_prod_model}..."
        )
        latest_raw_data_dict = get_latest_raw_data_window(
            db_config, tickers_prod_model, raw_data_lookback_days
        )

        if not latest_raw_data_dict or all(df.empty for df in latest_raw_data_dict.values()):
            logger.error("Failed to fetch any latest raw data.")
            return None

        # Filter out tickers for which no data was fetched
        valid_tickers_fetched = [
            t for t, df in latest_raw_data_dict.items() if not df.empty
        ]
        if not valid_tickers_fetched:
            logger.error("No valid raw data fetched for any target ticker.")
            return None
        
        # Ensure the order of tickers for processing matches tickers_prod_model for consistency with scalers
        # Reconstruct latest_raw_data_dict to maintain order and include only relevant tickers
        ordered_raw_data_dict = {}
        for ticker in tickers_prod_model:
            if ticker in latest_raw_data_dict and not latest_raw_data_dict[ticker].empty:
                ordered_raw_data_dict[ticker] = latest_raw_data_dict[ticker]
            else:
                logger.error(
                    f"Raw data for ticker {ticker} (expected by production model) "
                    f"is missing or empty. Cannot proceed."
                )
                # Or handle by predicting for subset, but for now, require all.
                return None
        
        current_num_stocks = len(ordered_raw_data_dict)
        if current_num_stocks != num_stocks_prod_model:
            logger.error(
                f"Data fetched for {current_num_stocks} stocks, but production model "
                f"expects {num_stocks_prod_model}. Mismatch."
            )
            return None

        # 3. Apply technical indicators
        logger.info("Applying technical indicators to the latest raw data...")
        data_with_ta = add_technical_indicators(ordered_raw_data_dict.copy())

        # 4. Preprocess (e.g., handle NaNs from TAs, but 'Target' is not created here)
        data_preprocessed_for_pred = {}
        for ticker, df in data_with_ta.items():
            df_filled = df.ffill().bfill()
            missing_cols = [
                col for col in feature_columns_prod_model if col not in df_filled.columns
            ]
            if missing_cols:
                logger.error(
                    f"Ticker {ticker}: Missing expected feature columns after TA: {missing_cols}"
                )
                return None
            df_filled = df_filled.fillna(0)
            data_preprocessed_for_pred[ticker] = df_filled[feature_columns_prod_model]

        # 5. Align data across tickers for the feature columns
        # Create a common index from the preprocessed data
        common_idx = pd.Index([])
        for df in data_preprocessed_for_pred.values():
            common_idx = common_idx.union(df.index)
        common_idx = common_idx.sort_values()

        if common_idx.empty:
            logger.error(
                "Common index is empty after preprocessing for prediction. Cannot create input."
            )
            return None

        # Align each ticker's DataFrame to the common index
        aligned_feature_data_list = []
        # Iterate in the order of production model's tickers
        for ticker in tickers_prod_model:
            df = data_preprocessed_for_pred[ticker]
            df_aligned = df.reindex(common_idx).ffill().bfill()
            aligned_feature_data_list.append(df_aligned.values)

        # Stack into (timesteps, num_stocks, num_features)
        if not aligned_feature_data_list:
            logger.error("No data to stack after alignment.")
            return None
            
        final_feature_array = np.zeros(
            (len(common_idx), num_stocks_prod_model, num_features_prod_model)
        )
        for i in range(num_stocks_prod_model):
            final_feature_array[:, i, :] = aligned_feature_data_list[i]
        
        # 6. Extract the last 'sequence_length' timesteps
        if final_feature_array.shape[0] < sequence_length:
            logger.error(
                f"Not enough timesteps ({final_feature_array.shape[0]}) after processing "
                f"to form a sequence of length {sequence_length}."
            )
            return None
        
        # (seq_len, num_stocks, num_features)
        input_sequence_raw = final_feature_array[-sequence_length:, :, :]
        logger.info(f"Raw input sequence for prediction shape: {input_sequence_raw.shape}")

        # 7. Scale the sequence using loaded scalers_x
        input_sequence_scaled = np.zeros_like(input_sequence_raw)
        for stock_idx in range(num_stocks_prod_model):
            for feature_idx in range(num_features_prod_model):
                scaler = scalers_x[stock_idx][feature_idx]
                feature_slice = input_sequence_raw[:, stock_idx, feature_idx].reshape(-1, 1)
                input_sequence_scaled[:, stock_idx, feature_idx] = scaler.transform(
                    feature_slice
                ).flatten()
        
        logger.info(f"Scaled input sequence shape: {input_sequence_scaled.shape}")

        # 8. Reshape for model input (add batch dimension)
        # Model expects (batch_size, seq_len, num_stocks, num_features)
        # (1, seq_len, num_stocks, num_features)
        model_input_sequence = np.expand_dims(input_sequence_scaled, axis=0)
        logger.info(f"Final model input sequence shape: {model_input_sequence.shape}")

        # 9. Save to a temporary file and print path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"prediction_input_sequence_{timestamp}.npy"
        output_file_path = Path(output_dir) / output_filename
        
        np.save(output_file_path, model_input_sequence)
        logger.info(f"Saved prepared input sequence to: {output_file_path.resolve()}")
        
        return str(output_file_path.resolve())

    except Exception as e:
        logger.error(f"Error in run_prepare_input: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare input sequence for daily stock prediction."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/params.yaml',
        help='Path to the configuration file (e.g., config/params.yaml)'
    )
    parser.add_argument(
        '--production_model_training_run_id',
        type=str,
        required=True,
        help="The dataset run_id used when the current production model was trained "
             "(for loading correct scalers/features)."
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        # Default temporary directory
        default='/tmp/stock_prediction_inputs',
        help="Directory to save the output .npy file."
    )
    args = parser.parse_args()

    config_path_resolved = Path(args.config).resolve()
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)
    
    logger.info(
        f"Starting input preparation with config: {config_path_resolved}, "
        f"prod model training run_id: {args.production_model_training_run_id}"
    )

    output_file = run_prepare_input(
        str(config_path_resolved),
        args.production_model_training_run_id,
        args.output_dir
    )

    if output_file:
        logger.info(f"Successfully prepared input sequence: {output_file}")
        # For Airflow XCom capture
        print(f"OUTPUT_PATH:{output_file}")
    else:
        logger.error("Failed to prepare input sequence.")
        sys.exit(1)