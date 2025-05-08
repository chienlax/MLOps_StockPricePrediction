# src/features/build_features.py
import argparse
import yaml
import logging
import sys
import os
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from src.utils.db_utils import (
        get_db_connection, # Not directly used here, but load_processed_features_from_db uses it
        load_processed_features_from_db,
        save_scaled_features,
        save_scalers
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1])) # Add 'src' to path
    from utils.db_utils import (
        load_processed_features_from_db,
        save_scaled_features,
        save_scalers
    )

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------

# --- Building features functions ---
def create_sequences(data_x, data_y, seq_len, pred_len):
    """Create sequences for time series forecasting"""
    sequences_x = []
    sequences_y = []
    for i in range(len(data_x) - seq_len - pred_len + 1):
        seq_x = data_x[i:i + seq_len]
        seq_y = data_y[i + seq_len:i + seq_len + pred_len]
        sequences_x.append(seq_x)
        sequences_y.append(seq_y)
    return np.array(sequences_x), np.array(sequences_y)

def scale_data(X_train, X_test, y_train, y_test, num_features, num_stocks):
    """Scale the data using separate MinMaxScaler for each feature and each stock"""
    # Create a 2D array of scalers for features (num_stocks x num_features)
    scalers_x = [[MinMaxScaler() for _ in range(num_features)] for _ in range(num_stocks)]
    X_train_scaled = np.zeros_like(X_train)
    X_test_scaled = np.zeros_like(X_test)
    
    # Scale each feature for each stock separately
    for stock_idx in range(num_stocks):
        for feature_idx in range(num_features):
            # Fit on training data only
            feature_data = X_train[:, :, stock_idx, feature_idx].reshape(-1, 1)
            scalers_x[stock_idx][feature_idx].fit(feature_data)
            
            # Transform both train and test
            X_train_scaled[:, :, stock_idx, feature_idx] = scalers_x[stock_idx][feature_idx].transform(
                X_train[:, :, stock_idx, feature_idx].reshape(-1, 1)).reshape(X_train.shape[0], -1)
            X_test_scaled[:, :, stock_idx, feature_idx] = scalers_x[stock_idx][feature_idx].transform(
                X_test[:, :, stock_idx, feature_idx].reshape(-1, 1)).reshape(X_test.shape[0], -1)
    
    # Scale targets
    y_scalers = [MinMaxScaler() for _ in range(num_stocks)]
    y_train_scaled = np.zeros_like(y_train)
    y_test_scaled = np.zeros_like(y_test)
    
    for stock_idx in range(num_stocks):
        y_train_for_stock = y_train[:, 0, stock_idx].reshape(-1, 1)
        y_scalers[stock_idx].fit(y_train_for_stock)
        
        y_train_scaled[:, 0, stock_idx] = y_scalers[stock_idx].transform(y_train_for_stock).flatten()
        y_test_scaled[:, 0, stock_idx] = y_scalers[stock_idx].transform(
            y_test[:, 0, stock_idx].reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scalers_x, y_scalers

# ------------------------------------------------------

def run_feature_building(config_path: str, run_id_arg: str) -> Optional[str]:
    """
    Run feature building process using data from a specific run_id.
    Loads processed data, creates sequences, scales data, and saves scaled features and scalers to DB.
    Args:
        config_path (str): Path to the params.yaml configuration file.
        run_id_arg (str): The specific run_id for the processed dataset to use.
    Returns:
        Optional[str]: The run_id_arg if successful, None otherwise.
    """

    # with open(config_path, 'r') as f:
    #     params = yaml.safe_load(f)

    # # Load database configuration
    # db_config = params['database']
    # logger.info(f"Using PostgreSQL database at {db_config['host']}:{db_config['port']}")
    
    # # Get run_id from params if specified, otherwise generate a new one
    # run_id = params.get('run_id')
    # if not run_id:
    #     run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # # Feature engineering parameters
    # seq_len = params['feature_engineering']['sequence_length']
    # pred_len = params['feature_engineering']['prediction_length']
    # train_ratio = params.get('feature_engineering', {}).get('train_ratio', 0.8)

    # # 1. Load Processed Data from database
    # logger.info(f"--- Loading Processed Data from database ---")
    # processed_data_dict = load_processed_features_from_db(db_config, run_id)
    
    # if not processed_data_dict:
    #     logger.error("No processed data found in database")
    #     raise ValueError("No processed data found in database")
    
    # processed_data = processed_data_dict['processed_data']
    # targets = processed_data_dict['targets']
    # feature_columns = processed_data_dict['feature_columns']
    # tickers = processed_data_dict['tickers']
    
    # # If we loaded data from a different run_id, update our run_id to maintain relationship
    # if run_id != processed_data_dict['run_id']:
    #     run_id = processed_data_dict['run_id']
    #     logger.info(f"Updated run_id to match processed data: {run_id}")
    
    # logger.info("--- Creating sequences ---")
    # num_stocks = len(tickers)
    # num_features = processed_data.shape[2]
    
    # # Split data into train and test sets
    # train_size = int(len(processed_data) * train_ratio)
    # train_data = processed_data[:train_size]
    # test_data = processed_data[train_size:]
    # train_targets = targets[:train_size]
    # test_targets = targets[train_size:]
    
    # # Create sequences for training and test sets
    # X_train, y_train = create_sequences(train_data, train_targets, seq_len, pred_len)
    # X_test, y_test = create_sequences(test_data, test_targets, seq_len, pred_len)
    
    # logger.info(f"Created sequences with shapes: X_train {X_train.shape}, y_train {y_train.shape}")
    # logger.info(f"Test sequences shapes: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # # 3. Scale the data
    # logger.info("--- Scaling data ---")
    # X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scalers_x, y_scalers = scale_data(
    #     X_train, X_test, y_train, y_test, num_features, num_stocks
    # )
    # logger.info("Data scaling completed")

    # # 4. Save Scaled Data to database
    # logger.info(f"--- Saving Scaled Data to database ---")
    # save_scaled_features(db_config, run_id, 'X_train', X_train_scaled)

    # # 5. Save Scaled Data to database
    # logger.info(f"--- Saving Scaled Data to database ---")
    # save_scaled_features(db_config, run_id, 'X_train', X_train_scaled)
    # save_scaled_features(db_config, run_id, 'y_train', y_train_scaled)
    # save_scaled_features(db_config, run_id, 'X_test', X_test_scaled)
    # save_scaled_features(db_config, run_id, 'y_test', y_test_scaled)
    # logger.info("--- Finished Saving Scaled Data ---")

    # # 6. Save Scalers to database
    # logger.info(f"--- Saving Scalers to database ---")
    # scalers_dict = {'scalers_x': scalers_x, 'y_scalers': y_scalers}
    # save_scalers(db_config, run_id, scalers_dict)
    # logger.info("--- Finished Saving Scalers ---")
    
    # return run_id

    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)

        db_config = params['database']
        logger.info(f"Feature building using PostgreSQL database at {db_config['host']}:{db_config['port']}")
        
        # Use the passed run_id_arg directly
        current_run_id = run_id_arg 
        logger.info(f"Targeting dataset with run_id: {current_run_id}")
        
        feature_eng_params = params['feature_engineering']
        seq_len = feature_eng_params['sequence_length']
        pred_len = feature_eng_params['prediction_length']
        train_ratio = feature_eng_params.get('train_ratio', 0.8)

        # 1. Load Processed Data from database using current_run_id
        logger.info(f"--- Loading Processed Data from database for run_id: {current_run_id} ---")
        processed_data_dict = load_processed_features_from_db(db_config, run_id=current_run_id)
        
        if not processed_data_dict:
            logger.error(f"No processed data found in database for run_id: {current_run_id}. Cannot build features.")
            return None # Indicate failure
        
        processed_data_np = processed_data_dict['processed_data'] # Shape: (timesteps, stocks, features)
        targets_np = processed_data_dict['targets']             # Shape: (timesteps, stocks)
        feature_columns = processed_data_dict['feature_columns']
        tickers = processed_data_dict['tickers']
        
        if processed_data_np is None or targets_np is None or processed_data_np.size == 0 or targets_np.size == 0:
            logger.error(f"Loaded processed data or targets are empty/None for run_id: {current_run_id}.")
            return None

        logger.info(f"Loaded processed data shape: {processed_data_np.shape}, targets shape: {targets_np.shape}")
        logger.info("--- Finished Loading Processed Data ---")
        
        # 2. Split data into train and test sets BEFORE creating sequences
        logger.info(f"--- Splitting data into train/test (ratio: {train_ratio}) ---")
        if len(processed_data_np) == 0:
            logger.error("Processed data is empty, cannot split.")
            return None
        train_size = int(len(processed_data_np) * train_ratio)
        if train_size < seq_len + pred_len : # Need enough data for at least one sequence
            logger.error(f"Train size ({train_size}) is too small to create any sequences with seq_len={seq_len}, pred_len={pred_len}.")
            return None

        train_features_raw = processed_data_np[:train_size]
        test_features_raw = processed_data_np[train_size:]
        train_targets_raw = targets_np[:train_size]
        test_targets_raw = targets_np[train_size:]
        logger.info(f"Raw train features shape: {train_features_raw.shape}, Raw test features shape: {test_features_raw.shape}")

        # 3. Create sequences for training and test sets
        logger.info("--- Creating sequences ---")
        X_train, y_train = create_sequences(train_features_raw, train_targets_raw, seq_len, pred_len)
        X_test, y_test = create_sequences(test_features_raw, test_targets_raw, seq_len, pred_len)
        
        if X_train.size == 0 or y_train.size == 0:
            logger.error(f"Failed to create training sequences. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}. Check data length and sequence parameters.")
            return None
        # X_test or y_test can be empty if test_features_raw is too short, which is acceptable.
        if X_test.size == 0 and test_features_raw.size > 0 : # only log error if raw test data existed but no sequences were made
            logger.warning(f"Test sequences (X_test) are empty. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}. This might be due to short test data length.")

        logger.info(f"Created sequences. X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # 4. Scale the data
        num_stocks = X_train.shape[2] # X_train shape: (samples, seq_len, num_stocks, num_features)
        num_features = X_train.shape[3]
        
        logger.info("--- Scaling data ---")
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scalers_x, y_scalers = scale_data(
            X_train, X_test, y_train, y_test, num_features, num_stocks
        )
        logger.info("Data scaling completed.")

        # 5. Save Scaled Data to database, associated with current_run_id
        logger.info(f"--- Saving Scaled Data to database for run_id: {current_run_id} ---")
        save_scaled_features(db_config, current_run_id, 'X_train', X_train_scaled)
        save_scaled_features(db_config, current_run_id, 'y_train', y_train_scaled)
        if X_test_scaled.size > 0: # Only save if not empty
            save_scaled_features(db_config, current_run_id, 'X_test', X_test_scaled)
        if y_test_scaled.size > 0: # Only save if not empty
            save_scaled_features(db_config, current_run_id, 'y_test', y_test_scaled)
        logger.info("--- Finished Saving Scaled Data ---")

        # 6. Save Scalers to database, associated with current_run_id
        logger.info(f"--- Saving Scalers to database for run_id: {current_run_id} ---")
        scalers_dict = {'scalers_x': scalers_x, 'y_scalers': y_scalers, 'tickers': tickers, 'num_features': num_features}
        save_scalers(db_config, current_run_id, scalers_dict)
        logger.info("--- Finished Saving Scalers ---")
        
        return current_run_id # Return the run_id to confirm success
    
    except Exception as e:
        logger.error(f"Error in run_feature_building for run_id {run_id_arg}: {e}", exc_info=True)
        return None

#------------------------------------------------------

if __name__ == '__main__':
    # try:
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--config', type=str, 
    #                       required=False,
    #                       help='Path to the configuration file (params.yaml)')

    #     # Check if arguments were passed
    #     if len(sys.argv) == 1:
    #         config_path = 'params.yaml'
    #         logger.info(f"No config specified, using default: {config_path}")
    #     else:
    #         args = parser.parse_args()
    #         config_path = args.config

    #     # Verify config file exists
    #     if not os.path.exists(config_path):
    #         logger.error(f"Configuration file not found: {config_path}")
    #         sys.exit(1)

    #     # Verify database configuration
    #     with open(config_path, 'r') as f:
    #         config = yaml.safe_load(f)
    #         if 'database' not in config:
    #             logger.error("Database configuration missing from params.yaml")
    #             sys.exit(1)
    #         required_db_fields = ['dbname', 'user', 'password', 'host', 'port']
    #         missing_fields = [field for field in required_db_fields if field not in config['database']]
    #         if missing_fields:
    #             logger.error(f"Missing required database fields: {missing_fields}")
    #             sys.exit(1)

    #     run_feature_building(config_path)
    #     logger.info("Feature building completed successfully")
        
    # except Exception as e:
    #     logger.error(f"Error in feature building: {e}", exc_info=True)
    #     sys.exit(1)

    parser = argparse.ArgumentParser(description="Feature building script for stock prediction.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/params.yaml', # Assuming script run from project root
        help='Path to the configuration file (e.g., config/params.yaml)'
    )
    parser.add_argument(
        '--run_id',
        type=str,
        required=True, 
        help='The run_id of the processed dataset (from processed_feature_data table) to use for feature building.'
    )
    args = parser.parse_args()
    config_path_arg = args.config
    cli_run_id_arg = args.run_id # Use a distinct name for the CLI argument

    # Resolve config path
    config_path_resolved = Path(config_path_arg)
    if not config_path_resolved.is_absolute():
        if (Path.cwd() / config_path_resolved).exists():
            config_path_resolved = (Path.cwd() / config_path_resolved).resolve()
        elif (Path(__file__).parent.parent.parent / config_path_resolved).exists():
            config_path_resolved = (Path(__file__).parent.parent.parent / config_path_resolved).resolve()
        else:
            logger.error(f"Configuration file not found: {config_path_arg}")
            sys.exit(1)
    
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)

    logger.info(f"Starting feature building script with resolved config: {config_path_resolved} for run_id: {cli_run_id_arg}")

    try:
        # Basic validation of database config in params.yaml
        with open(config_path_resolved, 'r') as f:
            config = yaml.safe_load(f)
            if 'database' not in config:
                logger.error("Database configuration missing from params.yaml")
                sys.exit(1)
            # Further checks for dbname, user, etc., can be added if desired

        # Call run_feature_building with the CLI-provided run_id
        returned_run_id = run_feature_building(str(config_path_resolved), run_id_arg=cli_run_id_arg)
        
        if returned_run_id:
            logger.info(f"Feature building completed successfully for run_id: {returned_run_id}")
            print(f"FEATURE_BUILD_SUCCESS_RUN_ID:{returned_run_id}") # For Airflow or capture
        else:
            logger.error(f"Feature building failed for run_id: {cli_run_id_arg}. Check logs.")
            sys.exit(1)
            
    except yaml.YAMLError as e_yaml:
        logger.error(f"Error parsing configuration file {config_path_resolved}: {e_yaml}", exc_info=True)
        sys.exit(1)
    except Exception as e_main:
        logger.error(f"Fatal error in feature building script for run_id {cli_run_id_arg}: {e_main}", exc_info=True)
        sys.exit(1)
