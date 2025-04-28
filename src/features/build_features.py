# src/features/build_features.py
import argparse
import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import logging
from datetime import datetime
from src.utils.db_utils import (
    get_db_connection, 
    load_processed_features_from_db,
    save_scaled_features,
    save_scalers
)
import sys
import os

# Set up logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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

def run_feature_building(config_path: str):
    """Run feature building process with PostgreSQL database."""
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)

    # Load database configuration
    db_config = params['database']
    logger.info(f"Using PostgreSQL database at {db_config['host']}:{db_config['port']}")
    
    # Get run_id from params if specified, otherwise generate a new one
    run_id = params.get('run_id')
    if not run_id:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Feature engineering parameters
    seq_len = params['feature_engineering']['sequence_length']
    pred_len = params['feature_engineering']['prediction_length']
    train_ratio = params.get('feature_engineering', {}).get('train_ratio', 0.8)

    # 1. Load Processed Data from database
    logger.info(f"--- Loading Processed Data from database ---")
    processed_data_dict = load_processed_features_from_db(db_config, run_id)
    
    if not processed_data_dict:
        logger.error("No processed data found in database")
        raise ValueError("No processed data found in database")
    
    processed_data = processed_data_dict['processed_data']
    targets = processed_data_dict['targets']
    feature_columns = processed_data_dict['feature_columns']
    tickers = processed_data_dict['tickers']
    
    # If we loaded data from a different run_id, update our run_id to maintain relationship
    if run_id != processed_data_dict['run_id']:
        run_id = processed_data_dict['run_id']
        logger.info(f"Updated run_id to match processed data: {run_id}")
    
    logger.info("--- Creating sequences ---")
    num_stocks = len(tickers)
    num_features = processed_data.shape[2]
    
    # Split data into train and test sets
    train_size = int(len(processed_data) * train_ratio)
    train_data = processed_data[:train_size]
    test_data = processed_data[train_size:]
    train_targets = targets[:train_size]
    test_targets = targets[train_size:]
    
    # Create sequences for training and test sets
    X_train, y_train = create_sequences(train_data, train_targets, seq_len, pred_len)
    X_test, y_test = create_sequences(test_data, test_targets, seq_len, pred_len)
    
    logger.info(f"Created sequences with shapes: X_train {X_train.shape}, y_train {y_train.shape}")
    logger.info(f"Test sequences shapes: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # 3. Scale the data
    logger.info("--- Scaling data ---")
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scalers_x, y_scalers = scale_data(
        X_train, X_test, y_train, y_test, num_features, num_stocks
    )
    logger.info("Data scaling completed")

    # 4. Save Scaled Data to database
    logger.info(f"--- Saving Scaled Data to database ---")
    save_scaled_features(db_config, run_id, 'X_train', X_train_scaled)

    # 5. Save Scaled Data to database
    logger.info(f"--- Saving Scaled Data to database ---")
    save_scaled_features(db_config, run_id, 'X_train', X_train_scaled)
    save_scaled_features(db_config, run_id, 'y_train', y_train_scaled)
    save_scaled_features(db_config, run_id, 'X_test', X_test_scaled)
    save_scaled_features(db_config, run_id, 'y_test', y_test_scaled)
    logger.info("--- Finished Saving Scaled Data ---")

    # 6. Save Scalers to database
    logger.info(f"--- Saving Scalers to database ---")
    scalers_dict = {'scalers_x': scalers_x, 'y_scalers': y_scalers}
    save_scalers(db_config, run_id, scalers_dict)
    logger.info("--- Finished Saving Scalers ---")
    
    return run_id

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, 
                          required=False,
                          help='Path to the configuration file (params.yaml)')

        # Check if arguments were passed
        if len(sys.argv) == 1:
            config_path = 'params.yaml'
            logger.info(f"No config specified, using default: {config_path}")
        else:
            args = parser.parse_args()
            config_path = args.config

        # Verify config file exists
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)

        # Verify database configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'database' not in config:
                logger.error("Database configuration missing from params.yaml")
                sys.exit(1)
            required_db_fields = ['dbname', 'user', 'password', 'host', 'port']
            missing_fields = [field for field in required_db_fields if field not in config['database']]
            if missing_fields:
                logger.error(f"Missing required database fields: {missing_fields}")
                sys.exit(1)

        run_feature_building(config_path)
        logger.info("Feature building completed successfully")
        
    except Exception as e:
        logger.error(f"Error in feature building: {e}", exc_info=True)
        sys.exit(1)