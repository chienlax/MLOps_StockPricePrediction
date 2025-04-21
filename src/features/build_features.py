# src/features/build_features.py
import argparse
import pickle
import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Goal: Load processed data, create sequences, split into train/test, scale data, and save scaled data and scalers

# Move create_sequences and scale_data here.
# Add argparse.
# Load parameters from params.yaml.
# Load processed data from the path specified in params.yaml.
# Perform train/test split.
# Scale data and save X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled using np.savez.
# Save the fitted scalers_x and y_scalers using pickle.


# --- Building features functions ---
# create_sequences(data_x, data_y, seq_len, pred_len) -> np.ndarray, np.ndarray
# scale_data(X_train, X_test, y_train, y_test, num_features, num_stocks) -> scaled_data..., scalers...

#%%
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

#%%
# --- Modify scale_data slightly ---
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
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)

    # Load paths and parameters
    processed_data_path = Path(params['output_paths']['processed_data_path'])
    split_data_output_path = Path(params['output_paths']['split_data_path'])
    scalers_output_path = Path(params['output_paths']['scalers_path'])
    split_data_output_path.parent.mkdir(parents=True, exist_ok=True)
    scalers_output_path.parent.mkdir(parents=True, exist_ok=True)

    seq_len = params['feature_engineering']['sequence_length']
    pred_len = params['feature_engineering']['prediction_length']
    train_ratio = 0.8 # Or load from params if needed

    # 1. Load Processed Data
    print(f"--- Loading Processed Data from {processed_data_path} ---")
    data = np.load(processed_data_path, allow_pickle=True)
    processed_data = data['processed_data']
    targets = data['targets']
    # feature_columns = data['feature_columns'] # Not strictly needed here
    # tickers = data['tickers'] # Not strictly needed here
    print("--- Finished Loading Processed Data ---")

    num_features = processed_data.shape[2]
    num_stocks = processed_data.shape[1]

    # 2. Create Sequences
    print("--- Creating Sequences ---")
    X_sequences, y_sequences = create_sequences(processed_data, targets, seq_len, pred_len)
    print(f"X sequences shape: {X_sequences.shape}")
    print(f"y sequences shape: {y_sequences.shape}")
    print("--- Finished Creating Sequences ---")

    # 3. Train-test split
    print("--- Splitting Data ---")
    train_size = int(len(X_sequences) * train_ratio)
    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]
    X_test = X_sequences[train_size:]
    y_test = y_sequences[train_size:]
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    print("--- Finished Splitting Data ---")

    # 4. Scale data
    print("--- Scaling Data ---")
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scalers_x, y_scalers = scale_data(
        X_train, X_test, y_train, y_test, num_features, num_stocks
    )
    print("--- Finished Scaling Data ---")

    # 5. Save Scaled Data
    print(f"--- Saving Scaled Data to {split_data_output_path} ---")
    np.savez(
        split_data_output_path,
        X_train_scaled=X_train_scaled,
        y_train_scaled=y_train_scaled,
        X_test_scaled=X_test_scaled,
        y_test_scaled=y_test_scaled
    )
    print("--- Finished Saving Scaled Data ---")

    # 6. Save Scalers
    print(f"--- Saving Scalers to {scalers_output_path} ---")
    with open(scalers_output_path, 'wb') as f:
        pickle.dump({'scalers_x': scalers_x, 'y_scalers': y_scalers}, f)
    print("--- Finished Saving Scalers ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (params.yaml)')
    args = parser.parse_args()
    run_feature_building(args.config)