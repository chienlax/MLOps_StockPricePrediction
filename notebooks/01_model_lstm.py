# %%
# %pip install --quiet yfinance ta optuna torch scikit-learn matplotlib 
import time

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import optuna
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


# %%
def load_data(tickers_list, period="3y", interval="1d"):
    ticker_data = {}
    for t in tickers_list:
        try:
            ticker_data[t] = yf.Ticker(t).history(period=period, interval=interval)
            print(f"Loaded data for {t}")
            time.sleep(15)  
        except Exception as e:
            print(f"Error loading {t}: {e}")
    
    # Filter out empty dataframes
    ticker_data = {ticker: data for ticker, data in ticker_data.items() if not data.empty}
    return ticker_data

# %%
def add_technical_indicators(ticker_data):
    for t in ticker_data:
        df = ticker_data[t]
        # Core indicators
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], 50).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], 200).ema_indicator()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        df['MACD'] = ta.trend.MACD(df['Close'], window_fast=12, window_sign=9, window_slow=26).macd()
        df['BB_High'] = ta.volatility.BollingerBands(df['Close'], window=15, window_dev=2).bollinger_hband()
        df['BB_Low'] = ta.volatility.BollingerBands(df['Close'], window=15, window_dev=2).bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        
        # Additional trend indicators
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], 50).sma_indicator()
        df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
        df['PSAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()
        
        # Momentum indicators
        df['Stochastic_K'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        df['Stochastic_D'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch_signal()
        df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
        
        # Volatility indicators
        df['Donchian_High'] = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20).donchian_channel_hband()
        df['Donchian_Low'] = ta.volatility.DonchianChannel(df['High'], df['Low'], df['Close'], window=20).donchian_channel_lband()
        df['Keltner_High'] = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close']).keltner_channel_hband()
        df['Keltner_Low'] = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close']).keltner_channel_lband()
        
        # Price-based features
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Rate_Of_Change'] = ta.momentum.ROCIndicator(df['Close'], 12).roc()
        
        # Volume-based indicators
        df['Volume_SMA'] = ta.trend.SMAIndicator(df['Volume'], 20).sma_indicator()
        df['Chaikin_Money_Flow'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=20).chaikin_money_flow()
        df['Force_Index'] = ta.volume.ForceIndexIndicator(df['Close'], df['Volume']).force_index()
        
        # Trend-strength indicators
        df['DI_Positive'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx_pos()
        df['DI_Negative'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx_neg()
    
        ticker_data[t] = df
    return ticker_data

# %%
def preprocess_data(df):
    # Fill forward then backward to handle NaNs
    df = df.fillna(method='ffill').fillna(method='bfill')
    # Create target variable (next day's close price)
    df['Target'] = df['Close'].shift(-1)
    # Drop the last row since it will have NaN target
    df = df.dropna()
    return df

# %%
def align_and_process_data(ticker_data):
    tickers = list(ticker_data.keys())
    
    ticker_data = {t: preprocess_data(df) for t, df in ticker_data.items()}
    
    all_indices = set().union(*[ticker_data[d].index for d in ticker_data])
    aligned_data = {}
    
    for t in tickers:
        aligned_data[t] = ticker_data[t].reindex(index=all_indices).sort_index()
    
    # Get feature columns (excluding Target)
    first_ticker = tickers[0]
    feature_columns = [col for col in aligned_data[first_ticker].columns if col != 'Target']
    num_features = len(feature_columns)
    num_stocks = len(tickers)
    
    # Create 3D arrays: (timesteps, stocks, features)
    processed_data = np.zeros((len(all_indices), num_stocks, num_features))
    targets = np.zeros((len(all_indices), num_stocks))
    
    for i, ticker in enumerate(tickers):
        df = aligned_data[ticker][feature_columns + ['Target']].fillna(method='ffill').fillna(method='bfill')
        processed_data[:, i, :] = df[feature_columns].values
        targets[:, i] = df['Target'].values
    
    # Clean any remaining NaNs
    nan_mask = np.isnan(processed_data).any(axis=(1, 2)) | np.isnan(targets).any(axis=1)
    processed_data = processed_data[~nan_mask]
    targets = targets[~nan_mask]
    
    return processed_data, targets, feature_columns, tickers

# %%
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

# %%
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

# %%
def filter_correlated_features(ticker_data, threshold=0.9):
    """
    Analyze feature correlations and remove highly correlated features
    Returns filtered data and list of features to keep
    """
    print(f"Filtering highly correlated features with threshold {threshold}")
    
    # Use the first ticker as reference for correlation analysis
    first_ticker = list(ticker_data.keys())[0]
    df = ticker_data[first_ticker].copy()
    
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Create a mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Identified {len(to_drop)} highly correlated features to remove: {to_drop}")

    if 'Close' in to_drop:
        print("Excluding 'Close' column from removal list.")
        to_drop.remove('Close')

    print(f"Final list of {len(to_drop)} features to remove: {to_drop}")
    
    # Features to keep
    features_to_keep = [col for col in df.columns if col not in to_drop]
    
    # Filter features for all tickers
    filtered_ticker_data = {}
    for ticker, data in ticker_data.items():
        filtered_ticker_data[ticker] = data[features_to_keep]
    
    return filtered_ticker_data, features_to_keep

# %%
class StockLSTM(nn.Module):
    def __init__(self, num_stocks, num_features, hidden_size, num_layers, dropout_rate=0.2):
        super(StockLSTM, self).__init__()
        self.num_stocks = num_stocks
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_stocks * num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_stocks)  # Output for all stocks
        
    def forward(self, x):
        batch_size, seq_len, n_stocks, n_features = x.size()
        # Reshape input for LSTM: (batch, seq_len, num_stocks * features)
        x = x.reshape(batch_size, seq_len, n_stocks * n_features)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take only the last timestep's output
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        
        # Fully connected layer
        out = self.fc(lstm_out)
        
        # Reshape to (batch_size, 1, num_stocks) to match target shape
        return out.view(batch_size, 1, self.num_stocks)


# %%
class StockLSTMWithAttention(nn.Module):
    def __init__(self, num_stocks, num_features, hidden_size, num_layers, dropout_rate=0.2):
        super(StockLSTMWithAttention, self).__init__()
        self.num_stocks = num_stocks
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_stocks * num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=4,
            dropout=dropout_rate
        )
        
        # Dropout and fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_stocks)
        
    def forward(self, x):
        batch_size, seq_len, n_stocks, n_features = x.size()
        # Reshape input for LSTM: (batch, seq_len, num_stocks * features)
        x = x.reshape(batch_size, seq_len, n_stocks * n_features)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply self-attention
        # Reshape for attention: (seq_len, batch, hidden)
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(
            lstm_out_transposed, 
            lstm_out_transposed, 
            lstm_out_transposed
        )
        
        # Reshape back: (batch, seq_len, hidden)
        attn_out = attn_out.transpose(0, 1)
        
        # Take only the last timestep's output after attention
        final_out = attn_out[:, -1, :]
        
        # Fully connected layers
        final_out = self.dropout(final_out)
        final_out = self.fc1(final_out)
        final_out = self.relu(final_out)
        final_out = self.fc2(final_out)
        
        # Reshape to (batch_size, 1, num_stocks) to match target shape
        return final_out.view(batch_size, 1, self.num_stocks)

# %%
class StockLSTMWithCrossStockAttention(nn.Module):
    def __init__(self, num_stocks, num_features, hidden_size, num_layers, dropout_rate=0.2):
        super(StockLSTMWithCrossStockAttention, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Separate LSTM for each stock
        self.stock_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_rate if num_layers > 1 else 0
            ) for _ in range(num_stocks)
        ])
        
        # Cross-stock attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=4,
            dropout=dropout_rate
        )
        
        # Dropout and output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)  # One output per stock
        
    def forward(self, x):
        batch_size, seq_len, n_stocks, n_features = x.size()
        
        # Process each stock separately with its own LSTM
        stock_outputs = []
        for stock_idx in range(self.num_stocks):
            # Extract data for this stock
            stock_data = x[:, :, stock_idx, :]  # (batch, seq, features)
            
            # Run through stock-specific LSTM
            lstm_out, _ = self.stock_lstms[stock_idx](stock_data)
            
            # Take the last output
            last_out = lstm_out[:, -1, :]  # (batch, hidden)
            stock_outputs.append(last_out.unsqueeze(0))  # (1, batch, hidden)
        
        # Concatenate all stock outputs
        stock_outputs = torch.cat(stock_outputs, dim=0)  # (n_stocks, batch, hidden)
        
        # Apply cross-stock attention
        attn_out, _ = self.cross_attention(
            stock_outputs, 
            stock_outputs, 
            stock_outputs
        )
        
        # Process each stock's output after attention
        final_outputs = []
        for stock_idx in range(self.num_stocks):
            stock_repr = attn_out[stock_idx]  # (batch, hidden)
            stock_repr = self.dropout(stock_repr)
            stock_out = self.fc(stock_repr)  # (batch, 1)
            final_outputs.append(stock_out)
        
        # Stack and reshape to (batch, 1, n_stocks)
        final_output = torch.stack(final_outputs, dim=2)  # (batch, 1, n_stocks)
        return final_output

# %%
def evaluate_model(model, test_loader, criterion, y_scalers, device):
    """Evaluate the model and return predictions and metrics"""
    model.eval()
    total_test_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_sequences, batch_targets in test_loader:
            batch_sequences = batch_sequences.to(device)
            batch_targets = batch_targets.to(device)
            
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_targets)
            total_test_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
    
    avg_test_loss = total_test_loss / len(test_loader)
    
    # Combine batches
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Calculate additional metrics
    metrics = {
        'test_loss': avg_test_loss,
        'mse_per_stock': [],
        'mape_per_stock': [],
        'direction_accuracy': []
    }
    
    # Transform back to original scale and calculate metrics per stock
    num_stocks = predictions.shape[2]
    
    for stock_idx in range(num_stocks):
        # Inverse transform
        pred_stock = y_scalers[stock_idx].inverse_transform(predictions[:, 0, stock_idx].reshape(-1, 1))
        true_stock = y_scalers[stock_idx].inverse_transform(targets[:, 0, stock_idx].reshape(-1, 1))
        
        # Calculate MSE and MAPE
        mse = mean_squared_error(true_stock, pred_stock)
        mape = mean_absolute_percentage_error(true_stock, pred_stock)
        
        # Calculate direction accuracy
        pred_direction = np.diff(pred_stock.flatten())
        true_direction = np.diff(true_stock.flatten())
        direction_accuracy = np.mean((pred_direction * true_direction) > 0)
        
        metrics['mse_per_stock'].append(mse)
        metrics['mape_per_stock'].append(mape)
        metrics['direction_accuracy'].append(direction_accuracy)
    
    metrics['avg_mse'] = np.mean(metrics['mse_per_stock'])
    metrics['avg_mape'] = np.mean(metrics['mape_per_stock'])
    metrics['avg_direction_accuracy'] = np.mean(metrics['direction_accuracy'])
    
    return predictions, targets, metrics

# %%
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

# %%
def visualize_predictions(predictions, targets, y_scalers, tickers, num_points=20):
    """Visualize predictions vs actual values for each stock"""
    num_stocks = len(tickers)
    
    for stock_idx in range(num_stocks):
        # Inverse transform
        pred_stock = y_scalers[stock_idx].inverse_transform(predictions[-num_points:, 0, stock_idx].reshape(-1, 1))
        true_stock = y_scalers[stock_idx].inverse_transform(targets[-num_points:, 0, stock_idx].reshape(-1, 1))
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(true_stock, label=f'Actual Price ({tickers[stock_idx]})', color='blue')
        plt.plot(pred_stock, label=f'Predicted Price ({tickers[stock_idx]})', color='red')
        plt.title(f'Actual vs. Predicted Price for {tickers[stock_idx]} (Last {num_points} Timesteps)')
        plt.xlabel('Timestep')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save figure for MLflow
        plt_path = f'prediction_{tickers[stock_idx]}.png'
        plt.savefig(plt_path)
        mlflow.log_artifact(plt_path)
        plt.close()

# %%
def train_final_model(best_params, X_train, y_train, X_test, y_test, num_features, num_stocks, y_scalers, tickers, device):
    """Train the final model with the best hyperparameters"""
    
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
    
    # MLflow tracking for final model - with nested=True
    with mlflow.start_run(run_name="final_model", nested=True):
        # Log best parameters
        mlflow.log_params(best_params)
        mlflow.log_param('num_stocks', num_stocks)
        mlflow.log_param('num_features', num_features)
        mlflow.log_param('tickers', str(tickers))
        
        # Full training loop
        num_epochs = 50
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
        
        # Load best model for final evaluation
        model.load_state_dict(best_model)
        predictions, targets, final_metrics = evaluate_model(model, test_loader, criterion, y_scalers, device)
        
        # Log final metrics
        mlflow.log_metrics({
            'final_test_loss': final_metrics['test_loss'],
            'final_avg_mse': final_metrics['avg_mse'],
            'final_avg_mape': final_metrics['avg_mape'],
            'final_avg_direction_accuracy': final_metrics['avg_direction_accuracy']
        })
        
        # Log model
        mlflow.pytorch.log_model(model, "model")
        
        # Visualize predictions
        visualize_predictions(predictions, targets, y_scalers, tickers)
        
    return model, final_metrics

# %%
def main():
    # Set up MLflow experiment
    mlflow.set_experiment("Stock_Price_Prediction_LSTM_Enhanced")
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load ticker data
    tickers_list = ['AAPL','MA','CSCO','MSFT','AMZN','GOOG','IBM']
    ticker_data = load_data(tickers_list, period="3y", interval="1d")
    
    # Use a single MLflow run for the entire process
    with mlflow.start_run(run_name="enhanced_model"):
        # Add technical indicators
        ticker_data = add_technical_indicators(ticker_data)
        
        # Filter highly correlated features
        ticker_data, remaining_features = filter_correlated_features(ticker_data, threshold=0.9)
        print(f"Remaining features after correlation filtering: {len(remaining_features)}")
        mlflow.log_param("num_features_after_correlation_filter", len(remaining_features))
        
        # Process data
        processed_data, targets, feature_columns, tickers = align_and_process_data(ticker_data)
        print(f"Data shape: {processed_data.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Create sequences
        sequence_length = 30
        prediction_length = 1
        X_sequences, y_sequences = create_sequences(processed_data, targets, sequence_length, prediction_length)
        
        # Train-test split
        train_ratio = 0.8
        train_size = int(len(X_sequences) * train_ratio)
        X_train = X_sequences[:train_size]
        y_train = y_sequences[:train_size]
        X_test = X_sequences[train_size:]
        y_test = y_sequences[train_size:]
        
        # Scale data
        num_features = processed_data.shape[2]
        num_stocks = len(tickers)
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scalers_x, y_scalers = scale_data(
            X_train, X_test, y_train, y_test, num_features, num_stocks)
        
        # Hyperparameter optimization with Optuna
        # Important: End the current MLflow run before optimization and start a new one after
        mlflow.end_run()  # End the current run before optimization
        
        # Run optimization without MLflow
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(
            trial, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 
            num_features, num_stocks, y_scalers, device
        ), n_trials=20)
        
        print("Best hyperparameters:", study.best_params)
        
        # Start a new MLflow run for the final training
        with mlflow.start_run(run_name="enhanced_model_final"):
            # Log the best parameters from optimization
            mlflow.log_params(study.best_params)
            
            # Train final model with best parameters
            final_model, final_metrics = train_final_model(
                study.best_params, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
                num_features, num_stocks, y_scalers, tickers, device
            )
            
            print("Final model metrics:")
            for metric_name, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"{metric_name}: {value:.4f}")
            
            # Save the model
            torch.save(final_model, 'lstm_stock_model_enhanced.pth')
            print("Model saved to lstm_stock_model_enhanced.pth")

if __name__ == "__main__":
    main()
