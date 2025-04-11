# src/data/make_dataset.py
import argparse
import time
import pickle
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from pathlib import Path
import logging
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler() # Defaults to stderr
    handler.setLevel(logging.INFO) # <--- ADD THIS LINE
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def load_data(tickers_list, period, interval, fetch_delay, output_dir):
    """Loads data for tickers and saves each as a pickle file."""
    output_dir = Path(output_dir)
    
    # --- DEBUGGING ---
    abs_output_dir = output_dir.resolve() 
    logger.info(f"Attempting to use RELATIVE output dir: {output_dir}")
    logger.info(f"Attempting to use ABSOLUTE output dir: {abs_output_dir}")
    logger.info(f"Current Working Directory inside container: {os.getcwd()}")
    # --- END DEBUGGING ---

    try:
        logger.info(f"Attempting to create directory: {abs_output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Successfully called mkdir for: {abs_output_dir} (might not indicate success if permissions are wrong)")
    except Exception as e:
         logger.error(f"ERROR during mkdir for {abs_output_dir}: {e}", exc_info=True)
         # If mkdir fails, we definitely won't save files
         return 

    logger.info(f"Saving raw data to: {abs_output_dir}") # Log the absolute path

    for t in tickers_list:
        output_path = output_dir / f"{t}_raw.pkl"
        if output_path.exists():
            logger.info(f"Skipping download for {t}, file exists: {output_path}")
            continue
        try:
            data = yf.Ticker(t).history(period=period, interval=interval)
            if not data.empty:
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Loaded and saved raw data for {t} to {output_path}") 
            else:
                logger.warning(f"No data loaded for {t}")
            time.sleep(fetch_delay)
        except Exception as e:
            logger.error(f"Error loading {t}: {e}", exc_info=True)

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

def preprocess_data(df):
    # Fill forward then backward to handle NaNs
    # df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.ffill().bfill()
    # Create target variable (next day's close price)
    df['Target'] = df['Close'].shift(-1)
    # Drop the last row since it will have NaN target
    df = df.dropna()
    return df

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

def run_processing(config_path: str):
    """Main function to run all data processing steps."""
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug(f"Loaded parameters: {params}") # Use logger.debug for verbose info

        # Define paths from config
        raw_data_dir = Path(params['output_paths']['raw_data_template']).parent
        processed_output_path = Path(params['output_paths']['processed_data_path'])
        processed_output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Raw data directory: {raw_data_dir}")
        logger.info(f"Processed data output path: {processed_output_path}")

        tickers_list = params['data_loading']['tickers']
        period = params['data_loading']['period']
        interval = params['data_loading']['interval']
        fetch_delay = params['data_loading']['fetch_delay']
        corr_threshold = params['feature_engineering']['correlation_threshold']

        # 1. Load Raw Data (or ensure it's downloaded)
        logger.info("--- Starting Data Loading ---")
        load_data(tickers_list, period, interval, fetch_delay, raw_data_dir)
        logger.info("--- Finished Data Loading ---")

        # Load from saved pickles
        ticker_data = {}
        for t in tickers_list:
            raw_path = raw_data_dir / f"{t}_raw.pkl"
            if raw_path.exists():
                with open(raw_path, 'rb') as f:
                    ticker_data[t] = pickle.load(f)
            else:
                logger.warning(f"Raw data file not found for {t} at {raw_path}")

        # Filter out any tickers where data wasn't loaded
        ticker_data = {k: v for k, v in ticker_data.items() if v is not None and not v.empty}
        if not ticker_data:
            logger.error("No valid ticker data loaded. Exiting.")
            raise ValueError("No valid ticker data loaded. Exiting.")
        loaded_tickers = list(ticker_data.keys())

        # 2. Add Technical Indicators
        logger.info("--- Starting Feature Engineering (Indicators) ---")
        ticker_data = add_technical_indicators(ticker_data)
        logger.info("--- Finished Feature Engineering (Indicators) ---")

        # 3. Filter Correlated Features
        logger.info("--- Starting Feature Filtering ---")
        ticker_data, remaining_features = filter_correlated_features(ticker_data, corr_threshold)
        logger.info(f"Features remaining after filtering: {len(remaining_features)}")
        logger.info("--- Finished Feature Filtering ---")

        # 4. Align and Process
        logger.info("--- Starting Data Alignment & Processing ---")
        processed_data, targets, feature_columns, final_tickers = align_and_process_data(ticker_data)
        logger.info(f"Processed data shape: {processed_data.shape}")
        logger.info(f"Targets shape: {targets.shape}")
        logger.info(f"Final tickers in order: {final_tickers}")
        logger.info("--- Finished Data Alignment & Processing ---")

        # 5. Save Processed Data
        # logger.info(f"--- Saving Processed Data to {processed_output_path} ---")
        # try:
        #     np.savez(
        #         processed_output_path,
        #         processed_data=processed_data,
        #         targets=targets,
        #         feature_columns=np.array(feature_columns, dtype=object),
        #         tickers=np.array(final_tickers, dtype=object)
        #     )
        #     logger.info(f"Successfully saved processed data to {processed_output_path}")
        # except Exception as e:
        #         logger.error(f"Failed to save processed data to {processed_output_path}", exc_info=True)
        #         raise # Re-raise the exception so Airflow task fails
        # logger.info("--- Finished Saving Processed Data ---")

        absolute_save_path = processed_output_path.resolve()
        logger.info(f"Attempting to save to absolute path: {absolute_save_path}")
        try:
            np.savez(
                processed_output_path,
                processed_data=processed_data,
                targets=targets,
                feature_columns=np.array(feature_columns, dtype=object),
                tickers=np.array(final_tickers, dtype=object)
            )
            logger.info(f"Successfully saved processed data to {processed_output_path}")
            # Add an existence check right after saving
            if absolute_save_path.exists():
                logger.info(f"Verified file exists at: {absolute_save_path}")
            else:
                logger.error(f"!!! File DOES NOT exist immediately after saving at: {absolute_save_path}")
        except Exception as e:
            logger.error(f"Failed to save processed data to {processed_output_path}", exc_info=True)
            raise


    except Exception as e:
        logger.error("An error occurred during the data processing pipeline.", exc_info=True)
        raise # Re-raise the exception to ensure Airflow task fails

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        required=True, help='Path to the configuration file (params.yaml)')
    args = parser.parse_args()

    logger.info(f"Starting data processing script with config: {args.config}")
    print("--- PRINT TEST: Starting Data Loading ---")

    run_processing(args.config)
    logger.info("Data processing script finished.")