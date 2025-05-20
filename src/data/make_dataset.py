# src/data/make_dataset.py
import argparse
import time
import pickle
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Union, Any, Callable, Iterable
from pathlib import Path

try:
    from src.utils.db_utils import (
        setup_database,
        check_ticker_exists, 
        save_to_raw_table,
        load_data_from_db,
        save_processed_features_to_db,
        get_last_data_timestamp_for_ticker,
        get_db_connection 
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1])) # Add 'src' to path
    from utils.db_utils import (
        setup_database,
        check_ticker_exists,
        save_to_raw_table,
        load_data_from_db,
        save_processed_features_to_db,
        get_last_data_timestamp_for_ticker,
        get_db_connection
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ------------------------------------------------------

def load_data(tickers_list: list, period: str, interval: str, fetch_delay: int, db_config: dict):
    """
    Loads raw stock data for the given tickers from Yahoo Finance.
    - If a ticker has existing data in the database, it fetches new data incrementally.
    - Otherwise, it fetches data for the specified 'period'.
    Saves all fetched data to the 'raw_stock_data' table in the database.
    
    Args:
        tickers_list (list): List of stock tickers (e.g., ['AAPL', 'MSFT']).
        period (str): The default period to fetch if no data exists for a ticker (e.g., "1y", "max").
                      This is IGNORED if fetching incrementally for a ticker.
        interval (str): Data interval (e.g., "1d", "1h").
        fetch_delay (int): Seconds to wait between API calls for different tickers.
        db_config (dict): Database connection configuration.
    """
    # setup_database(db_config) # Moved to run_processing to be called once.

    logger.info("Starting raw data loading/updating from Yahoo Finance.")

    for t in tickers_list:
        last_known_date_in_db = get_last_data_timestamp_for_ticker(db_config, t)
        # last_known_date_in_db = None
        
        yf_start_date_str = None
        fetch_description = ""

        if last_known_date_in_db:
            # yfinance 'start' is inclusive. We want data *after* the last known date.
            # Ensure last_known_date_in_db is a datetime object for timedelta
            if isinstance(last_known_date_in_db, str): # Should not happen if db_utils returns datetime
                last_known_date_in_db = pd.to_datetime(last_known_date_in_db)

            yf_start_date = last_known_date_in_db + timedelta(days=1)
            yf_start_date_str = yf_start_date.strftime('%Y-%m-%d')
            
            # Check if start_date is in the future (e.g. trying to fetch on a weekend for next monday)
            if yf_start_date > datetime.now():
                logger.info(f"Calculated start date {yf_start_date_str} for {t} is in the future. No data to fetch.")
                continue

            fetch_description = f"incrementally for {t} from {yf_start_date_str}"
        else:
            # No data in DB for this ticker, fetch the full 'period'
            fetch_description = f"full period '{period}' for {t}"

        try:
            logger.info(f"Attempting to download data {fetch_description}")
            
            # If yf_start_date_str is set, 'period' will be ignored by yfinance.
            # If yf_start_date_str is None, 'period=period' will be used.
            # yfinance's history() period argument is used if start and end are not provided.
            # If start is provided, period is ignored.
            data = yf.Ticker(t).history(
                period=period if not yf_start_date_str else None,
                start=yf_start_date_str, # Will be None if fetching full period
                interval=interval
            )
            
            if data.empty:
                if yf_start_date_str:
                    logger.info(f"No new data found for {t} since {last_known_date_in_db.strftime('%Y-%m-%d') if last_known_date_in_db else 'beginning'}.")
                else:
                    logger.warning(f"No data returned by yfinance for {t} for period '{period}'.")
                continue # Move to the next ticker
                
            logger.info(f"Downloaded {len(data)} records for {t} from yfinance.")
            
            records_added = save_to_raw_table(t, data, db_config)
            logger.info(f"Saved/Updated {records_added} records for {t} in the database.")
            
            if fetch_delay > 0:
                time.sleep(fetch_delay)
            
        except Exception as e_ticker:
            logger.error(f"Error processing ticker {t}: {e_ticker}", exc_info=True)
            continue # Continue to the next ticker
            
    logger.info("Finished raw data loading/updating from Yahoo Finance.")

# ------------------------------------------------------

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

# ------------------------------------------------------

def preprocess_data(df):
    try:
        original_shape = df.shape
        # Fill forward then backward to handle NaNs
        df = df.ffill().bfill()
        
        # Check if we have NaNs after filling
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Still have {nan_count} NaN values after fill forward/backward")
            
        # Create target variable (next day's close price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop the last row since it will have NaN target
        df = df.dropna()
        
        logger.debug(f"Preprocessed data shape changed from {original_shape} to {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}", exc_info=True)
        raise

# ------------------------------------------------------

def align_and_process_data(ticker_data):
    try:
        tickers = list(ticker_data.keys())
        logger.info(f"Aligning and processing data for {len(tickers)} tickers")
        
        ticker_data = {t: preprocess_data(df) for t, df in ticker_data.items()}
        
        all_indices = set().union(*[ticker_data[d].index for d in ticker_data])
        logger.info(f"Combined dataset has {len(all_indices)} timepoints")
        
        aligned_data = {}
        
        for t in tickers:
            aligned_data[t] = ticker_data[t].reindex(index=all_indices).sort_index()
        
        # Get feature columns (excluding Target)
        first_ticker = tickers[0]
        feature_columns = [col for col in aligned_data[first_ticker].columns if col != 'Target']
        num_features = len(feature_columns)
        num_stocks = len(tickers)
        
        logger.info(f"Creating 3D array with dimensions: timepoints={len(all_indices)}, stocks={num_stocks}, features={num_features}")
        
        # Create 3D arrays: (timesteps, stocks, features)
        processed_data = np.zeros((len(all_indices), num_stocks, num_features))
        targets = np.zeros((len(all_indices), num_stocks))
        
        for i, ticker in enumerate(tickers):
            df = aligned_data[ticker][feature_columns + ['Target']].ffill().bfill() #fillna(method='ffill').fillna(method='bfill')
            processed_data[:, i, :] = df[feature_columns].values
            targets[:, i] = df['Target'].values
        
        # Check for NaNs
        nan_count = np.isnan(processed_data).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in processed data before cleaning")
            
        # Clean any remaining NaNs
        nan_mask = np.isnan(processed_data).any(axis=(1, 2)) | np.isnan(targets).any(axis=1)
        processed_data = processed_data[~nan_mask]
        targets = targets[~nan_mask]
        
        logger.info(f"Final processed data shape after NaN removal: {processed_data.shape}")
        logger.info(f"Final targets shape after NaN removal: {targets.shape}")
        
        return processed_data, targets, feature_columns, tickers
        
    except Exception as e:
        logger.error(f"Error in align_and_process_data: {e}", exc_info=True)
        raise

# ------------------------------------------------------

def filter_correlated_features(ticker_data, threshold=0.9):
    """
    Analyze feature correlations and remove highly correlated features
    Returns filtered data and list of features to keep
    """
    try:
        logger.info(f"Filtering highly correlated features with threshold {threshold}")
        
        # Use the first ticker as reference for correlation analysis
        first_ticker = list(ticker_data.keys())[0]
        df = ticker_data[first_ticker].copy()
        
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()
        
        # Create a mask for the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        logger.info(f"Identified {len(to_drop)} highly correlated features to remove")
        
        if 'Close' in to_drop:
            logger.info("Excluding 'Close' column from removal list.")
            to_drop.remove('Close')
        
        # Print top correlations for debugging
        top_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > threshold:
                    top_corr_pairs.append((col1, col2, corr_value))
        
        # Sort by correlation value and print top 10
        top_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        logger.info("Top correlated feature pairs:")
        for col1, col2, corr in top_corr_pairs[:10]:
            logger.info(f"  {col1} - {col2}: {corr:.4f}")
        
        logger.info(f"Final list of {len(to_drop)} features to remove")
        
        # Features to keep
        features_to_keep = [col for col in df.columns if col not in to_drop]
        
        # Filter features for all tickers
        filtered_ticker_data = {}
        for ticker, data in ticker_data.items():
            filtered_ticker_data[ticker] = data[features_to_keep]
        
        return filtered_ticker_data, features_to_keep
        
    except Exception as e:
        logger.error(f"Error in feature correlation filtering: {e}", exc_info=True)
        # Return original data if there's an error
        logger.info("Returning original data due to error in correlation filtering")
        return ticker_data, list(ticker_data[list(ticker_data.keys())[0]].columns)

# ------------------------------------------------------

def run_processing(config_path: str, mode: str = 'full_process') -> Optional[str]:
    """
    Main function to run data processing steps.
    - 'incremental_fetch' mode: Calls load_data to fetch only new raw data and saves to DB.
    - 'full_process' mode: Calls load_data to ensure raw data is up-to-date,
                           then processes all raw data from DB for training, saves features, and returns a run_id.
    """
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")

        # Get database configuration
        logger.info("Setting up/Verifying database schema...")
        db_config = params['database']
        setup_database(db_config)
        logger.info(f"Using database configuration: host={db_config['host']}, dbname={db_config['dbname']}")

        # Log key configuration parameters
        logger.info(f"Processing with parameters: tickers={params['data_loading']['tickers']}, "
                   f"period={params['data_loading']['period']}, "
                   f"interval={params['data_loading']['interval']}")

        data_loading_params = params['data_loading']
        tickers_list = data_loading_params['tickers']

        period = data_loading_params['period']
        default_fetch_period = data_loading_params['period']

        interval = data_loading_params['interval']
        fetch_delay = data_loading_params['fetch_delay']
        corr_threshold = params['feature_engineering']['correlation_threshold']

        if mode == 'incremental_fetch':
            logger.info("--- Starting Mode: Incremental Raw Data Fetch ---")
            load_data(tickers_list, default_fetch_period, interval, fetch_delay, db_config)
            logger.info("--- Finished Mode: Incremental Raw Data Fetch ---")
            return None
        
        elif mode == 'full_process':
            logger.info("--- Starting Mode: Full Data Processing for Training ---")
            
            logger.info("Step 1.1: Ensuring raw data is up-to-date (calling load_data)...")
            load_data(tickers_list, default_fetch_period, interval, fetch_delay, db_config)
            logger.info("Step 1.1: Finished ensuring raw data is up-to-date.")

            logger.info("Step 1.2: Loading all available raw data from database...")
            ticker_data_from_db = load_data_from_db(db_config, tickers_list)
            if not ticker_data_from_db or all(df.empty for df in ticker_data_from_db.values()):
                logger.error("No raw data found in the database for any ticker after load_data. Cannot proceed.")
                return None
            logger.info(f"Loaded data for {len(ticker_data_from_db)} tickers from database.")

            logger.info("Step 1.3: Adding technical indicators...")
            ticker_data_with_ta = add_technical_indicators(ticker_data_from_db.copy())
            logger.info("Step 1.3: Finished adding technical indicators.")
            
            logger.info("Step 1.4: Preprocessing individual ticker data...")
            ticker_data_preprocessed = {}
            for ticker, df_ta in ticker_data_with_ta.items():
                if df_ta.empty:
                    logger.warning(f"DataFrame for {ticker} is empty after TA. Skipping preprocessing for it.")
                    continue
                ticker_data_preprocessed[ticker] = preprocess_data(df_ta.copy())
            
            if not ticker_data_preprocessed or all(df.empty for df in ticker_data_preprocessed.values()):
                logger.error("No data available after preprocessing all tickers. Cannot proceed.")
                return None
            logger.info("Step 1.4: Finished preprocessing individual ticker data.")

            logger.info("Step 1.5: Aligning data and creating final numpy arrays...")
            processed_data_np, targets_np, feature_columns_list, final_tickers_list = align_and_process_data(
                ticker_data_preprocessed.copy() 
            )
            if processed_data_np is None or targets_np is None: # Check if align_and_process_data indicated failure
                logger.error("Failed to align and process data into numpy arrays. Cannot proceed.")
                return None
            logger.info("Step 1.5: Finished aligning data and creating numpy arrays.")

            current_run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            logger.info(f"Step 1.6: Saving processed feature data to database with run_id: {current_run_id}...")
            save_processed_features_to_db(
                db_config,
                processed_data_np,
                targets_np,
                feature_columns_list,
                final_tickers_list,
                current_run_id
            )
            logger.info(f"Step 1.6: Successfully saved processed feature data for run_id: {current_run_id}.")
            logger.info("--- Finished Mode: Full Data Processing for Training ---")
            return current_run_id

        else:
            logger.error(f"Invalid mode specified: {mode}. Choose 'incremental_fetch' or 'full_process'.")
            return None

    except Exception as e:
        logger.error(f"Error in run_processing (mode: {mode}): {e}", exc_info=True)
        if mode == 'full_process':
            return None
        raise

# ------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data ingestion and processing script for stock prediction.")
    parser.add_argument(
        '--config',
        type=str,
        default='config/params.yaml', 
        help='Path to the configuration file (e.g., config/params.yaml)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['incremental_fetch', 'full_process'],
        default='full_process', 
        help="Operation mode: 'incremental_fetch' for daily raw data updates, "
             "'full_process' for generating a complete training dataset with a new run_id."
    )
    args = parser.parse_args()
    config_path_arg = args.config
    mode_arg = args.mode

    # Resolve config path to be absolute for robustness
    config_path_resolved = Path(config_path_arg)
    if not config_path_resolved.is_absolute():
        # Try to resolve relative to the script's directory or project root
        # This logic might need adjustment based on how/where you run the script
        if (Path.cwd() / config_path_resolved).exists():
            config_path_resolved = (Path.cwd() / config_path_resolved).resolve()
        elif (Path(__file__).parent.parent.parent / config_path_resolved).exists(): # Assuming script is in src/data/
            config_path_resolved = (Path(__file__).parent.parent.parent / config_path_resolved).resolve()
        else:
            logger.error(f"Configuration file not found: {config_path_arg} (resolved to {config_path_resolved})")
            sys.exit(1)
    
    if not config_path_resolved.exists():
        logger.error(f"Configuration file not found: {config_path_resolved}")
        sys.exit(1)

    logger.info(f"Starting data processing script with resolved config: {config_path_resolved} in mode: {mode_arg}")

    try:
        with open(config_path_resolved, 'r') as f:
            config_params = yaml.safe_load(f)
            if 'database' not in config_params:
                logger.error("Database configuration missing from params.yaml")
                sys.exit(1)
            required_db_fields = ['dbname', 'user', 'password', 'host', 'port']
            missing_fields = [field for field in required_db_fields if field not in config_params['database']]
            if missing_fields:
                logger.error(f"Missing required database configuration fields: {missing_fields}")
                sys.exit(1)
        
        logger.info(f"Using PostgreSQL database at {config_params['database']['host']}:{config_params['database']['port']}")

        returned_value = run_processing(str(config_path_resolved), mode=mode_arg)
        
        if mode_arg == 'full_process':
            if returned_value: 
                logger.info(f"Full processing completed. Dataset run_id: {returned_value}")
                print(f"RUN_ID:{returned_value}") 
            else:
                logger.error("Full processing mode did not return a run_id. Check logs for errors.")
                sys.exit(1) 
        else: 
            logger.info(f"Incremental fetch mode completed.")
            
        logger.info(f"Data processing script (mode: {mode_arg}) finished successfully.")
        
    except yaml.YAMLError as e_yaml:
        logger.error(f"Error parsing configuration file {config_path_resolved}: {e_yaml}", exc_info=True)
        sys.exit(1)
    except Exception as e_main_script:
        logger.error(f"Fatal error in data processing script (mode: {mode_arg}): {e_main_script}", exc_info=True)
        sys.exit(1)
