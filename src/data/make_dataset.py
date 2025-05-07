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
from datetime import datetime
from utils.db_utils import (
    setup_database,
    check_ticker_exists,
    save_to_raw_table,
    load_data_from_db,
    save_processed_data_to_db,
    save_processed_features_to_db,
    get_db_connection
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler() # Defaults to stderr
    handler.setLevel(logging.INFO) # <--- ADD THIS LINE
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_data(tickers_list, period, interval, fetch_delay, db_config):
    """Loads data for tickers and saves to PostgreSQL database."""
    try:
        # Initialize database
        setup_database(db_config)
        logger.info("Initializing PostgreSQL database connection")

        for t in tickers_list:
            # Check if ticker data already exists
            if check_ticker_exists(t, db_config):
                logger.info(f"Skipping download for {t}, data exists in database")
                continue

            try:
                logger.info(f"Downloading data for {t}")
                data = yf.Ticker(t).history(period=period, interval=interval)
                
                if data.empty:
                    logger.warning(f"No data available for {t}")
                    continue
                    
                # Log the data shape for debugging
                logger.info(f"Downloaded {len(data)} records for {t}")
                
                # Save to database
                save_to_raw_table(t, data, db_config)
                time.sleep(fetch_delay)
                
            except Exception as e:
                logger.error(f"Error loading {t}: {e}", exc_info=True)
                continue

    except Exception as e:
        logger.error(f"Error in load_data function: {e}", exc_info=True)
        raise

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
            df = aligned_data[ticker][feature_columns + ['Target']].fillna(method='ffill').fillna(method='bfill')
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


def run_processing(config_path: str, mode: str = 'full_process') -> Optional[str]:

    """Main function to run all data processing steps."""
    try:
        with open(config_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")

        # Get database configuration
        db_config = params['database']
        logger.info(f"Using database configuration: host={db_config['host']}, dbname={db_config['dbname']}")

        # Log key configuration parameters
        logger.info(f"Processing with parameters: tickers={params['data_loading']['tickers']}, "
                   f"period={params['data_loading']['period']}, "
                   f"interval={params['data_loading']['interval']}")

        tickers_list = params['data_loading']['tickers']
        period = params['data_loading']['period']
        interval = params['data_loading']['interval']
        fetch_delay = params['data_loading']['fetch_delay']
        corr_threshold = params['feature_engineering']['correlation_threshold']

        # 1. Load Raw Data
        logger.info("--- Starting Data Loading ---")
        load_data(tickers_list, period, interval, fetch_delay, db_config)
        logger.info("--- Finished Data Loading ---")

        # 2. Load data from database
        logger.info("--- Loading Data from Database ---")
        ticker_data = load_data_from_db(db_config, tickers_list)
        logger.info(f"Loaded data for {len(ticker_data)} tickers from database")

        # ...rest of the function remains the same until saving...

        # 6. Save Processed Data to database
        logger.info(f"--- Saving Processed Data to database ---")
        try:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_processed_features_to_db(
                db_config,
                processed_data,
                targets,
                feature_columns,
                final_tickers,
                run_id
            )
            logger.info(f"Successfully saved processed data to database with run_id: {run_id}")
        except Exception as e:
            logger.error(f"Failed to save processed data to database", exc_info=True)
            raise
        logger.info("--- Finished Saving Processed Data ---")

    except Exception as e:
        logger.error(f"Error in run_processing: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    # try:
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--config', type=str, 
    #                       required=False,
    #                       help='Path to the configuration file (params.yaml)')

    #     # Check if arguments were passed
    #     if len(sys.argv) == 1:
    #         # No arguments, use default config path
    #         config_path = 'params.yaml'
    #         logger.info(f"No config specified, using default: {config_path}")
    #     else:
    #         # Parse arguments normally
    #         args = parser.parse_args()
    #         config_path = args.config

    #     # Verify config file exists
    #     if not os.path.exists(config_path):
    #         logger.error(f"Configuration file not found: {config_path}")
    #         sys.exit(1)

    #     # Verify database configuration exists
    #     with open(config_path, 'r') as f:
    #         config = yaml.safe_load(f)
    #         if 'database' not in config:
    #             logger.error("Database configuration missing from params.yaml")
    #             sys.exit(1)
    #         required_db_fields = ['dbname', 'user', 'password', 'host', 'port']
    #         missing_fields = [field for field in required_db_fields if field not in config['database']]
    #         if missing_fields:
    #             logger.error(f"Missing required database configuration fields: {missing_fields}")
    #             sys.exit(1)

    #     logger.info(f"Starting data processing script with config: {config_path}")
    #     logger.info(f"Using PostgreSQL database at {config['database']['host']}:{config['database']['port']}")

    #     run_processing(config_path)
    #     logger.info("Data processing script finished successfully.")
        
    # except yaml.YAMLError as e:
    #     logger.error(f"Error parsing configuration file: {e}", exc_info=True)
    #     sys.exit(1)
    # except Exception as e:
    #     logger.error(f"Fatal error in data processing script: {e}", exc_info=True)
    #     sys.exit(1)

    # -----------------------
    try:
        parser = argparse.ArgumentParser(description="Data processing script for stock prediction.")
        parser.add_argument(
            '--config',
            type=str,
            default='config/params.yaml', # Default path relative to project root
            help='Path to the configuration file (params.yaml)'
        )
        parser.add_argument(
            '--mode',
            type=str,
            choices=['incremental_fetch', 'full_process'],
            default='full_process', # Default to full processing if not specified
            help='Operation mode: "incremental_fetch" for daily raw data, "full_process" for complete training dataset generation.'
        )
        args = parser.parse_args()
        config_path = args.config
        mode = args.mode

        # Verify config file exists
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)

        # Verify database configuration exists
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if 'database' not in config:
                logger.error("Database configuration missing from params.yaml")
                sys.exit(1)
            required_db_fields = ['dbname', 'user', 'password', 'host', 'port']
            missing_fields = [field for field in required_db_fields if field not in config['database']]
            if missing_fields:
                logger.error(f"Missing required database configuration fields: {missing_fields}")
                sys.exit(1)

        logger.info(f"Starting data processing script with config: {config_path} in mode: {mode}")
        
        # Call run_processing with the mode
        # run_processing needs to be adapted to handle the mode and return run_id for 'full_process'
        if mode == 'full_process':
            # This function will now need to return the run_id
            generated_run_id = run_processing(config_path, mode=mode)
            if generated_run_id:
                print(f"Full processing completed. Dataset run_id: {generated_run_id}")
            else:
                logger.error("Full processing did not return a run_id.")
                sys.exit(1)
        else: # incremental_fetch
            run_processing(config_path, mode=mode)
        
        logger.info(f"Data processing script (mode: {mode}) finished successfully.")
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in data processing script: {e}", exc_info=True)
        sys.exit(1)