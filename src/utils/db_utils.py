import psycopg2
import psycopg2.extras
import logging
import pickle
import io
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def get_db_connection(db_config: dict) -> psycopg2.extensions.connection:
    """Create a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        return conn
    except Exception as e:
        logger.error(f"Error connecting to PostgreSQL database: {e}")
        raise

def setup_database(db_config: dict) -> None:
    """Create PostgreSQL database tables if they don't exist."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        # Raw stock data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_stock_data (
            id SERIAL PRIMARY KEY,
            ticker TEXT,
            date TIMESTAMP,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            dividends REAL,
            stock_splits REAL,
            fetch_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        )
        ''')
        
        # Create index for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_raw_ticker_date ON raw_stock_data (ticker, date)')
        
        # Processed feature data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_feature_data (
            run_id TEXT PRIMARY KEY,
            processed_data BYTEA,
            targets BYTEA,
            feature_columns_json TEXT,
            tickers_json TEXT,
            processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Scaled feature sets
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scaled_feature_sets (
            run_id TEXT,
            set_name TEXT,
            data_blob BYTEA,
            PRIMARY KEY (run_id, set_name)
        )
        ''')

        # Scalers
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS scalers (
            run_id TEXT PRIMARY KEY,
            scaler_blob BYTEA,
            scaling_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Latest predictions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS latest_predictions (
            prediction_timestamp TIMESTAMP,
            ticker TEXT NOT NULL,
            predicted_price REAL,
            model_run_id TEXT,
            PRIMARY KEY (ticker)
        )
        ''')

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS optimization_results (
            run_id TEXT PRIMARY KEY,
            best_params_json TEXT,
            optimization_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info("Database tables initialized successfully")
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise

def save_to_raw_table(ticker: str, data: pd.DataFrame, db_config: dict) -> int:
    """Save raw ticker data to database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        records_added = 0
        for idx, row in data.iterrows():
            date_str = idx.strftime('%Y-%m-%d %H:%M:%S')
            try:
                cursor.execute('''
                INSERT INTO raw_stock_data 
                (ticker, date, open, high, low, close, volume, dividends, stock_splits)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    dividends = EXCLUDED.dividends,
                    stock_splits = EXCLUDED.stock_splits
                ''', (
                    ticker, date_str, row['Open'], row['High'], row['Low'], 
                    row['Close'], row['Volume'], row['Dividends'], row['Stock Splits']
                ))
                records_added += 1
            except Exception as e:
                logger.error(f"Error inserting record for {ticker} on {date_str}: {e}")
                
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully added {records_added} records for {ticker} to database")
        return records_added
        
    except Exception as e:
        logger.error(f"Error saving data to raw table: {e}")
        raise


def serialize_numpy(array: np.ndarray) -> bytes:
    """Serialize numpy array to bytes using pickle."""
    return pickle.dumps(array)

def check_ticker_exists(ticker: str, db_config: dict) -> bool:
    """Check if ticker data already exists in the database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM raw_stock_data WHERE ticker = %s", (ticker,))
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return count > 0
    except Exception as e:
        logger.error(f"Error checking if ticker exists: {e}", exc_info=True)
        return False

def load_data_from_db(db_config: dict, tickers_list: list) -> dict:
    """Load ticker data from PostgreSQL database into DataFrames."""
    try:
        conn = get_db_connection(db_config)
        ticker_data = {}
        
        for t in tickers_list:
            query = """
                SELECT date, open, high, low, close, volume, dividends, stock_splits 
                FROM raw_stock_data 
                WHERE ticker = %s 
                ORDER BY date
            """
            df = pd.read_sql_query(query, conn, params=(t,), parse_dates=['date'])
            
            if df.empty:
                logger.warning(f"No data found in database for {t}")
                continue
                
            df.set_index('date', inplace=True)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            
            if not df.empty:
                logger.info(f"Loaded {len(df)} records for {t} from {df.index.min()} to {df.index.max()}")
                ticker_data[t] = df
        
        if not ticker_data:
            logger.error("No ticker data was loaded from the database!")
            raise ValueError("No ticker data loaded from database")
            
        if len(ticker_data) < len(tickers_list):
            missing = set(tickers_list) - set(ticker_data.keys())
            logger.warning(f"Some tickers were not loaded from database: {missing}")
        
        conn.close()
        return ticker_data
        
    except Exception as e:
        logger.error(f"Error loading data from database: {e}", exc_info=True)
        raise

def save_processed_features_to_db(db_config: dict, processed_data: np.ndarray, 
                                targets: np.ndarray, feature_columns: list, 
                                tickers: list, run_id: Optional[str] = None) -> str:
    """Save processed feature data to PostgreSQL database."""
    try:
        import json
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        processed_data_blob = serialize_numpy(processed_data)
        targets_blob = serialize_numpy(targets)
        feature_columns_json = json.dumps(feature_columns)
        tickers_json = json.dumps(tickers)
        
        cursor.execute('''
        INSERT INTO processed_feature_data 
        (run_id, processed_data, targets, feature_columns_json, tickers_json)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            processed_data = EXCLUDED.processed_data,
            targets = EXCLUDED.targets,
            feature_columns_json = EXCLUDED.feature_columns_json,
            tickers_json = EXCLUDED.tickers_json
        ''', (run_id, processed_data_blob, targets_blob, feature_columns_json, tickers_json))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Successfully saved processed feature data with run_id: {run_id}")
        return run_id
        
    except Exception as e:
        logger.error(f"Error saving processed feature data: {e}", exc_info=True)
        raise

def load_processed_features_from_db(db_config: dict, run_id: Optional[str] = None) -> Optional[dict]:
    """Load processed feature data from PostgreSQL database."""
    try:
        import json
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        if run_id is None:
            cursor.execute("""
                SELECT run_id, processed_data, targets, feature_columns_json, tickers_json 
                FROM processed_feature_data 
                ORDER BY processing_timestamp DESC LIMIT 1
            """)
        else:
            cursor.execute("""
                SELECT run_id, processed_data, targets, feature_columns_json, tickers_json 
                FROM processed_feature_data 
                WHERE run_id = %s
            """, (run_id,))
            
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result is None:
            logger.warning(f"No processed feature data found{' for run_id: ' + run_id if run_id else ''}")
            return None
            
        run_id, processed_data_blob, targets_blob, feature_columns_json, tickers_json = result
        
        return {
            'run_id': run_id,
            'processed_data': pickle.loads(processed_data_blob),
            'targets': pickle.loads(targets_blob),
            'feature_columns': json.loads(feature_columns_json),
            'tickers': json.loads(tickers_json)
        }
        
    except Exception as e:
        logger.error(f"Error loading processed feature data: {e}", exc_info=True)
        raise

def save_scaled_features(db_config: dict, run_id: str, set_name: str, data: np.ndarray) -> bool:
    """Save scaled feature set to PostgreSQL database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        data_blob = serialize_numpy(data)
        
        cursor.execute('''
        INSERT INTO scaled_feature_sets (run_id, set_name, data_blob)
        VALUES (%s, %s, %s)
        ON CONFLICT (run_id, set_name) DO UPDATE SET
            data_blob = EXCLUDED.data_blob
        ''', (run_id, set_name, data_blob))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Saved scaled feature set '{set_name}' for run_id: {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving scaled feature set: {e}", exc_info=True)
        raise

def load_scaled_features(db_config: dict, run_id: str, set_name: str) -> Optional[np.ndarray]:
    """Load scaled feature set from PostgreSQL database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT data_blob FROM scaled_feature_sets 
            WHERE run_id = %s AND set_name = %s
        """, (run_id, set_name))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result is None:
            logger.warning(f"No scaled feature set '{set_name}' found for run_id: {run_id}")
            return None
            
        return pickle.loads(result[0])
        
    except Exception as e:
        logger.error(f"Error loading scaled feature set: {e}", exc_info=True)
        raise

def save_scalers(db_config: dict, run_id: str, scalers_dict: dict) -> bool:
    """Save scalers to PostgreSQL database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        scaler_blob = pickle.dumps(scalers_dict)
        
        cursor.execute('''
        INSERT INTO scalers (run_id, scaler_blob)
        VALUES (%s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            scaler_blob = EXCLUDED.scaler_blob
        ''', (run_id, scaler_blob))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Saved scalers for run_id: {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving scalers: {e}", exc_info=True)
        raise

def load_scalers(db_config: dict, run_id: str) -> Optional[dict]:
    """Load scalers from PostgreSQL database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        cursor.execute("SELECT scaler_blob FROM scalers WHERE run_id = %s", (run_id,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result is None:
            logger.warning(f"No scalers found for run_id: {run_id}")
            return None
            
        return pickle.loads(result[0])
        
    except Exception as e:
        logger.error(f"Error loading scalers: {e}", exc_info=True)
        raise

def save_optimization_results(db_config: dict, run_id: str, best_params: dict) -> bool:
    """Save optimization results to PostgreSQL database."""
    try:
        import json
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        best_params_json = json.dumps(best_params)
        
        cursor.execute('''
        INSERT INTO optimization_results (run_id, best_params_json)
        VALUES (%s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            best_params_json = EXCLUDED.best_params_json
        ''', (run_id, best_params_json))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Saved optimization results for run_id: {run_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving optimization results: {e}", exc_info=True)
        raise

def load_optimization_results(db_config: dict, run_id: str) -> Optional[dict]:
    """Load optimization results from PostgreSQL database."""
    try:
        import json
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        cursor.execute("SELECT best_params_json FROM optimization_results WHERE run_id = %s", (run_id,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if result is None:
            logger.warning(f"No optimization results found for run_id: {run_id}")
            return None
            
        return json.loads(result[0])
        
    except Exception as e:
        logger.error(f"Error loading optimization results: {e}", exc_info=True)
        raise

def save_prediction(db_config: dict, ticker: str, predicted_price: float, model_run_id: Optional[str] = None) -> bool:
    """Save latest prediction for a ticker to PostgreSQL database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO latest_predictions 
        (prediction_timestamp, ticker, predicted_price, model_run_id)
        VALUES (CURRENT_TIMESTAMP, %s, %s, %s)
        ON CONFLICT (ticker) DO UPDATE SET
            prediction_timestamp = CURRENT_TIMESTAMP,
            predicted_price = EXCLUDED.predicted_price,
            model_run_id = EXCLUDED.model_run_id
        ''', (ticker, predicted_price, model_run_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"Saved prediction for {ticker}: {predicted_price}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving prediction: {e}", exc_info=True)
        raise

def get_latest_predictions(db_config: dict, tickers: Optional[list] = None) -> dict:
    """Get latest predictions from PostgreSQL database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        
        if tickers is None:
            cursor.execute("""
                SELECT prediction_timestamp, ticker, predicted_price, model_run_id 
                FROM latest_predictions
                ORDER BY ticker
            """)
        else:
            cursor.execute("""
                SELECT prediction_timestamp, ticker, predicted_price, model_run_id 
                FROM latest_predictions
                WHERE ticker = ANY(%s)
                ORDER BY ticker
            """, (tickers,))
            
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        predictions = {
            row[1]: {
                'timestamp': row[0],
                'predicted_price': row[2],
                'model_run_id': row[3]
            }
            for row in results
        }
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error getting latest predictions: {e}", exc_info=True)
        raise