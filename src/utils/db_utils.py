"""Database utilities for stock price prediction operations."""

# Standard library imports
import io
import logging
import os
import pickle
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# Configure logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_db_connection(db_config: dict) -> psycopg2.extensions.connection:
    """Create a PostgreSQL database connection."""
    try:
        conn = psycopg2.connect(
            dbname=db_config["dbname"],
            user=db_config["user"],
            password=db_config["password"],
            host=db_config["host"],
            port=db_config["port"],
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
        cursor.execute(
            """
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
        """
        )

        # Create index for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_raw_ticker_date ON raw_stock_data (ticker, date)"
        )

        # Processed feature data
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS processed_feature_data (
            run_id TEXT PRIMARY KEY,
            processed_data BYTEA,
            targets BYTEA,
            feature_columns_json TEXT,
            tickers_json TEXT,
            processing_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        # Scaled feature sets
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS scaled_feature_sets (
            run_id TEXT,
            set_name TEXT,
            data_blob BYTEA,
            PRIMARY KEY (run_id, set_name)
        )
        """
        )

        # Scalers
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS scalers (
            run_id TEXT PRIMARY KEY,
            scaler_blob BYTEA,
            scaling_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        # Latest predictions table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS latest_predictions (
            target_prediction_date DATE NOT NULL, -- Date for which the prediction is made
            ticker TEXT NOT NULL,
            predicted_price REAL,
            model_mlflow_run_id TEXT,
            prediction_logged_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (target_prediction_date, ticker) -- Composite Primary Key
        )
        """
        )
        logger.info("Ensured 'latest_predictions' table schema is up-to-date.")

        # Add index for predictions
        cursor.execute(
            """
        CREATE INDEX IF NOT EXISTS idx_latest_predictions_ticker_target_date 
        ON latest_predictions (ticker, target_prediction_date DESC);
        """
        )

        # Optimization results
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS optimization_results (
            run_id TEXT PRIMARY KEY,
            best_params_json TEXT,
            optimization_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        )

        # Performance log table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS model_performance_log (
            log_id SERIAL PRIMARY KEY,
            prediction_date DATE NOT NULL,
            ticker TEXT NOT NULL,
            actual_price REAL,
            predicted_price REAL,
            mae REAL,
            rmse REAL,
            mape REAL,
            direction_accuracy REAL,
            model_mlflow_run_id TEXT,
            evaluation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (prediction_date, ticker, model_mlflow_run_id)
        )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_perf_log_date_ticker "
            "ON model_performance_log (prediction_date, ticker)"
        )

        conn.commit()
        cursor.close()
        conn.close()

        logger.info("Database tables initialized successfully")

    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise

    finally:
        if "conn" in locals() and conn:  # Ensure conn was defined
            if "cursor" in locals() and cursor:
                cursor.close()
            conn.close()


def save_to_raw_table(
    ticker_symbol: str, data_df: pd.DataFrame, db_config: dict
) -> int:
    """
    Save or update raw ticker data in the database using batch insertion.

    Converts NumPy types to Python native types before insertion.

    Args:
        ticker_symbol: Stock ticker symbol
        data_df: DataFrame containing stock data
        db_config: Database configuration dictionary

    Returns:
        int: Number of processed rows
    """
    conn = None
    cursor = None

    # Define the order of columns from the DataFrame to match the SQL query
    df_cols_ordered = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]

    # SQL for batch insert/update
    sql_upsert = """
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
            stock_splits = EXCLUDED.stock_splits;
    """
    rows_to_upsert = []
    for date_index, row in data_df.iterrows():
        # date_index is typically a pandas.Timestamp, which psycopg2 handles well
        current_row_values = [ticker_symbol, date_index]  # ticker, date

        for col_name in df_cols_ordered:
            value = row[col_name]
            if pd.isna(value):
                current_row_values.append(None)
            # isinstance(value, np.generic) catches most numpy scalar types
            elif isinstance(value, np.floating):  # np.float16, np.float32, np.float64
                current_row_values.append(float(value))
            elif isinstance(value, np.integer):  # np.int8, np.int16, np.int32, np.int64
                current_row_values.append(int(value))
            elif isinstance(value, (float, int)):  # Python native numeric types
                current_row_values.append(value)
            else:
                # For other types, log a warning and consider how to handle
                logger.warning(
                    f"Unexpected data type for {ticker_symbol} - {col_name} on "
                    f"{date_index}: {type(value)}, value: {value}. "
                    f"Attempting to cast to float, or inserting NULL."
                )
                try:
                    current_row_values.append(float(value))  # Try to cast
                except (ValueError, TypeError):
                    current_row_values.append(None)  # Fallback to NULL if cast fails

        rows_to_upsert.append(tuple(current_row_values))

    if not rows_to_upsert:
        logger.info(f"No data rows prepared for upsert for ticker {ticker_symbol}.")
        return 0

    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        # Use executemany for batch upsert
        cursor.executemany(sql_upsert, rows_to_upsert)

        num_processed_rows = len(rows_to_upsert)

        conn.commit()
        logger.info(
            f"Successfully upserted {num_processed_rows} data points for "
            f"{ticker_symbol} into database."
        )
        return num_processed_rows

    except psycopg2.Error as e_db:
        logger.error(
            f"Database error during batch upsert for {ticker_symbol}: {e_db}",
            exc_info=True,
        )
        if conn:
            conn.rollback()  # Rollback the entire batch on any error
        raise  # Re-raise to indicate failure to the calling function
    except Exception as e_general:
        logger.error(
            f"General error during batch upsert for {ticker_symbol}: {e_general}",
            exc_info=True,
        )
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def serialize_numpy(array: np.ndarray) -> bytes:
    """Serialize numpy array to bytes using pickle."""
    return pickle.dumps(array)


def check_ticker_exists(ticker: str, db_config: dict) -> bool:
    """Check if ticker data already exists in the database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM raw_stock_data WHERE ticker = %s", (ticker,)
        )
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
            df = pd.read_sql_query(query, conn, params=(t,), parse_dates=["date"])

            if df.empty:
                logger.warning(f"No data found in database for {t}")
                continue

            df.set_index("date", inplace=True)
            df.columns = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Dividends",
                "Stock Splits",
            ]

            if not df.empty:
                logger.info(
                    f"Loaded {len(df)} records for {t} from "
                    f"{df.index.min()} to {df.index.max()}"
                )
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


def save_processed_features_to_db(
    db_config: dict,
    processed_data: np.ndarray,
    targets: np.ndarray,
    feature_columns: list,
    tickers: list,
    run_id: Optional[str] = None,
) -> str:
    """Save processed feature data to PostgreSQL database."""
    try:
        import json

        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        processed_data_blob = serialize_numpy(processed_data)
        targets_blob = serialize_numpy(targets)
        feature_columns_json = json.dumps(feature_columns)
        tickers_json = json.dumps(tickers)

        cursor.execute(
            """
        INSERT INTO processed_feature_data 
        (run_id, processed_data, targets, feature_columns_json, tickers_json)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            processed_data = EXCLUDED.processed_data,
            targets = EXCLUDED.targets,
            feature_columns_json = EXCLUDED.feature_columns_json,
            tickers_json = EXCLUDED.tickers_json
        """,
            (
                run_id,
                processed_data_blob,
                targets_blob,
                feature_columns_json,
                tickers_json,
            ),
        )

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Successfully saved processed feature data with run_id: {run_id}")
        return run_id

    except Exception as e:
        logger.error(f"Error saving processed feature data: {e}", exc_info=True)
        raise


def load_processed_features_from_db(
    db_config: dict, run_id: Optional[str] = None
) -> Optional[dict]:
    """Load processed feature data from PostgreSQL database."""
    try:
        import json

        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        if run_id is None:
            cursor.execute(
                """
                SELECT run_id, processed_data, targets, feature_columns_json, tickers_json 
                FROM processed_feature_data 
                ORDER BY processing_timestamp DESC LIMIT 1
            """
            )
        else:
            cursor.execute(
                """
                SELECT run_id, processed_data, targets, feature_columns_json, tickers_json 
                FROM processed_feature_data 
                WHERE run_id = %s
            """,
                (run_id,),
            )

        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result is None:
            logger.warning(
                f"No processed feature data found"
                f"{' for run_id: ' + run_id if run_id else ''}"
            )
            return None

        (
            run_id,
            processed_data_blob,
            targets_blob,
            feature_columns_json,
            tickers_json,
        ) = result

        return {
            "run_id": run_id,
            "processed_data": pickle.loads(processed_data_blob),
            "targets": pickle.loads(targets_blob),
            "feature_columns": json.loads(feature_columns_json),
            "tickers": json.loads(tickers_json),
        }

    except Exception as e:
        logger.error(f"Error loading processed feature data: {e}", exc_info=True)
        raise


def save_scaled_features(
    db_config: dict, run_id: str, set_name: str, data: np.ndarray
) -> bool:
    """Save scaled feature set to PostgreSQL database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        data_blob = serialize_numpy(data)

        cursor.execute(
            """
        INSERT INTO scaled_feature_sets (run_id, set_name, data_blob)
        VALUES (%s, %s, %s)
        ON CONFLICT (run_id, set_name) DO UPDATE SET
            data_blob = EXCLUDED.data_blob
        """,
            (run_id, set_name, data_blob),
        )

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"Saved scaled feature set '{set_name}' for run_id: {run_id}")
        return True

    except Exception as e:
        logger.error(f"Error saving scaled feature set: {e}", exc_info=True)
        raise


def load_scaled_features(
    db_config: dict, run_id: str, set_name: str
) -> Optional[np.ndarray]:
    """Load scaled feature set from PostgreSQL database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT data_blob FROM scaled_feature_sets 
            WHERE run_id = %s AND set_name = %s
        """,
            (run_id, set_name),
        )

        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result is None:
            logger.warning(
                f"No scaled feature set '{set_name}' found for run_id: {run_id}"
            )
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

        cursor.execute(
            """
        INSERT INTO scalers (run_id, scaler_blob)
        VALUES (%s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            scaler_blob = EXCLUDED.scaler_blob
        """,
            (run_id, scaler_blob),
        )

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

        cursor.execute(
            """
        INSERT INTO optimization_results (run_id, best_params_json)
        VALUES (%s, %s)
        ON CONFLICT (run_id) DO UPDATE SET
            best_params_json = EXCLUDED.best_params_json
        """,
            (run_id, best_params_json),
        )

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

        cursor.execute(
            "SELECT best_params_json FROM optimization_results WHERE run_id = %s",
            (run_id,),
        )

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


def save_prediction(
    db_config: dict,
    ticker: str,
    predicted_price: float,
    model_mlflow_run_id: Optional[str],
    target_prediction_date_str: str,  # Expecting 'YYYY-MM-DD'
) -> bool:
    """
    Save a prediction for a ticker for a specific target date to PostgreSQL database.

    Updates if a prediction for that ticker and target_date already exists.

    Args:
        db_config: Database configuration dictionary
        ticker: Stock ticker symbol
        predicted_price: Predicted price value
        model_mlflow_run_id: MLflow run ID of the model used
        target_prediction_date_str: Date for which prediction is made ('YYYY-MM-DD')

    Returns:
        bool: True if successful
    """
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO latest_predictions 
        (target_prediction_date, ticker, predicted_price, model_mlflow_run_id, 
         prediction_logged_timestamp)
        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (target_prediction_date, ticker) DO UPDATE SET
            predicted_price = EXCLUDED.predicted_price,
            model_mlflow_run_id = EXCLUDED.model_mlflow_run_id,
            prediction_logged_timestamp = CURRENT_TIMESTAMP 
        """,
            (target_prediction_date_str, ticker, predicted_price, model_mlflow_run_id),
        )

        conn.commit()
        logger.info(
            f"Saved/Updated prediction for {ticker} "
            f"(Target Date: {target_prediction_date_str}): "
            f"{predicted_price:.4f} (Model Run: {model_mlflow_run_id})"
        )
        return True

    except Exception as e:
        logger.error(
            f"Error saving prediction for {ticker} "
            f"(Target Date: {target_prediction_date_str}): {e}",
            exc_info=True,
        )
        raise  # Or return False

    finally:
        if "conn" in locals() and conn:
            if "cursor" in locals() and cursor:
                cursor.close()
            conn.close()


def get_latest_predictions(db_config: dict, tickers: Optional[list] = None) -> dict:
    """Get latest predictions from PostgreSQL database."""
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        if tickers is None:
            cursor.execute(
                """
                SELECT prediction_timestamp, ticker, predicted_price, model_run_id 
                FROM latest_predictions
                ORDER BY ticker
            """
            )
        else:
            cursor.execute(
                """
                SELECT prediction_timestamp, ticker, predicted_price, model_run_id 
                FROM latest_predictions
                WHERE ticker = ANY(%s)
                ORDER BY ticker
            """,
                (tickers,),
            )

        results = cursor.fetchall()
        cursor.close()
        conn.close()

        predictions = {
            row[1]: {
                "timestamp": row[0],
                "predicted_price": row[2],
                "model_run_id": row[3],
            }
            for row in results
        }

        return predictions

    except Exception as e:
        logger.error(f"Error getting latest predictions: {e}", exc_info=True)
        raise


def get_prediction_for_date_ticker(
    db_config: dict, target_date_str: str, ticker: str  # 'YYYY-MM-DD'
) -> Optional[dict]:
    """
    Retrieve the predicted price and model_mlflow_run_id for a specific ticker and target date.

    Args:
        db_config: Database configuration dictionary
        target_date_str: Target prediction date string ('YYYY-MM-DD')
        ticker: Stock ticker symbol

    Returns:
        Optional[dict]: Dictionary with prediction info or None if not found
    """
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        cursor.execute(
            """
            SELECT predicted_price, model_mlflow_run_id
            FROM latest_predictions
            WHERE target_prediction_date = %s AND ticker = %s
        """,
            (target_date_str, ticker),
        )

        result = cursor.fetchone()

        if result:
            return dict(result)  # Convert DictRow to dict
        else:
            return None
    except Exception as e:
        logger.error(
            f"Error getting prediction for {ticker} on {target_date_str}: {e}",
            exc_info=True,
        )
        return None  # Or raise
    finally:
        if "conn" in locals() and conn:
            if "cursor" in locals() and cursor:
                cursor.close()
            conn.close()


def get_predictions_for_ticker_in_daterange(
    db_config: dict,
    ticker: str,
    start_date_str: str,  # 'YYYY-MM-DD'
    end_date_str: str,  # 'YYYY-MM-DD'
) -> pd.DataFrame:
    """
    Retrieve all predictions for a ticker within a specified date range.

    Args:
        db_config: Database configuration dictionary
        ticker: Stock ticker symbol
        start_date_str: Start date string ('YYYY-MM-DD')
        end_date_str: End date string ('YYYY-MM-DD')

    Returns:
        pd.DataFrame: DataFrame with prediction results
    """
    try:
        conn = get_db_connection(db_config)
        # Using pandas.read_sql for convenience
        query = """
            SELECT target_prediction_date, predicted_price
            FROM latest_predictions
            WHERE ticker = %s 
              AND target_prediction_date >= %s 
              AND target_prediction_date <= %s
            ORDER BY target_prediction_date ASC;
        """
        df = pd.read_sql_query(
            query, conn, params=(ticker.upper(), start_date_str, end_date_str)
        )
        # Ensure target_prediction_date is datetime object for merging
        if not df.empty:
            df["target_prediction_date"] = pd.to_datetime(df["target_prediction_date"])
        return df
    except Exception as e:
        logger.error(
            f"Error getting predictions for {ticker} in range "
            f"{start_date_str}-{end_date_str}: {e}",
            exc_info=True,
        )
        return pd.DataFrame()  # Return empty DataFrame on error
    finally:
        if "conn" in locals() and conn:
            conn.close()


def get_all_distinct_tickers_from_predictions(db_config: dict) -> list:
    """
    Retrieve a list of all unique ticker symbols from the latest_predictions table.

    Args:
        db_config: Database configuration dictionary

    Returns:
        list: List of unique ticker symbols
    """
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT DISTINCT ticker FROM latest_predictions ORDER BY ticker ASC;"
        )
        results = cursor.fetchall()

        return [row[0] for row in results]
    except Exception as e:
        logger.error(
            f"Error getting distinct tickers from predictions: {e}", exc_info=True
        )
        return []
    finally:
        if "conn" in locals() and conn:
            if "cursor" in locals() and cursor:  # Ensure cursor was defined
                cursor.close()
            conn.close()


def get_latest_prediction_for_all_tickers(db_config: dict) -> list:
    """
    Retrieve the most recent prediction for every ticker from the latest_predictions table.

    A prediction is defined as "latest" by the most recent target_prediction_date.

    Args:
        db_config: Database configuration dictionary

    Returns:
        list: List of dictionaries with latest prediction data for each ticker
    """
    try:
        conn = get_db_connection(db_config)
        # Use DictCursor to get results as dictionaries
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # Query to get the latest prediction for each ticker
        # Uses DISTINCT ON which is specific to PostgreSQL and efficient for this
        query = """
            SELECT DISTINCT ON (ticker) 
                   ticker, 
                   target_prediction_date, 
                   predicted_price, 
                   model_mlflow_run_id
            FROM latest_predictions
            ORDER BY ticker, target_prediction_date DESC;
        """
        cursor.execute(query)
        results = cursor.fetchall()

        predictions_list = []
        for row in results:
            predictions_list.append(
                {
                    "ticker": row["ticker"],
                    "predicted_price": row["predicted_price"],
                    # Format date to string if it's a date object from DB
                    "date": (
                        row["target_prediction_date"].isoformat()
                        if isinstance(row["target_prediction_date"], date)
                        else str(row["target_prediction_date"])
                    ),
                    "model_mlflow_run_id": row["model_mlflow_run_id"],
                }
            )
        return predictions_list

    except Exception as e:
        logger.error(
            f"Error getting latest predictions for all tickers from DB: {e}",
            exc_info=True,
        )
        return []  # Return empty list on error
    finally:
        if "conn" in locals() and conn:
            if "cursor" in locals() and cursor:
                cursor.close()
            conn.close()


def get_latest_target_date_prediction_for_ticker(
    db_config: dict, ticker: str
) -> Optional[dict]:
    """
    Retrieve the latest prediction for a specific ticker.

    Args:
        db_config: Database configuration dictionary
        ticker: Stock ticker symbol

    Returns:
        Optional[dict]: Latest prediction info or None if not found
    """
    conn = None
    cursor = None
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        query = """
            SELECT target_prediction_date, predicted_price, model_mlflow_run_id
            FROM latest_predictions
            WHERE ticker = %s
            ORDER BY target_prediction_date DESC
            LIMIT 1;
        """
        cursor.execute(query, (ticker.upper(),))
        result = cursor.fetchone()

        if result:
            # Log exactly what is being fetched
            logger.info(
                f"DB_UTILS: Fetched for {ticker}: "
                f"target_date={result['target_prediction_date']}, "
                f"raw_predicted_price={result['predicted_price']} "
                f"(type: {type(result['predicted_price'])})"
            )

            predicted_price_val = result["predicted_price"]
            final_predicted_price = None
            if predicted_price_val is not None:
                if isinstance(predicted_price_val, Decimal):
                    final_predicted_price = float(predicted_price_val)
                elif isinstance(predicted_price_val, (float, int)):
                    final_predicted_price = float(predicted_price_val)
                else:
                    try:
                        final_predicted_price = float(predicted_price_val)
                        logger.warning(
                            f"DB_UTILS: predicted_price for {ticker} was type "
                            f"{type(predicted_price_val)}, successfully cast to float."
                        )
                    except (ValueError, TypeError):
                        logger.error(
                            f"DB_UTILS: Could not convert predicted_price "
                            f"'{predicted_price_val}' (type: {type(predicted_price_val)}) "
                            f"to float for {ticker}. Setting to None."
                        )
                        final_predicted_price = None

            return {
                "target_prediction_date": (
                    result["target_prediction_date"].isoformat()
                    if isinstance(result["target_prediction_date"], date)
                    else str(result["target_prediction_date"])
                ),
                "predicted_price": final_predicted_price,  # Use the processed value
                "model_mlflow_run_id": result["model_mlflow_run_id"],
            }
        else:
            logger.info(f"DB_UTILS: No prediction found for {ticker}.")
            return None
    except Exception as e:
        logger.error(
            f"DB_UTILS: Error in get_latest_target_date_prediction_for_ticker "
            f"for {ticker}: {e}",
            exc_info=True,
        )
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_raw_stock_data_for_period(
    db_config: dict, ticker: str, end_date_obj: date, num_days: int
) -> pd.DataFrame:
    """
    Retrieve raw stock data for a ticker for a specified number of trading days.

    Args:
        db_config: Database configuration dictionary
        ticker: Stock ticker symbol
        end_date_obj: End date object
        num_days: Number of days to retrieve

    Returns:
        pd.DataFrame: DataFrame with raw stock data
    """
    conn = None
    try:
        conn = get_db_connection(db_config)
        query = """
            SELECT date, close
            FROM raw_stock_data
            WHERE ticker = %s AND date <= %s
            ORDER BY date DESC
            LIMIT %s;
        """
        df = pd.read_sql_query(
            query, conn, params=(ticker.upper(), end_date_obj, num_days)
        )

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])  # Ensure date is datetime
            df = df.sort_values(by="date", ascending=True).reset_index(drop=True)
            logger.info(
                f"Fetched {len(df)} raw_stock_data points for {ticker} "
                f"ending on/before {end_date_obj} for chart."
            )
        else:
            logger.warning(
                f"No raw_stock_data found for {ticker} for chart period "
                f"ending {end_date_obj}."
            )
        return df

    except Exception as e:
        logger.error(
            f"Error fetching raw stock data for {ticker} for chart: {e}", exc_info=True
        )
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


def save_daily_performance_metrics(
    db_config: dict,
    prediction_date: str,  # Expecting 'YYYY-MM-DD' string
    ticker: str,
    metrics_dict: Dict[str, float],
    model_mlflow_run_id: str,
) -> bool:
    """
    Save daily model performance metrics to the database.

    Args:
        db_config: Database configuration dictionary
        prediction_date: Prediction date string ('YYYY-MM-DD')
        ticker: Stock ticker symbol
        metrics_dict: Dictionary of performance metrics
        model_mlflow_run_id: MLflow run ID of the model

    Returns:
        bool: True if successful
    """
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO model_performance_log
        (prediction_date, ticker, actual_price, predicted_price, mae, rmse, 
         mape, direction_accuracy, model_mlflow_run_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (prediction_date, ticker, model_mlflow_run_id) DO UPDATE SET
            actual_price = EXCLUDED.actual_price,
            predicted_price = EXCLUDED.predicted_price,
            mae = EXCLUDED.mae,
            rmse = EXCLUDED.rmse,
            mape = EXCLUDED.mape,
            direction_accuracy = EXCLUDED.direction_accuracy,
            evaluation_timestamp = CURRENT_TIMESTAMP
        """,
            (
                prediction_date,
                ticker,
                metrics_dict.get("actual_price"),
                metrics_dict.get("predicted_price"),
                metrics_dict.get("mae"),
                metrics_dict.get("rmse"),
                metrics_dict.get("mape"),
                metrics_dict.get("direction_accuracy"),
                model_mlflow_run_id,
            ),
        )

        conn.commit()
        logger.info(
            f"Saved performance metrics for {ticker} on {prediction_date} "
            f"(Model Run ID: {model_mlflow_run_id})"
        )
        return True
    except Exception as e:
        logger.error(
            f"Error saving daily performance metrics for {ticker} on "
            f"{prediction_date}: {e}",
            exc_info=True,
        )
        raise
    finally:
        if "conn" in locals() and conn:
            cursor.close()
            conn.close()


def get_recent_performance_metrics(
    db_config: dict, ticker: str, days_lookback: int
) -> pd.DataFrame:
    """
    Retrieve recent performance metrics for a ticker.

    Args:
        db_config: Database configuration dictionary
        ticker: Stock ticker symbol
        days_lookback: Number of days to look back

    Returns:
        pd.DataFrame: DataFrame with performance metrics
    """
    try:
        conn = get_db_connection(db_config)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_lookback)

        query = """
            SELECT prediction_date, actual_price, predicted_price, mae, rmse, 
                   mape, direction_accuracy, model_mlflow_run_id
            FROM model_performance_log
            WHERE ticker = %s AND prediction_date >= %s AND prediction_date <= %s
            ORDER BY prediction_date DESC
        """
        df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))

        logger.info(
            f"Retrieved {len(df)} recent performance records for {ticker} "
            f"(last {days_lookback} days)"
        )
        return df
    except Exception as e:
        logger.error(
            f"Error getting recent performance metrics for {ticker}: {e}", exc_info=True
        )
        raise
    finally:
        if "conn" in locals() and conn:
            conn.close()


def get_last_data_timestamp_for_ticker(
    db_config: dict, ticker: str
) -> Optional[datetime]:
    """
    Get the timestamp of the latest data point for a ticker in raw_stock_data.

    Args:
        db_config: Database configuration dictionary
        ticker: Stock ticker symbol

    Returns:
        Optional[datetime]: Latest timestamp or None if not found
    """
    try:
        conn = get_db_connection(db_config)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT MAX(date) FROM raw_stock_data WHERE ticker = %s", (ticker,)
        )
        result = cursor.fetchone()

        if result and result[0]:
            logger.debug(f"Last data timestamp for {ticker}: {result[0]}")
            return result[
                0
            ]  # This will be a datetime object if 'date' column is TIMESTAMP
        else:
            logger.info(f"No data found for {ticker}, will fetch full history.")
            return None
    except Exception as e:
        logger.error(
            f"Error getting last data timestamp for {ticker}: {e}", exc_info=True
        )
        raise
    finally:
        if "conn" in locals() and conn:
            cursor.close()
            conn.close()


def get_latest_raw_data_window(
    db_config: dict, tickers_list: list, window_size_days: int
) -> dict:
    """
    Load the most recent data window for specified tickers.

    Args:
        db_config: Database configuration dictionary
        tickers_list: List of stock ticker symbols
        window_size_days: Number of days in the window

    Returns:
        dict: Dictionary of DataFrames {ticker: df}
    """
    try:
        conn = get_db_connection(db_config)
        ticker_data_window = {}

        for t in tickers_list:
            query = """
                SELECT date, open, high, low, close, volume, dividends, stock_splits 
                FROM raw_stock_data 
                WHERE ticker = %s 
                ORDER BY date DESC
                LIMIT %s 
            """
            df = pd.read_sql_query(
                query, conn, params=(t, window_size_days), parse_dates=["date"]
            )

            if df.empty:
                logger.warning(
                    f"No raw data found in DB for {t} to create latest window."
                )
                continue

            # The query already sorts DESC, so we need to re-sort ASC for time series processing
            df = df.sort_values(by="date", ascending=True).set_index("date")
            df.columns = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Dividends",
                "Stock Splits",
            ]

            if len(df) < window_size_days:
                logger.warning(
                    f"Fetched {len(df)} records for {t}, less than requested "
                    f"window {window_size_days}."
                )

            ticker_data_window[t] = df
            logger.info(
                f"Loaded latest {len(df)} records for {t} for prediction input."
            )

        if not ticker_data_window:
            logger.error("No data loaded for any ticker for the latest window.")
            # Depending on strictness, you might raise an error here

        return ticker_data_window
    except Exception as e:
        logger.error(f"Error loading latest raw data window: {e}", exc_info=True)
        raise
    finally:
        if "conn" in locals() and conn:
            conn.close()
