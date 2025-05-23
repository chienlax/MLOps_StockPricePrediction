"""FastAPI main module for stock prediction service."""

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Path Definitions & Config Loading ---
PROJECT_ROOT_API = Path(__file__).resolve().parents[2]
CONFIG_FILE_FOR_API = PROJECT_ROOT_API / "config/params.yaml"
TEMPLATES_DIR = PROJECT_ROOT_API / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"

DB_CONFIG = None
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [API] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    sys.path.insert(0, str(PROJECT_ROOT_API / "src"))
    from utils.db_utils import (
        get_all_distinct_tickers_from_predictions,
        get_latest_prediction_for_all_tickers,
        get_latest_target_date_prediction_for_ticker,
        get_raw_stock_data_for_period,
        setup_database,
    )

    with open(CONFIG_FILE_FOR_API, "r") as f_params:
        params_config = yaml.safe_load(f_params)
    DB_CONFIG = params_config.get("database")

    if not DB_CONFIG:
        logger.error(
            "CRITICAL API STARTUP: Database configuration not found in params.yaml."
        )
    else:
        setup_database(DB_CONFIG)  # Ensure DB is set up before API starts
        logger.info("Database setup completed successfully.")

except ImportError as e_imp:
    logger.error(
        f"CRITICAL API STARTUP: Could not import from utils.db_utils: {e_imp}",
        exc_info=True,
    )
    DB_CONFIG = None
except FileNotFoundError:
    logger.error(f"CRITICAL API STARTUP: Config file {CONFIG_FILE_FOR_API} not found.")
    DB_CONFIG = None
except Exception as e_cfg:
    logger.error(
        f"CRITICAL API STARTUP: Error loading params.yaml or DB config: {e_cfg}",
        exc_info=True,
    )
    DB_CONFIG = None

app = FastAPI(
    title="Stock Prediction API",
    description="API to provide latest stock predictions and historical context.",
    version="1.0.4",
)

if STATIC_DIR.exists() and STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.warning(
        f"Static directory {STATIC_DIR} not found. Static files might not be served."
    )

if TEMPLATES_DIR.exists() and TEMPLATES_DIR.is_dir():
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
else:
    logger.error(
        f"Templates directory {TEMPLATES_DIR} not found. HTML templates will not be served."
    )
    templates = None  # type: ignore


@app.get("/predictions/latest_for_table")
async def get_latest_predictions_for_table_from_db():
    """Get latest prediction data for display in a table format."""
    if not DB_CONFIG:
        raise HTTPException(status_code=503, detail="Database service unavailable.")
    try:
        predictions = get_latest_prediction_for_all_tickers(DB_CONFIG)
        return {"status": "success", "data": predictions or []}
    except Exception as e:
        logger.error(
            f"API Error fetching latest predictions for table: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Error fetching latest predictions: {str(e)}"
        )


@app.get("/tickers")
async def get_available_tickers():
    """Get all available tickers that have predictions."""
    if not DB_CONFIG:
        raise HTTPException(status_code=503, detail="Database service unavailable.")
    try:
        tickers = get_all_distinct_tickers_from_predictions(DB_CONFIG)
        return {"status": "success", "tickers": tickers}
    except Exception as e:
        logger.error(f"API Error fetching distinct tickers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching tickers: {str(e)}")


@app.get("/historical_context_chart/{ticker}")
async def get_historical_context_for_prediction(ticker: str):
    """Get historical data and prediction for a specific ticker."""
    if not DB_CONFIG:
        logger.error("API /historical_context_chart: DB_CONFIG is not available.")
        raise HTTPException(status_code=503, detail="Database service unavailable.")

    ticker_upper = ticker.upper()
    logger.info(f"Fetching data for /historical_context_chart/{ticker_upper}")

    latest_pred_info = get_latest_target_date_prediction_for_ticker(
        DB_CONFIG, ticker_upper
    )
    logger.info(f"Prediction info from DB for {ticker_upper}: {latest_pred_info}")

    historical_actual_dates_list = []
    historical_actual_prices_list = []
    prediction_ref_date_str = None
    predicted_price_for_ref = None

    if latest_pred_info and latest_pred_info.get("target_prediction_date"):
        prediction_ref_date_str = latest_pred_info["target_prediction_date"]
        predicted_price_for_ref = latest_pred_info.get("predicted_price")
        logger.info(
            f"Predicted price for ref date {prediction_ref_date_str} for {ticker_upper}: "
            f"{predicted_price_for_ref} (type: {type(predicted_price_for_ref)})"
        )

        latest_target_prediction_date_obj = date.fromisoformat(prediction_ref_date_str)
        history_end_date_for_db_query = latest_target_prediction_date_obj - timedelta(
            days=1
        )
        num_historical_days = 14

        logger.info(
            f"Querying raw_stock_data for {ticker_upper} up to "
            f"{history_end_date_for_db_query} for {num_historical_days} days."
        )
        df_hist_from_db = get_raw_stock_data_for_period(
            DB_CONFIG, ticker_upper, history_end_date_for_db_query, num_historical_days
        )

        if not df_hist_from_db.empty:
            for index, row in df_hist_from_db.iterrows():
                dt_obj = row["date"]
                price_val = row["close"]
                if pd.notnull(dt_obj) and pd.notnull(price_val):
                    historical_actual_dates_list.append(dt_obj.strftime("%Y-%m-%d"))
                    historical_actual_prices_list.append(float(price_val))
            logger.info(
                f"Found {len(historical_actual_dates_list)} historical points "
                f"for {ticker_upper} from DB for chart."
            )
        else:
            logger.warning(
                f"No historical data found in raw_stock_data for {ticker_upper} for the chart."
            )
    else:
        logger.warning(
            f"No prediction reference date found in DB for ticker {ticker_upper}."
        )

    return_data = {
        "status": "success",
        "ticker": ticker_upper,
        "historical_dates": historical_actual_dates_list,
        "historical_actuals": historical_actual_prices_list,
        "prediction_reference_date": prediction_ref_date_str,
        "predicted_price_for_ref_date": predicted_price_for_ref,
    }
    logger.info(f"Returning data for {ticker_upper} for chart/display: {return_data}")
    return return_data


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Render the main dashboard HTML page."""
    if not templates:
        raise HTTPException(
            status_code=500, detail="Server configuration error: Templates not found."
        )
    return templates.TemplateResponse("index.html", {"request": request})
