# from fastapi import FastAPI, HTTPException, Request
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import HTMLResponse
# from pathlib import Path
# import json
# from datetime import date

# app = FastAPI(
#     title="Stock Prediction API",
#     description="API để cung cấp dự báo cổ phiếu mới nhất và theo ticker",
#     version="1.0.0"
# )

# # Các đường dẫn dữ liệu
# BASE_DATA_PATH_INSIDE_CONTAINER = Path("/opt/airflow/data") # Define base path based on volume mount
# PREDICTION_FILE_PATH = BASE_DATA_PATH_INSIDE_CONTAINER / "predictions/latest_predictions.json"  # Path("data/predictions/latest_predictions.json")
# HISTORICAL_DIR = BASE_DATA_PATH_INSIDE_CONTAINER / "predictions/historical"  # Path("data/predictions/historical")

# # Mount static
# # app.mount("/static", StaticFiles(directory="templates/static"), name="static")
# app.mount("/static", StaticFiles(directory="/opt/airflow/templates/static"), name="static")

# # Cấu hình Jinja2
# templates = Jinja2Templates(directory="templates")

# # ------------------ TẢI DỮ LIỆU ------------------

# def load_predictions():
#     """
#     Đọc file latest_predictions.json và trả về danh sách dự đoán dạng:
#     [{"ticker": str, "predicted_price": float, "date": str}, ...]
#     """
#     if not PREDICTION_FILE_PATH.exists():
#         raise HTTPException(status_code=404, detail="Prediction file not found")
#     try:
#         with open(PREDICTION_FILE_PATH, "r") as f:
#             data = json.load(f)

#         if isinstance(data, dict) and "predictions" in data:
#             date_str = data.get("date", date.today().isoformat())
#             predictions_dict = data["predictions"]
#             predictions_list = [
#                 {"ticker": ticker, "predicted_price": price, "date": date_str}
#                 for ticker, price in predictions_dict.items()
#             ]
#             return predictions_list

#         raise ValueError("Invalid format in latest_predictions.json")
#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail="Error parsing prediction file")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# def load_historical_predictions_for_ticker(ticker: str):
#     """
#     Duyệt tất cả các file trong data/predictions/historical/ và trả về
#     các dự đoán cho ticker tương ứng (theo thời gian).
#     """
#     if not HISTORICAL_DIR.exists():
#         raise HTTPException(status_code=404, detail="Historical directory not found")

#     results = []
#     for file in sorted(HISTORICAL_DIR.glob("*.json")):
#         try:
#             with open(file, "r") as f:
#                 data = json.load(f)
#                 date_str = data.get("date", file.stem)
#                 predictions = data.get("predictions", {})
#                 if ticker.upper() in predictions:
#                     results.append({
#                         "ticker": ticker.upper(),
#                         "predicted_price": predictions[ticker.upper()],
#                         "date": date_str
#                     })
#         except Exception as e:
#             print(f"Warning: Error reading {file.name}: {e}")
#             continue

#     return results

# # ------------------ API ------------------

# @app.get("/predictions/latest")
# async def get_latest_predictions():
#     """
#     API trả về các dự đoán mới nhất từ latest_predictions.json
#     """
#     predictions = load_predictions()
#     return {"status": "success", "data": predictions}

# @app.get("/predictions/historical/{ticker}")
# async def get_historical_predictions(ticker: str):
#     """
#     API trả về lịch sử dự đoán cho 1 ticker cụ thể từ nhiều ngày.
#     """
#     predictions = load_historical_predictions_for_ticker(ticker)
#     if not predictions:
#         raise HTTPException(status_code=404, detail=f"No historical predictions found for ticker: {ticker}")
#     return {"status": "success", "ticker": ticker.upper(), "data": predictions}

# @app.get("/", response_class=HTMLResponse)
# async def get_dashboard(request: Request):
#     """
#     Giao diện chính Dashboard
#     """
#     return templates.TemplateResponse("index.html", {"request": request})
# #uvicorn src.api.main:app --reload


from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import yfinance as yf
import yaml 
import sys
import numpy as np # For sanitize_float_list if kept for other parts
import logging

# --- Path Definitions & Config Loading ---
PROJECT_ROOT_API = Path(__file__).resolve().parents[2] 
CONFIG_FILE_FOR_API = PROJECT_ROOT_API / "config/params.yaml"
TEMPLATES_DIR = PROJECT_ROOT_API / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"

DB_CONFIG = None
logger = logging.getLogger(__name__) # Define logger early

try:
    # Ensure 'src' is in PYTHONPATH
    sys.path.insert(0, str(PROJECT_ROOT_API / "src"))
    from utils.db_utils import (
        get_latest_prediction_for_all_tickers,
        get_all_distinct_tickers_from_predictions,
        get_latest_target_date_prediction_for_ticker # Needed for the new chart
        # get_predictions_for_ticker_in_daterange # Might not be needed if old chart endpoint is removed
    )
    with open(CONFIG_FILE_FOR_API, 'r') as f_params:
        params_config = yaml.safe_load(f_params)
    DB_CONFIG = params_config.get('database')
    if not DB_CONFIG:
        logger.error("CRITICAL API STARTUP: Database configuration not found in params.yaml.")
except ImportError as e_imp:
    logger.error(f"CRITICAL API STARTUP: Could not import db_utils: {e_imp}", exc_info=True)
except FileNotFoundError:
    logger.error(f"CRITICAL API STARTUP: Config file {CONFIG_FILE_FOR_API} not found.")
except Exception as e_cfg:
    logger.error(f"CRITICAL API STARTUP: Error loading params.yaml or DB config: {e_cfg}", exc_info=True)

app = FastAPI(
    title="Stock Prediction API",
    description="API để cung cấp dự báo cổ phiếu mới nhất và theo ticker",
    version="1.0.0"
)

if STATIC_DIR.exists() and STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.warning(f"Static directory {STATIC_DIR} not found.")

if TEMPLATES_DIR.exists() and TEMPLATES_DIR.is_dir():
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
else:
    logger.error(f"Templates directory {TEMPLATES_DIR} not found.")
    templates = None


# --- API Endpoints ---

@app.get("/predictions/latest_for_table")
async def get_latest_predictions_for_table_from_db():
    if not DB_CONFIG: raise HTTPException(status_code=500, detail="DB config error.")
    try:
        predictions = get_latest_prediction_for_all_tickers(DB_CONFIG)
        return {"status": "success", "data": predictions or []}
    except Exception as e:
        logger.error(f"API Error fetching latest predictions from DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching latest predictions: {str(e)}")

@app.get("/tickers")
async def get_available_tickers():
    if not DB_CONFIG: raise HTTPException(status_code=500, detail="DB config error.")
    try:
        tickers = get_all_distinct_tickers_from_predictions(DB_CONFIG)
        return {"status": "success", "tickers": tickers}
    except Exception as e:
        logger.error(f"API Error fetching distinct tickers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching tickers: {str(e)}")

# --- NEW ENDPOINT FOR HISTORICAL CONTEXT CHART ---
@app.get("/historical_context_chart/{ticker}")
async def get_historical_context_for_prediction(ticker: str):
    if not DB_CONFIG:
        raise HTTPException(status_code=500, detail="Database configuration error.")

    ticker_upper = ticker.upper()

    # 1. Get the latest target_prediction_date for this ticker from our DB
    latest_pred_info = get_latest_target_date_prediction_for_ticker(DB_CONFIG, ticker_upper)

    if not latest_pred_info:
        # If no prediction exists for this ticker, we can't determine the context window.
        # Alternatively, could default to showing last 30 days of actuals if no prediction.
        # For now, require a prediction to define the context.
        logger.warning(f"No prediction found in DB for ticker {ticker_upper} to define historical context.")
        # Return empty arrays so frontend can handle it gracefully
        return {
            "status": "success", # Still a successful API call
            "ticker": ticker_upper,
            "dates": [],
            "actual_prices": []
        }
        # Or: raise HTTPException(status_code=404, detail=f"No prediction found for {ticker_upper} to base context on.")


    latest_target_prediction_date_obj = date.fromisoformat(latest_pred_info["target_prediction_date"])

    # 2. Define historical window: 30 trading days ending on the day *before* the prediction date.
    # Fetch a bit more (e.g., 45-50 calendar days) to account for non-trading days.
    history_end_date = latest_target_prediction_date_obj - timedelta(days=1)
    history_start_date = history_end_date - timedelta(days=50) # Fetch ~50 calendar days back

    actual_prices_list = []
    actual_dates_list = []

    try:
        logger.info(f"Fetching yfinance history for {ticker_upper} from {history_start_date} to {history_end_date}")
        stock_data_hist = yf.Ticker(ticker_upper).history(
            start=history_start_date, 
            end=history_end_date + timedelta(days=1) # yf 'end' is exclusive
        )
        
        if not stock_data_hist.empty:
            # Ensure we take the most recent data points if more than 30 trading days were fetched
            df_hist = stock_data_hist[['Close']].tail(30).copy() # Get last 30 available records in the window
            df_hist.index = df_hist.index.normalize().tz_localize(None) # Date only, no timezone
            
            for idx_date, row in df_hist.iterrows():
                actual_dates_list.append(idx_date.strftime('%Y-%m-%d'))
                actual_prices_list.append(row['Close'] if pd.notnull(row['Close']) else None)
        else:
            logger.warning(f"No yfinance historical data found for {ticker_upper} in range {history_start_date} to {history_end_date}")

    except Exception as e:
        logger.error(f"Error fetching yfinance data for {ticker_upper} in /historical_context_chart: {e}", exc_info=True)
        # Continue even if yfinance fails, will return empty actuals list

    # Sanitize for JSON (though yfinance usually returns clean floats or NaN)
    # Using a helper for this is good practice if complex data types could appear
    sanitized_actual_prices = []
    for price in actual_prices_list:
        if price is None or np.isnan(price) or np.isinf(price):
            sanitized_actual_prices.append(None)
        else:
            sanitized_actual_prices.append(float(price))

    return {
        "status": "success",
        "ticker": ticker_upper,
        "dates": actual_dates_list,                 # Dates for historical actuals
        "actual_prices": sanitized_actual_prices,   # Historical actual prices
        "prediction_reference_date": latest_pred_info["target_prediction_date"] # Include for context on UI
    }

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    if not templates:
        raise HTTPException(status_code=500, detail="Server configuration error: Templates not found.")
    return templates.TemplateResponse("index.html", {"request": request})