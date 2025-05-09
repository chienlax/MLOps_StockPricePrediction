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

# src/api/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import yfinance as yf
import yaml 
import sys # For sys.exit in case of critical config error

# --- Path Definitions - Define PROJECT_ROOT_API at the top module level ---
# Path to project root, assuming api/main.py is in src/api/
PROJECT_ROOT_API = Path(__file__).resolve().parents[2] 
CONFIG_FILE_FOR_API = PROJECT_ROOT_API / "config/params.yaml"

# Static files and Templates directories
TEMPLATES_DIR = PROJECT_ROOT_API / "templates"
STATIC_DIR = TEMPLATES_DIR / "static"


# --- Load Configuration and DB Utils ---
DB_CONFIG = None
# PREDICTIONS_DIR_FOR_API = None # This path will be derived after params_config is loaded

# Logger for FastAPI (define early)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Define logger at module level

try:
    # Ensure 'src' is in PYTHONPATH if db_utils is not found by default
    # This might be needed if PYTHONPATH isn't set correctly in the uvicorn environment
    sys.path.insert(0, str(PROJECT_ROOT_API / "src")) # Add src to path
    from utils.db_utils import (
        get_db_connection, 
        get_predictions_for_ticker_in_daterange,
        get_latest_prediction_for_all_tickers,
        get_all_distinct_tickers_from_predictions
    )
    
    with open(CONFIG_FILE_FOR_API, 'r') as f_params:
        params_config = yaml.safe_load(f_params)
    DB_CONFIG = params_config.get('database')
    
    if not DB_CONFIG:
        logger.error("CRITICAL: Database configuration not found in params.yaml. API cannot function.")
        # You might want to exit here if DB is absolutely critical for app startup
        # For now, other endpoints might work if they don't use DB_CONFIG.

except ImportError as e_imp:
    logger.error(f"CRITICAL: Could not import db_utils. Ensure PYTHONPATH is correct or utils are accessible: {e_imp}", exc_info=True)
    # API likely cannot function without db_utils.
    # Consider sys.exit(1) or raise a startup error Uvicorn can catch.
except FileNotFoundError:
    logger.error(f"CRITICAL: Config file {CONFIG_FILE_FOR_API} not found. API cannot load configuration.")
    # Consider sys.exit(1)
except Exception as e_cfg:
    logger.error(f"CRITICAL: Error loading params.yaml ({CONFIG_FILE_FOR_API}) or extracting DB config: {e_cfg}", exc_info=True)
    # Consider sys.exit(1)


app = FastAPI(
    title="Stock Prediction API",
    description="API để cung cấp dự báo cổ phiếu mới nhất và theo ticker",
    version="1.0.0"
)

# Mount static files
# Ensure STATIC_DIR is correctly defined (it is, at the top now)
if STATIC_DIR.exists() and STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.warning(f"Static directory {STATIC_DIR} not found. Static files will not be served.")

# Cấu hình Jinja2
# Ensure TEMPLATES_DIR is correctly defined (it is, at the top now)
if TEMPLATES_DIR.exists() and TEMPLATES_DIR.is_dir():
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
else:
    logger.error(f"Templates directory {TEMPLATES_DIR} not found. HTML responses will fail.")
    templates = None # Prevent further errors if templates are used when dir is missing


# --- API Endpoints ---

@app.get("/predictions/latest_for_table")
async def get_latest_predictions_for_table_from_db():
    if not DB_CONFIG:
        logger.error("API call to /predictions/latest_for_table failed: DB_CONFIG not loaded.")
        raise HTTPException(status_code=500, detail="Database configuration error.")
    
    try:
        predictions = get_latest_prediction_for_all_tickers(DB_CONFIG)
        if not predictions:
            logger.info("No latest predictions found in the database for the table.")
            return {"status": "success", "data": []} 
        return {"status": "success", "data": predictions}
    except Exception as e:
        logger.error(f"API Error fetching latest predictions from DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching latest predictions: {str(e)}")

@app.get("/tickers")
async def get_available_tickers():
    if not DB_CONFIG:
        logger.error("API call to /tickers failed: DB_CONFIG not loaded.")
        raise HTTPException(status_code=500, detail="Database configuration error.")
    try:
        tickers = get_all_distinct_tickers_from_predictions(DB_CONFIG)
        return {"status": "success", "tickers": tickers}
    except Exception as e:
        logger.error(f"API Error fetching distinct tickers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching tickers: {str(e)}")

@app.get("/chart_data/{ticker}")
async def get_chart_data(ticker: str):
    if not DB_CONFIG:
        logger.error(f"API call to /chart_data/{ticker} failed: DB_CONFIG not loaded.")
        raise HTTPException(status_code=500, detail="Database configuration error.")

    ticker_upper = ticker.upper() # Use a different variable name
    end_date = date.today()
    start_date_actuals = end_date - timedelta(days=45) 
    start_date_preds = end_date - timedelta(days=30)

    actual_prices_df = pd.DataFrame()
    try:
        stock_data = yf.Ticker(ticker_upper).history(start=start_date_actuals, end=end_date + timedelta(days=1))
        if not stock_data.empty:
            actual_prices_df = stock_data[['Close']].copy()
            actual_prices_df.index = actual_prices_df.index.normalize().tz_localize(None)
            actual_prices_df.rename(columns={'Close': 'actual_price'}, inplace=True)
            actual_prices_df = actual_prices_df.reset_index().rename(columns={'Date': 'date', 'index':'date'})
            actual_prices_df['date_str'] = actual_prices_df['date'].dt.strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"Error fetching yfinance data for {ticker_upper} in /chart_data: {e}")

    predicted_prices_df = get_predictions_for_ticker_in_daterange(
        DB_CONFIG, ticker_upper, start_date_preds.isoformat(), end_date.isoformat()
    )
    if not predicted_prices_df.empty:
        # Ensure column names match after potential rename in db_utils or here
        # Assuming get_predictions_for_ticker_in_daterange returns 'target_prediction_date' and 'predicted_price'
        predicted_prices_df.rename(columns={'target_prediction_date': 'date', 'predicted_price': 'predicted_price'}, inplace=True, errors='ignore')
        predicted_prices_df['date'] = pd.to_datetime(predicted_prices_df['date'])
        predicted_prices_df['date_str'] = predicted_prices_df['date'].dt.strftime('%Y-%m-%d')


    last_30_days_range = pd.date_range(start=start_date_preds, end=end_date, freq='D')
    chart_df = pd.DataFrame(last_30_days_range, columns=['date'])
    chart_df['date_str'] = chart_df['date'].dt.strftime('%Y-%m-%d')

    if not actual_prices_df.empty:
        chart_df = pd.merge(chart_df, actual_prices_df[['date_str', 'actual_price']], on='date_str', how='left')
    else:
        chart_df['actual_price'] = pd.NA # Use pandas NA for proper handling of missing numeric

    if not predicted_prices_df.empty:
        chart_df = pd.merge(chart_df, predicted_prices_df[['date_str', 'predicted_price']], on='date_str', how='left')
    else:
        chart_df['predicted_price'] = pd.NA

    chart_df = chart_df.sort_values(by='date').reset_index(drop=True)
    
    # Convert to float or None for JSON. pd.NA might not serialize well directly with FastAPI's default JSON encoder.
    # Or ensure FastAPI/Pydantic handles pd.NA correctly. For simplicity:
    chart_df['actual_price'] = chart_df['actual_price'].apply(lambda x: float(x) if pd.notna(x) else None)
    chart_df['predicted_price'] = chart_df['predicted_price'].apply(lambda x: float(x) if pd.notna(x) else None)


    # Check if there's any data to plot at all
    if chart_df['actual_price'].isnull().all() and chart_df['predicted_price'].isnull().all():
         logger.warning(f"No chartable data (actual or predicted) found for ticker: {ticker_upper} in the date range.")
         # Return success with empty data so frontend can handle it.
         return {
            "status": "success", # Still a successful API call, just no data
            "ticker": ticker_upper,
            "dates": [],
            "actual_prices": [],
            "predicted_prices": []
        }
        # Or raise HTTPException if preferred:
        # raise HTTPException(status_code=404, detail=f"No chartable historical or predicted data found for ticker: {ticker_upper}")


    return {
        "status": "success",
        "ticker": ticker_upper,
        "dates": chart_df['date_str'].tolist(),
        "actual_prices": chart_df['actual_price'].tolist(),
        "predicted_prices": chart_df['predicted_price'].tolist()
    }

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    if not templates:
        logger.error("Templates not initialized. Dashboard cannot be served.")
        raise HTTPException(status_code=500, detail="Server configuration error: Templates not found.")
    return templates.TemplateResponse("index.html", {"request": request})