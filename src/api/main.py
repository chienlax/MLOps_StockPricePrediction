from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
import json
from datetime import date

app = FastAPI(
    title="Stock Prediction API",
    description="API để cung cấp dự báo cổ phiếu mới nhất và theo ticker",
    version="1.0.0"
)

# Các đường dẫn dữ liệu
BASE_DATA_PATH_INSIDE_CONTAINER = Path("/opt/airflow/data") # Define base path based on volume mount
PREDICTION_FILE_PATH = BASE_DATA_PATH_INSIDE_CONTAINER / "predictions/latest_predictions.json"  # Path("data/predictions/latest_predictions.json")
HISTORICAL_DIR = BASE_DATA_PATH_INSIDE_CONTAINER / "predictions/historical"  # Path("data/predictions/historical")

# Mount static
# app.mount("/static", StaticFiles(directory="templates/static"), name="static")
app.mount("/static", StaticFiles(directory="/opt/airflow/templates/static"), name="static")

# Cấu hình Jinja2
templates = Jinja2Templates(directory="templates")

# ------------------ TẢI DỮ LIỆU ------------------

def load_predictions():
    """
    Đọc file latest_predictions.json và trả về danh sách dự đoán dạng:
    [{"ticker": str, "predicted_price": float, "date": str}, ...]
    """
    if not PREDICTION_FILE_PATH.exists():
        raise HTTPException(status_code=404, detail="Prediction file not found")
    try:
        with open(PREDICTION_FILE_PATH, "r") as f:
            data = json.load(f)

        if isinstance(data, dict) and "predictions" in data:
            date_str = data.get("date", date.today().isoformat())
            predictions_dict = data["predictions"]
            predictions_list = [
                {"ticker": ticker, "predicted_price": price, "date": date_str}
                for ticker, price in predictions_dict.items()
            ]
            return predictions_list

        raise ValueError("Invalid format in latest_predictions.json")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing prediction file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def load_historical_predictions_for_ticker(ticker: str):
    """
    Duyệt tất cả các file trong data/predictions/historical/ và trả về
    các dự đoán cho ticker tương ứng (theo thời gian).
    """
    if not HISTORICAL_DIR.exists():
        raise HTTPException(status_code=404, detail="Historical directory not found")

    results = []
    for file in sorted(HISTORICAL_DIR.glob("*.json")):
        try:
            with open(file, "r") as f:
                data = json.load(f)
                date_str = data.get("date", file.stem)
                predictions = data.get("predictions", {})
                if ticker.upper() in predictions:
                    results.append({
                        "ticker": ticker.upper(),
                        "predicted_price": predictions[ticker.upper()],
                        "date": date_str
                    })
        except Exception as e:
            print(f"Warning: Error reading {file.name}: {e}")
            continue

    return results

# ------------------ API ------------------

@app.get("/predictions/latest")
async def get_latest_predictions():
    """
    API trả về các dự đoán mới nhất từ latest_predictions.json
    """
    predictions = load_predictions()
    return {"status": "success", "data": predictions}

@app.get("/predictions/historical/{ticker}")
async def get_historical_predictions(ticker: str):
    """
    API trả về lịch sử dự đoán cho 1 ticker cụ thể từ nhiều ngày.
    """
    predictions = load_historical_predictions_for_ticker(ticker)
    if not predictions:
        raise HTTPException(status_code=404, detail=f"No historical predictions found for ticker: {ticker}")
    return {"status": "success", "ticker": ticker.upper(), "data": predictions}

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """
    Giao diện chính Dashboard
    """
    return templates.TemplateResponse("index.html", {"request": request})
#uvicorn src.api.main:app --reload
