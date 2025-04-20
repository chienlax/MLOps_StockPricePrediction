from fastapi import FastAPI, HTTPException
from pathlib import Path
import json
from datetime import date

app = FastAPI(
    title="Stock Prediction API",
    description="API để cung cấp dự báo cổ phiếu mới nhất và theo ticker",
    version="1.0.0"
)

# Path to JSON predictions file (relative to project root)
PREDICTION_FILE_PATH = Path("data/predictions/latest_predictions.json")


def load_predictions():
    """
    Load the predictions JSON file and return a list of prediction entries.
    Each entry is a dict: {"ticker": str, "predicted_price": float, "date": str}
    """
    if not PREDICTION_FILE_PATH.exists():
        raise HTTPException(status_code=404, detail="Prediction file not found")
    try:
        with open(PREDICTION_FILE_PATH, "r") as f:
            data = json.load(f)

        # New format: {"date": "2025-04-20", "predictions": {"AAPL": 215.92, ...}}
        if isinstance(data, dict) and "predictions" in data and isinstance(data["predictions"], dict):
            date_str = data.get("date", date.today().isoformat())
            predictions_dict = data["predictions"]
            predictions_list = [
                {"ticker": ticker, "predicted_price": price, "date": date_str}
                for ticker, price in predictions_dict.items()
            ]
            return predictions_list

        raise ValueError("Unexpected JSON format: expected a dictionary with 'predictions' key")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing prediction file")
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predictions/latest")
async def get_latest_predictions():
    """
    Return all latest predictions as a list.
    """
    predictions = load_predictions()
    return {"status": "success", "data": predictions}


@app.get("/predictions/historical/{ticker}")
async def get_historical_predictions(ticker: str):
    """
    Return predictions filtered by ticker (case-insensitive).
    """
    predictions = load_predictions()
    filtered = [item for item in predictions if item.get("ticker", "").lower() == ticker.lower()]
    if not filtered:
        raise HTTPException(status_code=404, detail=f"No predictions found for ticker: {ticker}")
    return {"status": "success", "ticker": ticker.upper(), "data": filtered}
