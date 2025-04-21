#!/usr/bin/env python
import argparse
import json
import yaml
import numpy as np
from pathlib import Path
import joblib
import torch
from datetime import date
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate latest stock price predictions and save daily historical JSON files"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to params.yaml"
    )
    return parser.parse_args()


def load_model_from_registry(experiment_name: str):
    """
    Load the PyTorch model from MLflow Registry (Production),
    fallback to most recent run if unavailable.
    """
    try:
        return mlflow.pytorch.load_model(f"models:/{experiment_name}/Production")
    except Exception:
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise ValueError(f"Experiment not found: {experiment_name}")
        runs = client.search_runs([exp.experiment_id], order_by=["attribute.start_time DESC"], max_results=1)
        if not runs:
            raise ValueError(f"No runs found in experiment {experiment_name}")
        run_id = runs[0].info.run_id
        return mlflow.pytorch.load_model(f"runs:/{run_id}/model")


def main():
    args = parse_args()

    # Load config
    cfg_path = Path(args.config)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Set paths from config
    split_data_path = Path(cfg['output_paths']['split_data_path'])
    scalers_path = Path(cfg['output_paths']['scalers_path'])
    processed_data_path = Path(cfg['output_paths']['processed_data_path'])
    predictions_dir = Path(cfg['output_paths'].get('predictions_dir', '/opt/airflow/data/predictions'))
    experiment_name = cfg['mlflow']['experiment_name']

    # Load latest test sequence
    if not split_data_path.exists():
        raise FileNotFoundError(f"Split data not found: {split_data_path}")
    data = np.load(split_data_path, allow_pickle=True)
    if 'X_test_scaled' in data:
        X_test = data['X_test_scaled']
    elif 'x_test' in data:
        X_test = data['x_test']
    else:
        raise KeyError("X_test_scaled or x_test not found in split data file")
    X_latest = X_test[-1:]
    X_tensor = torch.tensor(X_latest, dtype=torch.float32)

    # Load scalers
    if not scalers_path.exists():
        raise FileNotFoundError(f"Scalers not found: {scalers_path}")
    scaler_dict = joblib.load(scalers_path)
    y_scalers = scaler_dict.get('y_scalers')
    if y_scalers is None:
        raise KeyError("y_scalers key not found in scalers file")

    # Load model
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    model = load_model_from_registry(experiment_name)
    model.eval()

    # Predict
    with torch.no_grad():
        preds = model(X_tensor)
        preds_np = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else np.array(preds)

    # Load tickers list
    proc_data = np.load(processed_data_path, allow_pickle=True)
    tickers = proc_data['tickers'].tolist()

    # Inverse transform predictions
    results = {}
    for idx, ticker in enumerate(tickers):
        try:
            # Handle 3D output (batch, horizon, features)
            val_scaled = float(preds_np[0, 0, idx])
        except Exception:
            # 2D output (batch, features)
            val_scaled = float(preds_np[0, idx])
        scaler = y_scalers[idx]
        try:
            val = scaler.inverse_transform([[val_scaled]])[0][0]
        except Exception:
            val = val_scaled
        results[ticker] = val

    # Prepare output directories
    predictions_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = predictions_dir / 'historical'
    hist_dir.mkdir(parents=True, exist_ok=True)

    # Prepare payload
    today = date.today().isoformat()
    payload = {"date": today, "predictions": results}

    # Write latest
    latest_file = predictions_dir / 'latest_predictions.json'
    with open(latest_file, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved latest predictions to {latest_file}")

    # Write daily historical file
    hist_file = hist_dir / f"{today}.json"
    with open(hist_file, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Saved historical predictions to {hist_file}")

if _name_ == '_main_':
    main()