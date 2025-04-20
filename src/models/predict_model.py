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
        description="Generate latest stock price predictions"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to params.yaml"
    )
    return parser.parse_args()


def load_model_from_registry(experiment_name):
    # Try to load from model registry Production stage
    try:
        return mlflow.pytorch.load_model(f"models:/{experiment_name}/Production")
    except Exception:
        # Fallback: load latest run artifact
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise ValueError(f"Experiment not found: {experiment_name}")
        runs = client.search_runs([exp.experiment_id], order_by=["attribute.start_time DESC"], max_results=1)
        if not runs:
            raise ValueError(f"No runs found in experiment {experiment_name}")
        run_id = runs[0].info.run_id
        model_uri = f"runs:/{run_id}/model"
        return mlflow.pytorch.load_model(model_uri)


def main():
    args = parse_args()
    # Load configuration
    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Paths from config
    split_data_path = Path(cfg['output_paths']['split_data_path'])
    scalers_path = Path(cfg['output_paths']['scalers_path'])
    processed_data_path = Path(cfg['output_paths']['processed_data_path'])
    experiment_name = cfg['mlflow']['experiment_name']

    # Load split-scaled data
    if not split_data_path.exists():
        raise FileNotFoundError(f"Split data not found: {split_data_path}")
    data = np.load(split_data_path, allow_pickle=True)

    # Safely extract X_test_scaled or x_test
    if 'X_test_scaled' in data:
        X_test = data['X_test_scaled']
    elif 'x_test' in data:
        X_test = data['x_test']
    else:
        raise KeyError("X_test_scaled or x_test not found in split data file")

    # Prepare latest sequence for prediction
    X_latest = X_test[-1:]
    X_tensor = torch.tensor(X_latest, dtype=torch.float32)

    # Load scalers
    if not scalers_path.exists():
        raise FileNotFoundError(f"Scalers not found: {scalers_path}")
    scalers_dict = joblib.load(scalers_path)
    y_scalers = scalers_dict.get('y_scalers')
    if y_scalers is None:
        raise KeyError("y_scalers key not found in scalers file")

    # Load model
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
    model = load_model_from_registry(experiment_name)
    model.eval()

    # Predict
    with torch.no_grad():
        preds = model(X_tensor)
        preds_scaled = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else np.array(preds)

    # Load tickers
    proc_data = np.load(processed_data_path, allow_pickle=True)
    tickers = proc_data['tickers'].tolist()

    # Inverse scale and assemble
    results = {}
    for idx, ticker in enumerate(tickers):
        # Determine scaled value dimension
        try:
            # 3D array: (batch, pred_len, num_stocks)
            val_scaled = float(preds_scaled[0, 0, idx])
        except Exception:
            # 2D array: (batch, num_stocks)
            val_scaled = float(preds_scaled[0, idx])
        scaler = y_scalers[idx]
        try:
            val = scaler.inverse_transform([[val_scaled]])[0][0]
        except Exception:
            val = val_scaled
        results[ticker] = val

    # Save to JSON
    out_dir = Path("/opt/airflow/data/predictions")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "latest_predictions.json"
    payload = {"date": date.today().isoformat(), "predictions": results}
    with open(out_file, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"Saved latest predictions to {out_file}")


if __name__ == '__main__':
    main()
