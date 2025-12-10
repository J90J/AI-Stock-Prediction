import sys
import pathlib
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.model import PalantirLSTM
from src.utils import (
    load_ticker, compute_RSI, compute_MACD, compute_bollinger_width,
    compute_ROC, compute_ATR, compute_stochastic_k
)

# Constants (Must match training)
LOOKBACK = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FEATURE_COLS = [
    "Close", "Volume", "MA_5", "MA_10", "MA_20",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_width", "ROC_10", "ATR_14", "Stoch_K",
    "NAS_Close", "NAS_Volume", "NAS_ret_1", "NAS_ret_5"
]

def run_demo(data_dir, checkpoints_dir, results_dir):
    data_path = pathlib.Path(data_dir)
    checkpoints_path = pathlib.Path(checkpoints_dir)
    results_path = pathlib.Path(results_dir)
    results_path.mkdir(exist_ok=True, parents=True)

    # Check for resources
    model_path = checkpoints_path / "palantir_lstm.pth"
    feature_scaler_path = checkpoints_path / "feature_scaler.pkl"
    close_scaler_path = checkpoints_path / "close_scaler.pkl"
    pltr_path = data_path / "PLTR (11).csv"
    ixic_path = data_path / "^IXIC (10).csv"

    missing = []
    if not model_path.exists(): missing.append(str(model_path))
    if not feature_scaler_path.exists(): missing.append(str(feature_scaler_path))
    if not pltr_path.exists(): missing.append(str(pltr_path))

    if missing:
        print("Error: Missing required files to run demo.")
        print("Missing:", missing)
        print("Please train the model first by moving your data to 'data/' and running 'python src/main.py'")
        return

    print("Loading resources...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Scalers
    feature_scaler = joblib.load(feature_scaler_path)
    
    # Load Data (process last chunk for demo)
    print("Processing latest data...")
    pltr_df, _ = load_ticker(pltr_path)
    nasdaq_df, _ = load_ticker(ixic_path)

    # Feature Engineer (Duplicate logic - in prod this would be in a shared transform function)
    pltr_df["Date"] = pd.to_datetime(pltr_df["Date"])
    nasdaq_df["Date"] = pd.to_datetime(nasdaq_df["Date"])
    pltr_df = pltr_df.sort_values("Date").reset_index(drop=True)
    nasdaq_df = nasdaq_df.sort_values("Date").reset_index(drop=True)

    merged = pltr_df.merge(
        nasdaq_df[["Date", "Close", "Volume"]].rename(columns={"Close": "NAS_Close", "Volume": "NAS_Volume"}),
        on="Date", how="inner"
    )

    merged["MA_5"]  = merged["Close"].rolling(window=5).mean()
    merged["MA_10"] = merged["Close"].rolling(window=10).mean()
    merged["MA_20"] = merged["Close"].rolling(window=20).mean()
    merged["RSI_14"] = compute_RSI(merged["Close"])
    merged["MACD"], merged["MACD_signal"], merged["MACD_hist"] = compute_MACD(merged["Close"])
    merged["BB_width"] = compute_bollinger_width(merged["Close"])
    merged["ROC_10"]   = compute_ROC(merged["Close"])
    merged["ATR_14"]   = compute_ATR(merged)
    merged["Stoch_K"]  = compute_stochastic_k(merged)
    merged["NAS_ret_1"] = merged["NAS_Close"].pct_change(1)
    merged["NAS_ret_5"] = merged["NAS_Close"].pct_change(5)
    merged = merged.dropna().reset_index(drop=True)

    # Take last LOOKBACK days
    if len(merged) < LOOKBACK:
        print(f"Not enough data. Need {LOOKBACK} rows, have {len(merged)}")
        return

    last_segment = merged.iloc[-LOOKBACK:]
    last_close = last_segment["Close"].iloc[-1]
    last_date = last_segment["Date"].iloc[-1]

    # Prepare input
    features = last_segment[FEATURE_COLS].values
    features_scaled = feature_scaler.transform(features)
    
    X_input = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # Load Model
    model = PalantirLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Running prediction for date after {last_date.date()}...")
    with torch.no_grad():
        pred_ret, pred_up = model(X_input)

    pred_ret_val = float(pred_ret.cpu().numpy()[0, 0])
    p_up = float(pred_up.cpu().numpy()[0, 0])

    next_close_pred = last_close * (1.0 + pred_ret_val)
    direction = "UP" if p_up >= 0.5 else "DOWN"

    print(f"Last Actual Close: ${last_close:.2f}")
    print(f"Predicted Next Close: ${next_close_pred:.2f}")
    print(f"Predicted Return: {pred_ret_val*100:.2f}%")
    print(f"Direction Probability (UP): {p_up*100:.1f}% -> {direction}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(last_segment["Date"], last_segment["Close"], label="History (Last 60 Days)")
    plt.scatter(last_date + pd.Timedelta(days=1), next_close_pred, color="red", label="Prediction", marker="x", s=100)
    plt.title(f"Palantir Stock Prediction: {direction}")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    
    out_file = results_path / "prediction_plot.png"
    plt.savefig(out_file)
    print(f"Plot saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    run_demo(args.data_dir, args.checkpoints_dir, args.results_dir)
