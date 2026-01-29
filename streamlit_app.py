import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import pathlib
import sys

# Add project root to path
sys.path.append(str(pathlib.Path(__file__).parent))

from src.model import PalantirLSTM
from src.utils import (
    load_ticker, compute_RSI, compute_MACD, compute_bollinger_width,
    compute_ROC, compute_ATR, compute_stochastic_k
)
from src.fetch_data import fetch_all_data
from src.sentiment import get_current_sentiment

# Set page config
st.set_page_config(page_title="PLTR Stock AI", page_icon="📈", layout="wide")

# Constants
LOOKBACK = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FEATURE_COLS = [
    "Close", "Volume", "MA_5", "MA_10", "MA_20",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_width", "ROC_10", "ATR_14", "Stoch_K",
    "NAS_Close", "NAS_Volume", "NAS_ret_1", "NAS_ret_5"
]

@st.cache_resource
def load_resources(checkpoints_dir):
    checkpoints_path = pathlib.Path(checkpoints_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = checkpoints_path / "palantir_lstm.pth"
    feature_scaler_path = checkpoints_path / "feature_scaler.pkl"
    
    if not model_path.exists() or not feature_scaler_path.exists():
        return None, None, None

    # Load Scalers
    feature_scaler = joblib.load(feature_scaler_path)
    
    # Load Model
    model = PalantirLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, feature_scaler, device

def run_inference(data_dir, model, feature_scaler, device):
    data_path = pathlib.Path(data_dir)
    pltr_path = data_path / "PLTR_current.csv"
    ixic_path = data_path / "IXIC_current.csv"

    if not pltr_path.exists():
        return None, "Data files not found. Please fetch data first."

    # Load Data
    pltr_df, _ = load_ticker(pltr_path)
    nasdaq_df, _ = load_ticker(ixic_path)

    # Feature Engineer
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

    if len(merged) < LOOKBACK:
        return None, f"Not enough data. Need {LOOKBACK} rows, have {len(merged)}"

    last_segment = merged.iloc[-LOOKBACK:]
    last_close = last_segment["Close"].iloc[-1]
    last_date = last_segment["Date"].iloc[-1]

    features = last_segment[FEATURE_COLS].values
    features_scaled = feature_scaler.transform(features)
    
    X_input = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_ret, pred_up = model(X_input)

    pred_ret_val = float(pred_ret.cpu().numpy()[0, 0])
    p_up = float(pred_up.cpu().numpy()[0, 0])
    model_verdict = "UP" if p_up >= 0.5 else "DOWN"
    next_close_pred = last_close * (1.0 + pred_ret_val)

    return {
        "last_close": last_close,
        "last_date": last_date,
        "pred_ret": pred_ret_val,
        "p_up": p_up,
        "model_verdict": model_verdict,
        "next_close_pred": next_close_pred,
        "history": last_segment,
        "merged_data": merged
    }, None

# UI Layout
st.title("Palantir (PLTR) Stock Predictor 🤖")
st.markdown("Hybrid LSTM + Sentiment Analysis System")

# Sidebar
st.sidebar.header("Configuration")
data_dir = st.sidebar.text_input("Data Directory", "data")
checkpoints_dir = st.sidebar.text_input("Checkpoints Directory", "checkpoints")

# Main execution safely wrapped
try:
    if st.sidebar.button("Fetch Latest Data"):
        with st.spinner("Downloading data from Yahoo Finance..."):
            try:
                fetch_all_data(data_dir)
                st.sidebar.success("Data updated!")
            except Exception as e:
                st.sidebar.error(f"Error fetching data: {e}")

    # Auto-download if missing
    pltr_path = pathlib.Path(data_dir) / "PLTR_current.csv"
    if not pltr_path.exists():
        with st.spinner("Data not found. Downloading automatically..."):
            try:
                fetch_all_data(data_dir)
                st.success("Data downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to auto-download data: {e}")

    try:
        model, scaler, device = load_resources(checkpoints_dir)
    except Exception as e:
        st.error(f"Failed to load model/resources: {e}")
        st.info("Ensure that 'checkpoints/' directory exists and contains 'palantir_lstm.pth' and 'feature_scaler.pkl'.")
        st.stop()

    if model is None:
        st.error("Model or Scaler not found in checkpoints directory. Please train the model first.")
    else:
        # Run Inference
        res, error = run_inference(data_dir, model, scaler, device)
        
        if error:
            st.warning(error)
            if "not found" in str(error).lower():
                st.info("Try clicking 'Fetch Latest Data' in the sidebar.")
        else:
            # Sentiment Analysis
            with st.spinner("Analyzing News Sentiment..."):
                sentiment = get_current_sentiment("PLTR")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Price", f"${res['next_close_pred']:.2f}", f"{res['pred_ret']*100:.2f}%")
            
            with col2:
                st.metric("Technical Signal", res['model_verdict'], f"Confidence: {res['p_up']*100:.1f}%")
                
            with col3:
                st.metric("News Sentiment", sentiment['verdict'], f"Score: {sentiment['score']:.2f}")

            # Final Verdict Logic
            model_verdict = res['model_verdict']
            news_verdict = sentiment['verdict']
            
            final_rec = "HOLD"
            color = "gray"
            if model_verdict == "UP" and news_verdict == "POSITIVE":
                final_rec = "STRONG BUY"
                color = "green"
            elif model_verdict == "DOWN" and news_verdict == "NEGATIVE":
                final_rec = "STRONG SELL"
                color = "red"
            elif model_verdict == "UP":
                final_rec = "BUY (Model Bullish)"
                color = "blue"
            elif model_verdict == "DOWN":
                final_rec = "SELL (Model Bearish)"
                color = "orange"
                
            st.subheader(f"Final Verdict: :{color}[{final_rec}]")
            
            # Plotting
            st.markdown("### Forecast Visualization")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            history = res['history']
            ax.plot(history["Date"], history["Close"], label="History (60 Days)", color='blue')
            
            # Plot prediction
            pred_date = res['last_date'] + pd.Timedelta(days=1)
            ax.scatter(pred_date, res['next_close_pred'], color="red", label="Prediction", marker="x", s=150, zorder=5)
            
            ax.set_title(f"Palantir Hybrid Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # News
            st.markdown("### Top Headlines")
            for h in sentiment['headlines']:
                st.info(h)

except Exception as e:
    import traceback
    st.error(f"CRITICAL APP ERROR: {e}")
    st.text(traceback.format_exc())
    st.stop()
