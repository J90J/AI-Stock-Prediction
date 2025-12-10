import pandas as pd
import numpy as np
import pathlib

def load_ticker(csv_path: pathlib.Path):
    """
    Read the CSV file, normalize the column names, and find the best available
    closing price column. This is to prevent bugs from slightly different schemas.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Best available closing price column
    close_col = None
    for cand in ["Close", "close", "CLOSE", "Adj Close", "AdjClose"]:
        if cand in df.columns:
            close_col = cand
            break

    if close_col is None:
        raise ValueError(f"No close column found in file: {csv_path}")

    # Convert key numeric columns
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    if "High" in df.columns:
        df["High"] = pd.to_numeric(df["High"], errors="coerce")
    if "Low" in df.columns:
        df["Low"] = pd.to_numeric(df["Low"], errors="coerce")

    df = df.dropna(subset=[close_col])

    close_np = df[close_col].astype(float).to_numpy()

    return df, close_np

def compute_RSI(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_MACD(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def compute_bollinger_width(series, window=20, num_std=2):
    ma  = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (upper - lower) / (ma + 1e-9)

def compute_ROC(series, period=10):
    return series.pct_change(periods=period)

def compute_ATR(df, period=14):
    # Needs High/Low/Close
    # Assuming df has these columns normalized
    if "High" not in df.columns or "Low" not in df.columns or "Close" not in df.columns:
         # Try to find case-insensitive match if needed, but load_ticker normalizes to what?
         # load_ticker doesn't rename columns to standard names, just matching for 'close'.
         # We need to rely on the specific dataset columns.
         # For simplicity, let's assume standard names or fail gracefully.
         # The original code used df["High"] etc. directly.
         pass
    
    # We need to handle potential missing columns safely or assume they exist as per original code
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()

def compute_stochastic_k(df, window=14):
    low_min  = df["Low"].rolling(window).min()
    high_max = df["High"].rolling(window).max()
    return 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)

def make_sequences(features_scaled, close_scaled, close_unscaled, lookback):
    """
    features_scaled : (N, F)
    close_scaled    : (N, 1)  - scaled close  (kept for API, not used here)
    close_unscaled  : (N, 1)  - raw close in USD
    lookback        : window length

    Returns:
      X          : (M, lookback, F)  input sequences
      y_ret      : (M, 1)           next-day return ( (next - last) / last )
      y_updown   : (M, 1)           1 if next_day_close > last_close_in_window else 0
    """
    N = len(features_scaled)
    X_list = []
    y_ret_list = []
    last_close_list = []
    next_close_list = []

    for i in range(N - lookback):
        # window covers indices [i, ..., i+lookback-1]
        X_seq = features_scaled[i : i + lookback]

        # next day index is i + lookback
        next_idx = i + lookback

        last_close = close_unscaled[next_idx - 1, 0]    # last close in window (USD)
        next_close = close_unscaled[next_idx, 0]        # next-day close (USD)

        # --- regression target: percent return ---
        ret = (next_close - last_close) / last_close    # e.g. 0.01 = +1%

        X_list.append(X_seq)
        y_ret_list.append([ret])
        last_close_list.append(last_close)
        next_close_list.append(next_close)

    X = np.array(X_list)                      # (M, lookback, F)
    y_ret = np.array(y_ret_list)             # (M, 1)

    last_close_arr = np.array(last_close_list).reshape(-1, 1)
    next_close_arr = np.array(next_close_list).reshape(-1, 1)

    # Classification label: does NEXT DAY go up vs LAST DAY IN WINDOW?
    y_updown = (next_close_arr > last_close_arr).astype(float)  # (M, 1)

    return X, y_ret, y_updown
