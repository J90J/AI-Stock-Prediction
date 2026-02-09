import yfinance as yf
import pandas as pd
import pathlib

def fetch_all_data(ticker="PLTR", data_dir="data"):
    """
    Downloads full historical data for a specific ticker and ^IXIC using yfinance.
    Saves them to {ticker}_current.csv and IXIC_current.csv in data_dir.
    """
    data_path = pathlib.Path(data_dir)
    data_path.mkdir(exist_ok=True, parents=True)

    print(f"Downloading latest data for {ticker} from Yahoo Finance...")

    # 1. Target Stock
    print(f"Fetching {ticker}...")
    stock = yf.Ticker(ticker)
    stock_hist = stock.history(period="max")
    
    if stock_hist.empty:
        raise ValueError(f"No data found for ticker symbol '{ticker}'")

    # yfinance returns index as Datetime, reset to get 'Date' column
    stock_hist = stock_hist.reset_index()
    # Format Date to YYYY-MM-DD for consistency
    stock_hist["Date"] = stock_hist["Date"].dt.strftime("%Y-%m-%d")
    
    stock_file = data_path / f"{ticker}_current.csv"
    stock_hist.to_csv(stock_file, index=False)
    print(f"Saved {ticker} data to {stock_file} ({len(stock_hist)} rows)")

    # 2. NASDAQ (IVIC)
    print("Fetching ^IXIC (NASDAQ)...")
    ixic = yf.Ticker("^IXIC")
    ixic_hist = ixic.history(period="max")
    ixic_hist = ixic_hist.reset_index()
    ixic_hist["Date"] = ixic_hist["Date"].dt.strftime("%Y-%m-%d")

    ixic_file = data_path / "IXIC_current.csv"
    ixic_hist.to_csv(ixic_file, index=False)
    print(f"Saved NASDAQ data to {ixic_file} ({len(ixic_hist)} rows)")

if __name__ == "__main__":
    fetch_all_data()
