import json
import pathlib
import sys
import pandas as pd
import numpy as np
import torch
import joblib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.model import StockLSTM
from src.utils import (
    load_ticker, compute_RSI, compute_MACD, compute_bollinger_width,
    compute_ROC, compute_ATR, compute_stochastic_k
)

LOOKBACK = 60
HIDDEN_SIZE = 64
NUM_LAYERS = 2
FEATURE_COLS = [
    "Close", "Volume", "MA_5", "MA_10", "MA_20",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_width", "ROC_10", "ATR_14", "Stoch_K",
    "NAS_Close", "NAS_Volume", "NAS_ret_1", "NAS_ret_5"
]

TOOLS = [
    {
        "name": "fetch_data",
        "description": (
            "Downloads the latest 5 years of stock price data and NASDAQ index data "
            "for a given ticker from Yahoo Finance. Always call this before train_model "
            "if data might be stale or missing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL, PLTR, NVDA"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "train_model",
        "description": (
            "Trains the LSTM neural network for a given stock ticker. "
            "Call this after fetch_data when no model checkpoint exists, "
            "or when the user explicitly asks to retrain."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "epochs": {
                    "type": "integer",
                    "description": "Training epochs (default 30, max 50)"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "predict",
        "description": (
            "Runs the trained LSTM model to predict the next trading day's price "
            "direction and return for a stock. Requires a trained model checkpoint."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_sentiment",
        "description": (
            "Fetches recent news headlines for a stock and scores sentiment "
            "using VADER. Returns a score from -1 (very negative) to 1 (very positive), "
            "a verdict (POSITIVE/NEUTRAL/NEGATIVE), and the top headlines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                }
            },
            "required": ["ticker"]
        }
    }
]

SYSTEM_PROMPT = """You are an AI stock analysis agent. You have four tools:
- fetch_data: download latest price history for a stock
- train_model: train the LSTM prediction model
- predict: run the technical price-direction forecast
- get_sentiment: analyze recent news headlines

When a user asks about a stock or trading decision, follow this workflow:
1. fetch_data to get fresh data (always do this for the first request)
2. train_model if no checkpoint exists (check by trying predict first — it will tell you if missing)
3. predict to get the technical signal
4. get_sentiment to get the news signal
5. Synthesize results into a clear recommendation with confidence level

Keep responses concise and structured. Always note that predictions are probabilistic and not financial advice."""


def _run_inference(ticker: str, data_dir: str, checkpoints_dir: str) -> dict:
    data_path = pathlib.Path(data_dir)
    checkpoints_path = pathlib.Path(checkpoints_dir)

    model_path = checkpoints_path / f"{ticker}_lstm.pth"
    feature_scaler_path = checkpoints_path / f"{ticker}_feature_scaler.pkl"

    if not model_path.exists() or not feature_scaler_path.exists():
        return {"error": f"No model found for {ticker}. Please call train_model first."}

    stock_path = data_path / f"{ticker}_current.csv"
    ixic_path = data_path / "IXIC_current.csv"

    if not stock_path.exists():
        return {"error": f"No data found for {ticker}. Please call fetch_data first."}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_scaler = joblib.load(feature_scaler_path)

    model = StockLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    stock_df, _ = load_ticker(stock_path)
    nasdaq_df, _ = load_ticker(ixic_path)

    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    nasdaq_df["Date"] = pd.to_datetime(nasdaq_df["Date"])
    stock_df = stock_df.sort_values("Date").reset_index(drop=True)
    nasdaq_df = nasdaq_df.sort_values("Date").reset_index(drop=True)

    merged = stock_df.merge(
        nasdaq_df[["Date", "Close", "Volume"]].rename(
            columns={"Close": "NAS_Close", "Volume": "NAS_Volume"}
        ),
        on="Date", how="inner"
    )

    merged["MA_5"]       = merged["Close"].rolling(5).mean()
    merged["MA_10"]      = merged["Close"].rolling(10).mean()
    merged["MA_20"]      = merged["Close"].rolling(20).mean()
    merged["RSI_14"]     = compute_RSI(merged["Close"])
    merged["MACD"], merged["MACD_signal"], merged["MACD_hist"] = compute_MACD(merged["Close"])
    merged["BB_width"]   = compute_bollinger_width(merged["Close"])
    merged["ROC_10"]     = compute_ROC(merged["Close"])
    merged["ATR_14"]     = compute_ATR(merged)
    merged["Stoch_K"]    = compute_stochastic_k(merged)
    merged["NAS_ret_1"]  = merged["NAS_Close"].pct_change(1)
    merged["NAS_ret_5"]  = merged["NAS_Close"].pct_change(5)
    merged = merged.dropna().reset_index(drop=True)

    if len(merged) < LOOKBACK:
        return {"error": f"Not enough data. Need {LOOKBACK} rows, have {len(merged)}."}

    last_segment = merged.iloc[-LOOKBACK:]
    last_close   = float(last_segment["Close"].iloc[-1])
    last_date    = str(last_segment["Date"].iloc[-1].date())

    features        = last_segment[FEATURE_COLS].values
    features_scaled = feature_scaler.transform(features)
    X_input         = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_ret, pred_up = model(X_input)

    pred_ret_val = float(pred_ret.cpu().numpy()[0, 0])
    p_up         = float(pred_up.cpu().numpy()[0, 0])

    return {
        "ticker":               ticker,
        "last_close":           round(last_close, 2),
        "last_date":            last_date,
        "predicted_return_pct": round(pred_ret_val * 100, 2),
        "predicted_next_close": round(last_close * (1 + pred_ret_val), 2),
        "probability_up":       round(p_up * 100, 1),
        "direction":            "UP" if p_up >= 0.5 else "DOWN"
    }


def execute_tool(tool_name: str, tool_input: dict, data_dir: str, checkpoints_dir: str) -> dict:
    ticker = tool_input.get("ticker", "PLTR").upper()

    if tool_name == "fetch_data":
        from src.fetch_data import fetch_all_data
        fetch_all_data(ticker, data_dir)
        return {"status": "success", "message": f"Downloaded latest data for {ticker}."}

    elif tool_name == "train_model":
        from src.main import train_model
        epochs = int(tool_input.get("epochs", 30))
        train_model(ticker, data_dir, checkpoints_dir, epochs=epochs)
        return {"status": "success", "message": f"Model trained for {ticker} ({epochs} epochs)."}

    elif tool_name == "predict":
        return _run_inference(ticker, data_dir, checkpoints_dir)

    elif tool_name == "get_sentiment":
        from src.sentiment import get_current_sentiment
        return get_current_sentiment(ticker)

    return {"error": f"Unknown tool: {tool_name}"}


def _run_agent_anthropic(user_message: str, api_key: str, data_dir: str, checkpoints_dir: str):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            text = "".join(block.text for block in response.content if hasattr(block, "text"))
            yield {"type": "response", "text": text}
            break

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    yield {"type": "tool_start", "tool": block.name, "input": block.input}
                    result = execute_tool(block.name, block.input, data_dir, checkpoints_dir)
                    yield {"type": "tool_end", "tool": block.name, "result": result}
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     json.dumps(result)
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            break


def _run_agent_openai(user_message: str, api_key: str, data_dir: str, checkpoints_dir: str):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["input_schema"]
            }
        }
        for t in TOOLS
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message}
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=openai_tools,
            max_tokens=2048
        )

        choice = response.choices[0]

        if choice.finish_reason == "stop":
            yield {"type": "response", "text": choice.message.content or ""}
            break

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                tool_name  = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)
                yield {"type": "tool_start", "tool": tool_name, "input": tool_input}
                result = execute_tool(tool_name, tool_input, data_dir, checkpoints_dir)
                yield {"type": "tool_end", "tool": tool_name, "result": result}
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      json.dumps(result)
                })
        else:
            break


def run_agent(
    user_message: str,
    api_key: str,
    data_dir: str = "data",
    checkpoints_dir: str = "checkpoints",
    openai_api_key: str = ""
):
    """
    Generator that runs the stock analysis agent.
    Tries Anthropic (Claude) first; falls back to OpenAI (GPT-4o) on
    rate-limit or credit errors if an OpenAI key is provided.

    Yields dicts:
      {"type": "tool_start", "tool": str, "input": dict}
      {"type": "tool_end",   "tool": str, "result": dict}
      {"type": "notice",     "text": str}
      {"type": "response",   "text": str}
    """
    try:
        yield from _run_agent_anthropic(user_message, api_key, data_dir, checkpoints_dir)
    except Exception as e:
        err = str(e).lower()
        is_recoverable = any(k in err for k in ["rate", "credit", "429", "too many", "balance"])
        if openai_api_key and is_recoverable:
            yield {"type": "notice", "text": "Anthropic unavailable — switching to OpenAI GPT-4o as backup."}
            yield from _run_agent_openai(user_message, openai_api_key, data_dir, checkpoints_dir)
        else:
            raise
