# AI Stock Analysis Agent — LSTM + Sentiment + Claude

## What is this?

This project is an **AI agent** for stock market analysis. It combines a custom LSTM neural network, real-time news sentiment analysis, and Claude (Anthropic's AI) as the orchestrating brain.

Unlike a traditional ML dashboard with hardcoded buttons, this is a **chat-based agent**: you ask it a question, and it autonomously decides which tools to call, in what order, to answer you.

## How the Agent Works

```text
User: "Should I buy NVDA today?"
        ↓
Agent (Claude) decides what to do
        ↓
  fetch_data  →  downloads latest 5y of price history
  train_model →  trains the LSTM on that data
  predict     →  runs the technical price-direction forecast
  sentiment   →  analyzes recent news headlines (VADER)
        ↓
Agent synthesizes all signals → final recommendation
```

The agent has four tools built on the existing ML pipeline:

| Tool | What it does |
| --- | --- |
| `fetch_data` | Downloads stock + NASDAQ data via Yahoo Finance |
| `train_model` | Trains the LSTM neural network (regression + classification heads) |
| `predict` | Runs inference: predicted return %, direction (UP/DOWN), confidence |
| `get_sentiment` | VADER sentiment on latest news headlines |

## Model Architecture

- **Input**: 60 days of OHLCV + technical indicators (RSI, MACD, Bollinger Bands, ROC, ATR, Stochastic Oscillator) + NASDAQ correlations — 17 features total
- **Model**: 2-layer stacked LSTM (hidden size 64) with two output heads:
  1. **Regression head** — predicts next-day return (%)
  2. **Classification head** — predicts probability of price going UP

## Directory Structure

```text
├── app.py               # Streamlit chat UI (entry point)
├── streamlit_app.py     # Mirror of app.py for Streamlit Cloud
├── requirements.txt     # Python dependencies
├── src/
│   ├── agent.py         # Agent core: tools, loop, Claude integration
│   ├── model.py         # LSTM model definition
│   ├── utils.py         # Feature engineering helpers
│   ├── fetch_data.py    # Yahoo Finance data download
│   ├── sentiment.py     # VADER sentiment analysis
│   └── main.py          # Standalone training script
├── data/                # CSV data files (auto-generated)
├── checkpoints/         # Saved models and scalers (auto-generated)
└── demo/
    └── demo.py          # Legacy demo script
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

**Option A — environment variable:**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Option B — Streamlit secrets** (for Streamlit Cloud deployment):

Create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

**Option C** — paste the key directly into the sidebar when the app starts.

### 3. Run the app

```bash
streamlit run app.py
```

### 4. Start chatting

Try prompts like:

- *"Should I buy PLTR today?"*
- *"What is your outlook on AAPL?"*
- *"Analyze TSLA and give me a recommendation"*
- *"Retrain the model for MSFT with 50 epochs"*

## Manual Training (optional)

The agent handles everything automatically, but you can also run the pipeline manually:

```bash
# Download data
python -c "from src.fetch_data import fetch_all_data; fetch_all_data('AAPL')"

# Train model
python src/main.py --ticker AAPL --epochs 50
```

## Disclaimer

Predictions are probabilistic and for educational purposes only. This is not financial advice.

## Acknowledgments

- Data: Yahoo Finance via `yfinance`
- Model: PyTorch LSTM
- Sentiment: VADER (`vaderSentiment`)
- Agent brain: Claude (Anthropic)
