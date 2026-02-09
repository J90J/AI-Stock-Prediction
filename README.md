# AI Stock Prediction with LSTM + Sentiment

## Introduction

This project is an **AI-powered stock prediction system** that combines traditional Deep Learning with modern Sentiment Analysis. It is designed to work with **any stock ticker** (e.g., PLTR, AAPL, NVDA), allowing users to dynamically train models and generate predictions for their chosen companies.

The system uses an LSTM neural network to analyze 60 days of price/volume history to determine the technical trend, and then cross-references that with real-time news sentiment (analyzed via VADER) to generate a final trading recommendation.

## Features

* **Universal Stock Support**: Enter any ticker symbol to download data and generate predictions.
* **Automated Retraining**: The system automatically retrains the AI model on the new stock's data to ensure accuracy.
* **Hybrid Inference**: Combines technical analysis (LSTM) with news sentiment (VADER).
* **Interactive UI**: A Streamlit-based web interface for easy interaction.

## Directory Structure

```
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── src/
│   ├── main.py         # Training script (callable)
│   ├── model.py        # LSTM model definition
│   ├── utils.py        # Helper functions (data loading, indicators)
│   ├── fetch_data.py   # Data downloading logic
│   ├── sentiment.py    # VADER sentiment analysis
├── data/               # CSV data files (auto-generated)
├── checkpoints/        # Saved models and scalers
├── demo/
│   ├── demo.py         # Legacy demo script
└── results/            # Generated plots and predictions
```

## Setup Instructions

1. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Run the App**:

    ```bash
    streamlit run streamlit_app.py
    ```

3. **Using the App**:
    1. **Enter Ticker**: Type a stock symbol (e.g., `NVDA`) in the sidebar.
    2. **Fetch & Train**: Click **"Fetch Data & Retrain Model"**.
    3. **Wait**: The app will download latest data and train a fresh model (~30-60s).
    4. **Analyze**: View the predicted price, technical signal, and news sentiment.

## Training (Manual)

The app handles training automatically, but you can also run it manually:

```bash
# Fetch data for a specific ticker
python -c "from src.fetch_data import fetch_all_data; fetch_all_data('AAPL')"

# Train model for that ticker
python src/main.py --ticker AAPL --epochs 50
```

## Model Architecture

* **Input**: 60 days of OHLCV data + Technical Indicators (RSI, MACD, BB, ROC, ATR, Stochastic Oscillator) + NASDAQ Index correlations.
* **Model**: Multi-layer LSTM with 2 heads:
    1. **Regression Head**: Predicts the next day's return.
    2. **Classification Head**: Predicts the probability of the price moving UP.

## Acknowledgments

* Data sourced from Yahoo Finance.
* Built with PyTorch.
