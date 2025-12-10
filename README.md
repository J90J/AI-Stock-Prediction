# Palantir Stock Prediction with LSTM

## Project Overview
This project uses a Long Short-Term Memory (LSTM) neural network to predict the next-day closing price of Palantir (PLTR) stock. It leverages historical price data and technical indicators (RSI, MACD, Bollinger Bands, etc.) along with market context from the NASDAQ index.

The project is structured for reproducibility and ease of use, separating data processing, model definition, training, and inference.

## Directory Structure
```
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── src/
│   ├── main.py         # Training script
│   ├── model.py        # LSTM model definition
│   ├── utils.py        # Helper functions (data loading, indicators)
├── data/               # Place your CSV data files here
├── checkpoints/        # Saved models and scalers
├── demo/
│   ├── demo.py         # Demo script for inference
└── results/            # Generated plots and predictions
```

## Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    *   This project requires two CSV files:
        *   `PLTR (11).csv` (Palantir historical data)
        *   `^IXIC (10).csv` (NASDAQ historical data)
    *   Place these files inside the `data/` directory.

## How to Run

### 1. Training the Model
To train the model from scratch:
```bash
python src/main.py
```
This will:
*   Load data from `data/`.
*   Process features and split into train/test sets.
*   Train the LSTM model for 50 epochs.
*   Save the trained model (`palantir_lstm.pth`) and scalers to `checkpoints/`.

### 2. Running the Demo
Once the model is trained (or if you have downloaded pre-trained checkpoints):
```bash
python demo/demo.py
```
This will:
*   Load the model and latest data.
*   Predict the next day's closing price and direction.
*   Generate a plot in `results/prediction_plot.png`.

## Reproducibility
*   **Hyperparameters**:
    *   Lookback Window: 60 days
    *   Hidden Size: 64
    *   Layers: 2
    *   Dropout: 0.2
    *   Learning Rate: 0.001
    *   Epochs: 50
    *   Batch Size: 32
*   **Random Seed**: While PyTorch seeds are not explicitly fixed in this version, the training process is stable. For strict determinism, set torch/numpy seeds at the start of `src/main.py`.

## Model Information
*   **Architecture**: Multi-layer LSTM with two heads:
    1.  **Regression Head**: Predicts the continuous return.
    2.  **Classification Head**: Predicts the probability of the price moving UP.
*   **Features**: OHLCV data + Technical Indicators (RSI, MACD, BB, ROC, ATR, Stochastic Oscillator) + NASDAQ Index correlations.

## Acknowledgments
*   Data sourced from Yahoo Finance.
*   Built with PyTorch.
