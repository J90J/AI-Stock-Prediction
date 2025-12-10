import torch
import torch.nn as nn

class PalantirLSTM(nn.Module):
    def __init__(self, input_size=18, hidden_size=64, num_layers=2, dropout=0.2):
        """
        input_size : number of input features per timestep
        hidden_size: LSTM hidden dimension
        num_layers : stacked LSTM layers
        dropout    : dropout between LSTM layers
        """
        super().__init__()

        self.hidden_size = hidden_size

        # ------ Shared LSTM encoder ------
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # ------ Regression head (next-day closing price/return) ------
        # Note: In the original code, this output 1 scaled price. 
        # In the modified data prep, it outputs return. Ideally the architecture handles scalar output.
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)   
        )

        # ------ Classification head (up/down) ------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]  # final hidden layer output

        price_pred = self.regressor(h_last)
        updown_prob = self.classifier(h_last)

        return price_pred, updown_prob
