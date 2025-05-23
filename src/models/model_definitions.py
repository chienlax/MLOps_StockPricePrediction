"""Module containing PyTorch neural network model definitions for stock price prediction."""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Basic LSTM model for multi-stock price prediction.

    Takes sequence data for multiple stocks and predicts next price for all stocks.
    """

    def __init__(
        self, num_stocks, num_features, hidden_size, num_layers, dropout_rate=0.2
    ):
        """
        Initialize the LSTM model.

        Args:
            num_stocks: Number of stocks to predict
            num_features: Number of features per stock
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout_rate: Dropout probability for regularization
        """
        super(StockLSTM, self).__init__()
        self.num_stocks = num_stocks
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_stocks * num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_stocks)  # Output for all stocks

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_stocks, num_features)

        Returns:
            Predictions tensor of shape (batch_size, 1, num_stocks)
        """
        batch_size, seq_len, n_stocks, n_features = x.size()
        # Reshape input for LSTM: (batch, seq_len, num_stocks * features)
        x = x.reshape(batch_size, seq_len, n_stocks * n_features)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take only the last timestep's output
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)

        # Fully connected layer
        out = self.fc(lstm_out)

        # Reshape to (batch_size, 1, num_stocks) to match target shape
        return out.view(batch_size, 1, self.num_stocks)


class StockLSTMWithAttention(nn.Module):
    """
    LSTM model with self-attention for multi-stock price prediction.

    Enhances the basic LSTM by incorporating self-attention mechanism
    to capture temporal dependencies.
    """

    def __init__(
        self, num_stocks, num_features, hidden_size, num_layers, dropout_rate=0.2
    ):
        """
        Initialize the LSTM with attention model.

        Args:
            num_stocks: Number of stocks to predict
            num_features: Number of features per stock
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout_rate: Dropout probability for regularization
        """
        super(StockLSTMWithAttention, self).__init__()
        self.num_stocks = num_stocks
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=num_stocks * num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
        )

        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=4, dropout=dropout_rate
        )

        # Dropout and fully connected layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, num_stocks)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_stocks, num_features)

        Returns:
            Predictions tensor of shape (batch_size, 1, num_stocks)
        """
        batch_size, seq_len, n_stocks, n_features = x.size()
        # Reshape input for LSTM: (batch, seq_len, num_stocks * features)
        x = x.reshape(batch_size, seq_len, n_stocks * n_features)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Apply self-attention
        # Reshape for attention: (seq_len, batch, hidden)
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(
            lstm_out_transposed, lstm_out_transposed, lstm_out_transposed
        )

        # Reshape back: (batch, seq_len, hidden)
        attn_out = attn_out.transpose(0, 1)

        # Take only the last timestep's output after attention
        final_out = attn_out[:, -1, :]

        # Fully connected layers
        final_out = self.dropout(final_out)
        final_out = self.fc1(final_out)
        final_out = self.relu(final_out)
        final_out = self.fc2(final_out)

        # Reshape to (batch_size, 1, num_stocks) to match target shape
        return final_out.view(batch_size, 1, self.num_stocks)


class StockLSTMWithCrossStockAttention(nn.Module):
    """
    LSTM model with cross-stock attention for multi-stock price prediction.

    Uses separate LSTMs for each stock and applies attention across stocks
    to capture inter-stock relationships.
    """

    def __init__(
        self, num_stocks, num_features, hidden_size, num_layers, dropout_rate=0.2
    ):
        """
        Initialize the LSTM with cross-stock attention model.

        Args:
            num_stocks: Number of stocks to predict
            num_features: Number of features per stock
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout_rate: Dropout probability for regularization
        """
        super(StockLSTMWithCrossStockAttention, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Separate LSTM for each stock
        self.stock_lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=num_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout_rate if num_layers > 1 else 0,
                )
                for _ in range(num_stocks)
            ]
        )

        # Cross-stock attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=4, dropout=dropout_rate
        )

        # Dropout and output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)  # One output per stock

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, num_stocks, num_features)

        Returns:
            Predictions tensor of shape (batch_size, 1, num_stocks)
        """
        batch_size, seq_len, n_stocks, n_features = x.size()

        # Process each stock separately with its own LSTM
        stock_outputs = []
        for stock_idx in range(self.num_stocks):
            # Extract data for this stock
            stock_data = x[:, :, stock_idx, :]  # (batch, seq, features)

            # Run through stock-specific LSTM
            lstm_out, _ = self.stock_lstms[stock_idx](stock_data)

            # Take the last output
            last_out = lstm_out[:, -1, :]  # (batch, hidden)
            stock_outputs.append(last_out.unsqueeze(0))  # (1, batch, hidden)

        # Concatenate all stock outputs
        stock_outputs = torch.cat(stock_outputs, dim=0)  # (n_stocks, batch, hidden)

        # Apply cross-stock attention
        attn_out, _ = self.cross_attention(stock_outputs, stock_outputs, stock_outputs)

        # Process each stock's output after attention
        final_outputs = []
        for stock_idx in range(self.num_stocks):
            stock_repr = attn_out[stock_idx]  # (batch, hidden)
            stock_repr = self.dropout(stock_repr)
            stock_out = self.fc(stock_repr)  # (batch, 1)
            final_outputs.append(stock_out)

        # Stack and reshape to (batch, 1, n_stocks)
        final_output = torch.stack(final_outputs, dim=2)  # (batch, 1, n_stocks)
        return final_output
