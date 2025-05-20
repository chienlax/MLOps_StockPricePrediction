# tests/models/test_model_definitions.py
import pytest
import torch

# Assuming model_definitions.py is in src/models and src is in PYTHONPATH
# If not, adjust sys.path or use relative imports if tests are structured as a package
from models.model_definitions import StockLSTM, StockLSTMWithAttention, StockLSTMWithCrossStockAttention

# --- Fixtures for Model Parameters ---
@pytest.fixture
def model_params():
    return {
        "num_stocks": 2,
        "num_features": 5,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout_rate": 0.2,
        "batch_size": 4,
        "seq_len": 10
    }

# --- Helper to create dummy input tensor ---
def create_dummy_input(batch_size, seq_len, num_stocks, num_features):
    return torch.randn(batch_size, seq_len, num_stocks, num_features)

# --- Tests for StockLSTM ---
class TestStockLSTM:
    def test_lstm_instantiation(self, model_params):
        model = StockLSTM(
            num_stocks=model_params["num_stocks"],
            num_features=model_params["num_features"],
            hidden_size=model_params["hidden_size"],
            num_layers=model_params["num_layers"],
            dropout_rate=model_params["dropout_rate"]
        )
        assert isinstance(model, StockLSTM)
        assert model.num_stocks == model_params["num_stocks"]

    def test_lstm_forward_pass_shape(self, model_params):
        model = StockLSTM(
            num_stocks=model_params["num_stocks"],
            num_features=model_params["num_features"],
            hidden_size=model_params["hidden_size"],
            num_layers=model_params["num_layers"]
        )
        dummy_input = create_dummy_input(
            model_params["batch_size"], model_params["seq_len"],
            model_params["num_stocks"], model_params["num_features"]
        )
        output = model(dummy_input)
        # Expected output shape: (batch_size, 1, num_stocks)
        assert output.shape == (model_params["batch_size"], 1, model_params["num_stocks"])
        assert output.dtype == torch.float32

# --- Tests for StockLSTMWithAttention ---
class TestStockLSTMWithAttention:
    def test_lstm_attention_instantiation(self, model_params):
        model = StockLSTMWithAttention(
            num_stocks=model_params["num_stocks"],
            num_features=model_params["num_features"],
            hidden_size=model_params["hidden_size"],
            num_layers=model_params["num_layers"],
            dropout_rate=model_params["dropout_rate"]
        )
        assert isinstance(model, StockLSTMWithAttention)

    def test_lstm_attention_forward_pass_shape(self, model_params):
        model = StockLSTMWithAttention(
            num_stocks=model_params["num_stocks"],
            num_features=model_params["num_features"],
            hidden_size=model_params["hidden_size"],
            num_layers=model_params["num_layers"]
        )
        dummy_input = create_dummy_input(
            model_params["batch_size"], model_params["seq_len"],
            model_params["num_stocks"], model_params["num_features"]
        )
        output = model(dummy_input)
        assert output.shape == (model_params["batch_size"], 1, model_params["num_stocks"])
        assert output.dtype == torch.float32

# --- Tests for StockLSTMWithCrossStockAttention ---
class TestStockLSTMWithCrossStockAttention:
    def test_lstm_cross_attention_instantiation(self, model_params):
        model = StockLSTMWithCrossStockAttention(
            num_stocks=model_params["num_stocks"],
            num_features=model_params["num_features"],
            hidden_size=model_params["hidden_size"],
            num_layers=model_params["num_layers"],
            dropout_rate=model_params["dropout_rate"]
        )
        assert isinstance(model, StockLSTMWithCrossStockAttention)
        # Check if the correct number of stock-specific LSTMs are created
        assert len(model.stock_lstms) == model_params["num_stocks"]


    def test_lstm_cross_attention_forward_pass_shape(self, model_params):
        model = StockLSTMWithCrossStockAttention(
            num_stocks=model_params["num_stocks"],
            num_features=model_params["num_features"],
            hidden_size=model_params["hidden_size"],
            num_layers=model_params["num_layers"]
        )
        dummy_input = create_dummy_input(
            model_params["batch_size"], model_params["seq_len"],
            model_params["num_stocks"], model_params["num_features"]
        )
        output = model(dummy_input)
        assert output.shape == (model_params["batch_size"], 1, model_params["num_stocks"])
        assert output.dtype == torch.float32