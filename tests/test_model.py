import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Import all necessary model components and the dataset
# Assuming 'giems_lstm' is the top-level package accessible from the test directory
from giems_lstm.model import LSTMNet, GRUNet
from giems_lstm.model.lstmkan import LSTMNetKAN
from giems_lstm.model.grukan import GRUNetKAN
from giems_lstm.model.kanlinear import KANLinear
from giems_lstm.data.dataset import WetlandDataset


# ==========================================
# 1. KANLinear Layer Tests
# ==========================================


def test_kan_linear_shape():
    """Test the output shape of the KANLinear layer."""
    batch_size = 16
    in_features = 32
    out_features = 8

    model = KANLinear(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    out = model(x)
    assert out.shape == (
        batch_size,
        out_features,
    ), "KANLinear output shape is incorrect"


def test_kan_linear_update_grid():
    """Test the grid update functionality runs without error."""
    model = KANLinear(in_features=10, out_features=5)
    x = torch.randn(32, 10)
    try:
        model.update_grid(x)
    except Exception as e:
        pytest.fail(f"KANLinear.update_grid failed: {e}")


def test_kan_regularization():
    """Test the regularization loss calculation."""
    model = KANLinear(in_features=10, out_features=5)
    loss = model.regularization_loss()
    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0, "Regularization loss must be non-negative"


# ==========================================
# 2. Model Forward Pass Tests
# ==========================================


@pytest.mark.parametrize("model_class", [LSTMNet, GRUNet, LSTMNetKAN, GRUNetKAN])
def test_models_forward_shape(model_class):
    """
    Test that all models produce the correct output shape and range after a forward pass.
    """
    batch_size = 8
    seq_length = 12
    input_dim = 10
    hidden_dim = 16
    output_dim = 1
    n_layers = 2
    device = torch.device("cpu")

    model = model_class(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n_layers=n_layers,
        device=device,
    )

    # Input tensor shape: [Batch, Seq, Feature]
    x = torch.randn(batch_size, seq_length, input_dim).to(device)

    # Initialize hidden state
    h = model.init_hidden(batch_size)

    # Forward pass
    out, h_out = model(x, h)

    # Expected output shape: (Batch, output_dim) as the last sequence step is extracted
    assert out.shape == (
        batch_size,
        output_dim,
    ), f"{model_class.__name__} output shape is incorrect"

    # Check output range (due to Sigmoid activation on the output layer)
    assert (
        out.min() >= 0 and out.max() <= 1
    ), f"{model_class.__name__} output should be clipped within [0, 1] (Sigmoid)"


@pytest.mark.parametrize("model_class", [LSTMNet, GRUNet])
def test_models_backward(model_class):
    """
    Test that the backpropagation runs successfully and generates gradients
    for all parameters in non-KAN models.
    """
    device = torch.device("cpu")
    model = model_class(
        input_dim=5, hidden_dim=10, output_dim=1, n_layers=1, device=device
    )

    # Input data
    x = torch.randn(4, 10, 5)  # [Batch, Seq, Feature]
    h = model.init_hidden(4)
    target = torch.rand(4, 1)  # [Batch, Output_Dim]

    out, _ = model(x, h)
    loss = nn.MSELoss()(out, target)
    loss.backward()

    # Check for gradient existence
    for name, param in model.named_parameters():
        # Gradients should be computed for all trainable parameters
        assert param.grad is not None, f"Parameter {name} did not receive a gradient"


# ==========================================
# 3. Dataset Tests (Basic Checks)
# ==========================================


@pytest.fixture
def mock_data():
    """Generate synthetic time series data."""
    num_samples = 100
    features = np.random.rand(num_samples, 5).astype(np.float32)  # [T, D]
    target = np.random.rand(num_samples, 1).astype(np.float32)  # [T, 1]
    dates = pd.date_range(start="2020-01-01", periods=num_samples).tolist()
    return features, target, dates


def test_wetland_dataset_len(mock_data):
    """Test if the dataset length calculation is correct in training mode."""
    features, target, dates = mock_data
    seq_length = 10

    dataset = WetlandDataset(
        features_scaled=features,
        dates=dates,
        target_scaled=target,
        seq_length=seq_length,
    )

    # Expected number of windows = Total length - Sequence length + 1
    expected_len = len(features) - seq_length + 1
    assert len(dataset) == expected_len, "Dataset length calculation is incorrect"


def test_wetland_dataset_item(mock_data):
    """Test the structure and shape of a single item retrieved in training mode."""
    features, target, dates = mock_data
    seq_length = 5
    dataset = WetlandDataset(
        features_scaled=features,
        dates=dates,
        target_scaled=target,
        seq_length=seq_length,
    )

    # Get the first sample
    item = dataset[0]
    # Check return format (features_window, target_window)
    assert len(item) == 2

    x_window, y_window = item
    assert x_window.shape == (seq_length, 5)
    # Target (y) should be the value at the last time step of the window
    assert y_window.shape == (1,)


def test_dataset_predict_mode(mock_data):
    """Test the item retrieval in prediction mode (should only return features)."""
    features, _, dates = mock_data
    dataset = WetlandDataset(
        features_scaled=features,
        dates=dates,
        target_scaled=None,
        seq_length=10,
        predict_mode=True,
    )

    item = dataset[0]
    # In predict mode, it returns only the feature_window (np.ndarray)
    assert isinstance(item, np.ndarray)
    assert item.shape == (10, 5)


def test_nan_handling():
    """
    Test that windows with NaN targets (at the window end) are skipped in training mode.
    """
    features = np.random.rand(20, 2).astype(np.float32)
    target = np.random.rand(20, 1).astype(np.float32)

    # Set target at index 10 to NaN (this should affect all windows ending at 10)
    target[10] = np.nan
    dates = pd.date_range("20200101", periods=20).tolist()

    seq_length = 5
    dataset = WetlandDataset(
        features_scaled=features,
        dates=dates,
        target_scaled=target,
        seq_length=seq_length,
    )

    # Theoretical total windows = 20 - 5 + 1 = 16
    # Target indices for windows are 4, 5, ..., 19
    # The target index 10 is valid (>= 4).
    # Since one window is skipped, actual length should be 16 - 1 = 15
    total_windows = len(features) - seq_length + 1

    assert len(dataset) == total_windows - 1, "NaN target window was not skipped"
