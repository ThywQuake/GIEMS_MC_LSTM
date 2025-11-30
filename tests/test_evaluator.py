import pytest
import numpy as np
import torch
import os
import tempfile
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Import components from the correct package structure
from giems_lstm.engine.evaluator import Evaluator, sMAPE, nse
from giems_lstm.model.lstm import LSTMNet


# --- Metric Tests ---


def test_smape_perfect_match():
    """Test sMAPE for perfect prediction."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    # Perfect match -> sMAPE = 0
    assert sMAPE(y_true, y_pred) == 0.0


def test_smape_deviation():
    """Test sMAPE for a known deviation."""
    y_true = np.array([10.0])
    y_pred = np.array([20.0])
    # Calculation: 100 * |10-20| / ((|10|+|20|)/2) = 100 * 10 / 15 = 66.666...
    assert pytest.approx(sMAPE(y_true, y_pred), 0.01) == 66.67


def test_nse_perfect_match():
    """Test Nash-Sutcliffe Efficiency (NSE) for perfect prediction."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    # NSE = 1 - (0 / Denominator) = 1
    assert nse(y_true, y_pred) == 1.0


def test_nse_mean_prediction():
    """Test NSE when prediction equals the mean of true values."""
    y_true = np.array([10.0, 20.0, 30.0])
    # Mean of y_true is 20.0
    y_pred = np.array([20.0, 20.0, 20.0])
    # Numerator = Denominator, so NSE = 1 - 1 = 0
    assert nse(y_true, y_pred) == 0.0


# --- Evaluator Class Tests ---


@pytest.fixture
def mock_setup():
    """
    Sets up a mock LSTM model, DataLoader, and Scaler for Evaluator testing.
    """
    # 1. Create Mock Data
    input_dim = 2
    output_dim = 1
    seq_length = 5
    batch_size = 4
    num_samples = 10

    # Create random features and targets
    # X: [Batch, Seq, Dim], y: [Batch, Output_Dim]
    X = torch.randn(num_samples, seq_length, input_dim)
    y = torch.randn(num_samples, output_dim)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # 2. Create Mock Model
    device = torch.device("cpu")
    # LSTMNet(input_dim, hidden_dim, output_dim, n_layers, device)
    model = LSTMNet(input_dim, 10, output_dim, 1, device)

    # 3. Create Mock Scaler
    scaler = MinMaxScaler()
    # Fit scaler on dummy data so it's initialized for inverse_transform
    scaler.fit(np.array([[0], [100]]))

    return model, dataloader, scaler, device


def test_evaluator_inference_and_saving(mock_setup):
    """
    Test the full run of the evaluator, ensuring inference occurs and results are saved correctly.
    """
    model, dataloader, scaler, device = mock_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        eval_folder = os.path.join(tmp_dir, "eval")
        model_folder = os.path.join(tmp_dir, "models")

        os.makedirs(model_folder, exist_ok=True)

        # Save a dummy model checkpoint
        model_path = os.path.join(model_folder, "0_0.pth")
        torch.save(model.state_dict(), model_path)

        evaluator = Evaluator(
            model=model,
            lat_idx=0,
            lon_idx=0,
            train_dataloader=dataloader,
            test_dataloader=dataloader,
            eval_folder=eval_folder,
            model_folder=model_folder,
            target_scaler=scaler,
            device=device,
            debug=True,  # Enable debug mode
        )

        evaluator.run()

        # 1. Check if result file exists
        result_file = os.path.join(eval_folder, "0_0.npy")
        assert os.path.exists(result_file)

        # 2. Load result and check structure
        results = np.load(result_file, allow_pickle=True).item()
        assert "Y_inv" in results
        assert "metrics" in results

        # Check Y_inv keys
        y_inv = results["Y_inv"]
        expected_keys = ["y_pred_train", "y_true_train", "y_pred_test", "y_true_test"]
        for key in expected_keys:
            assert key in y_inv
            assert isinstance(y_inv[key], np.ndarray)
            # Should not be empty (we passed data)
            assert y_inv[key].size > 0

        # Check metrics keys
        metrics = results["metrics"]
        assert "train" in metrics
        assert "test" in metrics
        assert "R2" in metrics["train"]
        assert "RMSE" in metrics["train"]


def test_evaluator_skips_if_exists(mock_setup, caplog):
    """
    Test that the evaluator skips the run if the result file already exists
    and debug mode is False.
    """
    model, dataloader, scaler, device = mock_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        eval_folder = os.path.join(tmp_dir, "eval")
        model_folder = os.path.join(tmp_dir, "models")
        os.makedirs(eval_folder, exist_ok=True)
        os.makedirs(model_folder, exist_ok=True)

        # Create a fake existing result file
        result_file = os.path.join(eval_folder, "1_1.npy")
        np.save(result_file, {"dummy": 1})

        evaluator = Evaluator(
            model=model,
            lat_idx=1,
            lon_idx=1,
            train_dataloader=dataloader,
            test_dataloader=dataloader,
            eval_folder=eval_folder,
            model_folder=model_folder,
            target_scaler=scaler,
            device=device,
            debug=False,  # debug=False enables the check
        )

        # Run evaluator and capture logs
        with caplog.at_level("INFO"):
            evaluator.run()

        # Check logs for skip message
        assert "already exists" in caplog.text


def test_evaluator_missing_model(mock_setup, caplog):
    """
    Test that the evaluator handles the case where the model checkpoint is missing.
    """
    model, dataloader, scaler, device = mock_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        eval_folder = os.path.join(tmp_dir, "eval")
        model_folder = os.path.join(tmp_dir, "models")
        # model_folder is created but intentionally left empty, so the .pth file is missing

        evaluator = Evaluator(
            model=model,
            lat_idx=2,
            lon_idx=2,
            train_dataloader=dataloader,
            test_dataloader=dataloader,
            eval_folder=eval_folder,
            model_folder=model_folder,
            target_scaler=scaler,
            device=device,
            debug=False,
        )

        with caplog.at_level("WARNING"):
            evaluator.run()

        # Check logs for warning message
        assert "not found" in caplog.text
        # Result file should NOT be created
        result_file = os.path.join(eval_folder, "2_2.npy")
        assert not os.path.exists(result_file)
