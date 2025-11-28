import pytest
import numpy as np
import torch
import os
import tempfile
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import MinMaxScaler
from src.giems_lstm.engine.predictor import Predictor
from src.giems_lstm.model.lstm import LSTMNet
from src.giems_lstm.data.dataset import WetlandDataset


@pytest.fixture
def mock_setup():
    """
    Sets up common mocks for predictor tests.
    """
    # 1. Mock Dataset
    seq_length = 5
    num_windows = 10
    input_dim = 2

    # Create fake windows [Num_Windows, Seq_Length, Input_Dim]
    windows = [
        np.random.rand(seq_length, input_dim).astype(np.float32)
        for _ in range(num_windows)
    ]

    # Create fake dates (Total length = num_windows + seq_length - 1)
    # Because windows are created by sliding: length = T - seq + 1 => T = length + seq - 1
    total_len = num_windows + seq_length - 1
    dates = [f"2000-01-{i+1:02d}" for i in range(total_len)]

    mock_dataset = MagicMock(spec=WetlandDataset)
    mock_dataset.windows = windows
    mock_dataset.dates = dates
    mock_dataset.seq_length = seq_length

    # 2. Mock Scaler
    scaler = MinMaxScaler()
    scaler.fit(np.array([[0], [100]]))  # Scale 0-100 to 0-1

    # 3. Mock Model
    device = torch.device("cpu")
    model = LSTMNet(input_dim, 10, 1, 1, device)
    model.eval()

    return mock_dataset, scaler, model, device


def test_predictor_inference(mock_setup):
    """
    Test that inference runs and populates pred_scaled.
    """
    dataset, scaler, model, device = mock_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "pred.npy")

        predictor = Predictor(
            lat_idx=0,
            lon_idx=0,
            dataset=dataset,
            target_scaler=scaler,
            model=model,
            save_path=save_path,
            device=device,
            batch_size=2,
        )

        predictor._run_inference()

        # Check if pred_scaled is populated
        assert len(predictor.pred_scaled) == len(dataset.windows)
        assert isinstance(predictor.pred_scaled, np.ndarray)


def test_predictor_post_processing(mock_setup):
    """
    Test inverse scaling and thresholding.
    """
    dataset, scaler, model, device = mock_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "pred.npy")

        predictor = Predictor(
            lat_idx=0,
            lon_idx=0,
            dataset=dataset,
            target_scaler=scaler,
            model=model,
            save_path=save_path,
            device=device,
        )

        # Manually set pred_scaled with known values
        # 0.5 in scaled (0-1) -> 50 in original (0-100)
        # -0.1 should be thresholded to 0
        predictor.pred_scaled = np.array([0.5, -0.1, 0.0, 1.0])

        predictor._post_process_predictions()

        result = predictor.pred_final

        # Check thresholding
        assert result[1] == 0.0  # -0.1 -> thresholded

        # Check inverse transform approximate values
        assert pytest.approx(result[0], 0.1) == 50.0
        assert pytest.approx(result[3], 0.1) == 100.0


def test_predictor_backfill_logic(mock_setup):
    """
    Test that backfill adds the correct number of elements.
    We mock AutoReg to avoid complex dependency behavior in unit tests.
    """
    dataset, scaler, model, device = mock_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "pred.npy")

        predictor = Predictor(
            lat_idx=0,
            lon_idx=0,
            dataset=dataset,
            target_scaler=scaler,
            model=model,
            save_path=save_path,
            device=device,
        )

        # Set some dummy final predictions
        predictor.pred_final = np.linspace(10, 20, 10)  # 10 values

        # Dataset seq_length is 5 (from fixture)
        # Backfill length should be seq_length - 1 = 4

        # Mock AutoReg
        with patch("src.giems_lstm.engine.predictor.AutoReg") as MockAutoReg:
            mock_ar_instance = MockAutoReg.return_value
            mock_fit_result = mock_ar_instance.fit.return_value

            # Mock predict to return 4 values (backcast)
            mock_fit_result.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0])

            predictor._backfill()

            # Check length: 4 (backcast) + 10 (original) = 14
            assert len(predictor.pred_final) == 14

            # Check if backcast was prepended
            assert np.allclose(predictor.pred_final[:4], [1.0, 2.0, 3.0, 4.0])


def test_predictor_save_results(mock_setup):
    """
    Test saving to disk.
    """
    dataset, scaler, model, device = mock_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "pred.npy")

        predictor = Predictor(
            lat_idx=0,
            lon_idx=0,
            dataset=dataset,
            target_scaler=scaler,
            model=model,
            save_path=save_path,
            device=device,
        )

        # Manually populate pred_final to match dataset dates length
        # dataset.dates length from fixture is num_windows (10) + seq_len (5) - 1 = 14
        predictor.pred_final = np.zeros(14)

        predictor._save_results()

        assert os.path.exists(save_path)

        data = np.load(save_path, allow_pickle=True).item()
        assert "prediction" in data
        assert "date" in data
        assert len(data["prediction"]) == 14


def test_predictor_skip_existing(mock_setup, caplog):
    """Test that predictor skips if file exists and not debug."""
    dataset, scaler, model, device = mock_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "existing.npy")
        # Create file
        with open(save_path, "w") as f:
            f.write("test")

        predictor = Predictor(
            lat_idx=0,
            lon_idx=0,
            dataset=dataset,
            target_scaler=scaler,
            model=model,
            save_path=save_path,
            device=device,
            debug=False,
        )

        with caplog.at_level("WARNING"):
            predictor.run()

        assert "already exists" in caplog.text
