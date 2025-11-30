import pytest
import numpy as np
import torch
import os
import tempfile
from unittest.mock import MagicMock, patch
from sklearn.preprocessing import MinMaxScaler

# Import components from the correct package structure
from giems_lstm.engine.predictor import Predictor
from giems_lstm.model.lstm import LSTMNet
from giems_lstm.data.dataset import WetlandDataset


@pytest.fixture
def mock_setup():
    """
    Sets up common mocks for predictor tests: Dataset, Scaler, and Model.
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

    # Calculate the total time steps based on sliding window
    # Total length = num_windows + seq_length - 1
    total_len = num_windows + seq_length - 1  # 10 + 5 - 1 = 14
    dates = [f"2000-01-{i+1:02d}" for i in range(total_len)]

    mock_dataset = MagicMock(spec=WetlandDataset)
    mock_dataset.windows = windows
    mock_dataset.dates = dates
    mock_dataset.seq_length = seq_length

    # Mock __len__ to return the number of windows
    mock_dataset.__len__.return_value = num_windows

    # Mock __getitem__ for DataLoader to work
    mock_dataset.__getitem__.side_effect = windows

    # 2. Mock Scaler
    scaler = MinMaxScaler()
    # Fit scaler on dummy data so it's initialized for inverse_transform
    scaler.fit(np.array([[0], [100]]))

    # 3. Mock Model
    device = torch.device("cpu")
    model = LSTMNet(input_dim, 10, 1, 1, device)
    model.eval()

    return mock_dataset, scaler, model, device


def test_predictor_inference(mock_setup):
    """
    Test that inference runs correctly and populates pred_scaled array.
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

        # Check if pred_scaled is populated with one prediction per window
        assert len(predictor.pred_scaled) == len(dataset.windows)
        assert isinstance(predictor.pred_scaled, np.ndarray)


def test_predictor_post_processing(mock_setup):
    """
    Test inverse scaling and thresholding logic (0 to 100 range).
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

        # Manually set pred_scaled with known values (scaled 0-1 range)
        # 0.5 -> 50 in original (0-100)
        # -0.1 should be thresholded to 0 in scaled space, resulting in 0
        # 1.1 should be capped at 1.0 in scaled space, resulting in 100
        predictor.pred_scaled = np.array([0.5, -0.1, 0.0, 1.0, 1.1])

        predictor._post_process_predictions()

        result = predictor.pred_final

        # Check thresholding and capping
        assert pytest.approx(result[1]) == 0.0  # -0.1 -> 0
        assert (
            pytest.approx(result[4]) == 100.0
        )  # 1.1 -> 100 (due to internal clipping before inverse transform)

        # Check inverse transform approximate values
        assert pytest.approx(result[0]) == 50.0
        assert pytest.approx(result[3]) == 100.0


def test_predictor_backfill_logic(mock_setup):
    """
    Test that backfill uses AutoReg to prepend the correct number of elements.
    We mock AutoReg to isolate the backfill logic.
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

        # Set some dummy final predictions (num_windows = 10)
        predictor.pred_final = np.linspace(10, 20, 10)

        # Dataset seq_length is 5 (from fixture)
        # Backfill length should be seq_length - 1 = 4

        # Mock AutoReg and its methods
        with patch("giems_lstm.engine.predictor.AutoReg") as MockAutoReg:
            mock_ar_instance = MockAutoReg.return_value
            mock_fit_result = mock_ar_instance.fit.return_value

            # Mock predict to return 4 backcast values
            backcast_values = np.array([1.0, 2.0, 3.0, 4.0])
            mock_fit_result.predict.return_value = backcast_values

            predictor._backfill()

            # Check length: 4 (backcast) + 10 (original) = 14
            expected_length = len(predictor.pred_final)
            assert expected_length == 14

            # Check if backcast was prepended correctly
            assert np.allclose(predictor.pred_final[:4], backcast_values)
            # Check if original predictions are still in place
            assert np.allclose(predictor.pred_final[4:], np.linspace(10, 20, 10))


def test_predictor_save_results(mock_setup):
    """
    Test saving results to disk in the expected format.
    """
    dataset, scaler, model, device = mock_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "pred_0_0.npy")

        predictor = Predictor(
            lat_idx=0,
            lon_idx=0,
            dataset=dataset,
            target_scaler=scaler,
            model=model,
            save_path=save_path,
            device=device,
        )

        # Manually populate pred_final to match dataset dates length (14)
        predictor.pred_final = np.linspace(50, 60, 14)

        predictor._save_results()

        assert os.path.exists(save_path)

        data = np.load(save_path, allow_pickle=True).item()
        assert "prediction" in data
        assert "date" in data
        # Check lengths match the total time steps
        assert len(data["prediction"]) == 14
        assert len(data["date"]) == 14
        assert np.allclose(data["prediction"], predictor.pred_final)


def test_predictor_skip_existing(mock_setup, caplog):
    """
    Test that predictor skips the run if the result file exists and debug=False.
    """
    dataset, scaler, model, device = mock_setup

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = os.path.join(tmp_dir, "existing.npy")
        # Create fake existing result file
        with open(save_path, "w") as f:
            f.write("test data")

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

        # Check logs for skip message
        assert "already exists" in caplog.text
        # Ensure the model was not evaluated (e.g., inference not called)
        # This is implicitly checked since the core functions aren't called if skipped.
