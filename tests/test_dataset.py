import pytest
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.giems_lstm.data.dataset import WetlandDataset, wetland_dataloader


@pytest.fixture
def synthetic_data():
    """
    Creates synthetic scaled features, targets, and dates for testing.
    """
    seq_length = 5
    num_samples = 50
    num_features = 3

    # Generate random features
    features = np.random.rand(num_samples, num_features).astype(np.float32)

    # Generate targets (some with NaNs to test filtering)
    target = np.random.rand(num_samples, 1).astype(np.float32)
    # Introduce NaNs at specific indices to verify they are skipped in training mode
    target[10] = np.nan
    target[20] = np.nan

    # Generate dates
    start_date = pd.Timestamp("2000-01-01")
    dates = [start_date + pd.Timedelta(days=i) for i in range(num_samples)]

    return {
        "features": features,
        "target": target,
        "dates": dates,
        "seq_length": seq_length,
        "num_samples": num_samples,
        "num_features": num_features,
    }


def test_dataset_initialization(synthetic_data):
    """Test if the dataset initializes correctly."""
    dataset = WetlandDataset(
        features_scaled=synthetic_data["features"],
        dates=synthetic_data["dates"],
        target_scaled=synthetic_data["target"],
        seq_length=synthetic_data["seq_length"],
        predict_mode=False,
    )
    assert dataset.seq_length == synthetic_data["seq_length"]
    assert dataset.predict_mode is False


def test_create_windows_training(synthetic_data):
    """
    Test window creation in training mode.
    Should return (feature, target) tuples and skip samples where target is NaN.
    """
    seq_length = synthetic_data["seq_length"]
    dataset = WetlandDataset(
        features_scaled=synthetic_data["features"],
        dates=synthetic_data["dates"],
        target_scaled=synthetic_data["target"],
        seq_length=seq_length,
        predict_mode=False,
    )

    # Expected number of windows:
    # Total possible windows = num_samples - seq_length + 1
    # We introduced 2 NaNs in the target.
    # Logic: if np.isnan(target[target_index]).any() -> continue
    # target_index is the last index of the window.
    # The NaN indices are 10 and 20.
    # If a window ends at index 10 or 20, it should be skipped.

    total_windows = synthetic_data["num_samples"] - seq_length + 1
    # Both index 10 and 20 are >= seq_length-1 (4), so they are reachable as window ends
    expected_len = total_windows - 2

    assert len(dataset) == expected_len

    # Check item structure
    item = dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 2
    feature_window, target_window = item

    # Check shapes
    assert feature_window.shape == (seq_length, synthetic_data["num_features"])
    assert target_window.shape == (1,)


def test_create_windows_predict_mode(synthetic_data):
    """
    Test window creation in prediction mode.
    Should return only feature windows and NOT skip based on target (target can be None).
    """
    seq_length = synthetic_data["seq_length"]
    # Pass None as target to ensure it works without it
    dataset = WetlandDataset(
        features_scaled=synthetic_data["features"],
        dates=synthetic_data["dates"],
        target_scaled=None,
        seq_length=seq_length,
        predict_mode=True,
    )

    total_windows = synthetic_data["num_samples"] - seq_length + 1
    assert len(dataset) == total_windows

    # Check item structure (should be just features)
    item = dataset[0]
    # In predict mode, windows list contains just feature_window
    assert isinstance(item, np.ndarray)
    assert item.shape == (seq_length, synthetic_data["num_features"])


def test_getitem_dtypes(synthetic_data):
    """Test that __getitem__ returns correct data types (torch tensors are usually handled by DataLoader, but dataset returns numpy)."""
    dataset = WetlandDataset(
        features_scaled=synthetic_data["features"],
        dates=synthetic_data["dates"],
        target_scaled=synthetic_data["target"],
        seq_length=synthetic_data["seq_length"],
        predict_mode=False,
    )

    feat, targ = dataset[0]
    assert isinstance(feat, np.ndarray)
    assert isinstance(targ, np.ndarray)
    assert feat.dtype == np.float32
    assert targ.dtype == np.float32


def test_wetland_dataloader_split(synthetic_data):
    """Test if wetland_dataloader correctly splits train and test sets by year."""
    # Modify dates to span multiple years
    num_samples = synthetic_data["num_samples"]
    start_date = pd.Timestamp("2000-01-01")
    # Make half the data 2000, half 2001
    dates = [start_date + pd.Timedelta(days=i) for i in range(num_samples)]
    # Force some into 2001
    mid_point = num_samples // 2
    for i in range(mid_point, num_samples):
        dates[i] = dates[i].replace(year=2001)

    dataset = WetlandDataset(
        features_scaled=synthetic_data["features"],
        dates=dates,
        target_scaled=synthetic_data["target"],
        seq_length=synthetic_data["seq_length"],
        predict_mode=False,
    )

    train_years = [2000]
    batch_size = 4

    train_loader, test_loader = wetland_dataloader(
        dataset, train_years=train_years, batch_size=batch_size
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)

    # Check that train_loader only contains data from 2000
    # We can inspect the indices in the Subset
    train_indices = train_loader.dataset.indices
    for idx in train_indices:
        # dataset.dates_seq aligns with windows
        window_date = dataset.dates_seq[idx]
        assert window_date.year == 2000

    # Check that test_loader only contains data from 2001 (not in train_years)
    test_indices = test_loader.dataset.indices
    for idx in test_indices:
        window_date = dataset.dates_seq[idx]
        assert window_date.year != 2000


def test_dataloader_batching(synthetic_data):
    """Test that the DataLoader produces batches of correct shape."""
    dataset = WetlandDataset(
        features_scaled=synthetic_data["features"],
        dates=synthetic_data["dates"],
        target_scaled=synthetic_data["target"],
        seq_length=synthetic_data["seq_length"],
        predict_mode=False,
    )

    train_years = [2000]  # All data is 2000 in fixture default
    batch_size = 10

    train_loader, _ = wetland_dataloader(dataset, train_years, batch_size=batch_size)

    # Fetch one batch
    features_batch, target_batch = next(iter(train_loader))

    # Shape checks: [Batch, Seq, Feat]
    assert features_batch.shape == (
        batch_size,
        synthetic_data["seq_length"],
        synthetic_data["num_features"],
    )
    # Shape checks: [Batch, 1]
    assert target_batch.shape == (batch_size, 1)
