import pytest
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import components from the correct package structure
from giems_lstm.data.dataset import WetlandDataset, wetland_dataloader


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
    # Total possible windows = num_samples (50) - seq_length (5) + 1 = 46
    # NaNs are at index 10 and 20 (target index, which is the window end)
    expected_len = (synthetic_data["num_samples"] - seq_length + 1) - 2

    assert len(dataset) == expected_len

    # Check item structure and shapes
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
    Should return only feature windows and NOT skip based on target.
    """
    seq_length = synthetic_data["seq_length"]
    # Pass None as target to match prediction mode requirements
    dataset = WetlandDataset(
        features_scaled=synthetic_data["features"],
        dates=synthetic_data["dates"],
        target_scaled=None,
        seq_length=seq_length,
        predict_mode=True,
    )

    # All possible windows should be created
    total_windows = synthetic_data["num_samples"] - seq_length + 1
    assert len(dataset) == total_windows

    # Check item structure (should be just features)
    item = dataset[0]
    # In predict mode, __getitem__ returns just the feature window (np.ndarray)
    assert isinstance(item, np.ndarray)
    assert item.shape == (seq_length, synthetic_data["num_features"])


def test_getitem_dtypes(synthetic_data):
    """Test that __getitem__ returns correct data types (numpy arrays with float32)."""
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

    # Create dates spanning 2000 and 2001, ensuring window indices align
    dates = []
    for i in range(num_samples):
        # Samples 0-24 in 2000, 25-49 in 2001
        if i < 25:
            dates.append(start_date + pd.Timedelta(days=i))
        else:
            dates.append(start_date.replace(year=2001) + pd.Timedelta(days=i - 25))

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
    train_indices = train_loader.dataset.indices
    for idx in train_indices:
        # dataset.dates_seq is the date corresponding to the target (window end)
        window_date = dataset.dates_seq[idx]
        assert window_date.year == 2000

    # Check that test_loader only contains data NOT in train_years (i.e., 2001)
    test_indices = test_loader.dataset.indices
    for idx in test_indices:
        window_date = dataset.dates_seq[idx]
        assert window_date.year not in train_years


def test_dataloader_batching(synthetic_data):
    """Test that the DataLoader produces batches of correct shape."""
    dataset = WetlandDataset(
        features_scaled=synthetic_data["features"],
        dates=synthetic_data["dates"],
        target_scaled=synthetic_data["target"],
        seq_length=synthetic_data["seq_length"],
        predict_mode=False,
    )

    # Define train_years to ensure we select a non-empty subset
    start_year = synthetic_data["dates"][0].year
    train_years = [start_year]
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
