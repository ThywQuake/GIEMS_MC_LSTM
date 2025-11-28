import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.giems_lstm.engine.trainer import Trainer
from src.giems_lstm.model.lstm import LSTMNet
from src.giems_lstm.model.lstmkan import LSTMNetKAN


@pytest.fixture
def mock_data():
    """
    Creates mock training and testing data loaders.
    """
    # Parameters for mock data
    num_samples = 20
    seq_length = 5
    input_dim = 3
    output_dim = 1
    batch_size = 4

    # Create random features and targets
    # Features: [Batch, Seq, Dim]
    train_features = torch.randn(num_samples, seq_length, input_dim)
    train_targets = torch.randn(num_samples, output_dim)

    test_features = torch.randn(num_samples // 2, seq_length, input_dim)
    test_targets = torch.randn(num_samples // 2, output_dim)

    # Create TensorDatasets
    train_dataset = TensorDataset(train_features, train_targets)
    test_dataset = TensorDataset(test_features, test_targets)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


@pytest.fixture
def device():
    """
    Use CUDA if available, else CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_trainer_initialization(mock_data, device):
    """
    Test that the Trainer initializes with correct attributes.
    """
    train_loader, test_loader = mock_data

    trainer = Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        learn_rate=0.001,
        hidden_dim=16,
        n_layers=1,
        n_epochs=2,
        model_type="LSTM",
        verbose_epoch=1,
        patience=5,
        device=device,
        debug=True,
    )

    assert trainer.learn_rate == 0.001
    assert trainer.hidden_dim == 16
    assert trainer.model_type == "LSTM"
    # In debug mode, n_epochs is forced to 10 in the __init__ provided in source
    # We check if it adheres to the logic in the source code
    assert trainer.n_epochs == 10
    assert trainer.device == device


def test_trainer_run_lstm(mock_data, device):
    """
    Test a complete training run with the LSTM model.
    """
    train_loader, test_loader = mock_data

    # Use a small number of epochs for speed
    trainer = Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        learn_rate=0.01,
        hidden_dim=8,
        n_layers=1,
        n_epochs=1,  # Set to 1, but debug=False to respect this value
        model_type="LSTM",
        verbose_epoch=1,
        patience=2,
        device=device,
        debug=False,
    )

    model = trainer.run()

    # Check that a model was returned and is of correct type
    assert isinstance(model, LSTMNet)
    # Check that model is on the correct device
    assert next(model.parameters()).device.type == device.type
    # Check that best_loss was updated from infinity
    assert trainer.best_loss < float("inf")
    # Check that best_model_state is saved
    assert trainer.best_model_state is not None


def test_trainer_run_lstm_kan(mock_data, device):
    """
    Test a complete training run with the LSTM-KAN model.
    """
    train_loader, test_loader = mock_data

    trainer = Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        learn_rate=0.01,
        hidden_dim=8,
        n_layers=1,
        n_epochs=1,
        model_type="LSTM_KAN",
        verbose_epoch=1,
        patience=2,
        device=device,
        debug=False,
    )

    model = trainer.run()

    assert isinstance(model, LSTMNetKAN)
    assert trainer.best_loss < float("inf")


def test_early_stopping_logic(mock_data, device):
    """
    Test the early stopping mechanism directly.
    """
    train_loader, test_loader = mock_data

    trainer = Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        learn_rate=0.001,
        hidden_dim=8,
        n_layers=1,
        n_epochs=10,
        model_type="LSTM",
        verbose_epoch=1,
        patience=2,  # Early stopping after 2 epochs of no improvement
        device=device,
        debug=False,
    )

    # Manually setup model and learning for the internal state
    trainer._model_setup()

    # 1. First epoch: Loss 0.5 (Improvement over inf)
    should_continue = trainer._check_early_stopping(0.5)
    assert should_continue is True
    assert trainer.best_loss == 0.5
    assert trainer.epochs_no_improve == 0

    # 2. Second epoch: Loss 0.6 (No improvement)
    should_continue = trainer._check_early_stopping(0.6)
    assert should_continue is True
    assert trainer.best_loss == 0.5  # Remains 0.5
    assert trainer.epochs_no_improve == 1

    # 3. Third epoch: Loss 0.7 (No improvement - Hit patience limit of 2)
    # Note: logic is if epochs_no_improve >= patience -> return False
    # Here epochs_no_improve becomes 2.
    should_continue = trainer._check_early_stopping(0.7)
    assert should_continue is False  # Should stop
    assert trainer.epochs_no_improve == 2


def test_invalid_model_type(mock_data, device):
    """
    Test that passing an invalid model type raises a ValueError.
    """
    train_loader, test_loader = mock_data

    trainer = Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        learn_rate=0.01,
        hidden_dim=8,
        n_layers=1,
        n_epochs=1,
        model_type="INVALID_TYPE",
        verbose_epoch=1,
        patience=1,
        device=device,
    )

    with pytest.raises(ValueError, match="Unsupported model type"):
        trainer.run()
