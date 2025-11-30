import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Import components from the correct package structure
from giems_lstm.engine.trainer import Trainer
from giems_lstm.model.lstm import LSTMNet
from giems_lstm.model.lstmkan import LSTMNetKAN


@pytest.fixture
def mock_data():
    """
    Creates mock training and testing data loaders using TensorDataset.
    Data is randomized and shaped for LSTM training: [Batch, Sequence, Feature].
    """
    # Parameters for mock data
    num_samples = 20
    seq_length = 5
    input_dim = 3
    output_dim = 1
    batch_size = 4

    # Create random features and targets
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
    Determines the appropriate device for model testing (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_trainer_initialization(mock_data, device):
    """
    Test that the Trainer initializes with correct attributes, including
    checking the debug mode epoch override (n_epochs is set to 10 in debug mode).
    """
    train_loader, test_loader = mock_data

    trainer = Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        learn_rate=0.001,
        hidden_dim=16,
        n_layers=1,
        n_epochs=2,  # This value should be overridden by debug=True
        model_type="LSTM",
        verbose_epoch=1,
        patience=5,
        device=device,
        debug=True,
    )

    assert trainer.learn_rate == 0.001
    assert trainer.hidden_dim == 16
    assert trainer.model_type == "LSTM"
    # Check that debug mode overrides n_epochs
    assert trainer.n_epochs == 10
    assert trainer.device == device


def test_trainer_run_lstm(mock_data, device):
    """
    Test a complete training run using the base LSTM model and check results.
    """
    train_loader, test_loader = mock_data

    # Use a small number of epochs (1) for rapid testing
    trainer = Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        learn_rate=0.01,
        hidden_dim=8,
        n_layers=1,
        n_epochs=1,
        model_type="LSTM",
        verbose_epoch=1,
        patience=2,
        device=device,
        debug=False,
    )

    model = trainer.run()

    # Check the returned model instance
    assert isinstance(model, LSTMNet)
    assert next(model.parameters()).device.type == device.type
    # Check that the training successfully updated the best loss
    assert trainer.best_loss < float("inf")
    assert trainer.best_model_state is not None


def test_trainer_run_lstm_kan(mock_data, device):
    """
    Test a complete training run using the LSTM-KAN model variant.
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

    # Check the returned model instance
    assert isinstance(model, LSTMNetKAN)
    assert trainer.best_loss < float("inf")


def test_early_stopping_logic(mock_data, device):
    """
    Test the internal mechanism of the early stopping logic (patience counter).
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
        patience=2,  # Stop after 2 epochs with no improvement
        device=device,
        debug=False,
    )

    # Manually call setup to initialize the model and internal state
    trainer._model_setup()

    # 1. First check: Loss 0.5 (Improvement over infinity)
    should_continue = trainer._check_early_stopping(0.5)
    assert should_continue is True
    assert trainer.best_loss == 0.5
    assert trainer.epochs_no_improve == 0

    # 2. Second check: Loss 0.6 (No improvement)
    should_continue = trainer._check_early_stopping(0.6)
    assert should_continue is True
    assert trainer.best_loss == 0.5  # Best loss remains the same
    assert trainer.epochs_no_improve == 1

    # 3. Third check: Loss 0.7 (No improvement - hits patience limit)
    should_continue = trainer._check_early_stopping(0.7)
    assert should_continue is False  # Should trigger stop
    assert trainer.epochs_no_improve == 2


def test_invalid_model_type(mock_data, device):
    """
    Test that passing an unsupported model type raises a ValueError.
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
