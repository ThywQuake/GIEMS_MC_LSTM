import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch

# Import the main module object so we can manually inject mocks
import giems_lstm.main as main_module
from giems_lstm.main import app

# Initialize the CLI runner for Typer
runner = CliRunner()


@pytest.fixture
def mock_functions():
    """
    Fixture to mock the core logic functions in main.py.
    We manually inject Mocks into the main module's namespace because
    main.py uses lazy imports (via globals()), preventing standard patching.
    """
    # 1. Create Mocks
    mock_train = MagicMock()
    mock_predict = MagicMock()
    mock_collect = MagicMock()

    # 2. Manually inject into module namespace
    # This allows the CLI commands to call these names without error.
    main_module._train = mock_train
    main_module._predict = mock_predict
    main_module._collect = mock_collect

    # 3. Mock _uniform_entry
    with patch("giems_lstm.main._uniform_entry") as mock_entry:
        yield {
            "train": mock_train,
            "predict": mock_predict,
            "collect": mock_collect,
            "entry": mock_entry,
        }

    # 4. Cleanup (Teardown)
    # Remove injected attributes to prevent side effects in other tests.
    del main_module._train
    del main_module._predict
    del main_module._collect


def test_app_help():
    """
    Test that the --help flag works and displays the correct description and commands.
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "GIEMS-LSTM Training and Prediction CLI" in result.stdout
    assert "train" in result.stdout
    assert "predict" in result.stdout
    assert "collect" in result.stdout


def test_train_command_defaults(mock_functions):
    """
    Test the 'train' command with default arguments.
    """
    result = runner.invoke(app, ["train"])

    assert result.exit_code == 0
    # Verify _uniform_entry is called with defaults (debug=False, parallel=0, seed=3407)
    mock_functions["entry"].assert_called_once_with(False, 0, 3407)

    # Verify _train is called with defaults (thread_id=0, config_path="config/F.toml", debug=False, parallel=0)
    mock_functions["train"].assert_called_once_with(0, "config/F.toml", False, 0)


def test_train_command_custom_args(mock_functions):
    """
    Test the 'train' command with custom arguments.
    """
    args = [
        "train",
        "--config",
        "config/custom.toml",
        "--thread-id",
        "5",
        "--parallel",
        "4",
        "--debug",
        "--seed",
        "12345",
    ]
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    # Verify arguments are correctly passed to the entry point
    mock_functions["entry"].assert_called_once_with(True, 4, 12345)

    # Verify arguments are correctly passed to the train function
    mock_functions["train"].assert_called_once_with(5, "config/custom.toml", True, 4)


def test_predict_command_defaults(mock_functions):
    """
    Test the 'predict' command with default arguments.
    """
    result = runner.invoke(app, ["predict"])

    assert result.exit_code == 0
    mock_functions["entry"].assert_called_once_with(False, 0, 3407)
    # _predict(thread_id, config_path, debug, parallel)
    mock_functions["predict"].assert_called_once_with(0, "config/F.toml", False, 0)


def test_predict_command_custom_args(mock_functions):
    """
    Test the 'predict' command with custom arguments.
    """
    args = ["predict", "-c", "config/pred.toml", "-t", "2", "-p", "8", "-d", "-s", "42"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    mock_functions["entry"].assert_called_once_with(True, 8, 42)
    mock_functions["predict"].assert_called_once_with(2, "config/pred.toml", True, 8)


def test_collect_command_defaults(mock_functions):
    """
    Test the 'collect' command with default arguments.
    """
    result = runner.invoke(app, ["collect"])

    assert result.exit_code == 0
    # collect default parallel is 0
    mock_functions["entry"].assert_called_once_with(False, 0, 3407)
    # _collect(config_path, eval, parallel)
    mock_functions["collect"].assert_called_once_with("config/F.toml", False, 0)


def test_collect_command_custom_args(mock_functions):
    """
    Test the 'collect' command with custom arguments.
    """
    args = [
        "collect",
        "--config",
        "config/analysis.toml",
        "--parallel",
        "16",
        "--seed",
        "999",
        "--eval",  # flag, implies True
    ]
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    mock_functions["entry"].assert_called_once_with(False, 16, 999)
    mock_functions["collect"].assert_called_once_with("config/analysis.toml", True, 16)


def test_invalid_argument():
    """Test that providing an invalid argument results in a failure."""
    result = runner.invoke(app, ["train", "--invalid-arg", "123"])
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_entry_point_logic_with_parallel_check():
    """
    Test the internal logic of _uniform_entry function, checking parallel settings.
    We mock the dependencies (like torch and utility functions) by injecting
    them into the main module's namespace and patching _init_imports.
    """
    from giems_lstm.main import _uniform_entry

    # 1. Prepare mocks for external dependencies used globally in _uniform_entry
    mock_torch = MagicMock()
    mock_torch.set_num_threads = MagicMock()
    mock_logging_util = MagicMock()
    mock_seed_util = MagicMock()

    with (
        # 2. Patch the global variables *before* the function runs
        # FIX: 添加 create=True，因为 main 模块在初始时没有 torch 属性
        patch.object(main_module, "torch", mock_torch, create=True),
        patch.object(
            main_module, "_setup_global_logging", mock_logging_util, create=True
        ),
        patch.object(main_module, "_seed_everything", mock_seed_util, create=True),
        # 3. Patch _init_imports to prevent it from overwriting our mocks with real imports
        patch.object(main_module, "_init_imports", MagicMock()),
        # 4. Patch system calls
        patch("multiprocessing.cpu_count", return_value=8),
    ):
        # ... (后续代码保持不变)
        # Case 1: parallel=0 (Standard single process)
        _uniform_entry(debug=False, parallel=0, seed=123)
        mock_logging_util.assert_called_with(False, "Main")
        mock_seed_util.assert_called_with(123)
        mock_torch.set_num_threads.assert_not_called()

        # Reset mocks for Case 2
        mock_logging_util.reset_mock()
        mock_seed_util.reset_mock()
        mock_torch.set_num_threads.reset_mock()

        # Case 2: parallel=4 (Parallel mode enabled)
        _uniform_entry(debug=True, parallel=4, seed=456)
        mock_logging_util.assert_called_with(True, "Main")
        mock_seed_util.assert_called_with(456)
        mock_torch.set_num_threads.assert_called_with(2)
