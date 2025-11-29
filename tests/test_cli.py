import pytest
from typer.testing import CliRunner
from unittest.mock import patch

# 假设您的包结构正确安装，可以直接导入
# 如果在开发环境中，可能需要确保 src 在 PYTHONPATH 中
from giems_lstm.main import app

# 初始化 Typer 的测试运行器
runner = CliRunner()


@pytest.fixture
def mock_functions():
    """
    Mock 掉 main.py 中的核心逻辑函数，避免测试时执行实际训练/预测代码。
    """
    with (
        patch("giems_lstm.main._train") as mock_train,
        patch("giems_lstm.main._predict") as mock_predict,
        patch("giems_lstm.main._collect") as mock_collect,
        patch("giems_lstm.main._uniform_entry") as mock_entry,
    ):
        yield {
            "train": mock_train,
            "predict": mock_predict,
            "collect": mock_collect,
            "entry": mock_entry,
        }


def test_app_help():
    """测试帮助信息是否能正常显示"""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "GIEMS-LSTM Training and Prediction CLI" in result.stdout
    assert "train" in result.stdout
    assert "predict" in result.stdout
    assert "collect" in result.stdout


def test_train_command_defaults(mock_functions):
    """测试 train 命令使用默认参数"""
    result = runner.invoke(app, ["train"])

    assert result.exit_code == 0
    # 验证 _uniform_entry 被调用 (debug=False, parallel=0, seed=3407)
    mock_functions["entry"].assert_called_once_with(False, 0, 3407)

    # 验证 _train 被调用 (thread_id=0, config_path="config/F.toml", debug=False, parallel=0)
    mock_functions["train"].assert_called_once_with(0, "config/F.toml", False, 0)


def test_train_command_custom_args(mock_functions):
    """测试 train 命令使用自定义参数"""
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
    # 验证参数是否正确透传
    mock_functions["entry"].assert_called_once_with(True, 4, 12345)
    mock_functions["train"].assert_called_once_with(5, "config/custom.toml", True, 4)


def test_predict_command_defaults(mock_functions):
    """测试 predict 命令使用默认参数"""
    result = runner.invoke(app, ["predict"])

    assert result.exit_code == 0
    mock_functions["entry"].assert_called_once_with(False, 0, 3407)
    # _predict(thread_id, config_path, debug, para)
    mock_functions["predict"].assert_called_once_with(0, "config/F.toml", False, 0)


def test_predict_command_custom_args(mock_functions):
    """测试 predict 命令使用自定义参数"""
    args = ["predict", "-c", "config/pred.toml", "-t", "2", "-p", "8", "-d", "-s", "42"]
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    mock_functions["entry"].assert_called_once_with(True, 8, 42)
    mock_functions["predict"].assert_called_once_with(2, "config/pred.toml", True, 8)


def test_collect_command_defaults(mock_functions):
    """测试 collect 命令使用默认参数"""
    result = runner.invoke(app, ["collect"])

    assert result.exit_code == 0
    # collect 默认 parallel 是 4
    mock_functions["entry"].assert_called_once_with(False, 4, 3407)
    # _collect(config_path, parallel)
    mock_functions["collect"].assert_called_once_with("config/F.toml", 4)


def test_collect_command_custom_args(mock_functions):
    """测试 collect 命令使用自定义参数"""
    args = [
        "collect",
        "--config",
        "config/analysis.toml",
        "--parallel",
        "16",
        "--seed",
        "999",
    ]
    result = runner.invoke(app, args)

    assert result.exit_code == 0
    mock_functions["entry"].assert_called_once_with(False, 16, 999)
    mock_functions["collect"].assert_called_once_with("config/analysis.toml", 16)


def test_invalid_argument():
    """测试输入无效参数时的行为"""
    result = runner.invoke(app, ["train", "--invalid-arg", "123"])
    assert result.exit_code != 0
    # Change: check .output instead of .stdout for validation errors
    assert "No such option" in result.output


def test_entry_point_logic_with_parallel_check():
    """
    测试 _uniform_entry 内部逻辑 (mock logging 和 mp)
    因为这是一个辅助函数，我们可以单独 patch 它内部的依赖来测试逻辑分支。
    """
    from giems_lstm.main import _uniform_entry

    with (
        patch("giems_lstm.main.setup_global_logging") as mock_logging,
        patch("giems_lstm.main._seed_everything") as mock_seed,
        patch("multiprocessing.cpu_count", return_value=8),
        patch("torch.set_num_threads") as mock_torch_threads,
    ):
        # 测试 Case 1: parallel=0 (不设置多进程相关)
        _uniform_entry(debug=False, parallel=0, seed=123)
        mock_logging.assert_called_with(False, "Main")
        mock_seed.assert_called_with(123)
        mock_torch_threads.assert_not_called()

        # 测试 Case 2: parallel=4 (设置 workers)
        # cpu_count=8, parallel=4 -> workers = 2
        _uniform_entry(debug=True, parallel=4, seed=456)
        mock_logging.assert_called_with(True, "Main")
        mock_seed.assert_called_with(456)
        mock_torch_threads.assert_called_with(2)
