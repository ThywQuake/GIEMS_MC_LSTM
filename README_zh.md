# GIEMS-MC-LSTM

## 概述
**GIEMS-MC-LSTM** 项目实现了一个基于长短期记忆网络（LSTM）的深度学习模型，并在其输出层集成了类似 KAN（Kolmogorov-Arnold Network）的线性层，命名为 `LSTMNetKAN`。该模型旨在进行大规模地理空间时间序列预测。主要应用是利用多个环境变量来预测湿地动态（例如湿地面积分数 `fwet`）。

训练和预测流程针对大规模计算进行了优化，支持使用 Python 的 `multiprocessing` 模块进行并行处理，允许用户将任务分割成多个任务块进行分布式运行。

## 模型与数据

### 模型架构
核心模型为 `LSTMNetKAN`：
* **类型：** LSTM 与 KANLinear 输出层的组合。
* **参数示例 (来自 `config/E.toml`)：** 隐藏层维度：384；层数：2；序列长度：12。

### 输入变量
模型使用时间序列变量（TVARs）和常数变量（CVARs）的组合（配置来自 `config/E.toml`）：
* **TVARs (时间序列变量):** `giems2` (`fwet`)、`era5` (`tmp`)、`mswep` (`pre`)、`gleam` (`sm`)、`grace` (`lwe_thickness`)。
* **CVARs (常数变量):** `fcti`。

## 设置与安装

本项目使用 **`uv`** 进行依赖管理和构建。

1.  **克隆仓库。**
2.  **安装 `uv`：** 如果尚未安装，请参考官方文档安装 `uv`。
3.  **依赖安装与构建：** 使用 `uv sync` 命令同时安装依赖并将项目构建为一个可执行环境。

    ```bash
    uv sync
    ```
4.  **数据准备：** 确保所有必需的 NetCDF (`.nc`) 数据文件（TVARs、CVARs 和湿地掩膜 `mask`）已放置在配置文件中指定的路径（例如，`config/E.toml` 中的 `data/clean/` 目录）。

## 使用方法

项目安装构建后，您可以使用命令行工具 **`giems`** 来运行训练和预测流程。

### 1. 训练 (Training)

```bash
giems train [OPTIONS]
````

| 选项 (Option)       | 描述 (Description)                                                      | 默认值 (Default) |
| :------------------ | :---------------------------------------------------------------------- | :--------------- |
| `--config`, `-c`    | 配置文件 (TOML) 的路径。                                                | `config/E.toml`  |
| `--parallel`, `-p`  | 启动进行并行训练的本地进程数。若 \>1，则运行所有任务块。                | `1`              |
| `--thread-id`, `-t` | 要运行的特定任务块 ID（用于 Slurm 分割）。若 `--parallel` \> 1 则忽略。 | `0`              |
| `--debug`, `-d`     | 启用调试模式（例如，减少 epoch 数量）。                                 | `False`          |

**示例（并行训练）：**

```bash
# 使用 8 个 worker 进行并行训练 (基于 config/E.toml)
giems train -c config/E.toml -p 8
```

### 2\. 预测 (Prediction)

```bash
giems predict [OPTIONS]
```

| 选项 (Option)       | 描述 (Description)             | 默认值 (Default) |
| :------------------ | :----------------------------- | :--------------- |
| `--config`, `-c`    | 配置文件 (TOML) 的路径。       | `config/E.toml`  |
| `--parallel`, `-p`  | 启动进行并行预测的本地进程数。 | `1`              |
| `--thread-id`, `-t` | 要运行的特定任务块 ID。        | `0`              |
| `--debug`, `-d`     | 启用调试模式。                 | `False`          |

**示例（Slurm/分块预测）：**

```bash
# 预测特定的任务块（例如，ID 为 5 的任务块）
giems predict -c config/E.toml -t 5
```

## 配置

所有设置均通过 TOML 文件控制（例如 `config/E.toml`）。
