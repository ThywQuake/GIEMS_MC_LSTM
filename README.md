# GIEMS-MC-LSTM

## Overview
The **GIEMS-MC-LSTM** project implements a deep learning model based on Long Short-Term Memory (LSTM) networks, augmented with a KAN-like (Kolmogorov-Arnold Network) linear layer, named `LSTMNetKAN`. This model is designed for large-scale geospatial time-series prediction. The primary application involves predicting wetland dynamics (e.g., `fwet`, the fraction of wet area) using multiple environmental variables.

The training and prediction pipelines are optimized for large-scale computation, supporting parallel processing using Python's `multiprocessing` module, which allows tasks to be split into chunks for distributed execution (e.g., on a Slurm cluster).

## Model and Data

### Model Architecture
The core model is `LSTMNetKAN`:
* **Type:** LSTM combined with a KANLinear output layer.
* **Example Parameters (from `config/E.toml`):** Hidden Dimension: 384; Number of Layers: 2; Sequence Length: 12.

### Input Variables
The model uses a combination of Time-Series Variables (TVARs) and Constant Variables (CVARs) (configured in `config/E.toml`):
* **TVARs (Time-Series Variables):** `giems2` (`fwet`), `era5` (`tmp`), `mswep` (`pre`), `gleam` (`sm`), `grace` (`lwe_thickness`).
* **CVARs (Constant Variables):** `fcti`.

## Setup and Installation

This project uses **`uv`** for dependency management and building.

1.  **Clone the repository.**
2.  **Install `uv`:** If not already installed, please refer to the official documentation to install `uv`.
3.  **Install Dependencies and Build:** Use the `uv sync` command to simultaneously install dependencies and build the project into an executable environment.

    ```bash
    uv sync
    ```
4.  **Data Preparation:** Ensure all required NetCDF (`.nc`) data files (TVARs, CVARs, and the wetland `mask`) are placed in the paths specified in your configuration file (e.g., `data/clean/` directory for `config/E.toml`).

## Usage

After installation and build, you can use the command-line tool **`giems`** to run the training and prediction pipelines.

### 1. Training

```bash
giems train [OPTIONS]
````

| Option              | Description                                                                           | Default         |
| :------------------ | :------------------------------------------------------------------------------------ | :-------------- |
| `--config`, `-c`    | Path to the configuration TOML file.                                                  | `config/E.toml` |
| `--parallel`, `-p`  | Number of local processes to spawn for parallel training. If \>1, it runs all chunks. | `1`             |
| `--thread-id`, `-t` | Specific chunk ID to run (for Slurm splitting). Ignored if `--parallel` \> 1.         | `0`             |
| `--debug`, `-d`     | Enable debug mode (e.g., reduces the number of epochs).                               | `False`         |

**Example (Parallel Training):**

```bash
# Train using 8 CPU/GPU workers based on config/E.toml
giems train -c config/E.toml -p 8
```

### 2\. Prediction

```bash
giems predict [OPTIONS]
```

| Option              | Description                                                 | Default         |
| :------------------ | :---------------------------------------------------------- | :-------------- |
| `--config`, `-c`    | Path to the configuration TOML file.                        | `config/E.toml` |
| `--parallel`, `-p`  | Number of local processes to spawn for parallel prediction. | `1`             |
| `--thread-id`, `-t` | Specific chunk ID to run.                                   | `0`             |
| `--debug`, `-d`     | Enable debug mode.                                          | `False`         |

**Example (Slurm/Chunked Prediction):**

```bash
# Predict a specific chunk of tasks (e.g., chunk ID 5)
giems predict -c config/E.toml -t 5
```

## Configuration

All settings are controlled by a TOML file (e.g., `config/E.toml`).
