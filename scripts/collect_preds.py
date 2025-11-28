import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path so we can import the config module
# Assuming this script is at scripts/collect_preds.py, root is one level up relative to scripts folder
sys.path.append(str(Path(__file__).resolve().parents[1]))

from giems_lstm.config import Config


def parse_filename(filename: Path):
    """
    Extract lat_idx, lon_idx from filename 'lat_lon.npy'.
    Returns (lat_idx, lon_idx) or None if format is invalid.
    """
    try:
        stem = filename.stem
        parts = stem.split("_")
        if len(parts) != 2:
            return None
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def load_prediction(file_path: Path, lat_idx: int, lon_idx: int):
    """
    Helper function to load a single .npy file.
    Returns tuple (lat_idx, lon_idx, data_array).
    """
    try:
        data = np.load(file_path)
        # Squeeze dimensions if output is (Time, 1) -> (Time,)
        if data.ndim > 1:
            data = data.squeeze()
        return lat_idx, lon_idx, data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect distributed .npy predictions into a single NetCDF file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/E.toml",
        help="Path to config file (default: config/E.toml)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.nc",
        help="Output NetCDF filename (default: output.nc)",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=8,
        help="Number of threads for parallel reading (default: 8)",
    )
    parser.add_argument(
        "-f",
        "--freq",
        type=str,
        default="MS",
        help="Pandas frequency string for time axis (e.g., 'MS' for Month Start, 'D' for Day)",
    )

    args = parser.parse_args()

    # 1. Load Configuration
    config_path = args.config
    print(f"Loading config from {config_path}...")

    # We use mode="predict" to ensure we get the correct folder paths
    config = Config(config_path, mode="predict")
    pred_folder = config.pred_folder

    if not pred_folder.exists():
        print(f"Error: Prediction folder {pred_folder} does not exist.")
        sys.exit(1)

    print(f"Scanning files in {pred_folder}...")
    files = list(pred_folder.glob("*.npy"))
    if not files:
        print("No .npy files found.")
        sys.exit(0)

    # 2. Determine Coordinates and Dimensions from Config
    print("Loading reference coordinates from Config...")
    if not config.TVARs:
        print(
            "Error: No TVARs found in config. Ensure config is valid and files exist."
        )
        sys.exit(1)

    # Use the first variable to get the spatial grid reference
    ref_var_name = list(config.TVARs.keys())[0]
    ref_da = config.TVARs[ref_var_name]

    lats = ref_da.coords["lat"].values
    lons = ref_da.coords["lon"].values

    n_lat = len(lats)
    n_lon = len(lons)

    # Generate Time Axis
    dates = pd.date_range(
        start=config.predict.start_date, end=config.predict.end_date, freq=args.freq
    )
    n_time = len(dates)

    print(f"Target Grid Shape: Time={n_time}, Lat={n_lat}, Lon={n_lon}")

    # 3. Pre-allocate Global Array
    # Initialize with NaN
    print("Allocating memory for full dataset...")
    full_array = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    # 4. Parallel Loading
    print(f"Loading {len(files)} files using {args.parallel} threads...")

    tasks = []
    for f in files:
        indices = parse_filename(f)
        if indices:
            tasks.append((f, indices[0], indices[1]))

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_coords = {
            executor.submit(load_prediction, f, lat, lon): (lat, lon)
            for f, lat, lon in tasks
        }

        for future in tqdm(
            as_completed(future_to_coords), total=len(tasks), unit="file"
        ):
            result = future.result()
            if result is None:
                continue

            lat_idx, lon_idx, data = result

            # Validation
            if len(data) == n_time:
                full_array[:, lat_idx, lon_idx] = data
            else:
                # Handle truncated data if necessary
                valid_len = min(len(data), n_time)
                full_array[:valid_len, lat_idx, lon_idx] = data[:valid_len]

    # 5. Wrap in Xarray
    print("Wrapping data into Xarray Dataset...")
    ds = xr.Dataset(
        data_vars={"fwet": (("time", "lat", "lon"), full_array)},
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={
            "description": "Aggregated GIEMS-LSTM Predictions",
            "source_script": "collect_preds.py",
        },
    )

    # 6. Save to NetCDF
    output_path = Path(args.output)
    print(f"Saving to {output_path}...")

    encoding = {
        "fwet": {
            "zlib": True,
            "complevel": 5,
            "dtype": "float32",
            "_FillValue": np.nan,
        }
    }
    ds.to_netcdf(output_path, encoding=encoding)

    print(f"Done! Saved {output_path}")


if __name__ == "__main__":
    main()
