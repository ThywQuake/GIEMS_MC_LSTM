import logging
import os
import numpy as np
import torch
from functools import partial
import multiprocessing as mp

from giems_lstm.config import Config
from giems_lstm.data import (
    extract_window_data,
    WetlandDataset,
    fit_scalers,
    transform_features,
)
from giems_lstm.engine import Predictor
from giems_lstm.model import LSTMNetKAN
from .allocate_coords import _allocate_coords


def _predict_task(
    coord: tuple[int, int], config: Config, thread_id: int, device: torch.device
):
    logger = logging.getLogger()

    lat_idx, lon_idx = coord
    logger.info(f"[Thread {thread_id}] Predicting {lat_idx}, {lon_idx}")

    train_target, train_features, _ = extract_window_data(
        lat_idx,
        lon_idx,
        config.TVARs,
        config.CVARs,
        config.model.window_size,
        config.predict.train_start_date,
        config.predict.train_end_date,
    )
    target_scaler, feature_scalers = fit_scalers(train_target, train_features)
    pred_target, pred_features, pred_dates = extract_window_data(
        lat_idx,
        lon_idx,
        config.TVARs,
        config.CVARs,
        config.model.window_size,
        config.predict.start_date,
        config.predict.end_date,
    )
    pred_features_scaled = transform_features(pred_features, feature_scalers)
    dataset = WetlandDataset(
        features_scaled=pred_features_scaled,
        dates=pred_dates,
        target_scaled=None,
        seq_length=config.model.seq_length,
        predict_mode=True,
    )

    save_path = config.pred_folder / f"{lat_idx}_{lon_idx}.npy"
    if not config.sys.debug and save_path.exists() and not config.sys.cover_exist:
        logger.info(
            f"Predictions for location ({lat_idx}, {lon_idx}) already exist. Skipping..."
        )
        return
    try:
        model = LSTMNetKAN(
            input_dim=(len(config.TVARs) + len(config.CVARs) - 1)
            * (config.model.window_size**2)
            + 1,
            hidden_dim=config.model.hidden_dim,
            output_dim=1,
            n_layers=config.model.n_layers,
            device=device,
        )
        model.load_state_dict(
            torch.load(
                config.model_folder / f"{lat_idx}_{lon_idx}.pth", map_location=device
            )
        )
    except Exception as e:
        logger.error(f"Error loading model for {lat_idx},{lon_idx}: {e}")
        return

    try:
        predictor = Predictor(
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            dataset=dataset,
            target_scaler=target_scaler,
            model=model,
            device=device,
            batch_size=config.model.batch_size,
            save_path=save_path,
            debug=config.sys.debug,
        )
        predictor.run()
    except Exception as e:
        logger.error(f"Error during prediction for {lat_idx},{lon_idx}: {e}")
        return


def _predict(thread_id: int, config_path: str, debug: bool, para: int):
    # Each process needs to reload configuration and data references to avoid issues with multiprocessing sharing complex objects forked from the main process
    config = Config(config_path=config_path, mode="predict")
    if debug:
        config.sys.debug = True
    # Determine the range of tasks this process is responsible for
    mask = config.mask
    total_tasks = int(np.sum(mask))
    tasks_per_thread = config.sys.tasks_per_thread

    start_task = thread_id * tasks_per_thread
    end_task = min((thread_id + 1) * tasks_per_thread, total_tasks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_t = partial(
        _predict_task, config=config, thread_id=thread_id, device=device
    )

    predict_coords = _allocate_coords(mask, start_task, end_task)
    os.makedirs(config.pred_folder, exist_ok=True)
    if para <= 1:
        for i, (lat_idx, lon_idx) in enumerate(predict_coords):
            logging.info(f"Processing task {i + start_task + 1}/{end_task}")
            predict_t((lat_idx, lon_idx))
    else:
        with mp.Pool(processes=para) as pool:
            pool.map(predict_t, predict_coords)
