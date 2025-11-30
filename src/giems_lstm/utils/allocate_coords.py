import numpy as np


def _allocate_coords(mask: np.ndarray, start_task: int, end_task: int):
    task_counter = 0
    for lat_idx in range(mask.shape[0]):
        for lon_idx in range(mask.shape[1]):
            if not mask[lat_idx, lon_idx]:
                continue
            if task_counter < start_task:
                task_counter += 1
                continue
            elif task_counter >= end_task:
                return

            task_counter += 1
            yield (lat_idx, lon_idx)
