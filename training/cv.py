from typing import Generator, Tuple
import numpy as np

def expanding_window_split(n_samples: int, n_splits: int = 3, min_train_size: float = 0.5,
                           val_size: float = 0.15, purge: int = 1) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Purged chronological CV. Sample t has a label ending t + purge."""
    if n_samples <= 0 or purge < 0:
        return
    val_len = max(1, int(n_samples * val_size))
    first_val = int(n_samples * min_train_size)
    for idx in range(n_splits):
        val_start = first_val + idx * val_len
        val_end = min(val_start + val_len, n_samples)
        if val_start - purge < 1 or val_start >= val_end:
            break
        yield np.arange(0, val_start - purge), np.arange(val_start, val_end)
