"""Reference judge shock specifications matching R1_CONTRACT.md."""

from __future__ import annotations

import numpy as np


LAT_CENTRES = np.arange(6.5, 38.0, 1.0, dtype=np.float64)  # length 32, i = lat - 6.5
LON_CENTRES = np.arange(65.5, 101.0, 1.0, dtype=np.float64)  # length 36, j = lon - 65.5

CH_T2M = 0
CH_TP = 1
CH_U10 = 2
CH_V10 = 3
CH_MSL = 4
CH_TCWV = 5


def box_mask(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> np.ndarray:
    """Return [32, 36] boolean mask using inclusive bounds on cell centres."""
    lat_ok = (LAT_CENTRES >= lat_min) & (LAT_CENTRES <= lat_max)
    lon_ok = (LON_CENTRES >= lon_min) & (LON_CENTRES <= lon_max)
    return lat_ok[:, None] & lon_ok[None, :]


def build_warm_ocean(init_last: np.ndarray) -> np.ndarray:
    """Bay of Bengal warm-ocean shock: t2m += 3.0 K, tcwv += 0.15 * tcwv over lat 8..20, lon 82..94."""
    arr = np.asarray(init_last, dtype=np.float32)
    if arr.shape != (6, 32, 36):
        raise ValueError(f"Expected init_last shape (6, 32, 36), got {arr.shape}")
    delta = np.zeros((6, 32, 36), dtype=np.float32)
    mask = box_mask(8.0, 20.0, 82.0, 94.0)
    delta[CH_T2M, mask] = 3.0
    delta[CH_TCWV, mask] = 0.15 * arr[CH_TCWV, mask]
    return delta


def build_monsoon_stall(init_last: np.ndarray) -> np.ndarray:
    """Arabian Sea monsoon-stall shock: u10 -= 0.40 * u10, v10 -= 0.40 * v10 over lat 8..20, lon 65..75."""
    arr = np.asarray(init_last, dtype=np.float32)
    if arr.shape != (6, 32, 36):
        raise ValueError(f"Expected init_last shape (6, 32, 36), got {arr.shape}")
    delta = np.zeros((6, 32, 36), dtype=np.float32)
    mask = box_mask(8.0, 20.0, 65.0, 75.0)
    delta[CH_U10, mask] = -0.40 * arr[CH_U10, mask]
    delta[CH_V10, mask] = -0.40 * arr[CH_V10, mask]
    return delta


def build_null() -> np.ndarray:
    """Null shock: exact zeros of shape [6, 32, 36]."""
    return np.zeros((6, 32, 36), dtype=np.float32)
