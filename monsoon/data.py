"""Dataset loading, fractional time splits, normalization, and one-step Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import xarray as xr

from monsoon.climatology import DEFAULT_PERIOD, evaluate as eval_harmonic, sincos_from_doy


CH_TP = 1
DEFAULT_NC_PATH = Path(__file__).resolve().parent.parent / "data" / "india_climate_train.nc"


def load_dataset(nc_path: str | Path = DEFAULT_NC_PATH) -> dict[str, Any]:
    """Load dataset arrays and day-of-year indices without storing or printing dates."""
    with xr.open_dataset(nc_path) as ds:
        data = ds["data"].values.astype(np.float32)  # [T, 6, 32, 36]
        static = ds["static"].values.astype(np.float32)  # [2, 32, 36]
        lat = ds["lat"].values.astype(np.float64)  # [32]
        lon = ds["lon"].values.astype(np.float64)  # [36]
        doy = pd.DatetimeIndex(ds["time"].values).day_of_year.to_numpy(dtype=np.int64)
        channels = [str(c) for c in ds["channel"].values]
        static_channels = [str(s) for s in ds["static_channel"].values]
    return {
        "data": data,
        "static": static,
        "lat": lat,
        "lon": lon,
        "doy": doy,
        "channels": channels,
        "static_channels": static_channels,
    }


def split_indices(
    total_steps: int,
    train_frac: float = 0.82,
    val_frac: float = 0.09,
    test_frac: float = 0.09,
) -> dict[str, np.ndarray]:
    """Return contiguous time-ordered fractional index arrays (never split by year or date)."""
    if not np.isclose(train_frac + val_frac + test_frac, 1.0, atol=1e-6):
        raise ValueError("Split fractions must sum to 1.0")
    n_train = int(total_steps * train_frac)
    n_val = int(total_steps * val_frac)
    n_test = total_steps - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"All splits must be non-empty; got {(n_train, n_val, n_test)}")
    return {
        "train": np.arange(0, n_train, dtype=np.int64),
        "val": np.arange(n_train, n_train + n_val, dtype=np.int64),
        "test": np.arange(n_train + n_val, total_steps, dtype=np.int64),
    }


def to_model_space(x: np.ndarray) -> np.ndarray:
    """Convert physical state [..., 6, H, W] to model space (tp clamped >= 0 then log1p)."""
    out = np.array(x, dtype=np.float32, copy=True)
    out[..., CH_TP, :, :] = np.log1p(np.maximum(out[..., CH_TP, :, :], 0.0))
    return out


def from_model_space(y: np.ndarray) -> np.ndarray:
    """Convert model-space state [..., 6, H, W] back to physical units (tp expm1, clamped >= 0)."""
    out = np.array(y, dtype=np.float32, copy=True)
    out[..., CH_TP, :, :] = np.maximum(np.expm1(out[..., CH_TP, :, :]), 0.0)
    return out


def compute_norm_stats(
    data_phys_train: np.ndarray,
    clim_model_train: np.ndarray,
    static: np.ndarray,
    floor_median_frac: float = 0.25,
    rain_threshold_mm: float = 10.0,
) -> dict[str, np.ndarray]:
    """Compute per-pixel anomaly std and channel/static bounds strictly from TRAIN data."""
    y_train = to_model_space(data_phys_train).astype(np.float64)
    anom_train = y_train - clim_model_train.astype(np.float64)
    raw_sigma = np.std(anom_train, axis=0)  # [6, 32, 36]
    med_sigma = np.median(raw_sigma, axis=(1, 2))  # [6]
    floor = floor_median_frac * med_sigma[:, None, None]
    sigma_pix = np.maximum(raw_sigma, floor).astype(np.float32)

    chan_mean_model = y_train.mean(axis=(0, 2, 3)).astype(np.float32)
    chan_std_model = np.maximum(y_train.std(axis=(0, 2, 3)), 1e-6).astype(np.float32)
    oro_mean = np.float32(static[1].mean())
    oro_std = np.float32(max(float(static[1].std()), 1e-6))
    phys_min = data_phys_train.min(axis=(0, 2, 3)).astype(np.float32)
    phys_max = data_phys_train.max(axis=(0, 2, 3)).astype(np.float32)
    rain_pos_rate = np.float32(np.mean(data_phys_train[:, CH_TP] >= rain_threshold_mm))

    return {
        "sigma_pix": sigma_pix,
        "chan_mean_model": chan_mean_model,
        "chan_std_model": chan_std_model,
        "oro_mean": oro_mean,
        "oro_std": oro_std,
        "phys_min": phys_min,
        "phys_max": phys_max,
        "rain_pos_rate": rain_pos_rate,
    }


def save_norm(path: str | Path, norm: dict[str, np.ndarray]) -> None:
    """Save normalization arrays and training bounds to npz."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **norm)


def load_norm(path: str | Path) -> dict[str, np.ndarray]:
    """Load normalization arrays and training bounds from npz."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def build_static_norm(static: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    """Return [2, H, W] float32 with land_sea_mask and z-scored orography_m."""
    lsm = static[0].astype(np.float32)
    oro = ((static[1] - float(norm["oro_mean"])) / float(norm["oro_std"])).astype(np.float32)
    return np.stack([lsm, oro], axis=0)


def build_step_features(
    a_hist: np.ndarray,
    clim_norm_target: np.ndarray,
    static_norm: np.ndarray,
    doy_sincos_target: np.ndarray,
) -> np.ndarray:
    """Assemble 34-channel input feature map [34, H, W] from [3, 6, H, W] anomalies."""
    a_m2, a_m1, a_0 = a_hist[0], a_hist[1], a_hist[2]
    diff = a_0 - a_m1
    h, w = a_0.shape[1], a_0.shape[2]
    sc_grid = np.broadcast_to(
        np.asarray(doy_sincos_target, dtype=np.float32)[:, None, None],
        (2, h, w),
    )
    return np.concatenate([a_m2, a_m1, a_0, diff, clim_norm_target, static_norm, sc_grid], axis=0).astype(
        np.float32
    )


class OneStepDataset(Dataset):
    """Memory-lean one-step Dataset that builds 34-channel features per sample on the fly."""

    def __init__(
        self,
        data_phys: np.ndarray,
        doy: np.ndarray,
        split_idx: np.ndarray,
        harm_coef: np.ndarray,
        norm: dict[str, np.ndarray],
        static: np.ndarray,
        harm_k: int = 4,
        period: int = DEFAULT_PERIOD,
        anomaly_clip: float = 8.0,
        rain_threshold_mm: float = 10.0,
    ) -> None:
        super().__init__()
        self.data_phys = data_phys
        self.doy = doy
        self.period = int(period)
        self.rain_threshold_mm = float(rain_threshold_mm)

        y_all = to_model_space(data_phys)
        clim_all = eval_harmonic(harm_coef, doy / float(self.period), K=harm_k)
        sigma_pix = norm["sigma_pix"]
        self.anom_raw = ((y_all - clim_all) / sigma_pix).astype(np.float32)
        self.anom_clipped = np.clip(self.anom_raw, -anomaly_clip, anomaly_clip).astype(np.float32)

        # Precompute 365-entry normalized climatology lookup table [365, 6, 32, 36] (~9.7 MB)
        doys_365 = np.arange(1, self.period + 1, dtype=np.float64)
        clim_365 = eval_harmonic(harm_coef, doys_365 / float(self.period), K=harm_k)
        cmean = norm["chan_mean_model"][None, :, None, None]
        cstd = norm["chan_std_model"][None, :, None, None]
        self.clim_norm_table = ((clim_365 - cmean) / cstd).astype(np.float32)
        self.static_norm = build_static_norm(static, norm)

        # Valid t_last indices stay strictly inside split_idx: [t-2..t] and [t+1] in split_idx
        self.t_indices = split_idx[2:-1].astype(np.int64)

    def __len__(self) -> int:
        return len(self.t_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.t_indices[idx])
        a_hist = self.anom_clipped[t - 2 : t + 1]  # [3, 6, 32, 36]
        doy_target = int(self.doy[t]) + 1
        clim_norm_t = self.clim_norm_table[(doy_target - 1) % self.period]
        sc_t = sincos_from_doy(doy_target, period=self.period).astype(np.float32)
        feat = build_step_features(a_hist, clim_norm_t, self.static_norm, sc_t)

        a_t = a_hist[2]  # [6, 32, 36]
        a_target = self.anom_raw[t + 1]  # [6, 32, 36]
        tp_phys_target = self.data_phys[t + 1, CH_TP]  # [32, 36]
        event_target = (tp_phys_target >= self.rain_threshold_mm).astype(np.float32)[None, :, :]  # [1, 32, 36]

        return (
            torch.from_numpy(feat),
            torch.from_numpy(a_t.copy()),
            torch.from_numpy(a_target.copy()),
            torch.from_numpy(event_target),
            torch.from_numpy(tp_phys_target.copy()),
        )


class MultiStepDataset(Dataset):
    """Multi-step Dataset returning initial 3-day anomaly history and K future targets inside one split."""

    def __init__(
        self,
        data_phys: np.ndarray,
        doy: np.ndarray,
        split_idx: np.ndarray,
        harm_coef: np.ndarray,
        norm: dict[str, np.ndarray],
        steps: int,
        harm_k: int = 4,
        period: int = DEFAULT_PERIOD,
        anomaly_clip: float = 8.0,
        rain_threshold_mm: float = 10.0,
    ) -> None:
        super().__init__()
        self.data_phys = data_phys
        self.doy = doy
        self.steps = int(steps)
        self.period = int(period)
        self.rain_threshold_mm = float(rain_threshold_mm)

        y_all = to_model_space(data_phys)
        clim_all = eval_harmonic(harm_coef, doy / float(self.period), K=harm_k)
        sigma_pix = norm["sigma_pix"]
        self.anom_raw = ((y_all - clim_all) / sigma_pix).astype(np.float32)
        self.anom_clipped = np.clip(self.anom_raw, -anomaly_clip, anomaly_clip).astype(np.float32)

        # Valid start index t keeps [t-2..t] and [t+1..t+steps] strictly inside split_idx
        self.t_indices = split_idx[2 : len(split_idx) - self.steps].astype(np.int64)

    def __len__(self) -> int:
        return len(self.t_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self.t_indices[idx])
        a_init_hist = self.anom_clipped[t - 2 : t + 1]  # [3, 6, 32, 36]
        doy_last = int(self.doy[t])
        a_targets = self.anom_raw[t + 1 : t + 1 + self.steps]  # [K, 6, 32, 36]
        tp_phys_targets = self.data_phys[t + 1 : t + 1 + self.steps, CH_TP]  # [K, 32, 36]
        event_targets = (tp_phys_targets >= self.rain_threshold_mm).astype(np.float32)  # [K, 32, 36]

        return (
            torch.from_numpy(a_init_hist.copy()),
            torch.tensor(doy_last, dtype=torch.long),
            torch.from_numpy(a_targets.copy()),
            torch.from_numpy(event_targets),
            torch.from_numpy(tp_phys_targets.copy()),
        )
