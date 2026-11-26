"""Harmonic and empirical climatology (period 365 days, fitted on train only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_PERIOD = 365


def sincos_from_doy(doy: np.ndarray | int | float, period: int = DEFAULT_PERIOD) -> np.ndarray:
    """Compute [sin(2*pi*doy/period), cos(2*pi*doy/period)]."""
    angle = 2.0 * np.pi * np.asarray(doy, dtype=np.float64) / float(period)
    return np.stack([np.sin(angle), np.cos(angle)], axis=-1)


def doy_frac_from_sincos(s: np.ndarray | float, c: np.ndarray | float) -> np.ndarray:
    """Decode [sin, cos] back to fractional day-of-year in [0, 1)."""
    return (np.arctan2(s, c) % (2.0 * np.pi)) / (2.0 * np.pi)


def design(doy_frac: np.ndarray, K: int = 4) -> np.ndarray:
    """Build harmonic design matrix [T, 2K+1] for doy_frac = doy / 365.0."""
    frac = np.atleast_1d(np.asarray(doy_frac, dtype=np.float64))
    cols = [np.ones_like(frac)]
    for k in range(1, K + 1):
        ang = 2.0 * np.pi * k * frac
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    return np.stack(cols, axis=-1)


def fit(y: np.ndarray, doy_frac: np.ndarray, K: int = 4, ridge: float = 1e-3) -> np.ndarray:
    """Fit per-pixel harmonic climatology on model-space data y [T, C, H, W] (train only)."""
    A = design(doy_frac, K=K)
    T = A.shape[0]
    if y.shape[0] != T:
        raise ValueError(f"Time length mismatch: y has {y.shape[0]}, doy_frac has {T}")
    reg = ridge * np.eye(A.shape[1], dtype=np.float64)
    rhs = A.T @ y.reshape(T, -1).astype(np.float64)
    coef = np.linalg.solve(A.T @ A + reg, rhs)
    return coef.reshape(A.shape[1], *y.shape[1:]).astype(np.float32)


def evaluate(coef: np.ndarray, doy_frac: np.ndarray, K: int = 4) -> np.ndarray:
    """Evaluate harmonic climatology [T, C, H, W] from coefficients [2K+1, C, H, W]."""
    A = design(doy_frac, K=K)
    out = np.tensordot(A, coef.astype(np.float64), axes=(1, 0))
    return out.astype(np.float32)


def save_harmonics(path: str | Path, coef: np.ndarray, K: int = 4, period: int = DEFAULT_PERIOD) -> None:
    """Save only fitted harmonic coefficients and metadata (never daily data)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        coef=np.asarray(coef, dtype=np.float32),
        K=np.int64(K),
        period=np.int64(period),
    )


def load_harmonics(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Load harmonic coefficients, K, and period from npz."""
    with np.load(path) as z:
        return z["coef"].astype(np.float32), int(z["K"]), int(z["period"])


def fit_empirical(
    x: np.ndarray,
    doy: np.ndarray,
    smooth_window: int = 15,
    period: int = DEFAULT_PERIOD,
) -> np.ndarray:
    """Build circularly smoothed empirical DOY climatology [period, C, H, W] from train only.

    Day 366 is merged into day 365 before circular smoothing.
    """
    if smooth_window % 2 != 1 or smooth_window < 1:
        raise ValueError(f"smooth_window must be a positive odd integer, got {smooth_window}")
    doy_arr = np.asarray(doy, dtype=np.int64)
    if x.shape[0] != doy_arr.shape[0]:
        raise ValueError("Time length mismatch between x and doy")
    doy_clamped = np.minimum(np.maximum(doy_arr, 1), period)  # merge 366 -> 365
    sums = np.zeros((period, *x.shape[1:]), dtype=np.float64)
    counts = np.zeros(period, dtype=np.int64)
    np.add.at(sums, doy_clamped - 1, x.astype(np.float64))
    np.add.at(counts, doy_clamped - 1, 1)
    if np.any(counts == 0):
        raise ValueError("Every DOY 1..365 must have at least one training sample")
    daily = sums / counts[:, None, None, None]
    if smooth_window == 1:
        return daily.astype(np.float32)
    half = smooth_window // 2
    padded = np.pad(daily, ((half, half), (0, 0), (0, 0), (0, 0)), mode="wrap")
    csum = np.concatenate([np.zeros((1, *daily.shape[1:]), dtype=np.float64), np.cumsum(padded, axis=0)], axis=0)
    smoothed = (csum[smooth_window:] - csum[:-smooth_window]) / float(smooth_window)
    return smoothed.astype(np.float32)


def evaluate_empirical(table: np.ndarray, doy: np.ndarray, period: int = DEFAULT_PERIOD) -> np.ndarray:
    """Lookup empirical climatology for forecast day indices (1-based, wrapping mod 365)."""
    doy_arr = np.atleast_1d(np.asarray(doy, dtype=np.int64))
    idx = (doy_arr - 1) % period
    return table[idx].astype(np.float32)
