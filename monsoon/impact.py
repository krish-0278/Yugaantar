"""Climate impact proxy indices computed from physical rollouts and harmonic climatology."""

from __future__ import annotations

from typing import Any

import numpy as np

from monsoon.climatology import DEFAULT_PERIOD, evaluate as eval_harmonic
from monsoon.transforms import CH_TP, from_model_space


CH_T2M = 0
HEATWAVE_ANOM_K = 3.0
HEATWAVE_MIN_CONSEC_DAYS = 2
HEAVY_RAIN_MM = 64.5
VERY_HEAVY_RAIN_MM = 115.6
RAIN_DEFICIT_FRAC = 0.20


def rollout_climatology_phys(
    harm_coef: np.ndarray,
    doy_last: int,
    steps: int,
    *,
    harm_k: int = 4,
    period: int = DEFAULT_PERIOD,
) -> np.ndarray:
    """Evaluate physical-unit harmonic climatology [steps, 6, 32, 36] for days doy_last+1..doy_last+steps."""
    ks = np.arange(1, int(steps) + 1, dtype=np.int64)
    doys = ((int(doy_last) - 1 + ks) % int(period)) + 1
    doy_frac = doys.astype(np.float64) / float(period)
    clim_model = eval_harmonic(harm_coef, doy_frac, K=int(harm_k))
    return from_model_space(clim_model).astype(np.float32)


def _consecutive_run_mask(hot: np.ndarray, min_days: int = HEATWAVE_MIN_CONSEC_DAYS) -> np.ndarray:
    """Return boolean mask [T, ...] True for timesteps belonging to a True run of length >= min_days."""
    t_len = hot.shape[0]
    out = np.zeros_like(hot, dtype=bool)
    if t_len < min_days:
        return out
    flat = hot.reshape(t_len, -1)
    out_flat = out.reshape(t_len, -1)
    for col in range(flat.shape[1]):
        run = 0
        for row in range(t_len):
            if flat[row, col]:
                run += 1
            else:
                if run >= min_days:
                    out_flat[row - run : row, col] = True
                run = 0
        if run >= min_days:
            out_flat[t_len - run : t_len, col] = True
    return out


def compute_rollout_impacts(
    rollout_phys: np.ndarray,
    clim_phys: np.ndarray,
    land_mask: np.ndarray,
    lat: np.ndarray,
) -> dict[str, float | int]:
    """Compute land-only proxy impact indices for one [steps, 6, H, W] physical rollout.

    Indices (proxies only, daily-mean t2m, not an operational forecast):
      - heatwave_cell_days: land cell-days with t2m - clim_t2m >= 3.0 K for >= 2 consecutive days
      - heavy_rain_64_share / heavy_rain_115_share: share of land cell-days with tp >= 64.5 / 115.6 mm/day
      - rain_deficit_share: share of land cells with cumulative rollout tp > 20% below cumulative climatology
      - crop_stress_csi: land area-weighted mean of clip(0.6*waterDeficit + 0.4*heatStress, 0, 1),
        where waterDeficit = clip((cum_clim - cum_pred)/max(cum_clim, 1e-6), 0, 1) and
        heatStress = clip(mean_t(max(t2m - clim_t2m, 0)) / 3.0, 0, 1).
    """
    pred = np.asarray(rollout_phys, dtype=np.float64)
    clim = np.asarray(clim_phys, dtype=np.float64)
    if pred.shape != clim.shape or pred.ndim != 4 or pred.shape[1] != 6:
        raise ValueError(f"Expected matching [steps, 6, H, W] shapes, got {pred.shape} vs {clim.shape}")

    land = np.asarray(land_mask, dtype=np.float64) >= 0.5
    n_land = int(np.sum(land))
    if n_land == 0:
        raise ValueError("land_mask must contain at least one land cell (>= 0.5)")

    steps = pred.shape[0]
    lat_arr = np.asarray(lat, dtype=np.float64)
    if lat_arr.shape != (pred.shape[2],):
        raise ValueError(f"lat shape {lat_arr.shape} != ({pred.shape[2]},)")
    w_grid = np.cos(np.deg2rad(lat_arr))[:, None] * np.ones((1, pred.shape[3]), dtype=np.float64)
    w_land = w_grid[land]
    w_land = w_land / np.sum(w_land)

    t2m_anom = pred[:, CH_T2M][:, land] - clim[:, CH_T2M][:, land]  # [steps, n_land]
    hw_mask = _consecutive_run_mask(t2m_anom >= HEATWAVE_ANOM_K, min_days=HEATWAVE_MIN_CONSEC_DAYS)
    hw_cell_days = int(np.sum(hw_mask))

    tp_land = pred[:, CH_TP][:, land]  # [steps, n_land]
    total_cell_days = float(steps * n_land)
    heavy_64_share = float(np.sum(tp_land >= HEAVY_RAIN_MM) / total_cell_days)
    heavy_115_share = float(np.sum(tp_land >= VERY_HEAVY_RAIN_MM) / total_cell_days)

    cum_pred = np.sum(tp_land, axis=0)
    cum_clim = np.sum(clim[:, CH_TP][:, land], axis=0)
    has_rain = cum_clim > 1e-6
    rel_deficit = np.where(has_rain, (cum_clim - cum_pred) / np.maximum(cum_clim, 1e-6), 0.0)
    deficit_share = float(np.mean(rel_deficit > RAIN_DEFICIT_FRAC))

    water_deficit = np.clip(rel_deficit, 0.0, 1.0)
    heat_stress = np.clip(np.mean(np.maximum(t2m_anom, 0.0), axis=0) / HEATWAVE_ANOM_K, 0.0, 1.0)
    csi_cell = np.clip(0.6 * water_deficit + 0.4 * heat_stress, 0.0, 1.0)
    crop_stress_csi = float(np.sum(w_land * csi_cell))

    return {
        "heatwave_cell_days": hw_cell_days,
        "heavy_rain_64_share": heavy_64_share,
        "heavy_rain_115_share": heavy_115_share,
        "rain_deficit_share": deficit_share,
        "crop_stress_csi": crop_stress_csi,
    }


def compute_impacts_pair(
    prime_phys: np.ndarray,
    multiverse_phys: np.ndarray,
    clim_phys: np.ndarray,
    land_mask: np.ndarray,
    lat: np.ndarray,
) -> dict[str, Any]:
    """Compare Prime vs Multiverse impacts over the same rollout steps and land mask."""
    p = compute_rollout_impacts(prime_phys, clim_phys, land_mask, lat)
    m = compute_rollout_impacts(multiverse_phys, clim_phys, land_mask, lat)
    delta: dict[str, dict[str, float | None]] = {}
    for key in p:
        pv = float(p[key])
        mv = float(m[key])
        diff = mv - pv
        rel_pct = (100.0 * diff / pv) if abs(pv) > 1e-12 else None
        delta[key] = {"abs": diff, "pct": rel_pct}

    return {
        "steps": int(np.asarray(prime_phys).shape[0]),
        "n_land_cells": int(np.sum(np.asarray(land_mask) >= 0.5)),
        "prime": p,
        "multiverse": m,
        "delta": delta,
    }
