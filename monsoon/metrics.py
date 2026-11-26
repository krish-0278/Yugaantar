"""Local evaluation metrics matching the Round 1 scoring specification."""

from __future__ import annotations

import numpy as np


def lat_weights(lat: np.ndarray) -> np.ndarray:
    """Cosine-latitude weights normalized to mean 1.0."""
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    return w / w.mean()


def lat_weighted_rmse(pred: np.ndarray, true: np.ndarray, lat_w: np.ndarray) -> np.ndarray:
    """Compute per-channel lat-weighted RMSE over [N, C, H, W] (or [C, H, W])."""
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(true, dtype=np.float64)
    if p.ndim == 3:
        p = p[None, ...]
        t = t[None, ...]
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {t.shape}")
    w = np.asarray(lat_w, dtype=np.float64)
    w = w / w.mean()
    sq_err = (p - t) ** 2
    mse = np.mean(sq_err * w[None, None, :, None], axis=(0, 2, 3))
    return np.sqrt(mse)


def skill_score(
    rmse_model: np.ndarray,
    rmse_pers: np.ndarray,
    rmse_clim: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute per-channel skill 1 - RMSE_model / min(RMSE_pers, RMSE_clim) and 6-channel mean."""
    best_ref = np.minimum(np.asarray(rmse_pers, dtype=np.float64), np.asarray(rmse_clim, dtype=np.float64))
    ss = 1.0 - np.asarray(rmse_model, dtype=np.float64) / best_ref
    return ss, float(np.mean(ss))


def heavy_rain_csi_details(
    pred_tp: np.ndarray,
    true_tp: np.ndarray,
    threshold_mm: float = 10.0,
) -> dict[str, float | int | bool]:
    """Compute heavy-rain CSI = hits / (hits + misses + false_alarms) and event counts."""
    pred_ev = np.asarray(pred_tp, dtype=np.float64) >= threshold_mm
    true_ev = np.asarray(true_tp, dtype=np.float64) >= threshold_mm
    hits = int(np.sum(pred_ev & true_ev))
    misses = int(np.sum((~pred_ev) & true_ev))
    false_alarms = int(np.sum(pred_ev & (~true_ev)))
    denom = hits + misses + false_alarms
    no_events = denom == 0
    csi = 0.0 if no_events else float(hits / denom)
    return {
        "csi": csi,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "true_events": int(np.sum(true_ev)),
        "pred_events": int(np.sum(pred_ev)),
        "no_events": no_events,
    }


def heavy_rain_csi(
    pred_tp: np.ndarray,
    true_tp: np.ndarray,
    threshold_mm: float = 10.0,
) -> float:
    """Return heavy-rain CSI at threshold_mm (0.0 when there are zero events)."""
    return float(heavy_rain_csi_details(pred_tp, true_tp, threshold_mm=threshold_mm)["csi"])


def day8_spatial_variance_ratio(pred_day8: np.ndarray, true_day8: np.ndarray) -> np.ndarray:
    """Per-channel spatial variance ratio var(pred)/var(true) of the day-8 field, averaged over windows."""
    p = np.asarray(pred_day8, dtype=np.float64)
    t = np.asarray(true_day8, dtype=np.float64)
    if p.ndim == 3:
        p = p[None, ...]
        t = t[None, ...]
    if p.shape != t.shape:
        raise ValueError(f"Shape mismatch: {p.shape} vs {t.shape}")
    vp = np.var(p, axis=(-2, -1))
    vt = np.var(t, axis=(-2, -1))
    ratio = vp / np.maximum(vt, 1e-12)
    return np.mean(ratio, axis=0)
