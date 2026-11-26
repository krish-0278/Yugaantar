"""Shock causality guard: zero-shock identity, chessboard causal cone, and bounded response."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_cdt


def is_zero_perturbation(perturbation: np.ndarray | None, zero_eps: float = 1e-9) -> bool:
    """Return True if perturbation is None or max|perturbation| <= zero_eps."""
    if perturbation is None:
        return True
    arr = np.asarray(perturbation)
    if arr.size == 0:
        return True
    return bool(np.max(np.abs(arr)) <= zero_eps)


def causal_cone_mask(
    perturbation: np.ndarray,
    steps: int,
    cone_speed: float = 3.0,
    taper: float = 1.5,
) -> np.ndarray:
    """Compute [steps, H, W] Chebyshev causal-cone mask around non-zero shock cells."""
    delta = np.asarray(perturbation, dtype=np.float32)
    support = np.any(delta != 0.0, axis=0)  # [H, W]
    if not np.any(support):
        return np.zeros((steps, delta.shape[1], delta.shape[2]), dtype=np.float32)
    dist = distance_transform_cdt(~support, metric="chessboard").astype(np.float32)
    masks = [
        np.clip((float(cone_speed) * k - dist) / float(taper), 0.0, 1.0)
        for k in range(1, steps + 1)
    ]
    return np.stack(masks, axis=0).astype(np.float32)


def apply_shock_response(
    control: np.ndarray,
    pert: np.ndarray,
    perturbation: np.ndarray,
    channel_std_phys: np.ndarray,
    *,
    cone_speed: float = 3.0,
    taper: float = 1.5,
    cap_sigma: float = 4.0,
    t2m_min_k: float = 233.15,
    t2m_max_k: float = 333.0,
    use_cone: bool = True,
    use_cap: bool = True,
) -> np.ndarray:
    """Apply causal-cone mask and response cap to physical shock response pert - control."""
    resp = (pert - control).astype(np.float32, copy=True)
    if use_cone:
        mask = causal_cone_mask(perturbation, control.shape[0], cone_speed=cone_speed, taper=taper)
        resp = resp * mask[:, None, :, :]
    if use_cap:
        cap = float(cap_sigma) * np.asarray(channel_std_phys, dtype=np.float32)[None, :, None, None]
        resp = np.clip(resp, -cap, cap)

    out = (control + resp).astype(np.float32)
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    out[:, 0] = np.clip(out[:, 0], float(t2m_min_k), float(t2m_max_k))
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out
