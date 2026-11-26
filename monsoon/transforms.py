"""Pure-NumPy space conversions and normalization helpers for training and serving."""

from __future__ import annotations

from pathlib import Path

import numpy as np


CH_TP = 1


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


def load_norm(path: str | Path) -> dict[str, np.ndarray]:
    """Load normalization arrays and training bounds from npz."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def build_static_norm(static: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    """Return [2, H, W] float32 with land_sea_mask and z-scored orography_m."""
    lsm = static[0].astype(np.float32)
    oro = ((static[1] - float(norm["oro_mean"])) / float(norm["oro_std"])).astype(np.float32)
    return np.stack([lsm, oro], axis=0)
