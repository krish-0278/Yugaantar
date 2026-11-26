"""Autoregressive rollout shared by Stage B training, tuning, evaluation, and serving."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from monsoon.climatology import DEFAULT_PERIOD, evaluate as eval_harmonic, load_harmonics
from monsoon.data import (
    build_static_norm,
    from_model_space,
    load_norm,
    to_model_space,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HARMONICS = ROOT / "weights" / "clim_harmonics.npz"
DEFAULT_NORM = ROOT / "weights" / "norm.npz"
DEFAULT_STATIC = ROOT / "weights" / "static.npz"


def build_clim_norm_table(
    harm_coef: np.ndarray,
    norm: dict[str, np.ndarray],
    harm_k: int = 4,
    period: int = DEFAULT_PERIOD,
) -> np.ndarray:
    """Build [period, 6, H, W] normalized climatology table for steps 1..period."""
    doys = np.arange(1, period + 1, dtype=np.float64)
    clim_365 = eval_harmonic(harm_coef, doys / float(period), K=harm_k)
    cmean = norm["chan_mean_model"][None, :, None, None]
    cstd = norm["chan_std_model"][None, :, None, None]
    return ((clim_365 - cmean) / cstd).astype(np.float32)


@lru_cache(maxsize=2)
def load_rollout_context(
    harmonics_path: str | Path = DEFAULT_HARMONICS,
    norm_path: str | Path = DEFAULT_NORM,
    static_path: str | Path = DEFAULT_STATIC,
) -> dict[str, Any]:
    """Load bundled harmonics, normalization stats, and static fields once."""
    harm_coef, harm_k, period = load_harmonics(harmonics_path)
    norm = load_norm(norm_path)
    with np.load(static_path) as z:
        static = z["static"].astype(np.float32)
    static_norm = build_static_norm(static, norm)
    clim_norm_table = build_clim_norm_table(harm_coef, norm, harm_k=harm_k, period=period)
    return {
        "harm_coef": harm_coef,
        "harm_k": int(harm_k),
        "period": int(period),
        "norm": norm,
        "static": static,
        "static_norm": static_norm,
        "clim_norm_table": clim_norm_table,
    }


def rollout_tensor(
    model: torch.nn.Module,
    a_init_hist: torch.Tensor,
    doy_last: torch.Tensor | int,
    steps: int,
    clim_norm_table_t: torch.Tensor,
    static_norm_t: torch.Tensor,
    period: int = DEFAULT_PERIOD,
    anom_clamp: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable K-step autoregressive rollout in normalized anomaly space.

    Args:
        model: MonsoonNet returning (a_next [B,6,H,W], event_logit [B,1,H,W]).
        a_init_hist: [B, 3, 6, H, W] normalized anomalies, oldest first.
        doy_last: int or [B] int64 tensor for the last input day.
        steps: number of forecast days (>= 1).
        clim_norm_table_t: [period, 6, H, W] tensor on the model device.
        static_norm_t: [2, H, W] tensor on the model device.

    Returns:
        anomalies: [B, steps, 6, H, W] clamped to [-anom_clamp, +anom_clamp] (unshrunk).
        event_logits: [B, steps, H, W] heavy-rain logits.
    """
    bsz, _, _, h, w = a_init_hist.shape
    device = a_init_hist.device
    if isinstance(doy_last, int):
        doy_last_t = torch.full((bsz,), doy_last, dtype=torch.long, device=device)
    else:
        doy_last_t = doy_last.to(device=device, dtype=torch.long).view(bsz)

    static_b = static_norm_t.unsqueeze(0).expand(bsz, -1, -1, -1)
    a_m2 = a_init_hist[:, 0]
    a_m1 = a_init_hist[:, 1]
    a_0 = a_init_hist[:, 2]

    anomalies = []
    event_logits = []
    for k in range(1, steps + 1):
        doy_target = doy_last_t + k
        idx = torch.remainder(doy_target - 1, period)
        clim_norm_k = clim_norm_table_t[idx]  # [B, 6, H, W]

        ang = (2.0 * math.pi / float(period)) * doy_target.to(a_init_hist.dtype)
        sc = torch.stack([torch.sin(ang), torch.cos(ang)], dim=1).view(bsz, 2, 1, 1).expand(bsz, 2, h, w)

        diff = a_0 - a_m1
        feat = torch.cat([a_m2, a_m1, a_0, diff, clim_norm_k, static_b, sc], dim=1)
        a_next, ev_logit = model(feat, a_0)
        a_next = torch.clamp(a_next, -anom_clamp, anom_clamp)

        anomalies.append(a_next)
        event_logits.append(ev_logit[:, 0])
        a_m2, a_m1, a_0 = a_m1, a_0, a_next

    return torch.stack(anomalies, dim=1), torch.stack(event_logits, dim=1)


def from_anomaly(
    anomalies: np.ndarray,
    doy_target: int | np.ndarray,
    ctx: dict[str, Any] | None = None,
) -> np.ndarray:
    """Convert normalized anomaly field(s) to physical units via harmonic_clim + sigma_pix * a.

    Supports [6, 32, 36] with scalar doy_target, or [steps, 6, 32, 36] with [steps] doy_target.
    """
    if ctx is None:
        ctx = load_rollout_context()
    a = np.asarray(anomalies, dtype=np.float32)
    sigma_pix = ctx["norm"]["sigma_pix"]
    harm_coef = ctx["harm_coef"]
    harm_k = ctx["harm_k"]
    period = ctx["period"]

    if a.ndim == 3:
        d_arr = np.array([int(doy_target)], dtype=np.float64)
        clim = eval_harmonic(harm_coef, d_arr / float(period), K=harm_k)[0]
        y = clim + sigma_pix * a
        return from_model_space(y)
    if a.ndim == 4:
        d_arr = np.atleast_1d(np.asarray(doy_target, dtype=np.float64))
        if d_arr.shape[0] != a.shape[0]:
            raise ValueError(f"doy_target length {d_arr.shape[0]} != steps {a.shape[0]}")
        clim = eval_harmonic(harm_coef, d_arr / float(period), K=harm_k)
        y = clim + sigma_pix[None, :, :, :] * a
        return from_model_space(y)
    raise ValueError(f"Expected 3D or 4D anomaly array, got shape {a.shape}")


@torch.inference_mode()
def rollout(
    model: torch.nn.Module,
    init_phys: np.ndarray,
    doy_last: int,
    steps: int,
    delta: np.ndarray | None = None,
    ctx: dict[str, Any] | None = None,
    anom_clamp: float = 5.0,
    input_clip: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic model-space rollout from physical init_phys [3, 6, 32, 36].

    If delta is provided and non-zero, it is added to init_phys[-1] in physical space and tp
    is clamped >= 0 before conversion to model space. Exact zero delta takes the identical
    code path as delta=None.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if ctx is None:
        ctx = load_rollout_context()

    init_arr = np.array(init_phys, dtype=np.float32, copy=True)
    if init_arr.shape != (3, 6, 32, 36):
        raise ValueError(f"Expected init_phys shape (3, 6, 32, 36), got {init_arr.shape}")

    if delta is not None:
        d_arr = np.asarray(delta, dtype=np.float32)
        if d_arr.shape != (6, 32, 36):
            raise ValueError(f"Expected delta shape (6, 32, 36), got {d_arr.shape}")
        if np.any(d_arr != 0.0):
            init_arr[-1] = init_arr[-1] + d_arr
            init_arr[-1, 1] = np.maximum(init_arr[-1, 1], 0.0)

    y_init = to_model_space(init_arr)
    doy_init = np.array([doy_last - 2, doy_last - 1, doy_last], dtype=np.float64)
    clim_init = eval_harmonic(ctx["harm_coef"], doy_init / float(ctx["period"]), K=ctx["harm_k"])
    sigma_pix = ctx["norm"]["sigma_pix"]
    a_init = np.clip((y_init - clim_init) / sigma_pix[None, :, :, :], -input_clip, input_clip).astype(np.float32)

    device = next(model.parameters()).device
    a_hist_t = torch.from_numpy(a_init).unsqueeze(0).to(device)
    clim_table_t = torch.from_numpy(ctx["clim_norm_table"]).to(device)
    static_norm_t = torch.from_numpy(ctx["static_norm"]).to(device)

    model.eval()
    anom_t, ev_t = rollout_tensor(
        model=model,
        a_init_hist=a_hist_t,
        doy_last=int(doy_last),
        steps=int(steps),
        clim_norm_table_t=clim_table_t,
        static_norm_t=static_norm_t,
        period=ctx["period"],
        anom_clamp=anom_clamp,
    )
    return anom_t[0].cpu().numpy().astype(np.float32), ev_t[0].cpu().numpy().astype(np.float32)
