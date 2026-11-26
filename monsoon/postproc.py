"""Post-processing pipeline shared by evaluation and serving.

Fixed execution order inside ``apply(...)``:
    beta -> from_anomaly -> kappa -> tp variance step (chosen method) -> event coupling -> clamps
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from monsoon.rollout import from_anomaly, load_rollout_context


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POSTPROC_PATH = ROOT / "weights" / "postproc.json"
CH_T2M = 0
CH_TP = 1


@lru_cache(maxsize=2)
def load_postproc(path: str | Path = DEFAULT_POSTPROC_PATH) -> dict[str, Any]:
    """Load post-processing parameters from weights/postproc.json."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _step_params(postproc_cfg: dict[str, Any], steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand 10-step beta and kappa schedules to arbitrary ``steps``."""
    beta_10 = np.asarray(postproc_cfg["beta"], dtype=np.float32)  # [10, 6]
    kappa_10 = np.asarray(postproc_cfg["kappa"], dtype=np.float32)  # [10]
    tau = float(postproc_cfg.get("beta_tail_tau", 10.0))

    fit_k = beta_10.shape[0]
    beta_steps = np.empty((steps, 6), dtype=np.float32)
    kappa_steps = np.empty((steps,), dtype=np.float32)
    tail_idx = np.empty((steps,), dtype=np.int64)

    for k in range(1, steps + 1):
        if k <= fit_k:
            idx = k - 1
            beta_steps[k - 1] = beta_10[idx]
            kappa_steps[k - 1] = kappa_10[idx]
            tail_idx[k - 1] = idx
        else:
            idx = fit_k - 1
            beta_steps[k - 1] = beta_10[idx] * math.exp(-(k - fit_k) / tau)
            kappa_steps[k - 1] = kappa_10[idx]
            tail_idx[k - 1] = idx
    return beta_steps, kappa_steps, tail_idx


def _apply_qm_step(tp_field: np.ndarray, q_pred: np.ndarray, q_true: np.ndarray, lam: float) -> np.ndarray:
    """Blend physical tp with monotonic piecewise-linear quantile map (preserving exact zeros)."""
    if lam <= 0.0:
        return tp_field
    mapped = np.interp(tp_field, q_pred, q_true).astype(np.float32)
    mapped = np.where(tp_field <= 0.0, 0.0, np.maximum(mapped, 0.0)).astype(np.float32)
    return ((1.0 - lam) * tp_field + lam * mapped).astype(np.float32)


def _apply_pattern_gain_step(tp_field: np.ndarray, gain: float, post_kappa: float) -> np.ndarray:
    """Apply pattern gain tp_g = max(0, m + g*(tp - m)) and post-clamp kappa adjustment."""
    if gain <= 1.0 + 1e-9:
        return tp_field
    m = float(np.mean(tp_field))
    tp_g = np.maximum(0.0, m + gain * (tp_field - m)).astype(np.float32)
    return np.maximum(tp_g * float(post_kappa), 0.0).astype(np.float32)


def _event_thresholds_for_steps(postproc_cfg: dict[str, Any], steps: int) -> np.ndarray:
    """Linearly interpolate event probability thresholds between fitted leads."""
    fit_leads = np.asarray(postproc_cfg["event_fit_leads"], dtype=np.float64)
    fit_thrs = np.asarray(postproc_cfg["event_thresholds"], dtype=np.float64)
    ks = np.arange(1, steps + 1, dtype=np.float64)
    return np.interp(ks, fit_leads, fit_thrs).astype(np.float32)


def apply(
    anomalies: np.ndarray,
    event_logits: np.ndarray,
    doy_last: int,
    steps: int,
    postproc_cfg: dict[str, Any] | None = None,
    ctx: dict[str, Any] | None = None,
    *,
    use_beta: bool = True,
    use_kappa: bool = True,
    use_tp_var: bool = True,
    use_event: bool | None = None,
    use_clamp: bool = True,
    tp_var_method: str | None = None,
) -> np.ndarray:
    """Apply fixed post-processing chain to model-space rollout outputs.

    Order:
        beta -> from_anomaly -> kappa -> tp variance step (chosen method) -> event coupling -> clamps
    """
    if postproc_cfg is None:
        postproc_cfg = load_postproc()
    if ctx is None:
        ctx = load_rollout_context()

    a = np.array(anomalies[:steps], dtype=np.float32, copy=True)
    ev_logits = np.asarray(event_logits[:steps], dtype=np.float32)

    beta_steps, kappa_steps, tail_idx = _step_params(postproc_cfg, steps)

    # 1. Beta shrinkage (output anomalies only)
    if use_beta:
        a = a * beta_steps[:, :, None, None]

    # 2. Convert anomalies to physical units via harmonic climatology + sigma_pix * a
    doy_targets = int(doy_last) + np.arange(1, steps + 1, dtype=np.int64)
    phys = from_anomaly(a, doy_targets, ctx=ctx).astype(np.float32, copy=True)

    # 3. Per-lead multiplicative tp bias correction kappa_k
    if use_kappa:
        phys[:, CH_TP] = np.maximum(phys[:, CH_TP] * kappa_steps[:, None, None], 0.0)

    # 4. tp variance step (chosen method: "pattern_gain" or "qm")
    if use_tp_var:
        method = tp_var_method or str(postproc_cfg.get("tp_var_method", "qm"))
        if method == "pattern_gain":
            g_10 = np.asarray(postproc_cfg["pattern_gain"], dtype=np.float32)
            pk_10 = np.asarray(postproc_cfg["pattern_post_kappa"], dtype=np.float32)
            for k in range(steps):
                idx = int(tail_idx[k])
                phys[k, CH_TP] = _apply_pattern_gain_step(phys[k, CH_TP], float(g_10[idx]), float(pk_10[idx]))
        elif method == "qm":
            lam_10 = np.asarray(postproc_cfg["qm_lambda"], dtype=np.float32)
            q_pred_all = np.asarray(postproc_cfg["qm_q_pred"], dtype=np.float32)
            q_true_all = np.asarray(postproc_cfg["qm_q_true"], dtype=np.float32)
            for k in range(steps):
                idx = int(tail_idx[k])
                lam = float(lam_10[idx])
                if lam > 0.0:
                    phys[k, CH_TP] = _apply_qm_step(phys[k, CH_TP], q_pred_all[idx], q_true_all[idx], lam)
        elif method != "none":
            raise ValueError(f"Unknown tp_var_method: {method}")

    # 5. Event coupling
    do_event = bool(postproc_cfg.get("use_event_coupling", False)) if use_event is None else bool(use_event)
    if do_event:
        probs = 1.0 / (1.0 + np.exp(-np.clip(ev_logits, -40.0, 40.0)))
        thrs = _event_thresholds_for_steps(postproc_cfg, steps)[:, None, None]
        hi_mm = float(postproc_cfg.get("event_high_mm", 10.5))
        lo_mm = float(postproc_cfg.get("event_low_mm", 9.5))
        phys[:, CH_TP] = np.where(
            probs > thrs,
            np.maximum(phys[:, CH_TP], hi_mm),
            np.minimum(phys[:, CH_TP], lo_mm),
        ).astype(np.float32)

    # 6. Physical clamps
    if use_clamp:
        lo = np.asarray(postproc_cfg["clamp_lo"], dtype=np.float32)
        hi = np.asarray(postproc_cfg["clamp_hi"], dtype=np.float32)
        phys = np.clip(phys, lo[None, :, None, None], hi[None, :, None, None]).astype(np.float32)
        phys[:, CH_TP] = np.maximum(phys[:, CH_TP], 0.0)
        t_min = float(postproc_cfg.get("t2m_min_k", 233.15))
        t_max = float(postproc_cfg.get("t2m_max_k", 333.0))
        phys[:, CH_T2M] = np.clip(phys[:, CH_T2M], t_min, t_max)

    return phys
