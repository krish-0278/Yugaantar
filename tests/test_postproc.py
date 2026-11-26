"""Tests for monsoon/postproc.py fixed order, switchable flags, Method G/QM, and raw identity."""

import numpy as np

from monsoon.postproc import apply as apply_postproc
from monsoon.rollout import from_anomaly, load_rollout_context


def _toy_postproc_cfg(method: str = "pattern_gain") -> dict:
    return {
        "beta": np.full((10, 6), 0.8).tolist(),
        "beta_tail_tau": 10.0,
        "kappa": [1.05] * 10,
        "tp_var_method": method,
        "pattern_gain": [1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.4, 1.4],
        "pattern_post_kappa": [1.0, 1.0, 1.0, 1.0, 0.98, 0.96, 0.94, 0.92, 0.92, 0.92],
        "qm_lambda": [0.5] * 10,
        "qm_q_pred": [np.linspace(0.0, 50.0, 16).tolist() for _ in range(10)],
        "qm_q_true": [np.linspace(0.0, 60.0, 16).tolist() for _ in range(10)],
        "event_fit_leads": [1, 4, 8],
        "event_thresholds": [0.6, 0.55, 0.5],
        "event_high_mm": 10.5,
        "event_low_mm": 9.5,
        "use_event_coupling": True,
        "clamp_lo": [233.15, 0.0, -30.0, -30.0, 95000.0, 0.0],
        "clamp_hi": [333.0, 300.0, 30.0, 30.0, 106000.0, 100.0],
        "t2m_min_k": 233.15,
        "t2m_max_k": 333.0,
    }


def test_postproc_all_flags_off_reproduces_raw() -> None:
    ctx = load_rollout_context()
    pp = _toy_postproc_cfg("pattern_gain")
    rng = np.random.default_rng(12)
    anom = rng.normal(0.0, 1.5, size=(12, 6, 32, 36)).astype(np.float32)
    ev = rng.normal(0.0, 2.0, size=(12, 32, 36)).astype(np.float32)
    doy_last = 160

    out_flags_off = apply_postproc(
        anom, ev, doy_last, 12, pp, ctx,
        use_beta=False, use_kappa=False, use_tp_var=False, use_event=False, use_clamp=False,
    )
    expected_raw = from_anomaly(anom, doy_last + np.arange(1, 13, dtype=np.int64), ctx=ctx)
    assert np.array_equal(out_flags_off, expected_raw)


def test_postproc_methods_and_validity() -> None:
    ctx = load_rollout_context()
    anom = np.zeros((15, 6, 32, 36), dtype=np.float32)
    anom[:, 0] = 2.0
    anom[:, 1, :16, :18] = 1.5
    ev = np.full((15, 32, 36), -5.0, dtype=np.float32)
    ev[:, 10, 10] = 5.0

    for method in ("pattern_gain", "qm"):
        pp = _toy_postproc_cfg(method)
        out = apply_postproc(anom, ev, doy_last=160, steps=15, postproc_cfg=pp, ctx=ctx)
        assert out.shape == (15, 6, 32, 36)
        assert np.isfinite(out).all()
        assert float(out[:, 1].min()) >= 0.0
        assert float(out[:, 0].min()) >= 233.15
        assert float(out[:, 0].max()) <= 333.0
        assert float(out[0, 1, 10, 10]) >= 10.5
        assert float(out[0, 1, 0, 0]) <= 9.5

        out_no_ev = apply_postproc(anom, ev, doy_last=160, steps=15, postproc_cfg=pp, ctx=ctx, use_event=False)
        assert not np.array_equal(out[:, 1], out_no_ev[:, 1])
