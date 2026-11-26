"""Unit tests for climate impact proxy indices (monsoon/impact.py)."""

from __future__ import annotations

import numpy as np

from monsoon.impact import compute_impacts_pair, compute_rollout_impacts, rollout_climatology_phys


def test_heatwave_consecutive_days_and_shares_synthetic() -> None:
    # Small 2x2 grid with 2 land cells at row 0 and 2 ocean cells at row 1
    lat = np.array([10.5, 20.5], dtype=np.float64)
    land_mask = np.array([[1.0, 0.8], [0.2, 0.0]], dtype=np.float32)

    clim = np.zeros((4, 6, 2, 2), dtype=np.float32)
    clim[:, 0] = 300.0  # t2m
    clim[:, 1] = 10.0   # tp -> 40 mm cumulative over 4 days

    pred = clim.copy()
    # Land cell (0, 0): hot days 0,1 (consecutive=2 -> 2 cell-days) and isolated hot day 3 (not counted)
    pred[0, 0, 0, 0] = 303.5
    pred[1, 0, 0, 0] = 303.0
    pred[3, 0, 0, 0] = 304.0
    # Land cell (0, 1): hot days 0,2 (isolated -> 0 cell-days); ocean cell (1,0) hot all 4 days (ignored)
    pred[0, 0, 0, 1] = 305.0
    pred[2, 0, 0, 1] = 305.0
    pred[:, 0, 1, 0] = 310.0

    # Heavy rain: 1 land cell-day >= 115.6 (also >= 64.5), plus 1 >= 64.5
    pred[0, 1, 0, 0] = 120.0
    pred[1, 1, 0, 0] = 70.0
    # Land cell (0, 1) dry (0 mm total -> 100% deficit > 20%)
    pred[:, 1, 0, 1] = 0.0

    res = compute_rollout_impacts(pred, clim, land_mask, lat)
    assert res["heatwave_cell_days"] == 2
    assert np.isclose(res["heavy_rain_64_share"], 2.0 / 8.0)
    assert np.isclose(res["heavy_rain_115_share"], 1.0 / 8.0)
    assert np.isclose(res["rain_deficit_share"], 0.5)
    assert 0.0 < res["crop_stress_csi"] <= 1.0

    pair = compute_impacts_pair(pred, pred, clim, land_mask, lat)
    assert pair["steps"] == 4
    assert pair["n_land_cells"] == 2
    assert pair["delta"]["heatwave_cell_days"]["abs"] == 0.0


def test_rollout_climatology_phys_wraps_and_uses_expm1() -> None:
    # Constant harmonic coef in model space (mean log1p(10.0) on tp)
    coef = np.zeros((9, 6, 32, 36), dtype=np.float32)
    coef[0, 0] = 298.0
    coef[0, 1] = float(np.log1p(10.0))

    c_wrap = rollout_climatology_phys(coef, doy_last=364, steps=4, harm_k=4, period=365)
    c_start = rollout_climatology_phys(coef, doy_last=0, steps=2, harm_k=4, period=365)
    assert c_wrap.shape == (4, 6, 32, 36)
    assert np.isfinite(c_wrap).all()
    assert np.allclose(c_wrap[:, 1], 10.0, atol=1e-4)
    # Day 2 after 364 is doy 1 == Day 1 after 0
    assert np.allclose(c_wrap[1:], np.repeat(c_start[0:1], 3, axis=0), atol=1e-4)
