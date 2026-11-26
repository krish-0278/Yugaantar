"""Tests for metrics: lat-weighted RMSE, skill, heavy-rain CSI, and day-8 variance ratio."""

import numpy as np

from monsoon.metrics import (
    day8_spatial_variance_ratio,
    heavy_rain_csi,
    heavy_rain_csi_details,
    lat_weighted_rmse,
    lat_weights,
    skill_score,
)


def test_skill_baseline_vs_self_and_perfect() -> None:
    rng = np.random.default_rng(0)
    lat = np.arange(6.5, 38.0, 1.0)
    w = lat_weights(lat)
    assert np.isclose(w.mean(), 1.0)

    true = rng.normal(size=(4, 6, 32, 36))
    pers = true + rng.normal(scale=1.0, size=true.shape)
    clim = true + rng.normal(scale=1.5, size=true.shape)

    rmse_pers = lat_weighted_rmse(pers, true, w)
    rmse_clim = lat_weighted_rmse(clim, true, w)
    best_pred = np.where((rmse_pers <= rmse_clim)[None, :, None, None], pers, clim)
    rmse_best = lat_weighted_rmse(best_pred, true, w)

    ss_self, ss_self_mean = skill_score(rmse_best, rmse_pers, rmse_clim)
    assert np.allclose(ss_self, 0.0)
    assert np.isclose(ss_self_mean, 0.0)

    rmse_perf = lat_weighted_rmse(true, true, w)
    assert np.allclose(rmse_perf, 0.0)
    ss_perf, ss_perf_mean = skill_score(rmse_perf, rmse_pers, rmse_clim)
    assert np.allclose(ss_perf, 1.0)
    assert np.isclose(ss_perf_mean, 1.0)


def test_heavy_rain_csi_hand_checked() -> None:
    # 2 hits (12 vs 15, 10 vs 10), 1 miss (9 vs 11), 1 false alarm (14 vs 3), 2 correct negatives (0 vs 2, 9.9 vs 9.9)
    true_tp = np.array([[15.0, 10.0, 11.0], [3.0, 2.0, 9.9]])
    pred_tp = np.array([[12.0, 10.0, 9.0], [14.0, 0.0, 9.9]])
    details = heavy_rain_csi_details(pred_tp, true_tp, threshold_mm=10.0)
    assert details["hits"] == 2
    assert details["misses"] == 1
    assert details["false_alarms"] == 1
    assert np.isclose(details["csi"], 2.0 / 4.0)
    assert np.isclose(heavy_rain_csi(pred_tp, true_tp, threshold_mm=10.0), 0.5)

    # Zero-event case returns 0.0
    zero_details = heavy_rain_csi_details(np.full((2, 2), 5.0), np.full((2, 2), 8.0), threshold_mm=10.0)
    assert zero_details["no_events"] is True
    assert zero_details["csi"] == 0.0


def test_day8_spatial_variance_ratio_toy() -> None:
    rng = np.random.default_rng(1)
    true8 = rng.normal(loc=10.0, scale=2.0, size=(3, 6, 32, 36))
    # Constant shift leaves spatial variance unchanged -> ratio 1.0
    ratio_same = day8_spatial_variance_ratio(true8 + 5.0, true8)
    assert np.allclose(ratio_same, 1.0)
    # Scaling spatial anomalies by 0.5 -> variance ratio 0.25
    ratio_half = day8_spatial_variance_ratio(0.5 * true8, true8)
    assert np.allclose(ratio_half, 0.25)
    # Scaling by 2.0 -> variance ratio 4.0
    ratio_double = day8_spatial_variance_ratio(2.0 * true8, true8)
    assert np.allclose(ratio_double, 4.0)
