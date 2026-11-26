"""Tests for harmonic and empirical climatology (period 365, train-only fit)."""

from pathlib import Path

import numpy as np

from monsoon.climatology import (
    doy_frac_from_sincos,
    evaluate,
    fit,
    fit_empirical,
    load_harmonics,
    save_harmonics,
    sincos_from_doy,
)
from monsoon.data import split_indices


def test_doy_sincos_roundtrip() -> None:
    doys = np.arange(1, 367, dtype=np.int64)
    sc = sincos_from_doy(doys, period=365)
    assert sc.shape == (366, 2)
    decoded = doy_frac_from_sincos(sc[:, 0], sc[:, 1])
    expected = (doys % 365) / 365.0
    # Circular phase distance
    phase_err = np.abs(np.angle(np.exp(1j * 2.0 * np.pi * (decoded - expected)))) / (2.0 * np.pi)
    assert np.max(phase_err) < 1e-12
    # Day 366 wraps identically to day 1 under period 365
    assert np.allclose(sc[365], sc[0], atol=1e-12)


def test_harmonic_shape_and_save(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    T = 800
    doy = (np.arange(T) % 365) + 1
    y = rng.normal(size=(T, 6, 32, 36)).astype(np.float32)
    coef = fit(y, doy / 365.0, K=4, ridge=1e-3)
    assert coef.shape == (9, 6, 32, 36)

    out_path = tmp_path / "clim_harmonics.npz"
    save_harmonics(out_path, coef, K=4, period=365)
    with np.load(out_path) as z:
        assert set(z.files) == {"coef", "K", "period"}
    loaded_coef, k_val, period_val = load_harmonics(out_path)
    assert k_val == 4 and period_val == 365
    assert np.allclose(loaded_coef, coef)

    pred = evaluate(loaded_coef, np.array([1.0, 180.0]) / 365.0, K=4)
    assert pred.shape == (2, 6, 32, 36)
    assert np.isfinite(pred).all()


def test_no_val_test_leakage_in_fit() -> None:
    rng = np.random.default_rng(7)
    T = 500
    doy = (np.arange(T) % 365) + 1
    y_clean = rng.normal(size=(T, 6, 4, 4)).astype(np.float32)
    y_corrupt = y_clean.copy()

    splits = split_indices(T, 0.82, 0.09, 0.09)
    tr = splits["train"]
    held = np.concatenate([splits["val"], splits["test"]])
    y_corrupt[held] += 10000.0

    coef_clean = fit(y_clean[tr], doy[tr] / 365.0, K=4, ridge=1e-3)
    coef_corrupt = fit(y_corrupt[tr], doy[tr] / 365.0, K=4, ridge=1e-3)
    assert np.array_equal(coef_clean, coef_corrupt)

    emp_clean = fit_empirical(y_clean[tr], doy[tr], smooth_window=15, period=365)
    emp_corrupt = fit_empirical(y_corrupt[tr], doy[tr], smooth_window=15, period=365)
    assert emp_clean.shape == (365, 6, 4, 4)
    assert np.array_equal(emp_clean, emp_corrupt)
