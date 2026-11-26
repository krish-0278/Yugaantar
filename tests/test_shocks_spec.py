"""Tests for reference judge shocks in monsoon/shocks_spec.py."""

import numpy as np

from monsoon.shocks_spec import (
    CH_MSL,
    CH_T2M,
    CH_TCWV,
    CH_TP,
    CH_U10,
    CH_V10,
    LAT_CENTRES,
    LON_CENTRES,
    box_mask,
    build_monsoon_stall,
    build_null,
    build_warm_ocean,
)


def _sample_init_last() -> np.ndarray:
    x = np.zeros((6, 32, 36), dtype=np.float32)
    x[CH_T2M] = 300.0
    x[CH_TP] = 5.0
    x[CH_U10] = 6.0
    x[CH_V10] = 4.0
    x[CH_MSL] = 101000.0
    x[CH_TCWV] = 40.0
    return x


def test_warm_ocean_spec() -> None:
    init_last = _sample_init_last()
    delta = build_warm_ocean(init_last)
    assert delta.shape == (6, 32, 36)

    mask = box_mask(8.0, 20.0, 82.0, 94.0)
    i_idx, j_idx = np.where(mask)
    print(
        f"warm_ocean lat idx range: {i_idx.min()}..{i_idx.max()} "
        f"(centres {LAT_CENTRES[i_idx.min()]}..{LAT_CENTRES[i_idx.max()]}), "
        f"lon idx range: {j_idx.min()}..{j_idx.max()} "
        f"(centres {LON_CENTRES[j_idx.min()]}..{LON_CENTRES[j_idx.max()]})"
    )
    print(f"warm_ocean spatial cells={int(mask.sum())}, nonzero tensor entries={int(np.count_nonzero(delta))}")

    assert int(mask.sum()) == 12 * 12
    assert np.allclose(delta[CH_T2M, mask], 3.0)
    assert np.allclose(delta[CH_TCWV, mask], 6.0)
    assert np.count_nonzero(delta[CH_T2M]) == 144
    assert np.count_nonzero(delta[CH_TCWV]) == 144
    for ch in (CH_TP, CH_U10, CH_V10, CH_MSL):
        assert np.count_nonzero(delta[ch]) == 0


def test_monsoon_stall_spec() -> None:
    init_last = _sample_init_last()
    delta = build_monsoon_stall(init_last)
    assert delta.shape == (6, 32, 36)

    mask = box_mask(8.0, 20.0, 65.0, 75.0)
    i_idx, j_idx = np.where(mask)
    print(
        f"monsoon_stall lat idx range: {i_idx.min()}..{i_idx.max()} "
        f"(centres {LAT_CENTRES[i_idx.min()]}..{LAT_CENTRES[i_idx.max()]}), "
        f"lon idx range: {j_idx.min()}..{j_idx.max()} "
        f"(centres {LON_CENTRES[j_idx.min()]}..{LON_CENTRES[j_idx.max()]})"
    )
    print(f"monsoon_stall spatial cells={int(mask.sum())}, nonzero tensor entries={int(np.count_nonzero(delta))}")

    assert int(mask.sum()) == 12 * 10
    assert np.allclose(delta[CH_U10, mask], -2.4)
    assert np.allclose(delta[CH_V10, mask], -1.6)
    assert np.count_nonzero(delta[CH_U10]) == 120
    assert np.count_nonzero(delta[CH_V10]) == 120
    for ch in (CH_T2M, CH_TP, CH_MSL, CH_TCWV):
        assert np.count_nonzero(delta[ch]) == 0


def test_null_spec() -> None:
    delta = build_null()
    print(f"null nonzero entries={int(np.count_nonzero(delta))}")
    assert delta.shape == (6, 32, 36)
    assert np.array_equal(delta, np.zeros((6, 32, 36), dtype=np.float32))
