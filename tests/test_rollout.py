"""Tests for monsoon/rollout.py: determinism, 30-step real-data finiteness, and zero-delta equivalence."""

from pathlib import Path
import numpy as np
import torch

from monsoon.data import load_dataset
from monsoon.model import MonsoonNet
from monsoon.rollout import from_anomaly, load_rollout_context, rollout


ROOT = Path(__file__).resolve().parent.parent


def _load_stageA_model() -> MonsoonNet:
    ckpt = torch.load(ROOT / "weights" / "model_stageA.pt", map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    m = MonsoonNet(
        cin=int(mc["cin"]),
        cout=int(mc["cout"]),
        channels=int(mc["channels"]),
        n_blocks=int(mc["n_blocks"]),
        expansion=int(mc["expansion"]),
    )
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m


def test_rollout_deterministic_zero_delta_and_30steps() -> None:
    model = _load_stageA_model()
    ctx = load_rollout_context()
    ds = load_dataset()
    t0 = 5100  # inside held-out region
    init_phys = ds["data"][t0 - 2 : t0 + 1]
    doy_last = int(ds["doy"][t0])

    anom1, ev1 = rollout(model, init_phys, doy_last, steps=30, delta=None, ctx=ctx)
    anom2, ev2 = rollout(model, init_phys, doy_last, steps=30, delta=None, ctx=ctx)
    anom_z, ev_z = rollout(model, init_phys, doy_last, steps=30, delta=np.zeros((6, 32, 36), dtype=np.float32), ctx=ctx)

    assert anom1.shape == (30, 6, 32, 36)
    assert ev1.shape == (30, 32, 36)
    assert np.array_equal(anom1, anom2)
    assert np.array_equal(ev1, ev2)
    assert np.array_equal(anom1, anom_z)
    assert np.array_equal(ev1, ev_z)

    assert np.isfinite(anom1).all()
    assert np.isfinite(ev1).all()
    assert np.max(np.abs(anom1)) <= 5.0 + 1e-6

    doys = doy_last + np.arange(1, 31, dtype=np.int64)
    phys = from_anomaly(anom1, doys, ctx=ctx)
    assert phys.shape == (30, 6, 32, 36)
    assert np.isfinite(phys).all()
    assert float(phys[:, 1].min()) >= 0.0
