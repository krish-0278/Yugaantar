"""FastAPI TestClient tests for serve.py contract, edge cases, determinism, and error handling."""

import base64
import json

from fastapi.testclient import TestClient
import numpy as np

from monsoon.climatology import sincos_from_doy
from serve import app


client = TestClient(app)


def _sample_init() -> np.ndarray:
    rng = np.random.default_rng(123)
    base = np.array([296.0, 3.0, 1.5, 0.5, 101000.0, 35.0], dtype=np.float32)[None, :, None, None]
    noise = rng.normal(0.0, 0.5, size=(3, 6, 32, 36)).astype(np.float32)
    arr = (base + noise).astype(np.float32)
    arr[:, 1] = np.maximum(arr[:, 1], 0.0)
    return arr


def _b64(a: np.ndarray) -> dict:
    c = np.ascontiguousarray(a, dtype="<f4")
    return {"b64": base64.b64encode(c.tobytes()).decode("ascii"), "shape": list(c.shape)}


def test_health_endpoints() -> None:
    for fn in (client.post, client.get):
        r = fn("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ready", "model_name": "monsoon-twin-v1"}


def test_simulate_steps_determinism_and_zero_shock_identity() -> None:
    init = _sample_init()
    doy_sc = sincos_from_doy(160, period=365).tolist()

    r_none_1 = client.post(
        "/simulate",
        json={"init_states": init.tolist(), "doy_sincos": doy_sc, "steps": 10, "perturbation": None},
    )
    r_none_2 = client.post(
        "/simulate",
        json={"init_states": init.tolist(), "doy_sincos": doy_sc, "steps": 10, "perturbation": None},
    )
    assert r_none_1.status_code == 200 and r_none_2.status_code == 200
    js1, js2 = r_none_1.json(), r_none_2.json()
    assert js1["inference_ms"] > 0.0

    arr1 = np.asarray(js1["rollout"], dtype=np.float32)
    arr2 = np.asarray(js2["rollout"], dtype=np.float32)
    assert arr1.shape == (10, 6, 32, 36)
    assert np.isfinite(arr1).all()
    assert float(arr1[:, 1].min()) >= 0.0
    assert float(arr1[:, 0].min()) >= 233.15 and float(arr1[:, 0].max()) <= 333.0
    assert np.array_equal(arr1, arr2)

    # All-zeros perturbation and 1e-12 noise perturbation must be bitwise equal to None
    for pert in (
        np.zeros((6, 32, 36), dtype=np.float32),
        np.full((6, 32, 36), 1e-12, dtype=np.float32),
    ):
        r_p = client.post(
            "/simulate",
            json={"init_states": init.tolist(), "doy_sincos": doy_sc, "steps": 10, "perturbation": pert.tolist()},
        )
        assert r_p.status_code == 200
        arr_p = np.asarray(r_p.json()["rollout"], dtype=np.float32)
        assert np.array_equal(arr1, arr_p)

    # Different step horizons 1, 10, 30, 60
    for st in (1, 10, 30, 60):
        r_s = client.post(
            "/simulate",
            json={"init_states": init.tolist(), "doy_sincos": doy_sc, "steps": st, "perturbation": None},
        )
        assert r_s.status_code == 200
        out_s = np.asarray(r_s.json()["rollout"], dtype=np.float32)
        assert out_s.shape == (st, 6, 32, 36)
        assert np.isfinite(out_s).all()
        assert float(out_s[:, 1].min()) >= 0.0


def test_simulate_validation_400() -> None:
    init = _sample_init()
    doy_sc = sincos_from_doy(160, period=365).tolist()

    # Invalid step counts 0 and 61
    for bad_steps in (0, 61, -3, "10", True):
        r = client.post(
            "/simulate",
            json={"init_states": init.tolist(), "doy_sincos": doy_sc, "steps": bad_steps, "perturbation": None},
        )
        assert r.status_code == 400
        assert "error" in r.json()

    # Wrong shapes
    bad_payloads = [
        {"init_states": init[:2].tolist(), "doy_sincos": doy_sc, "steps": 10, "perturbation": None},
        {"init_states": init.tolist(), "doy_sincos": [0.5, 0.5, 0.5], "steps": 10, "perturbation": None},
        {"init_states": init.tolist(), "doy_sincos": [0.0, 0.0], "steps": 10, "perturbation": None},
        {"init_states": init.tolist(), "doy_sincos": doy_sc, "steps": 10, "perturbation": np.zeros((6, 32, 35)).tolist()},
    ]
    for p in bad_payloads:
        r = client.post("/simulate", json=p)
        assert r.status_code == 400
        assert "error" in r.json()

    # NaN input via base64 and raw JSON
    init_nan = init.copy()
    init_nan[0, 0, 0, 0] = np.nan
    r_nan_b64 = client.post(
        "/simulate",
        json={"init_states": _b64(init_nan), "doy_sincos": doy_sc, "steps": 10, "perturbation": None},
    )
    assert r_nan_b64.status_code == 400
    assert "error" in r_nan_b64.json()

    lst = init.tolist()
    lst[0][0][0][0] = "NaN"
    r_nan_json = client.post(
        "/simulate",
        content=json.dumps({"init_states": lst, "doy_sincos": doy_sc, "steps": 10, "perturbation": None}).replace('"NaN"', "NaN"),
        headers={"Content-Type": "application/json"},
    )
    assert r_nan_json.status_code == 400
    assert "error" in r_nan_json.json()
