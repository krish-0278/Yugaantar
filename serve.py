"""Offline HTTP inference server for Multiverse of Monsoon."""

from __future__ import annotations

import argparse
import base64
import threading
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import uvicorn

from monsoon.infer import Simulator


MODEL_NAME = "monsoon-twin-v1"
MIN_STEPS = 1
MAX_STEPS = 60

app = FastAPI()
_INFER_LOCK = threading.Lock()
_SIMULATOR: Simulator | None = None
_READY = False


def _warmup(sim: Simulator) -> None:
    """Run one 10-step control rollout and one 10-step shocked rollout before reporting ready."""
    rng = np.random.default_rng(0)
    init = np.zeros((3, 6, 32, 36), dtype=np.float32)
    init[:, 0] = 295.0
    init[:, 1] = 2.0
    init[:, 2] = 1.0
    init[:, 3] = 0.5
    init[:, 4] = 101000.0
    init[:, 5] = 30.0
    init = (init + rng.normal(0.0, 0.1, size=init.shape).astype(np.float32)).astype(np.float32)
    init[:, 1] = np.maximum(init[:, 1], 0.0)

    doy_sc = [0.0, 1.0]
    _ = sim.simulate(init, doy_sc, 10, None)
    pert = np.zeros((6, 32, 36), dtype=np.float32)
    pert[0, 4:10, 18:24] = 3.0
    _ = sim.simulate(init, doy_sc, 10, pert)


def _init_server() -> None:
    global _SIMULATOR, _READY
    sim = Simulator()
    _warmup(sim)
    _SIMULATOR = sim
    _READY = True


_init_server()


def _decode_array(raw: Any, expect_shape: tuple[int, ...], name: str) -> np.ndarray:
    """Decode nested lists (canonical) or {'b64', 'shape'} with strict shape/finiteness validation."""
    if isinstance(raw, dict):
        if "b64" not in raw or "shape" not in raw:
            raise ValueError(f"{name} dict must contain 'b64' and 'shape'")
        try:
            shape = tuple(int(s) for s in raw["shape"])
        except Exception as exc:
            raise ValueError(f"{name} invalid shape") from exc
        if shape != expect_shape:
            raise ValueError(f"{name} shape {shape} != expected {expect_shape}")
        try:
            buf = base64.b64decode(raw["b64"], validate=True)
        except Exception as exc:
            raise ValueError(f"{name} invalid base64 payload") from exc
        expected_bytes = int(np.prod(expect_shape)) * 4
        if len(buf) != expected_bytes:
            raise ValueError(f"{name} byte length does not match shape")
        arr = np.frombuffer(buf, dtype="<f4").reshape(expect_shape).astype(np.float32, copy=True)
    else:
        try:
            arr = np.asarray(raw, dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if arr.shape != expect_shape:
            raise ValueError(f"{name} shape {arr.shape} != expected {expect_shape}")

    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _parse_steps(raw: Any) -> int:
    if isinstance(raw, bool):
        raise ValueError("steps must be an integer between 1 and 60")
    if isinstance(raw, (int, np.integer)):
        val = int(raw)
    elif isinstance(raw, float) and raw.is_integer():
        val = int(raw)
    else:
        raise ValueError("steps must be an integer between 1 and 60")
    if val < MIN_STEPS or val > MAX_STEPS:
        raise ValueError(f"steps must be between {MIN_STEPS} and {MAX_STEPS}")
    return val


@app.post("/health")
@app.get("/health")
def health():
    if not _READY or _SIMULATOR is None:
        return JSONResponse(status_code=503, content={"status": "loading", "model_name": MODEL_NAME})
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    if not _READY or _SIMULATOR is None:
        return JSONResponse(status_code=503, content={"error": "model not ready"})

    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        for key in ("init_states", "doy_sincos", "steps"):
            if key not in body:
                raise ValueError(f"missing required field '{key}'")

        init = _decode_array(body["init_states"], (3, 6, 32, 36), "init_states")
        doy_sc = _decode_array(body["doy_sincos"], (2,), "doy_sincos")
        if float(doy_sc[0]) == 0.0 and float(doy_sc[1]) == 0.0:
            raise ValueError("doy_sincos cannot both be zero")
        steps = _parse_steps(body["steps"])

        raw_pert = body.get("perturbation")
        pert = None if raw_pert is None else _decode_array(raw_pert, (6, 32, 36), "perturbation")
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return JSONResponse(status_code=400, content={"error": "malformed JSON request"})

    try:
        with _INFER_LOCK:
            t0 = time.perf_counter()
            out = _SIMULATOR.simulate(init, doy_sc, steps, pert)
            inference_ms = (time.perf_counter() - t0) * 1000.0
        if not np.isfinite(out).all():
            return JSONResponse(status_code=500, content={"error": "non-finite model output"})
        return {"rollout": out.tolist(), "inference_ms": float(inference_ms)}
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return JSONResponse(status_code=500, content={"error": "internal inference error"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
