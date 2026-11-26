"""FastAPI dashboard backend and proxy server for Multiverse of Monsoon."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monsoon.climatology import evaluate, load_harmonics, sincos_from_doy
from monsoon.impact import compute_impacts_pair, rollout_climatology_phys
from monsoon.shocks_spec import (
    LAT_CENTRES,
    LON_CENTRES,
    box_mask,
    build_monsoon_stall,
    build_null,
    build_warm_ocean,
)
from monsoon.transforms import from_model_space


WEIGHTS_DIR = ROOT / "weights"
STATIC_DIR = Path(__file__).resolve().parent / "static"
GRID_PATH = STATIC_DIR / "assets" / "grid.json"
HARM_PATH = WEIGHTS_DIR / "clim_harmonics.npz"
STD_PATH = WEIGHTS_DIR / "channel_std_phys.json"
NC_PATH = ROOT / "data" / "india_climate_train.nc"

CHANNELS = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
CHANNEL_INFO = [
    {"key": "t2m", "label": "2m Temperature", "short": "t2m", "unit": "K"},
    {"key": "tp", "label": "Total Precipitation", "short": "tp", "unit": "mm/day"},
    {"key": "u10", "label": "10m Zonal Wind (u10)", "short": "u10", "unit": "m/s"},
    {"key": "v10", "label": "10m Meridional Wind (v10)", "short": "v10", "unit": "m/s"},
    {"key": "msl", "label": "Mean Sea-Level Pressure", "short": "msl", "unit": "Pa"},
    {"key": "tcwv", "label": "Total Column Water Vapour", "short": "tcwv", "unit": "kg/m^2"},
]

PRESETS_META = [
    {
        "id": "null",
        "label": "Null shock (zero control)",
        "summary": "All-zero perturbation across all channels; verifies deterministic zero difference.",
        "box": None,
    },
    {
        "id": "warm_bay_of_bengal",
        "label": "Warm Bay of Bengal",
        "summary": "+3.0 K t2m, +15% tcwv over 8-20 N, 82-94 E.",
        "box": {"lat_min": 8.0, "lat_max": 20.0, "lon_min": 82.0, "lon_max": 94.0},
    },
    {
        "id": "weaker_arabian_sea_wind",
        "label": "Weaker Arabian Sea wind",
        "summary": "-40% u10, -40% v10 over 8-20 N, 65-75 E.",
        "box": {"lat_min": 8.0, "lat_max": 20.0, "lon_min": 65.0, "lon_max": 75.0},
    },
    {
        "id": "custom",
        "label": "Custom shock",
        "summary": "Localized circular or smooth perturbation placed anywhere on the grid.",
        "box": None,
    },
]

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_STEPS = 30
MIN_STEPS = 1
MIN_DOY = 1
MAX_DOY = 365


def _load_static_bundle() -> dict[str, Any]:
    with open(GRID_PATH, "r", encoding="utf-8") as f:
        grid = json.load(f)
    with open(STD_PATH, "r", encoding="utf-8") as f:
        std_data = json.load(f)
    harm_coef, harm_k, harm_period = load_harmonics(HARM_PATH)

    lat = np.asarray(grid["lat"], dtype=np.float64)
    lon = np.asarray(grid["lon"], dtype=np.float64)
    land_sea = np.asarray(grid["land_sea_mask"], dtype=np.float32)
    land_mask = land_sea >= 0.5
    std_phys = np.asarray(std_data["std"], dtype=np.float64)

    return {
        "grid": grid,
        "lat": lat,
        "lon": lon,
        "land_mask": land_mask,
        "std_phys": std_phys,
        "harm_coef": harm_coef,
        "harm_k": int(harm_k),
        "harm_period": int(harm_period),
    }


BUNDLE = _load_static_bundle()

app = FastAPI(title="Multiverse of Monsoon Dashboard")
app.state.model_url = "http://127.0.0.1:8000"


def _parse_int(raw: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{name} must be an integer between {lo} and {hi}")
    if isinstance(raw, (int, np.integer)):
        val = int(raw)
    elif isinstance(raw, float) and raw.is_integer():
        val = int(raw)
    elif isinstance(raw, str):
        try:
            val = int(raw.strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer between {lo} and {hi}") from exc
    else:
        raise ValueError(f"{name} must be an integer between {lo} and {hi}")
    if val < lo or val > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return val


def _parse_float(raw: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number between {lo} and {hi}")
    val = float(raw)
    if not np.isfinite(val) or val < lo or val > hi:
        raise ValueError(f"{name} must be finite and between {lo} and {hi}")
    return val


def _validate_init_states(raw: Any, name: str = "init_states") -> np.ndarray:
    try:
        arr = np.asarray(raw, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if arr.shape != (3, 6, 32, 36):
        raise ValueError(f"{name} shape {arr.shape} != expected (3, 6, 32, 36)")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _build_climatology_init(doy: int) -> tuple[np.ndarray, list[int]]:
    period = BUNDLE["harm_period"]
    harm_coef = BUNDLE["harm_coef"]
    harm_k = BUNDLE["harm_k"]

    init_days = [((doy - 1 - offset) % period) + 1 for offset in (2, 1, 0)]
    doy_frac = np.asarray(init_days, dtype=np.float64) / float(period)
    y_mod = evaluate(harm_coef, doy_frac, K=harm_k)
    arr = from_model_space(y_mod).astype(np.float32)
    arr[:, 1] = np.maximum(arr[:, 1], 0.0)
    return arr, init_days


def _build_custom_perturbation(shock_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if "center_lat" not in shock_cfg or "center_lon" not in shock_cfg:
        raise ValueError("custom shock requires center_lat and center_lon")

    center_lat = _parse_float(shock_cfg["center_lat"], "center_lat", float(LAT_CENTRES[0]), float(LAT_CENTRES[-1]))
    center_lon = _parse_float(shock_cfg["center_lon"], "center_lon", float(LON_CENTRES[0]), float(LON_CENTRES[-1]))
    radius_cells = _parse_float(shock_cfg.get("radius_cells", 3.0), "radius_cells", 0.5, 12.0)

    falloff = shock_cfg.get("falloff", "smooth")
    if falloff not in ("flat", "smooth"):
        raise ValueError("falloff must be 'flat' or 'smooth'")

    effects = shock_cfg.get("effects")
    if not isinstance(effects, dict) or len(effects) == 0:
        raise ValueError("custom shock requires a non-empty effects mapping")

    std_phys = BUNDLE["std_phys"]
    parsed_effects: dict[int, float] = {}
    for ch_name, raw_val in effects.items():
        if ch_name not in CHANNELS:
            raise ValueError(f"unknown channel '{ch_name}' in effects")
        idx = CHANNELS.index(ch_name)
        max_mag = 4.0 * float(std_phys[idx])
        val = _parse_float(raw_val, f"effects.{ch_name}", -max_mag, max_mag)
        parsed_effects[idx] = val

    dy = LAT_CENTRES[:, None] - center_lat
    dx = LON_CENTRES[None, :] - center_lon
    dist = np.hypot(dy, dx)

    if falloff == "flat":
        weights = (dist <= radius_cells).astype(np.float32)
    else:
        inside = dist < radius_cells
        weights = np.where(
            inside,
            0.5 * (1.0 + np.cos(np.pi * dist / radius_cells)),
            0.0,
        ).astype(np.float32)

    if float(weights.sum()) == 0.0:
        flat_idx = int(np.argmin(dist))
        ii, jj = np.unravel_index(flat_idx, dist.shape)
        weights[ii, jj] = 1.0

    region_mask = weights > 0.0
    delta = np.zeros((6, 32, 36), dtype=np.float32)
    for idx, val in parsed_effects.items():
        delta[idx] = np.float32(val) * weights
    return delta, region_mask


def _build_perturbation(
    shock_cfg: Any,
    init_last: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray]:
    if shock_cfg is None:
        preset = "null"
        cfg: dict[str, Any] = {}
    elif isinstance(shock_cfg, dict):
        preset = str(shock_cfg.get("preset", "null"))
        cfg = shock_cfg
    else:
        raise ValueError("shock must be an object or null")

    if preset == "null":
        return preset, build_null(), np.zeros((32, 36), dtype=bool)
    if preset == "warm_bay_of_bengal":
        return preset, build_warm_ocean(init_last), box_mask(8.0, 20.0, 82.0, 94.0)
    if preset == "weaker_arabian_sea_wind":
        return preset, build_monsoon_stall(init_last), box_mask(8.0, 20.0, 65.0, 75.0)
    if preset == "custom":
        delta, mask = _build_custom_perturbation(cfg)
        return preset, delta, mask
    raise ValueError(f"unknown shock preset '{preset}'")


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return int(resp.status), json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"error": raw or f"upstream HTTP {exc.code}"}
        return int(exc.code), body


def _summarize_perturbation(pert: np.ndarray, region_mask: np.ndarray) -> dict[str, Any]:
    std_phys = BUNDLE["std_phys"]
    n_cells = int(region_mask.sum())
    per_channel: dict[str, dict[str, float]] = {}
    for c, ch in enumerate(CHANNELS):
        field = pert[c]
        max_abs = float(np.max(np.abs(field)))
        if n_cells > 0:
            region_vals = field[region_mask]
            mean_val = float(np.mean(region_vals))
            min_val = float(np.min(region_vals))
            max_val = float(np.max(region_vals))
        else:
            mean_val = min_val = max_val = 0.0
        per_channel[ch] = {
            "mean": round(mean_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "max_abs_std": round(max_abs / float(std_phys[c]), 4),
        }
    return {"n_cells": n_cells, "channels": per_channel}


def _compute_timeseries(control: np.ndarray, shocked: np.ndarray) -> dict[str, Any]:
    land = BUNDLE["land_mask"]
    all_ts: dict[str, dict[str, list[float]]] = {}
    land_ts: dict[str, dict[str, list[float]]] = {}
    for c, ch in enumerate(CHANNELS):
        p_all = control[:, c].mean(axis=(1, 2))
        m_all = shocked[:, c].mean(axis=(1, 2))
        p_land = control[:, c][:, land].mean(axis=1)
        m_land = shocked[:, c][:, land].mean(axis=1)
        all_ts[ch] = {
            "prime": np.round(p_all, 4).tolist(),
            "multiverse": np.round(m_all, 4).tolist(),
        }
        land_ts[ch] = {
            "prime": np.round(p_land, 4).tolist(),
            "multiverse": np.round(m_land, 4).tolist(),
        }
    return {"all": all_ts, "land": land_ts}


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    model_url = str(app.state.model_url).rstrip("/")
    try:
        status_code, body = _http_json(f"{model_url}/health", payload=None, timeout=4.0)
        return {
            "dashboard": "ok",
            "model_url": model_url,
            "model_reachable": status_code == 200,
            "model": body,
        }
    except Exception as exc:
        return {
            "dashboard": "ok",
            "model_url": model_url,
            "model_reachable": False,
            "model": {"status": "unreachable", "error": str(exc)},
        }


@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    std_phys = BUNDLE["std_phys"]
    channels_out = []
    for i, info in enumerate(CHANNEL_INFO):
        item = dict(info)
        item["std"] = round(float(std_phys[i]), 6)
        channels_out.append(item)
    return {
        "grid": BUNDLE["grid"],
        "channels": channels_out,
        "presets": PRESETS_META,
        "history_available": bool(NC_PATH.is_file()),
        "limits": {
            "min_steps": MIN_STEPS,
            "max_steps": MAX_STEPS,
            "min_doy": MIN_DOY,
            "max_doy": MAX_DOY,
            "min_radius_cells": 0.5,
            "max_radius_cells": 12.0,
        },
    }


@app.get("/api/init/climatology")
def api_init_climatology(doy: str | int = 166):
    try:
        doy_int = _parse_int(doy, "doy", MIN_DOY, MAX_DOY)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    arr, init_days = _build_climatology_init(doy_int)
    return {
        "source": "climatology",
        "label": "smooth climatology",
        "doy": doy_int,
        "init_days": init_days,
        "init_states": np.round(arr, 4).tolist(),
    }


@app.post("/api/init/upload")
async def api_init_upload(request: Request):
    body = await request.body()
    if not body:
        return JSONResponse(status_code=400, content={"error": "empty upload body"})
    if len(body) > MAX_UPLOAD_BYTES:
        return JSONResponse(status_code=413, content={"error": "uploaded file exceeds 16 MB limit"})

    content_type = (request.headers.get("content-type") or "").lower()
    doy_val: int | None = None

    try:
        looks_json = "json" in content_type or body.lstrip()[:1] in (b"{", b"[")
        if looks_json and not body.startswith(b"PK"):
            parsed = json.loads(body.decode("utf-8"))
            if isinstance(parsed, dict):
                if "init_states" in parsed:
                    raw_arr = parsed["init_states"]
                elif "data" in parsed:
                    raw_arr = parsed["data"]
                else:
                    raise ValueError("JSON upload must contain 'init_states' or be a 4D array")
                if parsed.get("doy") is not None:
                    doy_val = _parse_int(parsed["doy"], "doy", MIN_DOY, MAX_DOY)
            else:
                raw_arr = parsed
            arr = _validate_init_states(raw_arr, "init_states")
        else:
            with np.load(io.BytesIO(body), allow_pickle=False) as npz:
                if not hasattr(npz, "files") or len(npz.files) == 0:
                    raise ValueError("uploaded .npz archive is empty")
                key = None
                for candidate in ("init_states", "data", "arr_0"):
                    if candidate in npz.files:
                        key = candidate
                        break
                if key is None:
                    key = npz.files[0]
                arr = _validate_init_states(npz[key], f"npz['{key}']")
                if "doy" in npz.files:
                    raw_doy = np.asarray(npz["doy"]).item()
                    doy_val = _parse_int(raw_doy, "doy", MIN_DOY, MAX_DOY)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return JSONResponse(status_code=400, content={"error": "invalid .npz or .json init upload"})

    return {
        "source": "upload",
        "label": "uploaded init",
        "doy": doy_val,
        "init_states": np.round(arr, 4).tolist(),
    }


@app.get("/api/init/history")
def api_init_history(index: str | int = 2):
    if not NC_PATH.is_file():
        return JSONResponse(status_code=404, content={"error": "not available"})
    try:
        idx = _parse_int(index, "index", 2, 10_000_000)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    import xarray as xr

    try:
        with xr.open_dataset(NC_PATH) as ds:
            total = int(ds.sizes["time"])
            if idx >= total:
                return JSONResponse(status_code=400, content={"error": f"index must be between 2 and {total - 1}"})
            arr = ds["data"].isel(time=slice(idx - 2, idx + 1)).values.astype(np.float32)
            doy_raw = int(ds["time"].isel(time=idx).dt.dayofyear.item())
            doy_val = min(MAX_DOY, max(MIN_DOY, doy_raw))
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"failed to read history sample: {exc}"})

    return {
        "source": "history",
        "label": f"history sample #{idx}",
        "index": idx,
        "total": total,
        "doy": doy_val,
        "init_states": np.round(arr, 4).tolist(),
    }


@app.post("/api/run")
async def api_run(request: Request):
    wall_t0 = time.perf_counter()
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        doy = _parse_int(body.get("doy", 166), "doy", MIN_DOY, MAX_DOY)
        steps = _parse_int(body.get("steps", 10), "steps", MIN_STEPS, MAX_STEPS)

        init_raw = body.get("init_states")
        if init_raw is None and isinstance(body.get("init"), dict):
            init_raw = body["init"].get("init_states")
        if init_raw is None:
            init_arr, _ = _build_climatology_init(doy)
        else:
            init_arr = _validate_init_states(init_raw, "init_states")

        preset, pert, region_mask = _build_perturbation(body.get("shock"), init_arr[2])
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception:
        return JSONResponse(status_code=400, content={"error": "malformed JSON request"})

    model_url = str(app.state.model_url).rstrip("/")
    doy_sc = sincos_from_doy(doy, period=BUNDLE["harm_period"]).astype(np.float32)
    init_list = init_arr.tolist()
    doy_sc_list = [float(doy_sc[0]), float(doy_sc[1])]

    ctrl_payload = {
        "init_states": init_list,
        "doy_sincos": doy_sc_list,
        "steps": steps,
        "perturbation": None,
    }
    shock_payload = {
        "init_states": init_list,
        "doy_sincos": doy_sc_list,
        "steps": steps,
        "perturbation": pert.tolist(),
    }

    try:
        s1, ctrl_resp = _http_json(f"{model_url}/simulate", payload=ctrl_payload, timeout=30.0)
        if s1 != 200 or "rollout" not in ctrl_resp:
            err = ctrl_resp.get("error", f"model server returned HTTP {s1}")
            return JSONResponse(status_code=502 if s1 >= 500 else s1, content={"error": str(err)})

        s2, shock_resp = _http_json(f"{model_url}/simulate", payload=shock_payload, timeout=30.0)
        if s2 != 200 or "rollout" not in shock_resp:
            err = shock_resp.get("error", f"model server returned HTTP {s2}")
            return JSONResponse(status_code=502 if s2 >= 500 else s2, content={"error": str(err)})
    except Exception as exc:
        return JSONResponse(status_code=503, content={"error": f"model server unavailable: {exc}"})

    control = np.asarray(ctrl_resp["rollout"], dtype=np.float32)
    shocked = np.asarray(shock_resp["rollout"], dtype=np.float32)
    expected_shape = (steps, 6, 32, 36)
    if control.shape != expected_shape or shocked.shape != expected_shape:
        return JSONResponse(status_code=502, content={"error": "unexpected rollout shape from model server"})

    diff = shocked - control
    max_abs_diff = float(np.max(np.abs(diff)))

    clim_phys = rollout_climatology_phys(
        BUNDLE["harm_coef"],
        doy,
        steps,
        harm_k=BUNDLE["harm_k"],
        period=BUNDLE["harm_period"],
    )
    impact = compute_impacts_pair(
        control,
        shocked,
        clim_phys,
        BUNDLE["land_mask"],
        BUNDLE["lat"],
    )

    inf_ctrl = float(ctrl_resp.get("inference_ms", 0.0))
    inf_shock = float(shock_resp.get("inference_ms", 0.0))
    wall_ms = (time.perf_counter() - wall_t0) * 1000.0

    return {
        "doy": doy,
        "steps": steps,
        "preset": preset,
        "channels": list(CHANNELS),
        "region_mask": region_mask.tolist(),
        "perturbation_summary": _summarize_perturbation(pert, region_mask),
        "control": np.round(control, 4).tolist(),
        "shocked": np.round(shocked, 4).tolist(),
        "diff": np.round(diff, 4).tolist(),
        "max_abs_diff": max_abs_diff,
        "timeseries": _compute_timeseries(control, shocked),
        "impact": impact,
        "inference_ms": {
            "control": round(inf_ctrl, 2),
            "shocked": round(inf_shock, 2),
            "total": round(inf_ctrl + inf_shock, 2),
        },
        "wall_ms": round(wall_ms, 2),
    }


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model", type=str, default="http://127.0.0.1:8000")
    args = ap.parse_args()
    app.state.model_url = str(args.model).rstrip("/")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
