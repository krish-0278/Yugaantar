"""Unit, contract, static hygiene, and live integration tests for the dashboard."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastapi.testclient import TestClient
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.server import BUNDLE, CHANNELS, _build_perturbation, app
from monsoon.climatology import sincos_from_doy
from monsoon.shocks_spec import (
    LAT_CENTRES,
    LON_CENTRES,
    build_monsoon_stall,
    build_null,
    build_warm_ocean,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_preset_shocks_match_contract():
    rng = np.random.default_rng(7)
    init_last = np.abs(rng.normal(0.0, 2.0, size=(6, 32, 36)).astype(np.float32)) + 1.0
    init_last[0] = 298.0
    init_last[5] = 40.0

    _, p_null, mask_null = _build_perturbation({"preset": "null"}, init_last)
    np.testing.assert_array_equal(p_null, build_null())
    assert not mask_null.any()

    _, p_warm, _ = _build_perturbation({"preset": "warm_bay_of_bengal"}, init_last)
    np.testing.assert_allclose(p_warm, build_warm_ocean(init_last), atol=1e-6)

    _, p_stall, _ = _build_perturbation({"preset": "weaker_arabian_sea_wind"}, init_last)
    np.testing.assert_allclose(p_stall, build_monsoon_stall(init_last), atol=1e-6)


def test_custom_shock_falloff_flat_vs_smooth():
    init_last = np.zeros((6, 32, 36), dtype=np.float32)
    _, p_flat, m_flat = _build_perturbation(
        {
            "preset": "custom",
            "center_lat": 16.5,
            "center_lon": 85.5,
            "radius_cells": 3.0,
            "falloff": "flat",
            "effects": {"t2m": 3.0},
        },
        init_last,
    )
    _, p_smooth, m_smooth = _build_perturbation(
        {
            "preset": "custom",
            "center_lat": 16.5,
            "center_lon": 85.5,
            "radius_cells": 3.0,
            "falloff": "smooth",
            "effects": {"t2m": 3.0},
        },
        init_last,
    )
    i0 = int(np.where(LAT_CENTRES == 16.5)[0][0])
    j0 = int(np.where(LON_CENTRES == 85.5)[0][0])
    assert p_flat[0, i0, j0] == pytest.approx(3.0)
    assert p_smooth[0, i0, j0] == pytest.approx(3.0)
    assert m_flat.sum() >= m_smooth.sum() > 0
    # Strict compact support outside radius 3.0
    dy = LAT_CENTRES[:, None] - 16.5
    dx = LON_CENTRES[None, :] - 85.5
    outside = np.hypot(dy, dx) > 3.0
    assert np.all(p_flat[0, outside] == 0.0)
    assert np.all(p_smooth[0, outside] == 0.0)


def test_climatology_init_and_doy_wrap():
    client = TestClient(app)
    r = client.get("/api/init/climatology?doy=1")
    assert r.status_code == 200
    body = r.json()
    assert body["init_days"] == [364, 365, 1]
    arr = np.asarray(body["init_states"], dtype=np.float32)
    assert arr.shape == (3, 6, 32, 36)
    assert np.isfinite(arr).all()
    assert np.all(arr[:, 1] >= 0.0)

    for bad in (0, 366, -1, "abc"):
        r_bad = client.get(f"/api/init/climatology?doy={bad}")
        assert r_bad.status_code == 400
        assert "error" in r_bad.json()


def test_upload_raw_body_npz_and_json():
    client = TestClient(app)
    arr = np.full((3, 6, 32, 36), 2.5, dtype=np.float32)
    arr[:, 0] = 296.0

    buf = io.BytesIO()
    np.savez(buf, init_states=arr, doy=np.int32(180))
    r_npz = client.post(
        "/api/init/upload",
        content=buf.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert r_npz.status_code == 200
    assert r_npz.json()["doy"] == 180
    assert np.asarray(r_npz.json()["init_states"]).shape == (3, 6, 32, 36)

    r_json = client.post(
        "/api/init/upload",
        content=json.dumps({"init_states": arr.tolist(), "doy": 170}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    assert r_json.status_code == 200
    assert r_json.json()["doy"] == 170

    # Wrong shape rejected cleanly
    bad_buf = io.BytesIO()
    np.savez(bad_buf, init_states=np.zeros((2, 6, 32, 36), dtype=np.float32))
    assert client.post("/api/init/upload", content=bad_buf.getvalue()).status_code == 400
    assert client.post("/api/init/upload", content=b"").status_code == 400
    assert client.post("/api/init/upload", content=b"not valid").status_code == 400


def test_custom_shock_and_run_validation_400():
    client = TestClient(app)
    bad_payloads = [
        {"doy": 0, "steps": 5, "shock": {"preset": "null"}},
        {"doy": 166, "steps": 31, "shock": {"preset": "null"}},
        {"doy": 166, "steps": 5, "shock": {"preset": "bad_preset"}},
        {
            "doy": 166,
            "steps": 5,
            "shock": {
                "preset": "custom",
                "center_lat": 50.0,
                "center_lon": 85.0,
                "radius_cells": 3.0,
                "falloff": "smooth",
                "effects": {"t2m": 2.0},
            },
        },
        {
            "doy": 166,
            "steps": 5,
            "shock": {
                "preset": "custom",
                "center_lat": 16.5,
                "center_lon": 85.5,
                "radius_cells": 0.1,
                "falloff": "smooth",
                "effects": {"t2m": 2.0},
            },
        },
        {
            "doy": 166,
            "steps": 5,
            "shock": {
                "preset": "custom",
                "center_lat": 16.5,
                "center_lon": 85.5,
                "radius_cells": 3.0,
                "falloff": "cubic",
                "effects": {"t2m": 2.0},
            },
        },
        {
            "doy": 166,
            "steps": 5,
            "shock": {
                "preset": "custom",
                "center_lat": 16.5,
                "center_lon": 85.5,
                "radius_cells": 3.0,
                "falloff": "smooth",
                "effects": {"bad_channel": 2.0},
            },
        },
        {
            "doy": 166,
            "steps": 5,
            "shock": {
                "preset": "custom",
                "center_lat": 16.5,
                "center_lon": 85.5,
                "radius_cells": 3.0,
                "falloff": "smooth",
                "effects": {"t2m": 999.0},
            },
        },
    ]
    for payload in bad_payloads:
        resp = client.post("/api/run", json=payload)
        assert resp.status_code == 400
        assert "error" in resp.json()


def test_null_shock_calls_simulate_twice_with_zeros():
    recorded_perts: list = []

    class _MockHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready"}).encode("utf-8"))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            recorded_perts.append(body.get("perturbation"))
            steps = int(body["steps"])
            init_last = np.asarray(body["init_states"], dtype=np.float32)[-1]
            rollout = np.repeat(init_last[None], steps, axis=0)
            if body.get("perturbation") is not None:
                rollout = rollout + np.asarray(body["perturbation"], dtype=np.float32)[None]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"rollout": rollout.tolist(), "inference_ms": 1.2}).encode("utf-8"))

    port = _free_port()
    srv = HTTPServer(("127.0.0.1", port), _MockHandler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    prev_url = app.state.model_url
    try:
        app.state.model_url = f"http://127.0.0.1:{port}"
        client = TestClient(app)
        resp = client.post("/api/run", json={"doy": 166, "steps": 3, "shock": {"preset": "null"}})
        assert resp.status_code == 200
        data = resp.json()
        assert len(recorded_perts) == 2
        assert recorded_perts[0] is None
        pert_arr = np.asarray(recorded_perts[1], dtype=np.float32)
        assert pert_arr.shape == (6, 32, 36)
        assert np.all(pert_arr == 0.0)
        assert data["max_abs_diff"] == 0.0
        assert np.array_equal(np.asarray(data["control"]), np.asarray(data["shocked"]))
        assert data["impact"]["steps"] == 3
    finally:
        app.state.model_url = prev_url
        srv.shutdown()
        srv.server_close()


def test_dashboard_hygiene_and_no_heavy_imports():
    dash_files = [
        ROOT / "dashboard" / "server.py",
        ROOT / "dashboard" / "static" / "index.html",
        ROOT / "dashboard" / "static" / "style.css",
        ROOT / "dashboard" / "static" / "app.js",
    ]
    date_re = re.compile(r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}\b")
    wx_re = re.compile(r"\b(?:ecmwf|era5|noaa|imd|gfs|cdsapi|open-meteo)\b", re.IGNORECASE)
    url_re = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)

    for path in dash_files:
        text = path.read_text(encoding="utf-8")
        assert not date_re.search(text), f"Date string found in {path}"
        assert not wx_re.search(text), f"Weather service keyword found in {path}"
        for match in url_re.findall(text):
            assert match.startswith(("http://127.0.0.1", "http://localhost")), (
                f"External URL '{match}' found in {path}"
            )

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, dashboard.server; "
                "bad=[m for m in ('xarray','pandas','monsoon.data','monsoon.losses','train') if m in sys.modules]; "
                "assert not bad, bad"
            ),
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0


def test_live_serve_py_integration():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "serve.py"), "--port", str(port)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    prev_url = app.state.model_url
    try:
        model_url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 90.0
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate()
                pytest.fail(f"serve.py exited early ({proc.returncode}):\n{out}\n{err}")
            try:
                with urllib.request.urlopen(f"{model_url}/health", timeout=2.0) as resp:
                    if resp.status == 200 and json.loads(resp.read().decode("utf-8")).get("status") == "ready":
                        ready = True
                        break
            except Exception:
                time.sleep(1.0)
        assert ready, "serve.py did not become ready in time"

        app.state.model_url = model_url
        client = TestClient(app)

        r_null = client.post("/api/run", json={"doy": 166, "steps": 5, "shock": {"preset": "null"}})
        assert r_null.status_code == 200
        d_null = r_null.json()
        assert d_null["max_abs_diff"] == 0.0
        assert np.asarray(d_null["control"]).shape == (5, 6, 32, 36)

        r_custom = client.post(
            "/api/run",
            json={
                "doy": 166,
                "steps": 5,
                "shock": {
                    "preset": "custom",
                    "center_lat": 14.5,
                    "center_lon": 88.5,
                    "radius_cells": 3.0,
                    "falloff": "smooth",
                    "effects": {"t2m": 3.0, "tcwv": 5.0},
                },
            },
        )
        assert r_custom.status_code == 200
        d_custom = r_custom.json()
        assert d_custom["max_abs_diff"] > 0.0
        diff = np.asarray(d_custom["diff"], dtype=np.float32)
        assert np.isfinite(diff).all()

        # Verify day-1 energy localization around the custom circle
        dy = LAT_CENTRES[:, None] - 14.5
        dx = LON_CENTRES[None, :] - 88.5
        near_mask = np.hypot(dy, dx) <= 5.0  # within radius + 2 cells
        std = BUNDLE["std_phys"]
        norm_sq = (diff[0] / std[:, None, None]) ** 2
        total_e = float(norm_sq.sum())
        near_e = float(norm_sq[:, near_mask].sum())
        assert total_e > 0.0
        assert (near_e / total_e) >= 0.90
    finally:
        app.state.model_url = prev_url
        proc.terminate()
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
