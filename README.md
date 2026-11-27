# Multiverse of Monsoon

Local-physics surrogate weather simulator and counterfactual trajectory explorer for the Indian monsoon (`6.5°–37.5°N`, `65.5°–100.5°E`, `32 × 36` grid at `1°` daily resolution).

Given `3` days of physical history across `6` channels (`t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv`), the system rolls out `1` to `60` daily steps for an unperturbed **Prime** forecast alongside a localized **Multiverse** perturbation, serving both via an offline HTTP API and a 3D-relief cartographic dashboard.

## Quick Start

Requires **Python 3.12**. Weights, harmonic climatology, normalization statistics, and static fields are bundled in `weights/` and `config/` (no external downloads at runtime).

### 1. Install runtime dependencies and start the model server

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe serve.py --port 8000
```

Wait until `GET http://127.0.0.1:8000/health` returns `{"status": "ready", "model_name": "monsoon-twin-v1"}`. Use `127.0.0.1` instead of `localhost` on Windows to avoid IPv6 DNS fallback delay.

### 2. Start the dashboard (optional interactive explorer)

In a second terminal, with `serve.py` running on port `8000`:

```powershell
.\.venv\Scripts\python.exe dashboard\server.py --port 8080 --model http://127.0.0.1:8000
```

Open `http://127.0.0.1:8080` (`1280×720` or wider). The UI provides:
- `360×320` hillshaded **3D Relief** maps with marching-squares coastlines and a `Flat Grid` (`36×32`) toggle
- Start-state selection via smooth `K=4` climatology (`DOY 1–365`), drag-and-drop `.npz`/`.json` upload, or historical index (when `data/` is present)
- Preset shocks (`warm_bay_of_bengal`, `weaker_arabian_sea_wind`, `null`) and interactive drag-to-draw custom circular/rectangular shocks
- Synchronized crosshairs, domain-mean trajectory plots (all cells vs `736` land cells), PNG/CSV/JSON export, and live `monsoon/impact.py` proxy metrics

See [dashboard/README.md](dashboard/README.md) for the full `/api/*` specification.

### 3. Validate

With `serve.py` listening on port `8000`:

```powershell
.\.venv\Scripts\python.exe -m monsoon_r1.schema_validator http://127.0.0.1:8000
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"
```

## Results Summary (`C = 128`, Held-Out Evaluation)

Evaluated across `50` unseen `test` windows (`docs/phase3_results.md`, post-processing fitted strictly on `val`) and `96` de-duplicated held-out shock windows (`scripts/check_shocks.py`):

| Lead | Mean Skill vs Best Baseline | Heavy-Rain CSI (`>= 10 mm/d`) | Best Baseline CSI |
|---:|---:|---:|---:|
| Day 1 | `+0.3050` | `0.5166` | `0.3939` (Persistence) |
| Day 4 | `+0.1248` | `0.3924` | `0.2848` (Empirical Clim.) |
| Day 8 | `+0.0154` | `0.3494` | `0.2995` (Empirical Clim.) |

- **Anti-blur (Day-8 Spatial Variance Ratio, `test`):** `tp = 0.4846` (floor `>= 0.40`), non-rain channels `0.6223–1.0447`.
- **30-step stability (`300` rollouts, incl. `200` `1σ` noisy starts):** `0` NaN/Inf, `min tp = 0.0 mm/d`, `t2m ∈ [-32.76, +43.33] °C`, `±5σ` anomaly clamp hit rate `< 0.004%`.
- **Shock causality (`96` windows, `V1` unshrunk response + causal cone):** `warm_ocean` passes all checks `100.0%` (`96/96`), `null` is exact `0.0` difference, `monsoon_stall` passes `localized` / `no_teleport` / `stable` at `100.0%` and `propagates` at `45.8%` (`44/96`).
- **Offline latency (`4` CPU threads):** `0.12–0.18 s` per 10-step `/simulate` request (`~303 MB` peak RSS).

## Architecture & Method

1. **Anomaly space:** Inputs are converted relative to a train-fitted `4`-harmonic seasonal climatology (`period = 365`, `log1p`/`expm1` for `tp`) and pixel-wise standard deviations (`sigma_pix`).
2. **Strictly local core (`MonsoonNet`, `175,123` params):** `cin = 34` channels (3-day anomalies, day-to-day tendency, target climatology, static `land_sea_mask`/orography, `sin`/`cos` DOY) pass through a `3×3` stem, two depthwise-separable residual blocks (`C = 128`, replicate padding, per-pixel `LN2d`), and zero-initialized `1×1` heads (`a_next = sigmoid(rho) * a_t + scale * head(h)` plus a heavy-rain logit). Zero global pooling fixes the per-step receptive radius at `3` grid cells.
3. **Out-of-loop calibration:** Validation-fitted lead `beta` shrinkage, multiplicative `kappa` rain debiasing, day-8 `pattern_gain` (`g_8 = 1.4`), and event-head blending are applied outside the autoregressive loop (`monsoon/postproc.py`).
4. **Counterfactual `V1` shocks:** Prime retains `beta` shrinkage (`control_served`). Shocked rollouts compute `pert_nb - ctrl_nb` without `beta` attenuation, apply a Chebyshev causal cone (`cone_speed = 3.0` cells/day, `taper = 1.5`) and `±4σ` cap, and add the bounded response to `control_served`. Null shocks (`<= 1e-9`) return `control_served` identically.

## Repository Layout

```text
serve.py                  Offline FastAPI server (/health, /simulate)
train.py                  Stage A (1-step) and Stage B (BPTT rollout) trainer
config/default.json       Model, training, post-processing, and shock config
weights/                  Bundled model.pt, harmonics, norm stats, postproc.json, static grid
monsoon/
  model.py                MonsoonNet local CNN + damped persistence (175,123 params)
  climatology.py          4-harmonic DOY climatology fit/eval (period=365)
  transforms.py           Physical <-> log1p rain transforms (serving-safe)
  rollout.py              Autoregressive rollout loop (torch + numpy bridge)
  postproc.py             Out-of-loop beta, kappa, pattern_gain, and event coupling
  shock.py                Causal cone mask and 4-sigma cap
  shocks_spec.py          Warm-ocean, monsoon-stall, and null perturbation specs
  infer.py                Simulator runtime (Prime + V1 counterfactual path)
  impact.py               Land heatwave, heavy-rain, deficit, and crop-stress proxies
  data.py, losses.py, metrics.py, windows.py   Dataset, losses, metrics, held-out splits
dashboard/
  server.py               FastAPI proxy, raw upload, climatology, and history API
  static/                 Vanilla HTML/CSS/JS 3D-relief UI and SVG time-series charts
scripts/                  Data prep, baselines, tuning, evaluation, shock checks, figures, brief
tests/                    Unit, contract, schema, receptive-field, causality, and UI suites
brief/                    2-page brief source (brief.html), PDF (brief.pdf), and PNG figures
docs/                     Evaluation tables, decision log, training CSV logs
```

## Training & Reproducibility

Training and offline evaluation require `data/india_climate_train.nc` (`6209 × 6 × 32 × 36`, split `0.82` train / `0.09` val / `0.09` test, gitignored) and the packages in `requirements-train.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe scripts\prepare_data.py
.\.venv\Scripts\python.exe train.py --stage a
.\.venv\Scripts\python.exe train.py --stage b
.\.venv\Scripts\python.exe scripts\tune_postproc.py
```

Observed CPU training wall times (`8` CPU threads, `docs/train_log_stageA.csv`, `docs/train_log_stageB.csv`):
- **Stage A (`25` one-step epochs):** `1668.4 s` (`27.8 min`, `64.9–68.7 s/epoch`)
- **Stage B (`11` BPTT epochs: `K=2` × 3, `K=4` × 4, `K=8` × 4):** `1957.5 s` (`32.6 min`: `70.0–71.8 s/epoch` at `K=2`, `143.0–144.6 s/epoch` at `K=4`, `289.2–299.2 s/epoch` at `K=8`)

Reproduce evaluation reports, shock diagnostics, figures, and the 2-page PDF brief:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py
.\.venv\Scripts\python.exe scripts\check_shocks.py
.\.venv\Scripts\python.exe scripts\make_figures.py
.\.venv\Scripts\python.exe scripts\build_brief.py
```

## Limitations

- **Forecast horizon:** Mean skill approaches climatology by day 8 (`+0.0154` on `test`, `-0.0026` at day 10); the model is calibrated for short-to-medium (`1–7` day) dynamics.
- **Grid resolution:** Daily `1°` (~111 km) averaging smooths convective rainfall extremes and cannot resolve tropical cyclone eyewalls or sub-daily peaks (day-8 `tp` variance ratio `0.4846`).
- **Counterfactuals as surrogate behavior:** Shock trajectories reflect learned local network dynamics bounded by the `4σ` cap and causal cone, not calibrated physical re-runs. In weak Arabian Sea wind regimes, `-40%` `monsoon_stall` perturbations decay under `~0.83/day` persistence before outside-box energy crosses the `propagates` threshold (`45.8%` overall pass rate across `96` windows).
- **Dataset shift & impact proxies:** Future out-of-archive periods may score lower than the held-out test split. Metrics in `monsoon/impact.py` are experimental `1°` daily grid proxies without local crop calendars, soil moisture, or exposure weighting and are not operational advisories.

## Dependencies

Declared in `requirements.txt` and `requirements-train.txt`: `torch`, `numpy`, `scipy`, `pandas`, `xarray`, `scikit-learn`, `fastapi`, `uvicorn`, `matplotlib`, `h5netcdf`, `netCDF4`, `pytest`, `httpx`. Schema/repo validation uses `monsoon-r1-kit`; HTML-to-PDF rendering uses headless Microsoft Edge (`msedge.exe`).
