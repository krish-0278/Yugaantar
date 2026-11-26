# Multiverse of Monsoon

Local-physics surrogate weather simulator and counterfactual trajectory explorer for the Indian monsoon domain (`6.5°–37.5°N`, `65.5°–100.5°E`, `32 × 36` grid at `1°` resolution, `6` daily channels: `t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv`).

## Overview

Multiverse of Monsoon rolls out `1` to `60` daily steps from a `3`-day physical history using a strictly local neural atmospheric core paired with harmonic climatology and out-of-loop post-processing. Beyond a deterministic **Prime** forecast, callers can inject localized physical perturbations (Sea-surface / column-moisture warming, Arabian Sea wind stall, custom localized shocks, or an exact-zero null check) to inspect **Multiverse** trajectories, spatial spread, and downstream physical proxy impacts side by side.

## Requirements

- **Python**: `3.12` (`.venv` recommended on Windows)
- **Serving & runtime**: `torch`, `numpy`, `scipy`, `pandas`, `xarray`, `scikit-learn`, `fastapi`, `uvicorn` (`requirements.txt`)
- **Training, figures & testing** (optional): `matplotlib`, `h5netcdf`, `netCDF4`, `pytest`, `httpx` (`requirements-train.txt`)
- **Bundled artifacts**: Trained weights, normalization statistics, harmonic climatology, and post-processing config are bundled in `weights/` and `config/`. Training and offline dataset evaluation require `data/india_climate_train.nc` (gitignored, not bundled).

## Run the server

```powershell
pip install -r requirements.txt
python serve.py --port 8000
```

Wait until `GET http://127.0.0.1:8000/health` returns `{"status": "ready", "model_name": "monsoon-twin-v1"}`. Use `127.0.0.1` instead of `localhost` on Windows to avoid IPv6 DNS fallback latency.

## Validate

With the inference server running on port `8000`:

```powershell
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
```

Run the test suite and repository hygiene scanner:

```powershell
python -m pytest -v
python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=1))"
```

## Train

Requires `data/india_climate_train.nc` (`6209 × 6 × 32 × 36`, fractional contiguous split `0.82` train / `0.09` val / `0.09` test) and training extras:

```powershell
pip install -r requirements-train.txt
python scripts/prepare_data.py
python train.py --stage a
python train.py --stage b
python scripts/tune_postproc.py
```

Observed CPU training wall times (`docs/train_log_stageA.csv`, `docs/train_log_stageB.csv`, `8` CPU threads):
- **Stage A (`25` one-step epochs)**: `1377.2 s` total (`23.0 min`, `~53–65 s/epoch`)
- **Stage B (`11` multi-step BPTT epochs: `K=2` × 3, `K=4` × 4, `K=8` × 4)**: `1485.7 s` total (`24.8 min`: `~53 s/epoch` at `K=2`, `~109 s/epoch` at `K=4`, `~223 s/epoch` at `K=8`)

## Dashboard

Start both servers in separate terminals (offline, `http://127.0.0.1:8080`):

**Terminal 1 (model inference server, port `8000`):**

```powershell
.\.venv\Scripts\python.exe serve.py --port 8000
```

**Terminal 2 (dashboard backend & static UI, port `8080`):**

```powershell
.\.venv\Scripts\python.exe dashboard\server.py --port 8080 --model http://127.0.0.1:8000
```

Open `http://127.0.0.1:8080` in a desktop browser (`1280×720` or wider). See `dashboard/README.md` for API endpoints (`GET /api/health`, `GET /api/meta`, `GET /api/init/climatology`, `POST /api/init/upload`, `GET /api/init/history`, `POST /api/run`).

## Repository map

```text
serve.py                  FastAPI /health and /simulate inference server
train.py                  Stage A (1-step) and Stage B (BPTT rollout) trainer
config/default.json       Model, training, post-processing, and shock parameters
weights/                  Bundled checkpoint, harmonics, normalization, static grid
monsoon/
  model.py                MonsoonNet local residual CNN + damped persistence (106,771 params)
  climatology.py          4-harmonic DOY climatology fit and evaluation (period=365)
  transforms.py           Physical <-> log1p rain model-space transforms (serving-safe)
  rollout.py              Autoregressive rollout wrapper (torch + numpy bridge)
  postproc.py             Out-of-loop shrinkage, kappa debias, pattern gain, event coupling
  shock.py                Causal cone propagation mask and 4-sigma response cap
  shocks_spec.py          Reference warm-ocean, monsoon-stall, and null shock specs
  infer.py                Simulator runtime (Prime control + V1 unshrunk shock path)
  impact.py               Heatwave, heavy-rain, rain-deficit, and crop-stress proxies
  data.py, losses.py, metrics.py, windows.py   Dataset, losses, metrics, held-out windows
dashboard/
  server.py               FastAPI proxy & raw-body upload / climatology / history API
  static/                 Vanilla HTML/CSS/JS UI + canvas maps + inline SVG time series
scripts/                  Data prep, baselines, postproc tuning, evaluation, shock checks, figures
tests/                    Unit, contract, schema, shock causality, and dashboard suites
brief/                    2-page HTML/PDF technical brief and generated PNG figures
docs/                     Phase results, decision log, training CSV logs
```

## Method summary

- **Formulation & architecture**: Predicts standardized anomalies relative to a `4`-harmonic seasonal climatology (`period=365`). `MonsoonNet` (`106,771` parameters, `cin=34`, `96` hidden channels, `2` depthwise residual blocks) pairs learned per-channel damped persistence (`sigmoid(rho_logit) ≈ 0.83–0.85/day`) with strictly local `3×3` convolutions and per-pixel `LN2d` (`max_step_radius = 3`), outputting a `6`-channel anomaly innovation plus a `10 mm/day` heavy-rain event probability.
- **Training & drift/blur control**: Trained in Stage A (`Huber + 0.2 BCE + 0.05 GradDiff`, EMA) and Stage B multi-step BPTT (`K=2, 4, 8`) with `±5σ` rollout anomaly clamps. Out-of-loop post-processing applies lead-dependent `beta` shrinkage (fitted on `val`), `kappa` multiplicative `tp` debiasing, day-8 pattern gain (`g_8 = 1.4`) to keep rain spatial variance above the `0.40` floor (`0.5095` on `test`), and event-conditioned precipitation blending.
- **Counterfactual shock serving (`V1`)**: The Prime control forecast retains output `beta` shrinkage (`control_served`), whereas shocked trajectories compute the dynamical response `pert_nb - ctrl_nb` without output `beta` attenuation, apply a spatial causal cone (`cone_speed = 3.0` cells/day, `taper = 1.5`) and `4σ` cap, and add the bounded response to `control_served`. Null perturbations (`<= 1e-9`) return `control_served` identically.

## Reproducing reported numbers

With `data/india_climate_train.nc` available locally (outputs written to `docs/phase3_results.md`, `brief/figures/*.png`, and `brief/brief.pdf`):

```powershell
python scripts/evaluate.py
python scripts/check_shocks.py
python scripts/make_figures.py
python scripts/build_brief.py
```

## Limitations

- Forecast skill decays toward climatology by lead day `8` (`+0.0121` test mean skill vs empirical climatology; `+0.2919` at day `1`, `+0.1175` at day `4`).
- Precipitation day-8 spatial variance ratio is `0.5095` (`test`), passing the `0.40` floor while extreme tails remain smoothed at `1°` daily resolution (tropical cyclone tracks and sub-daily convection are not resolved).
- Counterfactual shock fields represent learned surrogate dynamics rather than calibrated physical ground truth: large `tp` responses saturate at the `4σ` cap on day `1` (`35%` of `warm_ocean` windows), and `monsoon_stall` passes the kit's `propagates` criterion on `40.6%` (`39/96`) of held-out windows because weak background winds decay under `~0.83/day` persistence.
- Post-training evaluation periods outside the historical window distribution may score lower than the held-out local test split.
- Dashboard impact indicators (`heatwave_cell_days`, `heavy_rain_*_share`, `rain_deficit_share`, `crop_stress_csi`) are experimental `1°` daily proxies without local crop or population exposure maps; deficit and crop-stress indices over short `5–7` day horizons are sensitive to synoptic noise and are not operational forecasts.

## Libraries

Core libraries declared in `requirements.txt` and `requirements-train.txt`: `torch`, `numpy`, `scipy`, `pandas`, `xarray`, `scikit-learn`, `fastapi`, `uvicorn`, `matplotlib`, `h5netcdf`, `netCDF4`, `pytest`, `httpx`. Validation utilities use `monsoon-r1-kit`; PDF conversion uses headless Microsoft Edge (`msedge.exe`).
