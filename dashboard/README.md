# Multiverse of Monsoon Dashboard

Counterfactual trajectory explorer comparing an unperturbed **Prime** forecast against a **Multiverse** shock rollout side-by-side over the Indian monsoon domain (`6.5°–37.5°N, 65.5°–100.5°E`, `32 × 36` at `1°`).

## Running Locally (Two Terminals)

1. **Terminal 1 — Start the inference server (`port 8000`)**:

```powershell
.\.venv\Scripts\python.exe serve.py --port 8000
```

Wait until `POST http://127.0.0.1:8000/health` reports `{"status": "ready", "model_name": "monsoon-twin-v1"}`.

2. **Terminal 2 — Start the dashboard server (`port 8080`)**:

```powershell
.\.venv\Scripts\python.exe dashboard\server.py --port 8080 --model http://127.0.0.1:8000
```

3. Open `http://127.0.0.1:8080` in a desktop browser (`1280×720` or wider).

## API Endpoints

- `GET /api/health` — Checks dashboard availability and proxies `{model}/health`.
- `GET /api/meta` — Returns grid coordinates, `land_sea_mask`, channel units/standard deviations, shock preset metadata, and `history_available`.
- `GET /api/init/climatology?doy=N` — Synthesizes 3 consecutive initial days ending at day-of-year `N` (`1–365`, wrapping cleanly across day `365 -> 1`) from `weights/clim_harmonics.npz` via `from_model_space`.
- `POST /api/init/upload` — Accepts raw `.npz` (`application/octet-stream`) or `.json` (`application/json`) bodies containing a finite `[3, 6, 32, 36]` initial state array.
- `GET /api/init/history?index=N` — Extracts `[3, 6, 32, 36]` and integer `doy` from `data/india_climate_train.nc` when present locally (returns `404 {"error": "not available"}` otherwise).
- `POST /api/run` — Sends two live `POST /simulate` calls (`perturbation=null` for Prime; `[6, 32, 36]` perturbation for Multiverse, including all-zeros for the `null` preset) and returns round-tripped rollouts, differences, domain-mean time series, and impact proxies.

## Shock Modes & Proxy Impacts

- **Presets**: `null` (exact all-zeros live check), `warm_bay_of_bengal` (`+3.0 K t2m`, `+15% tcwv` over `8–20°N, 82–94°E`), `weaker_arabian_sea_wind` (`-40% u10`, `-40% v10` over `8–20°N, 65–75°E`), and `custom` (circular `flat` or raised-cosine `smooth` perturbation centered anywhere on the grid).
- **Physical Proxy Impacts** (`monsoon/impact.py`): heatwave land cell-days (`t2m` anomaly `>= +3.0 K` for `>= 2` consecutive days), heavy rainfall shares (`tp >= 64.5` and `>= 115.6 mm/day`), land rainfall deficit (`> 20%` below seasonal climatology), and latitude-weighted crop stress (`CSI`).

## Limitations

- Harmonic climatology represents a smooth 4-harmonic seasonal cycle in `expm1` precipitation space.
- Rainfall deficit and crop stress (`CSI`) over short horizons (`<= 7` days) reflect synoptic variability and are noisy; longer rollouts (`10–30` days) provide steadier comparisons.
- All outputs are experimental `1°` daily surrogate diagnostics and not an operational forecast.
