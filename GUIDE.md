# Multiverse of Monsoon — Complete Beginner's Guide

> **Who this is for:** Someone who has never seen this codebase before. No prior knowledge of machine learning, weather forecasting, or Python is assumed. Every concept is introduced before it is used.

---

## Table of Contents

1. [What is this project?](#1-what-is-this-project)
2. [The big picture — how it works in plain English](#2-the-big-picture--how-it-works-in-plain-english)
3. [Key vocabulary](#3-key-vocabulary)
4. [The data: what the model sees](#4-the-data-what-the-model-sees)
5. [The neural network: MonsoonNet](#5-the-neural-network-monsoonnet)
6. [Training the model](#6-training-the-model)
7. [Post-processing: fixing the model's quirks](#7-post-processing-fixing-the-models-quirks)
8. [Counterfactual shocks: asking "what if?"](#8-counterfactual-shocks-asking-what-if)
9. [Impact proxies: translating forecasts into human terms](#9-impact-proxies-translating-forecasts-into-human-terms)
10. [The API server: serve.py](#10-the-api-server-servepy)
11. [The dashboard: the visual interface](#11-the-dashboard-the-visual-interface)
12. [How to run everything from scratch](#12-how-to-run-everything-from-scratch)
13. [How to train the model yourself](#13-how-to-train-the-model-yourself)
14. [File-by-file tour of the repository](#14-file-by-file-tour-of-the-repository)
15. [Performance numbers explained](#15-performance-numbers-explained)
16. [Known limitations — what this cannot do](#16-known-limitations--what-this-cannot-do)
17. [Frequently asked questions](#17-frequently-asked-questions)

---

## 1. What is this project?

**Multiverse of Monsoon** is a software system that:

1. **Forecasts weather** for the Indian subcontinent up to 10 days into the future, using a small trained neural network (not a physics simulation).
2. **Lets you run "what if?" experiments** — for example: *"What would the rainfall look like over the next 10 days if the Bay of Bengal were 3°C warmer right now?"* The system runs two parallel forecasts — one normal ("Prime") and one with the change applied ("Multiverse") — and shows you the difference.
3. **Estimates rough impact** — how many land grid-cells had a heatwave, whether there was a rain deficit, how stressed crops might be.

It is a **research surrogate model**, not an operational forecast service. Think of it as a fast, lightweight "toy physics engine" for the Indian monsoon that you can probe and experiment with.

---

## 2. The big picture — how it works in plain English

Here is the entire pipeline in simple steps:

```
Historical weather data (3 days)
        │
        ▼
  Subtract the "normal" for that time of year  ←─── Climatology (seasonal average)
        │
        ▼
  Feed the anomaly (departure from normal) into MonsoonNet
        │
        ▼
  MonsoonNet predicts the anomaly for tomorrow
        │
        ▼
  Add "normal" back → physical weather values for tomorrow
        │
        ▼
  Use tomorrow's output as input → repeat up to 60 days (autoregressive rollout)
        │
        ▼
  Post-process (fix known biases, boost spatial detail)
        │
        ▼
  Optional: apply a "shock" (Bay of Bengal warming etc.) and run a second forecast
        │
        ▼
  Show both forecasts side-by-side in the dashboard
```

The key insight is that predicting *anomalies* (departures from normal) is much easier than predicting absolute values, because seasonal patterns dominate the signal otherwise.

---

## 3. Key vocabulary

| Term | Simple explanation |
|---|---|
| **Anomaly** | The difference from the seasonal average. If today is 2°C hotter than usual for this time of year, the anomaly is +2. |
| **Climatology** | The expected "normal" value for each calendar day, computed by averaging many years of data. |
| **Harmonic climatology** | A smooth mathematical approximation of the seasonal cycle using sine and cosine waves (4 harmonics). Avoids memorising each day individually. |
| **DOY** | Day Of Year. DOY 1 = Jan 1, DOY 166 = mid-June (deep monsoon), DOY 365 = Dec 31. |
| **Autoregressive rollout** | Feeding the model's output back as input to predict the next day, repeatedly. Like asking someone to predict step 2 after predicting step 1. |
| **Prime** | The unmodified forecast — no shock applied. |
| **Multiverse** | The forecast with a shock (perturbation) applied. |
| **Shock / Perturbation** | A deliberate change applied to the initial conditions — e.g., warming the sea surface by 3 K. |
| **Counterfactual** | "What would have happened if X were different?" The Multiverse forecast is a counterfactual. |
| **CSI** | Critical Success Index. A score for how well the model predicts rain events. 1.0 is perfect, 0.0 is useless. |
| **Skill** | The improvement over the best naive baseline (persistence or climatology). Positive = better than the baseline. |
| **RMSE** | Root Mean Squared Error. Smaller = more accurate predictions. |
| **Beta shrinkage** | Reducing forecast amplitude slightly at longer leads to avoid overconfident predictions. |
| **Causal cone** | A physical constraint: a shock applied at one location can only spread as fast as atmospheric disturbances travel. Cells far away on day 1 are not affected; they can be affected by day 5 or 6. |

---

## 4. The data: what the model sees

### Geography

The model covers the **Indian subcontinent and surrounding oceans**:
- **Latitude:** 6.5°N to 37.5°N
- **Longitude:** 65.5°E to 100.5°E
- **Resolution:** 1° × 1° grid → **32 rows × 36 columns = 1,152 grid cells**

### Channels (weather variables)

Each grid cell has **6 variables** per day:

| Index | Name | Meaning | Unit |
|---|---|---|---|
| 0 | `t2m` | Air temperature at 2 m height | Kelvin (K) |
| 1 | `tp` | Total precipitation | mm/day |
| 2 | `u10` | East-West wind at 10 m (+ = blowing east) | m/s |
| 3 | `v10` | North-South wind at 10 m (+ = blowing north) | m/s |
| 4 | `msl` | Mean sea-level pressure | Pa |
| 5 | `tcwv` | Total column water vapour | kg/m² |

So the model's input is a **3-day history** shaped `[3, 6, 32, 36]` — 3 days × 6 channels × 32 latitude rows × 36 longitude columns.

### Training / Validation / Test split

The historical dataset `data/india_climate_train.nc` has 6,209 daily snapshots. These are divided:
- **Train (82%):** Used to fit the model weights.
- **Val (9%):** Used to tune the post-processing (never touched during training).
- **Test (9%):** Held out until the very end — 50 evaluation windows, completely unseen.

---

## 5. The neural network: MonsoonNet

**File:** `monsoon/model.py`

MonsoonNet is a small convolutional neural network with **106,771 parameters** (very small for a neural network — GPT-3 has 175 billion).

### What "local" means

The key design constraint: the network can only look at **nearby grid cells**. With a 3×3 convolution and 2 blocks, the maximum reach is 3 cells (≈ 3 degrees, ≈ 330 km). This is deliberate:
- It prevents the model from learning spurious long-range correlations from limited data.
- It makes shock responses physically causal (a perturbation in the Bay of Bengal cannot instantly affect Rajasthan on day 1).

### Architecture from bottom to top

```
Input features [B, 34, 32, 36]
  (34 = 3 days × 6 channels + 2 sin/cos DOY + 10 static grid features)
        │
   Stem Conv 3×3   →  96 channels
        │
   Block 1:
     DepthwiseConv 3×3 → LN2d → expand 1×1 → GELU → project 1×1 → residual add
        │
   Block 2: (same as Block 1)
        │
   Two output heads:
     Head A: 1×1 conv → 6 anomaly channels  (the 6-variable anomaly prediction)
     Head B: 1×1 conv → 1 heavy-rain probability logit
```

**LN2d** (LayerNorm 2D): normalises across channels *per pixel*, not spatially. This keeps each grid cell independent, reinforcing the local-only design.

### Damped persistence

The model does not just output what the network predicts raw. It blends it with the previous day's anomaly:

```
a_next = rho × a_today + scale × network_innovation
```

Where `rho ≈ 0.83–0.85` (a learned parameter). This is called **damped persistence** — the model starts from "yesterday was similar" and adds a correction. It is very effective for multi-day forecasts because atmospheric memory is real.

---

## 6. Training the model

**Files:** `train.py`, `monsoon/losses.py`, `monsoon/data.py`

Training happens in two stages.

### Stage A — One-step training (25 epochs)

The model is trained to predict **one day ahead** at a time. For each training window:
1. Take 3 days of real history → subtract climatology → get anomalies.
2. Feed into MonsoonNet → get predicted anomaly for day 4.
3. Compare to actual day 4.
4. Compute loss, update weights.

**Loss function:**
```
Loss = Huber(predicted, actual)        ← smooth version of mean absolute error
     + 0.2 × BCE(rain_prob, rain_event) ← binary cross-entropy for heavy rain
     + 0.05 × GradDiff                  ← penalise spatial blurring
```

**EMA:** Exponential Moving Average of weights — keeps a smoothed version of the weights that tends to generalise better.

**Wall time:** ~23 minutes on 8 CPU threads.

### Stage B — Multi-step training (11 epochs)

The model is unrolled **K steps** (K=2, then K=4, then K=8) and trained end-to-end. This is called **Back-Propagation Through Time (BPTT)**. It teaches the model to avoid drift over multiple days.

**Wall time:** ~25 minutes on 8 CPU threads.

**Total training time: ~48 minutes on a modern laptop CPU.** No GPU required.

---

## 7. Post-processing: fixing the model's quirks

**File:** `monsoon/postproc.py`

Even a good neural network has systematic biases. Post-processing fixes them *after* the model runs (without retraining).

| Step | What it fixes | How |
|---|---|---|
| **Beta shrinkage** | At longer leads, the model is overconfident — it predicts larger anomalies than reality. | Multiply anomaly by a lead-day-dependent factor `< 1`. Fitted on val. |
| **Kappa debiasing** | Precipitation is systematically too high or low. | Multiply rain by a fitted correction factor. |
| **Pattern gain (g₈ = 1.4)** | At day 8, the model's rain maps are too spatially smooth (blurry). | Boost the rain anomaly by 1.4× at day 8 to restore spatial variance above a 0.40 floor. |
| **Event coupling** | Heavy-rain events are blended back in using the model's event probability head. | Weighted blend of the model's rain prediction and the event signal. |
| **Physical bounds** | Temperature must stay between −40°C and +60°C. Rain must be ≥ 0. | Hard clips. |

---

## 8. Counterfactual shocks: asking "what if?"

**Files:** `monsoon/shock.py`, `monsoon/shocks_spec.py`, `dashboard/server.py`

### Built-in shock presets

| Shock name | What it does | Region |
|---|---|---|
| **Null shock** | Does nothing. The Multiverse equals the Prime. Used to verify the system is working correctly — differences must be exactly zero. | Everywhere |
| **warm_bay_of_bengal** | Adds +3 K to `t2m` and +15% to `tcwv` (column moisture) | Bay of Bengal: lat 8–20°N, lon 82–94°E |
| **weaker_arabian_sea_wind** | Reduces `u10` and `v10` by 40% (weakens the monsoon jet) | Arabian Sea: lat 8–20°N, lon 65–75°E |

### Custom shocks

You can also draw your own shock in the dashboard:
- Pick a center latitude/longitude.
- Pick a radius (in grid cells, max 15).
- Choose a channel (e.g., `t2m`) and a delta value (e.g., +3 K).
- Choose a falloff (sharp box or smooth Gaussian).

### How shocks propagate

**The causal cone** prevents physically impossible responses. Atmospheric disturbances travel at roughly 3 grid cells per day. So:
- On **day 1**, only cells within 3 cells of the shock box are affected.
- On **day 5**, cells up to 15 cells away can be affected.
- Far-away cells on day 1 always show zero difference.

**The 4σ cap** prevents unrealistically large responses. If the model predicts a response larger than 4 standard deviations of the channel's physical variability, it is clipped. This happens for 35% of warm-ocean windows on day 1 (the Bay of Bengal warming strongly forces the model at short leads).

### V1 shock serving

The Multiverse forecast is computed as:
```
Multiverse = Prime + shock_response
shock_response = (pert_without_beta - ctrl_without_beta)  ← raw dynamical signal
              × causal_cone_mask                           ← spatially gated
              clipped to ±4σ                              ← bounded
```

The "without beta" means the shrinkage correction is *not* applied to the response — this preserves the full dynamical signal of the shock.

---

## 9. Impact proxies: translating forecasts into human terms

**File:** `monsoon/impact.py`

The dashboard shows an "impact table" comparing Prime vs Multiverse across 4 proxy indicators. These are **rough proxies only** — not operational forecasts.

| Indicator | Definition | Threshold |
|---|---|---|
| **Heatwave cell-days** | Number of land grid-cell × day pairs where temperature anomaly ≥ +3 K for ≥ 2 consecutive days | +3 K, ≥ 2 days |
| **Heavy rain share (64 mm)** | Fraction of land cell-days with rainfall ≥ 64.5 mm/day | 64.5 mm/day |
| **Heavy rain share (115 mm)** | Fraction of land cell-days with rainfall ≥ 115.6 mm/day (very heavy) | 115.6 mm/day |
| **Rain deficit share** | Fraction of land cells where the total forecast rain is >20% below what climatology expects | >20% below clim |
| **Crop stress CSI** | Composite index = 0.6 × water deficit + 0.4 × heat stress, area-weighted over land | 0 to 1 |

For example, a "warm Bay of Bengal" shock typically increases the heatwave cell-days slightly (the ocean warming propagates onshore) and may change the rain deficit share depending on how the monsoon responds.

---

## 10. The API server: serve.py

**File:** `serve.py`

This is a FastAPI web server that exposes two HTTP endpoints:

### `GET /health`

Returns a JSON object confirming the model is loaded and ready:

```json
{"status": "ready", "model_name": "monsoon-twin-v1"}
```

### `POST /simulate`

Accepts:
```json
{
  "init_states": [ ... ],    // 3-day history array, shape [3, 6, 32, 36]
  "doy_sincos": [0.97, 0.26], // sin and cos of the day-of-year angle
  "steps": 10,               // how many days to forecast (1–60)
  "shock": {"kind": "null"}  // or a named shock, or a custom shock spec
}
```

Returns:
```json
{
  "rollout": [ ... ],        // shape [steps, 6, 32, 36] — the Prime forecast
  "inference_ms": 127        // wall time in milliseconds
}
```

The dashboard calls this endpoint internally. You can also call it from the command line or any HTTP client.

---

## 11. The dashboard: the visual interface

**Files:** `dashboard/server.py`, `dashboard/static/`

The dashboard is a web application running at `http://127.0.0.1:8080`. It has two panels:

### Left panel — Controls

1. **Start state** — Choose how to initialise the 3-day history:
   - **Climatology:** Use the seasonal average for the chosen DOY. Good for exploring idealized responses.
   - **Historical record:** Load a real 3-day window from the archived history (if the data file is present).
   - **Upload:** Upload your own `.npy` file shaped `[3, 6, 32, 36]`.

2. **Shock** — Choose a preset or draw a custom shock. Preset cards show a mini-map of the shock box. The "Null shock" preset runs both Prime and Multiverse identically, confirming they match.

3. **Run** — Sends the request to the model server and shows results.

### Right panel — Results

Three map views side by side:
- **Prime** — the unperturbed forecast
- **Multiverse** — the shocked forecast
- **Difference** (Multiverse − Prime) — where the shock had an effect

You can:
- Scrub through **days 1–10** with the day slider (keyboard: left/right arrows)
- **Play** the animation (spacebar)
- Switch between **channels** (tabs: t2m, tp, u10, v10, msl, tcwv)
- **Hover** any cell to see a crosshair that highlights the same cell on all three maps and shows exact values
- View the **time-series chart** of domain-mean values across all days
- Download the current view as **PNG**, the data as **CSV**, or the impact as **JSON**
- See the **run history** (last 5 runs in memory) and reload a previous run

The impact table at the bottom shows the 5 proxy indicators for both Prime and Multiverse, with differences highlighted.

---

## 12. How to run everything from scratch

### Requirements

- **Python 3.12** (other 3.x versions may work but are untested)
- A terminal / command prompt
- ~500 MB of disk space for the model weights and dashboard files
- No GPU needed — everything runs on CPU

### Step 1: Clone the repository

```powershell
git clone https://github.com/krish-0278/Yugaantar.git
cd Yugaantar
```

### Step 2: Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> **What is a virtual environment?** It is an isolated Python installation just for this project. It prevents conflicts with other Python packages you may have installed.

### Step 3: Start the model server

Open **Terminal 1**:

```powershell
.venv\Scripts\python.exe serve.py --port 8000
```

Wait until you see output indicating the server is running. Then verify it is healthy:

```powershell
# In a new terminal:
curl http://127.0.0.1:8000/health
# Expected: {"status":"ready","model_name":"monsoon-twin-v1"}
```

### Step 4: Start the dashboard server

Open **Terminal 2**:

```powershell
.venv\Scripts\python.exe dashboard\server.py --port 8080 --model http://127.0.0.1:8000
```

### Step 5: Open the dashboard

Open your browser and go to: **http://127.0.0.1:8080**

You should see the dashboard. Try the **"Try Warm Bay of Bengal"** button in the empty state to run your first experiment.

### Step 6: Run the test suite (optional but recommended)

```powershell
.venv\Scripts\python.exe -m pytest -v
```

Expected: **29 passed, 1 warning**.

### Step 7: Run the repo health scanner (optional)

```powershell
.venv\Scripts\python.exe -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=2))"
```

Expected: **15.0/15.0 points, 0 flags**.

---

## 13. How to train the model yourself

> **Note:** You need the raw training dataset `data/india_climate_train.nc` (not included in the repository — it is large and gitignored). Training also requires extra packages.

### Step 1: Install training extras

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

### Step 2: Prepare the data

```powershell
.venv\Scripts\python.exe scripts\prepare_data.py
```

This computes and saves the climatology harmonics, normalisation statistics, and static grid fields into `weights/`.

### Step 3: Stage A training (1-step, ~23 minutes)

```powershell
.venv\Scripts\python.exe train.py --stage a
```

Trains for 25 epochs on one-step predictions. Logs written to `docs/train_log_stageA.csv`.

### Step 4: Stage B training (multi-step BPTT, ~25 minutes)

```powershell
.venv\Scripts\python.exe train.py --stage b
```

Trains for 11 epochs with rollout lengths K=2, K=4, K=8. Logs written to `docs/train_log_stageB.csv`.

### Step 5: Tune post-processing

```powershell
.venv\Scripts\python.exe scripts\tune_postproc.py
```

Fits beta, kappa, pattern gain, and event coupling on the validation set. Writes results to `weights/postproc.json`.

### Step 6: Evaluate

```powershell
.venv\Scripts\python.exe scripts\evaluate.py
```

Computes RMSE, skill, CSI, and variance ratios on the held-out test set. Writes to `docs/phase3_results.md`.

### Step 7: Generate figures

```powershell
.venv\Scripts\python.exe scripts\make_figures.py
```

Produces:
- `brief/figures/skill_vs_lead.png` — skill vs lead day for model vs baselines
- `brief/figures/shock_4panel.png` — warm-ocean response maps at days 1/3/6/10
- `brief/figures/variance_ratio.png` — day-8 spatial variance ratio per channel

---

## 14. File-by-file tour of the repository

```
Yugaantar/
│
├── serve.py                  ← The model API server. Run this first.
├── train.py                  ← Training script (needs data/).
│
├── config/
│   └── default.json          ← All hyperparameters: model size, shock speeds,
│                               post-processing knobs, training schedule.
│
├── weights/                  ← Pre-trained model files. Never delete these.
│   ├── model.pt              ← PyTorch checkpoint (the trained neural network)
│   ├── clim_harmonics.npz    ← Harmonic climatology coefficients [9, 6, 32, 36]
│   ├── norm.npz              ← Mean and std for normalisation
│   ├── static.npz            ← Land mask, latitude weights, orography
│   ├── postproc.json         ← Beta, kappa, pattern-gain, event-coupling params
│   └── channel_std_phys.json ← Physical std per channel for the 4σ shock cap
│
├── monsoon/                  ← Core Python library
│   ├── model.py              ← MonsoonNet architecture (the neural network)
│   ├── climatology.py        ← Harmonic and empirical climatology utilities
│   ├── transforms.py         ← Physical ↔ model-space conversions (log1p rain etc.)
│   ├── rollout.py            ← Autoregressive rollout loop
│   ├── postproc.py           ← Beta shrinkage, kappa, pattern gain, event coupling
│   ├── shock.py              ← Causal cone mask and 4σ response cap
│   ├── shocks_spec.py        ← Reference shock specifications (warm ocean, stall, null)
│   ├── infer.py              ← High-level Simulator class (Prime + Multiverse)
│   ├── impact.py             ← Heatwave, rain, deficit, crop-stress proxy indices
│   ├── data.py               ← Dataset loading and window sampling
│   ├── losses.py             ← Huber + BCE + gradient-difference loss
│   ├── metrics.py            ← RMSE, skill, CSI, variance ratio
│   └── windows.py            ← Held-out evaluation window definitions
│
├── dashboard/
│   ├── server.py             ← Dashboard backend (FastAPI proxy + history API)
│   ├── static/
│   │   ├── index.html        ← The web page (single-page app)
│   │   ├── style.css         ← All visual styling
│   │   └── app.js            ← All interactivity: maps, charts, drag-draw, history
│   └── README.md             ← Dashboard API documentation
│
├── scripts/
│   ├── prepare_data.py       ← Compute and save climatology, norms, static fields
│   ├── evaluate.py           ← Full test-set evaluation (RMSE, skill, CSI, VR)
│   ├── check_shocks.py       ← Verify shock causality on held-out windows
│   ├── tune_postproc.py      ← Fit post-processing params on val
│   ├── make_figures.py       ← Generate brief/figures/*.png
│   └── build_brief.py        ← Build brief/brief.pdf from brief/brief.html
│
├── tests/
│   ├── test_model.py         ← NaN/stability tests for MonsoonNet
│   ├── test_rollout.py       ← Rollout determinism and zero-shock identity
│   ├── test_serve.py         ← API endpoint contract tests
│   ├── test_postproc.py      ← Post-processing correctness
│   ├── test_shocks_serving.py ← Shock causality on 96 windows
│   ├── test_shocks_spec.py   ← Reference shock specification tests
│   ├── test_receptive_field.py ← Confirm max reach is 3 cells
│   └── test_dashboard.py     ← Dashboard API and hygiene tests
│
├── brief/
│   ├── brief.html            ← 2-page technical brief (HTML source)
│   ├── brief.pdf             ← Compiled PDF of the brief
│   └── figures/
│       ├── dashboard.png     ← Screenshot of the dashboard after a run
│       ├── skill_vs_lead.png ← Model vs baseline skill plot
│       ├── shock_4panel.png  ← Warm-ocean shock response maps
│       └── variance_ratio.png ← Day-8 spatial variance ratio
│
├── docs/
│   ├── phase3_results.md     ← Full evaluation tables (RMSE, skill, CSI, VR)
│   ├── baselines.md          ← Baseline method descriptions
│   ├── DECISIONS.md          ← Design decision log
│   ├── train_log_stageA.csv  ← Stage A per-epoch training log
│   └── train_log_stageB.csv  ← Stage B per-epoch training log
│
├── README.md                 ← Quick-start reference (commands and repo map)
├── GUIDE.md                  ← This file
├── requirements.txt          ← Runtime Python packages
├── requirements-train.txt    ← Additional packages for training and figures
└── .gitignore                ← Excludes data/, weights not in repo, temp files
```

---

## 15. Performance numbers explained

All numbers are on the **held-out test set** (50 windows, never seen during training or tuning).

### Skill score

Skill is measured relative to the best of three naive baselines (persistence, harmonic climatology, empirical climatology):

```
Skill = 1 - RMSE(model) / RMSE(best_baseline)
```

A skill of `+0.29` means the model's error is 29% lower than the best baseline.

| Lead day | Skill (mean over all channels) | Interpretation |
|---:|---:|---|
| Day 1 | +0.2919 | Strong — model is clearly better than "yesterday repeated" |
| Day 4 | +0.1175 | Moderate — still meaningfully ahead of baselines |
| Day 8 | +0.0121 | Weak — nearly climatology-level by day 8 |

### Heavy-rain CSI at 10 mm/day threshold

| Lead day | Persistence | Climatology | Model |
|---:|---:|---:|---:|
| Day 1 | 0.394 | 0.224 | **0.515** |
| Day 4 | 0.209 | 0.285 | **0.388** |
| Day 8 | 0.171 | 0.300 | **0.343** |

The model consistently beats both baselines at detecting rain events ≥ 10 mm/day.

### Day-8 spatial variance ratio

This measures whether the model's spatial patterns have the right amplitude. A ratio of 1.0 is perfect; below 0.4 means the maps are too smooth (blurry).

| Channel | Model at day 8 | Pass band |
|---|---:|---|
| t2m | 1.05 | ✅ 0.40–2.50 |
| tp | **0.51** | ✅ ≥ 0.40 (target ≥ 0.40) |
| u10 | 0.64 | ✅ |
| v10 | 0.66 | ✅ |
| msl | 0.87 | ✅ |
| tcwv | 0.84 | ✅ |

Precipitation (`tp`) is the hardest channel — it barely passes. The pattern-gain step in post-processing is essential for this.

---

## 16. Known limitations — what this cannot do

1. **Skill collapses by day 8.** The model is competitive at 1–5 days. By day 8 it is barely ahead of climatology (`+0.012` skill). Do not use it for 10-day absolute forecasts.

2. **Precipitation is smooth.** At 1° daily resolution, intense convective storms, cyclone eyes, and sub-daily bursts are not captured. The model predicts area-averaged daily rain, not spot intensities.

3. **Shock responses are model behaviour, not physics.** The Multiverse forecast shows what this particular neural network does when you change the initial conditions. It is not a calibrated physical simulation. A +3 K warming in the Bay of Bengal produces a plausible-looking response, but the exact magnitude and location cannot be validated against real-world data.

4. **Large rain responses saturate.** For warm-ocean shocks, 35% of test windows hit the 4σ cap on day 1. The cap prevents absurd values but means the actual shock response may be larger than what is shown.

5. **Monsoon stall only partially propagates.** The monsoon-stall shock (weakening Arabian Sea winds) passes the causality criterion in only 40.6% of held-out windows. In the other 60%, the natural persistence of the initial state dominates over the perturbation.

6. **Impact proxies are experimental.** Heatwave cell-days, rain deficit, and crop stress CSI are simple thresholding operations on 1° daily values. They do not account for population density, crop types, soil moisture, irrigation, or sub-daily weather. They are indicators for exploration, not operational alerts.

7. **No future data.** The model was trained on historical data. Evaluation outside the training distribution (e.g., future climate scenarios) may score lower than the held-out test results.

---

## 17. Frequently asked questions

**Q: What does "surrogate model" mean?**
A: A surrogate model is a fast approximation of a more complex system. Real numerical weather models solve millions of physical equations and take hours on supercomputers. This model was trained to *mimic* the patterns in historical weather data using a neural network that runs in milliseconds on a laptop.

**Q: Why use anomalies instead of raw temperatures?**
A: Most of the signal in daily weather is the seasonal cycle — it is cold in January and hot in June. If you try to predict raw temperature, the model wastes most of its capacity learning this obvious pattern. By subtracting the climatology first, the model only has to learn the interesting departures from normal.

**Q: What is a 1° resolution grid?**
A: One degree of latitude or longitude is about 111 km at the equator (a bit less at higher latitudes). So each grid cell is roughly a 100 km × 100 km square. A city like Mumbai occupies less than one cell.

**Q: Why is there a 3-day history input?**
A: Weather has memory. Yesterday's temperature partly determines today's. The 3-day window gives the model enough context to infer the current atmospheric state and momentum (e.g., is a trough approaching? Is a heat wave building?).

**Q: Can I use this for real weather forecasting?**
A: No. This is a research tool for exploring counterfactual scenarios. It does not ingest real-time observations, it has no data assimilation, and its skill at longer leads is too low for operational use. For real forecasts, use services that run full numerical weather prediction systems.

**Q: Why are the weights already bundled in the repo?**
A: So you can run the server immediately without needing the training data (which is large and cannot be distributed). The weights are the trained neural network — they are the "brain" of the system. Training would reproduce them, but it is not required to use the system.

**Q: What is a "fast-forward merge" in git?**
A: When you merge a branch that is simply ahead of the current branch (no divergence), git can just move the pointer forward. No new merge commit is created. This is what happened when `ui-upgrade` was merged into `main` — all 5 commits from the UI upgrade were added cleanly.

**Q: How do I add a new shock type?**
A: Add a function `build_myshock(init_last: np.ndarray) -> np.ndarray` to `monsoon/shocks_spec.py` that returns a `[6, 32, 36]` array with your perturbation. Then wire it up in `dashboard/server.py` in the shock-dispatch section. The causal cone and cap are applied automatically.

**Q: What does "29 passed, 1 warning" mean in the test suite?**
A: There are 29 automated tests covering the model, API, shocks, dashboard, and repo hygiene. All pass. The 1 warning is a deprecation notice from the `httpx`/`starlette` test client — it does not affect correctness.

---

*This guide was written from the actual source code, evaluation results, and test suite. Every number comes from a real run.*
