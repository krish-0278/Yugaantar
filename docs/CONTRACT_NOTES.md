# Contract Notes (Round 1 Authoritative Specification)

`docs/R1_CONTRACT.md` is authoritative and overrides `docs/BUILD_GUIDE.md` wherever they differ.

## Authoritative Contract Overrides

1. **Grid:** 32x36, 1 degree cells. Lat centres 6.5..37.5 (south to north).
   Lon centres 65.5..100.5 (west to east). Lat index `i = lat - 6.5`, lon index `j = lon - 65.5`.
2. **DOY convention:** `doy_sincos = [sin(2*pi*doy/365), cos(2*pi*doy/365)]` for the LAST input day.
   Forecast step `k` is `doy + k`. Use period `365`, NOT `365.25`. `rollout[0] = day +1`.
3. **Judge shocks (built from `init_states[-1]`):**
   - `warm_ocean`: `t2m += 3.0 K` and `tcwv += 0.15 * current tcwv`, box `lat 8..20`, `lon 82..94`.
   - `monsoon_stall`: `u10 -= 0.40 * u10` and `v10 -= 0.40 * v10`, box `lat 8..20`, `lon 65..75`.
   - `null`: `perturbation = zeros`. Output must equal the unperturbed call (`max diff < 1e-3` of channel std).
4. **Scoring facts:**
   - Skill: lat-weighted RMSE vs the better of persistence and climatology, leads 1, 4, 8, all 6 channels, 5 windows.
   - Heavy-rain CSI at 10 mm/day, same leads.
   - Anti-blur: day-8 spatial variance between 40% and 250% of the real one, per channel.
   - Causality: response non-zero; at day 1 at least 60% of response energy within 3 cells of the box;
     propagates outward by day 4-8 at most 8 cells/day; never above 5 std.
   - Validity: no NaN/Inf, `rain >= 0`, `t2m` within `-40..+60 C`.
5. **Wire format & hosting:** nested JSON lists only (canonical; base64 dict tolerated by starter/validator).
   `/health` is `POST` and returns `{"status":"ready","model_name":"..."}`.
   Response under 80 MB. 10 steps under 5.0 s. Judge runs the repo offline.
   Static fields live in the `.nc` variable `static` (shape `2x32x36`: `land_sea_mask`, `orography_m`) and must be bundled.
6. **Guide changes:** delete technique T14 (physics coupling prior). Model per-step receptive radius must be `<= 3` cells
   (stem + 2 blocks). Causal cone: radius 3 cells at step 1, growing at most 8 cells/day (default 1.5 per step).
   Rain CSI threshold is 10 mm/day. Do not write the forbidden 4-character reanalysis name (`e-r-a-5`) or any weather-service name
   anywhere in the repo.

## Phase 0 Data & Validator Verification Findings

- **Dataset integrity (`data/india_climate_train.nc`):**
  - Dimensions: `time=6209`, `channel=6` (`t2m, tp, u10, v10, msl, tcwv`), `lat=32`, `lon=36`, `static_channel=2` (`land_sea_mask, orography_m`).
  - Zero NaN and zero Inf values across all dynamic channels and both static channels.
  - Time coordinate has 6209 contiguous daily steps (step diff = 1 day for all 6208 intervals, zero gaps).
  - Grid coordinates match item 1 exactly (`lat = 6.5..37.5`, `lon = 65.5..100.5`, 1.0 deg step).
- **Static fields (`static`):**
  - `land_sea_mask` is fractional in `[0.0, 1.0]` (914 unique values; 736/1152 cells > 0.5; 228/1152 cells == 0.0), not binary.
  - `orography_m` ranges from `-0.1563` to `5283.9370` m.
- **`stats.json` comparison:**
  - Keys: `['channels', 'mean', 'std']`.
  - Matches computed dataset mean/std closely; max absolute difference is `< 0.25` Pa on `msl` (`0.2438` Pa on mean, `5.959e-04` on std, attributable to float32 vs float64 reduction).
- **Leap-year day 366 handling:**
  - 4 occurrences of day 366 in training data. With period `365`, `2*pi*366/365` wraps smoothly modulo `2*pi` to the same `(sin, cos)` phase as day 1 (`max diff = 7.806e-16`).
- **Shock box boundary convention (ambiguity resolved):**
  - Box bounds `lat 8..20, lon 82..94` and `lat 8..20, lon 65..75` are implemented with inclusive bounds on cell centres (`(lat >= lo) & (lat <= hi)` and `(lon >= lo) & (lon <= hi)`).
  - `warm_ocean`: lat indices `2..13` (centres `8.5..19.5`), lon indices `17..28` (centres `82.5..93.5`), 144 cells.
  - `monsoon_stall`: lat indices `2..13` (centres `8.5..19.5`), lon indices `0..9` (centres `65.5..74.5`), 120 cells.
  - Verified identical to `monsoon_r1.contract.box_mask`.
- **Kit validator & starter server (`monsoon_r1`):**
  - `schema_validator` tests `POST /health` (polling up to 120 s) and `POST /simulate` across both canonical nested JSON lists and tolerated `{"b64": ..., "shape": ...}` payloads, for `no perturbation`, `warm_ocean`, and `zeros` (`steps=10`, `doy=160`).
  - `monsoon_r1.contract.MAX_STEPS = 14`, `SCORE_STEPS = 10`, `TIME_LIMIT_S = 5.0`.
  - Guide Sections 7.1/7.3 used `365.25` and Section 8 used `64.5 mm/day` — both are overridden by `R1_CONTRACT.md` (`365` and `10.0 mm/day`).
