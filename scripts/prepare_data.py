"""Dataset inspection, contract verification, doy check, and static grid/weights export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
NC_PATH = DATA_DIR / "india_climate_train.nc"
STATS_PATH = DATA_DIR / "stats.json"
GRID_JSON_PATH = ROOT / "dashboard" / "static" / "assets" / "grid.json"
STATIC_NPZ_PATH = ROOT / "weights" / "static.npz"
CHANNEL_STD_JSON_PATH = ROOT / "weights" / "channel_std_phys.json"
CONFIG_PATH = ROOT / "config" / "default.json"


def main() -> None:
    if not NC_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {NC_PATH}")
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Missing stats file: {STATS_PATH}")

    print("=== 1. Dataset Overview ===")
    ds = xr.open_dataset(NC_PATH)
    print(f"File: {NC_PATH}")
    print(f"Dims: {dict(ds.sizes)}")

    print("\nCoordinates:")
    for name, coord in ds.coords.items():
        print(f"  - {name}: dims={coord.dims}, shape={coord.shape}, dtype={coord.dtype}")
        if coord.ndim == 1 and coord.size <= 10 and not np.issubdtype(coord.dtype, np.datetime64):
            print(f"      values={coord.values.tolist()}")

    print("\nData variables:")
    for name, var in ds.data_vars.items():
        print(f"  - {name}: dims={var.dims}, shape={var.shape}, dtype={var.dtype}")

    print("\n=== 2. Grid Verification (Contract Item 1) ===")
    lat = ds["lat"].values.astype(np.float64)
    lon = ds["lon"].values.astype(np.float64)
    expected_lat = np.arange(6.5, 38.0, 1.0, dtype=np.float64)
    expected_lon = np.arange(65.5, 101.0, 1.0, dtype=np.float64)
    lat_match = bool(lat.shape == (32,) and np.array_equal(lat, expected_lat))
    lon_match = bool(lon.shape == (36,) and np.array_equal(lon, expected_lon))
    print(f"lat ({lat.shape}): {lat.tolist()}")
    print(f"lon ({lon.shape}): {lon.tolist()}")
    print(f"lat matches 6.5..37.5 step 1.0 (i = lat - 6.5): {lat_match}")
    print(f"lon matches 65.5..100.5 step 1.0 (j = lon - 65.5): {lon_match}")
    if not (lat_match and lon_match):
        raise ValueError("Grid coordinates do not match contract item 1.")

    print("\n=== 3. Time Coordinate ===")
    times = pd.DatetimeIndex(ds["time"].values)
    diffs_days = np.diff(times.values).astype("timedelta64[D]").astype(np.int64)
    unique_diffs, diff_counts = np.unique(diffs_days, return_counts=True)
    print(f"Count: {len(times)}")
    print(f"Range: {times[0].isoformat()} .. {times[-1].isoformat()}")
    print(f"Step diffs (days): {dict(zip(unique_diffs.tolist(), diff_counts.tolist()))}")

    print("\n=== 4. Dynamic Channel Statistics ===")
    data_arr = ds["data"].values.astype(np.float64)  # [T, C, H, W]
    channels = [str(c) for c in ds["channel"].values]
    print(f"{'ch':<6} {'nan_cnt':>8} {'inf_cnt':>8} {'min':>12} {'max':>12} {'mean':>12} {'std':>12}")
    for idx, ch in enumerate(channels):
        vals = data_arr[:, idx, :, :]
        nan_cnt = int(np.isnan(vals).sum())
        inf_cnt = int(np.isinf(vals).sum())
        cmin = float(np.nanmin(vals))
        cmax = float(np.nanmax(vals))
        cmean = float(np.nanmean(vals))
        cstd = float(np.nanstd(vals))
        print(f"{ch:<6} {nan_cnt:>8d} {inf_cnt:>8d} {cmin:>12.4f} {cmax:>12.4f} {cmean:>12.4f} {cstd:>12.4f}")

    print("\n=== 5. Static Variable Summary ===")
    static_f32 = ds["static"].values.astype(np.float32)  # [2, 32, 36]
    static_arr = static_f32.astype(np.float64)
    static_names = [str(s) for s in ds["static_channel"].values]
    print(f"{'static':<16} {'nan_cnt':>8} {'min':>12} {'max':>12} {'mean':>12} {'std':>12}")
    for idx, sname in enumerate(static_names):
        vals = static_arr[idx]
        nan_cnt = int(np.isnan(vals).sum())
        smin = float(np.nanmin(vals))
        smax = float(np.nanmax(vals))
        smean = float(np.nanmean(vals))
        sstd = float(np.nanstd(vals))
        print(f"{sname:<16} {nan_cnt:>8d} {smin:>12.4f} {smax:>12.4f} {smean:>12.4f} {sstd:>12.4f}")
    if "land_sea_mask" in static_names:
        lsm = static_arr[static_names.index("land_sea_mask")]
        print(
            f"land_sea_mask unique count={np.unique(lsm).size}, "
            f"cells>0.5={int((lsm > 0.5).sum())}/{lsm.size}, "
            f"cells==0={int((lsm == 0).sum())}/{lsm.size}"
        )

    print("\n=== 6. stats.json Summary ===")
    with STATS_PATH.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    print(f"stats.json keys: {list(stats.keys())}")
    print(f"stats.json channels: {stats['channels']}")
    print(f"stats.json mean: {stats['mean']}")
    print(f"stats.json std:  {stats['std']}")
    mean_diff = np.max(np.abs(np.array(stats["mean"]) - data_arr.mean(axis=(0, 2, 3))))
    std_diff = np.max(np.abs(np.array(stats["std"]) - data_arr.std(axis=(0, 2, 3))))
    print(f"Max abs diff vs computed dataset mean={mean_diff:.3e}, std={std_diff:.3e}")

    print("\n=== 7. DOY Convention Verification (Contract Item 2) ===")
    doy = times.day_of_year.to_numpy(dtype=np.int64)  # 1..366
    n_leap_366 = int((doy == 366).sum())
    leap_years = sorted(set(int(y) for y in times.year[doy == 366]))
    period = 365.0
    angle = 2.0 * np.pi * doy / period
    sin_doy = np.sin(angle)
    cos_doy = np.cos(angle)
    decoded_frac = (np.arctan2(sin_doy, cos_doy) / (2.0 * np.pi)) % 1.0
    expected_frac = (doy % 365) / period
    wrap_err = np.abs(np.angle(np.exp(1j * 2.0 * np.pi * (decoded_frac - expected_frac)))) / (2.0 * np.pi)
    max_wrap_err = float(np.max(wrap_err))

    print(f"DOY range in dataset: {int(doy.min())}..{int(doy.max())}")
    print(f"Period used: {int(period)} (NOT 365.25)")
    print("Encoding confirmed: doy_sincos = [sin(2*pi*doy/365), cos(2*pi*doy/365)] for LAST input day; step k uses doy + k.")
    print(f"Max phase round-trip error via arctan2(sin, cos): {max_wrap_err:.3e}")
    print(f"Leap-year day 366 occurrences: {n_leap_366} days (years: {leap_years})")
    s366, c366 = np.sin(2.0 * np.pi * 366 / period), np.cos(2.0 * np.pi * 366 / period)
    s001, c001 = np.sin(2.0 * np.pi * 1 / period), np.cos(2.0 * np.pi * 1 / period)
    print(
        f"Leap-day handling: with fixed period=365, doy=366 evaluates to "
        f"[sin={s366:.6f}, cos={c366:.6f}], which equals doy=1 [sin={s001:.6f}, cos={c001:.6f}] "
        f"(max diff={max(abs(s366 - s001), abs(c366 - c001)):.3e}); "
        f"if forecast step k yields doy+k > 365, modular 365 wrapping keeps the seasonal phase continuous."
    )

    print("\n=== 8. Export Static Grid & Bundled Static Weights ===")
    lsm_idx = static_names.index("land_sea_mask")
    oro_idx = static_names.index("orography_m")
    grid_payload = {
        "lat": [round(float(v), 2) for v in lat],
        "lon": [round(float(v), 2) for v in lon],
        "land_sea_mask": [[round(float(v), 2) for v in row] for row in static_f32[lsm_idx]],
        "orography_m": [[round(float(v), 2) for v in row] for row in static_f32[oro_idx]],
    }
    GRID_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GRID_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(grid_payload, f)
    print(f"Exported {GRID_JSON_PATH} ({GRID_JSON_PATH.stat().st_size} bytes)")

    STATIC_NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        STATIC_NPZ_PATH,
        static=static_f32,
        static_channels=np.array(static_names),
        lat=lat.astype(np.float32),
        lon=lon.astype(np.float32),
    )
    print(f"Exported {STATIC_NPZ_PATH} ({STATIC_NPZ_PATH.stat().st_size} bytes)")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    n_train = int(data_arr.shape[0] * float(cfg["split"]["train"]))
    train_std = data_arr[:n_train].std(axis=(0, 2, 3)).astype(np.float64)
    std_payload = {
        "channels": channels,
        "std": [round(float(v), 6) for v in train_std],
        "n_train": n_train,
    }
    with CHANNEL_STD_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(std_payload, f, indent=2)
    print(f"Exported TRAIN physical channel stds -> {CHANNEL_STD_JSON_PATH}: {std_payload['std']}")
    ds.close()


if __name__ == "__main__":
    main()
