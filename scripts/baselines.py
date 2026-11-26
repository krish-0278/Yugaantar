"""Evaluate persistence, harmonic climatology, and empirical climatology on val & test windows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monsoon.climatology import (
    evaluate as eval_harmonic,
    evaluate_empirical,
    fit as fit_harmonic,
    fit_empirical,
    save_harmonics,
)
from monsoon.data import CH_TP, from_model_space, load_dataset, split_indices, to_model_space
from monsoon.metrics import (
    day8_spatial_variance_ratio,
    heavy_rain_csi_details,
    lat_weighted_rmse,
    lat_weights,
)
from monsoon.windows import CATEGORIES, select_heldout_windows


CONFIG_PATH = ROOT / "config" / "default.json"
HARMONICS_PATH = ROOT / "weights" / "clim_harmonics.npz"
BASELINES_MD = ROOT / "docs" / "baselines.md"


def _evaluate_split(
    split_name: str,
    windows: list[dict[str, Any]],
    data_phys: np.ndarray,
    doy: np.ndarray,
    lat_w: np.ndarray,
    harm_coef: np.ndarray,
    emp_table: np.ndarray,
    harm_k: int,
    period: int,
    lead_days: list[int],
    rain_thresh: float,
) -> dict[str, Any]:
    n_win = len(windows)
    max_lead = max(lead_days)
    t_last_arr = np.array([w["t_last"] for w in windows], dtype=np.int64)

    # Build true and baseline arrays of shape [N, max_lead, C, H, W]
    true_roll = np.stack([data_phys[t + 1 : t + 1 + max_lead] for t in t_last_arr], axis=0)
    pers_roll = np.stack([np.repeat(data_phys[t][None], max_lead, axis=0) for t in t_last_arr], axis=0)

    target_doys = np.stack([doy[t] + np.arange(1, max_lead + 1, dtype=np.int64) for t in t_last_arr], axis=0)
    flat_doys = target_doys.reshape(-1)
    harm_model = eval_harmonic(harm_coef, flat_doys / float(period), K=harm_k)
    harm_roll = from_model_space(harm_model).reshape(n_win, max_lead, *data_phys.shape[1:])
    emp_roll = evaluate_empirical(emp_table, flat_doys, period=period).reshape(n_win, max_lead, *data_phys.shape[1:])

    baselines = {
        "persistence": pers_roll,
        "clim_harmonic": harm_roll,
        "clim_empirical": emp_roll,
    }

    rmse_by_lead: dict[int, dict[str, np.ndarray]] = {}
    csi_by_lead: dict[int, dict[str, dict[str, Any]]] = {}
    for lead in lead_days:
        idx = lead - 1
        true_l = true_roll[:, idx]
        rmse_by_lead[lead] = {}
        csi_by_lead[lead] = {}
        for name, roll in baselines.items():
            pred_l = roll[:, idx]
            rmse_by_lead[lead][name] = lat_weighted_rmse(pred_l, true_l, lat_w)
            csi_by_lead[lead][name] = heavy_rain_csi_details(pred_l[:, CH_TP], true_l[:, CH_TP], threshold_mm=rain_thresh)

    var_ratio_day8 = {}
    idx8 = 8 - 1
    for name, roll in baselines.items():
        var_ratio_day8[name] = day8_spatial_variance_ratio(roll[:, idx8], true_roll[:, idx8])

    return {
        "split": split_name,
        "n_windows": n_win,
        "rmse": rmse_by_lead,
        "csi": csi_by_lead,
        "var_ratio_day8": var_ratio_day8,
    }


def _render_split_md(
    res: dict[str, Any],
    windows: list[dict[str, Any]],
    shortfalls: dict[str, dict[str, int]],
    channels: list[str],
    lead_days: list[int],
    rain_thresh: float,
) -> list[str]:
    split_name = res["split"]
    lines: list[str] = [f"## Split: `{split_name}` ({res['n_windows']} windows)", ""]

    # Category counts & shortfall report
    counts = {c: sum(1 for w in windows if w["category"] == c) for c in CATEGORIES}
    lines.append("### Window Counts by Category (non-overlapping, >= 13 days apart)")
    lines.append("| Category | Count | Shortfall |")
    lines.append("|---|---:|---:|")
    for c in CATEGORIES:
        sf = shortfalls[c]["shortfall"] if c in shortfalls else 0
        lines.append(f"| {c} | {counts[c]} | {sf} |")
    lines.append("")

    # RMSE & best baseline per channel and lead
    lines.append("### Lat-Weighted RMSE & Best Baseline by Lead")
    for lead in lead_days:
        lines.append(f"#### Lead Day {lead}")
        lines.append("| Channel | Persistence | Clim (Harmonic) | Clim (Empirical) | Best (All 3) | Best (Pers vs Harm) | Harm - Emp |")
        lines.append("|---|---:|---:|---:|---|---|---:|")
        rp = res["rmse"][lead]["persistence"]
        rh = res["rmse"][lead]["clim_harmonic"]
        re_ = res["rmse"][lead]["clim_empirical"]
        for ci, ch in enumerate(channels):
            vals3 = {"persistence": rp[ci], "clim_harmonic": rh[ci], "clim_empirical": re_[ci]}
            best3 = min(vals3, key=vals3.get)
            best2 = "persistence" if rp[ci] <= rh[ci] else "clim_harmonic"
            diff_he = rh[ci] - re_[ci]
            lines.append(
                f"| `{ch}` | {rp[ci]:.4f} | {rh[ci]:.4f} | {re_[ci]:.4f} | {best3} | {best2} | {diff_he:+.4f} |"
            )
        lines.append("")

    # CSI & heavy-rain event counts
    lines.append(f"### Heavy-Rain CSI (>= {rain_thresh:.1f} mm/day) & Event Counts")
    lines.append("| Lead | Baseline | CSI | Hits | Misses | False Alarms | True Events (>=10 mm) | Pred Events (>=10 mm) |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for lead in lead_days:
        for name in ("persistence", "clim_harmonic", "clim_empirical"):
            d = res["csi"][lead][name]
            lines.append(
                f"| {lead} | {name} | {d['csi']:.4f} | {d['hits']} | {d['misses']} | {d['false_alarms']} | {d['true_events']} | {d['pred_events']} |"
            )
    lines.append("")

    # Day-8 spatial variance ratio
    lines.append("### Day-8 Spatial Variance Ratio (`var(pred) / var(true)`, target band `0.40 .. 2.50`)")
    lines.append("| Channel | Persistence | Clim (Harmonic) | Clim (Empirical) |")
    lines.append("|---|---:|---:|---:|")
    vp = res["var_ratio_day8"]["persistence"]
    vh = res["var_ratio_day8"]["clim_harmonic"]
    ve = res["var_ratio_day8"]["clim_empirical"]
    for ci, ch in enumerate(channels):
        lines.append(f"| `{ch}` | {vp[ci]:.4f} | {vh[ci]:.4f} | {ve[ci]:.4f} |")
    lines.append("")
    return lines


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ds = load_dataset()
    data_phys = ds["data"]
    lat = ds["lat"]
    doy = ds["doy"]
    channels = ds["channels"]
    T = data_phys.shape[0]

    splits = split_indices(
        T,
        train_frac=cfg["split"]["train"],
        val_frac=cfg["split"]["val"],
        test_frac=cfg["split"]["test"],
    )
    train_idx = splits["train"]
    heldout_idx = np.concatenate([splits["val"], splits["test"]])

    rain_thresh = float(cfg["metrics"]["rain_threshold_mm"])
    train_rain_frac = float(np.mean(data_phys[train_idx, CH_TP] >= rain_thresh))
    val_rain_frac = float(np.mean(data_phys[splits["val"], CH_TP] >= rain_thresh))
    test_rain_frac = float(np.mean(data_phys[splits["test"], CH_TP] >= rain_thresh))
    heldout_rain_frac = float(np.mean(data_phys[heldout_idx, CH_TP] >= rain_thresh))

    print("=== Heavy-Rain Cell Fractions (tp >= 10.0 mm/day) ===")
    print(f"Train split ({len(train_idx)} days):    {train_rain_frac:.6f} ({100.0 * train_rain_frac:.3f}%)")
    print(f"Val split ({len(splits['val'])} days):      {val_rain_frac:.6f} ({100.0 * val_rain_frac:.3f}%)")
    print(f"Test split ({len(splits['test'])} days):     {test_rain_frac:.6f} ({100.0 * test_rain_frac:.3f}%)")
    print(f"Held-out val+test ({len(heldout_idx)} days): {heldout_rain_frac:.6f} ({100.0 * heldout_rain_frac:.3f}%)")

    # Fit harmonic climatology (in model space) & empirical climatology (in physical space) on TRAIN ONLY
    harm_k = int(cfg["climatology"]["harmonics_k"])
    period = int(cfg["climatology"]["doy_period"])
    ridge = float(cfg["climatology"]["ridge"])
    smooth_win = int(cfg["climatology"]["empirical_smooth_window"])

    data_model = to_model_space(data_phys)
    harm_coef = fit_harmonic(data_model[train_idx], doy[train_idx] / float(period), K=harm_k, ridge=ridge)
    save_harmonics(HARMONICS_PATH, harm_coef, K=harm_k, period=period)
    print(f"\nSaved harmonic climatology coefficients {harm_coef.shape} -> {HARMONICS_PATH}")

    emp_table = fit_empirical(data_phys[train_idx], doy[train_idx], smooth_window=smooth_win, period=period)
    print(f"Built empirical climatology table {emp_table.shape} (train only, {smooth_win}-day circular smoothing)")

    # Harmonic climatology across all T days for data-driven heatwave window detection
    clim_phys_all = from_model_space(eval_harmonic(harm_coef, doy / float(period), K=harm_k))

    windows_by_split, shortfalls = select_heldout_windows(
        data=data_phys,
        doy=doy,
        clim_phys=clim_phys_all,
        splits=splits,
        n_in=int(cfg["windows"]["n_in"]),
        n_out=int(cfg["windows"]["n_out"]),
        per_category=int(cfg["windows"]["per_category"]),
        min_sep=int(cfg["windows"]["min_separation_days"]),
        seed=int(cfg["seed"]),
    )

    for split_name in ("val", "test"):
        if shortfalls[split_name]:
            print(f"Window shortfalls in {split_name}: {shortfalls[split_name]}")
        else:
            print(f"Window selection in {split_name}: all {len(CATEGORIES)} categories supplied {cfg['windows']['per_category']} non-overlapping windows.")

    lat_w = lat_weights(lat)
    lead_days = [int(x) for x in cfg["metrics"]["lead_days"]]
    res_val = _evaluate_split(
        "val", windows_by_split["val"], data_phys, doy, lat_w, harm_coef, emp_table, harm_k, period, lead_days, rain_thresh
    )
    res_test = _evaluate_split(
        "test", windows_by_split["test"], data_phys, doy, lat_w, harm_coef, emp_table, harm_k, period, lead_days, rain_thresh
    )

    # Print harmonic vs empirical comparison summary
    print("\n=== Harmonic vs Empirical Climatology RMSE (Harm - Emp) ===")
    for res in (res_val, res_test):
        print(f"[{res['split']}]")
        for lead in lead_days:
            rh = res["rmse"][lead]["clim_harmonic"]
            re_ = res["rmse"][lead]["clim_empirical"]
            diffs = ", ".join(f"{ch}: {rh[ci]-re_[ci]:+.4f}" for ci, ch in enumerate(channels))
            print(f"  Lead {lead}: {diffs}")

    # Write docs/baselines.md
    md_lines: list[str] = [
        "# Round 1 Baseline Evaluation (`val` and `test` Separate)",
        "",
        "Fitted strictly on the `train` split (first 82% = 5,091 days; `val` = 558 days; `test` = 560 days).",
        "Windows are non-overlapping (`min_separation_days = 13`, >= 10 days apart) and selected strictly inside each split.",
        "",
        "## Heavy-Rain Cell Fraction (`tp >= 10.0 mm/day`)",
        f"- `train` ({len(train_idx)} days): `{train_rain_frac:.6f}` ({100.0 * train_rain_frac:.3f}%)",
        f"- `val` ({len(splits['val'])} days): `{val_rain_frac:.6f}` ({100.0 * val_rain_frac:.3f}%)",
        f"- `test` ({len(splits['test'])} days): `{test_rain_frac:.6f}` ({100.0 * test_rain_frac:.3f}%)",
        f"- `held-out (val + test)` ({len(heldout_idx)} days): `{heldout_rain_frac:.6f}` ({100.0 * heldout_rain_frac:.3f}%)",
        "",
    ]
    md_lines.extend(_render_split_md(res_val, windows_by_split["val"], shortfalls["val"], channels, lead_days, rain_thresh))
    md_lines.extend(_render_split_md(res_test, windows_by_split["test"], shortfalls["test"], channels, lead_days, rain_thresh))

    BASELINES_MD.parent.mkdir(parents=True, exist_ok=True)
    BASELINES_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote {BASELINES_MD}")


if __name__ == "__main__":
    main()
