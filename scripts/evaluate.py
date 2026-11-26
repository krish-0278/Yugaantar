"""Comprehensive Phase 3b evaluation on VAL (tuned) and TEST windows plus 30-step stability & clamp checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monsoon.climatology import evaluate as eval_harmonic, evaluate_empirical, fit_empirical
from monsoon.data import CH_TP, from_model_space, load_dataset, split_indices, to_model_space
from monsoon.metrics import (
    day8_spatial_variance_ratio,
    heavy_rain_csi_details,
    lat_weighted_rmse,
    lat_weights,
)
from monsoon.model import MonsoonNet
from monsoon.postproc import apply as apply_postproc, load_postproc
from monsoon.rollout import from_anomaly, load_rollout_context, rollout
from monsoon.windows import select_heldout_windows


CONFIG_PATH = ROOT / "config" / "default.json"
MODEL_PATH = ROOT / "weights" / "model.pt"
POSTPROC_PATH = ROOT / "weights" / "postproc.json"
RESULTS_MD = ROOT / "docs" / "phase3_results.md"


def _eval_window_set(
    split_name: str,
    tag: str,
    windows: list[dict[str, Any]],
    model: MonsoonNet,
    data_phys: np.ndarray,
    doy: np.ndarray,
    lat_w: np.ndarray,
    emp_table: np.ndarray,
    pp_cfg: dict[str, Any],
    ctx: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    channels = cfg["channels"]
    lead_days = [int(x) for x in cfg["metrics"]["lead_days"]]
    max_lead = max(10, max(lead_days))
    harm_coef = ctx["harm_coef"]
    harm_k = ctx["harm_k"]
    period = ctx["period"]
    anom_clamp = float(cfg["stage_b"]["anom_clamp"])
    input_clip = float(cfg["norm"]["anomaly_clip"])
    rain_thresh = float(cfg["metrics"]["rain_threshold_mm"])

    n_win = len(windows)
    t_arr = np.array([w["t_last"] for w in windows], dtype=np.int64)

    true_roll = np.empty((n_win, max_lead, 6, 32, 36), dtype=np.float32)
    pers_roll = np.empty_like(true_roll)
    harm_roll = np.empty_like(true_roll)
    emp_roll = np.empty_like(true_roll)

    raw_roll = np.empty_like(true_roll)
    ab_beta = np.empty_like(true_roll)
    ab_kappa = np.empty_like(true_roll)
    ab_tp_var = np.empty_like(true_roll)
    ab_event = np.empty_like(true_roll)
    post_roll = np.empty_like(true_roll)

    for i, t in enumerate(t_arr):
        init_phys = data_phys[t - 2 : t + 1]
        doy_last = int(doy[t])
        doy_targets = doy_last + np.arange(1, max_lead + 1, dtype=np.int64)

        true_roll[i] = data_phys[t + 1 : t + 1 + max_lead]
        pers_roll[i] = np.repeat(data_phys[t][None], max_lead, axis=0)
        harm_roll[i] = from_model_space(eval_harmonic(harm_coef, doy_targets / float(period), K=harm_k))
        emp_roll[i] = evaluate_empirical(emp_table, doy_targets, period=period)

        a_out, ev_out = rollout(
            model, init_phys, doy_last, max_lead, delta=None, ctx=ctx, anom_clamp=anom_clamp, input_clip=input_clip
        )
        raw_roll[i] = from_anomaly(a_out, doy_targets, ctx=ctx)
        ab_beta[i] = apply_postproc(
            a_out, ev_out, doy_last, max_lead, pp_cfg, ctx,
            use_beta=True, use_kappa=False, use_tp_var=False, use_event=False, use_clamp=False,
        )
        ab_kappa[i] = apply_postproc(
            a_out, ev_out, doy_last, max_lead, pp_cfg, ctx,
            use_beta=True, use_kappa=True, use_tp_var=False, use_event=False, use_clamp=False,
        )
        ab_tp_var[i] = apply_postproc(
            a_out, ev_out, doy_last, max_lead, pp_cfg, ctx,
            use_beta=True, use_kappa=True, use_tp_var=True, use_event=False, use_clamp=False,
        )
        ab_event[i] = apply_postproc(
            a_out, ev_out, doy_last, max_lead, pp_cfg, ctx,
            use_beta=True, use_kappa=True, use_tp_var=True, use_event=True, use_clamp=False,
        )
        post_roll[i] = apply_postproc(
            a_out, ev_out, doy_last, max_lead, pp_cfg, ctx,
            use_beta=True, use_kappa=True, use_tp_var=True, use_event=True, use_clamp=True,
        )

    methods = {
        "persistence": pers_roll,
        "clim_harmonic": harm_roll,
        "clim_empirical": emp_roll,
        "model_raw": raw_roll,
        "+beta": ab_beta,
        "+kappa": ab_kappa,
        "+tp_var": ab_tp_var,
        "+event": ab_event,
        "model_postproc": post_roll,
    }

    rmse_tbl: dict[int, dict[str, np.ndarray]] = {}
    skill_tbl: dict[int, dict[str, tuple[np.ndarray, float]]] = {}
    csi_tbl: dict[int, dict[str, dict[str, Any]]] = {}

    for lead in lead_days:
        idx = lead - 1
        t_l = true_roll[:, idx]
        rmse_tbl[lead] = {}
        skill_tbl[lead] = {}
        csi_tbl[lead] = {}
        for mname, roll in methods.items():
            p_l = roll[:, idx]
            rmse_tbl[lead][mname] = lat_weighted_rmse(p_l, t_l, lat_w)
            csi_tbl[lead][mname] = heavy_rain_csi_details(p_l[:, CH_TP], t_l[:, CH_TP], threshold_mm=rain_thresh)

        best_base_rmse = np.minimum(
            rmse_tbl[lead]["persistence"],
            np.minimum(rmse_tbl[lead]["clim_harmonic"], rmse_tbl[lead]["clim_empirical"]),
        )
        for mname in methods:
            ss = 1.0 - rmse_tbl[lead][mname] / best_base_rmse
            skill_tbl[lead][mname] = (ss, float(np.mean(ss)))

    vr_day8: dict[str, np.ndarray] = {}
    for mname, roll in methods.items():
        vr_day8[mname] = day8_spatial_variance_ratio(roll[:, 7], true_roll[:, 7])

    return {
        "split": split_name,
        "tag": tag,
        "n_win": n_win,
        "channels": channels,
        "lead_days": lead_days,
        "rmse": rmse_tbl,
        "skill": skill_tbl,
        "csi": csi_tbl,
        "vr_day8": vr_day8,
    }


def _run_stability_30(
    model: MonsoonNet,
    data_phys: np.ndarray,
    doy: np.ndarray,
    splits: dict[str, np.ndarray],
    val_wins: list[dict[str, Any]],
    test_wins: list[dict[str, Any]],
    pp_cfg: dict[str, Any],
    ctx: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    steps = 30
    anom_clamp = float(cfg["stage_b"]["anom_clamp"])
    input_clip = float(cfg["norm"]["anomaly_clip"])
    harm_coef = ctx["harm_coef"]
    harm_k = ctx["harm_k"]
    period = ctx["period"]
    sigma_pix = ctx["norm"]["sigma_pix"]

    all_runs: list[np.ndarray] = []
    max_abs_anom = 0.0
    nan_inf_count = 0

    for w in val_wins + test_wins:
        t = int(w["t_last"])
        init_phys = data_phys[t - 2 : t + 1]
        doy_last = int(doy[t])
        anom_out, ev_out = rollout(model, init_phys, doy_last, steps, delta=None, ctx=ctx, anom_clamp=anom_clamp, input_clip=input_clip)
        if not np.isfinite(anom_out).all():
            nan_inf_count += int((~np.isfinite(anom_out)).sum())
        else:
            max_abs_anom = max(max_abs_anom, float(np.max(np.abs(anom_out))))
        phys_out = apply_postproc(anom_out, ev_out, doy_last, steps, pp_cfg, ctx)
        if not np.isfinite(phys_out).all():
            nan_inf_count += int((~np.isfinite(phys_out)).sum())
        all_runs.append(phys_out)

    rng = np.random.default_rng(int(cfg["seed"]) + 999)
    held_idx = np.concatenate([splits["val"][2:-steps], splits["test"][2:-steps]])
    sampled_t = rng.choice(held_idx, size=200, replace=True)

    noisy_clamp_hits = np.zeros(6, dtype=np.int64)
    noisy_total_per_ch = 200 * steps * 32 * 36

    for t in sampled_t:
        t = int(t)
        init_phys = data_phys[t - 2 : t + 1]
        doy_last = int(doy[t])
        doy_init = np.array([doy_last - 2, doy_last - 1, doy_last], dtype=np.int64)
        clim_init = eval_harmonic(harm_coef, doy_init / float(period), K=harm_k)
        y_init = to_model_space(init_phys)
        a_init = (y_init - clim_init) / sigma_pix[None]
        a_noisy = a_init + rng.normal(0.0, 1.0, size=a_init.shape).astype(np.float32)
        init_noisy = from_model_space(clim_init + sigma_pix[None] * a_noisy)
        init_noisy[:, 0] = np.clip(init_noisy[:, 0], 233.15, 333.0)

        anom_out, ev_out = rollout(model, init_noisy, doy_last, steps, delta=None, ctx=ctx, anom_clamp=anom_clamp, input_clip=input_clip)
        if not np.isfinite(anom_out).all():
            nan_inf_count += int((~np.isfinite(anom_out)).sum())
        else:
            max_abs_anom = max(max_abs_anom, float(np.max(np.abs(anom_out))))
            noisy_clamp_hits += (np.abs(anom_out) >= (anom_clamp - 1e-6)).sum(axis=(0, 2, 3))
        phys_out = apply_postproc(anom_out, ev_out, doy_last, steps, pp_cfg, ctx)
        if not np.isfinite(phys_out).all():
            nan_inf_count += int((~np.isfinite(phys_out)).sum())
        all_runs.append(phys_out)

    arr = np.stack(all_runs, axis=0)
    spatial_std_vs_step = arr.std(axis=(-2, -1)).mean(axis=0)
    noisy_clamp_frac = noisy_clamp_hits.astype(np.float64) / float(noisy_total_per_ch)
    return {
        "n_rollouts": arr.shape[0],
        "nan_inf_count": nan_inf_count,
        "min_tp": float(arr[:, :, 1].min()),
        "t2m_min_c": float(arr[:, :, 0].min() - 273.15),
        "t2m_max_c": float(arr[:, :, 0].max() - 273.15),
        "max_abs_anom": max_abs_anom,
        "spatial_std_vs_step": spatial_std_vs_step,
        "noisy_clamp_frac": noisy_clamp_frac,
    }


def _render_eval_md(res: dict[str, Any], tp_var_label: str) -> list[str]:
    channels = res["channels"]
    lead_days = res["lead_days"]
    lines = [f"### Split: `{res['split']}` — {res['tag']} ({res['n_win']} windows)", ""]

    rows_order = ["persistence", "clim_harmonic", "clim_empirical", "model_raw", "+beta", "+kappa", "+tp_var", "+event", "model_postproc"]
    disp_name = {k: (f"+tp_var ({tp_var_label})" if k == "+tp_var" else k) for k in rows_order}

    lines.append("#### Lat-Weighted RMSE & Skill vs Best Baseline (`min(pers, harm, emp)`)")
    for lead in lead_days:
        lines.append(f"##### Lead Day {lead} ({res['tag']})")
        lines.append("| Method | " + " | ".join(channels) + " | Mean Skill |")
        lines.append("|---|" + "---:|" * len(channels) + "---:|")
        for mname in rows_order:
            rm = res["rmse"][lead][mname]
            _, mean_ss = res["skill"][lead][mname]
            vals = " | ".join(f"{rm[c]:.4f}" for c in range(len(channels)))
            lines.append(f"| `{disp_name[mname]}` | {vals} | {mean_ss:+.4f} |")
        ss_post, mean_post = res["skill"][lead]["model_postproc"]
        ss_raw, mean_raw = res["skill"][lead]["model_raw"]
        lines.append(
            f"| `skill (model_raw)` | " + " | ".join(f"{ss_raw[c]:+.4f}" for c in range(len(channels))) + f" | {mean_raw:+.4f} |"
        )
        lines.append(
            f"| `skill (model_postproc)` | " + " | ".join(f"{ss_post[c]:+.4f}" for c in range(len(channels))) + f" | {mean_post:+.4f} |"
        )
        lines.append("")

    lines.append("#### Heavy-Rain CSI (>= 10.0 mm/day)")
    lines.append("| Lead | Method | CSI | Hits | Misses | False Alarms |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for lead in lead_days:
        for mname in rows_order:
            d = res["csi"][lead][mname]
            lines.append(f"| {lead} | `{disp_name[mname]}` | {d['csi']:.4f} | {d['hits']} | {d['misses']} | {d['false_alarms']} |")
    lines.append("")

    lines.append("#### Day-8 Spatial Variance Ratio (`var(pred) / var(true)`, pass band `0.40..2.50`, `tp >= 0.40` test)")
    lines.append("| Method | " + " | ".join(channels) + " |")
    lines.append("|---|" + "---:|" * len(channels))
    for mname in rows_order:
        vr = res["vr_day8"][mname]
        lines.append(f"| `{disp_name[mname]}` | " + " | ".join(f"{vr[c]:.4f}" for c in range(len(channels))) + " |")
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
    lat_w = lat_weights(lat)

    splits = split_indices(
        data_phys.shape[0],
        train_frac=float(cfg["split"]["train"]),
        val_frac=float(cfg["split"]["val"]),
        test_frac=float(cfg["split"]["test"]),
    )

    ctx = load_rollout_context()
    pp_cfg = load_postproc(POSTPROC_PATH)
    harm_coef = ctx["harm_coef"]
    harm_k = ctx["harm_k"]
    period = ctx["period"]
    smooth_win = int(cfg["climatology"]["empirical_smooth_window"])

    emp_table = fit_empirical(data_phys[splits["train"]], doy[splits["train"]], smooth_window=smooth_win, period=period)
    clim_all = from_model_space(eval_harmonic(harm_coef, doy / float(period), K=harm_k))

    win_by_split, _ = select_heldout_windows(
        data=data_phys,
        doy=doy,
        clim_phys=clim_all,
        splits=splits,
        n_in=int(cfg["windows"]["n_in"]),
        n_out=int(cfg["windows"]["n_out"]),
        per_category=int(cfg["windows"]["per_category"]),
        min_sep=int(cfg["windows"]["min_separation_days"]),
        seed=int(cfg["seed"]),
    )

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    model = MonsoonNet(
        cin=int(mc["cin"]),
        cout=int(mc["cout"]),
        channels=int(mc["channels"]),
        n_blocks=int(mc["n_blocks"]),
        expansion=int(mc["expansion"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    res_val = _eval_window_set(
        "val", "TUNED (in-sample for post-processing)", win_by_split["val"], model, data_phys, doy, lat_w, emp_table, pp_cfg, ctx, cfg
    )
    res_test = _eval_window_set(
        "test", "OFFICIAL GATE SET (unseen during training & tuning)", win_by_split["test"], model, data_phys, doy, lat_w, emp_table, pp_cfg, ctx, cfg
    )
    stab = _run_stability_30(
        model, data_phys, doy, splits, win_by_split["val"], win_by_split["test"], pp_cfg, ctx, cfg
    )

    tp_var_label = f"{pp_cfg.get('tp_var_method', 'qm')}, g8={pp_cfg.get('pattern_gain', [1]*8)[7]:.1f}"
    new_lines: list[str] = [
        "",
        "---",
        "",
        f"## Phase 3b Update — Pattern Gain (`{tp_var_label}`, val target `tp VR >= 0.45`)",
        "",
        "Parameters tuned strictly on `val` (rows under `val` marked `TUNED`); **only `test` columns count for the Phase 3b Gate.**",
        "",
    ]
    new_lines.extend(_render_eval_md(res_val, tp_var_label))
    new_lines.extend(_render_eval_md(res_test, tp_var_label))

    new_lines.extend([
        "### 30-Step Stability (50 val + 50 test + 200 1-sigma noisy starts = 300 rollouts)",
        f"- NaN/Inf count: `{stab['nan_inf_count']}`",
        f"- Min `tp`: `{stab['min_tp']:.6f}` mm/day",
        f"- `t2m` range: `{stab['t2m_min_c']:.2f} C` .. `{stab['t2m_max_c']:.2f} C` (allowed `-40.00 .. +60.00 C`)",
        f"- Max `|anomaly|` in rollout: `{stab['max_abs_anom']:.4f}` (clamp `5.0`)",
        "",
        "#### Fraction of Anomaly Values at `+-5.0` Clamp Over 200 Noisy-Start 30-Step Rollouts",
        "| Channel | Clamp Fraction | Percent | Flag (> 1%) |",
        "|---|---:|---:|---|",
    ])
    for c, ch in enumerate(channels):
        frac = float(stab["noisy_clamp_frac"][c])
        flag = "FLAG (> 1%)" if frac > 0.01 else "OK"
        new_lines.append(f"| `{ch}` | {frac:.6f} | {100.0 * frac:.4f}% | {flag} |")
    new_lines.append("")

    new_lines.extend([
        "#### Mean Spatial Std per Channel vs Step",
        "| Step | " + " | ".join(channels) + " |",
        "|---:|" + "---:|" * len(channels),
    ])
    for step in (1, 4, 8, 10, 15, 20, 25, 30):
        vals = stab["spatial_std_vs_step"][step - 1]
        new_lines.append(f"| {step} | " + " | ".join(f"{vals[c]:.4f}" for c in range(len(channels))) + " |")
    new_lines.append("")

    # Preserve earlier Phase 3 section and append Phase 3b section (avoiding duplicate appends if re-run)
    marker = "## Phase 3b Update"
    existing = RESULTS_MD.read_text(encoding="utf-8") if RESULTS_MD.exists() else ""
    if marker in existing:
        base_text = existing.split("---\n\n" + marker)[0].rstrip()
    else:
        base_text = existing.rstrip()
    RESULTS_MD.write_text(base_text + "\n" + "\n".join(new_lines), encoding="utf-8")
    print(f"Appended Phase 3b results -> {RESULTS_MD}")

    print("\n=== NOISY-START +-5 CLAMP FRACTIONS (200 x 30 steps) ===")
    for c, ch in enumerate(channels):
        frac = float(stab["noisy_clamp_frac"][c])
        print(f"  {ch:<5}: {100.0 * frac:.4f}% ({'FLAG > 1%' if frac > 0.01 else 'OK'})")

    # Phase 3b Gate Summary on TEST windows
    print("\n=== PHASE 3b GATE SUMMARY (TEST WINDOWS) ===")
    r4_mod = res_test["rmse"][4]["model_postproc"]
    r4_pers = res_test["rmse"][4]["persistence"]
    g1 = bool(np.all(r4_mod < r4_pers))
    print(f"Gate 1 (lead 4 beats persistence on all 6 channels): {'PASS' if g1 else 'FAIL'}")
    for c, ch in enumerate(channels):
        print(f"   {ch}: model={r4_mod[c]:.4f} vs pers={r4_pers[c]:.4f}")

    for lead in (1, 4, 8):
        ss, mean_ss = res_test["skill"][lead]["model_postproc"]
        ch_str = ", ".join(f"{channels[c]}={ss[c]:+.4f}" for c in range(len(channels)))
        if lead == 8:
            ok8 = (mean_ss >= 0.0) and (float(ss[CH_TP]) >= -0.06)
            print(f"Gate 2 Lead 8 mean_skill={mean_ss:+.4f} (>=0), tp_skill={ss[CH_TP]:+.4f} (>=-0.06): {'PASS' if ok8 else 'FAIL'} | {ch_str}")
        else:
            print(f"Gate 2 Lead {lead} mean_skill={mean_ss:+.4f}: {'PASS' if mean_ss > 0 else 'FAIL'} | {ch_str}")

    vr8 = res_test["vr_day8"]["model_postproc"]
    g3 = bool(np.all((vr8 >= 0.40) & (vr8 <= 2.50)) and vr8[CH_TP] >= 0.40)
    print(f"Gate 3 (day-8 variance ratio in [0.40, 2.50], tp >= 0.40): {'PASS' if g3 else 'FAIL'} | "
          + ", ".join(f"{channels[c]}={vr8[c]:.4f}" for c in range(len(channels))))

    g4 = True
    for lead in (4, 8):
        csi_mod = res_test["csi"][lead]["model_postproc"]["csi"]
        csi_emp = res_test["csi"][lead]["clim_empirical"]["csi"]
        ok_l = csi_mod > csi_emp
        g4 &= ok_l
        print(f"Gate 4 Lead {lead} CSI: model={csi_mod:.4f} vs emp_clim={csi_emp:.4f} ({'PASS' if ok_l else 'FAIL'})")

    g5 = (
        stab["nan_inf_count"] == 0
        and stab["min_tp"] >= 0.0
        and stab["t2m_min_c"] >= -40.0
        and stab["t2m_max_c"] <= 60.0
    )
    print(f"Gate 5 (30-step stability): {'PASS' if g5 else 'FAIL'} "
          f"(nan={stab['nan_inf_count']}, min_tp={stab['min_tp']:.4f}, t2m=[{stab['t2m_min_c']:.2f}, {stab['t2m_max_c']:.2f}] C)")


if __name__ == "__main__":
    main()
