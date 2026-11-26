"""Fit Phase 3b post-processing parameters strictly on VAL and compare Method G vs Method QM."""

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
from monsoon.postproc import apply as apply_postproc
from monsoon.rollout import from_anomaly, load_rollout_context, rollout
from monsoon.windows import select_heldout_windows


CONFIG_PATH = ROOT / "config" / "default.json"
MODEL_PATH = ROOT / "weights" / "model.pt"
POSTPROC_PATH = ROOT / "weights" / "postproc.json"


def _build_qm_knots(pred_vals: np.ndarray, true_vals: np.ndarray) -> tuple[list[float], list[float]]:
    qs = np.concatenate(
        [
            np.linspace(0.0, 0.85, 86),
            np.linspace(0.855, 0.99, 28),
            np.linspace(0.991, 1.0, 10),
        ]
    )
    qp = np.quantile(pred_vals.astype(np.float64), qs)
    qt = np.quantile(true_vals.astype(np.float64), qs)
    uniq_x: list[float] = [0.0]
    uniq_y: list[float] = [0.0]
    for xp, yt in zip(qp, qt):
        if xp > uniq_x[-1] + 1e-6:
            uniq_x.append(round(float(xp), 5))
            uniq_y.append(round(float(max(yt, 0.0)), 5))
    if len(uniq_x) == 1:
        uniq_x.append(1.0)
        uniq_y.append(1.0)
    return uniq_x, uniq_y


def _apply_pattern_gain_batch(tp_batch: np.ndarray, g: float) -> np.ndarray:
    """Apply max(0, m + g*(tp - m)) per rollout day on [N, H, W]."""
    if g <= 1.0 + 1e-9:
        return tp_batch.copy()
    m = tp_batch.mean(axis=(-2, -1), keepdims=True)
    return np.maximum(0.0, m + g * (tp_batch - m)).astype(np.float32)


def main() -> int:
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
    val_idx = splits["val"]

    ctx = load_rollout_context()
    harm_coef = ctx["harm_coef"]
    harm_k = ctx["harm_k"]
    period = ctx["period"]
    norm = ctx["norm"]
    sigma_pix = norm["sigma_pix"]

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

    pp_cfg = cfg["postproc"]
    steps = int(pp_cfg["fit_steps"])
    anom_clamp = float(cfg["stage_b"]["anom_clamp"])
    input_clip = float(cfg["norm"]["anomaly_clip"])
    rain_thresh = float(cfg["metrics"]["rain_threshold_mm"])
    tp_vr_target = float(pp_cfg["tp_var_ratio_target"])  # 0.45

    val_starts = val_idx[2 : len(val_idx) - steps].astype(np.int64)
    print(f"Generating {steps}-step rollouts on {len(val_starts)} VAL start days (target tp var ratio >= {tp_vr_target:.2f})...", flush=True)

    anom_preds = np.empty((len(val_starts), steps, 6, 32, 36), dtype=np.float32)
    ev_logits = np.empty((len(val_starts), steps, 32, 36), dtype=np.float32)
    true_phys = np.empty((len(val_starts), steps, 6, 32, 36), dtype=np.float32)
    true_anom = np.empty((len(val_starts), steps, 6, 32, 36), dtype=np.float32)

    for i, t in enumerate(val_starts):
        init_phys = data_phys[t - 2 : t + 1]
        doy_last = int(doy[t])
        a_out, ev_out = rollout(model, init_phys, doy_last, steps, delta=None, ctx=ctx, anom_clamp=anom_clamp, input_clip=input_clip)
        anom_preds[i] = a_out
        ev_logits[i] = ev_out
        tr_slice = data_phys[t + 1 : t + 1 + steps]
        true_phys[i] = tr_slice
        doy_targets = doy_last + np.arange(1, steps + 1, dtype=np.int64)
        clim_t = eval_harmonic(harm_coef, doy_targets / float(period), K=harm_k)
        true_anom[i] = (to_model_space(tr_slice) - clim_t) / sigma_pix[None]

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
    val_win_t = [w["t_last"] for w in win_by_split["val"]]
    start_map = {int(t): idx for idx, t in enumerate(val_starts)}
    win_pos = np.array([start_map[int(t)] for t in val_win_t], dtype=np.int64)

    # Best baseline tp RMSE & CSI on val windows at leads [1, 4, 8]
    best_tp_base_rmse: dict[int, float] = {}
    emp_tp_csi: dict[int, float] = {}
    for lead in (1, 4, 8):
        k = lead - 1
        t_win = true_phys[win_pos, k, CH_TP]
        pers_win = np.array([data_phys[t, CH_TP] for t in val_win_t], dtype=np.float32)
        harm_win = np.array([clim_all[t + lead, CH_TP] for t in val_win_t], dtype=np.float32)
        emp_win = np.array([evaluate_empirical(emp_table, np.array([doy[t] + lead]), period=period)[0, CH_TP] for t in val_win_t], dtype=np.float32)
        r_p = float(lat_weighted_rmse(pers_win[:, None], t_win[:, None], lat_w)[0])
        r_h = float(lat_weighted_rmse(harm_win[:, None], t_win[:, None], lat_w)[0])
        r_e = float(lat_weighted_rmse(emp_win[:, None], t_win[:, None], lat_w)[0])
        best_tp_base_rmse[lead] = min(r_p, r_h, r_e)
        emp_tp_csi[lead] = float(heavy_rain_csi_details(emp_win, t_win, threshold_mm=rain_thresh)["csi"])

    # 1. Beta[k, c] through origin on VAL
    b_lo, b_hi = float(pp_cfg["beta_clip"][0]), float(pp_cfg["beta_clip"][1])
    beta = np.zeros((steps, 6), dtype=np.float64)
    for k in range(steps):
        for c in range(6):
            p = anom_preds[:, k, c].astype(np.float64)
            y = true_anom[:, k, c].astype(np.float64)
            beta[k, c] = np.clip(float(np.sum(p * y) / max(float(np.sum(p * p)), 1e-12)), b_lo, b_hi)

    shrunk_anom = anom_preds * beta[None, :, :, None, None].astype(np.float32)
    phys_beta = np.empty_like(true_phys)
    for i, t in enumerate(val_starts):
        doy_targets = int(doy[t]) + np.arange(1, steps + 1, dtype=np.int64)
        phys_beta[i] = from_anomaly(shrunk_anom[i], doy_targets, ctx=ctx)

    # 2. kappa_k on VAL
    kappa = np.ones(steps, dtype=np.float64)
    phys_kappa = phys_beta.copy()
    for k in range(steps):
        kappa[k] = float(np.sum(true_phys[:, k, CH_TP])) / max(float(np.sum(phys_beta[:, k, CH_TP])), 1e-12)
        phys_kappa[:, k, CH_TP] = np.maximum(phys_beta[:, k, CH_TP] * kappa[k], 0.0)

    # --- Method G (Pattern Gain) ---
    g_grid = [float(x) for x in pp_cfg["pattern_gain_grid"]]
    print("\n=== Method G (Pattern Gain) Lead-8 Pareto on VAL Windows ===")
    g8_chosen = None
    pareto_g: list[tuple[float, float, float, float]] = []
    for g in g_grid:
        tp_g_all = _apply_pattern_gain_batch(phys_kappa[:, 7, CH_TP], g)
        post_k8 = float(np.sum(true_phys[:, 7, CH_TP])) / max(float(np.sum(tp_g_all)), 1e-12)
        tp_g_win = np.maximum(tp_g_all[win_pos] * post_k8, 0.0)
        true_w8 = true_phys[win_pos, 7, CH_TP]
        vr = float(day8_spatial_variance_ratio(tp_g_win[:, None], true_w8[:, None])[0])
        rm = float(lat_weighted_rmse(tp_g_win[:, None], true_w8[:, None], lat_w)[0])
        csi_val = float(heavy_rain_csi_details(tp_g_win, true_w8, threshold_mm=rain_thresh)["csi"])
        pareto_g.append((g, vr, rm, csi_val))
        print(f"  g={g:.1f}: post_kappa={post_k8:.4f}, vr={vr:.4f}, tp_rmse={rm:.4f}, csi={csi_val:.4f}")
        if g8_chosen is None and vr >= tp_vr_target:
            g8_chosen = g

    # --- Method QM (Quantile Map Blend) ---
    raw_knots_p, raw_knots_t = [], []
    for k in range(steps):
        qp, qt = _build_qm_knots(phys_kappa[:, k, CH_TP], true_phys[:, k, CH_TP])
        raw_knots_p.append(qp)
        raw_knots_t.append(qt)

    qm_q_pred, qm_q_true = [], []
    for k in range(steps):
        xp = np.array(raw_knots_p[k], dtype=np.float64)
        yt = np.array(raw_knots_t[k], dtype=np.float64)
        u_old = np.linspace(0.0, 1.0, len(xp))
        u_new = np.linspace(0.0, 1.0, 128)
        qm_q_pred.append([round(float(v), 5) for v in np.interp(u_new, u_old, xp)])
        qm_q_true.append([round(float(v), 5) for v in np.interp(u_new, u_old, yt)])

    lam_cands = [float(x) for x in pp_cfg["qm_lam_candidates"]]
    print("\n=== Method QM (Quantile Map Blend) Lead-8 Pareto on VAL Windows ===")
    pareto_qm: list[tuple[float, float, float, float]] = []
    qm_lambda = np.zeros(steps, dtype=np.float64)
    for k in range(steps):
        qp_k = np.asarray(qm_q_pred[k], dtype=np.float32)
        qt_k = np.asarray(qm_q_true[k], dtype=np.float32)
        tp_k_win = phys_kappa[win_pos, k, CH_TP]
        true_k_win = true_phys[win_pos, k, CH_TP]
        mapped_win = np.interp(tp_k_win, qp_k, qt_k).astype(np.float32)
        mapped_win = np.where(tp_k_win <= 0.0, 0.0, np.maximum(mapped_win, 0.0)).astype(np.float32)

        chosen_lam = None
        for lam in lam_cands:
            blend_win = (1.0 - lam) * tp_k_win + lam * mapped_win
            vr = float(day8_spatial_variance_ratio(blend_win[:, None], true_k_win[:, None])[0])
            rm = float(lat_weighted_rmse(blend_win[:, None], true_k_win[:, None], lat_w)[0])
            csi_val = float(heavy_rain_csi_details(blend_win, true_k_win, threshold_mm=rain_thresh)["csi"])
            if k == 7:
                pareto_qm.append((lam, vr, rm, csi_val))
                print(f"  lam={lam:.2f}: vr={vr:.4f}, tp_rmse={rm:.4f}, csi={csi_val:.4f}")
            if chosen_lam is None and vr >= tp_vr_target:
                chosen_lam = lam
        qm_lambda[k] = 1.0 if chosen_lam is None else chosen_lam

    g_reaches = g8_chosen is not None
    qm_reaches = max(p[1] for p in pareto_qm) >= tp_vr_target
    if not g_reaches and not qm_reaches:
        print(f"ALERT: Neither Method G nor Method QM reached tp var ratio >= {tp_vr_target:.2f} at lead 8. Stopping!")
        return 2

    # Build schedules for both methods
    g_sched = np.array([1.0 + ((g8_chosen or 1.0) - 1.0) * np.clip(((k + 1) - 4.0) / 4.0, 0.0, 1.0) for k in range(steps)], dtype=np.float64)
    g_post_kappa = np.ones(steps, dtype=np.float64)
    phys_g = phys_kappa.copy()
    for k in range(steps):
        tp_g_all = _apply_pattern_gain_batch(phys_kappa[:, k, CH_TP], float(g_sched[k]))
        g_post_kappa[k] = float(np.sum(true_phys[:, k, CH_TP])) / max(float(np.sum(tp_g_all)), 1e-12)
        phys_g[:, k, CH_TP] = np.maximum(tp_g_all * g_post_kappa[k], 0.0)

    phys_qm = phys_kappa.copy()
    for k in range(steps):
        qp_k = np.asarray(qm_q_pred[k], dtype=np.float32)
        qt_k = np.asarray(qm_q_true[k], dtype=np.float32)
        tp_k_all = phys_kappa[:, k, CH_TP]
        mapped_all = np.interp(tp_k_all, qp_k, qt_k).astype(np.float32)
        mapped_all = np.where(tp_k_all <= 0.0, 0.0, np.maximum(mapped_all, 0.0)).astype(np.float32)
        phys_qm[:, k, CH_TP] = (1.0 - qm_lambda[k]) * tp_k_all + qm_lambda[k] * mapped_all

    # Compare both methods on VAL windows at leads 1, 4, 8
    print("\n=== Method G vs Method QM on VAL Windows (Leads 1, 4, 8) ===")
    print(f"{'Method':<14} {'Lead':>4} {'tp_RMSE':>10} {'tp_Skill':>10} {'tp_VR':>10} {'CSI':>10}")
    metrics_cmp: dict[str, dict[int, tuple[float, float, float, float]]] = {"pattern_gain": {}, "qm": {}}
    for mname, arr in (("pattern_gain", phys_g), ("qm", phys_qm)):
        for lead in (1, 4, 8):
            k = lead - 1
            pred_w = arr[win_pos, k, CH_TP]
            true_w = true_phys[win_pos, k, CH_TP]
            rm = float(lat_weighted_rmse(pred_w[:, None], true_w[:, None], lat_w)[0])
            sk = 1.0 - rm / best_tp_base_rmse[lead]
            vr = float(day8_spatial_variance_ratio(pred_w[:, None], true_w[:, None])[0])
            csi_v = float(heavy_rain_csi_details(pred_w, true_w, threshold_mm=rain_thresh)["csi"])
            metrics_cmp[mname][lead] = (rm, sk, vr, csi_v)
            print(f"{mname:<14} {lead:>4} {rm:>10.4f} {sk:>+10.4f} {vr:>10.4f} {csi_v:>10.4f}")

    # Pick method with lower tp RMSE at lead 8 subject to vr >= 0.45 (ties: higher CSI at lead 8)
    valid_cands = []
    if g_reaches:
        valid_cands.append(("pattern_gain", metrics_cmp["pattern_gain"][8][0], -metrics_cmp["pattern_gain"][8][3]))
    if qm_reaches:
        valid_cands.append(("qm", metrics_cmp["qm"][8][0], -metrics_cmp["qm"][8][3]))
    valid_cands.sort(key=lambda x: (round(x[1], 6), x[2]))
    chosen_method = valid_cands[0][0]
    chosen_phys = phys_g if chosen_method == "pattern_gain" else phys_qm
    print(f"\nChosen tp variance method: {chosen_method} (g_8={g8_chosen}, qm_lam_8={qm_lambda[7]:.2f})")

    # Event coupling check on VAL windows
    fit_leads = [int(x) for x in pp_cfg["event_fit_leads"]]
    hi_mm = float(pp_cfg["event_high_mm"])
    lo_mm = float(pp_cfg["event_low_mm"])
    probs_all = 1.0 / (1.0 + np.exp(-np.clip(ev_logits, -40.0, 40.0)))

    fit_thrs = []
    for lead in fit_leads:
        k = lead - 1
        p_k = probs_all[:, k]
        t_k = true_phys[:, k, CH_TP]
        best_thr, best_csi = 0.5, -1.0
        for thr in np.linspace(0.10, 0.90, 81):
            csi_val = float(heavy_rain_csi_details(np.where(p_k > thr, hi_mm, lo_mm), t_k, threshold_mm=rain_thresh)["csi"])
            if csi_val > best_csi:
                best_csi = csi_val
                best_thr = float(thr)
        fit_thrs.append(best_thr)

    ks = np.arange(1, steps + 1, dtype=np.float64)
    interp_thrs = np.interp(ks, np.asarray(fit_leads, dtype=np.float64), np.asarray(fit_thrs, dtype=np.float64))

    print("\n=== Event Coupling Check After New tp Step (VAL Windows) ===")
    csi_with: dict[int, float] = {}
    csi_without: dict[int, float] = {}
    for lead in fit_leads:
        k = lead - 1
        tp_no = chosen_phys[win_pos, k, CH_TP]
        tp_yes = np.where(probs_all[win_pos, k] > interp_thrs[k], np.maximum(tp_no, hi_mm), np.minimum(tp_no, lo_mm))
        t_w = true_phys[win_pos, k, CH_TP]
        csi_no = float(heavy_rain_csi_details(tp_no, t_w, threshold_mm=rain_thresh)["csi"])
        csi_yes = float(heavy_rain_csi_details(tp_yes, t_w, threshold_mm=rain_thresh)["csi"])
        r_no = float(lat_weighted_rmse(tp_no[:, None], t_w[:, None], lat_w)[0])
        r_yes = float(lat_weighted_rmse(tp_yes[:, None], t_w[:, None], lat_w)[0])
        csi_without[lead] = csi_no
        csi_with[lead] = csi_yes
        print(
            f"  Lead {lead}: CSI without={csi_no:.4f}, with={csi_yes:.4f} "
            f"(emp_clim={emp_tp_csi[lead]:.4f}) | RMSE without={r_no:.4f}, with={r_yes:.4f}"
        )

    # Task B: keep event coupling only if CSI at leads 4 and 8 still beats empirical climatology on val (0.334 and 0.327)
    beats_emp_with = (csi_with[4] > emp_tp_csi[4]) and (csi_with[8] > emp_tp_csi[8])
    improves_csi = (csi_with[4] >= csi_without[4]) and (csi_with[8] >= csi_without[8])
    use_event_coupling = bool(beats_emp_with and improves_csi)
    print(f"Event coupling decision: use_event_coupling={use_event_coupling}")

    # Physical clamps
    phys_min = norm["phys_min"].astype(np.float64)
    phys_max = norm["phys_max"].astype(np.float64)
    rng = phys_max - phys_min
    margin = float(pp_cfg["clamp_margin_frac"])
    clamp_lo = phys_min - margin * rng
    clamp_hi = phys_max + margin * rng
    clamp_lo[CH_TP] = max(clamp_lo[CH_TP], 0.0)
    clamp_lo[0] = max(clamp_lo[0], float(pp_cfg["t2m_min_k"]))
    clamp_hi[0] = min(clamp_hi[0], float(pp_cfg["t2m_max_k"]))

    postproc_payload: dict[str, Any] = {
        "channels": channels,
        "fit_steps": steps,
        "beta": [[round(float(v), 6) for v in row] for row in beta],
        "beta_tail_tau": float(pp_cfg["beta_tail_tau"]),
        "kappa": [round(float(v), 6) for v in kappa],
        "tp_var_method": chosen_method,
        "pattern_gain": [round(float(v), 6) for v in g_sched],
        "pattern_post_kappa": [round(float(v), 6) for v in g_post_kappa],
        "qm_lambda": [round(float(v), 4) for v in qm_lambda],
        "qm_q_pred": qm_q_pred,
        "qm_q_true": qm_q_true,
        "event_fit_leads": fit_leads,
        "event_thresholds": [round(float(v), 4) for v in fit_thrs],
        "event_high_mm": hi_mm,
        "event_low_mm": lo_mm,
        "use_event_coupling": use_event_coupling,
        "clamp_lo": [round(float(v), 4) for v in clamp_lo],
        "clamp_hi": [round(float(v), 4) for v in clamp_hi],
        "t2m_min_k": float(pp_cfg["t2m_min_k"]),
        "t2m_max_k": float(pp_cfg["t2m_max_k"]),
    }

    POSTPROC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POSTPROC_PATH.open("w", encoding="utf-8") as f:
        json.dump(postproc_payload, f, indent=2)
    print(f"\nSaved Phase 3b postproc parameters -> {POSTPROC_PATH} ({POSTPROC_PATH.stat().st_size} bytes)")

    out_check = apply_postproc(anom_preds[win_pos[0]], ev_logits[win_pos[0]], int(doy[val_starts[win_pos[0]]]), steps, postproc_payload, ctx)
    assert np.isfinite(out_check).all()
    assert float(out_check[:, CH_TP].min()) >= 0.0
    assert float(out_check[:, 0].min()) >= 233.15 and float(out_check[:, 0].max()) <= 333.0
    return 0


if __name__ == "__main__":
    sys.exit(main())
