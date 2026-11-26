"""Evaluate Stage A 1-step predictions on all val days, all test days, and val windows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monsoon.climatology import evaluate as eval_harmonic, evaluate_empirical, fit_empirical, load_harmonics
from monsoon.data import CH_TP, OneStepDataset, from_model_space, load_dataset, load_norm, split_indices
from monsoon.metrics import heavy_rain_csi_details, lat_weighted_rmse, lat_weights, skill_score
from monsoon.model import MonsoonNet
from monsoon.windows import select_heldout_windows


CONFIG_PATH = ROOT / "config" / "default.json"
HARMONICS_PATH = ROOT / "weights" / "clim_harmonics.npz"
NORM_PATH = ROOT / "weights" / "norm.npz"
MODEL_PATH = ROOT / "weights" / "model.pt"


@torch.no_grad()
def _predict_dataset(
    model: MonsoonNet,
    ds: OneStepDataset,
    harm_coef: np.ndarray,
    sigma_pix: np.ndarray,
    harm_k: int,
    period: int,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds_a = []
    preds_ev = []
    for feat, a_t, _, _, _ in loader:
        a_pred, ev_logit = model(feat, a_t)
        preds_a.append(a_pred.numpy())
        preds_ev.append(torch.sigmoid(ev_logit[:, 0]).numpy())

    pred_a = np.concatenate(preds_a, axis=0)
    pred_prob = np.concatenate(preds_ev, axis=0)

    t_idx = ds.t_indices
    doy_target = ds.doy[t_idx] + 1
    clim_target = eval_harmonic(harm_coef, doy_target / float(period), K=harm_k)
    pred_phys = from_model_space(clim_target + sigma_pix[None] * pred_a)
    return pred_phys, pred_prob


def _best_csi_threshold(probs: np.ndarray, tp_true: np.ndarray, rain_thresh: float) -> tuple[float, float]:
    best_thr = 0.5
    best_csi = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        csi = heavy_rain_csi_details(
            np.where(probs >= thr, rain_thresh + 1.0, 0.0),
            tp_true,
            threshold_mm=rain_thresh,
        )["csi"]
        if float(csi) > best_csi:
            best_csi = float(csi)
            best_thr = float(thr)
    return best_thr, best_csi


def _report_split(
    name: str,
    t_idx: np.ndarray,
    data_phys: np.ndarray,
    doy: np.ndarray,
    pred_phys: np.ndarray,
    init_phys: np.ndarray,
    emp_table: np.ndarray,
    lat_w: np.ndarray,
    channels: list[str],
    kappa_train: float,
    period: int,
) -> None:
    true_phys = data_phys[t_idx + 1]
    pers_phys = data_phys[t_idx]
    clim_emp_phys = evaluate_empirical(emp_table, doy[t_idx] + 1, period=period)

    pred_kappa = pred_phys.copy()
    pred_kappa[:, CH_TP] = np.maximum(pred_phys[:, CH_TP] * kappa_train, 0.0)

    rmse_pers = lat_weighted_rmse(pers_phys, true_phys, lat_w)
    rmse_clim = lat_weighted_rmse(clim_emp_phys, true_phys, lat_w)
    rmse_init = lat_weighted_rmse(init_phys, true_phys, lat_w)
    rmse_raw = lat_weighted_rmse(pred_phys, true_phys, lat_w)
    rmse_kap = lat_weighted_rmse(pred_kappa, true_phys, lat_w)

    ss_raw, ss_raw_mean = skill_score(rmse_raw, rmse_pers, rmse_clim)
    ss_kap, ss_kap_mean = skill_score(rmse_kap, rmse_pers, rmse_clim)
    imp_pct = (1.0 - rmse_kap / rmse_pers) * 100.0

    print(f"\n=== Lead-1 Evaluation: {name} (N={len(t_idx)}) ===")
    header = f"{'Row':<22} " + " ".join(f"{ch:>10}" for ch in channels)
    print(header)
    print("-" * len(header))
    for label, arr in (
        ("persistence", rmse_pers),
        ("clim_empirical", rmse_clim),
        ("init_damped_pers", rmse_init),
        ("model (raw)", rmse_raw),
        ("model, tp with kappa", rmse_kap),
    ):
        print(f"{label:<22} " + " ".join(f"{v:10.4f}" for v in arr))
    print(f"{'RMSE impr vs pers (%)':<22} " + " ".join(f"{v:+9.2f}%" for v in imp_pct))
    print(f"{'Skill (raw)':<22} " + " ".join(f"{v:+10.4f}" for v in ss_raw) + f" | mean={ss_raw_mean:+.4f}")
    print(f"{'Skill (with kappa)':<22} " + " ".join(f"{v:+10.4f}" for v in ss_kap) + f" | mean={ss_kap_mean:+.4f}")


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ds = load_dataset()
    data_phys = ds["data"]
    static = ds["static"]
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
    harm_k = int(cfg["climatology"]["harmonics_k"])
    period = int(cfg["climatology"]["doy_period"])
    smooth_win = int(cfg["climatology"]["empirical_smooth_window"])
    anom_clip = float(cfg["norm"]["anomaly_clip"])
    rain_thresh = float(cfg["metrics"]["rain_threshold_mm"])

    harm_coef, _, _ = load_harmonics(HARMONICS_PATH)
    norm = load_norm(NORM_PATH)
    emp_table = fit_empirical(data_phys[splits["train"]], doy[splits["train"]], smooth_window=smooth_win, period=period)

    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    trained_model = MonsoonNet(
        cin=int(mc["cin"]),
        cout=int(mc["cout"]),
        channels=int(mc["channels"]),
        n_blocks=int(mc["n_blocks"]),
        expansion=int(mc["expansion"]),
    )
    trained_model.load_state_dict(ckpt["state_dict"])
    trained_model.eval()

    init_model = MonsoonNet(
        cin=int(mc["cin"]),
        cout=int(mc["cout"]),
        channels=int(mc["channels"]),
        n_blocks=int(mc["n_blocks"]),
        expansion=int(mc["expansion"]),
    )
    init_model.eval()

    print(f"Loaded checkpoint: {MODEL_PATH} (best_epoch={ckpt['best_epoch']}, params={ckpt['param_count']})")
    rho_vals = torch.sigmoid(trained_model.rho_logit).detach().numpy()
    scale_vals = trained_model.scale.detach().numpy()
    print("Learned rho_c = sigmoid(rho_logit): " + ", ".join(f"{ch}={rho_vals[i]:.4f}" for i, ch in enumerate(channels)))
    print("Learned scale_c:                    " + ", ".join(f"{ch}={scale_vals[i]:.4f}" for i, ch in enumerate(channels)))

    train_ds = OneStepDataset(data_phys, doy, splits["train"], harm_coef, norm, static, harm_k, period, anom_clip, rain_thresh)
    val_ds = OneStepDataset(data_phys, doy, splits["val"], harm_coef, norm, static, harm_k, period, anom_clip, rain_thresh)
    test_ds = OneStepDataset(data_phys, doy, splits["test"], harm_coef, norm, static, harm_k, period, anom_clip, rain_thresh)

    sigma_pix = norm["sigma_pix"]
    tr_pred, tr_prob = _predict_dataset(trained_model, train_ds, harm_coef, sigma_pix, harm_k, period)
    va_pred, va_prob = _predict_dataset(trained_model, val_ds, harm_coef, sigma_pix, harm_k, period)
    te_pred, te_prob = _predict_dataset(trained_model, test_ds, harm_coef, sigma_pix, harm_k, period)

    _, _ = tr_pred, te_pred
    va_init, _ = _predict_dataset(init_model, val_ds, harm_coef, sigma_pix, harm_k, period)
    te_init, _ = _predict_dataset(init_model, test_ds, harm_coef, sigma_pix, harm_k, period)

    tr_true_tp = data_phys[train_ds.t_indices + 1, CH_TP]
    va_true_tp = data_phys[val_ds.t_indices + 1, CH_TP]
    te_true_tp = data_phys[test_ds.t_indices + 1, CH_TP]

    kappa_train = float(tr_true_tp.sum() / max(float(tr_pred[:, CH_TP].sum()), 1e-12))
    kappa_val = float(va_true_tp.sum() / max(float(va_pred[:, CH_TP].sum()), 1e-12))
    print(f"\ntp bias correction kappa (train-fitted): {kappa_train:.4f}")
    print(f"tp bias correction kappa (val-fitted, ref): {kappa_val:.4f}")

    _report_split("ALL VAL DAYS", val_ds.t_indices, data_phys, doy, va_pred, va_init, emp_table, lat_w, channels, kappa_train, period)
    _report_split("ALL TEST DAYS", test_ds.t_indices, data_phys, doy, te_pred, te_init, emp_table, lat_w, channels, kappa_train, period)

    # Val windows from monsoon/windows.py
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
    val_win_t = np.array([w["t_last"] for w in win_by_split["val"]], dtype=np.int64)
    val_map = {int(t): idx for idx, t in enumerate(val_ds.t_indices)}
    win_pos = np.array([val_map[int(t)] for t in val_win_t], dtype=np.int64)

    _report_split(
        "VAL WINDOWS (50 non-overlapping)",
        val_ds.t_indices[win_pos],
        data_phys,
        doy,
        va_pred[win_pos],
        va_init[win_pos],
        emp_table,
        lat_w,
        channels,
        kappa_train,
        period,
    )

    # Event head CSI evaluation
    thr_val, csi_val_on_val = _best_csi_threshold(va_prob, va_true_tp, rain_thresh)
    thr_train, csi_tr_on_tr = _best_csi_threshold(tr_prob, tr_true_tp, rain_thresh)

    def _csi_at(probs: np.ndarray, tp_true: np.ndarray, thr: float) -> float:
        return float(
            heavy_rain_csi_details(
                np.where(probs >= thr, rain_thresh + 1.0, 0.0),
                tp_true,
                threshold_mm=rain_thresh,
            )["csi"]
        )

    pers_val_csi = heavy_rain_csi_details(data_phys[val_ds.t_indices, CH_TP], va_true_tp, rain_thresh)["csi"]
    pers_test_csi = heavy_rain_csi_details(data_phys[test_ds.t_indices, CH_TP], te_true_tp, rain_thresh)["csi"]
    pers_win_csi = heavy_rain_csi_details(
        data_phys[val_ds.t_indices[win_pos], CH_TP], data_phys[val_ds.t_indices[win_pos] + 1, CH_TP], rain_thresh
    )["csi"]

    print("\n=== Event Head CSI (>= 10.0 mm/day) at Lead 1 ===")
    print(f"VAL-fitted threshold   : {thr_val:.3f} -> CSI: train={_csi_at(tr_prob, tr_true_tp, thr_val):.4f}, "
          f"val={csi_val_on_val:.4f}, test={_csi_at(te_prob, te_true_tp, thr_val):.4f}, "
          f"val_windows={_csi_at(va_prob[win_pos], data_phys[val_ds.t_indices[win_pos] + 1, CH_TP], thr_val):.4f}")
    print(f"TRAIN-fitted threshold : {thr_train:.3f} -> CSI: train={csi_tr_on_tr:.4f}, "
          f"val={_csi_at(va_prob, va_true_tp, thr_train):.4f}, test={_csi_at(te_prob, te_true_tp, thr_train):.4f}, "
          f"val_windows={_csi_at(va_prob[win_pos], data_phys[val_ds.t_indices[win_pos] + 1, CH_TP], thr_train):.4f}")
    print(f"Persistence CSI        : val_all={pers_val_csi:.4f}, test_all={pers_test_csi:.4f}, "
          f"val_windows={pers_win_csi:.4f} (docs/baselines.md: val=0.4256, test=0.3939)")


if __name__ == "__main__":
    main()
