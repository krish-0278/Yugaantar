"""Training pipeline for MonsoonNet: Stage A (1-step) and Stage B (BPTT rollout curriculum)."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import ctypes
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from monsoon.climatology import (
    evaluate as eval_harmonic,
    evaluate_empirical,
    fit as fit_harmonic,
    fit_empirical,
    load_harmonics,
    save_harmonics,
)
from monsoon.data import (
    MultiStepDataset,
    OneStepDataset,
    build_static_norm,
    compute_norm_stats,
    from_model_space,
    load_dataset,
    load_norm,
    save_norm,
    split_indices,
    to_model_space,
)
from monsoon.losses import StageALoss
from monsoon.metrics import lat_weighted_rmse, lat_weights, skill_score
from monsoon.model import MonsoonNet
from monsoon.rollout import build_clim_norm_table, from_anomaly, load_rollout_context, rollout, rollout_tensor
from monsoon.windows import select_heldout_windows


ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "config" / "default.json"
HARMONICS_PATH = ROOT / "weights" / "clim_harmonics.npz"
NORM_PATH = ROOT / "weights" / "norm.npz"
STATIC_PATH = ROOT / "weights" / "static.npz"
MODEL_PATH = ROOT / "weights" / "model.pt"
MODEL_STAGEA_PATH = ROOT / "weights" / "model_stageA.pt"
STAGEB_RESUME_PATH = ROOT / "runs" / "stageB_last.pt"
LOG_STAGEA_PATH = ROOT / "docs" / "train_log_stageA.csv"
LOG_STAGEB_PATH = ROOT / "docs" / "train_log_stageB.csv"


@contextmanager
def prevent_windows_sleep():
    """Keep Windows awake while training runs without modifying global power settings."""
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    enabled = False
    if os.name == "nt" and hasattr(ctypes, "windll"):
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
            enabled = True
            print("Windows sleep inhibited for training duration.", flush=True)
        except Exception:
            enabled = False
    try:
        yield
    finally:
        if enabled:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)
            except Exception:
                pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_dynamic_noise(feat: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return feat
    out = feat.clone()
    out[:, :24] = out[:, :24] + sigma * torch.randn_like(out[:, :24])
    return out


def _update_ema(ema_model: MonsoonNet, model: MonsoonNet, decay: float) -> None:
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


def _build_model(cfg: dict[str, Any], device: torch.device) -> MonsoonNet:
    mc = cfg["model"]
    return MonsoonNet(
        cin=int(mc["cin"]),
        cout=int(mc["cout"]),
        channels=int(mc["channels"]),
        n_blocks=int(mc["n_blocks"]),
        expansion=int(mc["expansion"]),
    ).to(device)


def run_overfit_test(
    train_ds: OneStepDataset,
    cfg: dict[str, Any],
    lat_w_t: torch.Tensor,
    pos_weight: float,
    device: torch.device,
) -> tuple[float, float]:
    """Overfit 64 training samples with input noise=0 and EMA=off; check >10x loss drop."""
    sa = cfg["stage_a"]
    n_sub = int(sa["overfit_samples"])
    n_steps = int(sa["overfit_steps"])
    step_idx = np.linspace(0, len(train_ds) - 1, n_sub, dtype=np.int64)
    sub_ds = Subset(train_ds, step_idx.tolist())
    loader = DataLoader(
        sub_ds,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(int(cfg["seed"])),
    )

    set_seed(int(cfg["seed"]))
    model = _build_model(cfg, device)
    criterion = StageALoss(
        lat_w=lat_w_t,
        pos_weight=pos_weight,
        huber_delta=float(sa["huber_delta"]),
        rain_threshold_mm=float(cfg["metrics"]["rain_threshold_mm"]),
        tp_tail_weight=float(sa["tp_tail_weight"]),
        bce_weight=float(sa["bce_weight"]),
        grad_weight=float(sa["grad_loss_weight"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    first_loss = 0.0
    last_loss = 0.0
    step = 0
    model.train()
    while step < n_steps:
        for feat, a_t, a_target, ev_target, tp_phys in loader:
            step += 1
            feat = feat.to(device)
            a_t = a_t.to(device)
            a_target = a_target.to(device)
            ev_target = ev_target.to(device)
            tp_phys = tp_phys.to(device)

            opt.zero_grad(set_to_none=True)
            a_pred, ev_logit = model(feat, a_t)
            loss, _ = criterion(a_pred, ev_logit, a_target, ev_target, tp_phys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(sa["grad_clip"]))
            opt.step()
            sched.step()
            val = float(loss.detach().item())
            if step == 1:
                first_loss = val
            last_loss = val
            if step >= n_steps:
                break

    model.eval()
    with torch.no_grad():
        full_loader = DataLoader(sub_ds, batch_size=n_sub, shuffle=False, num_workers=0)
        f_feat, f_at, f_tar, f_ev, f_tp = next(iter(full_loader))
        ap, el = model(f_feat.to(device), f_at.to(device))
        full_end_loss, _ = criterion(ap, el, f_tar.to(device), f_ev.to(device), f_tp.to(device))
        full_end = float(full_end_loss.item())

    ratio_step = first_loss / max(last_loss, 1e-12)
    ratio_full = first_loss / max(full_end, 1e-12)
    status = "PASS" if max(ratio_step, ratio_full) > 10.0 else "BELOW_10X"
    print(
        f"Overfit test ({n_sub} samples, {n_steps} steps, noise=0, EMA=off): "
        f"step1_loss={first_loss:.6f} -> step{n_steps}_loss={last_loss:.6f} (ratio={ratio_step:.2f}x), "
        f"full64_end_loss={full_end:.6f} (ratio={ratio_full:.2f}x) [{status}]",
        flush=True,
    )
    return first_loss, last_loss


@torch.no_grad()
def evaluate_val_stage_a(
    model: MonsoonNet,
    val_ds: OneStepDataset,
    val_loader: DataLoader,
    criterion: StageALoss,
    harm_coef: np.ndarray,
    sigma_pix: np.ndarray,
    lat_w: np.ndarray,
    harm_k: int,
    period: int,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    preds_anom = []
    for feat, a_t, a_target, ev_target, tp_phys in val_loader:
        feat = feat.to(device)
        a_t = a_t.to(device)
        a_target = a_target.to(device)
        ev_target = ev_target.to(device)
        tp_phys = tp_phys.to(device)
        a_pred, ev_logit = model(feat, a_t)
        loss, _ = criterion(a_pred, ev_logit, a_target, ev_target, tp_phys)
        bs = feat.shape[0]
        total_loss += float(loss.item()) * bs
        n_samples += bs
        preds_anom.append(a_pred.cpu().numpy())

    pred_a = np.concatenate(preds_anom, axis=0)
    t_idx = val_ds.t_indices
    doy_target = val_ds.doy[t_idx] + 1
    clim_target = eval_harmonic(harm_coef, doy_target / float(period), K=harm_k)
    pred_phys = from_model_space(clim_target + sigma_pix[None] * pred_a)
    true_phys = val_ds.data_phys[t_idx + 1]
    pers_phys = val_ds.data_phys[t_idx]

    rmse_model = lat_weighted_rmse(pred_phys, true_phys, lat_w)
    rmse_pers = lat_weighted_rmse(pers_phys, true_phys, lat_w)
    return total_loss / max(n_samples, 1), rmse_model, rmse_pers


def train_stage_a(
    cfg: dict[str, Any],
    ds: dict[str, Any],
    splits: dict[str, np.ndarray],
    harm_coef: np.ndarray,
    norm: dict[str, np.ndarray],
    lat_w: np.ndarray,
    pos_weight: float,
    device: torch.device,
    skip_overfit: bool,
) -> None:
    seed = int(cfg["seed"])
    data_phys = ds["data"]
    static = ds["static"]
    doy = ds["doy"]
    channels = ds["channels"]

    harm_k = int(cfg["climatology"]["harmonics_k"])
    period = int(cfg["climatology"]["doy_period"])
    anom_clip = float(cfg["norm"]["anomaly_clip"])
    rain_thresh = float(cfg["metrics"]["rain_threshold_mm"])
    lat_w_t = torch.from_numpy(lat_w.astype(np.float32))

    train_ds = OneStepDataset(
        data_phys, doy, splits["train"], harm_coef, norm, static, harm_k, period, anom_clip, rain_thresh
    )
    val_ds = OneStepDataset(
        data_phys, doy, splits["val"], harm_coef, norm, static, harm_k, period, anom_clip, rain_thresh
    )
    print(f"OneStepDataset sizes: train={len(train_ds)}, val={len(val_ds)}")

    if not skip_overfit:
        run_overfit_test(train_ds, cfg, lat_w_t, pos_weight, device)

    sa = cfg["stage_a"]
    set_seed(seed)
    model = _build_model(cfg, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"MonsoonNet C={cfg['model']['channels']}, blocks={cfg['model']['n_blocks']}, radius={model.radius}, params={param_count}")

    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    criterion = StageALoss(
        lat_w=lat_w_t,
        pos_weight=pos_weight,
        huber_delta=float(sa["huber_delta"]),
        rain_threshold_mm=rain_thresh,
        tp_tail_weight=float(sa["tp_tail_weight"]),
        bce_weight=float(sa["bce_weight"]),
        grad_weight=float(sa["grad_loss_weight"]),
    ).to(device)

    epochs = int(sa["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(sa["lr"]), weight_decay=float(sa["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    dl_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(sa["batch_size"]),
        shuffle=True,
        num_workers=int(sa["num_workers"]),
        generator=dl_gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(sa["batch_size"]),
        shuffle=False,
        num_workers=int(sa["num_workers"]),
    )

    LOG_STAGEA_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ["epoch", "train_loss", "val_loss", "val_skill_mean", "secs", *[f"val_rmse_{ch}" for ch in channels]]

    best_skill = -1e9
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    noise_sigma = float(sa["input_noise_sigma"])
    ema_decay = float(sa["ema_decay"])
    grad_clip = float(sa["grad_clip"])

    with LOG_STAGEA_PATH.open("w", newline="", encoding="utf-8") as log_f:
        writer = csv.writer(log_f)
        writer.writerow(header)

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            model.train()
            tr_loss_sum = 0.0
            tr_count = 0
            for feat, a_t, a_target, ev_target, tp_phys in train_loader:
                feat = _add_dynamic_noise(feat, noise_sigma).to(device)
                a_t = a_t.to(device)
                a_target = a_target.to(device)
                ev_target = ev_target.to(device)
                tp_phys = tp_phys.to(device)

                opt.zero_grad(set_to_none=True)
                a_pred, ev_logit = model(feat, a_t)
                loss, _ = criterion(a_pred, ev_logit, a_target, ev_target, tp_phys)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                _update_ema(ema_model, model, ema_decay)

                bs = feat.shape[0]
                tr_loss_sum += float(loss.detach().item()) * bs
                tr_count += bs

            sched.step()
            train_loss = tr_loss_sum / max(tr_count, 1)
            val_loss, val_rmse, val_pers_rmse = evaluate_val_stage_a(
                ema_model, val_ds, val_loader, criterion, harm_coef, norm["sigma_pix"], lat_w, harm_k, period, device
            )
            val_skill = float(np.mean(1.0 - val_rmse / val_pers_rmse))
            secs = time.perf_counter() - t0

            if val_skill > best_skill:
                best_skill = val_skill
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}

            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_skill:.6f}", f"{secs:.2f}", *[f"{v:.4f}" for v in val_rmse]])
            log_f.flush()
            rmse_str = " ".join(f"{ch}={val_rmse[i]:.3f}" for i, ch in enumerate(channels))
            print(
                f"Epoch {epoch:02d}/{epochs} | {secs:5.2f}s | train={train_loss:.4f} | "
                f"val={val_loss:.4f} | skill_vs_pers={val_skill:+.4f} | {rmse_str}",
                flush=True,
            )

    assert best_state is not None
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "config": cfg,
            "stage": "A",
            "best_epoch": best_epoch,
            "best_val_skill_mean": best_skill,
            "param_count": param_count,
        },
        MODEL_PATH,
    )
    print(f"\nSaved best Stage A EMA checkpoint (epoch {best_epoch}, val_skill_vs_pers={best_skill:+.4f}) -> {MODEL_PATH}")


@torch.no_grad()
def evaluate_val_stage_b(
    model: MonsoonNet,
    val_ds: MultiStepDataset,
    val_loader: DataLoader,
    criterion: StageALoss,
    clim_norm_table_t: torch.Tensor,
    static_norm_t: torch.Tensor,
    harm_coef: np.ndarray,
    sigma_pix: np.ndarray,
    lat_w: np.ndarray,
    K: int,
    harm_k: int,
    period: int,
    anom_clamp: float,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    step_k_preds = []
    for a_hist, doy_last, a_targets, ev_targets, tp_targets in val_loader:
        a_hist = a_hist.to(device)
        doy_last_t = doy_last.to(device)
        a_targets = a_targets.to(device)
        ev_targets = ev_targets.to(device)
        tp_targets = tp_targets.to(device)

        a_preds, ev_logits = rollout_tensor(
            model, a_hist, doy_last_t, K, clim_norm_table_t, static_norm_t, period=period, anom_clamp=anom_clamp
        )
        loss = torch.tensor(0.0, device=device)
        for k in range(K):
            step_loss, _ = criterion(
                a_preds[:, k],
                ev_logits[:, k : k + 1],
                a_targets[:, k],
                ev_targets[:, k : k + 1],
                tp_targets[:, k],
            )
            loss = loss + step_loss
        bs = a_hist.shape[0]
        total_loss += float(loss.item()) * bs
        n_samples += bs
        step_k_preds.append(a_preds[:, K - 1].cpu().numpy())

    pred_a_k = np.concatenate(step_k_preds, axis=0)
    t_idx = val_ds.t_indices
    doy_k = val_ds.doy[t_idx] + K
    clim_k = eval_harmonic(harm_coef, doy_k / float(period), K=harm_k)
    pred_phys_k = from_model_space(clim_k + sigma_pix[None] * pred_a_k)
    true_phys_k = val_ds.data_phys[t_idx + K]
    rmse_k = lat_weighted_rmse(pred_phys_k, true_phys_k, lat_w)
    return total_loss / max(n_samples, 1), rmse_k


def evaluate_raw_val_windows(
    model: MonsoonNet,
    cfg: dict[str, Any],
    ds: dict[str, Any],
    splits: dict[str, np.ndarray],
    harm_coef: np.ndarray,
    lat_w: np.ndarray,
) -> bool:
    """Print raw Stage B skill vs best baseline at leads 1, 4, 8 on val windows. Returns True if OK."""
    ctx = load_rollout_context()
    data_phys = ds["data"]
    doy = ds["doy"]
    channels = ds["channels"]
    harm_k = int(cfg["climatology"]["harmonics_k"])
    period = int(cfg["climatology"]["doy_period"])
    smooth_win = int(cfg["climatology"]["empirical_smooth_window"])
    anom_clamp = float(cfg["stage_b"]["anom_clamp"])
    input_clip = float(cfg["norm"]["anomaly_clip"])
    lead_days = [int(x) for x in cfg["metrics"]["lead_days"]]
    max_lead = max(lead_days)

    clim_all = from_model_space(eval_harmonic(harm_coef, doy / float(period), K=harm_k))
    emp_table = fit_empirical(data_phys[splits["train"]], doy[splits["train"]], smooth_window=smooth_win, period=period)
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
    val_wins = win_by_split["val"]
    t_arr = np.array([w["t_last"] for w in val_wins], dtype=np.int64)

    raw_preds = []
    true_roll = []
    pers_roll = []
    harm_roll = []
    emp_roll = []
    for t in t_arr:
        init_phys = data_phys[t - 2 : t + 1]
        doy_last = int(doy[t])
        anom_out, _ = rollout(model, init_phys, doy_last, max_lead, delta=None, ctx=ctx, anom_clamp=anom_clamp, input_clip=input_clip)
        doy_targets = doy_last + np.arange(1, max_lead + 1, dtype=np.int64)
        phys_out = from_anomaly(anom_out, doy_targets, ctx=ctx)
        raw_preds.append(phys_out)
        true_roll.append(data_phys[t + 1 : t + 1 + max_lead])
        pers_roll.append(np.repeat(data_phys[t][None], max_lead, axis=0))
        harm_roll.append(from_model_space(eval_harmonic(harm_coef, doy_targets / float(period), K=harm_k)))
        emp_roll.append(evaluate_empirical(emp_table, doy_targets, period=period))

    raw_arr = np.stack(raw_preds, axis=0)
    true_arr = np.stack(true_roll, axis=0)
    pers_arr = np.stack(pers_roll, axis=0)
    harm_arr = np.stack(harm_roll, axis=0)
    emp_arr = np.stack(emp_roll, axis=0)

    print("\n=== Raw Stage B Skill vs Best Baseline on VAL Windows (no post-processing) ===")
    min_lead8_skill = 1e9
    for lead in lead_days:
        idx = lead - 1
        r_mod = lat_weighted_rmse(raw_arr[:, idx], true_arr[:, idx], lat_w)
        r_per = lat_weighted_rmse(pers_arr[:, idx], true_arr[:, idx], lat_w)
        r_har = lat_weighted_rmse(harm_arr[:, idx], true_arr[:, idx], lat_w)
        r_emp = lat_weighted_rmse(emp_arr[:, idx], true_arr[:, idx], lat_w)
        r_best = np.minimum(r_per, np.minimum(r_har, r_emp))
        ss = 1.0 - r_mod / r_best
        ss_str = " ".join(f"{ch}={ss[i]:+.4f}" for i, ch in enumerate(channels))
        print(f"Lead {lead}: mean={float(np.mean(ss)):+.4f} | {ss_str}", flush=True)
        if lead == 8:
            min_lead8_skill = float(np.min(ss))

    min_allowed = float(cfg["stage_b"]["raw_lead8_min_skill"])
    if min_lead8_skill < min_allowed:
        print(f"ALERT: raw Stage B lead-8 min channel skill {min_lead8_skill:+.4f} < {min_allowed:+.4f}. Stopping before tuning!", flush=True)
        return False
    return True


def train_stage_b(
    cfg: dict[str, Any],
    ds: dict[str, Any],
    splits: dict[str, np.ndarray],
    harm_coef: np.ndarray,
    norm: dict[str, np.ndarray],
    lat_w: np.ndarray,
    pos_weight: float,
    device: torch.device,
    resume: bool,
) -> int:
    seed = int(cfg["seed"])
    data_phys = ds["data"]
    static = ds["static"]
    doy = ds["doy"]
    channels = ds["channels"]

    harm_k = int(cfg["climatology"]["harmonics_k"])
    period = int(cfg["climatology"]["doy_period"])
    anom_clip = float(cfg["norm"]["anomaly_clip"])
    rain_thresh = float(cfg["metrics"]["rain_threshold_mm"])
    sb = cfg["stage_b"]
    sa = cfg["stage_a"]
    anom_clamp = float(sb["anom_clamp"])
    noise_sigma = float(sb["input_noise_sigma"])
    ema_decay = float(sb["ema_decay"])
    grad_clip = float(sb["grad_clip"])
    subsample_frac = float(sb["subsample_frac"])
    max_k8_secs = float(sb["max_k8_epoch_secs"])
    curriculum = sb["curriculum"]

    clim_norm_table_t = torch.from_numpy(build_clim_norm_table(harm_coef, norm, harm_k=harm_k, period=period)).to(device)
    static_norm_t = torch.from_numpy(build_static_norm(static, norm)).to(device)
    lat_w_t = torch.from_numpy(lat_w.astype(np.float32))

    criterion = StageALoss(
        lat_w=lat_w_t,
        pos_weight=pos_weight,
        huber_delta=float(sa["huber_delta"]),
        rain_threshold_mm=rain_thresh,
        tp_tail_weight=float(sa["tp_tail_weight"]),
        bce_weight=float(sa["bce_weight"]),
        grad_weight=float(sa["grad_loss_weight"]),
    ).to(device)

    model = _build_model(cfg, device)
    ema_model = _build_model(cfg, device)
    param_count = sum(p.numel() for p in model.parameters())

    start_stage_idx = 0
    start_epoch = 1
    resume_ckpt: dict[str, Any] | None = None

    if resume and STAGEB_RESUME_PATH.exists():
        resume_ckpt = torch.load(STAGEB_RESUME_PATH, map_location=device, weights_only=False)
        saved_stage = int(resume_ckpt["stage_idx"])
        saved_epoch = int(resume_ckpt["epoch"])
        saved_stage_epochs = int(curriculum[saved_stage]["epochs"])
        if saved_epoch < saved_stage_epochs:
            start_stage_idx = saved_stage
            start_epoch = saved_epoch + 1
            model.load_state_dict(resume_ckpt["model_state"])
            ema_model.load_state_dict(resume_ckpt["ema_state"])
            print(f"Resuming Stage B from {STAGEB_RESUME_PATH} at stage_idx={start_stage_idx} (K={curriculum[start_stage_idx]['K']}), epoch={start_epoch}", flush=True)
        else:
            start_stage_idx = saved_stage + 1
            start_epoch = 1
            best_prev = resume_ckpt["best_stage_ema_state"]
            model.load_state_dict(best_prev)
            ema_model.load_state_dict(best_prev)
            print(f"Resuming Stage B at next stage_idx={start_stage_idx} from best EMA of K={resume_ckpt['K']}", flush=True)
    else:
        init_path = MODEL_STAGEA_PATH if MODEL_STAGEA_PATH.exists() else MODEL_PATH
        ckpt = torch.load(init_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        ema_model.load_state_dict(ckpt["state_dict"])
        print(f"Initialized Stage B from {init_path} (params={param_count})", flush=True)

    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    STAGEB_RESUME_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_STAGEB_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = ["K", "epoch", "train_loss", "val_loss", "secs", *[f"val_rmse_stepK_{ch}" for ch in channels]]
    log_mode = "a" if (resume and resume_ckpt is not None and LOG_STAGEB_PATH.exists()) else "w"

    final_best_loss = float("inf")
    final_best_state: dict[str, torch.Tensor] | None = None

    with LOG_STAGEB_PATH.open(log_mode, newline="", encoding="utf-8") as log_f:
        writer = csv.writer(log_f)
        if log_mode == "w":
            writer.writerow(header)

        for stage_idx in range(start_stage_idx, len(curriculum)):
            K = int(curriculum[stage_idx]["K"])
            stage_epochs = int(curriculum[stage_idx]["epochs"])
            train_ms = MultiStepDataset(data_phys, doy, splits["train"], harm_coef, norm, K, harm_k, period, anom_clip, rain_thresh)
            val_ms = MultiStepDataset(data_phys, doy, splits["val"], harm_coef, norm, K, harm_k, period, anom_clip, rain_thresh)
            val_loader = DataLoader(val_ms, batch_size=int(sb["batch_size"]), shuffle=False, num_workers=int(sb["num_workers"]))

            opt = torch.optim.AdamW(model.parameters(), lr=float(sb["lr"]), weight_decay=float(sb["weight_decay"]))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stage_epochs)

            ep0 = start_epoch if (resume_ckpt is not None and stage_idx == start_stage_idx and start_epoch > 1) else 1
            if ep0 > 1 and resume_ckpt is not None:
                opt.load_state_dict(resume_ckpt["opt_state"])
                sched.load_state_dict(resume_ckpt["sched_state"])
                best_stage_loss = float(resume_ckpt["best_stage_loss"])
                best_stage_state = {k: v.clone() for k, v in resume_ckpt["best_stage_ema_state"].items()}
            else:
                best_stage_loss = float("inf")
                best_stage_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}

            print(f"\n--- Stage B Curriculum K={K} ({stage_epochs} epochs, train_total={len(train_ms)}, subsample={subsample_frac:.0%}, val={len(val_ms)}) ---", flush=True)

            for epoch in range(ep0, stage_epochs + 1):
                t0 = time.perf_counter()
                rng = np.random.default_rng(seed + stage_idx * 1000 + epoch)
                n_sub = max(1, int(round(len(train_ms) * subsample_frac)))
                chosen = np.sort(rng.choice(len(train_ms), size=n_sub, replace=False))
                sub_ds = Subset(train_ms, chosen.tolist())
                dl_gen = torch.Generator().manual_seed(seed + stage_idx * 1000 + epoch)
                train_loader = DataLoader(
                    sub_ds,
                    batch_size=int(sb["batch_size"]),
                    shuffle=True,
                    num_workers=int(sb["num_workers"]),
                    generator=dl_gen,
                )

                model.train()
                tr_loss_sum = 0.0
                tr_count = 0
                for a_hist, doy_last, a_targets, ev_targets, tp_targets in train_loader:
                    if noise_sigma > 0.0:
                        a_hist = a_hist + noise_sigma * torch.randn_like(a_hist)
                    a_hist = a_hist.to(device)
                    doy_last_t = doy_last.to(device)
                    a_targets = a_targets.to(device)
                    ev_targets = ev_targets.to(device)
                    tp_targets = tp_targets.to(device)

                    opt.zero_grad(set_to_none=True)
                    a_preds, ev_logits = rollout_tensor(
                        model, a_hist, doy_last_t, K, clim_norm_table_t, static_norm_t, period=period, anom_clamp=anom_clamp
                    )
                    loss = torch.tensor(0.0, device=device)
                    for k in range(K):
                        step_loss, _ = criterion(
                            a_preds[:, k],
                            ev_logits[:, k : k + 1],
                            a_targets[:, k],
                            ev_targets[:, k : k + 1],
                            tp_targets[:, k],
                        )
                        loss = loss + step_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
                    _update_ema(ema_model, model, ema_decay)

                    bs = a_hist.shape[0]
                    tr_loss_sum += float(loss.detach().item()) * bs
                    tr_count += bs

                sched.step()
                train_loss = tr_loss_sum / max(tr_count, 1)
                val_loss, val_rmse_k = evaluate_val_stage_b(
                    ema_model, val_ms, val_loader, criterion, clim_norm_table_t, static_norm_t,
                    harm_coef, norm["sigma_pix"], lat_w, K, harm_k, period, anom_clamp, device
                )
                secs = time.perf_counter() - t0

                if val_loss < best_stage_loss:
                    best_stage_loss = val_loss
                    best_stage_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}

                torch.save(
                    {
                        "stage_idx": stage_idx,
                        "K": K,
                        "epoch": epoch,
                        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "ema_state": {k: v.detach().cpu() for k, v in ema_model.state_dict().items()},
                        "opt_state": opt.state_dict(),
                        "sched_state": sched.state_dict(),
                        "best_stage_loss": best_stage_loss,
                        "best_stage_ema_state": best_stage_state,
                        "config": cfg,
                    },
                    STAGEB_RESUME_PATH,
                )

                writer.writerow([K, epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{secs:.2f}", *[f"{v:.4f}" for v in val_rmse_k]])
                log_f.flush()
                rmse_str = " ".join(f"{ch}={val_rmse_k[i]:.3f}" for i, ch in enumerate(channels))
                print(
                    f"K={K} Ep {epoch:02d}/{stage_epochs} | {secs:6.2f}s | train={train_loss:.4f} | "
                    f"val={val_loss:.4f} (best={best_stage_loss:.4f}) | step{K}_rmse: {rmse_str}",
                    flush=True,
                )

                if K == 8 and secs > max_k8_secs:
                    print(f"ALERT: K=8 epoch took {secs:.1f}s > {max_k8_secs:.1f}s limit. Stopping to report.", flush=True)
                    return 3

            # Carry best EMA checkpoint from stage K into next stage
            model.load_state_dict({k: v.to(device) for k, v in best_stage_state.items()})
            ema_model.load_state_dict({k: v.to(device) for k, v in best_stage_state.items()})
            final_best_loss = best_stage_loss
            final_best_state = best_stage_state
            start_epoch = 1

    if final_best_state is None and resume_ckpt is not None:
        final_best_state = resume_ckpt["best_stage_ema_state"]
        final_best_loss = float(resume_ckpt["best_stage_loss"])

    assert final_best_state is not None
    torch.save(
        {
            "state_dict": final_best_state,
            "config": cfg,
            "stage": "B",
            "best_val_loss_k8": final_best_loss,
            "param_count": param_count,
        },
        MODEL_PATH,
    )
    print(f"\nSaved final Stage B EMA checkpoint (K=8 best_val_loss={final_best_loss:.4f}) -> {MODEL_PATH}", flush=True)

    # Evaluate raw Stage B model on val windows before any tuning
    eval_model = _build_model(cfg, torch.device("cpu"))
    eval_model.load_state_dict(final_best_state)
    eval_model.eval()
    ok_raw = evaluate_raw_val_windows(eval_model, cfg, ds, splits, harm_coef, lat_w)
    return 0 if ok_raw else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MonsoonNet (Stage A or Stage B).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--stage", choices=("a", "b"), default="a")
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = int(cfg["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        n_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(n_threads)
        print(f"Device: CPU (threads={n_threads})", flush=True)
    else:
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})", flush=True)

    ds = load_dataset()
    data_phys = ds["data"]
    static = ds["static"]
    lat = ds["lat"]
    doy = ds["doy"]
    T = data_phys.shape[0]

    splits = split_indices(
        T,
        train_frac=float(cfg["split"]["train"]),
        val_frac=float(cfg["split"]["val"]),
        test_frac=float(cfg["split"]["test"]),
    )
    train_idx = splits["train"]

    harm_k = int(cfg["climatology"]["harmonics_k"])
    period = int(cfg["climatology"]["doy_period"])
    ridge = float(cfg["climatology"]["ridge"])

    if HARMONICS_PATH.exists():
        harm_coef, loaded_k, loaded_p = load_harmonics(HARMONICS_PATH)
        if loaded_k != harm_k or loaded_p != period:
            raise ValueError("Existing clim_harmonics.npz disagrees with config")
    else:
        y_train = to_model_space(data_phys[train_idx])
        harm_coef = fit_harmonic(y_train, doy[train_idx] / float(period), K=harm_k, ridge=ridge)
        save_harmonics(HARMONICS_PATH, harm_coef, K=harm_k, period=period)

    rain_thresh = float(cfg["metrics"]["rain_threshold_mm"])
    if NORM_PATH.exists() and args.stage == "b":
        norm = load_norm(NORM_PATH)
    else:
        clim_train = eval_harmonic(harm_coef, doy[train_idx] / float(period), K=harm_k)
        norm = compute_norm_stats(
            data_phys_train=data_phys[train_idx],
            clim_model_train=clim_train,
            static=static,
            floor_median_frac=float(cfg["norm"]["sigma_floor_median_frac"]),
            rain_threshold_mm=rain_thresh,
        )
        save_norm(NORM_PATH, norm)
        print(f"Saved TRAIN normalization stats -> {NORM_PATH}", flush=True)

    lat_w = lat_weights(lat)
    p_rain = float(norm["rain_pos_rate"])
    pos_weight = min(float(cfg["stage_a"]["bce_pos_weight_max"]), (1.0 - p_rain) / max(p_rain, 1e-6))

    with prevent_windows_sleep():
        if args.stage == "a":
            train_stage_a(cfg, ds, splits, harm_coef, norm, lat_w, pos_weight, device, args.skip_overfit)
            return 0
        return train_stage_b(cfg, ds, splits, harm_coef, norm, lat_w, pos_weight, device, args.resume)


if __name__ == "__main__":
    sys.exit(main())
