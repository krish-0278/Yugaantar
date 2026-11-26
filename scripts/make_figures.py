"""Generate publication figures for brief/brief.pdf from real model and evaluation data."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monsoon.climatology import evaluate as eval_harmonic, evaluate_empirical, fit_empirical, sincos_from_doy
from monsoon.data import from_model_space, load_dataset, split_indices
from monsoon.infer import Simulator
from monsoon.metrics import day8_spatial_variance_ratio, lat_weighted_rmse, lat_weights
from monsoon.model import MonsoonNet
from monsoon.postproc import apply as apply_postproc, load_postproc
from monsoon.rollout import load_rollout_context, rollout
from monsoon.shocks_spec import LAT_CENTRES, LON_CENTRES, build_warm_ocean
from monsoon.windows import select_heldout_windows


FIG_DIR = ROOT / "brief" / "figures"
CONFIG_PATH = ROOT / "config" / "default.json"
MODEL_PATH = ROOT / "weights" / "model.pt"
POSTPROC_PATH = ROOT / "weights" / "postproc.json"
STATIC_PATH = ROOT / "weights" / "static.npz"

INK = "#1d1d1b"
MUTED = "#6b675d"
RULE = "#cfc9bb"
PANEL = "#faf8f3"
ACCENT = "#0f5c5c"
SHOCK = "#c8402a"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": PANEL,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.grid": True,
            "grid.color": RULE,
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
        }
    )


def _draw_coastlines(ax: plt.Axes, land_mask: np.ndarray) -> None:
    """Draw exact 0.5 land-sea cell boundary contour on [32, 36] grid."""
    h, w = land_mask.shape
    for i in range(h):
        lat_bot = LAT_CENTRES[i] - 0.5
        lat_top = LAT_CENTRES[i] + 0.5
        for j in range(w):
            lon_l = LON_CENTRES[j] - 0.5
            lon_r = LON_CENTRES[j] + 0.5
            v = bool(land_mask[i, j])
            if i + 1 < h and v != bool(land_mask[i + 1, j]):
                ax.plot([lon_l, lon_r], [lat_top, lat_top], color=INK, lw=0.7, alpha=0.85)
            if j + 1 < w and v != bool(land_mask[i, j + 1]):
                ax.plot([lon_r, lon_r], [lat_bot, lat_top], color=INK, lw=0.7, alpha=0.85)


def compute_test_curves() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ds = load_dataset()
    data_phys = ds["data"]
    lat = ds["lat"]
    doy = ds["doy"]
    channels = list(ds["channels"])
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
    anom_clamp = float(cfg["stage_b"]["anom_clamp"])
    input_clip = float(cfg["norm"]["anomaly_clip"])
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
    test_wins = win_by_split["test"]

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

    max_lead = 10
    n_win = len(test_wins)
    true_roll = np.empty((n_win, max_lead, 6, 32, 36), dtype=np.float32)
    pers_roll = np.empty_like(true_roll)
    harm_roll = np.empty_like(true_roll)
    emp_roll = np.empty_like(true_roll)
    post_roll = np.empty_like(true_roll)

    for i, w in enumerate(test_wins):
        t = int(w["t_last"])
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
        post_roll[i] = apply_postproc(
            a_out, ev_out, doy_last, max_lead, pp_cfg, ctx,
            use_beta=True, use_kappa=True, use_tp_var=True, use_event=True, use_clamp=True,
        )

    methods = {
        "persistence": pers_roll,
        "clim_harmonic": harm_roll,
        "clim_empirical": emp_roll,
        "model_postproc": post_roll,
    }

    leads = list(range(1, 11))
    mean_skill: dict[str, list[float]] = {m: [] for m in methods}
    for lead in leads:
        idx = lead - 1
        t_l = true_roll[:, idx]
        rmses = {m: lat_weighted_rmse(roll[:, idx], t_l, lat_w) for m, roll in methods.items()}
        best_base = np.minimum(rmses["persistence"], np.minimum(rmses["clim_harmonic"], rmses["clim_empirical"]))
        for m in methods:
            ss = 1.0 - rmses[m] / best_base
            mean_skill[m].append(float(np.mean(ss)))

    vr_day8 = {m: day8_spatial_variance_ratio(roll[:, 7], true_roll[:, 7]) for m, roll in methods.items()}

    # Pick first test window near monsoon season (DOY 150..240) if available
    shock_win = next((w for w in test_wins if 150 <= int(doy[int(w["t_last"])]) <= 240), test_wins[0])
    t_sw = int(shock_win["t_last"])
    init_sw = data_phys[t_sw - 2 : t_sw + 1]
    doy_sw = int(doy[t_sw])

    return {
        "n_test_win": n_win,
        "channels": channels,
        "leads": leads,
        "mean_skill": mean_skill,
        "vr_day8": vr_day8,
        "shock_win": {"t_last": t_sw, "doy": doy_sw, "category": shock_win.get("category", "test")},
        "init_sw": init_sw,
    }


def make_skill_vs_lead(data: dict[str, object]) -> None:
    leads = data["leads"]
    ms = data["mean_skill"]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=220)
    ax.axhline(0.0, color=MUTED, lw=0.9, ls="--", zorder=1)
    ax.plot(leads, ms["model_postproc"], color=ACCENT, lw=2.2, marker="o", ms=4.5, label="Model (postproc)", zorder=4)
    ax.plot(leads, ms["clim_empirical"], color=INK, lw=1.4, ls="-.", marker="s", ms=3.5, label="Empirical climatology", zorder=3)
    ax.plot(leads, ms["clim_harmonic"], color=MUTED, lw=1.2, ls=":", label="Harmonic climatology", zorder=2)
    ax.plot(leads, ms["persistence"], color=SHOCK, lw=1.5, ls="--", marker="^", ms=3.8, label="Persistence", zorder=3)

    ax.set_xlim(0.7, 10.3)
    ax.set_xticks(leads)
    ax.set_ylim(-0.58, 0.36)
    ax.set_xlabel("Lead Day")
    ax.set_ylabel("Mean Skill vs Best Baseline")
    ax.set_title(f"Forecast Skill vs Lead Day (N = {data['n_test_win']} Held-Out Test Windows)")
    ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor=RULE, fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "skill_vs_lead.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)

    print("\n=== FIGURE 1: skill_vs_lead.png ===")
    print("lead | model_postproc | persistence | clim_empirical | clim_harmonic")
    for idx, d in enumerate(leads):
        print(
            f"  {d:2d} | {ms['model_postproc'][idx]:+14.4f} | {ms['persistence'][idx]:+11.4f} | "
            f"{ms['clim_empirical'][idx]:+14.4f} | {ms['clim_harmonic'][idx]:+13.4f}"
        )


def make_variance_ratio(data: dict[str, object]) -> None:
    channels = data["channels"]
    vr_model = data["vr_day8"]["model_postproc"]
    vr_emp = data["vr_day8"]["clim_empirical"]

    x = np.arange(len(channels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=220)
    b1 = ax.bar(x - width / 2, vr_model, width, color=ACCENT, edgecolor=INK, linewidth=0.6, label="Model (postproc)")
    b2 = ax.bar(x + width / 2, vr_emp, width, color=RULE, edgecolor=INK, linewidth=0.6, label="Empirical climatology")
    ax.axhline(0.40, color=SHOCK, lw=1.6, ls="--", label="0.40 anti-blur floor")

    for rect in b1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.025, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)
    for rect in b2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.025, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylim(0.0, 1.34)
    ax.set_ylabel("Day-8 Spatial Variance Ratio")
    ax.set_title(f"Day-8 Spatial Variance Ratio per Channel (N = {data['n_test_win']} Test Windows)")
    ax.legend(loc="upper center", ncol=3, frameon=True, facecolor="#ffffff", edgecolor=RULE, fontsize=7.8)
    fig.tight_layout()
    out = FIG_DIR / "variance_ratio.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)

    print("\n=== FIGURE 3: variance_ratio.png ===")
    print("channel | model_postproc | clim_empirical | clim_harmonic | persistence")
    for idx, ch in enumerate(channels):
        print(
            f"  {ch:5s} | {vr_model[idx]:14.4f} | {vr_emp[idx]:14.4f} | "
            f"{data['vr_day8']['clim_harmonic'][idx]:13.4f} | {data['vr_day8']['persistence'][idx]:11.4f}"
        )


def make_shock_4panel(data: dict[str, object]) -> None:
    static = np.load(STATIC_PATH)["static"]
    land_mask = static[0] >= 0.5

    init_phys = data["init_sw"]
    doy_last = int(data["shock_win"]["doy"])
    doy_sc = sincos_from_doy(doy_last, period=365)

    sim = Simulator()
    pert = build_warm_ocean(init_phys[2])
    ctrl = sim.simulate(init_phys, doy_sc, 10, None)
    shock = sim.simulate(init_phys, doy_sc, 10, pert)
    resp = shock - ctrl  # [10, 6, 32, 36]

    days = [1, 3, 6, 10]
    row_specs = [
        (0, "t2m response (K)", float(np.max(np.abs(resp[[d - 1 for d in days], 0])))),
        (1, "tp response (mm/day)", float(np.max(np.abs(resp[[d - 1 for d in days], 1])))),
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "monsoon_div", ["#215e8a", "#94bcd2", "#f4f1ea", "#e29270", "#c8402a"]
    )
    extent = [LON_CENTRES[0] - 0.5, LON_CENTRES[-1] + 0.5, LAT_CENTRES[0] - 0.5, LAT_CENTRES[-1] + 0.5]

    fig, axes = plt.subplots(2, 4, figsize=(8.8, 4.4), dpi=220, sharex=True, sharey=True)
    for r, (ch_idx, row_label, vabs) in enumerate(row_specs):
        vmax = max(vabs, 1e-4)
        im = None
        for c_idx, day in enumerate(days):
            ax = axes[r, c_idx]
            ax.grid(False)
            field = resp[day - 1, ch_idx]
            im = ax.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
            _draw_coastlines(ax, land_mask)
            ax.plot(
                [81.5, 94.5, 94.5, 81.5, 81.5],
                [7.5, 7.5, 20.5, 20.5, 7.5],
                color=SHOCK,
                ls="--",
                lw=1.1,
            )
            if r == 0:
                ax.set_title(f"Day {day}", fontsize=9, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(f"{row_label}\nLat (°N)", fontsize=8)
            if r == 1:
                ax.set_xlabel("Lon (°E)", fontsize=8)
            ax.tick_params(labelsize=7)
        cbar = fig.colorbar(im, ax=axes[r, :], fraction=0.018, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

    sw = data["shock_win"]
    fig.suptitle(
        f"Warm Bay of Bengal Response via monsoon.infer.Simulator (test t={sw['t_last']}, DOY={sw['doy']}, {sw['category']})",
        fontsize=10,
        y=0.98,
    )
    out = FIG_DIR / "shock_4panel.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"\n=== FIGURE 2: shock_4panel.png (test t_last={sw['t_last']}, DOY={sw['doy']}) ===")
    box = ((LAT_CENTRES >= 8.0) & (LAT_CENTRES <= 20.0))[:, None] & ((LON_CENTRES >= 82.0) & (LON_CENTRES <= 94.0))[None, :]
    print("day | t2m_box_mean(K) | t2m_max_abs(K) | tp_box_mean(mm/d) | tp_max_abs(mm/d)")
    for day in days:
        rt = resp[day - 1, 0]
        rp = resp[day - 1, 1]
        print(
            f" {day:2d} | {rt[box].mean():+15.4f} | {np.max(np.abs(rt)):14.4f} | "
            f"{rp[box].mean():+17.4f} | {np.max(np.abs(rp)):16.4f}"
        )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()
    data = compute_test_curves()
    make_skill_vs_lead(data)
    make_shock_4panel(data)
    make_variance_ratio(data)


if __name__ == "__main__":
    main()
