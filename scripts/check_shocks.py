"""Diagnostic comparison of shock-response variants (V0-V3) on de-duplicated windows."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monsoon_r1 import contract as r1_contract, scoring as r1_scoring

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monsoon.climatology import evaluate as eval_harmonic, sincos_from_doy
from monsoon.data import from_model_space, load_dataset, split_indices
from monsoon.infer import Simulator, decode_doy_last
from monsoon.postproc import apply as apply_postproc
from monsoon.rollout import rollout
from monsoon.shock import causal_cone_mask
from monsoon.shocks_spec import build_monsoon_stall, build_warm_ocean
from monsoon.windows import deduplicate_windows, select_heldout_windows


CONFIG_PATH = ROOT / "config" / "default.json"


def _compute_variant_output(
    sim: Simulator,
    anom_ctrl: np.ndarray,
    ev_ctrl: np.ndarray,
    anom_pert: np.ndarray,
    ev_pert: np.ndarray,
    control: np.ndarray,
    perturbation: np.ndarray,
    doy_last: int,
    steps: int,
    variant: str,
) -> np.ndarray:
    """Compute physical shocked rollout for diagnostic variants V0..V3 without changing serve path."""
    pp = sim.postproc_cfg
    ctx = sim.ctx
    std = sim.channel_std_phys
    cap = float(sim.cap_sigma) * std[None, :, None, None]

    if variant == "V0":
        pert_phys = apply_postproc(anom_pert, ev_pert, doy_last, steps, postproc_cfg=pp, ctx=ctx).astype(np.float32)
        resp = (pert_phys - control).astype(np.float32, copy=True)
    elif variant == "V1":
        ctrl_dyn = apply_postproc(
            anom_ctrl, ev_ctrl, doy_last, steps, postproc_cfg=pp, ctx=ctx, use_beta=False
        ).astype(np.float32)
        pert_dyn = apply_postproc(
            anom_pert, ev_pert, doy_last, steps, postproc_cfg=pp, ctx=ctx, use_beta=False
        ).astype(np.float32)
        resp = (pert_dyn - ctrl_dyn).astype(np.float32, copy=True)
    elif variant in ("V2", "V3"):
        ctrl_dyn = apply_postproc(
            anom_ctrl,
            ev_ctrl,
            doy_last,
            steps,
            postproc_cfg=pp,
            ctx=ctx,
            use_beta=False,
            use_tp_var=False,
            use_event=False,
        ).astype(np.float32)
        pert_dyn = apply_postproc(
            anom_pert,
            ev_pert,
            doy_last,
            steps,
            postproc_cfg=pp,
            ctx=ctx,
            use_beta=False,
            use_tp_var=False,
            use_event=False,
        ).astype(np.float32)
        resp = (pert_dyn - ctrl_dyn).astype(np.float32, copy=True)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    mask = causal_cone_mask(perturbation, steps, cone_speed=sim.cone_speed, taper=sim.taper)
    resp = resp * mask[:, None, :, :]

    if variant == "V3":
        resp = (cap * np.tanh(resp / np.maximum(cap, 1e-12))).astype(np.float32)
    else:
        resp = np.clip(resp, -cap, cap).astype(np.float32)

    out = (control + resp).astype(np.float32)
    out[:, 1] = np.maximum(out[:, 1], 0.0)
    out[:, 0] = np.clip(out[:, 0], sim.t2m_min_k, sim.t2m_max_k)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    ds = load_dataset()
    data_phys = ds["data"]
    doy = ds["doy"]
    splits = split_indices(
        data_phys.shape[0],
        train_frac=float(cfg["split"]["train"]),
        val_frac=float(cfg["split"]["val"]),
        test_frac=float(cfg["split"]["test"]),
    )

    sim = Simulator()
    clim_all = from_model_space(
        eval_harmonic(sim.ctx["harm_coef"], doy / float(sim.ctx["period"]), K=sim.ctx["harm_k"])
    )
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
    raw_wins = win_by_split["val"] + win_by_split["test"]
    wins = deduplicate_windows(raw_wins)
    n_val = sum(1 for w in wins if w["split"] == "val")
    n_test = sum(1 for w in wins if w["split"] == "test")
    print(f"Windows: raw={len(raw_wins)} -> de-duplicated={len(wins)} (val={n_val}, test={n_test})")

    steps = 10
    std = sim.channel_std_phys
    channels = cfg["channels"]
    shocks_spec = {
        "warm_ocean": {
            "builder": build_warm_ocean,
            "r1_box": r1_contract.BOX_BAY_OF_BENGAL,
            "shocked_chs": (0, 5),
        },
        "monsoon_stall": {
            "builder": build_monsoon_stall,
            "r1_box": r1_contract.BOX_ARABIAN_SEA,
            "shocked_chs": (2, 3),
        },
    }

    # Precompute 10-step model-space rollouts once per window & shock
    precomp: list[dict[str, Any]] = []
    with torch.inference_mode():
        for w in wins:
            t = int(w["t_last"])
            init = data_phys[t - 2 : t + 1]
            doy_sc = sincos_from_doy(int(doy[t]), period=365).tolist()
            doy_last = decode_doy_last(doy_sc)
            anom_ctrl, ev_ctrl = rollout(
                sim.model,
                init,
                doy_last,
                steps,
                delta=None,
                ctx=sim.ctx,
                anom_clamp=sim.anom_clamp,
                input_clip=sim.input_clip,
            )
            control = apply_postproc(
                anom_ctrl, ev_ctrl, doy_last, steps, postproc_cfg=sim.postproc_cfg, ctx=sim.ctx
            ).astype(np.float32)

            entry: dict[str, Any] = {
                "doy_last": doy_last,
                "anom_ctrl": anom_ctrl,
                "ev_ctrl": ev_ctrl,
                "control": control,
                "shocks": {},
            }
            for sname, spec in shocks_spec.items():
                delta = spec["builder"](init[-1])
                anom_pert, ev_pert = rollout(
                    sim.model,
                    init,
                    doy_last,
                    steps,
                    delta=delta,
                    ctx=sim.ctx,
                    anom_clamp=sim.anom_clamp,
                    input_clip=sim.input_clip,
                )
                entry["shocks"][sname] = (delta, anom_pert, ev_pert)
            precomp.append(entry)

    variants = ("V0", "V1", "V2", "V3")
    stats_std = np.asarray(json.loads((ROOT / "data" / "stats.json").read_text(encoding="utf-8"))["std"], dtype=np.float32)
    print("\n=== VARIANT COMPARISON (CONES ON, 10-STEP ROLLOUT, N={} UNIQUE WINDOWS) ===".format(len(wins)))
    for sname, spec in shocks_spec.items():
        r1_box = spec["r1_box"]
        near = r1_scoring.near_mask(r1_box)
        schs = spec["shocked_chs"]
        print(f"\n--- Shock: {sname} (shocked channels: {[channels[c] for c in schs]}) ---")
        header = (
            f"{'Var':<3} | {'fin':>5} {'nz':>5} {'loc':>5} {'prop':>6} {'ntel':>5} {'stab':>5} {'all':>6}"
            f" | {'c1%':>5} {'c2%':>5} {'stat%':>5} | {'R_min':>6} {'R_p10':>6} {'R_med':>6} | {'max|z|':>6} | "
            + " ".join(f"d8_{channels[c]}" for c in schs)
        )
        print(header)
        print("-" * len(header))

        for var in variants:
            checks = []
            checks_stats = []
            c1_list = []
            c2_list = []
            r_vals = []
            max_z = 0.0
            d8_ch_max = {c: [] for c in schs}
            d8_ch_rms = {c: [] for c in schs}

            for item in precomp:
                delta, anom_pert, ev_pert = item["shocks"][sname]
                out = _compute_variant_output(
                    sim,
                    item["anom_ctrl"],
                    item["ev_ctrl"],
                    anom_pert,
                    ev_pert,
                    item["control"],
                    delta,
                    item["doy_last"],
                    steps,
                    var,
                )
                ck = r1_scoring.causality_checks(item["control"], out, r1_box, std)
                ck_s = r1_scoring.causality_checks(item["control"], out, r1_box, stats_std)
                checks.append(ck)
                checks_stats.append(ck_s)

                d = out - item["control"]
                z = np.abs(d / std[None, :, None, None])
                max_z = max(max_z, float(np.max(z)))
                e = r1_scoring._energy(d, std)
                out_e = np.array([e[k][~near].sum() for k in range(steps)], dtype=np.float64)
                late = float(out_e[3:8].mean())
                emax = float(e.sum(axis=(1, 2)).max())
                c1_list.append(bool(late > 1.5 * float(out_e[0]) + 1e-12))
                c2_list.append(bool(late > 0.02 * max(emax, 1e-12)))
                r_vals.append(late / (0.02 * max(emax, 1e-12)))

                for c in schs:
                    d8_ch_max[c].append(float(np.max(z[7, c])))
                    d8_ch_rms[c].append(float(np.sqrt(np.mean((d[7, c] / std[c]) ** 2))))

            r_arr = np.asarray(r_vals, dtype=np.float64)
            rates = {
                k: 100.0 * float(np.mean([bool(c[k]) for c in checks]))
                for k in ("finite", "nonzero", "localized", "propagates", "no_teleport", "stable")
            }
            all_pass = 100.0 * float(
                np.mean(
                    [
                        bool(c["finite"] and c["nonzero"] and c["localized"] and c["propagates"] and c["stable"])
                        for c in checks
                    ]
                )
            )
            stats_pass = 100.0 * float(
                np.mean(
                    [
                        bool(c["finite"] and c["nonzero"] and c["localized"] and c["propagates"] and c["stable"])
                        for c in checks_stats
                    ]
                )
            )
            c1_pct = 100.0 * float(np.mean(c1_list))
            c2_pct = 100.0 * float(np.mean(c2_list))
            d8_str = " ".join(
                f"{np.median(d8_ch_max[c]):.3f}({np.median(d8_ch_rms[c]):.3f})" for c in schs
            )
            print(
                f"{var:<3} | {rates['finite']:5.1f} {rates['nonzero']:5.1f} {rates['localized']:5.1f} "
                f"{rates['propagates']:6.1f} {rates['no_teleport']:5.1f} {rates['stable']:5.1f} {all_pass:6.1f}"
                f" | {c1_pct:5.1f} {c2_pct:5.1f} {stats_pass:5.1f}"
                f" | {np.min(r_arr):6.3f} {np.percentile(r_arr, 10):6.3f} {np.median(r_arr):6.3f}"
                f" | {max_z:6.4f} | {d8_str}"
            )


if __name__ == "__main__":
    main()
