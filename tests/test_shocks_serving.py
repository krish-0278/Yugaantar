"""Shock causality & locality verification across all 96 de-duplicated val+test windows (cone ON and cone OFF)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from monsoon_r1 import contract as r1_contract, scoring as r1_scoring
from scipy.ndimage import distance_transform_cdt, distance_transform_edt

from monsoon.climatology import evaluate as eval_harmonic, sincos_from_doy
from monsoon.data import from_model_space, load_dataset, split_indices
from monsoon.infer import Simulator
from monsoon.shocks_spec import box_mask, build_monsoon_stall, build_null, build_warm_ocean
from monsoon.windows import deduplicate_windows, select_heldout_windows


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "default.json"


def _fmt_mmm(vals: np.ndarray) -> str:
    return f"{float(np.min(vals)):.4f} / {float(np.median(vals)):.4f} / {float(np.max(vals)):.4f}"


def _evaluate_shocks(
    sim: Simulator,
    windows_meta: list[dict[str, Any]],
    windows_data: list[tuple[np.ndarray, list[float]]],
    use_cone: bool,
) -> dict[str, Any]:
    std = sim.channel_std_phys
    channels = sim.cfg["channels"]

    specs = {
        "warm_ocean": {
            "builder": build_warm_ocean,
            "box": (8.0, 20.0, 82.0, 94.0),
            "r1_box": r1_contract.BOX_BAY_OF_BENGAL,
            "shocked_chs": (0, 5),
        },
        "monsoon_stall": {
            "builder": build_monsoon_stall,
            "box": (8.0, 20.0, 65.0, 75.0),
            "r1_box": r1_contract.BOX_ARABIAN_SEA,
            "shocked_chs": (2, 3),
        },
    }

    summary: dict[str, Any] = {}
    null_eq_same_len = []
    null_cross_len_diff = []
    controls_30 = []
    for init_states, doy_sc in windows_data:
        c10 = sim.simulate(init_states, doy_sc, 10, None, use_cone=use_cone)
        c30 = sim.simulate(init_states, doy_sc, 30, None, use_cone=use_cone)
        controls_30.append(c30)
        null_10 = sim.simulate(init_states, doy_sc, 10, build_null(), use_cone=use_cone)
        null_eq_same_len.append(bool(np.array_equal(null_10, c10)))
        null_cross_len_diff.append(float(np.max(np.abs(null_10 - c30[:10]) / std[None, :, None, None])))

    summary["null_eq_same_len"] = np.array(null_eq_same_len, dtype=bool)
    summary["null_cross_len_diff"] = np.array(null_cross_len_diff, dtype=np.float64)

    for sname, spec in specs.items():
        bm = box_mask(*spec["box"])
        d_cheb = distance_transform_cdt(~bm, metric="chessboard").astype(np.float64)
        d_euc = distance_transform_edt(~bm).astype(np.float64)
        near_cheb = d_cheb <= 3.0
        near_euc = d_euc <= 3.0

        d1_cheb = []
        d1_euc = []
        d1_ch_max = []
        d1_tp_max_mm = []
        d1_tp_cap_hits = []
        mean_abs_u10_box = []
        beyond_cheb = {1: [], 4: [], 6: [], 8: []}
        beyond_euc = {1: [], 4: [], 6: [], 8: []}
        front_radii = []
        front_max_jump = []
        max_30_z = []
        nan_inf_total = 0

        kit_checks = []
        kit_fail_indices = []
        tp_cap_mm = 4.0 * float(std[1])

        for idx, (init_states, doy_sc) in enumerate(windows_data):
            delta = spec["builder"](init_states[-1])
            c30 = controls_30[idx]
            p30 = sim.simulate(init_states, doy_sc, 30, delta, use_cone=use_cone)
            if not np.isfinite(p30).all():
                nan_inf_total += int((~np.isfinite(p30)).sum())

            ck = r1_scoring.causality_checks(c30[:10], p30[:10], spec["r1_box"], std)
            ck_all = bool(ck["finite"] and ck["nonzero"] and ck["localized"] and ck["propagates"] and ck["stable"])
            kit_checks.append(ck)
            if not ck_all:
                kit_fail_indices.append((idx, windows_meta[idx]["split"], windows_meta[idx]["category"], ck))

            resp30 = p30 - c30
            z30 = np.abs(resp30) / std[None, :, None, None]
            max_30_z.append(float(np.max(z30)))

            max_tp_d1 = float(np.max(np.abs(resp30[0, 1])))
            d1_tp_max_mm.append(max_tp_d1)
            d1_tp_cap_hits.append(bool(max_tp_d1 >= tp_cap_mm - 1e-4))
            mean_abs_u10_box.append(float(np.mean(np.abs(init_states[-1, 2][bm]))))

            resp8 = resp30[:8]
            energy8 = ((resp8 / std[None, :, None, None]) ** 2).sum(axis=1)
            e1_tot = float(energy8[0].sum())

            d1_cheb.append(float(energy8[0][near_cheb].sum() / max(e1_tot, 1e-30)))
            d1_euc.append(float(energy8[0][near_euc].sum() / max(e1_tot, 1e-30)))
            d1_ch_max.append(z30[0].max(axis=(1, 2)))

            for day in (1, 4, 6, 8):
                ek = energy8[day - 1]
                etot = float(ek.sum())
                beyond_cheb[day].append(float(ek[~near_cheb].sum() / max(etot, 1e-30)))
                beyond_euc[day].append(float(ek[~near_euc].sum() / max(etot, 1e-30)))

            r_days = []
            max_d = int(d_cheb.max())
            for day in range(8):
                ek = energy8[day]
                etot = float(ek.sum())
                r95 = 0
                if etot > 0:
                    for r in range(max_d + 1):
                        if float(ek[d_cheb <= r].sum() / etot) >= 0.95:
                            r95 = r
                            break
                r_days.append(r95)
            front_radii.append(r_days)
            front_max_jump.append(int(np.max(np.diff(r_days))))

        front_arr = np.array(front_radii, dtype=np.float64)
        prop_e_ok = np.mean(np.array(beyond_cheb[6]) > np.array(beyond_cheb[1]))
        prop_r_ok = np.mean(front_arr[:, 7] >= (front_arr[:, 0] + 1.0))
        prop_both_ok = np.mean(
            (np.array(beyond_cheb[6]) > np.array(beyond_cheb[1])) & (front_arr[:, 7] >= (front_arr[:, 0] + 1.0))
        )

        kit_rates = {
            key: float(np.mean([bool(c[key]) for c in kit_checks]))
            for key in ("finite", "nonzero", "localized", "propagates", "no_teleport", "stable")
        }
        kit_rates["all_pass"] = float(
            np.mean([bool(c["finite"] and c["nonzero"] and c["localized"] and c["propagates"] and c["stable"]) for c in kit_checks])
        )

        summary[sname] = {
            "d1_cheb": np.array(d1_cheb),
            "d1_euc": np.array(d1_euc),
            "d1_ch_max": np.stack(d1_ch_max, axis=0),
            "d1_tp_max_mm": np.array(d1_tp_max_mm, dtype=np.float64),
            "d1_tp_cap_hits": np.array(d1_tp_cap_hits, dtype=bool),
            "mean_abs_u10_box": np.array(mean_abs_u10_box, dtype=np.float64),
            "beyond_cheb": {d: np.array(v) for d, v in beyond_cheb.items()},
            "beyond_euc": {d: np.array(v) for d, v in beyond_euc.items()},
            "front_radii": front_arr,
            "front_max_jump": np.array(front_max_jump),
            "max_30_z": np.array(max_30_z),
            "nan_inf_total": nan_inf_total,
            "prop_e_ok": float(prop_e_ok),
            "prop_r_ok": float(prop_r_ok),
            "prop_both_ok": float(prop_both_ok),
            "kit_checks": kit_checks,
            "kit_rates": kit_rates,
            "kit_fail_indices": kit_fail_indices,
            "shocked_chs": spec["shocked_chs"],
            "channels": channels,
            "n_windows": len(windows_data),
        }
    return summary


def _print_table(title: str, summ: dict[str, Any]) -> None:
    print(f"\n==================== {title} ====================")
    print(
        f"Null shock same-length (10v10) np.array_equal: "
        f"{int(summ['null_eq_same_len'].sum())}/{len(summ['null_eq_same_len'])}"
    )
    print(
        f"Null shock cross-length (10v30) max |diff|/std (min/med/max): "
        f"{_fmt_mmm(summ['null_cross_len_diff'])}"
    )
    for sname in ("warm_ocean", "monsoon_stall"):
        d = summ[sname]
        chs = d["channels"]
        n_win = d["n_windows"]
        print(f"\n--- Shock: {sname} ({n_win} de-duplicated windows) ---")
        print(f"1a. Day-1 energy <= 3 cells (Chebyshev) min/med/max : {_fmt_mmm(d['d1_cheb'])}")
        print(f"1b. Day-1 energy <= 3 cells (Euclidean) min/med/max : {_fmt_mmm(d['d1_euc'])}")
        print("2.  Day-1 max |resp|/std per channel (min/med/max):")
        for ci, ch in enumerate(chs):
            print(f"      {ch:<5}: {_fmt_mmm(d['d1_ch_max'][:, ci])}")
        print(
            f"    Day-1 tp 4-sigma cap hits: {int(d['d1_tp_cap_hits'].sum())}/{n_win} "
            f"({100.0 * float(np.mean(d['d1_tp_cap_hits'])):.2f}%), "
            f"median max |day-1 tp resp| = {float(np.median(d['d1_tp_max_mm'])):.4f} mm/day"
        )
        print("3a. Energy > 3 cells (Chebyshev) on days 1,4,6,8 (min/med/max):")
        for day in (1, 4, 6, 8):
            print(f"      day {day}: {_fmt_mmm(d['beyond_cheb'][day])}")
        print("3b. Energy > 3 cells (Euclidean) on days 1,4,6,8 (min/med/max):")
        for day in (1, 4, 6, 8):
            print(f"      day {day}: {_fmt_mmm(d['beyond_euc'][day])}")
        print("4.  95%-energy front radius (Chebyshev) days 1..8 (min/med/max) [info only]:")
        for day in range(1, 9):
            print(f"      day {day}: {_fmt_mmm(d['front_radii'][:, day - 1])}")
        print(f"    Max day-to-day front increase (min/med/max) [info only]: {_fmt_mmm(d['front_max_jump'])}")
        print(
            f"    Propagation shares [info]: beyond3(d6>d1)={100*d['prop_e_ok']:.1f}%, "
            f"radius(d8>=d1+1)={100*d['prop_r_ok']:.1f}%, BOTH={100*d['prop_both_ok']:.1f}%"
        )
        print(f"5.  30-step max |resp|/std (min/med/max): {_fmt_mmm(d['max_30_z'])}, NaN/Inf={d['nan_inf_total']}")
        kr = d["kit_rates"]
        print(
            "6.  Kit causality_checks pass rates: "
            + ", ".join(f"{k}={100*v:.1f}%" for k, v in kr.items())
        )
        if sname == "monsoon_stall":
            u10 = d["mean_abs_u10_box"]
            props = np.array([bool(c["propagates"]) for c in d["kit_checks"]], dtype=bool)
            strata = [
                ("< 3 m/s", u10 < 3.0),
                ("3 to 5 m/s", (u10 >= 3.0) & (u10 < 5.0)),
                (">= 5 m/s", u10 >= 5.0),
            ]
            print("    monsoon_stall kit propagates stratified by box mean |u10| [info]:")
            for label, mask in strata:
                cnt = int(mask.sum())
                ok = int(props[mask].sum())
                pct = (100.0 * ok / cnt) if cnt > 0 else 0.0
                print(f"      {label:<12}: {ok:2d}/{cnt:2d} ({pct:5.1f}%)")
        if d["kit_fail_indices"]:
            print(f"    Failing windows ({len(d['kit_fail_indices'])}):")
            for idx, sp, cat, ck in d["kit_fail_indices"]:
                print(
                    f"      idx={idx} ({sp}/{cat}): nonzero={ck['nonzero']}, localized={ck['localized']} "
                    f"(near_d1={ck['near_frac_day1']:.4f}), propagates={ck['propagates']}, "
                    f"no_teleport={ck['no_teleport']} (far_max={ck['far_frac_max']:.4f}), stable={ck['stable']}"
                )


def test_shocks_across_96_windows_cone_on_and_off() -> None:
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
    clim_all = from_model_space(eval_harmonic(sim.ctx["harm_coef"], doy / float(sim.ctx["period"]), K=sim.ctx["harm_k"]))
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
    all_wins = deduplicate_windows(win_by_split["val"] + win_by_split["test"])
    assert len(all_wins) == 96

    windows_data = []
    for w in all_wins:
        t = int(w["t_last"])
        init_states = data_phys[t - 2 : t + 1]
        doy_sc = sincos_from_doy(int(doy[t]), period=365).tolist()
        windows_data.append((init_states, doy_sc))

    summ_on = _evaluate_shocks(sim, all_wins, windows_data, use_cone=True)
    summ_off = _evaluate_shocks(sim, all_wins, windows_data, use_cone=False)

    _print_table("CONE ON (PRODUCTION SERVING PATH - V1)", summ_on)
    _print_table("CONE OFF (ABLATION: NETWORK ALONE - V1)", summ_off)

    # Null consistency assertions
    assert bool(np.all(summ_on["null_eq_same_len"]))
    assert float(np.max(summ_on["null_cross_len_diff"])) <= 1e-3

    # Assertions on CONE ON path: kit causality checks + locality + boundedness
    for sname in ("warm_ocean", "monsoon_stall"):
        d = summ_on[sname]
        assert d["nan_inf_total"] == 0
        assert float(np.min(d["d1_cheb"])) >= 0.80
        assert float(np.min(d["d1_euc"])) >= 0.80
        assert float(np.max(d["max_30_z"])) <= 4.0 + 1e-5
        for ch_idx in d["shocked_chs"]:
            assert float(np.min(d["d1_ch_max"][:, ch_idx])) > 0.0

        kr = d["kit_rates"]
        assert kr["finite"] == 1.0
        assert kr["nonzero"] == 1.0
        assert kr["localized"] == 1.0
        assert kr["no_teleport"] == 1.0
        assert kr["stable"] == 1.0

    assert summ_on["warm_ocean"]["kit_rates"]["propagates"] >= 0.95, (
        f"warm_ocean propagates {summ_on['warm_ocean']['kit_rates']['propagates']:.3f} < 0.95"
    )
