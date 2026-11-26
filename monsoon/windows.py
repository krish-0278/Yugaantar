"""Data-driven evaluation window selection on held-out splits (never by date)."""

from __future__ import annotations

from typing import Any

import numpy as np


CH_T2M = 0
CH_TP = 1
CH_MSL = 4
CATEGORIES = ("rain_onset", "heavy_rain", "heatwave", "cold_disturbance", "random")


def _greedy_non_overlapping(
    candidates: np.ndarray,
    scores: np.ndarray,
    count: int,
    min_sep: int,
) -> list[int]:
    """Select up to `count` indices ordered by descending score with |i - j| >= min_sep."""
    order = np.argsort(-scores, kind="stable")
    chosen: list[int] = []
    for pos in order:
        idx = int(candidates[pos])
        if all(abs(idx - c) >= min_sep for c in chosen):
            chosen.append(idx)
            if len(chosen) >= count:
                break
    return chosen


def select_split_windows(
    data: np.ndarray,
    doy: np.ndarray,
    clim_phys: np.ndarray,
    split_idx: np.ndarray,
    split_name: str,
    n_in: int = 3,
    n_out: int = 10,
    per_category: int = 10,
    min_sep: int = 13,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Select non-overlapping held-out windows within one split by data-driven rules.

    Each selected index `t_last` uses inputs `[t_last - (n_in - 1) .. t_last]` and targets
    `[t_last + 1 .. t_last + n_out]`, all strictly inside `split_idx`.
    `min_sep >= 13` guarantees both non-overlapping 13-day spans and >= 10 days separation.
    """
    start = int(split_idx[0]) + (n_in - 1)
    end = int(split_idx[-1]) - n_out  # inclusive upper bound for t_last
    if end < start:
        raise ValueError(f"Split '{split_name}' too short for n_in={n_in}, n_out={n_out}")
    candidates = np.arange(start, end + 1, dtype=np.int64)

    tp = data[:, CH_TP]
    t2m_anom = data[:, CH_T2M] - clim_phys[:, CH_T2M]
    msl = data[:, CH_MSL]

    domain_mean_tp = tp.mean(axis=(1, 2))
    domain_max_tp = tp.max(axis=(1, 2))

    # Category 1: rain onset = largest 7-day rise in domain-mean tp during forecast
    onset_scores = np.array([domain_mean_tp[t + 7] - domain_mean_tp[t] for t in candidates], dtype=np.float64)
    # Category 2: heavy rain = largest domain-max tp across leads 1..8
    heavy_scores = np.array([domain_max_tp[t + 1 : t + 9].max() for t in candidates], dtype=np.float64)
    # Category 3: heatwave = largest peak spatial-mean t2m anomaly over leads 1..4
    anom_domain = t2m_anom.mean(axis=(1, 2))
    heat_scores = np.array([anom_domain[t + 1 : t + 5].max() for t in candidates], dtype=np.float64)
    # Category 4: cold-season disturbance = largest msl change within cold season (doy <= 60 or >= 305)
    cold_mask = np.array([(doy[t] <= 60 or doy[t] >= 305) for t in candidates], dtype=bool)
    cold_candidates = candidates[cold_mask]
    cold_scores = np.array(
        [
            max(
                float(np.sqrt(np.mean((msl[t + k] - msl[t]) ** 2)))
                for k in range(1, 5)
            )
            for t in cold_candidates
        ],
        dtype=np.float64,
    )

    cat_chosen: dict[str, list[int]] = {
        "rain_onset": _greedy_non_overlapping(candidates, onset_scores, per_category, min_sep),
        "heavy_rain": _greedy_non_overlapping(candidates, heavy_scores, per_category, min_sep),
        "heatwave": _greedy_non_overlapping(candidates, heat_scores, per_category, min_sep),
        "cold_disturbance": _greedy_non_overlapping(cold_candidates, cold_scores, per_category, min_sep),
    }

    # Category 5: random (seeded, non-overlapping within random category)
    rng = np.random.default_rng(seed + (101 if split_name == "val" else 202))
    rand_scores = rng.random(len(candidates))
    cat_chosen["random"] = _greedy_non_overlapping(candidates, rand_scores, per_category, min_sep)

    windows: list[dict[str, Any]] = []
    shortfalls: dict[str, dict[str, int]] = {}
    for cat in CATEGORIES:
        picked = cat_chosen[cat]
        if len(picked) < per_category:
            shortfalls[cat] = {"requested": per_category, "found": len(picked), "shortfall": per_category - len(picked)}
        for t_last in picked:
            windows.append({"split": split_name, "category": cat, "t_last": int(t_last)})

    return windows, shortfalls


def select_heldout_windows(
    data: np.ndarray,
    doy: np.ndarray,
    clim_phys: np.ndarray,
    splits: dict[str, np.ndarray],
    n_in: int = 3,
    n_out: int = 10,
    per_category: int = 10,
    min_sep: int = 13,
    seed: int = 0,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, int]]]]:
    """Select tagged windows separately for 'val' and 'test' held-out splits."""
    out_windows: dict[str, list[dict[str, Any]]] = {}
    out_shortfalls: dict[str, dict[str, dict[str, int]]] = {}
    for split_name in ("val", "test"):
        w, sf = select_split_windows(
            data=data,
            doy=doy,
            clim_phys=clim_phys,
            split_idx=splits[split_name],
            split_name=split_name,
            n_in=n_in,
            n_out=n_out,
            per_category=per_category,
            min_sep=min_sep,
            seed=seed,
        )
        out_windows[split_name] = w
        out_shortfalls[split_name] = sf
    return out_windows, out_shortfalls


def deduplicate_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate windows sharing the same start index ``t_last``, preserving first-seen order."""
    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for w in windows:
        t = int(w["t_last"])
        if t not in seen:
            seen.add(t)
            unique.append(w)
    return unique
