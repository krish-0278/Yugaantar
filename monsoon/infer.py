"""Read-only inference simulator converting physical inputs to physical rollouts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from monsoon.model import MonsoonNet
from monsoon.postproc import apply as apply_postproc, load_postproc
from monsoon.rollout import load_rollout_context, rollout
from monsoon.shock import apply_shock_response, is_zero_perturbation


ROOT = Path(__file__).resolve().parent.parent


def decode_doy_last(doy_sincos: np.ndarray | list[float] | tuple[float, float]) -> int:
    """Decode [sin(2*pi*doy/365), cos(2*pi*doy/365)] -> integer doy in 1..365."""
    s = float(doy_sincos[0])
    c = float(doy_sincos[1])
    ang = math.atan2(s, c) % (2.0 * math.pi)
    doy = int(round(ang * 365.0 / (2.0 * math.pi)))
    return 365 if doy == 0 or doy >= 365 else doy


class Simulator:
    """CPU-only inference simulator loading all weights and static fields once at startup."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        num_threads: int = 4,
        *,
        use_unshrunk_shock: bool = True,
    ) -> None:
        self.root = Path(root_dir).resolve() if root_dir is not None else ROOT
        torch.set_num_threads(int(num_threads))
        self.use_unshrunk_shock = bool(use_unshrunk_shock)

        with (self.root / "config" / "default.json").open("r", encoding="utf-8") as f:
            self.cfg: dict[str, Any] = json.load(f)

        weights_dir = self.root / "weights"
        self.ctx = load_rollout_context(
            harmonics_path=weights_dir / "clim_harmonics.npz",
            norm_path=weights_dir / "norm.npz",
            static_path=weights_dir / "static.npz",
        )
        self.postproc_cfg = load_postproc(weights_dir / "postproc.json")
        with (weights_dir / "channel_std_phys.json").open("r", encoding="utf-8") as f:
            self.channel_std_phys = np.asarray(json.load(f)["std"], dtype=np.float32)

        ckpt = torch.load(weights_dir / "model.pt", map_location="cpu", weights_only=False)
        mc = ckpt["config"]["model"]
        self.model = MonsoonNet(
            cin=int(mc["cin"]),
            cout=int(mc["cout"]),
            channels=int(mc["channels"]),
            n_blocks=int(mc["n_blocks"]),
            expansion=int(mc["expansion"]),
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.anom_clamp = float(self.cfg["stage_b"]["anom_clamp"])
        self.input_clip = float(self.cfg["norm"]["anomaly_clip"])
        shock_cfg = self.cfg["shock"]
        self.cone_speed = float(shock_cfg["cone_speed"])
        self.taper = float(shock_cfg["taper"])
        self.cap_sigma = float(shock_cfg["cap_sigma"])
        self.zero_eps = float(shock_cfg["zero_eps"])
        self.t2m_min_k = float(self.postproc_cfg.get("t2m_min_k", 233.15))
        self.t2m_max_k = float(self.postproc_cfg.get("t2m_max_k", 333.0))

    def simulate(
        self,
        init_states: np.ndarray,
        doy_sincos: np.ndarray | list[float] | tuple[float, float],
        steps: int,
        perturbation: np.ndarray | None = None,
        *,
        use_cone: bool = True,
        use_cap: bool = True,
        use_unshrunk_shock: bool | None = None,
    ) -> np.ndarray:
        """Run deterministic rollout in physical units [steps, 6, 32, 36] float32."""
        init_arr = np.asarray(init_states, dtype=np.float32)
        if init_arr.shape != (3, 6, 32, 36):
            raise ValueError(f"Expected init_states shape (3, 6, 32, 36), got {init_arr.shape}")
        if not np.isfinite(init_arr).all():
            raise ValueError("init_states contains non-finite values")

        sc_arr = np.asarray(doy_sincos, dtype=np.float64)
        if sc_arr.shape != (2,) or not np.isfinite(sc_arr).all():
            raise ValueError("doy_sincos must be 2 finite floats")

        steps_int = int(steps)
        if steps_int < 1:
            raise ValueError(f"steps must be >= 1, got {steps_int}")

        doy_last = decode_doy_last(sc_arr)

        with torch.inference_mode():
            anom_ctrl, ev_ctrl = rollout(
                self.model,
                init_arr,
                doy_last,
                steps_int,
                delta=None,
                ctx=self.ctx,
                anom_clamp=self.anom_clamp,
                input_clip=self.input_clip,
            )
            control = apply_postproc(
                anom_ctrl,
                ev_ctrl,
                doy_last,
                steps_int,
                postproc_cfg=self.postproc_cfg,
                ctx=self.ctx,
            ).astype(np.float32)

            if is_zero_perturbation(perturbation, zero_eps=self.zero_eps):
                return control

            pert_arr = np.asarray(perturbation, dtype=np.float32)
            if pert_arr.shape != (6, 32, 36) or not np.isfinite(pert_arr).all():
                raise ValueError("perturbation must have finite shape (6, 32, 36)")

            anom_pert, ev_pert = rollout(
                self.model,
                init_arr,
                doy_last,
                steps_int,
                delta=pert_arr,
                ctx=self.ctx,
                anom_clamp=self.anom_clamp,
                input_clip=self.input_clip,
            )

            unshrunk = self.use_unshrunk_shock if use_unshrunk_shock is None else bool(use_unshrunk_shock)
            if unshrunk:
                # V1: shock response from dynamical core without beta shrinkage, added to served control.
                ctrl_nb = apply_postproc(
                    anom_ctrl,
                    ev_ctrl,
                    doy_last,
                    steps_int,
                    postproc_cfg=self.postproc_cfg,
                    ctx=self.ctx,
                    use_beta=False,
                ).astype(np.float32)
                pert_nb = apply_postproc(
                    anom_pert,
                    ev_pert,
                    doy_last,
                    steps_int,
                    postproc_cfg=self.postproc_cfg,
                    ctx=self.ctx,
                    use_beta=False,
                ).astype(np.float32)
                return apply_shock_response(
                    control,
                    pert_nb,
                    pert_arr,
                    self.channel_std_phys,
                    baseline_for_resp=ctrl_nb,
                    cone_speed=self.cone_speed,
                    taper=self.taper,
                    cap_sigma=self.cap_sigma,
                    t2m_min_k=self.t2m_min_k,
                    t2m_max_k=self.t2m_max_k,
                    use_cone=use_cone,
                    use_cap=use_cap,
                )

            pert = apply_postproc(
                anom_pert,
                ev_pert,
                doy_last,
                steps_int,
                postproc_cfg=self.postproc_cfg,
                ctx=self.ctx,
            ).astype(np.float32)

            return apply_shock_response(
                control,
                pert,
                pert_arr,
                self.channel_std_phys,
                cone_speed=self.cone_speed,
                taper=self.taper,
                cap_sigma=self.cap_sigma,
                t2m_min_k=self.t2m_min_k,
                t2m_max_k=self.t2m_max_k,
                use_cone=use_cone,
                use_cap=use_cap,
            )
