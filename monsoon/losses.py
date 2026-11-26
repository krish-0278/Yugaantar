"""Loss functions for Stage A one-step training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


CH_TP = 1


def lat_weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_w: torch.Tensor,
    tp_true_phys: torch.Tensor,
    delta: float = 1.0,
    rain_threshold_mm: float = 10.0,
    tp_tail_weight: float = 3.0,
) -> torch.Tensor:
    """Latitude-weighted Huber loss in normalized anomaly space with heavy-rain tail weighting."""
    err = pred - target
    abs_err = torch.abs(err)
    huber = torch.where(abs_err <= delta, 0.5 * err * err, delta * (abs_err - 0.5 * delta))

    weights = lat_w.view(1, 1, -1, 1).expand_as(huber).clone()
    tail_mult = 1.0 + tp_tail_weight * (tp_true_phys >= rain_threshold_mm).to(huber.dtype)
    weights[:, CH_TP, :, :] = weights[:, CH_TP, :, :] * tail_mult
    return (huber * weights).mean()


def gradient_diff_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lat_w: torch.Tensor,
) -> torch.Tensor:
    """Latitude-weighted spatial gradient-difference L1 loss along y and x."""
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_true = target[:, :, 1:, :] - target[:, :, :-1, :]
    w_y = 0.5 * (lat_w[1:] + lat_w[:-1])
    loss_y = (torch.abs(dy_pred - dy_true) * w_y.view(1, 1, -1, 1)).mean()

    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_true = target[:, :, :, 1:] - target[:, :, :, :-1]
    loss_x = (torch.abs(dx_pred - dx_true) * lat_w.view(1, 1, -1, 1)).mean()
    return 0.5 * (loss_y + loss_x)


class StageALoss(nn.Module):
    """Combined Stage A loss: Huber + 0.2 * BCE(event) + 0.05 * GradDiff."""

    def __init__(
        self,
        lat_w: torch.Tensor,
        pos_weight: float,
        huber_delta: float = 1.0,
        rain_threshold_mm: float = 10.0,
        tp_tail_weight: float = 3.0,
        bce_weight: float = 0.2,
        grad_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer("lat_w", lat_w.clone().float())
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32))
        self.huber_delta = float(huber_delta)
        self.rain_threshold_mm = float(rain_threshold_mm)
        self.tp_tail_weight = float(tp_tail_weight)
        self.bce_weight = float(bce_weight)
        self.grad_weight = float(grad_weight)

    def forward(
        self,
        pred_anom: torch.Tensor,
        event_logit: torch.Tensor,
        target_anom: torch.Tensor,
        event_target: torch.Tensor,
        tp_true_phys: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        huber = lat_weighted_huber_loss(
            pred_anom,
            target_anom,
            self.lat_w,
            tp_true_phys,
            delta=self.huber_delta,
            rain_threshold_mm=self.rain_threshold_mm,
            tp_tail_weight=self.tp_tail_weight,
        )
        bce = F.binary_cross_entropy_with_logits(event_logit, event_target, pos_weight=self.pos_weight)
        grad = gradient_diff_loss(pred_anom, target_anom, self.lat_w)
        total = huber + self.bce_weight * bce + self.grad_weight * grad
        parts = {
            "total": float(total.detach().item()),
            "huber": float(huber.detach().item()),
            "bce": float(bce.detach().item()),
            "grad": float(grad.detach().item()),
        }
        return total, parts
