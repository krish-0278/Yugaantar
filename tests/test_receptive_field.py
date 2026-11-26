"""Receptive-field test proving per-step Chebyshev radius <= 3 at centre and corner pixels."""

import copy
import torch
import torch.nn as nn

from monsoon.model import MonsoonNet


def _check_pixel_radius(model: MonsoonNet, i0: int, j0: int, max_radius: int = 3) -> int:
    feat = torch.randn(1, 34, 32, 36, requires_grad=True)
    a_t = torch.randn(1, 6, 32, 36, requires_grad=True)
    a_next, event_logit = model(feat, a_t)
    target = a_next[0, :, i0, j0].sum() + event_logit[0, 0, i0, j0]
    target.backward()

    grad_feat = feat.grad[0].abs().sum(dim=0)  # [32, 36]
    grad_at = a_t.grad[0].abs().sum(dim=0)  # [32, 36]
    nonzero_mask = (grad_feat > 0) | (grad_at > 0)
    ii, jj = torch.where(nonzero_mask)
    assert len(ii) > 0, "Expected non-zero gradients after head re-initialization"
    dist = torch.maximum((ii - i0).abs(), (jj - j0).abs())
    max_dist = int(dist.max().item())
    assert max_dist <= max_radius, f"Pixel ({i0},{j0}) max nonzero gradient distance {max_dist} > {max_radius}"
    return max_dist


def test_receptive_field_centre_and_corner() -> None:
    torch.manual_seed(0)
    base = MonsoonNet(cin=34, cout=6, channels=96, n_blocks=2, expansion=2)
    model = copy.deepcopy(base)
    nn.init.normal_(model.head.weight, std=0.1)
    nn.init.normal_(model.head.bias, std=0.1)
    nn.init.normal_(model.event_head.weight, std=0.1)
    model.eval()

    d_centre = _check_pixel_radius(model, 16, 18, max_radius=3)
    d_corner = _check_pixel_radius(model, 0, 0, max_radius=3)
    print(f"Receptive field max Chebyshev distance: centre(16,18)={d_centre}, corner(0,0)={d_corner}")
    assert d_centre <= 3
    assert d_corner <= 3
