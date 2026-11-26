"""Unit tests for MonsoonNet shapes, zero-init behavior, parameter count, and numerical stability."""

import torch

from monsoon.model import MonsoonNet


def test_model_shapes_and_zero_init() -> None:
    torch.manual_seed(0)
    model = MonsoonNet(cin=34, cout=6, channels=96, n_blocks=2, expansion=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MonsoonNet parameter count: {n_params}")
    assert model.radius == 3

    feat = torch.randn(4, 34, 32, 36)
    a_t = torch.randn(4, 6, 32, 36)
    a_next, event_logit = model(feat, a_t)

    assert a_next.shape == (4, 6, 32, 36)
    assert event_logit.shape == (4, 1, 32, 36)

    rho = torch.sigmoid(model.rho_logit).view(1, -1, 1, 1)
    expected = rho * a_t
    assert torch.equal(a_next, expected)


def test_model_no_nan_100_batches() -> None:
    torch.manual_seed(1)
    model = MonsoonNet(cin=34, cout=6, channels=96, n_blocks=2, expansion=2)
    model.eval()
    with torch.no_grad():
        for _ in range(100):
            feat = torch.randn(2, 34, 32, 36) * 2.0
            a_t = torch.randn(2, 6, 32, 36) * 2.0
            a_next, event_logit = model(feat, a_t)
            assert torch.isfinite(a_next).all()
            assert torch.isfinite(event_logit).all()
