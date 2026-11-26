"""MonsoonNet local-receptive-field residual architecture (step radius <= 3 cells)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LN2d(nn.Module):
    """LayerNorm over channels per pixel only (no spatial mixing)."""

    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class Block(nn.Module):
    """Residual local conv block: out = x + proj(GELU(expand(LN2d(depthwise(x)))))."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
            groups=channels,
        )
        self.norm = LN2d(channels)
        self.expand = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(self.act(self.expand(self.norm(self.depthwise(x)))))


class MonsoonNet(nn.Module):
    """Damped-persistence + strictly local conv innovation network with heavy-rain event head."""

    def __init__(
        self,
        cin: int = 34,
        cout: int = 6,
        channels: int = 96,
        n_blocks: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.channels = channels
        self.n_blocks = n_blocks
        self.expansion = expansion

        self.stem = nn.Conv2d(cin, channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.blocks = nn.Sequential(*[Block(channels, expansion=expansion) for _ in range(n_blocks)])
        self.head = nn.Conv2d(channels, cout, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.event_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.rho_logit = nn.Parameter(torch.full((cout,), 2.0))
        self.scale = nn.Parameter(torch.full((cout,), 0.5))

    @property
    def radius(self) -> int:
        """Per-step Chebyshev receptive radius in grid cells (1 + n_blocks = 3)."""
        return 1 + len(self.blocks)

    def forward(self, feat: torch.Tensor, a_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next-day normalized anomalies [B, 6, H, W] and heavy-rain event logits [B, 1, H, W]."""
        h = self.blocks(self.stem(feat))
        rho = torch.sigmoid(self.rho_logit).view(1, -1, 1, 1)
        scale = self.scale.view(1, -1, 1, 1)
        a_next = rho * a_t + scale * self.head(h)
        event_logit = self.event_head(h)
        return a_next, event_logit
