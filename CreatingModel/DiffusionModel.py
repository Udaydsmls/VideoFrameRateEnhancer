import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding used to condition on a noise level."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    args = t.view(-1, 1) * freqs.view(1, -1)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DiffusionInterpolator(nn.Module):
    """Single-pass conditional denoiser conditioned on a scalar noise level."""

    def __init__(self, base_channels: int = 64, num_channels: int = 3, time_dim: int = 128) -> None:
        super().__init__()
        c = base_channels
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.in_conv = nn.Conv2d(2 * num_channels, c, 3, padding=1)
        self.down1 = _ResBlock(c, c * 2, time_dim)
        self.down2 = _ResBlock(c * 2, c * 4, time_dim)
        self.mid = _ResBlock(c * 4, c * 4, time_dim)
        self.up2 = _ResBlock(c * 4 + c * 2, c * 2, time_dim)
        self.up1 = _ResBlock(c * 2 + c, c, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, num_channels, 3, padding=1),
        )

    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor,
        noise_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = frame1.shape[0]
        if noise_level is None:
            noise_level = torch.zeros(batch, device=frame1.device, dtype=frame1.dtype)
        t_emb = self.time_mlp(sinusoidal_embedding(noise_level, self.time_dim))

        x = self.in_conv(torch.cat([frame1, frame2], dim=1))
        d1 = self.down1(F.avg_pool2d(x, 2), t_emb)
        d2 = self.down2(F.avg_pool2d(d1, 2), t_emb)
        m = self.mid(d2, t_emb)
        u2 = self.up2(torch.cat([F.interpolate(m, size=d1.shape[-2:], mode="nearest"), d1], dim=1), t_emb)
        u1 = self.up1(torch.cat([F.interpolate(u2, size=x.shape[-2:], mode="nearest"), x], dim=1), t_emb)
        return torch.sigmoid(self.out(u1))
