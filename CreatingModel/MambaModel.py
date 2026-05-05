import torch
from torch import nn
from torch.nn import functional as F


class _SelectiveSSM(nn.Module):
    """Selective state-space scan along a sequence axis."""

    def __init__(self, dim: int, state_dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dt_proj = nn.Linear(dim, dim)
        self.B_proj = nn.Linear(dim, state_dim)
        self.C_proj = nn.Linear(dim, state_dim)
        log_a = torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
        self.A_log = nn.Parameter(log_a.repeat(dim, 1))
        self.D = nn.Parameter(torch.ones(dim))
        self.out_proj = nn.Linear(dim, dim)

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        x_in, gate = self.in_proj(x).chunk(2, dim=-1)
        x_in = F.silu(x_in)
        gate = F.silu(gate)

        delta = F.softplus(self.dt_proj(x_in))
        a = -torch.exp(self.A_log).unsqueeze(0).unsqueeze(0)
        a_bar = torch.exp(delta.unsqueeze(-1) * a)
        b = self.B_proj(x_in).unsqueeze(2)
        b_bar = delta.unsqueeze(-1) * b
        u = x_in.unsqueeze(-1)
        bu = b_bar * u

        h = torch.zeros_like(bu[:, 0])
        outputs = []
        for t in range(bu.shape[1]):
            h = a_bar[:, t] * h + bu[:, t]
            outputs.append(h)
        h_seq = torch.stack(outputs, dim=1)

        c = self.C_proj(x_in).unsqueeze(2)
        y = (h_seq * c).sum(dim=-1) + self.D * x_in
        return self.out_proj(y * gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward = self._scan(x)
        backward = self._scan(torch.flip(x, dims=[1]))
        return forward + torch.flip(backward, dims=[1])


class _MambaBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ssm = _SelectiveSSM(dim, state_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class MambaInterpolator(nn.Module):
    """Bidirectional Mamba-style frame interpolator (pure PyTorch)."""

    def __init__(
        self,
        num_channels: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        patch_size: int = 8,
        state_dim: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stem = nn.Conv2d(2 * num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(
            [_MambaBlock(embed_dim, state_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.skip_conv = nn.Conv2d(2 * num_channels, embed_dim // 2, 3, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(embed_dim + embed_dim // 2, embed_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, num_channels, 3, padding=1),
        )

    def forward(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([frame1, frame2], dim=1)
        b, _, h, w = x.shape
        ph = (self.patch_size - h % self.patch_size) % self.patch_size
        pw = (self.patch_size - w % self.patch_size) % self.patch_size
        x_padded = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x

        tokens = self.stem(x_padded)
        gh, gw = tokens.shape[-2:]
        seq = tokens.flatten(2).transpose(1, 2)
        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq)
        feat = seq.transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
        feat = F.interpolate(feat, size=x_padded.shape[-2:], mode="bilinear", align_corners=False)

        out = self.head(torch.cat([feat, self.skip_conv(x_padded)], dim=1))
        if ph or pw:
            out = out[..., :h, :w]
        return torch.sigmoid(out)
