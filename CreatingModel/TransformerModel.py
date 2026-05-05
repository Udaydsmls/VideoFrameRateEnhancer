import torch
from torch import nn
from torch.nn import functional as F


def _positional_embedding(length: int, dim: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(dim, device=device, dtype=dtype).unsqueeze(0)
    angle = pos / torch.pow(10000.0, (2 * (i // 2)) / dim)
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe.unsqueeze(0)


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.body = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


class TransformerInterpolator(nn.Module):
    """ViT-style frame interpolator with a convolutional skip path."""

    def __init__(
        self,
        num_channels: int = 3,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 6,
        patch_size: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.stem = nn.Conv2d(2 * num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(
            [_Block(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
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
        seq = seq + _positional_embedding(seq.shape[1], self.embed_dim, seq.device, seq.dtype)
        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq)
        feat = seq.transpose(1, 2).reshape(b, self.embed_dim, gh, gw)
        feat = F.interpolate(feat, size=x_padded.shape[-2:], mode="bilinear", align_corners=False)

        out = self.head(torch.cat([feat, self.skip_conv(x_padded)], dim=1))
        if ph or pw:
            out = out[..., :h, :w]
        return torch.sigmoid(out)
