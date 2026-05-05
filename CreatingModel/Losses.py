import torch
from torch import nn
from torch.nn import functional as F


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    kernel_2d = g.t() @ g
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Mean structural similarity between two tensors of shape (B, C, H, W)."""
    channels = pred.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, channels, pred.device, pred.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(pred, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=channels)
    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(pred * pred, kernel, padding=pad, groups=channels) - mu_xx
    sigma_yy = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu_yy
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    return (numerator / denominator).mean()


class L1SSIMLoss(nn.Module):
    """L1 + (1 - SSIM); keeps sharpness while preserving overall fidelity."""

    def __init__(self, ssim_weight: float = 0.5) -> None:
        super().__init__()
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target) + self.ssim_weight * (1 - ssim(pred, target))
