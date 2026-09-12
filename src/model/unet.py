"""
unet.py
-------
U-Net encoder and decoder building blocks for OceanEmbed.

Design
------
Each encoder block:
    Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU → CBAM → MaxPool2d

Each decoder block:
    Bilinear upsample (to skip's exact spatial size)
    → concat skip connection
    → Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU → CBAM

Using bilinear upsample + size matching (rather than ConvTranspose2d) avoids
checkerboard artefacts and handles non-power-of-2 spatial dimensions cleanly.
The NIO PoC grid (50 × 120) produces 25→12→6 under max pooling; interpolating
to skip.shape[2:] restores the exact upstream size without any cropping/padding.

Bottleneck:
    Conv(3×3) → BN → ReLU → Conv(3×3) → BN → ReLU
    (same channel count in and out — no CBAM needed, no pooling)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.model.cbam import CBAM


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Two consecutive Conv2d→BatchNorm2d→ReLU layers with 3×3 kernels.

    Bias is disabled because it is redundant with BatchNorm's learned shift.
    Padding=1 preserves spatial dimensions.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class EncoderBlock(nn.Module):
    """
    One encoder stage: double conv → CBAM → MaxPool.

    Returns
    -------
    skip : Tensor  (B, out_ch, H, W)    Feature map before downsampling.
                   Used as the skip connection for the matching decoder block.
    down : Tensor  (B, out_ch, H/2, W/2)  Downsampled features passed deeper.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        reduction_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.conv = _double_conv(in_ch, out_ch)
        self.cbam = CBAM(out_ch, reduction_ratio=reduction_ratio)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.cbam(self.conv(x))  # (B, out_ch, H, W)
        down = self.pool(skip)           # (B, out_ch, H//2, W//2)
        return skip, down


class Bottleneck(nn.Module):
    """
    Deepest layer — double conv at the coarsest spatial resolution.
    No pooling; no CBAM (spatial size already small).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _double_conv(in_ch, out_ch)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """
    One decoder stage: upsample → concat skip → double conv → CBAM.

    The upsample step uses bilinear interpolation to the exact spatial size
    of the corresponding skip connection.  This cleanly handles the case where
    MaxPool(25) = 12 (floor division), so upsample(12) × 2 = 24 ≠ 25.

    Parameters
    ----------
    in_ch    : Channels coming from the previous (deeper) decoder or bottleneck.
    skip_ch  : Channels coming from the encoder skip connection.
    out_ch   : Output channels of this decoder block.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        reduction_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.conv = _double_conv(in_ch + skip_ch, out_ch)
        self.cbam = CBAM(out_ch, reduction_ratio=reduction_ratio)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x    : (B, in_ch,   H_low, W_low)   Upward-flowing features.
        skip : (B, skip_ch, H,     W    )   Encoder skip connection.

        Returns
        -------
        Tensor  (B, out_ch, H, W)
        """
        # Upsample to exactly match skip's spatial dims
        x = F.interpolate(
            x, size=skip.shape[2:], mode="bilinear", align_corners=False
        )
        x = torch.cat([x, skip], dim=1)   # (B, in_ch + skip_ch, H, W)
        return self.cbam(self.conv(x))
