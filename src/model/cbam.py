"""
cbam.py
-------
Convolutional Block Attention Module (CBAM).

Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
https://arxiv.org/abs/1807.06521

CBAM applies two sequential attention gates:
  1. Channel attention  — "what" to focus on  (recalibrates channel weights)
  2. Spatial attention  — "where" to focus on (recalibrates spatial weights)

Both gates use global average pooling and global max pooling in parallel so the
module captures both the mean feature response and peak activations.

This implementation uses the standard CBAM design with:
  - Shared MLP weights for channel attention (avg + max → same FC layers)
  - 7×7 depthwise conv for spatial attention (large receptive field)
  - No bias in attention convolutions (attention maps should be zero-centered)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ChannelAttention(nn.Module):
    """
    Channel attention gate.

    Produces a per-channel scale vector in [0, 1] by:
      1. Squeezing spatial dimensions with avg-pool and max-pool.
      2. Passing each through a shared two-layer MLP (bottleneck).
      3. Summing the two responses, applying sigmoid.

    Parameters
    ----------
    in_channels     : Number of input feature channels C.
    reduction_ratio : MLP bottleneck = C // reduction_ratio.
                      Minimum bottleneck size is 4 to preserve expressiveness.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 8) -> None:
        super().__init__()
        bottleneck = max(4, in_channels // reduction_ratio)
        # Shared MLP — applied to both avg-pool and max-pool descriptors
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, in_channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        Tensor  (B, C, H, W) — channel-recalibrated features.
        """
        B, C, _, _ = x.shape

        # Global descriptors: (B, C)
        avg_desc = F.adaptive_avg_pool2d(x, 1).view(B, C)
        max_desc = F.adaptive_max_pool2d(x, 1).view(B, C)

        # Shared MLP + element-wise sum + sigmoid → scale: (B, C, 1, 1)
        scale = torch.sigmoid(
            self.shared_mlp(avg_desc) + self.shared_mlp(max_desc)
        ).view(B, C, 1, 1)

        return x * scale


class SpatialAttention(nn.Module):
    """
    Spatial attention gate.

    Produces a spatial map in [0, 1] by:
      1. Compressing channel dimension with avg and max pooling.
      2. Concatenating the two maps and passing through a conv layer.
      3. Applying sigmoid.

    Parameters
    ----------
    kernel_size : Conv kernel size for the spatial attention map.
                  Must be odd.  Default 7 (matches original CBAM paper).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        Tensor  (B, C, H, W) — spatially-recalibrated features.
        """
        # Channel descriptors at each spatial position: (B, 1, H, W) each
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values

        # Concatenate and convolve → spatial attention map: (B, 1, H, W)
        pool_cat = torch.cat([avg_map, max_map], dim=1)  # (B, 2, H, W)
        scale = torch.sigmoid(self.conv(pool_cat))        # (B, 1, H, W)

        return x * scale


class CBAM(nn.Module):
    """
    Full CBAM: channel attention followed by spatial attention.

    Drop-in refinement module — output has the same shape as input.

    Parameters
    ----------
    in_channels     : Number of feature channels.
    reduction_ratio : Bottleneck ratio for channel attention MLP.
    spatial_kernel  : Conv kernel size for spatial attention (must be odd).
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 8,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction_ratio)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        Tensor  (B, C, H, W) — channel- and spatially-recalibrated features.
        """
        x = self.channel(x)
        x = self.spatial(x)
        return x
