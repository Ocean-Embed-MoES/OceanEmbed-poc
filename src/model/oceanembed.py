"""
oceanembed.py
-------------
OceanEmbed model — CBAM-augmented U-Net for subsurface ocean temperature
reconstruction from multi-channel surface observations.

Architecture (PoC scale)
------------------------

    Input:  (B, T*C_in, H, W) = (B, 24, 50, 120)
            24 channels = 3-day window × 8 surface channels

    Stem:   1×1 conv  →  (B, 32, 50, 120)
            Projects the stacked temporal channels to the encoder's
            first feature dimension.

    Encoder (3 stages, each = double conv + CBAM + MaxPool):
            enc1:  (B,  32, 50, 120) → skip1, down → (B,  32, 25, 60)
            enc2:  (B,  64, 25,  60) → skip2, down → (B,  64, 12, 30)
            enc3:  (B, 128, 12,  30) → skip3, down → (B, 128,  6, 15)

    Bottleneck (double conv, no pool):
            (B, 128, 6, 15) → (B, 128, 6, 15)

    Decoder (3 stages, each = bilinear upsample + skip cat + double conv + CBAM):
            dec1:  upsample(6→12) + skip3 → (B,  64, 12, 30)
            dec2:  upsample(12→25) + skip2 → (B,  32, 25, 60)
            dec3:  upsample(25→50) + skip1 → (B,  32, 50, 120)

    Head:   1×1 conv  →  (B, 15, 50, 120)
            One output channel per PS-standard depth level.
            No output activation — predictions are in normalized space.

Full design comparison
----------------------
    Dimension       Full design         PoC (this file)
    ─────────────────────────────────────────────────────
    Grid            0.25°, 100×240      0.5°, 50×120
    Input channels  12 (+ static)       8 × 3-day window → 24
    Temporal window 7 days              3 days
    Enc channels    64→128→256→512      32→64→128
    Bottleneck      512                 128
    Depth outputs   15                  15 (unchanged — PS requirement)
    Attention       CBAM everywhere     CBAM everywhere

Parameter count (approximate): ~450 k
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.model.unet import Bottleneck, DecoderBlock, EncoderBlock


class OceanEmbedModel(nn.Module):
    """
    CBAM-augmented U-Net for subsurface ocean temperature reconstruction.

    Parameters
    ----------
    in_channels  : Total input channels = temporal_window × surface_channels.
                   Default 24 (= 3 days × 8 channels).
    enc_channels : Channel progression for the three encoder stages.
                   Default [32, 64, 128].
    n_depths     : Number of output depth levels.
                   Default 15 (PS-standard: 0–1000 m).
    """

    def __init__(
        self,
        in_channels: int = 24,
        enc_channels: list[int] = None,
        n_depths: int = 15,
    ) -> None:
        super().__init__()
        if enc_channels is None:
            enc_channels = [32, 64, 128]

        e1, e2, e3 = enc_channels

        # ── Stem: project input channels to enc1 feature space ────────────────
        # Using 1×1 conv so the spatial information is untouched.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, e1, kernel_size=1, bias=False),
            nn.BatchNorm2d(e1),
            nn.ReLU(inplace=True),
        )

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = EncoderBlock(e1, e1)   # 32 → 32
        self.enc2 = EncoderBlock(e1, e2)   # 32 → 64
        self.enc3 = EncoderBlock(e2, e3)   # 64 → 128

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = Bottleneck(e3, e3)   # 128 → 128

        # ── Decoder ───────────────────────────────────────────────────────────
        # dec1: from bottleneck (e3=128) + skip3 (e3=128) → e2=64
        # dec2: from dec1 (e2=64)        + skip2 (e2=64)  → e1=32
        # dec3: from dec2 (e1=32)        + skip1 (e1=32)  → e1=32
        self.dec1 = DecoderBlock(in_ch=e3, skip_ch=e3, out_ch=e2)
        self.dec2 = DecoderBlock(in_ch=e2, skip_ch=e2, out_ch=e1)
        self.dec3 = DecoderBlock(in_ch=e1, skip_ch=e1, out_ch=e1)

        # ── Output head ───────────────────────────────────────────────────────
        # 1×1 conv maps e1 feature channels to n_depths prediction channels.
        # No activation: predictions are in normalized (Z-score) space and
        # can be negative.
        self.head = nn.Conv2d(e1, n_depths, kernel_size=1)

        # Store config for repr / checkpointing
        self.in_channels  = in_channels
        self.enc_channels = enc_channels
        self.n_depths     = n_depths

        # Weight initialisation
        self._init_weights()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)
            Normalized surface input window.

        Returns
        -------
        Tensor  (B, n_depths, H, W)
            Predicted subsurface temperature (normalized).
        """
        # Stem
        x = self.stem(x)                        # (B, e1, H, W)

        # Encode
        skip1, x = self.enc1(x)                 # skip1: (B, e1, H,   W)
        skip2, x = self.enc2(x)                 # skip2: (B, e2, H/2, W/2)
        skip3, x = self.enc3(x)                 # skip3: (B, e3, H/4, W/4)

        # Bottleneck
        x = self.bottleneck(x)                  # (B, e3, H/4, W/4)

        # Decode (skip connections added in reverse order)
        x = self.dec1(x, skip3)                 # (B, e2, H/4, W/4) → H/4
        x = self.dec2(x, skip2)                 # (B, e1, H/2, W/2)
        x = self.dec3(x, skip1)                 # (B, e1, H,   W)

        # Output
        return self.head(x)                     # (B, n_depths, H, W)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self) -> None:
        """
        Kaiming He initialisation for Conv2d layers (suited for ReLU networks).
        BatchNorm initialized with weight=1, bias=0 (default, made explicit).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def __repr__(self) -> str:
        return (
            f"OceanEmbedModel(\n"
            f"  in_channels={self.in_channels},\n"
            f"  enc_channels={self.enc_channels},\n"
            f"  n_depths={self.n_depths},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )
