"""
loss.py
-------
Depth-stratified masked RMSE loss for OceanEmbed.

Motivation
----------
Ocean temperature reconstruction has very different difficulty by depth zone:

  Mixed layer (0–30 m)    — tightly coupled to SST; model learns it easily
  Thermocline (50–200 m)  — sharpest gradients; most sensitive to Ekman pumping
                            and mesoscale eddies; the hardest zone and the most
                            oceanographically important for the PS requirements
  Deep ocean (300–1000 m) — weak seasonal variability; model gets these
                            "almost for free" with little gradient signal

Weighting the thermocline more heavily pushes the model to spend capacity
where it is most needed, rather than optimising the easy surface layers.

Loss formula
------------
For each depth zone z (with index set I_z and weight w_z):

    RMSE_z = sqrt(  sum_{b,i,h,w} (pred - tgt)²·mask  /  (N_valid · |I_z|)  )

    L = sum_z ( w_z · RMSE_z ) / sum_z(w_z)

where N_valid = number of valid (ocean) pixels in the batch.

A small epsilon (1e-8) is added under the sqrt for numerical stability near
zero loss.  The eps is small enough to have negligible effect in practice.

Implementation note
-------------------
The function returns both the scalar loss (for .backward()) and a dict of
per-zone RMSE values (for logging / tensorboard) as a named tuple so callers
can unpack cleanly.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor


# ── Depth zone definitions ───────────────────────────────────────────────────
# These must be consistent with the 15 PS-standard depths in dataset.py.
# Standard depths: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
# Indices:          0  1   2   3   4   5   6    7    8    9   10   11   12   13   14

_DEPTH_ZONES: list[tuple[str, list[int]]] = [
    ("mixed_layer", [0, 1, 2, 3, 4]),           # 0–30 m
    ("thermocline",  [5, 6, 7, 8, 9, 10]),       # 50–200 m
    ("deep_ocean",   [11, 12, 13, 14]),           # 300–1000 m
]


class LossOutput(NamedTuple):
    """Return type of DepthStratifiedLoss.forward()."""
    total:       Tensor               # scalar loss for backprop
    mixed_layer: float                # RMSE of mixed-layer depths (normalised)
    thermocline: float                # RMSE of thermocline depths (normalised)
    deep_ocean:  float                # RMSE of deep-ocean depths (normalised)


class DepthStratifiedLoss(nn.Module):
    """
    Depth-stratified masked RMSE loss.

    Parameters
    ----------
    mixed_layer_weight : Loss weight for mixed-layer depths (0–30 m).
    thermocline_weight : Loss weight for thermocline depths (50–200 m).
    deep_ocean_weight  : Loss weight for deep-ocean depths (300–1000 m).
    eps                : Epsilon added under sqrt for numerical stability.
    """

    def __init__(
        self,
        mixed_layer_weight: float = 1.0,
        thermocline_weight: float = 2.0,
        deep_ocean_weight:  float = 0.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.eps = eps
        self._zone_weights = {
            "mixed_layer": mixed_layer_weight,
            "thermocline":  thermocline_weight,
            "deep_ocean":   deep_ocean_weight,
        }
        self._total_weight = sum(self._zone_weights.values())

    def forward(
        self,
        pred:   Tensor,
        target: Tensor,
        mask:   Tensor,
    ) -> LossOutput:
        """
        Parameters
        ----------
        pred   : (B, D, H, W) float32  — model predictions in normalised space.
        target : (B, D, H, W) float32  — GLORYS targets in normalised space.
        mask   : (B, H, W)    bool     — True = valid ocean pixel.

        Returns
        -------
        LossOutput  Named tuple: (total, mixed_layer, thermocline, deep_ocean).
                    total is the differentiable scalar for loss.backward().
                    Zone values are plain floats for logging.
        """
        mask_f  = mask.unsqueeze(1).float()                    # (B, 1, H, W)
        n_valid = mask_f.sum().clamp(min=1.0)                  # scalar

        total_loss = pred.new_zeros(1)
        zone_rmse: dict[str, float] = {}

        for zone_name, indices in _DEPTH_ZONES:
            weight = self._zone_weights[zone_name]
            n_idx  = len(indices)

            # Squared error, masked to ocean pixels  (B, n_idx, H, W)
            sq_err = (pred[:, indices] - target[:, indices]) ** 2 * mask_f

            # Mean over valid spatial pixels and depth indices in this zone
            zone_mse = sq_err.sum() / (n_valid * n_idx)

            # RMSE (with eps for stability) weighted by zone weight
            zone_rmse_val = torch.sqrt(zone_mse + self.eps)
            total_loss    = total_loss + weight * zone_rmse_val

            zone_rmse[zone_name] = float(zone_rmse_val.detach())

        total_loss = total_loss / self._total_weight

        return LossOutput(
            total       = total_loss,
            mixed_layer = zone_rmse["mixed_layer"],
            thermocline = zone_rmse["thermocline"],
            deep_ocean  = zone_rmse["deep_ocean"],
        )
