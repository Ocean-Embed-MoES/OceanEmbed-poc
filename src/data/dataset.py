"""
dataset.py
----------
PyTorch Dataset for OceanEmbed PoC.

Reads preprocessed NetCDF files (produced by scripts/prepare_data.py) and
provides rolling temporal windows of normalized surface inputs paired with
the GLORYS subsurface temperature target at the 15 PS-standard depth levels.

Expected file layout (produced by prepare_data.py)
---------------------------------------------------
  outputs/processed/
    train/
      inputs_2015.nc    (time, channel, latitude, longitude)  physical units
      targets_2015.nc   (time, depth,   latitude, longitude)  °C
      ...
    val/
      inputs_2018.nc
      targets_2018.nc
    test/
      inputs_2019.nc
      targets_2019.nc
      inputs_2020.nc
      targets_2020.nc
    norm_stats.json       Z-score stats computed from training set

Each sample
-----------
  x    : (temporal_window × n_channels, H, W)  e.g. (24, 50, 120)
          Oldest day first, newest last along the channel axis.
  y    : (n_depths, H, W)                       e.g. (15, 50, 120)
          Normalized GLORYS temperature at PS-standard depths.
  mask : (H, W)  bool
          True for valid ocean pixels.  Used by the loss to exclude land.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from src.preprocessing.normalize import load_stats, normalize


# ── Channel / depth metadata — must match prepare_data.py exactly ─────────────
CHANNEL_NAMES = [
    "sst", "grad_sst", "sla", "grad_sla",
    "sss", "wsc", "u_curr", "v_curr",
]
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_LABELS    = [f"d{d}m" for d in STANDARD_DEPTHS]


class OceanEmbedDataset(Dataset):
    """
    Rolling-window dataset for OceanEmbed PoC.

    Loads preprocessed data from disk, applies Z-score normalization,
    and returns (input_window, target, ocean_mask) triplets.

    Parameters
    ----------
    split           : One of "train", "val", "test".
    processed_dir   : Root directory that contains split sub-folders and
                      norm_stats.json.  Relative paths are resolved from
                      the project root (the directory containing this file).
    temporal_window : Number of consecutive daily snapshots stacked along
                      the channel axis.  Must match the value used by the
                      model's temporal aggregation layer.
    """

    def __init__(
        self,
        split: str,
        processed_dir: pathlib.Path | str,
        temporal_window: int = 3,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test' — got '{split}'")

        self.split    = split
        self.proc_dir = pathlib.Path(processed_dir)
        self.tw       = temporal_window

        # ── Load normalization stats (computed on training set only) ──────────
        stats_path = self.proc_dir / "norm_stats.json"
        all_stats  = load_stats(stats_path)
        self._input_stats  = {k: all_stats["inputs"][k]  for k in CHANNEL_NAMES}
        self._target_stats = {k: all_stats["targets"][k] for k in DEPTH_LABELS}

        # ── Load and concatenate all preprocessed years for this split ─────────
        split_dir    = self.proc_dir / split
        input_files  = sorted(split_dir.glob("inputs_*.nc"))
        target_files = sorted(split_dir.glob("targets_*.nc"))

        if not input_files:
            raise FileNotFoundError(
                f"No preprocessed input files found in {split_dir}.\n"
                "Run `python scripts/prepare_data.py` first."
            )
        if len(input_files) != len(target_files):
            raise RuntimeError(
                f"Mismatch: {len(input_files)} input files vs "
                f"{len(target_files)} target files in {split_dir}."
            )

        inp_ds = xr.open_mfdataset(input_files,  combine="by_coords")
        tgt_ds = xr.open_mfdataset(target_files, combine="by_coords")

        # Shape: (T, C, H, W) and (T, D, H, W)
        raw_inputs  = inp_ds["inputs"].values.astype(np.float32)
        raw_targets = tgt_ds["targets"].values.astype(np.float32)

        T_in, T_tgt = raw_inputs.shape[0], raw_targets.shape[0]
        if T_in != T_tgt:
            raise RuntimeError(
                f"Input time length ({T_in}) != target time length ({T_tgt})."
            )

        # ── Build land/ocean mask ─────────────────────────────────────────────
        # A pixel is ocean if at least one depth level is non-NaN in the target.
        # We compute this before NaN→0 replacement so the mask is clean.
        self.mask = np.any(~np.isnan(raw_targets[0]), axis=0)  # (H, W)  bool

        # ── Replace NaN with 0.0 before normalization ─────────────────────────
        # The loss function uses the mask to ignore land pixels, so filling
        # land NaNs with 0 is safe and avoids propagating NaN through the model.
        raw_inputs  = np.nan_to_num(raw_inputs,  nan=0.0)
        raw_targets = np.nan_to_num(raw_targets, nan=0.0)

        # ── Apply Z-score normalization ────────────────────────────────────────
        self.inputs  = normalize(raw_inputs,  self._input_stats,  CHANNEL_NAMES, channel_axis=1)
        self.targets = normalize(raw_targets, self._target_stats, DEPTH_LABELS,  channel_axis=1)

        self.T = T_in
        # First valid index needs tw-1 preceding days
        self._valid = list(range(self.tw - 1, self.T))

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x    : float32 tensor  (tw × C, H, W)  — normalized surface window
        y    : float32 tensor  (D, H, W)        — normalized subsurface target
        mask : bool tensor     (H, W)           — True = valid ocean pixel
        """
        t = self._valid[idx]

        # Stack temporal window: days [t-tw+1, ..., t], oldest first
        window = self.inputs[t - self.tw + 1 : t + 1]   # (tw, C, H, W)
        x = window.reshape(-1, *window.shape[2:])         # (tw*C, H, W)

        y    = self.targets[t]   # (D, H, W)
        mask = self.mask         # (H, W)

        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.copy()),
            torch.from_numpy(mask),
        )

    # ── Convenience properties ─────────────────────────────────────────────────

    @property
    def n_input_channels(self) -> int:
        """Total input channels = temporal_window × surface_channels."""
        return self.tw * len(CHANNEL_NAMES)

    @property
    def n_depths(self) -> int:
        """Number of output depth levels."""
        return len(DEPTH_LABELS)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """(H, W) grid shape."""
        return self.inputs.shape[2], self.inputs.shape[3]

    def __repr__(self) -> str:
        H, W = self.spatial_shape
        return (
            f"OceanEmbedDataset(split='{self.split}', "
            f"samples={len(self)}, "
            f"x_shape=({self.n_input_channels},{H},{W}), "
            f"y_shape=({self.n_depths},{H},{W}))"
        )
