"""
normalize.py
------------
Z-score normalization utilities for OceanEmbed.

Statistics (mean, std) are computed from the TRAINING SET ONLY and then
applied uniformly to train / val / test splits.  This prevents data leakage
and ensures the test set is truly unseen.

Two sets of statistics are maintained:
  inputs  — per-channel stats (one entry per surface input channel)
  targets — per-depth-level stats (one entry per PS-standard depth)

Per-depth normalization for the target is important because deeper layers
have much smaller temperature variance than the mixed layer; uniform
normalization would cause the loss to be dominated by shallow depths.

The stats are saved to / loaded from a JSON file so they can be reproduced
exactly across runs without re-processing the training data.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np


# ── Stats I/O ─────────────────────────────────────────────────────────────────

def save_stats(stats: dict, path: pathlib.Path) -> None:
    """Serialize normalization stats to *path* as a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_stats(path: pathlib.Path) -> dict:
    """Load normalization stats from a JSON file produced by :func:`save_stats`."""
    if not path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {path}\n"
            "Run scripts/prepare_data.py first to generate them."
        )
    return json.loads(path.read_text())


# ── Stats computation ─────────────────────────────────────────────────────────

def compute_channel_stats(
    arr: np.ndarray,
    names: list[str],
    channel_axis: int = 1,
) -> dict[str, dict[str, float]]:
    """
    Compute per-channel mean and std, ignoring NaN values.

    Parameters
    ----------
    arr          : Array of shape (T, C, H, W) where C is along *channel_axis*.
    names        : Channel names, length C.
    channel_axis : Axis index for the channel dimension.

    Returns
    -------
    dict  { name: { "mean": float, "std": float } }
    """
    n_channels = arr.shape[channel_axis]
    if n_channels != len(names):
        raise ValueError(
            f"Expected {len(names)} channels, got {n_channels}"
        )

    stats: dict[str, dict[str, float]] = {}
    for c, name in enumerate(names):
        idx = [slice(None)] * arr.ndim
        idx[channel_axis] = c
        data  = arr[tuple(idx)]
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            raise ValueError(f"Channel '{name}' is entirely NaN — cannot compute stats.")
        stats[name] = {
            "mean": float(np.mean(valid)),
            "std":  float(np.std(valid)),
        }
    return stats


# ── Normalization / denormalization ───────────────────────────────────────────

def normalize(
    arr: np.ndarray,
    stats: dict[str, dict[str, float]],
    names: list[str],
    channel_axis: int = 1,
) -> np.ndarray:
    """
    Apply Z-score normalization channel-wise along *channel_axis*.

    Each channel c is normalized as:
        out[c] = (arr[c] - mean_c) / std_c

    NaN values (land, missing data) are propagated as-is.
    If std_c < 1e-8 (constant channel), std_c is treated as 1.0 to avoid /0.

    Parameters
    ----------
    arr          : Float array of shape (..., C, ...).
    stats        : Output of :func:`compute_channel_stats`.
    names        : Names for each position along *channel_axis*, length C.
    channel_axis : Axis index for the channel dimension.

    Returns
    -------
    np.ndarray  Same shape as *arr*, dtype float32.
    """
    out = arr.astype(np.float32).copy()
    for c, name in enumerate(names):
        idx = [slice(None)] * out.ndim
        idx[channel_axis] = c
        mean = np.float32(stats[name]["mean"])
        std  = np.float32(max(stats[name]["std"], 1e-8))
        out[tuple(idx)] = (out[tuple(idx)] - mean) / std
    return out


def denormalize(
    arr: np.ndarray,
    stats: dict[str, dict[str, float]],
    names: list[str],
    channel_axis: int = 1,
) -> np.ndarray:
    """
    Reverse Z-score normalization.  Inverse of :func:`normalize`.

    Parameters are identical to :func:`normalize`.

    Returns
    -------
    np.ndarray  Values in original physical units, dtype float32.
    """
    out = arr.astype(np.float32).copy()
    for c, name in enumerate(names):
        idx = [slice(None)] * out.ndim
        idx[channel_axis] = c
        mean = np.float32(stats[name]["mean"])
        std  = np.float32(max(stats[name]["std"], 1e-8))
        out[tuple(idx)] = out[tuple(idx)] * std + mean
    return out
