"""
metrics.py
----------
Pure evaluation metric functions for OceanEmbed PoC validation.

All functions take flat 1-D arrays of predictions and observations
(already masked and depth-selected) and return scalar floats.

Metric suite
------------
  rmse         : Root Mean Squared Error (°C)
  bias         : Mean systematic error — positive = warm bias (°C)
  r_squared    : Coefficient of determination R² (dimensionless)
  pearson_r    : Pearson linear correlation coefficient (dimensionless)

depth_profile_metrics()
  Compute the full suite at every depth level given (N, D, H, W) arrays.
  Returns a nested dict keyed by depth label for easy JSON serialisation.
"""
from __future__ import annotations

import numpy as np


# ── Scalar metrics ─────────────────────────────────────────────────────────────

def rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    """Root Mean Squared Error (°C). Both arrays should be 1-D."""
    return float(np.sqrt(np.nanmean((pred - obs) ** 2)))


def bias(pred: np.ndarray, obs: np.ndarray) -> float:
    """Mean bias: pred − obs.  Positive = model runs warm (°C)."""
    return float(np.nanmean(pred - obs))


def r_squared(pred: np.ndarray, obs: np.ndarray) -> float:
    """Coefficient of determination R² = 1 − SS_res / SS_tot."""
    ss_res = np.nansum((obs - pred) ** 2)
    ss_tot = np.nansum((obs - np.nanmean(obs)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def pearson_r(pred: np.ndarray, obs: np.ndarray) -> float:
    """Pearson linear correlation coefficient."""
    mask = np.isfinite(pred) & np.isfinite(obs)
    if mask.sum() < 2:
        return float("nan")
    p, o = pred[mask], obs[mask]
    cov  = np.mean((p - p.mean()) * (o - o.mean()))
    denom = p.std() * o.std()
    if denom < 1e-12:
        return float("nan")
    return float(cov / denom)


# ── Multi-depth evaluation ─────────────────────────────────────────────────────

def depth_profile_metrics(
    pred:         np.ndarray,
    obs:          np.ndarray,
    depth_labels: list[str],
    depth_axis:   int = 1,
) -> dict[str, dict[str, float]]:
    """
    Compute RMSE, Bias, R², Pearson-r at each depth level.

    Parameters
    ----------
    pred         : Array with a depth dimension, e.g. (N, D, ...) or (D, ...).
    obs          : Same shape as pred.
    depth_labels : Human-readable label for each depth, e.g. ['d5m', 'd10m', ...].
    depth_axis   : Axis index for the depth dimension.

    Returns
    -------
    dict  { depth_label: {'rmse': float, 'bias': float, 'r2': float, 'r': float} }
    """
    results: dict[str, dict[str, float]] = {}
    n_depths = pred.shape[depth_axis]
    assert len(depth_labels) == n_depths, (
        f"depth_labels length {len(depth_labels)} != n_depths {n_depths}"
    )

    for d, label in enumerate(depth_labels):
        # Extract this depth level and flatten to 1-D
        idx       = [slice(None)] * pred.ndim
        idx[depth_axis] = d
        p_flat = pred[tuple(idx)].ravel()
        o_flat = obs[tuple(idx)].ravel()

        # Drop NaN pairs (land / missing)
        valid  = np.isfinite(p_flat) & np.isfinite(o_flat)
        p_flat = p_flat[valid]
        o_flat = o_flat[valid]

        if len(p_flat) < 2:
            results[label] = {"rmse": float("nan"), "bias": float("nan"),
                               "r2": float("nan"),   "r":    float("nan"),
                               "n_valid": 0}
            continue

        results[label] = {
            "rmse": rmse(p_flat, o_flat),
            "bias": bias(p_flat, o_flat),
            "r2":   r_squared(p_flat, o_flat),
            "r":    pearson_r(p_flat, o_flat),
            "n_valid": int(valid.sum()),
        }

    return results


def summary_table(results: dict[str, dict[str, float]], source: str) -> str:
    """
    Format a depth-profile metric dict as a human-readable text table.
    """
    lines = [
        f"  {'depth':<8}  {'RMSE':>7}  {'Bias':>7}  {'R²':>6}  {'r':>6}  {'N':>8}",
        f"  {'-' * 50}",
    ]
    for label, m in results.items():
        if m["n_valid"] == 0:
            lines.append(f"  {label:<8}  {'N/A':>7}  {'N/A':>7}  {'N/A':>6}  {'N/A':>6}  {'N/A':>8}")
        else:
            lines.append(
                f"  {label:<8}"
                f"  {m['rmse']:>7.4f}"
                f"  {m['bias']:>7.4f}"
                f"  {m['r2']:>6.4f}"
                f"  {m['r']:>6.4f}"
                f"  {m['n_valid']:>8,}"
            )
    return "\n".join(lines)
