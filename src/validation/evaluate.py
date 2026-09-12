"""
evaluate.py
-----------
OceanEmbed PoC — test-set inference and multi-source validation.

Validation hierarchy (per problem statement §3.6b)
---------------------------------------------------
PRIMARY   : INCOIS LAS VAM     — in-situ optimally interpolated, 10-day, 1°
            INCOIS McCreary    — model+in-situ blended, 10-day, 1°
SUPPLEMENTARY : ARMOR3D        — satellite+in-situ blended, daily, 0.125°
                                 (excluded from primary ranking due to
                                  resolution constraint)

Memory-efficient design
-----------------------
The naive approach of loading all 729 predictions + a full year of ARMOR3D
would require ~6 GB of RAM — OOM on a laptop.

Instead:
  - Predictions (N=729, 15, 50, 120) at 0.5° ≈ 263 MB — kept in RAM.
  - INCOIS comparison: only the ~72 matched timesteps are regridded (one at a
    time), not all 729. Peak memory per step ≈ 1 MB.
  - ARMOR3D comparison: processed month-by-month (~400 MB/month), immediately
    resampled to the model's own 0.5° grid (not upscaled to 0.125°). Running
    statistics accumulated per depth — no large arrays kept alive.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from datetime import datetime
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.cache_loader import (
    get_armor3d, get_incois_mccreary, get_incois_vam,
)
from src.data.dataset    import OceanEmbedDataset
from src.model           import OceanEmbedModel
from src.preprocessing.normalize import load_stats
from src.preprocessing.regrid    import make_target_grid
from src.validation.metrics      import summary_table

# ── Depth constants ────────────────────────────────────────────────────────────
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_LABELS    = [f"d{d}m" for d in STANDARD_DEPTHS]

# Depths that exist in INCOIS ZAX (no 0 m level in INCOIS product)
_INCOIS_ZAX = [5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400,
               500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000]
COMMON_DEPTHS_INCOIS = [d for d in STANDARD_DEPTHS if d in _INCOIS_ZAX]
COMMON_LABELS_INCOIS = [f"d{d}m" for d in COMMON_DEPTHS_INCOIS]
COMMON_IDX_INCOIS    = [STANDARD_DEPTHS.index(d) for d in COMMON_DEPTHS_INCOIS]


# ── Running stats accumulator (online / one-pass) ──────────────────────────────

class _RunningStats:
    """
    Accumulate RMSE, Bias, Pearson-r statistics incrementally without
    storing all observations in RAM.

    Uses Welford-style running sums so any number of timesteps can be
    processed one-at-a-time with O(1) extra memory.
    """

    __slots__ = ("_n", "_sum_sq_err", "_sum_diff",
                 "_sum_p", "_sum_o", "_sum_pp", "_sum_oo", "_sum_po")

    def __init__(self) -> None:
        self._n          = 0
        self._sum_sq_err = 0.0   # sum (pred - obs)^2
        self._sum_diff   = 0.0   # sum (pred - obs)
        self._sum_p      = 0.0
        self._sum_o      = 0.0
        self._sum_pp     = 0.0
        self._sum_oo     = 0.0
        self._sum_po     = 0.0   # sum (pred * obs)

    def update(self, pred_flat: np.ndarray, obs_flat: np.ndarray) -> None:
        """Accept 1-D arrays of valid (non-NaN) pixels for one timestep."""
        n = len(pred_flat)
        if n == 0:
            return
        diff = pred_flat - obs_flat
        self._n          += n
        self._sum_sq_err += float(np.sum(diff * diff))
        self._sum_diff   += float(np.sum(diff))
        self._sum_p      += float(np.sum(pred_flat))
        self._sum_o      += float(np.sum(obs_flat))
        self._sum_pp     += float(np.sum(pred_flat * pred_flat))
        self._sum_oo     += float(np.sum(obs_flat  * obs_flat))
        self._sum_po     += float(np.sum(pred_flat * obs_flat))

    def result(self) -> dict:
        n = self._n
        if n < 2:
            return {"rmse": float("nan"), "bias": float("nan"),
                    "r2":   float("nan"), "r":    float("nan"),
                    "n_valid": 0}

        rmse_val = float(np.sqrt(self._sum_sq_err / n))
        bias_val = float(self._sum_diff / n)

        pm = self._sum_p / n
        om = self._sum_o / n
        cov  = self._sum_po / n - pm * om
        varp = max(self._sum_pp / n - pm * pm, 0.0)
        varo = max(self._sum_oo / n - om * om, 0.0)
        denom = float(np.sqrt(varp * varo))
        r_val = float(cov / denom) if denom > 1e-12 else float("nan")

        # R² = 1 − SS_res / SS_tot  (using obs variance as SS_tot denominator)
        ss_tot = self._sum_oo - n * om * om
        r2_val = float(1.0 - self._sum_sq_err / ss_tot) if ss_tot > 1e-12 else float("nan")

        return {
            "rmse":    rmse_val,
            "bias":    bias_val,
            "r2":      r2_val,
            "r":       r_val,
            "n_valid": n,
        }


def _month_periods(start: str, end: str) -> Iterator[tuple[str, str]]:
    """Yield (month_start, month_end) pairs between start and end dates."""
    for period in pd.period_range(start=start, end=end, freq="M"):
        yield (
            period.start_time.strftime("%Y-%m-%d"),
            period.end_time.strftime("%Y-%m-%d"),
        )


# ── Evaluator ─────────────────────────────────────────────────────────────────

class OceanEmbedEvaluator:
    """
    Runs inference and multi-source validation for OceanEmbedModel.

    Parameters
    ----------
    checkpoint_path : Path to best.pt checkpoint.
    cfg             : Full YAML config dict (from poc.yaml).
    device          : Torch device.
    """

    def __init__(
        self,
        checkpoint_path: pathlib.Path,
        cfg:             dict,
        device:          torch.device,
    ) -> None:
        self.cfg    = cfg
        self.device = device

        # ── Norm stats ────────────────────────────────────────────────────────
        proc_root  = _ROOT / cfg["data"]["processed_dir"]
        self.stats = load_stats(proc_root / "norm_stats.json")

        # ── Model ─────────────────────────────────────────────────────────────
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        tw   = cfg["data"]["temporal_window"]
        self.model = OceanEmbedModel(
            in_channels  = tw * len(cfg["channels"]["names"]),
            enc_channels = cfg["model"]["enc_channels"],
            n_depths     = len(cfg["standard_depths"]),
        ).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        val_epoch = ckpt.get("epoch", "?")
        val_loss  = ckpt.get("val_loss", float("nan"))
        print(f"  Loaded checkpoint: epoch {val_epoch}, val_loss {val_loss:.4f}")

        # ── Target grid ───────────────────────────────────────────────────────
        d = cfg["data"]
        self.target_lats, self.target_lons = make_target_grid(
            d["lat_min"], d["lat_max"],
            d["lon_min"], d["lon_max"],
            resolution=d["grid_res"],
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run_inference(
        self,
        loader:  DataLoader,
        dataset: OceanEmbedDataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the model on all test samples and denormalize to °C.

        Returns
        -------
        preds_C : (N, 15, H, W)  float32  — predicted temperature in °C
        times   : (N,)           datetime64[ns] — date of each prediction
        """
        t0 = time.perf_counter()
        preds_norm_list: list[np.ndarray] = []

        for x, _, _mask in loader:
            x   = x.to(self.device)
            out = self.model(x).cpu().numpy()    # (B, 15, H, W) normalised
            preds_norm_list.append(out)

        preds_norm = np.concatenate(preds_norm_list, axis=0)   # (N, 15, H, W)

        # Denormalize from Z-score to °C using per-depth norm stats
        target_stats = self.stats["targets"]
        preds_C      = np.empty_like(preds_norm)
        for di, label in enumerate(DEPTH_LABELS):
            mean = float(target_stats[label]["mean"])
            std  = max(float(target_stats[label]["std"]), 1e-10)
            preds_C[:, di] = preds_norm[:, di] * std + mean

        # Timestamps: one per sample = the last day of its temporal window
        times = dataset.times[dataset.valid]

        elapsed = time.perf_counter() - t0
        print(f"  Inference: {len(preds_C)} samples, {elapsed:.1f}s")
        print(f"  T range: [{preds_C.min():.1f}, {preds_C.max():.1f}] °C")
        return preds_C, times

    # ── INCOIS comparison — per-timestep regrid ────────────────────────────────

    def _compare_incois(
        self,
        preds_C: np.ndarray,
        times:   np.ndarray,
        start:   str,
        end:     str,
        source:  str,          # "vam" or "mccreary"
    ) -> dict:
        """
        Compare model predictions against one INCOIS product.

        Memory strategy: align ~72 timesteps first, then regrid and compare
        one timestep at a time — peak extra memory ≈ a few MB.

        Depths: 14 common (INCOIS has no 0 m level).
        Spatial: model 0.5° bilinear-regridded to INCOIS 1°.
        """
        print(f"\n  INCOIS {source.upper()} [{start} → {end}] ...")

        loader_fn = get_incois_vam if source == "vam" else get_incois_mccreary
        obs_var   = "TEMP"         if source == "vam" else "T_ANALYZED"

        ds = loader_fn(start, end)
        d  = self.cfg["data"]
        ds = ds.sel(
            latitude  = slice(d["lat_min"], d["lat_max"]),
            longitude = slice(d["lon_min"], d["lon_max"]),
        )
        tgt_lats = ds.latitude.values
        tgt_lons = ds.longitude.values
        print(f"    Grid: {len(tgt_lats)} lat × {len(tgt_lons)} lon | {len(ds.time)} timesteps")

        # Align times
        model_pdx  = pd.DatetimeIndex(times)
        incois_pdx = pd.DatetimeIndex(ds.time.values)
        pairs: list[tuple[int, int]] = []
        for pi, pt in enumerate(incois_pdx):
            diffs = np.abs((model_pdx - pt).days)
            mi    = int(diffs.argmin())
            if diffs[mi] <= 5:
                pairs.append((mi, pi))
        print(f"    Aligned pairs: {len(pairs)}")

        # Running stats per common depth
        accum = {lbl: _RunningStats() for lbl in COMMON_LABELS_INCOIS}

        for mi, pi in pairs:
            # Regrid this one model snapshot (15, H, W) → (15, tgt_H, tgt_W)
            snap_da = xr.DataArray(
                preds_C[mi],
                dims   = ["depth_idx", "latitude", "longitude"],
                coords = {"latitude": self.target_lats, "longitude": self.target_lons},
            )
            snap_rg = snap_da.interp(
                latitude  = tgt_lats,
                longitude = tgt_lons,
                method    = "linear",
            ).values   # (15, tgt_H, tgt_W)

            # Observation at this timestep — select matching depths
            obs = ds[obs_var].isel(time=pi).sel(
                ZAX=COMMON_DEPTHS_INCOIS
            ).values   # (14, tgt_H, tgt_W)

            for k, (di, lbl) in enumerate(zip(COMMON_IDX_INCOIS, COMMON_LABELS_INCOIS)):
                p_flat = snap_rg[di].ravel()
                o_flat = obs[k].ravel()
                valid  = np.isfinite(p_flat) & np.isfinite(o_flat)
                accum[lbl].update(p_flat[valid], o_flat[valid])

        return {lbl: accum[lbl].result() for lbl in COMMON_LABELS_INCOIS}

    # ── ARMOR3D comparison — month-by-month ────────────────────────────────────

    def _compare_armor3d(
        self,
        preds_C: np.ndarray,
        times:   np.ndarray,
        start:   str,
        end:     str,
    ) -> dict:
        """
        Compare model predictions against ARMOR3D (supplementary).

        Memory strategy: load one month at a time (~400 MB), immediately
        resample ARMOR3D to the model's own 0.5° grid (avoiding any upscale
        to 0.125°), interpolate depths to our 15 PS-standard levels, then
        compare and discard — peak extra memory ≈ 400 MB.
        """
        print(f"\n  ARMOR3D [{start} → {end}] (supplementary, month-by-month) ...")

        model_pdx = pd.DatetimeIndex(times)
        std_depths = np.array(STANDARD_DEPTHS, dtype=np.float64)

        # Running stats per all 15 depths
        accum  = {lbl: _RunningStats() for lbl in DEPTH_LABELS}
        n_matched = 0

        for m_start, m_end in _month_periods(start, end):
            try:
                armor = get_armor3d(m_start, m_end)
            except Exception as exc:
                print(f"    {m_start[:7]}: skip ({exc})")
                continue

            # Resample ARMOR3D (0.125°, up to 480×200) → model 0.5° grid
            # interp is safe here: ~31 days × 37 depths × 50 × 120 ≈ 28 MB
            armor_rg = armor["to"].interp(
                latitude  = self.target_lats,
                longitude = self.target_lons,
                method    = "linear",
            )   # (days_in_month, depth_armor, 50, 120)

            # Interpolate ARMOR3D depths to our 15 PS-standard depths
            armor_interp = armor_rg.interp(
                depth  = std_depths,
                method = "linear",
                kwargs = {"fill_value": float("nan"), "bounds_error": False},
            )   # (days_in_month, 15, 50, 120)

            armor_vals = armor_interp.values   # (days_in_month, 15, 50, 120)
            armor_pdx  = pd.DatetimeIndex(armor.time.values)

            for ai, at in enumerate(armor_pdx):
                diffs = np.abs((model_pdx - at).days)
                mi    = int(diffs.argmin())
                if diffs[mi] > 0:
                    continue   # ARMOR3D is daily — only exact matches

                n_matched += 1
                for di, lbl in enumerate(DEPTH_LABELS):
                    p_flat = preds_C[mi, di].ravel()
                    o_flat = armor_vals[ai, di].ravel()
                    valid  = np.isfinite(p_flat) & np.isfinite(o_flat)
                    accum[lbl].update(p_flat[valid], o_flat[valid])

            # Free this month's data before loading the next
            del armor, armor_rg, armor_interp, armor_vals

        print(f"    Matched days: {n_matched}")
        return {lbl: accum[lbl].result() for lbl in DEPTH_LABELS}

    # ── Orchestrate ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_loader:  DataLoader,
        test_dataset: OceanEmbedDataset,
        out_dir:      pathlib.Path,
    ) -> dict:
        """
        Full evaluation: inference → INCOIS VAM → McCreary → ARMOR3D.

        Saves validation_results.json and validation_summary.txt to out_dir.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        d          = self.cfg["data"]
        test_start = d["test_start"][:4]
        test_end   = d["test_end"][:4]

        print("\n  Running inference ...")
        preds_C, times = self.run_inference(test_loader, test_dataset)

        results: dict = {
            "meta": {
                "generated":         datetime.now().isoformat(),
                "test_period":       f"{test_start}–{test_end}",
                "n_test_days":       int(preds_C.shape[0]),
                "grid":              "0.5°  50×120  NIO",
                "depths_m":          STANDARD_DEPTHS,
                "note_primary":      "INCOIS LAS VAM + McCreary are PRIMARY validation",
                "note_supplementary":"ARMOR3D is SUPPLEMENTARY — resampled to model 0.5° grid",
            },
            "primary":       {},
            "supplementary": {},
        }

        # ── INCOIS VAM (PRIMARY) ──────────────────────────────────────────────
        for yr in [test_start, test_end]:
            key = f"incois_vam_{yr}"
            try:
                r = self._compare_incois(
                    preds_C, times, f"{yr}-01-01", f"{yr}-12-31", source="vam"
                )
                results["primary"][key] = r
                _print_metrics(key, r)
            except Exception as exc:
                print(f"  ⚠ {key}: {exc}")
                results["primary"][key] = {"error": str(exc)}

        # ── INCOIS McCreary (PRIMARY) ─────────────────────────────────────────
        for yr in [test_start, test_end]:
            key = f"incois_mccreary_{yr}"
            try:
                r = self._compare_incois(
                    preds_C, times, f"{yr}-01-01", f"{yr}-12-31", source="mccreary"
                )
                results["primary"][key] = r
                _print_metrics(key, r)
            except Exception as exc:
                print(f"  ⚠ {key}: {exc}")
                results["primary"][key] = {"error": str(exc)}

        # ── ARMOR3D (SUPPLEMENTARY) ───────────────────────────────────────────
        for yr in [test_start, test_end]:
            key = f"armor3d_{yr}"
            try:
                r = self._compare_armor3d(
                    preds_C, times, f"{yr}-01-01", f"{yr}-12-31"
                )
                results["supplementary"][key] = r
                _print_metrics(key, r)
            except Exception as exc:
                print(f"  ⚠ {key}: {exc}")
                results["supplementary"][key] = {"error": str(exc)}

        # ── Save ──────────────────────────────────────────────────────────────
        json_path = out_dir / "validation_results.json"
        json_path.write_text(json.dumps(results, indent=2))

        txt_path = out_dir / "validation_summary.txt"
        txt_path.write_text(_build_summary(results))

        print(f"\n  Saved: {json_path}")
        print(f"  Saved: {txt_path}")
        return results


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _print_metrics(key: str, metrics: dict) -> None:
    if not metrics or "error" in metrics:
        return
    first = next(iter(metrics.values()), {})
    if "rmse" not in first:
        return
    print(f"\n  {key}:")
    print(summary_table(metrics, key))


def _build_summary(results: dict) -> str:
    lines = [
        "OceanEmbed PoC — Validation Summary",
        "=" * 65,
        f"Generated:   {results['meta']['generated']}",
        f"Test period: {results['meta']['test_period']}",
        f"Test days:   {results['meta']['n_test_days']}",
        "",
        "PRIMARY VALIDATION",
        "  INCOIS LAS VAM + McCreary (in-situ, 10-day, 1°)",
        "-" * 65,
    ]
    for key, m in results["primary"].items():
        lines.append(f"\n{key}:")
        if "error" in m:
            lines.append(f"  ERROR: {m['error']}")
        else:
            lines.append(summary_table(m, key))

    lines += [
        "",
        "SUPPLEMENTARY VALIDATION",
        "  ARMOR3D (blended, daily, resampled to model 0.5° grid)",
        "-" * 65,
    ]
    for key, m in results["supplementary"].items():
        lines.append(f"\n{key}:")
        if "error" in m:
            lines.append(f"  ERROR: {m['error']}")
        else:
            lines.append(summary_table(m, key))

    return "\n".join(lines)
