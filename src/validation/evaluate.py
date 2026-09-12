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
                                  resolution constraint — 1° vs 0.5° model)

Workflow
--------
  1. Ensure test split (2019–2020) is preprocessed; run it if not.
  2. Run model inference on every test day → (N_test, 15, 50, 120) in °C.
  3. For INCOIS (10-day, 1°):
       - Find nearest-day model prediction for each INCOIS timestep.
       - Bilinear-regrid model 0.5° → 1°.
       - Select 14 common depths (INCOIS has no 0 m level).
       - Compute RMSE, Bias, R², r at each depth.
  4. For ARMOR3D (daily, 0.125°):
       - Direct date-matched subset.
       - Bilinear-regrid model 0.5° → 0.125°.
       - Interpolate ARMOR3D to our 15 PS-standard depths.
       - Compute metrics at each depth.
  5. Save structured results to outputs/metrics/validation_results.json
     and a human-readable summary to outputs/metrics/validation_summary.txt.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader

# Allow running as a module from any directory
_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.cache_loader import (
    get_armor3d, get_incois_mccreary, get_incois_vam,
)
from src.data.dataset       import OceanEmbedDataset
from src.model              import OceanEmbedModel
from src.preprocessing.normalize import load_stats, denormalize
from src.preprocessing.regrid    import make_target_grid
from src.validation.metrics import depth_profile_metrics, summary_table

# ── Depth constants ────────────────────────────────────────────────────────────
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_LABELS    = [f"d{d}m" for d in STANDARD_DEPTHS]

# Depths that exist in INCOIS ZAX (no 0 m level)
INCOIS_ZAX = [5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0,
              200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0,
              900.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]

# Our standard depths that overlap with INCOIS (excludes 0 m)
COMMON_DEPTHS_INCOIS = [d for d in STANDARD_DEPTHS if float(d) in INCOIS_ZAX]
COMMON_LABELS_INCOIS = [f"d{d}m" for d in COMMON_DEPTHS_INCOIS]
COMMON_IDX_INCOIS    = [STANDARD_DEPTHS.index(d) for d in COMMON_DEPTHS_INCOIS]


class OceanEmbedEvaluator:
    """
    Runs inference and multi-source validation for OceanEmbedModel.

    Parameters
    ----------
    checkpoint_path : Path to best.pt checkpoint.
    cfg             : Full YAML config dict.
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
        ckpt = torch.load(checkpoint_path, map_location=device)
        tw   = cfg["data"]["temporal_window"]
        self.model = OceanEmbedModel(
            in_channels  = tw * len(cfg["channels"]["names"]),
            enc_channels = cfg["model"]["enc_channels"],
            n_depths     = len(cfg["standard_depths"]),
        ).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        val_epoch  = ckpt.get("epoch", "?")
        val_loss   = ckpt.get("val_loss", float("nan"))
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
        loader: DataLoader,
        dataset: OceanEmbedDataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the model on all samples in loader and denormalize to °C.

        Returns
        -------
        preds_C : (N, 15, H, W)  float32  — predicted temperature in °C
        times   : (N,)           datetime64 — date of each prediction
        masks   : (H, W)         bool       — shared ocean mask
        """
        preds_norm_list = []
        t0 = time.perf_counter()

        for x, _, mask in loader:
            x = x.to(self.device)
            out = self.model(x).cpu().numpy()        # (B, 15, H, W) normalised
            preds_norm_list.append(out)

        preds_norm = np.concatenate(preds_norm_list, axis=0)  # (N, 15, H, W)

        # Denormalize each depth channel from Z-score to °C
        target_stats = self.stats["targets"]
        preds_C = np.zeros_like(preds_norm)
        for di, label in enumerate(DEPTH_LABELS):
            mean = target_stats[label]["mean"]
            std  = max(target_stats[label]["std"], 1e-10)
            preds_C[:, di] = preds_norm[:, di] * std + mean

        # Timestamps corresponding to each sample (last day of the temporal window)
        valid_indices = dataset.valid          # list of day-indices used as last-window-day
        all_times     = dataset.times          # full array of daily timestamps
        times         = all_times[valid_indices]

        # Ocean mask from the first sample (same for all)
        _, _, mask0 = dataset[0]
        ocean_mask = mask0.numpy().astype(bool)

        elapsed = time.perf_counter() - t0
        print(f"  Inference done: {len(preds_C)} samples in {elapsed:.1f}s")
        return preds_C, times, ocean_mask

    # ── Spatial regridding helper ──────────────────────────────────────────────

    @staticmethod
    def _regrid_predictions(
        preds:     np.ndarray,
        times:     np.ndarray,
        src_lats:  np.ndarray,
        src_lons:  np.ndarray,
        tgt_lats:  np.ndarray,
        tgt_lons:  np.ndarray,
    ) -> xr.DataArray:
        """
        Convert a (N, D, H, W) numpy array to xr.DataArray and bilinear-regrid
        to (tgt_lats, tgt_lons) using xr.interp.
        """
        da = xr.DataArray(
            preds.astype(np.float32),
            dims   = ["time", "depth_idx", "latitude", "longitude"],
            coords = {
                "time":      times,
                "depth_idx": np.arange(preds.shape[1]),
                "latitude":  src_lats,
                "longitude": src_lons,
            },
        )
        return da.interp(
            latitude  = tgt_lats,
            longitude = tgt_lons,
            method    = "linear",
        )

    # ── Time alignment helper ──────────────────────────────────────────────────

    @staticmethod
    def _align_times(
        model_times:    np.ndarray,
        product_times:  np.ndarray,
        max_delta_days: int = 5,
    ) -> list[tuple[int, int]]:
        """
        Find pairs (model_idx, product_idx) where model date is nearest to
        product date within max_delta_days.
        """
        mt = pd.DatetimeIndex(model_times)
        pairs: list[tuple[int, int]] = []
        for pi, pt in enumerate(pd.DatetimeIndex(product_times)):
            diffs = np.abs((mt - pt).days)
            mi    = int(diffs.argmin())
            if diffs[mi] <= max_delta_days:
                pairs.append((mi, pi))
        return pairs

    # ── INCOIS comparison ──────────────────────────────────────────────────────

    def _compare_incois(
        self,
        preds_C:    np.ndarray,
        times:      np.ndarray,
        start:      str,
        end:        str,
        source:     str,          # "vam" or "mccreary"
    ) -> dict:
        """
        Compare model predictions against one INCOIS product.

        Only depths present in both INCOIS ZAX and our STANDARD_DEPTHS
        are evaluated (14 depths — no 0 m level in INCOIS).
        Predictions are bilinear-regridded from 0.5° to INCOIS 1°.
        """
        print(f"\n  Comparing with INCOIS {source.upper()} ({start} → {end}) ...")

        # Load INCOIS
        if source == "vam":
            ds     = get_incois_vam(start, end)
            var    = "TEMP"
        else:
            ds     = get_incois_mccreary(start, end)
            var    = "T_ANALYZED"

        # Subset to our NIO domain
        d = self.cfg["data"]
        ds = ds.sel(
            latitude  = slice(d["lat_min"], d["lat_max"]),
            longitude = slice(d["lon_min"], d["lon_max"]),
        )
        tgt_lats = ds.latitude.values
        tgt_lons = ds.longitude.values
        print(f"    INCOIS grid: {len(tgt_lats)} lat × {len(tgt_lons)} lon (1°)")
        print(f"    INCOIS timesteps: {len(ds.time)}")

        # Regrid all model predictions to INCOIS 1° grid
        pred_rg = self._regrid_predictions(
            preds_C, times, self.target_lats, self.target_lons,
            tgt_lats, tgt_lons,
        )                                                # (N, 15, H_1, W_1)

        # Time alignment: find model predictions nearest each INCOIS timestep
        pairs = self._align_times(times, ds.time.values)
        print(f"    Aligned pairs:  {len(pairs)}")

        # Build aligned arrays at common depths
        pred_aligned = np.stack([
            pred_rg.isel(time=mi, depth_idx=COMMON_IDX_INCOIS).values
            for mi, _ in pairs
        ], axis=0)                    # (P, 14, H_1, W_1)

        obs_aligned = np.stack([
            ds[var].isel(time=pi).sel(ZAX=COMMON_DEPTHS_INCOIS).values
            for _, pi in pairs
        ], axis=0)                    # (P, 14, H_1, W_1)

        # Compute per-depth metrics
        results = depth_profile_metrics(pred_aligned, obs_aligned, COMMON_LABELS_INCOIS)
        return results

    # ── ARMOR3D comparison ─────────────────────────────────────────────────────

    def _compare_armor3d(
        self,
        preds_C:  np.ndarray,
        times:    np.ndarray,
        start:    str,
        end:      str,
    ) -> dict:
        """
        Compare model predictions against ARMOR3D (supplementary).

        ARMOR3D is daily at 0.125°.  Model is regridded to ARMOR3D grid;
        ARMOR3D is interpolated to our 15 PS-standard depths.
        """
        print(f"\n  Comparing with ARMOR3D [{start} → {end}] (supplementary) ...")

        ds       = get_armor3d(start, end)
        tgt_lats = ds.latitude.values
        tgt_lons = ds.longitude.values
        print(f"    ARMOR3D grid:  {len(tgt_lats)} lat × {len(tgt_lons)} lon (0.125°)")
        print(f"    ARMOR3D timesteps: {len(ds.time)}")

        # Regrid all model predictions to ARMOR3D 0.125° grid
        pred_rg = self._regrid_predictions(
            preds_C, times, self.target_lats, self.target_lons,
            tgt_lats, tgt_lons,
        )                                                # (N, 15, H_A, W_A)

        # Time alignment: ARMOR3D is daily — match exact dates
        pairs = self._align_times(times, ds.time.values, max_delta_days=0)
        print(f"    Date-matched pairs: {len(pairs)}")

        if len(pairs) == 0:
            print("    ⚠ No matching dates found — check test period vs ARMOR3D cache")
            return {}

        # Interpolate ARMOR3D to our standard depths
        std_depths_da = xr.DataArray(
            np.array(STANDARD_DEPTHS, dtype=np.float64), dims="depth"
        )

        pred_aligned = []
        obs_aligned  = []
        for mi, ai in pairs:
            p_snap = pred_rg.isel(time=mi).values   # (15, H_A, W_A)
            o_raw  = ds["to"].isel(time=ai)          # (D_armor, H_A, W_A)
            o_interp = o_raw.interp(
                depth  = std_depths_da,
                method = "linear",
                kwargs = {"fill_value": "extrapolate", "bounds_error": False},
            ).values                                  # (15, H_A, W_A)
            pred_aligned.append(p_snap)
            obs_aligned.append(o_interp)

        pred_aligned = np.stack(pred_aligned, axis=0)   # (P, 15, H_A, W_A)
        obs_aligned  = np.stack(obs_aligned,  axis=0)   # (P, 15, H_A, W_A)

        results = depth_profile_metrics(pred_aligned, obs_aligned, DEPTH_LABELS)
        return results

    # ── Orchestrate ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        test_loader: DataLoader,
        test_dataset: OceanEmbedDataset,
        out_dir: pathlib.Path,
    ) -> dict:
        """
        Run full evaluation: inference → INCOIS VAM → McCreary → ARMOR3D.

        Returns the results dict and saves JSON + text summary.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        d = self.cfg["data"]
        test_start = d["test_start"][:4]   # "2019"
        test_end   = d["test_end"][:4]     # "2020"

        # ── 1. Inference ──────────────────────────────────────────────────────
        print("\n  Running inference on test split ...")
        preds_C, times, _ = self.run_inference(test_loader, test_dataset)
        print(f"  Predictions: {preds_C.shape}  range [{preds_C.min():.1f}, {preds_C.max():.1f}] °C")

        results: dict = {
            "meta": {
                "generated":   datetime.now().isoformat(),
                "test_period": f"{test_start}–{test_end}",
                "n_test_days": int(preds_C.shape[0]),
                "grid":        f"0.5°  50×120  NIO",
                "depths_m":    STANDARD_DEPTHS,
                "note_primary":      "INCOIS LAS VAM + McCreary are the PRIMARY validation datasets",
                "note_supplementary":"ARMOR3D is SUPPLEMENTARY — lower weight due to resolution constraint",
            },
            "primary":      {},
            "supplementary":{},
        }

        # ── 2. INCOIS VAM (PRIMARY) ───────────────────────────────────────────
        for yr in [test_start, test_end]:
            key = f"incois_vam_{yr}"
            print(f"\n─── INCOIS VAM {yr} ───")
            try:
                r = self._compare_incois(
                    preds_C, times,
                    start=f"{yr}-01-01", end=f"{yr}-12-31",
                    source="vam",
                )
                results["primary"][key] = r
                _print_metrics(key, r)
            except Exception as exc:
                print(f"    ⚠ INCOIS VAM {yr} failed: {exc}")
                results["primary"][key] = {"error": str(exc)}

        # ── 3. INCOIS McCreary (PRIMARY) ──────────────────────────────────────
        for yr in [test_start, test_end]:
            key = f"incois_mccreary_{yr}"
            print(f"\n─── INCOIS McCreary {yr} ───")
            try:
                r = self._compare_incois(
                    preds_C, times,
                    start=f"{yr}-01-01", end=f"{yr}-12-31",
                    source="mccreary",
                )
                results["primary"][key] = r
                _print_metrics(key, r)
            except Exception as exc:
                print(f"    ⚠ INCOIS McCreary {yr} failed: {exc}")
                results["primary"][key] = {"error": str(exc)}

        # ── 4. ARMOR3D (SUPPLEMENTARY) ────────────────────────────────────────
        for yr in [test_start, test_end]:
            key = f"armor3d_{yr}"
            print(f"\n─── ARMOR3D {yr} (supplementary) ───")
            try:
                r = self._compare_armor3d(
                    preds_C, times,
                    start=f"{yr}-01-01", end=f"{yr}-12-31",
                )
                results["supplementary"][key] = r
                _print_metrics(key, r)
            except Exception as exc:
                print(f"    ⚠ ARMOR3D {yr} failed: {exc}")
                results["supplementary"][key] = {"error": str(exc)}

        # ── 5. Save outputs ───────────────────────────────────────────────────
        json_path = out_dir / "validation_results.json"
        json_path.write_text(json.dumps(results, indent=2))
        print(f"\n  Saved: {json_path}")

        txt_path = out_dir / "validation_summary.txt"
        txt_path.write_text(_build_summary(results))
        print(f"  Saved: {txt_path}")

        return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_metrics(key: str, metrics: dict) -> None:
    """Print a compact metric table to stdout."""
    if "error" in metrics:
        return
    first = next(iter(metrics.values()), None)
    if first is None or "rmse" not in first:
        return
    print(f"\n  {key}:")
    print(summary_table(metrics, key))


def _build_summary(results: dict) -> str:
    """Build a readable text summary file."""
    lines = [
        "OceanEmbed PoC — Validation Summary",
        "=" * 65,
        f"Generated:   {results['meta']['generated']}",
        f"Test period: {results['meta']['test_period']}",
        f"Test days:   {results['meta']['n_test_days']}",
        f"Grid:        {results['meta']['grid']}",
        "",
        "PRIMARY VALIDATION",
        "  INCOIS LAS VAM + McCreary (in-situ based, 10-day, 1°)",
        "-" * 65,
    ]
    for key, metrics in results["primary"].items():
        lines.append(f"\n{key}:")
        if "error" in metrics:
            lines.append(f"  ERROR: {metrics['error']}")
        else:
            lines.append(summary_table(metrics, key))

    lines += [
        "",
        "SUPPLEMENTARY VALIDATION",
        "  ARMOR3D (satellite+in-situ blended, daily, 0.125°)",
        "  Note: lower weight due to 1° resolution constraint vs 0.5° model",
        "-" * 65,
    ]
    for key, metrics in results["supplementary"].items():
        lines.append(f"\n{key}:")
        if "error" in metrics:
            lines.append(f"  ERROR: {metrics['error']}")
        else:
            lines.append(summary_table(metrics, key))

    return "\n".join(lines)
