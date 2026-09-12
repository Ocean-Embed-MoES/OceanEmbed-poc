#!/usr/bin/env python3
"""
scripts/prepare_data.py
-----------------------
End-to-end preprocessing pipeline for OceanEmbed PoC.

Strategy (memory-efficient)
---------------------------
Surface inputs (SST, SLA, SSS, currents, winds) are small after NIO slicing
and are loaded for the whole year at once (~100–700 MB each).

GLORYS (11.4 GB/year) is processed in MONTHLY chunks to keep peak RAM below
~2 GB.  Each month's (30, 36, 301, 721) thetao slice is regridded and depth-
interpolated independently, then concatenated before saving.

Usage
-----
  python scripts/prepare_data.py               # all splits
  python scripts/prepare_data.py --splits train val
  python scripts/prepare_data.py --dry-run
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import xarray as xr
import yaml

from src.data.cache_loader import (
    get_currents, get_glorys, get_ssh, get_sss, get_sst, get_winds,
)
from src.preprocessing.features  import gradient_logmag, wind_stress_curl
from src.preprocessing.normalize import compute_channel_stats, save_stats
from src.preprocessing.regrid    import make_target_grid, regrid


# ── Constants (must match src/data/dataset.py) ────────────────────────────────
CHANNEL_NAMES   = [
    "sst", "grad_sst", "sla", "grad_sla",
    "sss", "wsc", "u_curr", "v_curr",
]
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_LABELS    = [f"d{d}m" for d in STANDARD_DEPTHS]

# Process GLORYS in chunks of this many days to keep RAM under ~2 GB
GLORYS_CHUNK_DAYS = 30


# ── Surface input processing (yearly — small after NIO slice) ─────────────────

def _process_surface_inputs(
    year: int,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load + regrid + feature-engineer all surface inputs for one year.

    Returns
    -------
    inputs : np.ndarray  (T, 8, H, W)  float32
    times  : np.ndarray  datetime64 array of length T
    """
    start, end = f"{year}-01-01", f"{year}-12-31"

    print("    Loading SST ...", end=" ", flush=True)
    sst_raw = get_sst(start, end, lat_min, lat_max, lon_min, lon_max)
    sst_raw.name = "sst"
    print(sst_raw.shape)

    print("    Loading SLA ...", end=" ", flush=True)
    sla_raw = get_ssh(start, end, lat_min, lat_max, lon_min, lon_max)
    sla_raw.name = "sla"
    print(sla_raw.shape)

    print("    Loading SSS ...", end=" ", flush=True)
    sss_raw = get_sss(start, end, lat_min, lat_max, lon_min, lon_max)
    sss_raw.name = "sss"
    print(sss_raw.shape)

    print("    Loading currents ...", end=" ", flush=True)
    curr_raw = get_currents(start, end, lat_min, lat_max, lon_min, lon_max)
    print(f"u={curr_raw['u'].shape}")

    print("    Loading winds (6-h → daily) ...", end=" ", flush=True)
    wind_raw = get_winds(start, end, lat_min, lat_max, lon_min, lon_max)
    print(f"uwnd={wind_raw['uwnd'].shape}")

    print("    Regriding surface inputs to 0.5° ...", end=" ", flush=True)
    t_rg = time.perf_counter()

    sst_rg    = regrid(sst_raw,          target_lats, target_lons).load()
    sla_rg    = regrid(sla_raw,          target_lats, target_lons).load()
    sss_rg    = regrid(sss_raw,          target_lats, target_lons).load()
    u_curr_rg = regrid(curr_raw["u"],    target_lats, target_lons).load()
    v_curr_rg = regrid(curr_raw["v"],    target_lats, target_lons).load()
    uwnd_rg   = regrid(wind_raw["uwnd"], target_lats, target_lons).load()
    vwnd_rg   = regrid(wind_raw["vwnd"], target_lats, target_lons).load()
    print(f"{time.perf_counter() - t_rg:.0f}s")

    print("    Computing ∇SST, ∇SLA, WSC ...", end=" ", flush=True)
    t_feat = time.perf_counter()
    sst_rg.name = "sst"
    sla_rg.name = "sla"
    grad_sst = gradient_logmag(sst_rg)
    grad_sla = gradient_logmag(sla_rg)
    wsc      = wind_stress_curl(uwnd_rg, vwnd_rg)
    print(f"{time.perf_counter() - t_feat:.0f}s")

    inputs = np.stack([
        sst_rg.values,     # 0  SST
        grad_sst.values,   # 1  ∇SST
        sla_rg.values,     # 2  SLA
        grad_sla.values,   # 3  ∇SLA
        sss_rg.values,     # 4  SSS
        wsc.values,        # 5  WSC
        u_curr_rg.values,  # 6  U_curr
        v_curr_rg.values,  # 7  V_curr
    ], axis=1).astype(np.float32)              # (T, 8, H, W)

    times = sst_rg.time.values
    print(f"    Surface inputs: {inputs.shape}  {inputs.nbytes / 1e6:.0f} MB")
    return inputs, times


# ── GLORYS processing (monthly chunks — avoids loading 11 GB at once) ─────────

def _process_glorys_chunked(
    year: int,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    n_days: int,
) -> np.ndarray:
    """
    Load, regrid, and depth-interpolate GLORYS in monthly chunks.

    Processes GLORYS_CHUNK_DAYS days at a time so peak RAM is ~1-2 GB
    instead of 11 GB for the full year.

    Returns
    -------
    targets : np.ndarray  (T, 15, H, W)  float32  °C
    """
    std_depths = np.array(STANDARD_DEPTHS, dtype=np.float64)
    targets_list: list[np.ndarray] = []

    # Build list of (month_start, month_end) strings covering the year
    import calendar
    month_ranges = []
    for month in range(1, 13):
        _, last_day = calendar.monthrange(year, month)
        ms = f"{year}-{month:02d}-01"
        me = f"{year}-{month:02d}-{last_day:02d}"
        month_ranges.append((ms, me, month))

    for ms, me, month in month_ranges:
        print(f"    GLORYS {year}-{month:02d} ...", end=" ", flush=True)
        t_chunk = time.perf_counter()

        glorys_raw = get_glorys(ms, me, lat_min, lat_max, lon_min, lon_max)
        thetao = glorys_raw["thetao"]   # (days, 36, 301, 721) — lazy

        # Regrid spatially to 0.5° (still lazy until .load())
        thetao_rg = regrid(
            thetao, target_lats, target_lons,
            lat_dim="latitude", lon_dim="longitude",
        )

        # Depth interpolation — keeps lazy
        thetao_interp = thetao_rg.interp(
            depth=std_depths,
            method="linear",
            kwargs={"fill_value": "extrapolate", "bounds_error": False},
        )

        # Now trigger the actual I/O + computation
        chunk_arr = thetao_interp.values.astype(np.float32)  # (days, 15, 50, 120)
        targets_list.append(chunk_arr)

        elapsed = time.perf_counter() - t_chunk
        mb = chunk_arr.nbytes / 1e6
        print(f"shape={chunk_arr.shape}  {mb:.0f} MB  {elapsed:.0f}s")

        # Explicitly close to free file handles
        glorys_raw.close()

    targets = np.concatenate(targets_list, axis=0)   # (T, 15, 50, 120)

    if targets.shape[0] != n_days:
        print(f"    ⚠ Time mismatch: expected {n_days} days, got {targets.shape[0]}")

    print(f"    GLORYS targets: {targets.shape}  {targets.nbytes / 1e6:.0f} MB")
    return targets


# ── Per-year orchestration ─────────────────────────────────────────────────────

def process_year(
    year: int,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    out_dir: pathlib.Path,
) -> None:
    """
    Process one calendar year and save NetCDF output files.

    Saved files
    -----------
    inputs_{year}.nc   — (T, 8,  50, 120)  float32  physical units
    targets_{year}.nc  — (T, 15, 50, 120)  float32  °C
    """
    import calendar
    t0 = time.perf_counter()
    print(f"\n  ── {year} ({year}-01-01 → {year}-12-31) ──")

    # Leap year check
    n_days = 366 if calendar.isleap(year) else 365

    # ── Surface inputs (fast — all in memory) ──────────────────────────────
    inputs, times = _process_surface_inputs(
        year, lat_min, lat_max, lon_min, lon_max, target_lats, target_lons,
    )

    # ── GLORYS target (monthly chunks) ─────────────────────────────────────
    print("    Processing GLORYS in monthly chunks ...")
    targets = _process_glorys_chunked(
        year, lat_min, lat_max, lon_min, lon_max, target_lats, target_lons,
        n_days=len(times),
    )

    # Align time axes in case GLORYS has different coverage at year edges
    T = min(len(times), targets.shape[0])
    inputs  = inputs[:T]
    targets = targets[:T]
    times   = times[:T]

    # ── Save ────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_inputs = xr.Dataset(
        {"inputs": (["time", "channel", "latitude", "longitude"], inputs)},
        coords={
            "time":      times,
            "channel":   CHANNEL_NAMES,
            "latitude":  target_lats,
            "longitude": target_lons,
        },
        attrs={
            "description": "OceanEmbed PoC — 8-channel surface inputs",
            "channels":    ", ".join(CHANNEL_NAMES),
            "year":        str(year),
            "grid_res_deg": "0.5",
            "domain":      "NIO 5-30N 45-105E",
        },
    )
    ds_targets = xr.Dataset(
        {"targets": (["time", "depth", "latitude", "longitude"], targets)},
        coords={
            "time":      times,
            "depth":     STANDARD_DEPTHS,
            "latitude":  target_lats,
            "longitude": target_lons,
        },
        attrs={
            "description": "OceanEmbed PoC — GLORYS thetao at 15 PS-standard depths",
            "source":      "Copernicus GLORYS12 (cmems_mod_glo_phy_my_0.083deg_P1D-m)",
            "depths_m":    str(STANDARD_DEPTHS),
            "year":        str(year),
            "units":       "degC",
        },
    )

    inp_path = out_dir / f"inputs_{year}.nc"
    tgt_path = out_dir / f"targets_{year}.nc"
    print("    Writing NetCDF ...", end=" ", flush=True)
    t_write = time.perf_counter()
    ds_inputs.to_netcdf(inp_path,  format="NETCDF4")
    ds_targets.to_netcdf(tgt_path, format="NETCDF4")
    print(f"{time.perf_counter() - t_write:.0f}s")

    elapsed = time.perf_counter() - t0
    print(f"    ✓ inputs_{year}.nc   {inp_path.stat().st_size / 1e6:.0f} MB")
    print(f"    ✓ targets_{year}.nc  {tgt_path.stat().st_size / 1e6:.0f} MB")
    print(f"    Year {year} done in {elapsed / 60:.1f} min")


# ── Normalization stats from training files ────────────────────────────────────

def compute_and_save_stats(
    train_dir: pathlib.Path,
    stats_path: pathlib.Path,
) -> None:
    """
    Load all training-year inputs/targets, compute Z-score stats (training
    data ONLY), and save to *stats_path*.
    """
    print("\n  Computing normalization statistics from training data ...")

    input_files  = sorted(train_dir.glob("inputs_*.nc"))
    target_files = sorted(train_dir.glob("targets_*.nc"))

    print(f"  Reading {len(input_files)} input files ...")
    all_inputs = np.concatenate(
        [xr.open_dataset(f)["inputs"].values  for f in input_files],  axis=0
    ).astype(np.float32)

    print(f"  Reading {len(target_files)} target files ...")
    all_targets = np.concatenate(
        [xr.open_dataset(f)["targets"].values for f in target_files], axis=0
    ).astype(np.float32)

    print(f"  Training inputs  shape: {all_inputs.shape}")
    print(f"  Training targets shape: {all_targets.shape}")

    input_stats  = compute_channel_stats(all_inputs,  CHANNEL_NAMES, channel_axis=1)
    target_stats = compute_channel_stats(all_targets, DEPTH_LABELS,  channel_axis=1)

    save_stats({"inputs": input_stats, "targets": target_stats}, stats_path)
    print(f"  Saved: {stats_path}")

    print("\n  Input channel statistics:")
    print(f"  {'channel':<12}  {'mean':>10}  {'std':>10}")
    print(f"  {'-'*36}")
    for name, s in input_stats.items():
        print(f"  {name:<12}  {s['mean']:>10.4f}  {s['std']:>10.4f}")

    print("\n  Target depth statistics (°C):")
    print(f"  {'depth':<10}  {'mean':>8}  {'std':>8}")
    print(f"  {'-'*30}")
    for name, s in target_stats.items():
        print(f"  {name:<10}  {s['mean']:>8.3f}  {s['std']:>8.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _years_in_range(start: str, end: str) -> list[int]:
    return list(range(int(start[:4]), int(end[:4]) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OceanEmbed PoC — end-to-end preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",  default="configs/poc.yaml")
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text())
    d   = cfg["data"]

    lat_min, lat_max = d["lat_min"], d["lat_max"]
    lon_min, lon_max = d["lon_min"], d["lon_max"]

    target_lats, target_lons = make_target_grid(
        lat_min, lat_max, lon_min, lon_max, resolution=d["grid_res"]
    )
    proc_root = _ROOT / d["processed_dir"]

    split_years = {
        "train": _years_in_range(d["train_start"], d["train_end"]),
        "val":   _years_in_range(d["val_start"],   d["val_end"]),
        "test":  _years_in_range(d["test_start"],  d["test_end"]),
    }

    print("OceanEmbed PoC — Preprocessing Pipeline")
    print(f"Config : {cfg_path}")
    print(f"Output : {proc_root}")
    print(f"Grid   : {len(target_lats)} × {len(target_lons)}  @ {d['grid_res']}°  "
          f"({lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E)")
    print(f"Splits : {args.splits}")
    for sp, yrs in split_years.items():
        if sp in args.splits:
            print(f"  {sp}: {yrs}")
    print(f"GLORYS : processed in monthly chunks (peak RAM ~2 GB/year)")

    if args.dry_run:
        print("\n[dry-run] Exiting without processing.")
        return

    total_t0 = time.perf_counter()

    for split in args.splits:
        years   = split_years[split]
        out_dir = proc_root / split
        print(f"\n{'═' * 55}")
        print(f"  Split: {split.upper()}  ({min(years)}–{max(years)})")
        print(f"{'═' * 55}")
        for year in years:
            process_year(
                year, lat_min, lat_max, lon_min, lon_max,
                target_lats, target_lons, out_dir,
            )

    if "train" in args.splits:
        compute_and_save_stats(
            train_dir  = proc_root / "train",
            stats_path = proc_root / "norm_stats.json",
        )

    elapsed = time.perf_counter() - total_t0
    print(f"\n✅ Preprocessing complete in {elapsed / 60:.1f} min.")
    print(f"   Files: {proc_root}")


if __name__ == "__main__":
    main()
