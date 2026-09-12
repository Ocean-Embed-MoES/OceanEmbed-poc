#!/usr/bin/env python3
"""
scripts/prepare_data.py
-----------------------
End-to-end preprocessing pipeline for OceanEmbed PoC.

What this script does
---------------------
For each split (train 2015–2017, val 2018, test 2019–2020), and for each
calendar year within the split:

  1. Load all 5 PS-required surface variables from the local SSD cache.
  2. Regrid every variable to the common 0.5° × 0.5° NIO target grid.
  3. Compute derived features:
       ∇SST  — log-magnitude SST gradient
       ∇SLA  — log-magnitude SLA gradient
       WSC   — Wind Stress Curl  (winds discarded afterwards)
  4. Stack into an 8-channel input array  (T, 8, 50, 120).
  5. Load GLORYS, regrid to 0.5°, and interpolate from native depth levels
     to the 15 PS-standard depths.
  6. Save inputs and targets as NetCDF files in outputs/processed/{split}/.

After all years are saved:
  7. Compute Z-score normalization statistics from the TRAINING SET ONLY.
  8. Save stats to outputs/processed/norm_stats.json.

Usage
-----
  python scripts/prepare_data.py               # uses configs/poc.yaml
  python scripts/prepare_data.py --config path/to/other.yaml
  python scripts/prepare_data.py --splits train val   # process specific splits
  python scripts/prepare_data.py --dry-run            # print plan, do nothing

Notes
-----
- The script processes one year at a time to keep peak memory low.
- Nothing is normalized here — raw physical-unit data is saved.
  Normalization is applied on-the-fly in dataset.py.
- Re-running overwrites existing output files for the same year.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Allow imports from the project root regardless of working directory
_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import xarray as xr
import yaml

from src.data.cache_loader import (
    get_currents,
    get_glorys,
    get_ssh,
    get_sss,
    get_sst,
    get_winds,
)
from src.preprocessing.features  import gradient_logmag, wind_stress_curl
from src.preprocessing.normalize import compute_channel_stats, save_stats
from src.preprocessing.regrid    import make_target_grid, regrid


# ── Constants ─────────────────────────────────────────────────────────────────
# These must match src/data/dataset.py exactly.
CHANNEL_NAMES   = [
    "sst", "grad_sst", "sla", "grad_sla",
    "sss", "wsc", "u_curr", "v_curr",
]
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_LABELS    = [f"d{d}m" for d in STANDARD_DEPTHS]


# ── Per-year processing ────────────────────────────────────────────────────────

def process_year(
    year: int,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    out_dir: pathlib.Path,
) -> None:
    """
    Process one calendar year end-to-end and save the results to *out_dir*.

    Output files
    ------------
    inputs_{year}.nc   — shape (T, 8, H, W),  float32, physical units
    targets_{year}.nc  — shape (T, 15, H, W), float32, °C
    """
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    t0    = time.perf_counter()

    print(f"\n  ── {year} ({start} → {end}) ──")

    # ── 1. Load raw surface inputs ─────────────────────────────────────────
    print("    Loading SST ...", end=" ", flush=True)
    sst_raw = get_sst(start, end, lat_min, lat_max, lon_min, lon_max)
    sst_raw.name = "sst"
    print(f"{sst_raw.shape}")

    print("    Loading SLA ...", end=" ", flush=True)
    sla_raw = get_ssh(start, end, lat_min, lat_max, lon_min, lon_max)
    sla_raw.name = "sla"
    print(f"{sla_raw.shape}")

    print("    Loading SSS ...", end=" ", flush=True)
    sss_raw = get_sss(start, end, lat_min, lat_max, lon_min, lon_max)
    sss_raw.name = "sss"
    print(f"{sss_raw.shape}")

    print("    Loading currents ...", end=" ", flush=True)
    curr_raw = get_currents(start, end, lat_min, lat_max, lon_min, lon_max)
    print(f"u={curr_raw['u'].shape}")

    print("    Loading winds (6-h → daily) ...", end=" ", flush=True)
    wind_raw = get_winds(start, end, lat_min, lat_max, lon_min, lon_max)
    print(f"uwnd={wind_raw['uwnd'].shape}")

    # ── 2. Load GLORYS target ───────────────────────────────────────────────
    print("    Loading GLORYS ...", end=" ", flush=True)
    glorys_raw = get_glorys(start, end, lat_min, lat_max, lon_min, lon_max)
    thetao     = glorys_raw["thetao"]
    print(f"thetao={thetao.shape}")

    native_depths = thetao.depth.values.astype(float)
    print(f"    GLORYS native depths: {len(native_depths)} levels "
          f"[{native_depths[0]:.2f} … {native_depths[-1]:.2f} m]")

    # ── 3. Regrid all surface inputs to 0.5° target grid ───────────────────
    print("    Regriding to 0.5° ...")
    sst_rg     = regrid(sst_raw,          target_lats, target_lons)
    sla_rg     = regrid(sla_raw,          target_lats, target_lons)
    sss_rg     = regrid(sss_raw,          target_lats, target_lons)
    u_curr_rg  = regrid(curr_raw["u"],    target_lats, target_lons)
    v_curr_rg  = regrid(curr_raw["v"],    target_lats, target_lons)
    uwnd_rg    = regrid(wind_raw["uwnd"], target_lats, target_lons)
    vwnd_rg    = regrid(wind_raw["vwnd"], target_lats, target_lons)

    # Force computation (triggers lazy I/O) before feature derivation
    sst_rg    = sst_rg.load()
    sla_rg    = sla_rg.load()
    sss_rg    = sss_rg.load()
    u_curr_rg = u_curr_rg.load()
    v_curr_rg = v_curr_rg.load()
    uwnd_rg   = uwnd_rg.load()
    vwnd_rg   = vwnd_rg.load()

    # ── 4. Compute derived features ─────────────────────────────────────────
    print("    Computing ∇SST, ∇SLA, WSC ...")
    grad_sst = gradient_logmag(sst_rg)
    grad_sla = gradient_logmag(sla_rg)
    wsc      = wind_stress_curl(uwnd_rg, vwnd_rg)
    # uwnd_rg and vwnd_rg are no longer needed

    # ── 5. Stack into (T, 8, H, W) ─────────────────────────────────────────
    # Channel order must match CHANNEL_NAMES above.
    channels = [
        sst_rg.values,      # 0  SST
        grad_sst.values,    # 1  ∇SST
        sla_rg.values,      # 2  SLA
        grad_sla.values,    # 3  ∇SLA
        sss_rg.values,      # 4  SSS
        wsc.values,         # 5  WSC
        u_curr_rg.values,   # 6  U_curr
        v_curr_rg.values,   # 7  V_curr
    ]
    inputs = np.stack(channels, axis=1).astype(np.float32)  # (T, 8, H, W)
    print(f"    inputs  shape={inputs.shape}  dtype={inputs.dtype}")

    # Retrieve the time axis from the regrided SST
    times = sst_rg.time.values

    # ── 6. Regrid GLORYS and interpolate to 15 PS-standard depths ───────────
    print("    Regriding GLORYS + interpolating to 15 standard depths ...")
    thetao_rg = regrid(
        thetao, target_lats, target_lons,
        lat_dim="latitude", lon_dim="longitude",
    ).load()

    # Linear interpolation along depth dimension.
    # fill_value="extrapolate" handles the 0 m surface depth, which is
    # slightly shallower than GLORYS' first native level (≈ 0.494 m).
    thetao_interp = thetao_rg.interp(
        depth=np.array(STANDARD_DEPTHS, dtype=np.float64),
        method="linear",
        kwargs={"fill_value": "extrapolate", "bounds_error": False},
    )
    targets = thetao_interp.values.astype(np.float32)  # (T, 15, H, W)
    print(f"    targets shape={targets.shape}  dtype={targets.dtype}")

    # ── 7. Save to NetCDF ───────────────────────────────────────────────────
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
            "description": "OceanEmbed PoC — preprocessed 8-channel surface inputs",
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
    ds_inputs.to_netcdf(inp_path,  format="NETCDF4")
    ds_targets.to_netcdf(tgt_path, format="NETCDF4")

    elapsed = time.perf_counter() - t0
    print(f"    Saved {inp_path.name}  ({inp_path.stat().st_size / 1e6:.0f} MB)")
    print(f"    Saved {tgt_path.name} ({tgt_path.stat().st_size / 1e6:.0f} MB)")
    print(f"    Done in {elapsed:.0f}s")


# ── Normalization stats from training files ────────────────────────────────────

def compute_and_save_stats(
    train_dir: pathlib.Path,
    stats_path: pathlib.Path,
) -> None:
    """
    Load all training-year inputs and targets, compute per-channel / per-depth
    Z-score statistics, and save to *stats_path*.

    Reading from saved NetCDF files rather than keeping year arrays in memory
    avoids holding 3 years of data in RAM simultaneously.
    """
    print("\n  Computing normalization statistics from training data ...")

    input_files  = sorted(train_dir.glob("inputs_*.nc"))
    target_files = sorted(train_dir.glob("targets_*.nc"))

    all_inputs  = np.concatenate(
        [xr.open_dataset(f)["inputs"].values  for f in input_files],  axis=0
    ).astype(np.float32)
    all_targets = np.concatenate(
        [xr.open_dataset(f)["targets"].values for f in target_files], axis=0
    ).astype(np.float32)

    print(f"    Training inputs  shape: {all_inputs.shape}")
    print(f"    Training targets shape: {all_targets.shape}")

    input_stats  = compute_channel_stats(all_inputs,  CHANNEL_NAMES,  channel_axis=1)
    target_stats = compute_channel_stats(all_targets, DEPTH_LABELS,   channel_axis=1)

    stats = {"inputs": input_stats, "targets": target_stats}
    save_stats(stats, stats_path)

    print(f"  Saved: {stats_path}")
    print("\n  Input channel statistics:")
    print(f"  {'channel':<12}  {'mean':>10}  {'std':>10}")
    print(f"  {'-'*38}")
    for name, s in input_stats.items():
        print(f"  {name:<12}  {s['mean']:>10.4f}  {s['std']:>10.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _years_in_range(start: str, end: str) -> list[int]:
    return list(range(int(start[:4]), int(end[:4]) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OceanEmbed PoC — end-to-end preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/poc.yaml",
        help="Path to YAML config (default: configs/poc.yaml)",
    )
    parser.add_argument(
        "--splits", nargs="+", choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Which splits to process (default: all three)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the processing plan and exit without doing anything",
    )
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text())

    d        = cfg["data"]
    lat_min  = d["lat_min"]
    lat_max  = d["lat_max"]
    lon_min  = d["lon_min"]
    lon_max  = d["lon_max"]

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
    print(f"Config:   {cfg_path}")
    print(f"Output:   {proc_root}")
    print(f"Grid:     {len(target_lats)} × {len(target_lons)}  "
          f"@ {d['grid_res']}°  ({lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E)")
    print(f"Splits:   {args.splits}")
    for sp, yrs in split_years.items():
        if sp in args.splits:
            print(f"  {sp}: {yrs}")

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

    # Normalization stats — only after training files are written
    if "train" in args.splits:
        compute_and_save_stats(
            train_dir  = proc_root / "train",
            stats_path = proc_root / "norm_stats.json",
        )

    elapsed = time.perf_counter() - total_t0
    print(f"\n✅ Preprocessing complete in {elapsed / 60:.1f} min.")
    print(f"   Processed files: {proc_root}")


if __name__ == "__main__":
    main()
