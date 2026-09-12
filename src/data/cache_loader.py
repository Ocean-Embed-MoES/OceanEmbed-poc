"""
cache_loader.py
---------------
Read-only access layer to the pre-cached OceanEmbed dataset.

All data was pre-downloaded by the local-cache-framework notebook and stored
as NetCDF files on the local SSD.  This module reads those files directly —
no network access, no credentials required.

Cache layout
------------
  /run/media/mohan-krishna/vamSHE/data-pool/
    cache/
      sst/            *.nc  (time, latitude, longitude)  °C
      sss/            *.nc  (time, latitude, longitude)  PSU
      ssh/            *.nc  (time, latitude, longitude)  m  (SLA)
      currents/       *.nc  (time, latitude, longitude)  m/s  [global]
      winds/          *.nc  (time, latitude, longitude)  m/s  [global, 6-hourly]
      glorys/         *.nc  (time, depth, latitude, longitude)  °C
      argo/           *.nc  (time, depth, latitude, longitude)  °C  [ARMOR3D]
      incois_argo_vam/        *.nc  (time, ZAX, latitude, longitude)
      incois_argo_mccreary/   *.nc  (time, ZAX, latitude, longitude)
    meta/
      cache_index.json        chunk registry used by the cache engine

Public functions
----------------
  get_sst       → xr.DataArray  (time, latitude, longitude)  °C
  get_sss       → xr.DataArray  (time, latitude, longitude)  PSU
  get_ssh       → xr.DataArray  (time, latitude, longitude)  m  (SLA)
  get_currents  → xr.Dataset    {u, v}  m/s
  get_winds     → xr.Dataset    {uwnd, vwnd}  m/s  [daily-mean, NIO slice]
  get_glorys    → xr.Dataset    {thetao}  °C  (time, depth, lat, lon)
  get_armor3d   → xr.Dataset    {to}  °C   [supplementary validation only]
  get_incois_vam       → xr.Dataset  {TEMP, TERR, SAL, SERR}  10-day
  get_incois_mccreary  → xr.Dataset  {T_ANALYZED, ...}        10-day
"""
from __future__ import annotations

import json
import pathlib
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


# ── Cache paths ──────────────────────────────────────────────────────────────
_DATA_ROOT   = pathlib.Path("/run/media/mohan-krishna/vamSHE/data-pool")
_CACHE_ROOT  = _DATA_ROOT / "cache"
_CACHE_INDEX = _DATA_ROOT / "meta" / "cache_index.json"


# ── Internal helpers ─────────────────────────────────────────────────────────

def _load_index() -> dict:
    if not _CACHE_INDEX.exists():
        raise FileNotFoundError(
            f"Cache index not found: {_CACHE_INDEX}\n"
            "Make sure the SSD is mounted at /run/media/mohan-krishna/vamSHE/"
        )
    return json.loads(_CACHE_INDEX.read_text())


def _bbox_covers(
    entry: dict,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> bool:
    """True if the cached chunk's bbox fully contains the requested bbox."""
    try:
        return (
            float(entry["lat_min"]) <= lat_min
            and float(entry["lat_max"]) >= lat_max
            and float(entry["lon_min"]) <= lon_min
            and float(entry["lon_max"]) >= lon_max
        )
    except (KeyError, ValueError, TypeError):
        return False


def _norm_time(
    obj: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """Coerce 'time' coordinate to datetime64[ns] (handles cftime objects)."""
    if "time" not in getattr(obj, "dims", getattr(obj, "sizes", {})):
        return obj
    try:
        t = pd.DatetimeIndex(obj.time.values).to_numpy().astype("datetime64[ns]")
    except Exception:
        t = np.array([np.datetime64(str(v)[:19], "ns") for v in obj.time.values])
    return obj.assign_coords(time=t)


def _load_chunks(
    variable: str,
    start: str,
    end: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.Dataset:
    """
    Find cached NetCDF chunks for *variable* that spatially cover the
    requested bbox, merge them along the time axis, and slice to [start, end].
    """
    idx = _load_index()

    covering = [
        e for e in idx.values()
        if e.get("variable") == variable
        and _bbox_covers(e, lat_min, lat_max, lon_min, lon_max)
        and pathlib.Path(e["file"]).exists()
    ]
    if not covering:
        raise RuntimeError(
            f"No cached chunks for variable='{variable}' covering "
            f"bbox=({lat_min},{lat_max},{lon_min},{lon_max}).\n"
            "Check that the SSD is mounted and the cache is populated."
        )

    req_s = date.fromisoformat(start)
    req_e = date.fromisoformat(end)
    relevant = sorted(
        [
            c for c in covering
            if date.fromisoformat(c["end"])   >= req_s
            and date.fromisoformat(c["start"]) <= req_e
        ],
        key=lambda c: c["start"],
    )
    if not relevant:
        raise RuntimeError(
            f"No cache chunks for variable='{variable}' "
            f"overlap with {start}→{end}."
        )

    if len(relevant) == 1:
        ds = xr.open_dataset(relevant[0]["file"])
    else:
        ds = xr.open_mfdataset(
            [c["file"] for c in relevant],
            combine    = "by_coords",
            use_cftime = True,   # force cftime; _norm_time converts to datetime64[ns]
        )

    ds = _norm_time(ds)
    if "time" in ds.sizes:
        ds = ds.sel(time=slice(start, end))
    return ds


# ── Public API ───────────────────────────────────────────────────────────────

def get_sst(
    start: str,
    end: str,
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lon_min: float = 45.0,
    lon_max: float = 105.0,
) -> xr.DataArray:
    """
    Sea Surface Temperature — Copernicus OSTIA (0.05°, daily).

    Cached extent: 2015–2020, NIO (5–30°N, 45–105°E).

    Returns
    -------
    xr.DataArray  (time, latitude, longitude)  °C
    """
    ds = _load_chunks("sst", start, end, lat_min, lat_max, lon_min, lon_max)
    da = ds["analysed_sst"]
    da.name = "sst"
    return da


def get_sss(
    start: str,
    end: str,
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lon_min: float = 45.0,
    lon_max: float = 105.0,
) -> xr.DataArray:
    """
    Sea Surface Salinity — Copernicus SMAP/SMOS (0.125°, daily).

    Cached extent: 2015–2020, NIO.

    Returns
    -------
    xr.DataArray  (time, latitude, longitude)  PSU
    """
    ds = _load_chunks("sss", start, end, lat_min, lat_max, lon_min, lon_max)
    da = ds["sos"]
    da.name = "sss"
    return da


def get_ssh(
    start: str,
    end: str,
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lon_min: float = 45.0,
    lon_max: float = 105.0,
) -> xr.DataArray:
    """
    Sea Level Anomaly — Copernicus DUACS (0.25°, daily).

    Used as the SSH proxy.  The problem statement lists SSH/SLA as
    interchangeable inputs; DUACS provides SLA (sea level anomaly).

    Cached extent: 2015–2020, NIO.

    Returns
    -------
    xr.DataArray  (time, latitude, longitude)  m
    """
    ds = _load_chunks("ssh", start, end, lat_min, lat_max, lon_min, lon_max)
    da = ds["sla"]
    da.name = "sla"
    return da


def _find_global_chunks(variable: str, start: str, end: str) -> list:
    """
    Return cache chunks for *variable* that overlap [start, end] in time,
    without any bbox filter.  Used for globally-cached datasets (currents,
    winds) where the stored bbox varies across years (-180/180 vs 0/360).
    """
    idx   = _load_index()
    req_s = date.fromisoformat(start)
    req_e = date.fromisoformat(end)
    return sorted(
        [
            e for e in idx.values()
            if e.get("variable") == variable
            and pathlib.Path(e["file"]).exists()
            and date.fromisoformat(e["end"])   >= req_s
            and date.fromisoformat(e["start"]) <= req_e
        ],
        key=lambda e: e["start"],
    )


def _promote_latlon(ds: xr.Dataset) -> xr.Dataset:
    """
    OSCAR stores float coordinates as non-dimension 'lat'/'lon' variables
    while the actual array dimensions are integer-indexed 'latitude'/'longitude'.
    This helper promotes 'lat'/'lon' to proper dimension coordinates and renames
    the dimensions so that xarray's label-based .sel() works correctly.
    """
    rename_dims: dict[str, str] = {}
    assign: dict[str, xr.DataArray] = {}

    # Check if lat/lon are non-dimension coordinates that match a dimension
    for src, dim in [("lat", "latitude"), ("lon", "longitude")]:
        if src in ds.coords and dim in ds.dims and src not in ds.dims:
            assign[dim] = ds[src]   # promote float coord to dimension coord
    if assign:
        ds = ds.assign_coords(assign)

    return ds


def get_currents(
    start: str,
    end: str,
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lon_min: float = 45.0,
    lon_max: float = 105.0,
) -> xr.Dataset:
    """
    Surface ocean currents — NASA OSCAR V2.0 (0.25°, daily).

    Cached at global extent; this function slices to the requested bbox.
    OSCAR's cached files use integer-indexed dimensions with float 'lat'/'lon'
    non-dimension coordinates — this is handled transparently.

    Returns
    -------
    xr.Dataset  variables: u, v  (m s⁻¹)
                dims: (time, latitude, longitude)
    """
    chunks = _find_global_chunks("currents", start, end)
    if not chunks:
        raise RuntimeError(
            f"No cached currents chunks overlap with {start}→{end}."
        )

    if len(chunks) == 1:
        ds = _norm_time(xr.open_dataset(chunks[0]["file"]))
    else:
        # Open each chunk independently and normalise its time before concat.
        # This avoids xr.combine_by_coords crashing on mixed calendar types
        # (e.g. OSCAR 2020 uses 'julian' while 2019 uses 'proleptic_gregorian').
        parts = [_norm_time(xr.open_dataset(c["file"])) for c in chunks]
        ds    = xr.concat(parts, dim="time").sortby("time")
        # Drop duplicate timestamps that can arise when chunk boundaries overlap
        _, unique_idx = np.unique(ds.time.values, return_index=True)
        ds = ds.isel(time=unique_idx)

    if "time" in ds.sizes:
        ds = ds.sel(time=slice(start, end))

    # Promote non-dimension lat/lon floats to proper dimension coordinates
    ds = _promote_latlon(ds)

    # Now slice by float coordinate value
    ds = ds.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
    return ds[["u", "v"]]


def get_winds(
    start: str,
    end: str,
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lon_min: float = 45.0,
    lon_max: float = 105.0,
) -> xr.Dataset:
    """
    10-m surface winds — NASA CCMP V3.1 (0.25°, 6-hourly → daily mean).

    Wind components are returned for use ONLY in computing Wind Stress Curl
    (WSC).  Raw uwnd/vwnd are never fed to the model as inputs.

    Returns
    -------
    xr.Dataset  variables: uwnd, vwnd  (m s⁻¹)
                dims: (time, latitude, longitude)  [daily mean]
    """
    chunks = _find_global_chunks("winds", start, end)
    if not chunks:
        raise RuntimeError(
            f"No cached winds chunks overlap with {start}→{end}."
        )

    ds = (
        xr.open_dataset(chunks[0]["file"])
        if len(chunks) == 1
        else xr.open_mfdataset([c["file"] for c in chunks], combine="by_coords", use_cftime=True)
    )
    ds = _norm_time(ds)
    if "time" in ds.sizes:
        ds = ds.sel(time=slice(start, end))

    # CCMP uses latitude/longitude as true dimension coordinates
    lat_dim = "latitude" if "latitude" in ds.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds.dims else "lon"
    ds = ds.sel({
        lat_dim: slice(lat_min, lat_max),
        lon_dim: slice(lon_min, lon_max),
    })
    if lat_dim != "latitude" or lon_dim != "longitude":
        ds = ds.rename({lat_dim: "latitude", lon_dim: "longitude"})

    # Resample 6-hourly → daily mean
    return ds[["uwnd", "vwnd"]].resample(time="1D").mean()


def get_glorys(
    start: str,
    end: str,
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lon_min: float = 45.0,
    lon_max: float = 105.0,
) -> xr.Dataset:
    """
    3D ocean temperature — Copernicus GLORYS12 reanalysis (0.083°, daily).

    This is the training target.  Native depth levels are non-round values
    (0.494 m → ~1062 m); use glorys['thetao'].depth.values to inspect them.
    The preprocessing pipeline interpolates these to the 15 PS-standard depths.

    Cached extent: 2015–2020, NIO.

    Returns
    -------
    xr.Dataset  variable: thetao  (°C)
                dims: (time, depth, latitude, longitude)
    """
    ds = _load_chunks("glorys", start, end, lat_min, lat_max, lon_min, lon_max)
    return ds[["thetao"]]


def get_armor3d(
    start: str,
    end: str,
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lon_min: float = 45.0,
    lon_max: float = 105.0,
) -> xr.Dataset:
    """
    Gridded Argo temperature (ARMOR3D) — Copernicus (0.125°, daily).

    Used as SUPPLEMENTARY validation only.
    Primary independent validation uses INCOIS VAM + McCreary.

    Cached extent: 2015–2020, NIO.

    Returns
    -------
    xr.Dataset  variable: to  (°C)
                dims: (time, depth, latitude, longitude)
    """
    ds = _load_chunks("argo", start, end, lat_min, lat_max, lon_min, lon_max)
    return ds[["to"]]


def get_incois_vam(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> xr.Dataset:
    """
    INCOIS Argo — Variational Analysis Methodology.
    PRIMARY independent validation dataset (true in-situ, never in training).

    Fully cached as a single file; always loads instantly.

    Returns
    -------
    xr.Dataset  variables: TEMP (°C), TERR, SAL, SERR
                dims: (time, ZAX, latitude, longitude)
                ZAX: 24 depth levels [5, 10, 20, ..., 2000 m]
                cadence: 10-day  |  resolution: 1°
                coverage: 2004–2026, 29.5°S–29.5°N, 30.5–119.5°E
    """
    fp = _CACHE_ROOT / "incois_argo_vam"
    files = sorted(fp.glob("*.nc"))
    if not files:
        raise RuntimeError(f"INCOIS VAM not cached — no .nc files in {fp}")
    ds = (
        xr.open_mfdataset(files, combine="by_coords", use_cftime=True)
        if len(files) > 1
        else xr.open_dataset(files[0])
    )
    ds = _norm_time(ds)
    if start or end:
        ds = ds.sel(time=slice(start, end))
    return ds


def get_incois_mccreary(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> xr.Dataset:
    """
    INCOIS Argo — Kessler-McCreary Methodology.
    PRIMARY independent validation dataset (true in-situ, never in training).

    Cached in 4 chunks covering 2001–2026; merged seamlessly.

    Returns
    -------
    xr.Dataset  variables: T_ANALYZED, T_MEAN, T_STDEV, T_RMSE, ...
                dims: (time, ZAX, latitude, longitude)
                ZAX: 24 depth levels [5, 10, 20, ..., 2000 m]
                cadence: 10-day  |  resolution: 1°
                coverage: 2001–2026, 29.5°S–29.5°N, 30.5–119.5°E
    """
    fp = _CACHE_ROOT / "incois_argo_mccreary"
    files = sorted(fp.glob("*.nc"))
    if not files:
        raise RuntimeError(f"INCOIS McCreary not cached — no .nc files in {fp}")
    ds = (
        xr.open_mfdataset(files, combine="by_coords", use_cftime=True)
        if len(files) > 1
        else xr.open_dataset(files[0])
    )
    ds = _norm_time(ds)
    if start or end:
        ds = ds.sel(time=slice(start, end))
    return ds
