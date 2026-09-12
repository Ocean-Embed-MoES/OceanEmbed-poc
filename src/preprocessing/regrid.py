"""
regrid.py
---------
Regrid surface variable DataArrays to the common PoC target grid.

Target grid
-----------
  Resolution : 0.5° × 0.5°
  Domain     : North Indian Ocean — 5–30°N, 45–105°E
  Shape      : 50 latitude × 120 longitude cells
  Cell centres: offset half a cell inward from the domain boundary
                lat [5.25, 5.75, ..., 29.75]
                lon [45.25, 45.75, ..., 104.75]

Method
------
Bilinear interpolation via xarray.DataArray.interp (backed by scipy).
This is appropriate for smooth oceanographic fields and handles the
resolution differences between inputs (0.05°–0.25°) and the target (0.5°).
NaN pixels (land, missing data) propagate naturally through interpolation.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


def make_target_grid(
    lat_min: float = 5.0,
    lat_max: float = 30.0,
    lon_min: float = 45.0,
    lon_max: float = 105.0,
    resolution: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the target latitude and longitude coordinate arrays.

    Cell centres are placed at half-resolution offset from the domain edge,
    so the grid cells align cleanly with integer-degree boundaries.

    Parameters
    ----------
    lat_min, lat_max : float  Domain latitude bounds (degrees N).
    lon_min, lon_max : float  Domain longitude bounds (degrees E).
    resolution       : float  Grid cell size (degrees).

    Returns
    -------
    lats, lons : np.ndarray  1-D float64 coordinate arrays.

    Examples
    --------
    >>> lats, lons = make_target_grid()
    >>> lats.shape, lons.shape
    ((50,), (120,))
    >>> lats[0], lats[-1]
    (5.25, 29.75)
    """
    half = resolution / 2.0
    lats = np.arange(lat_min + half, lat_max, resolution)
    lons = np.arange(lon_min + half, lon_max, resolution)
    return lats, lons


def regrid(
    da: xr.DataArray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.DataArray:
    """
    Bilinear interpolation of a DataArray onto the target lat/lon grid.

    Works for any number of leading dimensions (time, depth, …).
    Out-of-range pixels are set to NaN (no extrapolation).

    Parameters
    ----------
    da           : Input DataArray.  Must have *lat_dim* and *lon_dim*.
    target_lats  : Target latitude coordinates (degrees N).
    target_lons  : Target longitude coordinates (degrees E).
    lat_dim      : Name of the latitude dimension in *da*.
    lon_dim      : Name of the longitude dimension in *da*.

    Returns
    -------
    xr.DataArray  On the target grid; all other dimensions preserved.
    """
    return da.interp(
        {lat_dim: target_lats, lon_dim: target_lons},
        method="linear",
    )
