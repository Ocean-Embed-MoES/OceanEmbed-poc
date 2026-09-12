"""
features.py
-----------
Compute physics-informed derived surface features for OceanEmbed.

Three derived features (per the architecture design document):

  ∇SST   log|∇SST|   Log-magnitude of the SST horizontal gradient.
                      Captures ocean fronts and mesoscale eddies.
                      Reduces upper-ocean RMSE compared to raw SST alone.

  ∇SLA   log|∇SLA|   Log-magnitude of the SLA horizontal gradient.
                      Frontal sharpness; geostrophic current gradients.

  WSC    Wind Stress Curl   ∂(τ_y)/∂x − ∂(τ_x)/∂y
                      Ekman pumping (WSC > 0 → upwelling) and suction signal.
                      Computed from the bulk aerodynamic formula:
                        τ = ρ_air · C_D · |U| · U_wind

Design note
-----------
Raw wind components (uwnd, vwnd) are NEVER passed as model inputs.
The architecture document is explicit: "We do NOT include raw U_wind,
V_wind directly."  They are used here solely to compute WSC and are
then discarded.  Only WSC enters the 8-channel feature tensor.
"""
from __future__ import annotations

import numpy as np
import xarray as xr


# ── Physical constants ────────────────────────────────────────────────────────
R_EARTH: float = 6_371_000.0   # Earth mean radius  (m)
RHO_AIR: float = 1.225          # Air density        (kg m⁻³)
C_D:     float = 1.5e-3         # Bulk drag coeff    (dimensionless)
EPS:     float = 1e-6           # Floor for log(magnitude + ε)


# ── Internal: spherical gradient ─────────────────────────────────────────────

def _spherical_gradients(
    da: xr.DataArray,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Physical horizontal gradients (∂f/∂x, ∂f/∂y) in SI units per metre.

    Uses centred finite differences on a regular lat/lon grid, with correct
    spherical-Earth geometry to convert angular spacing to metres:

        ∂f/∂x = (1 / (R·cos φ)) · ∂f/∂λ   [east,  units m⁻¹]
        ∂f/∂y = (1 / R)          · ∂f/∂φ   [north, units m⁻¹]

    where φ is latitude (radians) and λ is longitude (radians).

    numpy.gradient is used with explicit spacing so boundary values are
    computed with one-sided differences automatically.

    Parameters
    ----------
    da      : DataArray with *lat_dim* and *lon_dim* as the last two dimensions.
    lat_dim : Name of the latitude dimension.
    lon_dim : Name of the longitude dimension.

    Returns
    -------
    dfx, dfy : np.ndarray  East and north gradient components.
               Same shape as da.values.
    """
    lats = da[lat_dim].values.astype(np.float64)
    lons = da[lon_dim].values.astype(np.float64)

    dlat_rad = float(np.mean(np.diff(lats))) * (np.pi / 180.0)
    dlon_rad = float(np.mean(np.diff(lons))) * (np.pi / 180.0)

    vals = da.values.astype(np.float64)
    ndim = vals.ndim

    # ── ∂f/∂y (north) ────────────────────────────────────────────────────────
    # Uniform meridional spacing: dy = R · Δφ  (metres)
    dy = R_EARTH * dlat_rad
    dfy = np.gradient(vals, dy, axis=-2)

    # ── ∂f/∂x (east) ─────────────────────────────────────────────────────────
    # numpy.gradient(vals, dlon_rad, axis=-1) gives ∂f/∂λ  [units / radian]
    # Divide by R·cos(lat) to convert to units / metre.
    dfx_per_rad = np.gradient(vals, dlon_rad, axis=-1)

    cos_lat = np.cos(lats * (np.pi / 180.0))               # (nlat,)
    cos_lat = np.where(np.abs(cos_lat) < 1e-10, 1e-10, cos_lat)  # avoid /0 at poles

    # Broadcast to (..., nlat, nlon)
    cos_bc = cos_lat.reshape((1,) * (ndim - 2) + (len(lats), 1))
    dfx = dfx_per_rad / (R_EARTH * cos_bc)

    return dfx.astype(np.float32), dfy.astype(np.float32)


# ── Public feature functions ──────────────────────────────────────────────────

def gradient_logmag(
    da: xr.DataArray,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.DataArray:
    """
    Log-magnitude of the horizontal gradient of a surface field.

        out = log( √((∂f/∂x)² + (∂f/∂y)²) + ε )

    Uses ε = 1e-6 for numerical stability where the gradient is near-zero.

    Parameters
    ----------
    da      : Surface field DataArray (e.g., SST or SLA).
              Must have *lat_dim* and *lon_dim* as its last two dimensions.
    lat_dim : Name of the latitude dimension.
    lon_dim : Name of the longitude dimension.

    Returns
    -------
    xr.DataArray  Same dims and coords as *da*.
    """
    dfx, dfy = _spherical_gradients(da, lat_dim, lon_dim)
    magnitude = np.sqrt(dfx ** 2 + dfy ** 2)
    result    = np.log(magnitude + EPS).astype(np.float32)
    name = f"grad_{da.name}" if da.name else "grad"
    return xr.DataArray(result, dims=da.dims, coords=da.coords, name=name)


def wind_stress_curl(
    uwnd: xr.DataArray,
    vwnd: xr.DataArray,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.DataArray:
    """
    Wind Stress Curl (WSC) from daily-mean 10-m wind components.

        WSC = ∂(τ_y)/∂x − ∂(τ_x)/∂y

    where wind stress uses the bulk aerodynamic formula:
        τ_x = ρ_air · C_D · |U| · uwnd    [N m⁻²]
        τ_y = ρ_air · C_D · |U| · vwnd    [N m⁻²]
        |U| = √(uwnd² + vwnd²)             [m s⁻¹]

    Constants: ρ_air = 1.225 kg m⁻³,  C_D = 1.5 × 10⁻³.

    Positive WSC → Ekman suction (upwelling).
    Negative WSC → Ekman pumping (downwelling).

    Parameters
    ----------
    uwnd, vwnd : Eastward and northward 10-m wind components (m s⁻¹).
                 Must share the same dims and coords.
    lat_dim    : Name of the latitude dimension.
    lon_dim    : Name of the longitude dimension.

    Returns
    -------
    xr.DataArray  WSC field (N m⁻³), same dims/coords as *uwnd*.
    """
    u = uwnd.values.astype(np.float64)
    v = vwnd.values.astype(np.float64)

    wind_speed = np.sqrt(u ** 2 + v ** 2)           # |U|  (m s⁻¹)
    tau_x = RHO_AIR * C_D * wind_speed * u           # τ_x  (N m⁻²)
    tau_y = RHO_AIR * C_D * wind_speed * v           # τ_y  (N m⁻²)

    tx_da = xr.DataArray(tau_x, dims=uwnd.dims, coords=uwnd.coords)
    ty_da = xr.DataArray(tau_y, dims=vwnd.dims, coords=vwnd.coords)

    # ∂(τ_y)/∂x  and  ∂(τ_x)/∂y  in N m⁻³
    dtau_y_dx, _         = _spherical_gradients(ty_da, lat_dim, lon_dim)
    _,         dtau_x_dy = _spherical_gradients(tx_da, lat_dim, lon_dim)

    wsc = (dtau_y_dx - dtau_x_dy).astype(np.float32)

    return xr.DataArray(
        wsc,
        dims=uwnd.dims,
        coords=uwnd.coords,
        name="wsc",
        attrs={
            "long_name": "Wind Stress Curl",
            "units":     "N m-3",
            "positive":  "upwelling (Ekman suction)",
        },
    )
