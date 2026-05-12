"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import re
import datetime as dt
import numpy as np
import xarray as xr


def x2xdb_np(xls):
    """
    Convert linear-scale values to log scale (dB).

    Parameters
    ----------
    xls : float or array
        Values in linear scale.

    Returns
    -------
    xdb : float or array
        Values in dB scale.

    Notes
    -----
    .. math::  dBx = 10log_{10}x

    Examples
    --------
    >>> # Convert radar reflectivity in linear scale (mm^6 m^-3) to dBZ.
    >>> import towerpy as tp
    >>> zls = 39811
    >>> zdbz = tp.utils.radutilities.x2xdb(zls)
    >>> print(zdbz)
    Out[0]: 46.000030866277406
    """
    xls = np.array(xls)
    xdb = 10*np.log10(xls)
    return xdb


def xdb2x_np(xdb):
    """
    Convert log scale (dB) values to linear-scale.

    Parameters
    ----------
    xdb : float or array
        Values in dB scale.

    Returns
    -------
    xls : float or array
        Values in linear scale.

    Notes
    -----
    .. math::  x = 10^{0.1*dBx}

    Examples
    --------
    >>> # Convert radar reflectivity in dBZ to linear scale (mm^6 m^-3).
    >>> import towerpy as tp
    >>> dbz = 50
    >>> zls = tp.utils.radutilities.xdb2x(dbz)
    >>> print(zls)
    Out[0]: 100000.0
    """
    xdb = np.array(xdb)
    xls = 10 ** (0.1*xdb)
    return xls


# =============================================================================
# %% xarray implementation
# =============================================================================

DISTANCE_UNITS = {
    "m":  {"m", "meter", "meters", "metre", "metres"},
    "km": {"km", "kilometer", "kilometers", "kilometre", "kilometres"},
    }

ANGLE_UNITS = {
    "deg": {"deg", "degree", "degrees",
            "degrees_north", "degrees_east"  # not sure
            },
    "rad": {"rad", "radian", "radians"},
    }

# Reverse lookup: map any known string → canonical unit
UNIT_LOOKUP = {
    u: canon for canon, variants in {**DISTANCE_UNITS, **ANGLE_UNITS}.items()
    for u in variants}

# Conversion logic

def convert(var, target_unit: str):
    """
    Convert a DataArray to the given canonical unit.

    Parameters
    ----------
    var : xr.DataArray
        Input array with a 'units' attribute.
    target_unit : str
        Canonical target unit, e.g. "m", "km", "rad", "deg".

    Returns
    -------
    xr.DataArray
        Converted array with updated 'units' attribute.

    Raises
    ------
    ValueError
        If the input unit is unknown or incompatible with the target.
    """
    target_unit = target_unit.strip().lower()

    if target_unit not in UNIT_LOOKUP.values():
        raise ValueError(f"Unknown target unit {target_unit!r}")

    # Extract and normalise input unit
    raw = var.attrs.get("units", "").strip().lower()
    if not raw:
        raise ValueError("No 'units' attribute found on the DataArray. "
                         "Please set var.attrs['units'] to one of: "
                         f"{sorted(UNIT_LOOKUP.keys())}")
    src_unit = UNIT_LOOKUP.get(raw)
    if src_unit is None:
        raise ValueError(f"Unsupported unit {raw!r}. Check var.attrs['units']"
                         f" Expected one of: {sorted(UNIT_LOOKUP.keys())}")
    # Identity
    if src_unit == target_unit:
        return var

    # Distance conversions
    if src_unit in DISTANCE_UNITS and target_unit in DISTANCE_UNITS:
        return _convert_distance(var, src_unit, target_unit)

    # Angle conversions
    if src_unit in ANGLE_UNITS and target_unit in ANGLE_UNITS:
        return _convert_angle(var, src_unit, target_unit)

    raise ValueError(
        f"Incompatible unit conversion: {src_unit!r} → {target_unit!r}")


# Unit converters

def _convert_distance(var, src, dst):
    out = var.copy()

    if src == "m" and dst == "km":
        out = out / 1000
    elif src == "km" and dst == "m":
        out = out * 1000
    else:
        raise ValueError(f"Unsupported distance conversion {src} → {dst}")

    out.attrs = var.attrs.copy()
    out.attrs["units"] = dst
    return out


def _convert_angle(var, src, dst):
    out = var.copy()

    if src == "deg" and dst == "rad":
        out = np.deg2rad(out)
    elif src == "rad" and dst == "deg":
        out = np.rad2deg(out)
    else:
        raise ValueError(f"Unsupported angle conversion {src} → {dst}")

    out.attrs = var.attrs.copy()
    out.attrs["units"] = dst
    return out

# Scale conversions

def x2xdb(xls):
    """
    Convert linear-scale values to logarithmic scale (dB).

    Parameters
    ----------
    xls : float, array-like, or xr.DataArray
        Values in linear scale.

    Returns
    -------
    xdb : same type as input
        Values in dB scale.

    Notes
    -----
    dB(x) = 10 * log10(x)
    """
    if isinstance(xls, xr.DataArray):
        out = 10 * np.log10(xls)
        out.attrs = xls.attrs.copy()
        out.attrs["scale"] = "dB"
        return out

    xls = np.asarray(xls)
    return 10 * np.log10(xls)


def xdb2x(xdb):
    """
    Convert logarithmic-scale (dB) values to linear scale.

    Parameters
    ----------
    xdb : float, array-like, or xr.DataArray
        Values in dB scale.

    Returns
    -------
    xls : same type as input
        Values in linear scale.

    Notes
    -----
    x = 10 ** (0.1 * dB)
    """
    if isinstance(xdb, xr.DataArray):
        out = 10 ** (0.1 * xdb)
        out.attrs = xdb.attrs.copy()
        out.attrs["scale"] = "linear"
        return out

    xdb = np.asarray(xdb)
    return 10 ** (0.1 * xdb)


def dtm_to_np64(dt):
    """
    Convert python datetime to numpy.datetime64 with microsecond precision.
    """
    return np.datetime64(dt, 'us')

def np64_to_dtm(t):
    """
    Convert numpy.datetime64, xarray.DataArray, or integer nanoseconds
    to Python datetime.datetime safely.
    """

    # Case 0: xarray DataArray → extract scalar
    if isinstance(t, xr.DataArray):
        t = t.item()

    # Case 1: already Python datetime
    if isinstance(t, dt.datetime):
        return t

    # Case 2: numpy.datetime64
    if isinstance(t, np.datetime64):
        # Downcast to milliseconds to avoid overflow
        t_ms = t.astype('datetime64[ms]').astype('int64')
        return dt.datetime.utcfromtimestamp(t_ms / 1000)

    # Case 3: integer nanoseconds since epoch
    if isinstance(t, (int, np.integer)):
        # Convert ns → ms
        t_ms = t // 1_000_000
        return dt.datetime.utcfromtimestamp(t_ms / 1000)

    raise TypeError(f"Unsupported datetime type: {type(t)}")

UNIT_ALIASES = {
    "meterspersecond": "m/s",
    "metersperseconds": "m/s",
    "m/s": "m/s",
    "mm/h": "mm/h",
    "mmh-1": "mm/h",
    "mmperhour": "mm/h",
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "deg/km": "deg/km",
    "degree/km": "deg/km",
    "degrees/km": "deg/km",
    'degreesperkilometer': "deg/km",
    "dbm": "dBm",
    "dbz": "dBZ",
    "db": "dB",
    "unorm": "unorm",
    "relative_frequency": "unorm",
    }

UNIT_ALIASES = {"m/s": "m/s",
                "mm/h": "mm/h",
                "deg": "deg",
                "deg/km": "deg/km",
                "db/km": "dB/km",
                "dBm": "dBm",
                "dbz": "dBZ",
                "db": "dB",
                "dbm": "dBm",
                "unorm": "unorm",
                "relative_frequency": "unorm"}


def _normalise_units(u):
    """Normalise units string into a known canonical form."""
    if not u:
        return ""
    # Canonical formatting
    u = u.strip().lower()
    u = u.replace("−", "-").replace("–", "-")  # unicode dashes
    u = u.replace("per ", "/").replace("per", "/")
    u = u.replace(" ", "")
    # Regex-based normalisation
    if re.match(r"meters?/seconds?$", u):
        u = "m/s"
    elif re.match(r"m/?s-?1$", u):
        u = "m/s"
    elif re.match(r"mm/?h-?1$", u):
        u = "mm/h"
    elif u in ("degree", "degrees"):
        u = "deg"
    elif u in ("degree/km", "degrees/km", "degkm-1", "degperkm"):
        u = "deg/km"
    elif re.match(r"degrees?/kilometers?$", u):
        u = "deg/km"
    elif re.match(r"db/?kilometers?$", u):
        u = "dB/km"
    # dBZ, dB, dBm case-insensitive handling
    if u == "dbz":
        return "dBZ"
    if u == "db":
        return "dB"
    if u == "dbm":
        return "dBm"
    # Final canonical mapping
    u = UNIT_ALIASES.get(u, u)
    return u


def _safe_units(ds):
    """Return a normalised units string, mapping to safe defaults."""
    u = ds.attrs.get("units", "")
    if not u:
        return ""
    if u == "unitless":
        return "-"
    return u
