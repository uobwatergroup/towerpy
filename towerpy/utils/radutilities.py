"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import copy
import datetime as dt
import fnmatch
import json
from pathlib import Path
import re
import cartopy.io.shapereader as shpreader
from scipy import interpolate
import numpy as np
import xarray as xr
from ..utils.unit_conversion import np64_to_dtm


def find_nearest(iarray, val2search, mode="any"):
    """
    Return the index of the closest value to a given number.

    Parameters
    ----------
    iarray : array
             Input array.
    val2search : float or int
                 Value to search into the array.
    mode : {"any", "major", "minor"}, optional
        - "any": closest value in the array (default)
        - "major": closest local maximum
        - "minor": closest local minimum

    Returns
    -------
    idx : float or int
        Index into the array, or None if no candidate found.

    """
    a = np.asarray(iarray)

    if mode == "any":
        return int(np.abs(a - val2search).argmin())

    # interior extrema candidates
    if mode == "major":
        mask = (a[1:-1] > a[:-2]) & (a[1:-1] > a[2:])
    elif mode == "minor":
        mask = (a[1:-1] < a[:-2]) & (a[1:-1] < a[2:])
    else:
        raise ValueError("mode must be 'any', 'major', or 'minor'")

    candidates = np.where(mask)[0] + 1  # shift to original indices

    # if none found, fall back to nearest anywhere
    if candidates.size == 0:
        return int(np.abs(a - val2search).argmin())

    return int(candidates[np.abs(a[candidates] - val2search).argmin()])


def normalisenan(a):
    """
    Scale input vectors to unit norm, ignoring any NaNs.

    Parameters
    ----------
    a : array
        The data to normalise, element by element.

    Returns
    -------
    normarray : array
        Normalised input a.

    """
    normarray = (a-np.nanmin(a))/(np.nanmax(a)-np.nanmin(a))
    return normarray


def normalisenanvalues(a, vmin, vmax):
    """
    Scale input vectors to unit norm, by scaling the vector to given values.

    Parameters
    ----------
    a : array
        The data to normalize.
    vmin : float or int
        Minimum value used to scale the data.
    vmax : float or int
        Maximum value used to scale the data.

    Returns
    -------
    normarray : array
        Normalised data.

    """
    normarray = (a-vmin)/(vmax-vmin)
    return normarray


def fillnan1d(x):
    """
    Fill nan value with last non nan value.

    Parameters
    ----------
    x : array
        The data to be filled.

    Returns
    -------
    xf
        Array with nan values filtered.

    """
    x = np.array(x, dtype=float)
    mask = np.isnan(x)
    idx = np.where(~mask, np.arange(mask.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def interp_nan(x, y, kind='linear', nan_type='mask'):
    """
    Interpolate 1-D arrays to fill nan-masked values.

    Parameters
    ----------
    x : array_like
        1-D array.
    y : array_like
        1-D array.
    kind : str, optional
        Specifies the kind of interpolation used in the
        scipy.interpolate.interp1d function. The default is 'linear'.
    nan_type : str, optional
        Type of non-valid values, either 'nan' or 'mask'.
        The default is 'mask'.
    """
    if nan_type == 'mask':
        idx_valid = np.ma.where(np.isfinite(y))
    elif nan_type == 'nan':
        idx_valid = np.where(np.isfinite(y))
    f = interpolate.interp1d(x[idx_valid], y[idx_valid], bounds_error=False,
                             kind=kind)
    if nan_type == 'mask':
        nanitp = np.where(~np.isfinite(y).mask, y, f(x))
    elif nan_type == 'nan':
        nanitp = np.where(np.isfinite(y), y, f(x))
    return nanitp


def maf_radial(rad_vars, maf_len=3, maf_ignorenan=True, maf_extendvalid=False,
               maf_params=None):
    r"""
    Apply a Moving-Average Filter to variables along the radial direction.

    Parameters
    ----------
    rad_vars : dict
        Radar variables to be smoothed.
    maf_len : int, optional
        Odd number used to apply a moving average filter to each beam and
        smooth the signal. The default is 3.
    maf_ignorenan : bool, optional
        Set to False if nan values shall not be filtered. The default is True.
    maf_params : dict, optional
        Filters the radar variable using min and max constraints.
        The default are:

        :math:`ZH` [dBZ]: [-np.inf, np.inf]

        :math:`Z_{DR}` [dB]: [-np.inf, np.inf]

        :math:`\Phi_{DP}` [deg]: [-np.inf, np.inf]

        :math:`\rho_{HV}` [-]: [-np.inf, np.inf]

        :math:`V` [m/s]: [-np.inf, np.inf]

        :math:`LDR` [dB]: [-np.inf, np.inf]35, 35, 3]

    Returns
    -------
    mafvars : dict
        Transformed data.

    """
    lpv = {'ZH [dBZ]': [-np.inf, np.inf], 'ZDR [dB]': [-np.inf, np.inf],
           'PhiDP [deg]': [-np.inf, np.inf], 'rhoHV [-]': [-np.inf, np.inf],
           'V [m/s]': [-np.inf, np.inf], 'LDR [dB]': [-np.inf, np.inf],
           'Rainfall [mm/h]': [-np.inf, np.inf],
           'KDP [deg/km]': [-np.inf, np.inf]}
    for rkey in rad_vars.keys():
        if rkey not in lpv:
            lpv[rkey] = [-np.inf, np.inf]
    if maf_params is not None:
        lpv.update(maf_params)
    for k, v in lpv.items():
        v.append(maf_len)

    if maf_ignorenan:
        m = np.full(rad_vars[list(rad_vars.keys())[0]].shape, 0.)
        for k, v in rad_vars.items():
            m[v < lpv[k][0]] = np.nan
            m[v > lpv[k][1]] = np.nan
        vars_mask = {keys: np.ma.masked_invalid(np.ma.masked_array(values, m))
                     for keys, values in rad_vars.items()}
    else:
        m = np.full(rad_vars[list(rad_vars.keys())[0]].shape, 1.)
        for k, v in rad_vars.items():
            m[v < lpv[k][0]] = np.nan
            m[v > lpv[k][1]] = np.nan
        vars_mask = {keys: values * m
                     for keys, values in rad_vars.items()}

    mafvars = {}
    if maf_extendvalid and not maf_ignorenan:
        for k, v in vars_mask.items():
            rvarmaf = []
            for beam in v:
                # extend last valid value with ignore nan False
                mask = np.isnan(beam)
                wl = np.ones(maf_len, dtype=int)
                amaf = np.convolve(np.where(mask, 0, beam), wl,
                                   mode='same')/np.convolve(~mask, wl,
                                                            mode='same')
                rvarmaf.append(amaf)
            mafvars[k] = np.array(rvarmaf)
    elif not maf_extendvalid:
        for k, v in vars_mask.items():
            rvarmaf = []
            for beam in v:
                amaf = np.ma.convolve(beam,
                                      np.ones(lpv[k][2])/lpv[k][2],
                                      mode='same')
                rvarmaf.append(amaf)
            mafvars[k] = np.array(rvarmaf)
    return mafvars


def get_datashp(fname, key2read=None):
    """
    Read in data from *.shp files using cartopy.

    Parameters
    ----------
    fname : str
        Name of the *.shp file.
    key2read : str, optional
        Name of the feature to retrieve from the *.shp file.
        The default is None.

    Returns
    -------
    shpdatalist : list
        Features extrated from the file.

    """
    shpdata = shpreader.Reader(fname)
    shpdata1 = shpdata.records()
    shpattr = next(shpdata1)
    print('The available key-attributes of the shapefile are: \n' +
          f' {sorted(shpattr.attributes.keys())}')
    if key2read is None:
        key_att = input('Enter key attribute:')
    else:
        key_att = key2read
    print(f'Reading shapefile using -{key_att}- as key-attribute')
    # getshpdata = lambda shpdata1: shpdata1.attributes[key_att]
    gshpdat = sorted(shpdata.records(),
                     key=lambda shpdata1: shpdata1.attributes[key_att])
    shpdatalist = [i.attributes for i in gshpdat]
    return shpdatalist


def get_windows_data(wdw_size, wdw_coords, array2extract):
    """
    Retrieve data from a PPI scan using size-defined windows.

    Parameters
    ----------
    wdw_size : 2-element tuple or list of int
        Size of the window [row, cols]. Must be odd numbers.
    wdw_coords :  2-element tuple or list of int/floats
        Coordinates within the PPI scan of the centre of the window to extract.
    array2extract : array
        Data array from which the data will bve retrieved.

    Returns
    -------
    wdw : list
        Retrieved data.

    """
    if all([i % 2 for i in wdw_size]):
        start_row_index = wdw_coords[0] - wdw_size[0]//2
        end_row_index = wdw_coords[0] + (wdw_size[0]//2) + 1
        start_column_index = wdw_coords[1] - wdw_size[1]//2
        end_column_index = wdw_coords[1] + (wdw_size[1]//2) + 1
        wdw = array2extract[start_row_index:end_row_index,
                            start_column_index:end_column_index]
    else:
        wdw = print('The window rows/columns must be and odd numer')
    return wdw


def rolling_window(a, window, mode='constant', constant_values=np.nan):
    """
    Compute a rolling window using np.lib.stride_tricks for fast computation.

    Parameters
    ----------
    a : array
        Array to be smoothed.
    window : 2-element tuple or list, optional
        Window size (m, n) used to apply a moving average filter. m and n must
        be odd numbers for the m-rays and n-gates. The default is (1, 3).
    mode : str, optional
        Mode used to pad the array. See numpy.pad for more information.
        The default is 'constant'.
    constant_values : int or float, optional
        Used in 'constant'. The values to set the padded values for each axis.
        The default is np.nan.

    Notes
    -----
        It is expected to pad arrays that represent radar data in polar format.
        Thus, the rays are wrapped, and the gates are extended for consistency.
    """
    if mode == 'edge':
        apad = np.pad(a, ((0, 0), (window[1]//2, window[1]//2)),
                      mode='edge')
    elif mode == 'constant':
        apad = np.pad(a, ((0, 0), (window[1]//2, window[1]//2)),
                      mode='constant', constant_values=(constant_values))
    if window[0] > 1:
        apad = np.pad(apad, ((window[0]//2, window[0]//2), (0, 0)),
                      mode='wrap')
    return np.lib.stride_tricks.sliding_window_view(apad, window)[:, :, 0]


def compute_texture(tpy_coordlist, rad_vars, wdw_size=[3, 3], classid=None):
    """
    Compute the texture of given arrays.

    Parameters
    ----------
    tpy_coordlist : 3-element tuple or list of int
        Coordinates and classID of a given pixel.
    rad_vars : dict
        Radar variables used to compute the texture.
    wdw_size :  2-element tuple or list of int
        Size of the window [row, cols]. Must be odd numbers. The default is
        [3, 3].
    classid : dict
        Key/values of the echoes classification:
            'precipi' = 0

            'clutter' = 5

    Returns
    -------
    rvars : dict
        Texture values.
    """
    echoesID = {'precipi': 0, 'clutter': 5}
    if classid is not None:
        echoesID.update(classid)

    vval = {keid: {nvar: [vvar[nval[0], nval[1]] for nidx, nval
                          in enumerate(tpy_coordlist) if nval[2] == veid]
                   for nvar, vvar in rad_vars.items()}
            for keid, veid in echoesID.items()}

    vtxt = {keid: {'s'+nvar: [np.nanstd(get_windows_data(wdw_size, nval[:-1],
                                                         vvar), ddof=1)
                              for nidx, nval
                              in enumerate(tpy_coordlist) if nval[2] == veid]
                   for nvar, vvar in rad_vars.items()}
            for keid, veid in echoesID.items()}

    rvars = {k: vtxt[k] | vval[k] for k in echoesID.keys()}
    return rvars


def idx_consecutive(array1d, step_size=1, group_size=1):
    """
    Find the index of consecutive non-nan values within a 1D array.

    Parameters
    ----------
    array1d : array_like
        Input 1-D array.
    stepsize : int or float, optional
        Difference between consecutive elements. The default is 1.
    group_size : int, optional
        Minimum size of the grouped consecutive-valid values.
        The default is 3.
    """
    # IDX of non-nan values
    idx_valid = np.nonzero(~np.isnan(array1d))[0]
    # Group consecutive non-nan values
    cons_group = np.split(
        idx_valid, np.where(np.diff(idx_valid) != step_size)[0]+1)
    cons_idx = np.hstack([i for i in cons_group if len(i) >= group_size])
    return cons_idx


# def linspace_step(start, stop, step):
#     """Like np.linspace but uses step instead of num."""
#     return np.linspace(start, stop, int((stop - start) / step + 1))

def linspace_step(start: float, stop: float, step: float) -> np.ndarray:
    """
    Generate a 1D array from start to stop (inclusive) with a fixed step size.

    Notes
    -----
    THe function:
    * validates inputs
    * avoids floating*point drift
    * guarantees inclusion of stop
    """
    if step <= 0:
        raise ValueError("step must be positive.")
    if stop < start:
        raise ValueError("stop must be >= start.")
    # Compute number of steps using rounding to avoid float drift
    n = int(round((stop - start) / step)) + 1
    # Generate values
    arr = start + step * np.arange(n)
    # Ensure final value is exactly stop
    arr[-1] = stop
    return arr

# =============================================================================
# %% xarray implementation
# =============================================================================

# =============================================================================
# %%% UKMO datasets
# =============================================================================

def ukmo_year_directory(root_directory, year, rsite, modep, elev):
    """
    Build the canonical CEDA UKMO Nimrod single-site directory path.

    Example:
        <root>/ukmo-nimrod/data/single-site/storage_by_year/2023/chenies/raw-dual-polar/lpel0/
    """
    return (Path(root_directory) / 'ukmo-nimrod' / 'data' / 'single-site'
            / "storage_by_year" / f"{year:04d}" / rsite / "raw-dual-polar"
            / f"{modep}el{elev}")


def find_UKMO_rfile(root_directory, rsite, moder, modep, elev, target_time,
                    tolerance=dt.timedelta(minutes=5), return_time_diff=False):
    """
    Find the closest radar file to `target_time`.

    Parameters
    ----------
    root_directory : Path-like
        Directory containing radar files. The storage layout follows the CEDA
        UKMO Nimrod single-site archive structure.
    rsite : str
        Radar site identifier (e.g., "jersey", "chenies"). 
    moder : str
        Dual-polarisation mode for the "aug" field ("zdr" or "ldr").
    modep : str
        Pulse mode ("sp" or "lp").
    elev : int
        Elevation angle (e.g., 4).
    target_time : datetime.datetime
        Desired timestamp to match.
    tolerance : datetime.timedelta, optional
        Maximum allowed absolute time difference. Default is ±5 minutes.
    return_time_diff : bool, optional
        If True, return both the file path and the signed time difference.
        If False (default), return only the file path.

    Returns
    -------
    Path or (Path, datetime.timedelta) or None
        - If return_time_difference=False:
              Path to the closest file, or None if no file is within tolerance.
        - If return_time_difference=True:
              (Path, signed_time_difference), or None if no file is within tolerance.

        The signed time difference is:
            file_time - target_time
        Negative values indicate the file is earlier than the target time.

    Notes
    -----
    Files are expected to follow the naming pattern:
        metoffice-c-band-rain-radar_{rsite}_YYYYMMDDHHMM_raw-dual-polar-aug{moder}-{modep}-el{elev}.dat

    The timestamp (YYYYMMDDHHMM) is extracted from the filename and compared to
    `target_time`. The file with the smallest absolute time difference is selected.
    If the closest file lies outside the specified `tolerance`, the function
    returns None.
    """

    year_dir = ukmo_year_directory(root_directory, target_time.year, rsite,
                                   modep, elev)

    if not year_dir.is_dir():
        return None

    pattern = re.compile(rf"metoffice-c-band-rain-radar_{rsite}_([0-9]{{12}})"
                         rf"_raw-dual-polar-aug{moder}-{modep}-el{elev}\.dat")
    candidates = []
    for f in year_dir.glob("*.dat"):
        m = pattern.match(f.name)
        if not m:
            continue
        timestamp_str = m.group(1)
        file_time = dt.datetime.strptime(timestamp_str, "%Y%m%d%H%M")
        diff = file_time - target_time
        abs_diff = abs(diff)
        candidates.append((abs_diff, diff, f))
    if not candidates:
        return None
    # Pick the smallest absolute difference
    candidates.sort(key=lambda x: x[0])
    abs_diff, signed_diff, best_file = candidates[0]
    if abs_diff > tolerance:
        return None
    if return_time_diff:
        return best_file, signed_diff
    return best_file


def list_UKMO_rfiles(root_directory, rsite, moder, modep, elev, start_time,
                     stop_time, return_time_diff=False):
    """
    List radar files whose timestamps fall between `start_time` and `stop_time`

    Parameters
    ----------
    root_directory : Path-like
        Directory containing radar files. The storage layout follows the CEDA
        UKMO Nimrod single-site archive structure.
    rsite : str
        Radar site identifier (e.g., "jersey", "chenies"). 
    moder : str
        Dual-polarisation mode for the "aug" field ("zdr" or "ldr").
    modep : str
        Pulse mode ("sp" or "lp").
    elev : int
        Elevation angle (e.g., 4).
    start_time : datetime.datetime
        Start of the inclusive time window.
    stop_time : datetime.datetime
        End of the inclusive time window.
    return_time_diff : bool, optional
        If True, return (Path, file_time - start_time) tuples.
        If False (default), return only Paths.

    Returns
    -------
    list of Path or list of (Path, datetime.timedelta)
        All matching files sorted by timestamp. If no files fall within the
        specified time window, an empty list is returned.

    Notes
    -----
    Files are expected to follow the naming pattern:
        metoffice-c-band-rain-radar_{rsite}_YYYYMMDDHHMM_raw-dual-polar-aug{moder}-{modep}-el{elev}.dat
    """

    if start_time > stop_time:
        raise ValueError("start_time must be <= stop_time")

    pattern = re.compile(rf"metoffice-c-band-rain-radar_{rsite}_([0-9]{{12}})"
                         rf"_raw-dual-polar-aug{moder}-{modep}-el{elev}\.dat")
    results = []
    # Loop over all years in the time window
    for year in range(start_time.year, stop_time.year + 1):
        year_dir = ukmo_year_directory(root_directory, year, rsite, modep, elev)
        if not year_dir.is_dir():
            continue
        for f in year_dir.glob("*.dat"):
            m = pattern.match(f.name)
            if not m:
                continue
            timestamp_str = m.group(1)
            file_time = dt.datetime.strptime(timestamp_str, "%Y%m%d%H%M")
            if start_time <= file_time <= stop_time:
                diff = file_time - start_time
                results.append((file_time, diff, f))
    # Sort chronologically
    results.sort(key=lambda x: x[0])
    if return_time_diff:
        return [(f, diff) for _, diff, f in results]
    else:
        return [f for _, _, f in results]

# =============================================================================
# %%% Processing
# =============================================================================
def normalise_with_bounds(da, vmin, vmax):
    r"""
    Clip a field to the interval ``[vmin, vmax]`` and linearly normalise it
    to the range ``[0, 1]``.

    Parameters
    ----------
    da : xarray.DataArray or array-like
        Input data to be normalised.
    vmin : float
        Lower bound for clipping.
    vmax : float
        Upper bound for clipping. Must satisfy ``vmax > vmin``.

    Returns
    -------
    xarray.DataArray
        Normalised data in the range ``[0, 1]`` after clipping to
        ``[vmin, vmax]``.

    Notes
    -----
    * The transformation applied is:
        .. math::
            x_\mathrm{norm} =
            \frac{\min(\max(x, v_\mathrm{min}), v_\mathrm{max}) - v_\mathrm{min}}
                 {v_\mathrm{max} - v_\mathrm{min}}

    """

    if not isinstance(da, xr.DataArray):
        da = xr.DataArray(da)
    da_clipped = da.clip(vmin, vmax)
    return (da_clipped - vmin) / (vmax - vmin)


def normalise_auto_bounds(da):
    r"""
    Linearly normalise a field to the range ``[0, 1]`` using its own
    minimum and maximum values.

    Parameters
    ----------
    da : xarray.DataArray or array-like
        Input data to be normalised.

    Returns
    -------
    xarray.DataArray
        Normalised data in the range ``[0, 1]``. NaNs in the input are
        preserved. If all finite values are equal, the result will contain
        NaNs due to division by zero.

    Notes
    -----
    * The transformation applied is:
        .. math::
            x_\mathrm{norm} =
            \frac{x - x_\mathrm{min}}{x_\mathrm{max} - x_\mathrm{min}}
        where ``x_min`` and ``x_max`` are computed ignoring NaNs.
    * NaNs are ignored when computing the bounds.

    """
    if not isinstance(da, xr.DataArray):
        da = xr.DataArray(da)
    mn = da.min(skipna=True)
    mx = da.max(skipna=True)
    return (da - mn) / (mx - mn)


def find_nearest_index(arr, value, mode="any"):
    """
    Find the index of the element in ``arr`` that is closest to ``value``,
    optionally restricted to local maxima or minima.

    Parameters
    ----------
    arr : array-like
        Input 1D array of numeric values.
    value : float or int
        Target value used to determine proximity.
    mode : {"any", "major", "minor"}, default "any"
        Selection mode:

        - ``"any"``   – return the index of the element closest to ``value``.
        - ``"major"`` – restrict the search to local maxima.
        - ``"minor"`` – restrict the search to local minima.

        If no local maxima/minima exist for ``"major"`` or ``"minor"``,
        the function falls back to ``"any"``.

    Returns
    -------
    int
        Index of the selected element in ``arr``.

    Notes
    -----
    * NaNs are ignored when computing distances via ``np.nanargmin``.
    * Local maxima/minima are detected by comparing each element with its
      immediate neighbours.
    """
    if mode == "any":
        return int(np.nanargmin(np.abs(arr - value)))

    left  = np.r_[np.nan, arr[:-1]]
    right = np.r_[arr[1:], np.nan]

    if mode == "major":
        mask = (arr > left) & (arr > right)
    elif mode == "minor":
        mask = (arr < left) & (arr < right)
    else:
        raise ValueError("mode must be 'any', 'major', or 'minor'")

    if not np.any(mask):
        return int(np.nanargmin(np.abs(arr - value)))

    cand = np.abs(arr - value)
    cand[~mask] = np.nan
    return int(np.nanargmin(cand))


def robust_minmax(arr, method="quantile", floor=None, **kwargs):
    """
    Compute robust min/max values for a NumPy array, resistant to outliers.

    Parameters
    ----------
    arr : array-like
        Input data (NaNs are ignored).
    method : str, optional
        Robustness strategy:
        - "quantile": use lower/upper quantiles
        - "mad": median ± k * MAD
        - "iqr": Tukey fences (Q1 - k*IQR, Q3 + k*IQR)
    floor : float or None, optional
        If given, the lower bound will be clamped to at least this value.
        Useful for non-negative domains (e.g. heights).
    kwargs : dict
        Extra parameters depending on method:
        - quantile: q_low=0.01, q_high=0.99
        - mad: k=3.0
        - iqr: k=1.5

    Returns
    -------
    (robust_min, robust_max)
    """
    arr = np.asarray(arr).ravel()
    arr = arr[~np.isnan(arr)]  # drop NaNs

    if arr.size == 0:
        return np.nan, np.nan

    if method == "quantile":
        q_low = kwargs.get("q_low", 0.01)
        q_high = kwargs.get("q_high", 0.99)
        lower, upper = np.quantile(arr, [q_low, q_high])

    elif method == "mad":
        k = kwargs.get("k", 3.0)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        lower, upper = med - k * mad, med + k * mad

    elif method == "iqr":
        k = kwargs.get("k", 1.5)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
    
    elif method == "simple":
        lower, upper = np.nanmin(arr), np.nanmax(arr)
    
    else:
        raise ValueError(f"Unknown method: {method}")

    # Enforce floor if requested
    if floor is not None:
        lower = max(lower, floor)

    return lower, upper


def xr_hist2d(x, y, x_edges, y_edges, dim):
    """
    Compute a vectorised 2D histogram of two DataArrays.

    Parameters
    ----------
    x, y : xr.DataArray
        Input variables to histogram over, sharing the same core dimensions.
    x_edges, y_edges : array-like
        Bin edges for the x and y axes, defining the histogram grid.
    dim : list[str]
        Dimensions over which the histogram is computed and reduced.

    Returns
    -------
    xr.DataArray
        A 2D histogram with labelled ``x_bin`` and ``y_bin`` dimensions,
        vectorised over all non-core dimensions.

    Notes
    -----
    1. x corresponds to SNR, y corresponds to rhoHV in the noise-correction
    workflow.

    """
    def _hist2d(a, b):
        # Collapse core dims; vectorisation happens over non-core dims
        H, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=[x_edges, y_edges])
        return H

    return xr.apply_ufunc(
        _hist2d, x, y, input_core_dims=[dim, dim],
        output_core_dims=[["x_bin", "y_bin"]],
        vectorize=True, dask="parallelized",
        output_dtypes=[float]).assign_coords(x_bin=x_edges[:-1],
                                             y_bin=y_edges[:-1])


def pad_range(da, pad, range_dim="range"):
    """Pad the range dimension on both sides with NaNs."""
    return da.pad({range_dim: (pad, pad)}, mode="constant",
                  constant_values=np.nan)


def despike_isolated(da, window, range_dim="range"):
    """Flag isolated non‑null gates whose neighbours are both null."""
    pad = window // 2
    da_pad = pad_range(da, pad, range_dim)

    left = da_pad.shift({range_dim: 1})
    right = da_pad.shift({range_dim: -1})

    keep = ~(da_pad.notnull() & left.isnull() & right.isnull())

    core = keep.isel({range_dim: slice(pad, pad + da.sizes[range_dim])})

    return core


def std_mask_threshold(std, thr, window, range_dim="range"):
    """Flag gates where the rolling stdexceeds a threshold."""
    pad = window // 2
    std_pad = pad_range(std, pad, range_dim)

    left = std_pad.shift({range_dim: 1})
    right = std_pad.shift({range_dim: -1})

    drop = (std_pad >= thr) & ((left >= thr) | (right >= thr))
    keep = ~drop
    core = keep.isel({range_dim: slice(pad, pad + std.sizes[range_dim])})

    return core


def std_mask_isolated(std, window, range_dim="range"):
    """Flag isolated non‑null std gates lacking valid neighbours."""

    pad = window // 2
    std_pad = pad_range(std, pad, range_dim)

    left = std_pad.shift({range_dim: 1})
    right = std_pad.shift({range_dim: -1})

    drop = std_pad.notnull() & (left.isnull() | right.isnull())
    keep = ~drop
    core = keep.isel({range_dim: slice(pad, pad + std.sizes[range_dim])})

    return core


def rolling_std_xr(da, mov_avrgf_len=(1, 3), *, azimuth_dim="azimuth",
                   range_dim="range"):
    """Compute a 1D or 2D rolling standard deviation over range and azimuth."""
    # assign wsize
    m, n = mov_avrgf_len
    # 1D rolling
    if m == 1:
        return (da.rolling({range_dim: n}, center=True)
                .construct("window")
                .reduce(np.nanstd, dim="window"))
    # 2D rolling
    return (da.rolling({azimuth_dim: m, range_dim: n}, center=True)
            .construct({azimuth_dim: "window_az", range_dim: "window_rg"})
            .reduce(np.nanstd, dim=("window_az", "window_rg")))


def anchor_edges(x):
    """Return a copy where leading/trailing NaNs are replaced with the nearest valid values.
    Internal use only (for stable rolling stats)."""
    x = np.asarray(x, float).copy()
    mask = np.isnan(x)
    if mask.all():
        return x
    # Fill leading
    first_valid = np.argmax(~mask)
    x[:first_valid] = x[first_valid]
    # Fill trailing
    last_valid = len(x) - 1 - np.argmax(~mask[::-1])
    x[last_valid+1:] = x[last_valid]
    return x


def sliding_windows(x, window):
    """Centered sliding windows with edge anchoring, returns (n, window)."""
    x_anchored = anchor_edges(x)
    half = window // 2
    # Pad with the anchored edge values so windows near edges remain valid
    left_pad = x_anchored[0]
    right_pad = x_anchored[-1]
    padded = np.pad(x_anchored, (half, window - half - 1), mode="constant",
                    constant_values=(left_pad, right_pad))
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(padded, window)


def edge_preserving_rolling_mean(x, window):
    x = np.asarray(x, float)
    n = len(x)
    out = np.empty(n, float)

    # Use anchored edges for internal computation
    w = sliding_windows(x, window)
    center = np.mean(w, axis=1)

    # Warm-up: cumulative mean starting at the first valid
    mask = np.isnan(x)
    if mask.all():
        return np.full(n, np.nan)
    first_valid = np.argmax(~mask)
    last_valid = n - 1 - np.argmax(~mask[::-1])

    # Warm-up (from first_valid inclusive)
    cum = np.cumsum(anchor_edges(x))
    count = np.cumsum(~np.isnan(anchor_edges(x)))
    for i in range(0, first_valid):
        out[i] = np.nan  # preserve leading NaNs in the baseline output if desired
    for i in range(first_valid, min(first_valid + window // 2, n)):
        # cumulative mean of anchored series avoids dilution
        out[i] = (cum[i] - (cum[first_valid-1] if first_valid > 0 else 0)) / \
                 (count[i] - (count[first_valid-1] if first_valid > 0 else 0))

    # Middle (centered rolling mean)
    mid_start = window // 2
    mid_end = n - (window // 2)
    out[mid_start:mid_end] = center[mid_start:mid_end]

    # Cool-down: cumulative mean backwards to last_valid
    for i in range(max(last_valid - window // 2 + 1, mid_end), last_valid + 1):
        segment = anchor_edges(x)[i:last_valid+1]
        out[i] = np.mean(segment)

    # Preserve trailing NaNs beyond last_valid
    for i in range(last_valid + 1, n):
        out[i] = np.nan

    return out


def fill_both(da, dim="range"):
    """Forward + backward fill along a dimension."""
    # Forward fill
    fwd = da.interpolate_na(dim=dim, method="nearest", fill_value="extrapolate")
    # Reverse *data* only, keep coords intact
    rev_data = fwd.data[..., ::-1]   # works for Dask arrays too?
    rev = xr.DataArray(rev_data, dims=fwd.dims,
                       coords={k: (v if k != dim else fwd.coords[k])
                               for k, v in fwd.coords.items()},
                       attrs=fwd.attrs,
                       name=fwd.name)
    # Forward fill again on reversed data
    rev_filled = rev.interpolate_na(dim=dim, method="nearest",
                                    fill_value="extrapolate")
    # Reverse data back
    out_data = rev_filled.data[..., ::-1]
    out = xr.DataArray(out_data, dims=rev_filled.dims, coords=rev_filled.coords,
                       attrs=rev_filled.attrs, name=rev_filled.name)
    return out


def detect_and_clean(raw, window=12, k=3, replace="baseline"):
    """
    Detect outliers and return baseline, mask, and cleaned series.

    Parameters
    ----------
    raw : array-like
        Input array (NaNs allowed).
    window : int
        Rolling window length.
    k : float
        MAD multiplier.
    replace : {"baseline", "nan", "interp"}
        Strategy for handling outliers:
        - "baseline": replace with rolling baseline
        - "nan": replace with NaN
        - "interp": replace with linear interpolation

    Returns
    -------
    baseline : ndarray
        Edge-preserving rolling mean baseline.
    outliers : ndarray of bool
        Boolean mask of detected outliers.
    cleaned : ndarray
        Cleaned series according to `replace`.
    """
    # Fill NaNs for baseline calculation
    x = fillnan1d(raw)
    baseline = edge_preserving_rolling_mean(x, window)
    residuals = np.abs(x - baseline)
    mad_vals = rolling_mad(x, window)
    eps = 1e-6
    outliers = residuals > k * (mad_vals + eps)

    cleaned = x.copy()
    if replace == "baseline":
        cleaned[outliers] = baseline[outliers]
    elif replace == "nan":
        cleaned[outliers] = np.nan
    elif replace == "interp":
        cleaned[outliers] = np.nan
        # Simple linear interpolation over NaNs
        nans = np.isnan(cleaned)
        if np.any(nans):
            idx = np.arange(len(cleaned))
            cleaned[nans] = np.interp(idx[nans], idx[~nans], cleaned[~nans])
    else:
        raise ValueError(f"Unknown replace strategy: {replace}")

    # return baseline, outliers, cleaned
    return cleaned


def rolling_mad(x, window):
    x = np.asarray(x, float)
    w = sliding_windows(x, window)  # windows built from anchored edges

    # Compute median and MAD ignoring NaNs in the original x per window
    # Use a mask derived from original x to avoid NaN contamination
    # Build a mask window for each center position
    n = len(x)
    half = window // 2
    # Pad the NaN mask similarly to align with windows
    mask = np.isnan(x)
    left = np.repeat(mask[0], half)
    right = np.repeat(mask[-1], window - half - 1)
    mask_padded = np.concatenate([left, mask, right])
    from numpy.lib.stride_tricks import sliding_window_view
    mask_w = sliding_window_view(mask_padded, window)  # (n, window)

    # For each window, compute median over values where original x is not NaN
    med = np.empty(n, float)
    mad = np.empty(n, float)
    for i in range(n):
        valid = ~mask_w[i]
        vals = w[i][valid]
        # If none valid, set NaN
        if vals.size == 0:
            med[i] = np.nan
            mad[i] = np.nan
        else:
            m = np.median(vals)
            med[i] = m
            mad[i] = np.median(np.abs(vals - m))
    return mad

# =============================================================================
# %%% Dataset attrs
# =============================================================================
def scan_midtime(arr):
    """
    Return the midpoint of a datetime64 array both as numpy.datetime64
    and as a Python datetime.

    Parameters
    ----------
    arr : array-like of numpy.datetime64
        Typically something like merged.time.values.

    Returns
    -------
    mid_np : numpy.datetime64
        Midpoint timestamp in numpy datetime64.
    mid_py : datetime.datetime
        Midpoint timestamp converted to Python datetime.
    """
    arr = np.asarray(arr)

    # Extract min/max
    bounds = np.array([arr.min(), arr.max()])

    # Compute midpoint in integer space
    mid_int = bounds.astype(int).mean().astype(bounds.dtype)

    # Convert to Python datetime
    mid_py = np64_to_dtm(mid_int)

    return mid_int, mid_py


def getcoordunits(ds, coord_name, default=""):
    return ds.coords[coord_name].attrs.get("units", default)


def resolve_rect_coords(ds, coord_names=None):
    """
    Resolve rectangular coordinate names using:
    1. explicit user override
    2. automatic detection
    """
    # 1. User override
    if coord_names:
        xname = coord_names.get("x")
        yname = coord_names.get("y")
        if xname in ds.coords and yname in ds.coords:
            return xname, yname

    # 2. Automatic detection
    candidates = [("grid_rectx", "grid_recty"),
                  ("x", "y"),
                  ("xc", "yc"),
                  ("projection_x_coordinate", "projection_y_coordinate")]
    for xname, yname in candidates:
        if xname in ds.coords and yname in ds.coords:
            return xname, yname
    return None, None


def auto_assign_elevation(sweep, azimuth_dim="azimuth"):
    """
    Normalise elevation coordinate in a sweep:
    - If elevation exists and is constant, collapse to scalar.
    - If elevation varies, keep it as azimuth-dependent.
    - Preserve existing attrs.
    """
    if "elevation" not in sweep.coords:
        return sweep  # nothing to do

    elev = sweep.coords["elevation"]
    vals = elev.values
    attrs = elev.attrs.copy()

    if np.allclose(vals, vals[0]):
        # Collapse to scalar
        sweep = sweep.drop_vars("elevation")
        sweep = sweep.assign_coords(elevation=float(vals[0]))
        sweep.coords["elevation"].attrs.update(attrs)
    else:
        # Already varying, just ensure attrs are preserved
        sweep.coords["elevation"].attrs.update(attrs)
    return sweep


def _extract_timestamp_from_attrs(attrs):
    """Return the oldest parseable timestamp from attrs."""
    candidates = []

    for key, val in attrs.items():
        if not isinstance(val, str):
            continue
        try:
            ts = np.datetime64(val)
            candidates.append(ts)
        except Exception:
            continue

    if not candidates:
        return None

    # Choose the oldest timestamp (valid time, not creation time)
    best = min(candidates)

    py_dt = dt.datetime.utcfromtimestamp(
        best.astype("datetime64[s]").astype(int)
    )
    return py_dt


def resolve_attr(ds, key, default=None, return_all=False):
    """
    Resolve an attribute inside ds.attrs using nested lookup with support
    for wildcard segments (e.g. 'dataset*/how/radconstH').

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset whose attrs may contain nested dicts.
    key : str
        Path-like key, segments separated by '/' or '.'.
        Supports wildcards via fnmatch (e.g. 'dataset*/how/radconstH').
    default : any
        Value returned if no match is found.
    return_all : bool
        If True, return list of all matches. Otherwise return first match.

    Returns
    -------
    any or list
    """
    parts = key.split("/")

    def _search(node, idx):
        if idx == len(parts):
            return [node]

        if not isinstance(node, dict):
            return []

        matches = []
        for k, v in node.items():
            # split metadata key into its own segments
            k_parts = k.split("/")

            consumed = 0
            # match as many segments as possible
            while (
                idx + consumed < len(parts)
                and consumed < len(k_parts)
                and fnmatch.fnmatch(k_parts[consumed], parts[idx + consumed])
            ):
                consumed += 1

            if consumed == 0:
                continue

            # if we consumed all segments of k_parts, descend into v
            if consumed == len(k_parts):
                matches.extend(_search(v, idx + consumed))

        return matches

    results = _search(ds.attrs, 0)
    if return_all:
        return results if results else default
    return results[0] if results else default


# def get_attrval(attr, sweep, default=None, required=True):
#     """
#     Resolve a metadata attribute from a sweep using flexible candidate paths.

#     Parameters
#     ----------
#     sweep : xarray.Dataset
#         Sweep dataset with nested attrs.
#     attr : str
#         Logical attribute name, e.g. "beamwidth", "radconstH".
#     default : any
#         Value returned if nothing is found (unless required=True).
#     required : bool
#         If True, raise KeyError when attribute is missing.

#     Returns
#     -------
#     any
#     """
#     candidates = [f"how/{attr}",
#                   f"what/{attr}",
#                   f"where/{attr}",
#                   attr,  # top-level
#                   f"{attr}",
#                   # f"{attr}_h",
#                   # f"{attr}_v",
#                   f"dataset*/how/{attr}",
#                   f"dataset*/what/{attr}",
#                   f"dataset*/where/{attr}",
#                   f"scan*/how/{attr}",]

#     for key in candidates:
#         val = resolve_attr(sweep, key, default=None)
#         if val is not None:
#             return val
#     if required:
#         raise KeyError( f"Required attribute '{attr}' not found in metadata. "
#                        f"Candidates tried: {candidates}" )
#     return default

def get_attrval(attr, sweep, default=None, required=True):
    """
    Resolve a metadata attribute from a sweep using flexible candidate paths.
    """

    candidates = []

    # how / what / where
    for base in ["how", "what", "where"]:
        candidates.append(f"{base}/{attr}")
        candidates.append(f"{base}_{attr}")

    # top-level
    candidates.append(attr)

    # dataset* and scan*
    for base in ["dataset*", "scan*"]:
        # legacy nested
        candidates.append(f"{base}/how/{attr}")
        # sanitized nested
        candidates.append(f"{base}_how/{attr}")
        # sanitized flattened
        candidates.append(f"{base}_how_{attr}")

    # Try all candidates
    for key in candidates:
        val = resolve_attr(sweep, key, default=None)
        if val is not None:
            return val

    if required:
        raise KeyError(
            f"Required attribute '{attr}' not found. Candidates tried: {candidates}"
        )
    return default


def _as_dataset(obj):
    """Return (dataset, varname) for either a Dataset or DataArray."""
    if isinstance(obj, xr.DataArray):
        # Wrap into a dataset
        ds = obj.to_dataset(name=obj.name or "data")

        # Preserve DataArray attributes
        if obj.attrs:
            ds.attrs.update(obj.attrs)

        return ds, list(ds.data_vars)[0]
    else:
        return obj, None


def _safe_metadata(ds, elev_dim="elevation"):
    """Return dict with safe radar metadata (elevation, time, radar name, sweep mode)."""

    out = {}

    # ------------------------------------------------------------------
    # Elevation handling
    # ------------------------------------------------------------------
    if elev_dim in ds.coords:
        elev = ds[elev_dim].values
        # Convert to 1‑D array safely
        elev = np.asarray(elev).ravel()
        if elev.size == 0:
            out["elev_str"] = ""
        else:
            # Check if all elevations are (almost) identical
            if np.allclose(elev, elev[0], rtol=0, atol=1e-0):
                out["elev_str"] = f"{float(elev[0]):.1f}deg"
            else:
                # Mixed elevations -> show min–max
                out["elev_str"] = f"{float(elev.min()):.2f}–{float(elev.max()):.2f}deg"
    else:
        # Fallback: old attribute
        elev = ds.attrs.get("elev_ang")
        if isinstance(elev, (int, float)):
            out["elev_str"] = f"{elev:.1f}deg"
        else:
            out["elev_str"] = ""

    # ------------------------------------------------------------------
    # Time handling
    # ------------------------------------------------------------------
    py_dt = None
    # 1. Try coordinate
    if "time" in ds.coords:
        try:
            t = ds["time"].values
            ts = t if np.ndim(t) == 0 else t[0]
            py_dt = dt.datetime.utcfromtimestamp(
                ts.astype("datetime64[s]").astype(int))
        except Exception:
            py_dt = None
    # 2. Fallback: infer from attributes
    if py_dt is None:
        py_dt = _extract_timestamp_from_attrs(ds.attrs)
    # 3. Format or fallback
    out["dt_str"] = py_dt.strftime("%Y-%m-%d %H:%M:%S") if py_dt else ""
    # ------------------------------------------------------------------
    # Radar name
    # ------------------------------------------------------------------
    # Radar name
    where = ds.attrs.get("where")

    if isinstance(where, dict):
        out["rname"] = where.get("site_name", "Radar")
    elif "site_name" in ds.attrs:
        out["rname"] = ds.attrs["site_name"]
    else:
        out["rname"] = "Radar"
    # ------------------------------------------------------------------
    # Sweep mode
    # ------------------------------------------------------------------
    try:
        out["swp_mode"] = str(ds.sweep_mode.values)
    except Exception:
        out["swp_mode"] = ""

    return out


def _deep_update(base, updates):
    """Recursively merge nested dictionaries, overwriting non-dict values."""
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else: base[key] = value
    return base


def safe_assign_variable(ds, name, da, *, new_attrs=None):
    """
    Safely overwrite a variable in a Dataset while preserving all coordinate
    attributes and encodings.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    name : str
        Variable name to assign/overwrite.
    da : xr.DataArray
        New data array.
    new_attrs : dict, optional
        Attributes to merge into the variable after assignment.

    Returns
    -------
    xr.Dataset
        Dataset with variable replaced and coordinate attrs preserved.
    """

    # Snapshot coordinate attrs + encodings
    coord_attrs = {c: ds[c].attrs.copy() for c in ds.coords}
    coord_enc = {c: ds[c].encoding.copy() for c in ds.coords}

    # Assign variable
    ds[name] = da

    # Restore coordinate attrs + encodings
    for c in coord_attrs:
        ds[c].attrs = coord_attrs[c]
        ds[c].encoding = coord_enc[c]

    # Update variable attrs if requested
    if new_attrs is not None:
        merged = {**ds[name].attrs, **new_attrs}
        ds[name].attrs = merged

    return ds


def apply_offset_ppi(ds, var2correct, offset, *, output_mode="preserve",
                     output_name=None, return_only_corrected=False,
                     azimuth_dim="azimuth", provenance_name="apply_offset"):
    """
    Apply an offset to a single PPI radar variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Input PPI dataset. Must contain a valid azimuth coordinate and the
        variable to correct.
    var2correct : str
        Name of the variable to correct.
    offset : float, array-like, or xarray.DataArray
        Offset to subtract. May be:
        - scalar (same for all rays)
        - 1D array/DataArray indexed by azimuth (per-ray offset)
    output_mode : ["preserve", "overwrite", "rename"], optional
        Controls how the corrected variable is written:

        * "preserve"  : keep the original variable and write the corrected
                        variable to ``output_name``. If ``output_name`` is
                        None, defaults to ``<var2correct>_QC``.
        * "overwrite" : overwrite the original variable in place.
                        ``output_name`` is ignored.
        * "rename"    : drop the original variable and write the corrected
                        variable to ``output_name``. ``output_name`` must be
                        provided.
    output_name : str, optional
        Name of the corrected variable when ``output_mode`` is "preserve" or
        "rename". If None and ``output_mode="preserve"``, defaults to
        ``<var2correct>_QC``. Required when ``output_mode="rename"``.
    return_only_corrected : bool, optional
        If True, return only the corrected variable as a DataArray.
        If False (default), return the full dataset.
    azimuth_dim : str, optional
        Name of azimuth dimension.
    provenance_name : str, optional
        Name recorded in provenance metadata.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Full dataset or only the corrected variable, depending on
        ``return_only_corrected``.

    Notes
    -----
    * This function is PPI-specific. It assumes that ``var2correct`` is
      indexed by azimuth and range. Applying it to QVPs, RHIs, or other
      products will raise errors or produce incorrect results.
    * Offset broadcasting follows the azimuth dimension. If ``offset`` is
      1D, its dimension must match ``azimuth_dim``.
    * Variable-level and dataset-level provenance are updated to reflect
      the correction step, including the chosen ``output_mode`` and
      ``output_name``.
    """
    from ..io import modeltp as mdtp

    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f

    # Validate variable
    if var2correct not in ds.data_vars:
        raise KeyError(f"Variable {var2correct} not found in dataset.")

    # Validate output_mode
    if output_mode not in ("preserve", "overwrite", "rename"):
        raise ValueError("output_mode must be one of "
                         "'preserve', 'overwrite', 'rename'.")

    # Validate output_name for rename mode
    if output_mode == "rename" and output_name is None:
        raise ValueError("output_name must be provided when "
                         "output_mode='rename'.")

    # Normalise offset
    if not isinstance(offset, xr.DataArray):
        offset = xr.DataArray(offset)

    var_da = ds[var2correct]

    # Broadcast offset if needed
    offset_b = offset
    if azimuth_dim in var_da.dims and offset_b.ndim == 1:
        offset_b = offset_b.broadcast_like(
            var_da.sel({azimuth_dim: var_da[azimuth_dim]}))

    # Determine output_name for preserve mode
    if output_mode == "preserve" and output_name is None:
        output_name = "%s_QC" % var2correct

    # Apply correction
    corrected = (var_da - offset_b).rename(output_name)

    # Build attrs: canonical + parent
    parent_attrs = var_da.attrs.copy()
    canonical_attrs = sweep_vars_attrs_f.get(output_name, {}).copy()
    merged_attrs = dict(canonical_attrs)
    merged_attrs.update(parent_attrs)

    # Update correction_chain and provenance
    merged_attrs = add_correction_step(
        parent_attrs=merged_attrs,
        step=provenance_name,
        parent=var2correct,
        params={"offset": (offset.values.tolist()
                           if hasattr(offset, "values") else offset),
                "output_mode": output_mode,
                "output_name": output_name},
        outputs=[output_name],
        mode="overwrite" if output_mode == "overwrite" else "preserve",
        module_provenance="towerpy.utils.radutilities.apply_offset_ppi",)
    corrected.attrs = merged_attrs

    # Assign corrected variable
    ds_out = ds.copy()

    if output_mode == "overwrite":
        ds_out = safe_assign_variable(ds_out, var2correct, corrected)
        assigned_name = var2correct

    elif output_mode == "preserve":
        ds_out = safe_assign_variable(ds_out, output_name, corrected)
        assigned_name = output_name

    elif output_mode == "rename":
        ds_out = safe_assign_variable(ds_out, output_name, corrected)
        assigned_name = output_name
        if output_name != var2correct:
            ds_out = ds_out.drop_vars(var2correct)

    # Merge canonical attrs again after assignment
    old_attrs = ds_out[assigned_name].attrs.copy()
    canonical = sweep_vars_attrs_f.get(assigned_name, {})
    merged = dict(old_attrs)
    merged.update(canonical)
    ds_out = safe_assign_variable(ds_out, assigned_name, ds_out[assigned_name],
                                  new_attrs=merged)

    # Dataset-level provenance
    params = {"var2correct": var2correct,
              "offset": (offset.values.tolist()
                         if hasattr(offset, "values") else offset),
              "output_mode": output_mode,
              "output_name": output_name,
              "azimuth_dim": azimuth_dim}
    ds_out = record_provenance(
        ds_out, step=provenance_name,
        inputs=[var2correct], outputs=[assigned_name],
        parameters=params,
        module_provenance="towerpy.utils.radutilities.apply_offset_ppi",)

    # Optional: return only corrected variable
    if return_only_corrected:
        return ds_out[assigned_name]

    return ds_out

# =============================================================================
# %%% Provenance
# =============================================================================

def record_provenance(ds, step, outputs, parameters, inputs=None,
                      module_provenance=None, extra_attrs=None):
    """
    Update a Dataset’s provenance chain by recording processing steps, inputs,
    outputs, and parameters.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose provenance metadata is to be updated.
    step : str
        Logical step name contributing to the correction chain, also used
        for grouping related function calls
    outputs : list[str]
        Variables produced or modified by the function.
    parameters : dict
        Parameter values used during the function call.
    inputs : list[str], optional
        Variables read or required by the function.
    module_provenance : str, optional
        Name of the module or library that performed the step.
        Defaults to "towerpy".
    extra_attrs : dict, optional
        Additional attributes to attach directly to the Dataset.

    Returns
    -------
    xr.Dataset
        The Dataset with an updated ``processing_chain`` attribute reflecting
        the new provenance entry.
    """
    ds = ds.copy()
    # Normalise inputs/outputs
    inputs = sorted(set(inputs or []))
    outputs = sorted(set(outputs))
    # Make parameters JSON‑serialisable
    def _py(v):
        if isinstance(v, np.generic):
            return v.item()
        return v
    parameters = {k: _py(v) for k, v in parameters.items()}
    # Load existing chain
    raw_chain = ds.attrs.get("processing_chain", [])
    if isinstance(raw_chain, str):
        try:
            chain = json.loads(raw_chain)
        except Exception:
            chain = []
    else:
        chain = list(raw_chain)
    # Try to merge with existing entry
    existing = None
    for entry in chain:
        if entry.get("step") == step:
            existing = entry
            break
    if existing:
        # Merge outputs
        existing["outputs"] = sorted(set(existing.get("outputs", [])).union(outputs))
        # Merge parameters
        # raw = existing["parameters"]
        # params_dict = json.loads(raw) if isinstance(raw, str) else raw
        params_dict = existing["parameters"]
        for k, v in parameters.items():
            if k in params_dict and params_dict[k] != v:
                existing = None
                break
            params_dict[k] = v
        if existing:
            # existing["parameters"] = _maybe_json_encode(params_dict)
            existing["parameters"] = params_dict
            # Merge inputs
            existing["inputs"] = sorted(set(existing.get("inputs", [])).union(inputs))
    if existing is None:
        # Create new step entry
        entry = {
            "step": step,
            "inputs": inputs,
            "outputs": outputs,
            # "parameters": _maybe_json_encode(parameters),
            "parameters": parameters,
            "module_provenance": module_provenance or "towerpy",
        }
        chain.append(entry)
    else:
        entry = existing
    # Apply dataset-level extra attrs
    if extra_attrs:
        for k, v in extra_attrs.items():
            # entry[k] = _maybe_json_encode(_py(v))
            entry[k] = _py(v)
    # Write back processing_chain
    # ds.attrs["processing_chain"] = _maybe_json_encode(chain)
    ds.attrs["processing_chain"] = chain
    return ds


def add_correction_step(parent_attrs, *, step, parent, params, outputs,
                        mode=None, extra_attrs=None, module_provenance=None):
    """
    Append a correction step to a variable's provenance chain and update
    all associated provenance attributes in a consistent way.

    Parameters
    ----------
    parent_attrs : dict
        Existing attrs of the parent variable or dataset.
    step : str
        Name of the processing step (e.g. "attenuation_correction").
    parent : str
        Name of the parent variable.
    params : dict
        Parameters used in this step. Must be JSON-serialisable.
    outputs : list[str]
        Names of output variables produced by this step.
    mode : {"overwrite", "preserve"}, optional
        If None, defaults to "preserve".
    module_provenance : str, default "Towerpy"
        Identifier for the provenance source.

    Returns
    -------
    dict
        Updated attrs including correction_chain, provenance fields, and mode.
    """
    if module_provenance is None:
        module_provenance = "Towerpy"
    # Determine mode
    if mode is None:
        mode = "preserve"
    # Build chain entry
    chain_entry = {
        "step": step, "params": params, "parent": parent, "mode": mode,
        "outputs": outputs, "module_provenance": module_provenance}
    if extra_attrs:
        for k, v in extra_attrs.items():
            chain_entry[k] = v
    # Merge with previous chain
    prev_chain = parent_attrs.get("correction_chain", [])
    if isinstance(prev_chain, str):
        prev_chain = json.loads(prev_chain)
    new_chain = prev_chain + [chain_entry]
    # Build updated attrs
    new_attrs = dict(parent_attrs)
    new_attrs.update({
        "correction_chain": new_chain,
        # "correction_chain": _maybe_json_encode(new_chain),
        "correction": step, "parent": parent, "mode": mode,
        # "provenance": provenance_source, "provenance_step": step,
        # "provenance_base": parent,
        })
    return new_attrs


def apply_correction_chain(ds, varname, step, params, mask=None, suffix="_corr",
                           corrected_field=None, module_provenance=None,
                           extra_attrs=None):
    """
    Apply a correction mask to a variable and add it back into the Dataset,
    appending provenance-aware metadata to track the full correction chain.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable to correct.
    varname : str
        Name of the variable to correct.
    mask : xarray.DataArray
        Boolean or categorical mask aligned with ds[varname].
    step : str
        Short tag describing the correction step (e.g. "SNR_correction").
    params : dict
        Parameters used in the correction (e.g. {"min_snr": 5}).
    suffix : str, optional
        Suffix for the corrected variable name. Default "_corr".
    corrected_field : xarray.DataArray, optional
        The corrected variable produced by this step. When provided, it will
        be used as the output of the correction step.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with new corrected variable added.
    """
    if module_provenance is None:
        module_provenance = "Towerpy"
    corrected_name = f"{varname}{suffix}"
    # 1. Determine corrected data
    if corrected_field is not None:
        corrected = corrected_field
    elif mask is not None:
        corrected = ds[varname].where(mask == 0)
    else:
        corrected = ds[varname]
    # 2. Build provenance entry
    new_entry = {"step": step, "params": params, "parent": varname,
                 "outputs": [corrected_name],
                 "mode": "overwrite" if suffix == "" else "preserve",
                 "module_provenance": module_provenance
                 }
    if extra_attrs:
        for k, v in extra_attrs.items():
            new_entry[k] = v
    # chain = ds[varname].attrs.get("correction_chain", [])
    # chain.append(new_entry)
    raw_chain = ds[varname].attrs.get("correction_chain", [])
    if isinstance(raw_chain, str):
        try:
            chain = json.loads(raw_chain)
        except Exception:
            chain = []
    else:
        chain = list(raw_chain)

    chain.append(new_entry)
    # Base metadata update (variable-level only)
    meta_update = {"correction": step,
                   "correction_params": params,
                   "parent": varname,
                   "history": ds[varname].attrs.get(
                       "history", "") + f" | {new_entry}",
                   # "correction_chain": _maybe_json_encode(chain),
                   "correction_chain": chain,
                   # "provenance": "Towerpy",
                   # "provenance_step": step,
                   # "provenance_base": varname,
                   "mode": new_entry["mode"]}

    # Apply variable-level attrs and insert into dataset
    corrected.attrs = {**ds[varname].attrs, **meta_update}
    # encoded = {k: _maybe_json_encode(v) for k, v in meta_update.items()}
    # corrected.attrs = {**ds[varname].attrs, **encoded}
    ds = safe_assign_variable(ds, corrected_name, corrected)
    return ds


def merge_in_time(datasets, *, height_res=None, height_tol=1e-5,
                  extra_attrs=None):
    """
    Merge a list of time-indexed datasets into a single time-sorted dataset.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        Each must contain a scalar ``time`` coordinate.
        Optionally may contain a 1D ``height`` coordinate.
    height_res : float or None, optional
        If provided, interpolate all datasets with a ``height`` dimension
        onto a common height grid with this vertical resolution (km).

    Returns
    -------
    ds : xarray.Dataset
        Dataset with dimension ``time`` (and ``height`` if present),
        containing all variables stacked along time.
    """
    if not datasets:
        raise ValueError("merge_in_time: received an empty list.")

    # Validate types and extract times
    times = []
    for i, ds in enumerate(datasets):
        if not isinstance(ds, xr.Dataset):
            raise TypeError(f"#{i} is not an xarray.Dataset.")
        if "time" not in ds.coords or ds["time"].dims != ():
            raise ValueError(f"#{i} does not contain a scalar 'time' coord.")
        times.append(np.datetime64(ds["time"].item(), "ns"))

    if len(times) != len(set(times)):
        raise ValueError("Duplicate time coordinates detected.")

    # Handle height dimension if present
    has_height = all("height" in ds.coords for ds in datasets)

    if has_height and height_res is not None:
        max_height = max(float(ds.height.max()) for ds in datasets)
        common_height = np.arange(0.0, max_height + height_res, height_res)
        datasets = [ds.interp(height=common_height) for ds in datasets]

    elif has_height and height_res is None:
        ref_height = datasets[0]["height"]
        for ds in datasets[1:]:
            if not np.allclose(ds["height"], ref_height, rtol=0,
                               atol=height_tol,equal_nan=True):
                raise ValueError(
                    "Height grids differ beyond tolerance. Provide height_res "
                    f"or increase height_tol (current={height_tol}).")

    # Concatenate
    out = xr.concat(datasets, dim="time", coords="minimal", compat="override")
    out = out.assign_coords(time=("time",
                                  np.array(times, dtype="datetime64[ns]")))
    out = out.sortby("time")
    # Remove per-scan metadata that should not persist
    DROP_ATTRS = {"scan_datetime_unix_ns", "scan_datetime_iso",
                  "scan_datetime_unit", "input_processing_chain",}
    for attr in DROP_ATTRS:
        out.attrs.pop(attr, None)
    # Provenance
    input_chains = [copy.deepcopy(ds.attrs["input_processing_chain"])
                    for ds in datasets if "input_processing_chain" in ds.attrs]
    out.attrs["source_input_processing_chains"] = input_chains
    out = record_provenance(
        out, step="merge_inputs_time",
        inputs=[v for ds in datasets for v in ds.data_vars],
        outputs=list(out.data_vars),
        parameters={"n_datasets": len(datasets), "height_res_km": height_res,
                    "has_height": has_height},
        module_provenance="towerpy.utils.merge_in_time",
        extra_attrs=extra_attrs)
    return out


# =============================================================================
# %%% Encode/decode
# =============================================================================

_PROV_KEYS_DATASET = {'processing_chain', 'source_input_processing_chains',
                      'input_processing_chain'}

_PROV_KEYS_VARIABLE = {'correction_chain', 'correction_params'}

_NESTED_CHAIN_KEYS = {"source_input_processing_chains"}

_STRUCTURED_DATASET_ATTRS = {"where"}


def _sanitize_for_json(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _maybe_json_encode(value):
    if isinstance(value, (dict, list)):
        return json.dumps(_sanitize_for_json(value))
    return value


def _validate_nested_chain(value, key):
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list, got {type(value)}")
    for inner in value:
        if not isinstance(inner, list):
            raise ValueError(f"{key} must be list-of-lists, got inner {type(inner)}")


def encode_provenance(ds):
    ds = ds.copy()
    for key in _PROV_KEYS_DATASET:
        if key in ds.attrs:
            if key in _NESTED_CHAIN_KEYS:
                _validate_nested_chain(ds.attrs[key], key)
            ds.attrs[key] = _maybe_json_encode(ds.attrs[key])
    for var in ds.data_vars:
        attrs = ds[var].attrs
        for key in _PROV_KEYS_VARIABLE:
            if key in attrs:
                attrs[key] = _maybe_json_encode(attrs[key])
    return ds

def encode_cf_value(value):
    # Already a JSON string? leave it
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except Exception:
            return value  # plain string is CF-safe

    # CF-safe primitive types
    if isinstance(value, (bytes, int, float, bool)):
        return value

    # NumPy scalar types
    if isinstance(value, np.generic):
        return value.item()

    # NumPy arrays are CF-safe
    if isinstance(value, np.ndarray):
        return value

    # Lists/tuples of numbers/bools → CF-safe list
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (int, float, bool, np.generic)) for v in value):
            return [v.item() if isinstance(v, np.generic) else v for v in value]
        # structured list → JSON string
        return json.dumps(_sanitize_for_json(value))

    # Dicts → JSON string
    if isinstance(value, dict):
        return json.dumps(_sanitize_for_json(value))

    # Fallback → JSON string
    return json.dumps(_sanitize_for_json(value))


def encode_cf_flags(attrs):
    if "flags" not in attrs:
        return attrs

    flags = attrs["flags"]
    if not isinstance(flags, dict):
        return attrs

    if "flag_values" in attrs or "flag_meanings" in attrs:
        return attrs

    attrs["flag_values"] = list(flags.values())
    attrs["flag_meanings"] = " ".join(
        k.replace(" ", "_").replace("[", "").replace("]", "")
        for k in flags.keys())
    attrs["flags_json"] = json.dumps(_sanitize_for_json(flags))
    del attrs["flags"]

    return attrs


def encode_cf_attrs(attrs):
    attrs = encode_cf_flags(dict(attrs))
    for key, val in list(attrs.items()):
        attrs[key] = encode_cf_value(val)
    return attrs


def encode_CF(ds):
    ds = encode_provenance(ds)
    ds = ds.copy()

    ds.attrs = encode_cf_attrs(ds.attrs)

    for var in ds.data_vars:
        ds[var].attrs = encode_cf_attrs(ds[var].attrs)

    for coord in ds.coords:
        ds[coord].attrs = encode_cf_attrs(ds[coord].attrs)

    return ds


def _maybe_json_decode(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def decode_provenance(ds):
    ds = ds.copy()

    for key in _PROV_KEYS_DATASET:
        if key not in ds.attrs:
            continue
        decoded = _maybe_json_decode(ds.attrs[key])
        if key == "processing_chain" and isinstance(decoded, list):
            for entry in decoded:
                if "parameters" in entry:
                    entry["parameters"] = _maybe_json_decode(entry["parameters"])
        if key in _NESTED_CHAIN_KEYS and isinstance(decoded, list):
            out = []
            for item in decoded:
                inner = _maybe_json_decode(item)
                if isinstance(inner, list):
                    for entry in inner:
                        if "parameters" in entry:
                            entry["parameters"] = _maybe_json_decode(entry["parameters"])
                out.append(inner)
            decoded = out
        ds.attrs[key] = decoded

    for key in _STRUCTURED_DATASET_ATTRS:
        if key in ds.attrs:
            ds.attrs[key] = _maybe_json_decode(ds.attrs[key])

    for var in ds.data_vars:
        attrs = ds[var].attrs
        for key in _PROV_KEYS_VARIABLE:
            if key not in attrs:
                continue
            decoded = _maybe_json_decode(attrs[key])
            if key == "correction_chain" and isinstance(decoded, list):
                for entry in decoded:
                    if "params" in entry:
                        entry["params"] = _maybe_json_decode(entry["params"])
            attrs[key] = decoded

    return ds


def decode_cf_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def decode_cf_flags(attrs):
    if "flags" in attrs:
        return attrs
    if "flags_json" in attrs:
        try:
            attrs["flags"] = json.loads(attrs["flags_json"])
        except Exception:
            pass
    return attrs


def decode_cf_attrs(attrs):
    attrs = decode_cf_flags(dict(attrs))
    for key, val in list(attrs.items()):
        attrs[key] = decode_cf_value(val)
    return attrs


def decode_CF(ds):
    ds = decode_provenance(ds)
    ds = ds.copy()

    ds.attrs = decode_cf_attrs(ds.attrs)

    for var in ds.data_vars:
        ds[var].attrs = decode_cf_attrs(ds[var].attrs)

    for coord in ds.coords:
        ds[coord].attrs = decode_cf_attrs(ds[coord].attrs)

    return ds
