"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np
from scipy import interpolate
import cartopy.io.shapereader as shpreader


def find_nearest(iarray, val2search):
    """
    Return the index of the closest value to a given number.

    Parameters
    ----------
    iarray : array
             Input array.
    val2search : float or int
                 Value to search into the array.

    Returns
    -------
    idx : float or int
        Index into the array.

    """
    a = np.asarray(iarray)
    idx = (np.abs(a - val2search)).argmin()
    return idx


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
    x = np.array(x)
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

    Returns
    -------
    TYPE
        DESCRIPTION.

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

        :math:`ZH` [dBZ]: [np.NINF, np.inf]

        :math:`Z_{DR}` [dB]: [np.NINF, np.inf]

        :math:`\Phi_{DP}` [deg]: [np.NINF, np.inf]

        :math:`\rho_{HV}` [-]: [np.NINF, np.inf]

        :math:`V` [m/s]: [np.NINF, np.inf]

        :math:`LDR` [dB]: [np.NINF, np.inf]35, 35, 3]

    Returns
    -------
    mafvars : dict
        Transformed data.

    """
    lpv = {'ZH [dBZ]': [np.NINF, np.inf], 'ZDR [dB]': [np.NINF, np.inf],
           'PhiDP [deg]': [np.NINF, np.inf], 'rhoHV [-]': [np.NINF, np.inf],
           'V [m/s]': [np.NINF, np.inf], 'LDR [dB]': [np.NINF, np.inf],
           'Rainfall [mm/h]': [np.NINF, np.inf],
           'KDP [deg/km]': [np.NINF, np.inf]}
    for rkey in rad_vars.keys():
        if rkey not in lpv:
            lpv[rkey] = [np.NINF, np.inf]
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
