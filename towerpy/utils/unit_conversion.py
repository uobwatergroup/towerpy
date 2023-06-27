"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np


def x2xdb(xls):
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


def xdb2x(xdb):
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
