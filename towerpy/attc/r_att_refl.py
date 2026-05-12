"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import copy
import json
import numpy as np
from scipy.interpolate import interp1d
import xarray as xr
from ..datavis import rad_display as rdd
from ..utils import unit_conversion as tpuc
from ..utils.radutilities import (apply_correction_chain, record_provenance,
                                  safe_assign_variable, _maybe_json_encode)
from ..utils.unit_conversion import x2xdb, xdb2x


class Attn_Refl_Relation:
    r"""
    A class to compute the :math:`A_{H,V}(Z_{H,V})`.

    Attributes
    ----------
    elev_angle : float
        Elevation angle at which the scan was taken, in deg.
    file_name : str
        Name of the file containing radar data.
    scandatetime : datetime
        Date and time of scan.
    site_name : str
        Name of the radar site.
    """

    def __init__(self, radobj=None):
        self.elev_angle = getattr(radobj, 'elev_angle',
                                  None) if radobj else None
        self.file_name = getattr(radobj, 'file_name',
                                 None) if radobj else None
        self.scandatetime = getattr(radobj, 'scandatetime',
                                    None) if radobj else None
        self.site_name = getattr(radobj, 'site_name',
                                 None) if radobj else None

    def ah_zh(self, rad_vars, var2calc='ZH [dBZ]', rband='C', temp=20.,
              coeff_a=None, coeff_b=None, zh_lower_lim=20., zh_upper_lim=50.,
              rhohv_lim=0.95, copy_ofr=True, data2correct=None,
              plot_method=False):
        r"""
        Compute the :math:`A_H-Z_H` relation.

        Parameters
        ----------
        rad_vars : dict
            Radar object containing at least the specific attenuation
            :math:`(A_H)` in dB/km, or the calibrated horizontal reflectivity
            :math:`(Z_H)` in dBZ, that will be used for calculations.
        var2calc : str
            Radar variable to be computed. The string has to be one of
            'AH [dB/km]' or 'ZH [dBZ]'. The default is 'ZH [dBZ]'.
        rband: str
            Frequency band according to the wavelength of the radar.
            The string has to be one of 'C' or 'X'. The default is 'C'.
        temp: float
            Temperature, in :math:`^{\circ}C`, used to derive the coefficients
            according to [1]_. The default is 20.
        coeff_a, coeff_b: float
            Override the default coefficients of the :math:`A_H(Z_H)`
            relationship. The default are None.
        zh_lower_lim, zh_upper_lim : floats
            Thresholds in :math:`Z_H` for the :math:`A_H(Z_H)` relationship.
            Default is :math:`20 < Z_H < 50 dBZ`.
        rhohv_lim : float
            Threshold in :math:`\rho_{HV}` to filter out invalid values.
            Default is 0.95.
        copy_ofr : bool, optional
            If True, original values are used to populate out-of-range values,
            i.e., values below or above zh_limits. The default is True.
        data2correct : dict, optional
            Dictionary that will be updated with the computed variable.
            The default is None.
        plot_method : bool, optional
            Plot the :math:`A_H-Z_H` relation. The default is False.

        Returns
        -------
         vars : dict
            AH [dB/km]:
                Specific attenuation at horizontal polarisation.
            ZH [dBZ]:
                Reflectivity at horizontal polarisation not affected by partial
                beam blockage, radar miscalibration or the impact of wet radom.
            coeff_a, coeff_b:
                Interpolated coefficients of the :math:`A_H(Z_H)` relation.

        Math
        ----
        .. [Eq.1]
        .. math::  A_H = aZ_h^b
        where :math:`Z_h = 10^{0.1*Z_H}`, :math:`Z_H` in dBZ and
        :math:`A_H` in dB/km.

        Notes
        -----
        Standard values according to [1]_

        References
        ----------
        .. [1] Diederich, M., Ryzhkov, A., Simmer, C., Zhang, P., & Trömel, S.
         (2015). Use of Specific Attenuation for Rainfall Measurement at X-Band
         Radar Wavelengths. Part I: Radar Calibration and Partial Beam Blockage
         Estimation. Journal of Hydrometeorology, 16(2), 487-502.
         https://doi.org/10.1175/JHM-D-14-0066.1
        """
        if coeff_a is None and coeff_b is None:
            # Default values for the temp
            temps = np.array((0, 10, 20, 30), dtype=np.float64)
            # Default values for C- and X-band radars
            coeffs_a = {'X': np.array((1.62e-4, 1.15e-4, 7.99e-5, 5.5e-5),
                                      dtype=np.float64),
                        'C': np.array((4.27e-5, 2.89e-5, 2.09e-5, 1.59e-5),
                                      dtype=np.float64)}
            coeffs_b = {'X': np.array((0.74, 0.78, 0.82, 0.86),
                                      dtype=np.float64),
                        'C': np.array((0.73, 0.75, 0.76, 0.77),
                                      dtype=np.float64)}
            # Interpolate the temp, and coeffs to set the coeffs
            icoeff_a = interp1d(temps, coeffs_a.get(rband))
            icoeff_b = interp1d(temps, coeffs_b.get(rband))
            coeff_a = icoeff_a(temp)
            coeff_b = icoeff_b(temp)
        if var2calc == 'ZH [dBZ]':
            # Copy the original dict to keep variables unchanged
            rvars = copy.deepcopy(rad_vars)
            # Computes Zh (linear units)
            r_ahzhl = (((1 / coeff_a)**(1/coeff_b))
                       * rvars['AH [dB/km]'] ** (1 / coeff_b))
            r_ahzh = {}
            # Computes ZH (dBZ)
            r_ahzh['ZH [dBZ]'] = tpuc.x2xdb(r_ahzhl)
            # Filter values using a lower limit
            r_ahzh['ZH [dBZ]'][rvars['ZH [dBZ]'] < zh_lower_lim] = np.nan
            # Filter values using an upper limit
            r_ahzh['ZH [dBZ]'][rvars['ZH [dBZ]'] > zh_upper_lim] = np.nan
            # Filter invalid values
            r_ahzh['ZH [dBZ]'][rvars['rhoHV [-]'] < rhohv_lim] = np.nan
            if copy_ofr and 'ZH [dBZ]' in rad_vars.keys():
                # Filter invalid values
                # Use original values to populate with out-of-range values.
                ind = np.isneginf(r_ahzh['ZH [dBZ]'])
                r_ahzh['ZH [dBZ]'][ind] = rvars['ZH [dBZ]'][ind]
                ind = np.isnan(r_ahzh['ZH [dBZ]'])
                r_ahzh['ZH [dBZ]'][ind] = rvars['ZH [dBZ]'][ind]
            zh_diff = (tpuc.xdb2x(rad_vars['ZH [dBZ]'])
                       / tpuc.xdb2x(r_ahzh['ZH [dBZ]']))
            zh_diff[np.isinf(zh_diff)] = np.nan
            zh_diffdbZ = tpuc.x2xdb(zh_diff)
            r_ahzh['diff [dBZ]'] = zh_diffdbZ
        if var2calc == 'AH [dB/km]':
            # Copy the original dict to keep variables unchanged
            rvars = copy.deepcopy(rad_vars)
            # Filter values using a lower limit
            rvars['ZH [dBZ]'][rad_vars['ZH [dBZ]'] < zh_lower_lim] = np.nan
            # Filter values using an upper limit
            rvars['ZH [dBZ]'][rad_vars['ZH [dBZ]'] > zh_upper_lim] = np.nan
            # Filter invalid values
            rvars['ZH [dBZ]'][rvars['rhoHV [-]'] < rhohv_lim] = np.nan
            r_ahzh = {}
            r_ahzh['AH [dB/km]'] = (
                coeff_a * (tpuc.xdb2x(rvars['ZH [dBZ]']) ** coeff_b))
            # Filter invalid values
            # ind = np.isneginf(r_ahzh['AH [dB/km]'])
            # r_ahzh['AH [dB/km]'][ind] = rad_vars['ZH [dBZ]'][ind]
            # # Filter invalid values
            # ind = np.isnan(r_ahzh['ZH [dBZ]'])
            # r_ahzh['ZH [dBZ]'][ind] = rad_vars['ZH [dBZ]'][ind]
            if copy_ofr and 'AH [dB/km]' in rad_vars.keys():
                # Filter invalid values
                # Use original values to populate with out-of-range values.
                ind = np.isneginf(r_ahzh['AH [dB/km]'])
                r_ahzh['AH [dB/km]'][ind] = rvars['AH [dB/km]'][ind]
                ind = np.isnan(r_ahzh['AH [dB/km]'])
                r_ahzh['AH [dB/km]'][ind] = rvars['AH [dB/km]'][ind]
                ah_diff = rad_vars['AH [dB/km]'] - r_ahzh['AH [dB/km]']
                ah_diff[np.isinf(ah_diff)] = np.nan
                r_ahzh['diff [dB/km]'] = ah_diff
        self.vars = r_ahzh
        self.coeff_a = coeff_a
        self.coeff_b = coeff_b
        if data2correct is not None:
            # Copy the original dict to keep variables unchanged
            data2cc = copy.deepcopy(data2correct)
            data2cc.update(r_ahzh)
            self.vars = data2cc

        if plot_method:
            if var2calc == 'ZH [dBZ]':
                rdd.plot_zhah(rad_vars, r_ahzh, temp, coeff_a, coeff_b,
                              coeffs_a.get(rband), coeffs_b.get(rband), temps,
                              zh_lower_lim, zh_upper_lim)

    def av_zv(self, rad_vars, var2calc='ZV [dBZ]', rband='C', temp=10.,
              coeff_a=None, coeff_b=None, zv_lower_lim=20., zv_upper_lim=50.,
              rhohv_lim=0.95, copy_ofr=True, data2correct=None,
              plot_method=False):
        r"""
        Compute the :math:`A_V-Z_V` relation.

        Parameters
        ----------
        rad_vars : dict
            Radar object containing at least the specific attenuation
            :math:`(A_V)` in dB/km, or the calibrated vertical reflectivity
            :math:`(Z_V)` in dBZ, that will be used for calculations.
        var2calc : str
            Radar variable to be computed. The string has to be one of
            'AV [dB/km]' or 'ZV [dBZ]'. The default is 'ZV [dBZ]'.
        rband: str
            Frequency band according to the wavelength of the radar.
            The string has to be one of 'C' or 'X'. The default is 'C'.
        temp: float
            Temperature, in :math:`^{\circ}C`, used to derive the coefficients
            according to [1]_. The default is 10.
        coeff_a, coeff_b: float
            Override the default coefficients of the :math:`A_V(Z_V)`
            relationship. The default are None.
        zv_lower_lim, zv_upper_lim : floats
            Thresholds in :math:`Z_V` for the :math:`A_V(Z_V)` relationship.
            Default is :math:`20 < Z_V < 50 dBZ`.
        rhohv_lim : float
            Threshold in :math:`\rho_{HV}` to filter out invalid values.
            Default is 0.95.
        copy_ofr : bool, optional
            If True, original values are used to populate out-of-range values,
            i.e., values below or above zv_limits. The default is True.
        data2correct : dict, optional
            Dictionary that will be updated with the computed variable.
            The default is None.
        plot_method : bool, optional
            Plot the :math:`A_V-Z_V` relation. The default is False.

        Returns
        -------
         vars : dict
            AV [dB/km]:
                Specific attenuation at vertical polarisation.
            ZV [dBZ]:
                Reflectivity at vertical polarisation not affected by partial
                beam blockage, radar miscalibration or the impact of wet radom.
            coeff_a, coeff_b:
                Interpolated coefficients of the :math:`A_V(Z_V)` relation.

        Math
        ----
        .. [Eq.1]
        .. math::  A_V = aZ_v^b
        where :math:`Z_v = 10^{0.1*Z_V}`, :math:`Z_V` in dBZ and
        :math:`A_V` in dB/km.

        Notes
        -----
        Standard values according to [1]_

        References
        ----------
        .. [1] Diederich, M., Ryzvkov, A., Simmer, C., Zvang, P., & Trömel, S.
         (2015). Use of Specific Attenuation for Rainfall Measurement at X-Band
         Radar Wavelengths. Part I: Radar Calibration and Partial Beam Blockage
         Estimation. Journal of Hydrometeorology, 16(2), 487-502.
         https://doi.org/10.1175/JHM-D-14-0066.1
        """
        if coeff_a is None and coeff_b is None:
            # Default values for the temp
            temps = np.array((0, 10, 20, 30), dtype=np.float64)
            # Default values for C- and X-band radars
            coeffs_a = {'X': np.array((1.35e-4, 9.47e-5, 6.5e-5, 4.46e-5),
                                      dtype=np.float64),
                        'C': np.array((3.87e-5, 2.67e-5, 1.97e-5, 1.53e-5),
                                      dtype=np.float64)}
            coeffs_b = {'X': np.array((0.78, 0.82, 0.86, 0.89),
                                      dtype=np.float64),
                        'C': np.array((0.75, 0.77, 0.78, 0.78),
                                      dtype=np.float64)}
            # Interpolate the temp, and coeffs to set the coeffs
            icoeff_a = interp1d(temps, coeffs_a.get(rband))
            icoeff_b = interp1d(temps, coeffs_b.get(rband))
            coeff_a = icoeff_a(temp)
            coeff_b = icoeff_b(temp)
        if var2calc == 'ZV [dBZ]':
            # Copy the original dict to keep variables unchanged
            rvars = copy.deepcopy(rad_vars)
            # Computes Zv
            r_avzvl = (((1 / coeff_a)**(1/coeff_b))
                       * rvars['AV [dB/km]'] ** (1 / coeff_b))
            r_avzv = {}
            # Computes ZV
            r_avzv['ZV [dBZ]'] = tpuc.x2xdb(r_avzvl)
            # Filter values using a lower limit
            r_avzv['ZV [dBZ]'][rvars['ZV [dBZ]'] < zv_lower_lim] = np.nan
            # Filter values using an upper limit
            r_avzv['ZV [dBZ]'][rvars['ZV [dBZ]'] > zv_upper_lim] = np.nan
            # Filter invalid values
            r_avzv['ZV [dBZ]'][rvars['rhoHV [-]'] < rhohv_lim] = np.nan
            if copy_ofr and 'ZV [dBZ]' in rad_vars.keys():
                # Filter invalid values
                # Use original values to populate with out-of-range values.
                ind = np.isneginf(r_avzv['ZV [dBZ]'])
                r_avzv['ZV [dBZ]'][ind] = rvars['ZV [dBZ]'][ind]
                ind = np.isnan(r_avzv['ZV [dBZ]'])
                r_avzv['ZV [dBZ]'][ind] = rvars['ZV [dBZ]'][ind]
            zv_diff = (tpuc.xdb2x(rad_vars['ZV [dBZ]'])
                       / tpuc.xdb2x(r_avzv['ZV [dBZ]']))
            zv_diff[np.isinf(zv_diff)] = np.nan
            zv_diffdbZ = tpuc.x2xdb(zv_diff)
            r_avzv['diff [dBZ]'] = zv_diffdbZ
        if var2calc == 'AV [dB/km]':
            # Copy the original dict to keep variables unchanged
            rvars = copy.deepcopy(rad_vars)
            # Filter values using a lower limit
            rvars['ZV [dBZ]'][rad_vars['ZV [dBZ]'] < zv_lower_lim] = np.nan
            # Filter values using an upper limit
            rvars['ZV [dBZ]'][rad_vars['ZV [dBZ]'] > zv_upper_lim] = np.nan
            # Filter invalid values
            rvars['ZV [dBZ]'][rvars['rhoHV [-]'] < rhohv_lim] = np.nan
            r_avzv = {}
            r_avzv['AV [dB/km]'] = (
                coeff_a * (tpuc.xdb2x(rvars['ZV [dBZ]']) ** coeff_b))
            # Filter invalid values
            # ind = np.isneginf(r_avzv['AV [dB/km]'])
            # r_avzv['AV [dB/km]'][ind] = rad_vars['ZV [dBZ]'][ind]
            # # Filter invalid values
            # ind = np.isnan(r_avzv['ZV [dBZ]'])
            # r_avzv['ZV [dBZ]'][ind] = rad_vars['ZV [dBZ]'][ind]
            if copy_ofr and 'AV [dB/km]' in rad_vars.keys():
                # Filter invalid values
                # Use original values to populate with out-of-range values.
                ind = np.isneginf(r_avzv['AV [dB/km]'])
                r_avzv['AV [dB/km]'][ind] = rvars['AV [dB/km]'][ind]
                ind = np.isnan(r_avzv['AV [dB/km]'])
                r_avzv['AV [dB/km]'][ind] = rvars['AV [dB/km]'][ind]
                av_diff = rad_vars['AV [dB/km]'] - r_avzv['AV [dB/km]']
                av_diff[np.isinf(av_diff)] = np.nan
                r_avzv['diff [dB/km]'] = av_diff
        self.vars = r_avzv
        self.coeff_a = coeff_a
        self.coeff_b = coeff_b
        if data2correct is not None:
            # Copy the original dict to keep variables unchanged
            data2cc = copy.deepcopy(data2correct)
            data2cc.update(r_avzv)
            self.vars = data2cc

        if plot_method:
            rdd.plot_zhah(rad_vars, temp, coeff_a, coeff_b,
                          coeffs_a.get(rband), coeffs_b.get(rband), temps)

# =============================================================================
# %% xarray implementation
# =============================================================================


def _a_from_z_core(z_lin, a_coeff, b_coeff):
    return a_coeff * z_lin**b_coeff


def _z_from_a_core(a_lin, a_coeff, b_coeff):
    return (a_lin / a_coeff)**(1.0 / b_coeff)


def get_a_z_coeffs(pol="H", rband="C", temp=20, coeff_a_override=None,
                   coeff_b_override=None):
    """
    Return (a, b) coefficients for the A(Z) relation for either H or V
    polarisation, with optional overrides.

    Parameters
    ----------
    pol : {"H", "V"}
        Polarisation. Determines whether AH–ZH or AV–ZV tables are used.
    rband : {"C", "X"}
        Radar frequency band.
    temp : float
        Temperature in °C for interpolation.
    coeff_a_override, coeff_b_override : float, optional
        If both are provided, they override the table values.

    Returns
    -------
    (a, b) : tuple of floats
        Interpolated or overridden coefficients for the A(Z) relation.

    Notes
    -----
    1. Coefficients follow Diederich et al. (2015), JHM 16(2), 487–502.
    """
    pol = pol.upper()
    rband = rband.upper()
    # Overrides both coefficients
    if coeff_a_override is not None and coeff_b_override is not None:
        return float(coeff_a_override), float(coeff_b_override)
    # Temperature grid used in the paper
    temps = np.array((0.0, 10.0, 20.0, 30.0), dtype=float)
    # Coefficient tables
    if pol == "H":
        coeffs_a = {"X": np.array((1.62e-4, 1.15e-4, 7.99e-5, 5.50e-5)),
                    "C": np.array((4.27e-5, 2.89e-5, 2.09e-5, 1.59e-5)),
                    }
        coeffs_b = {"X": np.array((0.74, 0.78, 0.82, 0.86)),
                    "C": np.array((0.73, 0.75, 0.76, 0.77)),
                    }
    elif pol == "V":
        coeffs_a = {"X": np.array((1.35e-4, 9.47e-5, 6.50e-5, 4.46e-5)),
                    "C": np.array((3.87e-5, 2.67e-5, 1.97e-5, 1.53e-5)),
                    }
        coeffs_b = {"X": np.array((0.78, 0.82, 0.86, 0.89)),
                    "C": np.array((0.75, 0.77, 0.78, 0.78)),
                    }
    else:
        raise ValueError(f"Invalid pol={pol!r}. Must be 'H' or 'V'.")
    valid_rbands = set(list(coeffs_a.keys()) + list(coeffs_b.keys()))
    if rband not in valid_rbands:
        raise ValueError(f"Invalid band={rband!r}. Must be {valid_rbands}.")
    # Interpolate temperature
    a = np.interp(temp, temps, coeffs_a[rband])
    b = np.interp(temp, temps, coeffs_b[rband])

    return float(a), float(b)


def _compute_a_z(ds, *, pol, derive_from, z_name, a_name, out_name, diff_name,
                 rhohv_name, rband, temp, coeff_a, coeff_b, z_limits, rhohv_min,
                 copy_out_of_lims):
    """
    Core engine for A–Z relations (AH–ZH, AV–ZV and inverses).

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    pol : {"H", "V"}
        Polarisation.
    derive_from : {"radrefl", "spcfatt"}
        Whether to compute A from Z or Z from A.
    z_name, a_name : str
        Canonical variable names for Z and A.
    out_name : str
        Name of the output variable.
    diff_name : str or None
        Name of the diff variable.
    rhohv_name : str
        Name of the rhoHV variable.
    rband : {"C", "X"}
        Radar band.
    temp : float
        Temperature for coefficient interpolation.
    coeff_a, coeff_b : float
        Coefficients for the A(Z) relation.
    z_limits : (float, float)
        Lower and upper Z limits.
    rhohv_min : float
        Minimum rhoHV threshold.
    copy_out_of_lims : bool
        Whether to copy parent values for out‑of‑limits results.
    """
    if derive_from not in {"radrefl", "spcfatt"}:
        raise ValueError(f"derive_from must be 'radrefl' or 'spcfatt', got {derive_from!r}")

    if pol not in {"H", "V"}:
        raise ValueError(f"pol must be 'H' or 'V', got {pol!r}")

    if rhohv_name not in ds:
        raise KeyError(f"{rhohv_name!r} not found in Dataset")
    rhohv = ds[rhohv_name]
    zmin, zmax = z_limits
    # Dask-safe wrappers
    def _apply(func, da, **kwargs):
        return xr.apply_ufunc(func, da, kwargs=kwargs, dask="parallelized",
                              output_dtypes=[float])

    def _isneginf(da):
        return xr.apply_ufunc(np.isneginf, da, dask="parallelized",
                              output_dtypes=[bool])
    # FORWARD: AH/AV from ZH/ZV
    if derive_from == "radrefl":
        if z_name not in ds:
            raise KeyError(f"{z_name!r} not found in Dataset")
        z = ds[z_name]
        # Mask by Z-limits and rhoHV
        valid = (z >= zmin) & (z <= zmax) & (rhohv >= rhohv_min)
        z_masked = z.where(valid)
        # Convert Z (dBZ) -> Zh (linear)
        zh_lin = _apply(xdb2x, z_masked)
        # Apply AH/AV = a * Zh^b
        a_da = _apply(_a_from_z_core, zh_lin, a_coeff=coeff_a, b_coeff=coeff_b)
        # Copy out-of-range from parent A if requested
        parent_exists = a_name in ds
        if copy_out_of_lims and parent_exists:
            parent_a = ds[a_name]
            bad = _isneginf(a_da) | a_da.isnull()
            a_da = xr.where(bad, parent_a, a_da)
        # Compute diff if parent exists: A_parent - A_new
        diff_da = None
        if parent_exists:
            parent_a = ds[a_name]
            diff_da = parent_a - a_da
            diff_da = diff_da.where(~xr.apply_ufunc(
                np.isinf, diff_da, dask="parallelized", output_dtypes=[bool]))
        ds_out = ds.assign({out_name: a_da})
        if diff_name and diff_da is not None:
            ds_out = ds_out.assign({diff_name: diff_da})
        return ds_out
    # INVERSE: ZH/ZV from AH/AV
    if a_name not in ds:
        raise KeyError(f"{a_name!r} not found in Dataset")
    a_da = ds[a_name]
    # Only rhoHV mask for inverse relation
    valid = (rhohv >= rhohv_min)
    a_masked = a_da.where(valid)
    # Compute Zh (linear) from AH/AV
    zh_lin = _apply(_z_from_a_core, a_masked, a_coeff=coeff_a, b_coeff=coeff_b)
    # Convert Zh (linear) -> Z (dBZ)
    z_da = _apply(x2xdb, zh_lin)
    # Copy out-of-range from parent Z if requested
    parent_exists = z_name in ds
    if copy_out_of_lims and parent_exists:
        parent_z = ds[z_name]
        bad = _isneginf(z_da) | z_da.isnull()
        z_da = xr.where(bad, parent_z, z_da)
    # Compute diff if parent exists: Z_parent - Z_new (in dB)
    diff_da = None
    if parent_exists:
        parent_z = ds[z_name]
        parent_z_lin = _apply(xdb2x, parent_z)
        new_z_lin = _apply(xdb2x, z_da)
        ratio = parent_z_lin / new_z_lin
        ratio = ratio.where(~xr.apply_ufunc(
            np.isinf, ratio, dask="parallelized", output_dtypes=[bool]))
        diff_da = _apply(x2xdb, ratio)
    ds_out = ds.assign({out_name: z_da})
    if diff_name and diff_da is not None:
        ds_out = ds_out.assign({diff_name: diff_da})
    return ds_out

def rel_a_z(ds, pol="H", derive_from="spcfatt", inp_names=None, rband="C",
            temp=20., coeff_a=None, coeff_b=None, z_limits=(20., 50.),
            rhohv_min=0.95, copy_out_of_lims=True, apply_maf=False,
            threshold=0.25, mov_avrgf_len=(1, 5), apply_rr_only=True,
            detect_fval=True, merge_into_ds=True, replace_vars=False,
            modify_output=None,):
    r"""
    Compute the empirical :math:`A(Z)` or :math:`Z(A)` relationship for
    horizontal or vertical polarisation, following the formulation of
    Diederich et al. (2014).

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing Z, A, rhoHV, and optionally ML_PCP_CLASS,
        along with the polar coordinates (range, azimuth, elevation).
    pol : {"H", "V"}, default "H"
        Polarisation. Determines whether :math:`A_H(Z_H)` or :math:`A_V(Z_V)`
        are used.
    derive_from : {"radrefl", "spcfatt"}, default "spcfatt"
        Variable to derive: "radrefl" for A from Z, "spcfatt" for Z from A.
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{'azi': 'azimuth', 'rng': 'range', 'elv': 'elevation', "ZH": "DBZH",
        "ZV": "DBZV", "AH": "AH", "AV": "AV", "RHOHV": "RHOHV",
        "ML_PCP_CLASS": "ML_PCP_CLASS"}``.
    rband : {"C", "X"}, default "C"
        Radar band used to select default coefficients.
    temp : float, default 20.
        Temperature (°C) used for coefficient interpolation.
    coeff_a, coeff_b : float, optional
        Override the default coefficients of the :math:`A(Z)` relationship.
    z_limits : (float, float)
        Valid range of :math:`Z` (dBZ) for applying the :math:`A(Z)` model.
        Default is ``(20, 50)``.
    rhohv_min : float, default 0.95
        Minimum :math:`\rho_{HV}` threshold for valid gates.
    copy_out_of_lims : bool, default True
        If ``True``, copy parent values for gates outside ``z_limits``.
    apply_maf : bool, default False
        If ``True``, apply azimuth‑wise moving‑average smoothing to the
        :math:`Z` field derived from :math:`A`.
    mov_avrgf_len : tuple, default (1, 5)
        Moving‑average window length ``(n_range, n_azimuth)``. The window
        must be of the form ``(1, m)``
    threshold : float, default 0.25
        Minimum fraction of valid differences required per ray when applying
        moving‑average smoothing.
    apply_rr_only : bool, default True
        If ``True``, restrict smoothing to the precipitation region.
    detect_fval : bool, default True
        If ``True``, begin correction after the first finite non‑zero
        difference.
    merge_into_ds : bool, default True
        If True, merge corrected outputs into a full copy of the input dataset.
        If False, return a dataset containing only the corrected outputs.
    replace_vars : bool, default False
        If True, overwrite the parent variable (ZH/ZV or AH/AV).
        If False, corrected variables receive the scientific suffix "_ZAR"
        unless explicit names are provided via modify_output.
    modify_output : bool | list[str] | dict[str, str] | None
        Controls which logical outputs are written and how they are named.
        Logical outputs:
            - MAIN: "ZAR" (corrected Z or A)
            - DIFF: "DIFF"
            - MAF:  "MAF"
            - MAF_DIFF: "MAF_DIFF"

        None / True → write all applicable outputs.
        list → write only listed logical outputs.
        dict → map logical outputs to explicit output names.

    Returns
    -------
    xr.Dataset
        Dataset with computed A or Z field, and optional smoothed Z.

        A : dB/km
            Specific attenuation (when derived from reflectivity).
        Z dBZ: dBZ
            Reflectivity derived from specific attenuation. Z is not affected
            by partial beam blockage, radar miscalibration or the impact of wet
            radom.
        coeff_a, coeff_b:
            Interpolated coefficients of the :math:`A(Z)` relation.

    Notes
    -----
    * This function operates in native polar radar coordinates.
    * The empirical model follows [1]_ and uses the power‑law form:
        .. math::  A_{H/V} = aZ_{h/v}^b
        where
            - :math:`Z_{h/v} = 10^{0.1*Z_{H/V}}`
            - :math:`Z_{H/V}` in dBZ
            - :math:`A_{H/V}` in dB/km.


    References
    ----------
    .. [1] Diederich, M., Ryzhkov, A., Simmer, C., Zhang, P., & Trömel, S.
        (2014). Use of specific attenuation for rainfall measurement at X-Band
        radar wavelengths. Part I: Radar calibration and partial beam blockage
        estimation. Journal of Hydrometeorology, 16(2), 487–502.
        https://doi.org/10.1175/jhm-d-14-0066.1
    """
    from ..io import modeltp as mdtp

    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # 1. Resolve names
    defaults = {'azi': 'azimuth', 'rng': 'range', 'elv': 'elevation',
                "ZH": "DBZH", "ZV": "DBZV", "AH": "AH", "AV": "AV",
                "RHOHV": "RHOHV", "ML_PCP_CLASS": "ML_PCP_CLASS"}
    names = {**defaults, **(inp_names or {})}
    if pol == "H":
        z_name = names["ZH"]
        a_name = names["AH"]
    else:
        z_name = names["ZV"]
        a_name = names["AV"]
    rhohv_name = names["RHOHV"]
    pcp_name = names["ML_PCP_CLASS"]
    azi = names["azi"]
    rng = names["rng"]
    # 2. Prepare output dataset
    if merge_into_ds:
        ds_out = ds.copy()
    else:
        ds_out = xr.Dataset(coords=ds.coords, attrs=ds.attrs.copy())
    # 3. Compute A–Z relation
    a_coeff, b_coeff = get_a_z_coeffs(pol=pol, rband=rband, temp=temp,
                                      coeff_a_override=coeff_a,
                                      coeff_b_override=coeff_b)
    dstmp = _compute_a_z(ds, pol=pol, derive_from=derive_from, z_name=z_name,
                         a_name=a_name, rhohv_name=rhohv_name, rband=rband,
                         temp=temp, coeff_a=a_coeff, coeff_b=b_coeff,
                         z_limits=z_limits, rhohv_min=rhohv_min,
                         copy_out_of_lims=copy_out_of_lims,
                         out_name="__AZ_OUT_TMP__",
                         diff_name="__AZ_DIFF_TMP__")
    out_arr = dstmp["__AZ_OUT_TMP__"]
    diff_arr = dstmp.get("__AZ_DIFF_TMP__", None)
    # 4. Determine parent and output names
    parent_name = z_name if derive_from == "spcfatt" else a_name
    if replace_vars:
        main_out_name = parent_name
    else:
        main_out_name = f"{parent_name}_ZAR"
    # modify_output override
    if isinstance(modify_output, dict):
        main_out_name = modify_output.get("ZAR", main_out_name)
    # Apply correction chain
    ds_out = apply_correction_chain(
        ds_out, varname=parent_name, step="A_Z_relation",
        suffix="" if replace_vars else "_ZAR",
        corrected_field=out_arr,
        params={"pol": pol, "derive_from": derive_from, "rband": rband,
                "temp": float(temp), "coeff_a": float(a_coeff),
                "coeff_b": float(b_coeff), "z_limits": list(z_limits),
                "rhohv_min": float(rhohv_min),
                "copy_out_of_lims": bool(copy_out_of_lims)},
        module_provenance="towerpy.attc.r_att_refl.rel_a_z")
    # Add description when deriving Z from A
    # if derive_from == "spcfatt":
    #     desc = ds_out[main_out_name].attrs.get("description", "")
    #     extra = ("Corrects effects of partial beam blockage, "
    #              "radar miscalibration and the impact of wet radom.")
    #     ds_out[main_out_name].attrs["description"] = (
    #         desc.rstrip(".") + "; " + extra if desc else extra)
    # 5b. Optional diff field
    if diff_arr is not None and not apply_maf:
        if derive_from == "spcfatt":
            base_diff = f"{z_name}_DIFF"
        else:
            base_diff = f"{a_name}_DIFF"
        diff_out_name = base_diff
        if not replace_vars:
            diff_out_name = f"{base_diff}_ZAR"
        if isinstance(modify_output, dict):
            diff_out_name = modify_output.get("DIFF", diff_out_name)
        parent_attrs = ds[parent_name].attrs.copy()
        canon = sweep_vars_attrs_f.get(diff_out_name, {}).copy()
        attrs = {**parent_attrs, **canon}
        attrs["description"] = ("Difference between input and A(Z)/Z(A)-"
                                "derived field.")
        da_diff = xr.DataArray(diff_arr, dims=(azi, rng),
                               coords={azi: ds[azi], rng: ds[rng]},
                               attrs=attrs)
        ds_out = safe_assign_variable(ds_out, diff_out_name, da_diff)
    # 6. Optional az smoothing
    if apply_maf and derive_from == "spcfatt":
        pcp_region = ds.get(pcp_name, None)
        z_attc = ds[z_name]
        z_from_a = out_arr
        diffs = z_attc - z_from_a
        z_maf = apply_MAF_smoothing(z_attc=z_attc, z_from_a=z_from_a,
                                    diffs=diffs, pcp_region=pcp_region,
                                    threshold=threshold, rng_name=rng,
                                    azi_name=azi, mov_avrgf_len=mov_avrgf_len,
                                    apply_rr_only=apply_rr_only,
                                    detect_fval=detect_fval)
        # MAF naming logic
        if replace_vars:
            maf_name = parent_name
        else:
            maf_name = f"{parent_name}_ZAR_MAF"
        if isinstance(modify_output, dict):
            maf_name = modify_output.get("MAF", maf_name)
        # Build attrs
        parent_attrs = ds_out[main_out_name].attrs.copy()
        canon = sweep_vars_attrs_f.get(maf_name, {}).copy()
        attrs = {**parent_attrs, **canon}
        raw = attrs.get("correction_params", {})
        if isinstance(raw, str):
            try:
                corr_params = json.loads(raw)
            except Exception:
                corr_params = {}
        else:
            corr_params = dict(raw)
        corr_params["apply_maf"] = True
        corr_params["apply_maf_params"] = {
            "threshold": float(threshold),
            "mov_avrgf_len": tuple(mov_avrgf_len),
            "apply_rr_only": bool(apply_rr_only),
            "detect_fval": bool(detect_fval)}
        attrs["correction_params"] = _maybe_json_encode(corr_params)
        z_maf = z_maf.assign_attrs(attrs)
        ds_out = safe_assign_variable(ds_out, maf_name, z_maf)
        # MAF DIFF (never overwrites)
        parent_z = ds[z_name]
        ratio = xdb2x(parent_z) / xdb2x(z_maf)
        ratio = ratio.where(np.isfinite(ratio))
        diff_final = x2xdb(ratio)
        if replace_vars:
            maf_diff_name = f"{z_name}_DIFF"
        else:
            maf_diff_name = f"{z_name}_ZAR_MAF_DIFF"
        if isinstance(modify_output, dict):
            maf_diff_name = modify_output.get("MAF_DIFF", maf_diff_name)
        parent_attrs = parent_z.attrs.copy()
        canon = sweep_vars_attrs_f.get(maf_diff_name, {}).copy()
        attrs = {**parent_attrs, **canon}
        attrs["description"] = ("Difference between input Z and final "
                                "A(Z)-derived Z (smoothed).")
        da_diff = xr.DataArray(diff_final, dims=(azi, rng),
                               coords={azi: ds[azi], rng: ds[rng]},
                               attrs=attrs)
        ds_out = safe_assign_variable(ds_out, maf_diff_name, da_diff)
    # 7. Dataset-level provenance
    extra = {"step_description":
             ("Corrects effects of partial beam blockage, radar miscalibration"
              " and the impact of wet radom.")}
    created = []
    # main output name
    created.append(main_out_name)
    # optional diff
    if diff_arr is not None and not apply_maf:
        created.append(diff_out_name)
    # optional MAF + MAF_DIFF
    if apply_maf and derive_from == "spcfatt":
        created.append(maf_name)
        created.append(maf_diff_name)
    ds_out = record_provenance(
        ds_out, step="compute_A_Z",  # function="Z(A)_relation", 
        inputs=[z_name, a_name, rhohv_name], outputs=created,
        parameters={"pol": pol, "derive_from": derive_from, "rband": rband,
                    "temp": float(temp), "coeff_a": float(a_coeff),
                    "coeff_b": float(b_coeff), "z_limits": list(z_limits),
                    "rhohv_min": float(rhohv_min),
                    "copy_out_of_lims": bool(copy_out_of_lims),
                    "apply_maf": bool(apply_maf),
                    "threshold": float(threshold),
                    "mov_avrgf_len": tuple(mov_avrgf_len),
                    "apply_rr_only": bool(apply_rr_only),
                    "detect_fval": bool(detect_fval),
                    "merge_into_ds": bool(merge_into_ds),
                    "replace_vars": bool(replace_vars),
                    "modify_output": modify_output},
        extra_attrs=extra, module_provenance="towerpy.attc.r_att_refl.rel_a_z")
    return ds_out


def apply_MAF_smoothing(z_attc, z_from_a, diffs, pcp_region=None, threshold=0.25,
                      mov_avrgf_len=(1, 5), apply_rr_only=True,
                      detect_fval=True, azi_name="azimuth", rng_name="range"):
    """
    Path-wise smoothing of Z estimates.
    """
    # 1. Mask zeros as NaN
    diffs = diffs.where(diffs != 0)
    # 2. Count valid diffs per ray
    diffs_valid = diffs.notnull().sum(dim=rng_name)
    # 3. Count valid originals (with or without ML_PCP_CLASS)
    if pcp_region is not None and apply_rr_only:
        flags = pcp_region.attrs.get("flags", {})
        rain_flag = flags.get("rain", 1.0)
        rain_mask = (pcp_region == rain_flag)
        valid_orig = (z_attc.notnull() & rain_mask).sum(dim=rng_name)
    else:
        rain_mask = None
        valid_orig = z_attc.notnull().sum(dim=rng_name)
    # 4. Fraction of valid diffs
    valid_fraction = diffs_valid / valid_orig.where(valid_orig != 0)
    # 5. Median diff per ray (over range)
    median_diff = diffs.median(dim=rng_name, skipna=True).fillna(0)
    # 6. Smooth median along azimuth
    win = mov_avrgf_len[1]
    smoothed = (median_diff.rolling({azi_name: win}, center=True,
                                    min_periods=1).mean())
    # 7. Detect first finite non-zero diff per ray
    if detect_fval:
        nonzero_mask = (diffs != 0) & diffs.notnull()
        first_nonzero = nonzero_mask.argmax(dim=rng_name)
        rng_index = xr.DataArray(diffs[rng_name].data, dims=[rng_name],
                                 coords={rng_name: diffs[rng_name]})
        after_first = rng_index >= first_nonzero
    else:
        after_first = xr.ones_like(diffs, dtype=bool)
    # 8. Combine masks (broadcast row_mask to 2D)
    row_mask = (valid_fraction >= threshold)
    row_mask_2d = row_mask.broadcast_like(diffs)
    full_mask = row_mask_2d & after_first
    # 9. Apply correction (broadcast median/smoothed to Z shape)
    median_2d = median_diff.broadcast_like(z_attc)
    smoothed_2d = smoothed.broadcast_like(z_attc)
    corrected = xr.where(median_2d == 0, z_attc, z_attc - smoothed_2d)
    z_maf = xr.where(full_mask, corrected, z_from_a)
    # 10. Restrict to precipitation region
    if apply_rr_only and pcp_region is not None:
        z_maf = xr.where(rain_mask, z_maf, z_attc)
    return z_maf
