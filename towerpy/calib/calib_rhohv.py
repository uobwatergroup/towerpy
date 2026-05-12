"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import copy
import datetime as dt

import numpy as np
import xarray as xr
import xradar as xrd
from scipy.optimize import minimize_scalar
from sklearn.metrics import root_mean_squared_error as sklrmse

from ..datavis.rad_display import _plot_rhohvmethod_grid, _plot_rhohvmethod_single
from ..eclass.snr import signal2noiseratio
from ..utils.radutilities import (
    apply_correction_chain,
    record_provenance,
    safe_assign_variable,
    xr_hist2d,
)
from ..utils.unit_conversion import convert


class rhoHV_Calibration:
    r"""
    A class for calibrating the correlation coefficient :math:`\rho_{HV}`.

    Attributes
    ----------
    elev_angle : float
        Elevation angle at where the scan was taken, in degrees.
    file_name : str
        Name of the file containing radar data.
    scandatetime : datetime
        Date and time of scan.
    site_name : str
        Name of the radar site.
    vars : dict
        Corrected :math:`\rho_{HV}` and user-defined radar variables.
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
    
    def rhohv_noise_correction(self, rad_georef, rad_params, rad_vars,
                               mode="exp", exp_curvet=20.0, eps=0.005,
                               rhohv_theo=(0.9, 1.0), noise_level=(0, 100), 
                               bins_rho=(0.8, 1.1, 0.005), bins_snr=(5, 30, 0.1),
                               plot_method=False, data2correct=None):
        r"""
        Applies noise‑bias correction to :math:`\rho_{HV}`.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used for the correction method.
            The default is None.
        data2correct : dict, optional
            Dictionary to update the corrected :math:`\rho_{HV}`.
            The default is None.

        Notes
        -----
        1. Based on the method described in [1]_
        2. See the xarray implementation (rhohv_noisecorrection) for more details.

        References
        ----------
        .. [1] Ryzhkov, A. V.; Zrnic, D. S. (2019). Radar Polarimetry for
            Weather Observations (1st ed.). Springer International Publishing.
            https://doi.org/10.1007/978-3-030-05093-1
        """
        # =============================================================================
        # assign radar params
        # =============================================================================
        prms_mod = {k1.replace(' ', ('_')).replace('[m/s]', '[ms-1]'):
                    v1 for k1, v1 in rad_params.items()
                    if not isinstance(v1, dict) and not isinstance(v1, dt.datetime)
                    and not isinstance(v1, np.ndarray)}
        prms_mod2 = {
            f"{outer_key.replace('/', '_')}({inner_key})": value
            for outer_key, inner_dict in rad_params.items()
            if isinstance(inner_dict, dict) for inner_key, value in inner_dict.items()}
        prms_mod = prms_mod | prms_mod2
        prms_mod['datetime'] = rad_params['datetime'].strftime(
            "%Y-%m-%dT%H:%M:%S.%f_%Z%z")
        prms_mod['timestamp'] = rad_params['datetime'].timestamp()
        prms_mod['file_name'] = self.file_name
        # =============================================================================
        # assign rvars
        # =============================================================================
        sweep_vars_mapping = {'DBZH': 'ZH [dBZ]', 'ZDR': 'ZDR [dB]',
                              'PHIDP': 'PhiDP [deg]', 'RHOHV': 'rhoHV [-]',
                              'VRADH': 'V [m/s]'}
        sweep_vars_attrs_f = xrd.model.sweep_vars_mapping
        azimuth = np.rad2deg(rad_georef['azim [rad]'])
        elevation = np.rad2deg(rad_georef['elev [rad]'])
        elevation_attrs = xrd.model.get_elevation_attrs()
        rng = rad_georef['range [m]']
        azimuth_attrs = xrd.model.get_azimuth_attrs(azimuth)
        range_attrs = xrd.model.get_range_attrs(rng)
        sweep = xr.Dataset(coords=dict(azimuth=(["azimuth"], azimuth, azimuth_attrs), 
                           elevation=(["azimuth"], elevation, elevation_attrs),
                           range=(["range"], rng, range_attrs)))
        sweep = sweep.assign_coords(sweep_mode="azimuth_surveillance",
                                    longitude=rad_params['longitude [dd]'],
                                    latitude=rad_params['latitude [dd]'],
                                    altitude=rad_params['altitude [m]'],
                                    time=rad_params['aztime'])
        for k1, v1 in sweep_vars_mapping.items():
            if v1 in rad_vars.keys():
                sweep = sweep.assign({k1: (['azimuth', 'range'],
                                           rad_vars[v1])})
            else:
                print(f'{k1} not in data')
            if k1 in sweep_vars_attrs_f:
                sweep[k1].attrs = sweep_vars_attrs_f[k1]
            else:
                print(f'{k1} moment_attrs not assigned ')
        
        sweep.attrs = prms_mod
        for vv in sweep.data_vars:
            if sweep[vv].dtype == "float" or sweep[vv].dtype == "float32" or sweep[vv].dtype == "float64":
                sweep[vv].encoding = {'zlib': True, 'complevel': 6}

        rhohv_nc = rhohv_noisecorrection(
            sweep, inp_names={"DBZ": "DBZH", "rng": "range", "RHOHV": "RHOHV"},
            rhohv_theo=rhohv_theo, mode=mode, noise_level=noise_level,
            exp_curvet=exp_curvet, eps=eps, bins_rho=bins_rho, bins_snr=bins_snr,
            # preserve_original=False,
            replace_vars=True,
            # data2correct=None,
            # vars2correct=['RHOHV'],
            # vars2correct=['RHOHV'], out_names={'RHOHV': 'RHOHV'},
            mask={'RHOHV': "RHOHV"},
            plot_method=plot_method)
        
        if data2correct is None:
            self.vars = {'rhoHV [-]': rhohv_nc.RHOHV.values}
        else:
            data2cc = copy.deepcopy(data2correct)
            data2cc.update({'rhoHV [-]': rhohv_nc.RHOHV.values})
            self.vars = data2cc
        # self.noise_level_dB = rhohv_nc.attrs['noise_level_dB']
        self.noise_level_dB = (next(item['parameters']['noise_level_dB']
                        for item in rhohv_nc.attrs['processing_chain']
                        if item['step'] == 'rhohv_noisecorrection'))


# =============================================================================
# %% xarray implementation
# =============================================================================


def _build_theo_line(snr_centers, rhohv_theo, mode="linear", exp_curvet=20.0,
                     eps=0.005):
    """Internal helper for rhohv_noisecorrection."""
    rho0, rho_inf = rhohv_theo
    if mode == "linear":
        return np.linspace(rho0, rho_inf, len(snr_centers))
    elif mode == "exp":
        s0 = snr_centers[0]
        k = np.log((rho_inf - rho0) / eps) / (exp_curvet - s0)
        return rho_inf - (rho_inf - rho0) * np.exp(-k * (snr_centers - s0))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _rmse_objective(rc, Z, rng_km, rhohv_na, bins_snr, bins_rho, rhohv_theo,
                    mode="linear", exp_curvet=20.0, eps=0.005):
    '''Internal helper for rhohv_noisecorrection.'''
    snr_db = signal2noiseratio(Z, rng_km, rc, scale="db").rename("snr_db")
    snr_lin = signal2noiseratio(Z, rng_km, rc, scale="lin").rename("snr_lin")
    rhohv_corr = (rhohv_na * (1 + 1 / snr_lin)).rename("rhohv_corr")

    snr_edges = np.arange(*bins_snr)
    rho_edges = np.arange(*bins_rho)
    hist = xr_hist2d(snr_db, rhohv_corr, snr_edges, rho_edges,
                     dim=list(snr_db.dims))
    rhohv_bin_dim = "y_bin"
    idx = hist.argmax(dim=rhohv_bin_dim)

    rhohv_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    histmax = xr.apply_ufunc(lambda i: rhohv_centers[i], idx,
                             vectorize=True, dask="parallelized",
                             output_dtypes=[float]).values

    snr_centers = 0.5 * (snr_edges[:-1] + snr_edges[1:])
    theo_line = _build_theo_line(snr_centers, rhohv_theo, mode=mode, exp_curvet=exp_curvet, eps=eps)

    return sklrmse(theo_line, histmax)

def _optimise_noise_level(Z, rng_km, rhohv_na, bins_rho=(0.8, 1.1, 0.005),
                         bins_snr=(5, 30, 0.1), rhohv_theo=(0.90, 1.),
                         noise_level=(0, 100), mode="linear", exp_curvet=20.0,
                         eps=0.005):
    '''Internal helper for rhohv_noisecorrection.'''
    def objective(rc):
        return _rmse_objective(rc, Z, rng_km, rhohv_na,
                               bins_snr, bins_rho, rhohv_theo,
                               mode=mode, exp_curvet=exp_curvet, eps=eps)
    result = minimize_scalar(objective, bounds=noise_level, method="bounded")
    return result.x, result.fun


def rhohv_noisecorrection(ds, inp_names=None, rhohv_theo=(0.9, 1.0),
                          noise_level=(0, 100), bins_rho=(0.8, 1.1, 0.005),
                          bins_snr=(5, 30, 0.1), mode="exp", exp_curvet=20.0,
                          eps=0.005, mask=True, replace_vars=False,
                          plot_method=False):
    r"""
    Correct noise-bias in the radar correlation coefficient (rhoHV), by fitting
    a theoretical :math:`\rho_{HV}`–SNR curve and optimising the radar noise
    level, following Ryzhkov & Zrnić (2019).
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the radar reflectivity in dBZ, the raw correlation
        coefficient along with the polar coordinates (range, azimuth).
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{"rng": "range", "DBZ": "DBTH", "RHOHV": "URHOHV"}.``
    rhohv_theo : tuple of float, default (0.9, 1.0)
        Theoretical :math:`\rho_{HV}` range in rain
        ``(rhoHV_0, rhoHV_inf)`` used to build the model curve.
    noise_level : tuple of float, default (0, 100)
        Search interval (in dB) for optimising the radar noise level.
    bins_rho : tuple of float, default (0.8, 1.1, 0.005)
        Binning interval for :math:`\rho_{HV}` as ``(start, stop, step)``.
    bins_snr : tuple of float, default (5, 30, 0.1)
        Binning interval for SNR (in dB) as ``(start, stop, step)``.
    mode : {"linear", "exp"}, default "exp"
        Functional form of the theoretical :math:`\rho_{HV}`–SNR curve.
    exp_curvet : float, default 20.0
        Transition point (dB) for the exponential model.
    eps : float, default 0.005
        Small tolerance controlling the exponential decay.
    mask : bool, list of str, dict of str to str, or None, default True
        Controls which variables receive the noise correction.

        * ``None`` or ``False``: classification only; no correction applied.
        * ``True``: correct all relevant 2‑D variables (default: ``URHOHV``).
        * list of str: correct only the listed variables.
        * dict: map input variable names to explicit output names.
        Masking is applied after correction; variables not listed are untouched.
    replace_vars : bool, default False
        If True, overwrite the selected variables.
        If False, corrected variables receive a ``_QC`` suffix unless explicit
        names are provided via ``mask`` (dict form).
    plot_method : bool, default False
        If ``True``, produce diagnostic plots of the optimisation and
        histogram analysis.

    Returns
    -------
    xarray.Dataset
        Dataset containing the corrected :math:`\rho_{HV}` field and
        diagnostic attributes, including:
            - ``noise_level_dB`` – optimised noise level
            - ``objective_rmse`` – RMSE of the optimisation
            - ``rhohv_theo`` – theoretical bounds
            - ``mode``, ``exp_curvet``, ``eps`` – model parameters
    
    Notes
    -----
    * This function operates in native polar radar coordinates.
    * The method estimates the noise level that minimises the RMSE between the
      observed :math:`\rho_{HV}` distribution and a theoretical curve, then
      applies the correction:
        .. math::
                \rho_{HV(corr)} =
                \rho_{HV} \left( 1 + \frac{1}{\mathrm{SNR}_{\mathrm{lin}}} \right)
    * Units for range are inspected and converted to the appropriate units
      (km) when necessary.
    
    References
    ----------
    .. [1] Ryzhkov, A. V., & Zrnic, D. S. (2019). Radar polarimetry for
        weather observations. In Springer atmospheric sciences.
        https://doi.org/10.1007/978-3-030-05093-1
    """
    from ..io import modeltp as mdtp

    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # Resolve variable names
    defaults = {"rng": "range", "DBZ": "DBTH", "RHOHV": "URHOHV"}
    names = {**defaults, **(inp_names or {})}
    rng_km = convert(ds[names["rng"]], "km")
    DBZ = ds[names["DBZ"]]
    rhohv_na = ds[names["RHOHV"]]
    # Optimise noise level
    opt_noise, opt_rmse = _optimise_noise_level(
        DBZ, rng_km, rhohv_na, bins_rho, bins_snr, rhohv_theo, noise_level,
        mode=mode, exp_curvet=exp_curvet, eps=eps)
    # Apply correction
    snr_db = signal2noiseratio(DBZ, rng_km, opt_noise, scale="db").rename("snr_db")
    snr_lin = signal2noiseratio(DBZ, rng_km, opt_noise, scale="lin").rename("snr_lin")
    rhohv_corr = (rhohv_na * (1 + 1 / snr_lin)).rename("rhohv_corr")
    rhohv_final = rhohv_corr.rename(f"{names['RHOHV']}_corr")
    rhohv_final.attrs = sweep_vars_attrs_f.get('RHOHV', {})
    # Histogram diagnostics
    snr_edges = np.arange(*bins_snr)
    rho_edges = np.arange(*bins_rho)
    hist = xr_hist2d(snr_db, rhohv_final, snr_edges, rho_edges,
                     dim=list(snr_db.dims))
    # Extract maxima per SNR bin (bin centers)
    rhohv_bin_dim = "y_bin"
    idx = hist.argmax(dim=rhohv_bin_dim)

    rhohv_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    histmax = xr.apply_ufunc(lambda i: rhohv_centers[i], idx, vectorize=True,
                             dask="parallelized", output_dtypes=[float])
    # Define centers and theoretical line for plotting
    snr_centers = 0.5 * (snr_edges[:-1] + snr_edges[1:])
    # Theoretical line according to chosen mode
    theo_line = _build_theo_line(snr_centers, rhohv_theo,
                                 mode=mode, exp_curvet=exp_curvet, eps=eps)
    # Determine variables to correct
    # Default: only correct RHOHV
    default_var = names["RHOHV"]
    if mask is None or mask is False:
        # Classification only -> no correction applied
        return ds
    if mask is True:
        # Correct all relevant 2‑D variables (default: only RHOHV)
        vars_to_correct = [default_var]
        rename_map = {}
    elif isinstance(mask, (list, tuple, set)):
        vars_to_correct = list(mask)
        rename_map = {}
    elif isinstance(mask, dict):
        vars_to_correct = list(mask.keys())
        rename_map = mask
    else:
        raise TypeError("mask must be None, bool, list, or dict")
    # Validate variables
    missing = [v for v in vars_to_correct if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Variables not found in dataset: {missing}")
    # Prepare output dataset
    ds_out = ds.copy()
    corrected_vars = []
    # Apply correction to each selected variable
    for var in vars_to_correct:
        # Determine output name
        if isinstance(mask, dict):
            # Explicit renaming
            out_var = rename_map[var]
        else:
            if replace_vars:
                out_var = var
            else:
                out_var = f"{var}_QC"
        # Apply correction chain
        ds_out = apply_correction_chain(
            ds_out, varname=var, step="rhohv_nc",
            suffix="" if replace_vars or isinstance(mask, dict) else "_QC",
            corrected_field=rhohv_corr,
            params={"noise_level_dB": float(opt_noise)},
            module_provenance="towerpy.calib.calib_rhohv.rhohv_noisecorrection")
        # internal name created by apply_correction_chain
        internal_name = var if replace_vars else f"{var}_QC"
        # If explicit rename requested, rename internal_name -> out_var
        if isinstance(mask, dict):
            if internal_name in ds_out and internal_name != out_var:
                ds_out = ds_out.rename({internal_name: out_var})
        # Merge canonical attrs
        old_attrs = ds_out[out_var].attrs.copy()
        new_attrs = sweep_vars_attrs_f.get(out_var, {})
        merged = {**old_attrs, **new_attrs}
        ds_out = safe_assign_variable(ds_out, out_var, ds_out[out_var],
                                      new_attrs=merged)
        corrected_vars.append(out_var)
    # Provenance
    extra = {"step_description":
             ("Stabilises the correlation coefficient in regions of low "
              "reflectivity by correcting its dependence on the SNR.")}
    # extra = ("Stabilises the correlation coefficient in regions of low "
             # "reflectivity by correcting its dependence on the SNR.")
    params = {"noise_level_dB": float(opt_noise),
              "objective_rmse": float(opt_rmse),
              "rhohv_theo": rhohv_theo,
              "mode": mode,
              "exp_curvet": exp_curvet,
              "eps": eps,
              "noise_level_bounds": noise_level,
              "mask": mask,
              "replace_vars": replace_vars,
              "corrected_vars": corrected_vars
              }
    ds_out = record_provenance(
        ds_out, step="rhohv_noisecorrection",
        inputs = [names["DBZ"], names["rng"], names["RHOHV"]],
        outputs=corrected_vars, parameters=params, extra_attrs=extra,
        module_provenance="towerpy.calib.calib_rhohv.rhohv_noisecorrection")
    # Plotting
    if plot_method:
        _plot_rhohvmethod_single(snr_edges, rho_edges, hist, snr_db, rhohv_na,
                                 snr_centers, theo_line, histmax, opt_noise)
        _plot_rhohvmethod_grid(DBZ, rng_km, rhohv_na, bins_snr=bins_snr,
                               bins_rho=bins_rho, rhohv_theo=rhohv_theo,
                               opt_noise=opt_noise, mode=mode,
                               exp_curvet=exp_curvet, eps=eps)
    return ds_out
