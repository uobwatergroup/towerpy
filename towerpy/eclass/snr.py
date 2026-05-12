"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import warnings
import xarray as xr
import numpy as np
from ..io import modeltp as mdtp
from ..datavis import rad_display
from ..utils.radutilities import get_attrval, safe_assign_variable
from ..utils.radutilities import record_provenance, apply_correction_chain
from ..utils.unit_conversion import convert


warnings.filterwarnings("ignore", category=RuntimeWarning)


class SNR_Classif:
    r"""
    A class to compute the Signal-to-Noise Ratio on radar data.

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
        min_snr : float
            Reference noise value.
        snr_class : dict
            Results of the SNR method.
        vars : dict
            Radar variables with noise removed.
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

    def signalnoiseratio(self, rad_georef, rad_params, rad_vars, min_snr=0,
                         rad_cst=None, snr_linu=False, data2correct=None,
                         classid=None, plot_method=False):
        """
        Compute the SNR and discard data using a reference noise value.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used to compute the SNR.
        min_snr : float64, optional
            Reference noise value. The default is 0.
        data2correct : dict, optional
            Variables into which noise is removed. The default is None.
        plot_method : Bool, optional
            Plot the SNR classification method. The default is False.
        """
        self.echoesID = {'pcpn': 0,
                         'noise': 3}
        if classid is not None:
            self.echoesID.update(classid)
        if rad_cst:
            rc = rad_cst
        else:
            rc = rad_params['radar constant [dB]']
        rh, _ = np.meshgrid(rad_georef['range [m]']/1000,
                             rad_georef['azim [rad]'])
        snrc = rad_vars['ZH [dBZ]'] - 20*np.log10(rh) + rc
        idx = np.nonzero(snrc >= min_snr)
        snrclass = np.full(snrc.shape, np.nan)
        snrclass[idx] = 1
        snr = {'snrclass': snrclass, 'snr [dB]': snrc}
        if snr_linu is True:
            snrlu = 10 ** (0.1*snrc)
            snr['snr [linear]'] = snrlu
        if data2correct is not None:
            rdatsnr = data2correct.copy()
            for key in rdatsnr:
                rdatsnr[key] = rdatsnr[key]*snrclass
                self.vars = rdatsnr
        snr['snrclass'][np.isnan(snr['snrclass'])] = self.echoesID['noise']
        snr['snrclass'][snr['snrclass'] == 1] = self.echoesID['pcpn']
        if plot_method:
            rad_display.plot_snr(rad_georef, rad_params, snr, min_snr)
        self.min_snr = min_snr
        self.snr_class = snr

    @staticmethod
    def static_signalnoiseratio(rad_georef, rad_params, rad_vars,
                                min_snr=0, rad_cst=None, snr_linu=False,
                                data2correct=None, plot_method=False):
        """
        Compute the SNR (in dB) and discard data using a reference noise value.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used to compute the SNR..
        min_snr : float64, optional
            Reference noise value. The default is 0.
        data2correct : dict, optional
            Variables into which noise is removed. The default is None.
        plot_method : Bool, optional
            Plot the SNR classification method. The default is False.

        Returns
        -------
        Object containing the signal/noise classification.

        """
        if rad_cst:
            rc = rad_cst
        else:
            rc = rad_params['radar constant [dB]']
        rh, _ = np.meshgrid(rad_georef['range [m]']/1000,
                             rad_georef['azim [rad]'])
        snrc = rad_vars['ZH [dBZ]'] - 20*np.log10(rh) + rc
        idx = np.nonzero(snrc >= min_snr)
        snrclass = np.full(snrc.shape, np.nan)
        snrclass[idx] = 1
        snr = {'snrclass': snrclass, 'snr [dB]': snrc}
        if snr_linu is True:
            snrlu = 10 ** (0.1*snrc)
            snr['snr [linear]'] = snrlu
        if data2correct is not None:
            rdatsnr = data2correct.copy()
            for key in rdatsnr:
                rdatsnr[key] = rdatsnr[key]*snrclass
                return snr, rdatsnr
        return snr


# =============================================================================
# %% xarray implementation
# =============================================================================

def signal2noiseratio(Z, rng_km, rc, scale="db"):
    """
    Compute signal-to-noise ratio (SNR).

    Parameters
    ----------
    Z : array_like or xarray.DataArray
        Radar reflectivity in dBZ.
    rng_km : array_like or xarray.DataArray
        Range, in km.
    rc : float
        Radar constant, in dB.
    scale : {"db", "lin", "both"}, optional
        Output format:
        - "db"   : return SNR in dB (default)
        - "lin"  : return SNR in linear scale
        - "both" : return dict with both

    Returns
    -------
    snr : ndarray, DataArray, or dict
        Depending on `scale`:
        - "db"   -> snr_db
        - "lin"  -> snr_lin
        - "both" -> {"snr_db": snr_db, "snr_lin": snr_lin}
    """
    if scale == "db":
        return Z - 20.0 * np.log10(rng_km) + rc
    elif scale == "lin":
        snr_db = Z - 20.0 * np.log10(rng_km) + rc
        return 10.0 ** (snr_db / 10.0)
    elif scale == "both":
        snr_db = Z - 20.0 * np.log10(rng_km) + rc
        snr_lin = 10.0 ** (snr_db / 10.0)
        return {"snr_db": snr_db, "snr_lin": snr_lin}
    else: raise ValueError(f"Unknown scale '{scale}',"
                           " expected 'db', 'lin', or 'both'.")


def snr_classif(ds, inp_names=None, min_snr=0, rcst_dB=None, classid=None,
                snr_linu=False, mask=None, replace_vars=False):
    """
    Compute the signal-to-noise ratio (SNR) and classify gates as signal or
    noise using a reference noise value. Optionally apply SNR-based masking to
    selected variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the radar reflectivity in dBZ (e.g. "DBTH"),
        along with the polar coordinates (range, azimuth).
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset.
        Defaults: {'azi': 'azimuth', 'rng': 'range', 'DBZ': 'DBTH'}.
    min_snr : float, default 0
        Minimum signal‑to‑noise ratio threshold (in dB). Values below
        this threshold are classified as noise.
    rcst_dB : float, default None
        Radar constant (in dB). If None, the function attempts to retrieve
        the radar constant from metadata. If missing, default is 0.
    classid : dict, default None
        Override classification IDs for ``'pcpn'``, ``'noise'``. Defaults are
        ``{'pcpn': 0, 'noise': 3}``.
    snr_linu : bool, default False
        If ``True``, also compute linear SNR (``SNR``).    
    mask : bool, list of str, dict of str to str, or None, optional
        Controls which variables receive SNR-based masking.

        * ``None`` or ``False``: classification only; no masking applied.
        * ``True``: mask all 2‑D data variables in the dataset.
        * list of str: mask only the listed variables.
        * dict: map input variable names to explicit output names.
    replace_vars : bool, default False
        If True, overwrite selected variables.
        If False, masked variables receive a ``_QC`` suffix unless explicit
        names are provided via ``mask`` (dict form).

    Returns
    -------
    xarray.Dataset
        Dataset containing:
    
        - ``SNR_CLASS`` : classification field (signal / noise)
        - ``DBSNR``     : SNR in dB
        - ``SNR``       : linear SNR (if ``snr_linu=True``)
        - masked variables (if ``mask`` is True, a list, or a dict)
        - updated variable-level and dataset-level provenance
        
    Notes
    -----
    * Range coordinates are inspected and converted to kilometres when needed.
    """
    # Resolve variable names
    defaults = {"azi": "azimuth", "rng": "range",  "DBZ": "DBTH"}
    names = {**defaults, **(inp_names or {})}
    # Classification IDs
    echoesID = {'signal': 0, 'noise': 3}
    if classid is not None:
        echoesID.update(classid)
    # Resolve radar constant
    rc = (float(rcst_dB) if rcst_dB is not None
          else get_attrval("radconstH", ds, default=0))
    # SNR in dB
    rng_km = convert(ds[names["rng"]], "km").values
    DBZ = ds[names["DBZ"]]
    snr_dB = xr.apply_ufunc(
        signal2noiseratio, DBZ, rng_km, rc, kwargs={"scale": "db"},
        dask="parallelized", output_dtypes=[float], vectorize=True)
    # Classification array
    snrclass = xr.where(snr_dB >= min_snr, echoesID["signal"],
                        echoesID["noise"])
    # Write variables back
    dims = (names["azi"], names["rng"])
    coords = {names["azi"]: ds[names["azi"]], names["rng"]: ds[names["rng"]]}
    # Prepare output
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    ds_out = xr.Dataset(
        {"SNR_CLASS": xr.DataArray(
            snrclass, dims=dims, coords=coords,
            attrs=sweep_vars_attrs_f.get('SNR_CLASS', '')),
        "DBSNR": xr.DataArray(
            snr_dB, dims=dims, coords=coords,
            attrs=sweep_vars_attrs_f.get('DBSNR', ''))},
        coords=ds.coords, attrs=ds.attrs.copy())
    ds_out.SNR_CLASS.attrs.update({"units": f"flags [{len(echoesID)}]",
                                   "flags": echoesID})
    if snr_linu:
        ds_out["SNR"] = xr.DataArray(
            10.0 ** (0.1 * snr_dB), dims=dims, coords=coords,
            attrs=sweep_vars_attrs_f.get('SNRH', ''))
    outputs = list(ds_out.keys())
    # QC API: determine variables to mask
    if mask is None or mask is False:
        # classification-only mode: no masking, only SNR diagnostics
        return ds_out
    if mask is True:
        vars_to_correct = [v for v, da in ds.data_vars.items() if da.ndim == 2]
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
    ds_out2 = ds.copy()
    # Attach SNR outputs
    for key in ds_out:
        ds_out2 = safe_assign_variable(ds_out2, key, ds_out[key])
    corrected_vars = []
    for var in vars_to_correct:
        # Determine output name
        if isinstance(mask, dict):
            out_var = rename_map[var]
        else:
            out_var = var if replace_vars else f"{var}_QC"
        # Apply masking
        ds_out2 = apply_correction_chain(
            ds_out2, varname=var, step="snr_classif",
            mask=(snrclass == echoesID["noise"]),
            suffix="" if replace_vars or isinstance(mask, dict) else "_QC",
            params={"min_snr_threshold": float(min_snr)},
            module_provenance="towerpy.eclass.snr.snr_classif")
        # Rename if needed
        internal_name = var if replace_vars else f"{var}_QC"
        if isinstance(mask, dict) and internal_name != out_var:
            if internal_name in ds_out2:
                ds_out2 = ds_out2.rename({internal_name: out_var})
        # Merge canonical attrs
        old_attrs = ds_out2[out_var].attrs.copy()
        new_attrs = sweep_vars_attrs_f.get(out_var, {})
        merged = {**old_attrs, **new_attrs}
        ds_out2 = safe_assign_variable(ds_out2, out_var, ds_out2[out_var],
                                       new_attrs=merged)
        corrected_vars.append(out_var)
    outputs.extend(corrected_vars)
    # Provenance
    extra = {'step_description':
             ('Quantifies the level of desired signal relative to background ' 
              ' noise and removes data classified as noise.')}
    params = {"min_snr_threshold": float(min_snr),
              "reflectivity_var": names["DBZ"],
              "range_var": names["rng"],
              "azimuth_var": names["azi"],
              "radar_constant_value_dB": rc,
              "snr_linear": bool(snr_linu),
              "class_ids": echoesID,
              "mask": mask,
              "replace_vars": replace_vars,
              "corrected_vars": corrected_vars}
    ds_out2 = record_provenance(
        ds_out2, step="snr_computation_classif",
        inputs = [names["DBZ"], names["rng"], names["azi"]], outputs=outputs,
        parameters=params, extra_attrs=extra,
        module_provenance="towerpy.eclass.snr.snr_classif")
    return ds_out2
