"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import copy
import warnings

import numpy as np
import xarray as xr

from ..datavis import rad_display
from ..utils.radutilities import find_nearest, find_nearest_index, record_provenance


class ZDR_Calibration:
    r"""
    A class to calibrate the radar differential reflectivity.

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
    zdr_offset : dict
        Computed :math:`Z_{DR}` offset
    zdr_offset_stats : dict
        Stats calculated during the computation of the :math:`Z_{DR}` offset.
    vars : dict
        Offset-corrected :math:`(Z_{DR})` and user-defined radar variables.
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

    def offsetdetection_vps(self, pol_profs, mlyr=None, min_h=1.1, zhmin=5,
                            zhmax=30, rhvmin=0.98, minbins=2, stats=False,
                            plot_method=False, rad_georef=None, rad_vars=None):
        r"""
        Calculate the offset on :math:`Z_{DR}` using vertical profiles.

        Parameters
        ----------
        pol_profs : dict
            Profiles of polarimetric variables.
        mlyr : class
            Melting layer class containing the top and bottom boundaries of
            the ML. Only gates below the melting layer bottom (i.e. the rain
            region below the melting layer) are included in the method.
            If None, the default values of the melting level and the thickness
            of the melting layer are set to 5 and 0.5, respectively.
        min_h : float, optional
            Minimum height of usable data within the polarimetric profiles.
            The default is 1.1.
        zhmin : float, optional
            Threshold on :math:`Z_{H}` (in dBZ) related to light rain.
            The default is 5.
        zhmax : float, optional
            Threshold on :math:`Z_{H}` (in dBZ) related to light rain.
            The default is 30.
        rhvmin : float, optional
            Threshold on :math:`\rho_{HV}` (unitless) related to light rain.
            The default is 0.98.
        minbins : float, optional
            Consecutive bins of :math:`Z_{DR}` related to light rain.
            The default is 2.
        stats : dict, optional
            If True, the function returns stats related to the computation of
            the :math:`Z_{DR}` offset. The default is False.
        plot_method : Bool, optional
            Plot the offset detection method. The default is False.
        rad_georef : dict, optional
            Used only to depict the methodolgy. Georeferenced data containing
            descriptors of the azimuth, gate and beam height, amongst others.
            The default is None.
        rad_vars : dict, optional
            Used only to depict the methodolgy. Radar variables used for
            plotting the offset correction method. The default is None.

        Notes
        -----
        1. Based on the method described in [1]_ and [2]_

        References
        ----------
        .. [1] Gorgucci, E., Scarchilli, G., and Chandrasekar, V. (1999),
            A procedure to calibrate multiparameter weather radar using
            properties of the rain medium, IEEE T. Geosci. Remote, 37, 269–276,
            https://doi.org/10.1109/36.739161
        .. [2] Sanchez-Rivas, D. and Rico-Ramirez, M. A. (2022): "Calibration
            of radar differential reflectivity using quasi-vertical profiles",
            Atmos. Meas. Tech., 15, 503–520,
            https://doi.org/10.5194/amt-15-503-2022
        """
        if mlyr is None:
            mlvl = 5
            mlyr_thickness = 0.5
            mlyr_bottom = mlvl - mlyr_thickness
        else:
            mlvl = mlyr.ml_top
            mlyr_thickness = mlyr.ml_thickness
            mlyr_bottom = mlyr.ml_bottom
        if np.isnan(mlyr_bottom):
            boundaries_idx = [find_nearest(
                pol_profs.georef['profiles_height [km]'], min_h),
                find_nearest(pol_profs.georef['profiles_height [km]'],
                             mlvl-mlyr_thickness)]
        else:
            boundaries_idx = [find_nearest(
                pol_profs.georef['profiles_height [km]'], min_h),
                find_nearest(pol_profs.georef['profiles_height [km]'],
                             mlyr_bottom)]
        if boundaries_idx[1] <= boundaries_idx[0]:
            boundaries_idx = [np.nan]
        if np.isnan(mlvl) and np.isnan(mlyr_bottom):
            boundaries_idx = [np.nan]

        if any(np.isnan(boundaries_idx)):
            self.zdr_offset = 0
        else:
            profs = copy.deepcopy(pol_profs.vps)
            calzdr_vps = {k: v[boundaries_idx[0]:boundaries_idx[1]]
                          for k, v in profs.items()}

            calzdr_vps['ZDR [dB]'][calzdr_vps['ZH [dBZ]'] < zhmin] = np.nan
            calzdr_vps['ZDR [dB]'][calzdr_vps['ZH [dBZ]'] > zhmax] = np.nan
            calzdr_vps['ZDR [dB]'][calzdr_vps['rhoHV [-]'] < rhvmin] = np.nan
            if np.count_nonzero(~np.isnan(calzdr_vps['ZDR [dB]'])) <= minbins:
                calzdr_vps['ZDR [dB]'] *= np.nan

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                calzdrvps_mean = np.nanmean(calzdr_vps['ZDR [dB]'])
                calzdrvps_max = np.nanmax(calzdr_vps['ZDR [dB]'])
                calzdrvps_min = np.nanmin(calzdr_vps['ZDR [dB]'])
                calzdrvps_std = np.nanstd(calzdr_vps['ZDR [dB]'])
                calzdrvps_sem = (np.nanstd(
                    calzdr_vps['ZDR [dB]'])
                    / np.sqrt(len(calzdr_vps['ZDR [dB]'])))

            if not np.isnan(calzdrvps_mean):
                self.zdr_offset = calzdrvps_mean
            else:
                self.zdr_offset = 0
            if stats:
                self.zdr_offset_stats = {'offset_max': calzdrvps_max,
                                         'offset_min': calzdrvps_min,
                                         'offset_std': calzdrvps_std,
                                         'offset_sem': calzdrvps_sem,
                                         }
            if plot_method:
                rad_params = {}
                if self.elev_angle:
                    rad_params['elev_ang [deg]'] = self.elev_angle
                else:
                    rad_params['elev_ang [deg]'] = 'surveillance scan'
                if self.scandatetime:
                    rad_params['datetime'] = self.scandatetime
                else:
                    rad_params['datetime'] = None
                var = 'ZDR [dB]'
                rad_var = np.array([i[boundaries_idx[0]:boundaries_idx[1]]
                                    for i in rad_vars[var]], dtype=np.float64)
                rad_display.plot_offsetcorrection(
                    rad_georef, rad_params, rad_var,
                    var_offset=self.zdr_offset, var_name=var)

    def offsetdetection_qvps(self, pol_profs, mlyr=None, min_h=0., max_h=3.,
                             zhmin=0, zhmax=20, rhvmin=0.985, minbins=4,
                             zdr_0=0.182, stats=False):
        r"""
        Calculate the offset on :math:`Z_{DR}` using QVPs, acoording to [1]_.

        Parameters
        ----------
        pol_profs : dict
            Profiles of polarimetric variables.
        mlyr : class
            Melting layer class containing the top and bottom boundaries of
            the ML.
        min_h : float, optional
            Minimum height of usable data within the polarimetric profiles.
            The default is 0.
        max_h : float, optional
            Maximum height of usable data within the polarimetric profiles.
            The default is 3.
        zhmin : float, optional
            Threshold on :math:`Z_{H}` (in dBZ) related to light rain.
            The default is 0.
        zhmax : float, optional
            Threshold on :math:`Z_{H}` (in dBZ) related to light rain.
            The default is 20.
        rhvmin : float, optional
            Threshold on :math:`\rho_{HV}` (unitless) related to light rain.
            The default is 0.985.
        minbins : float, optional
            Consecutive bins of :math:`Z_{DR}` related to light rain.
            The default is 3.
        zdr_0 : float, optional
            Intrinsic value of :math:`Z_{DR}` in light rain at ground level.
            Defaults to 0.182.
        stats : dict, optional
            If True, the function returns stats related to the computation of
            the :math:`Z_{DR}` offset. The default is False.

        Notes
        -----
        1. Based on the method described in [1]

        References
        ----------
        .. [1] Sanchez-Rivas, D. and Rico-Ramirez, M. A. (2022): "Calibration
            of radar differential reflectivity using quasi-vertical profiles",
            Atmos. Meas. Tech., 15, 503–520,
            https://doi.org/10.5194/amt-15-503-2022
        """
        if mlyr is None:
            mlvl = 5
            mlyr_thickness = 0.5
            mlyr_bottom = mlvl - mlyr_thickness
        else:
            mlvl = mlyr.ml_top
            mlyr_thickness = mlyr.ml_thickness
            mlyr_bottom = mlyr.ml_bottom
        if np.isnan(mlyr_bottom):
            boundaries_idx = [find_nearest(pol_profs.georef['profiles_height [km]'], min_h),
                              find_nearest(pol_profs.georef['profiles_height [km]'],
                                           mlvl-mlyr_thickness)]
        else:
            boundaries_idx = [find_nearest(pol_profs.georef['profiles_height [km]'], min_h),
                              find_nearest(pol_profs.georef['profiles_height [km]'],
                                           mlyr_bottom)]
        if boundaries_idx[1] <= boundaries_idx[0]:
            boundaries_idx = [np.nan]
        if np.isnan(mlvl) and np.isnan(mlyr_bottom):
            boundaries_idx = [np.nan]

        maxheight = find_nearest(pol_profs.georef['profiles_height [km]'],
                                 max_h)

        if any(np.isnan(boundaries_idx)):
            self.zdr_offset = 0
        else:
            profs = copy.deepcopy(pol_profs.qvps)
            calzdr_qvps = {k: v[boundaries_idx[0]:boundaries_idx[1]]
                           for k, v in profs.items()}

            calzdr_qvps['ZDR [dB]'][calzdr_qvps['ZH [dBZ]'] < zhmin] = np.nan
            calzdr_qvps['ZDR [dB]'][calzdr_qvps['ZH [dBZ]'] > zhmax] = np.nan
            calzdr_qvps['ZDR [dB]'][calzdr_qvps['rhoHV [-]'] < rhvmin] = np.nan
            if np.count_nonzero(~np.isnan(calzdr_qvps['ZDR [dB]'])) <= minbins:
                calzdr_qvps['ZDR [dB]'] *= np.nan
            calzdr_qvps['ZDR [dB]'][calzdr_qvps['rhoHV [-]']>maxheight]=np.nan

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                calzdrqvps_mean = np.nanmean(calzdr_qvps['ZDR [dB]'])
                calzdrqvps_max = np.nanmax(calzdr_qvps['ZDR [dB]'])
                calzdrqvps_min = np.nanmin(calzdr_qvps['ZDR [dB]'])
                calzdrqvps_std = np.nanstd(calzdr_qvps['ZDR [dB]'])
                calzdrqvps_sem = np.nanstd(calzdr_qvps['ZDR [dB]'])/np.sqrt(len(calzdr_qvps['ZDR [dB]']))

            if not np.isnan(calzdrqvps_mean):
                self.zdr_offset = calzdrqvps_mean - zdr_0
            else:
                self.zdr_offset = 0

            if stats:
                self.zdr_offset_stats = {'offset_max': calzdrqvps_max,
                                         'offset_min': calzdrqvps_min,
                                         'offset_std': calzdrqvps_std,
                                         'offset_sem': calzdrqvps_sem,
                                         }

    def offset_correction(self, zdr2calib, zdr_offset=0, data2correct=None):
        """
        Correct the ZDR offset using a given value.

        Parameters
        ----------
        zdr2calib : array of float
            Offset-affected differential reflectiviy :math:`Z_{DR}` in dB.
        zdr_offset : float
            Differential reflectivity offset in dB. The default is 0.
        data2correct : dict, optional
            Dictionary to update the offset-corrected :math:`Z_{DR}`.
            The default is None.
        """
        if np.isnan(zdr_offset):
            zdr_offset = 0
        zdr_oc = copy.deepcopy(zdr2calib) - zdr_offset
        if data2correct is None:
            self.vars = {'ZDR [dB]': zdr_oc}
        else:
            data2cc = copy.deepcopy(data2correct)
            data2cc.update({'ZDR [dB]': zdr_oc})
            self.vars = data2cc

# =============================================================================
# %% xarray implementation
# =============================================================================

def _empty_stats():
    return xr.Dataset({"offset_max": ((), np.nan), "offset_min": ((), np.nan),
                       "offset_std": ((), np.nan), "offset_sem": ((), np.nan)})


def zdr_offsetdetection_vp(ds, mlyr=None, inp_names=None, min_h=1.1, minbins=2,
                           zhmin=5.0, zhmax=30.0, rhvmin=0.98,
                           return_stats=False):
    r"""
    Compute the ZDR calibration offset using vertical profiles (VPS)
    following Gorgucci et al. (1999) and Sanchez-Rivas & Rico-Ramirez (2022).

    Parameters
    ----------
    ds : xarray.Dataset
        VPS dataset containing ZDR, ZH, RHOHV, and a height coordinate.
    mlyr : xarray.Dataset or None, optional
        Dataset containing the melting-layer boundaries height.
        ``[MLYRTOP, MLYRBTM, MLYRTHK]``. Only gates below the melting layer
        bottom (i.e. the rain region below the melting layer) are included in
        the computation.
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{"ZDR": "ZDR", "ZH": "DBZH", "RHOHV": "RHOHV", "height": "height"}``
    min_h : float, default 1.1
        Minimum height (km) above ground to include in the VPS analysis.
    minbins : int, default 2
        Minimum number of valid bins required to compute the offset.
    zhmin : float, default 5.0
        Minimum ZH threshold (dBZ) for selecting light-rain gates.
    zhmax : float, default 30.0
        Maximum ZH threshold (dBZ) for selecting light-rain gates.
    rhvmin : float, default 0.98
        Minimum RHOHV threshold for selecting light-rain gates.
    return_stats : bool, default False
        If True, return (offset, stats_dataset).

    Returns
    -------
    offset : xr.Dataset
        Scalar ZDR offset, in dB.
    stats : xr.Dataset, optional
        Dataset with offset_max, offset_min, offset_std, offset_sem.
    
    Notes
    -----
    * The method estimates the differential reflectivity offset following [2]_:
          .. math:: Z^{O_{VP}}_{DR} = \frac{1}{n} \sum_{i=1}^{n} Z_{{DR}_{i}}
    * Only gates between `min_h` and the melting-layer bottom are used.
    * Only light-rain gates are used, using threshold in RHOHV and ZH, 
      according to [1]_
    
    References
    ----------
    .. [1] Gorgucci, E., Scarchilli, G., & Chandrasekar, V. (1999). A procedure
        to calibrate multiparameter weather radar using properties of the rain
        medium. IEEE Transactions on Geoscience and Remote Sensing, 37(1),
        269–276. https://doi.org/10.1109/36.739161
    .. [2] Sanchez-Rivas, D., & Rico-Ramirez, M. A. (2022). Calibration of
        radar differential reflectivity using quasi-vertical profiles.
        Atmospheric Measurement Techniques, 15(2), 503–520.
        https://doi.org/10.5194/amt-15-503-2022
    """
    # 1. Canonical variable mapping
    defaults = {"ZDR": "ZDR", "ZH": "DBZH", "RHOHV": "RHOHV",
                "height": "height"}
    names = {**defaults, **(inp_names or {})}
    height = ds[names["height"]]
    # 2. Determine melting-layer geometry
    # if detect_mlyr:
    #     mlyr_kwargs = mlyr_kwargs or {}
    #     mlyr = detect_mlyr_from_profiles(ds, **mlyr_kwargs)
    if mlyr is None:
        ml_top = 5.0
        ml_thk = 0.75
        ml_bottom = ml_top - ml_thk
    else:
        ml_top = float(mlyr["MLYRTOP"])
        ml_bottom = float(mlyr["MLYRBTM"])
        ml_thk = float(mlyr["MLYRTHK"])
    # 3. Height slicing using find_nearest_index
    hvals = height.values
    # Invalid MLyr → offset = 0
    if np.isnan(ml_bottom) or np.isnan(ml_top):
        offset = xr.DataArray(0.0, name="ZDR_OFFSET")
        stats = _empty_stats() if return_stats else None
    else:
        i0 = find_nearest_index(hvals, min_h)
        i1 = find_nearest_index(hvals, ml_bottom)

        if i1 <= i0:
            offset = xr.DataArray(0.0, name="ZDR_OFFSET")
            stats = _empty_stats() if return_stats else None
        else:
            ds_sel = ds.isel({names["height"]: slice(i0, i1)})
            zdr_sel = ds_sel[names["ZDR"]]
            zh_sel = ds_sel[names["ZH"]]
            rho_sel = ds_sel[names["RHOHV"]]
            # 4. Apply light-rain filtering
            mask = ((zh_sel >= zhmin) & (zh_sel <= zhmax) & (rho_sel >= rhvmin))
            zdr_filt = zdr_sel.where(mask)
            # 5. minbins
            valid_bins = zdr_filt.count(dim=names["height"])
            if valid_bins <= minbins:
                offset = xr.DataArray(0.0, name="ZDR_OFFSET")
                stats = _empty_stats() if return_stats else None
            else:
                # 6. Compute offset
                offset = zdr_filt.mean(dim=names["height"], skipna=True)
                offset = offset.rename("ZDR_OFFSET")
                # 7. Compute stats
                if return_stats:
                    stats = xr.Dataset(
                        {"offset_max": zdr_filt.max(dim=names["height"],
                                                    skipna=True),
                         "offset_min": zdr_filt.min(dim=names["height"],
                                                    skipna=True),
                         "offset_std": zdr_filt.std(dim=names["height"],
                                                    skipna=True),
                         "offset_sem": 
                             (zdr_filt.std(dim=names["height"], skipna=True)
                              / np.sqrt(valid_bins)),})
                else:
                    stats = None
    # 8. Build output dataset
    coords = {name: coord for name, coord in ds.coords.items()
              if coord.dims == ()}  # keep scalar coords only
    data_vars = {"ZDR_OFFSET": offset}
    if return_stats and stats is not None:
        for k, v in stats.data_vars.items():
            data_vars[k] = v
    ds_out = xr.Dataset(data_vars, coords=coords)
    # 9. Record dataset-level provenance
    #TODO: add step_description
    extra = {'step_description': ('')}
    params = {"min_h": min_h, "zhmin": zhmin, "zhmax": zhmax, "rhvmin": rhvmin,
              "minbins": minbins, "ml_top": ml_top, "ml_bottom": ml_bottom,
              "ml_thickness": ml_thk}
    outputs = 'ZDR_OFFSET'
    ds_out = record_provenance(
        ds_out, step="zdr_offsetdetection_vp",
        inputs=[names["ZDR"], names["ZH"], names["RHOHV"]], outputs=outputs,
        parameters=params, extra_attrs=extra,
        module_provenance="towerpy.calib.calib_zdr.zdr_offsetdetection_vp")
    return ds_out


def zdr_offsetdetection_qvp(ds, mlyr=None, inp_names=None, min_h=0., max_h=3.,
                            zhmin=0., zhmax=20., rhvmin=0.985, minbins=4,
                            zdr_0=0.182, return_stats=False):
    r"""
    Compute the ZDR calibration offset using quasi‑vertical profiles (QVPs),
    following Sanchez‑Rivas & Rico‑Ramirez (2022).

    Parameters
    ----------
    ds : xarray.Dataset
        QVP dataset containing ZDR, ZH, RHOHV, and a height coordinate.
    mlyr : xarray.Dataset or None, optional
        Dataset containing the melting-layer boundaries height.
        ``[MLYRTOP, MLYRBTM, MLYRTHK]``. Only gates below the melting layer
        bottom (i.e. the rain region below the melting layer) are included in
        the computation.
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{"ZDR": "ZDR", "ZH": "DBZH", "RHOHV": "RHOHV", "height": "height"}``.
    min_h : float, default 0.
        Minimum height (km) to include in the QVP analysis.
    max_h : float, default 3.
        Maximum height (km) to include in the QVP analysis.
    zhmin : float, default 0.
        Minimum :math:`Z_H` threshold (dBZ) for selecting light‑rain gates.
    zhmax : float, default 20.
        Maximum :math:`Z_H` threshold (dBZ) for selecting light‑rain gates.
    rhvmin : float, default 0.985
        Minimum :math:`\rho_{HV}` threshold for selecting light‑rain gates.
    minbins : int, default 4
        Minimum number of valid :math:`Z_{DR}` bins required to compute
        the offset.
    zdr_0 : float, default 0.182
        Intrinsic value of :math:`Z_{DR}` in light rain at ground level.
    return_stats : bool, default False
        If ``True``, return both the offset and a dataset of summary statistics.

    Returns
    -------
    offset : xarray.Dataset
        Scalar ZDR calibration offset, in dB:
    stats : xarray.Dataset, optional
        Dataset containing ``offset_max``, ``offset_min``, ``offset_std``,
        and ``offset_sem``. Returned only if ``return_stats=True``.

    Notes
    -----
    * The method estimates the intrinsic differential reflectivity offset by
      computing the difference between the mean observed Z_DR and the expected
      intrinsic value in light rain, as follws:
          .. math:: Z^{O_{QVP}}_{DR} = \left(\frac{1}{n} \sum_{i=1}^{n} Z_{{DR}_i} \right) - Z_{DR}^{gl}
    * Only gates between ``min_h`` and the melting‑layer bottom are used.
    * Only light‑rain gates are selected, using thresholds in ZH and RHOHV.
    * The method follows the QVP‑based calibration approach described in [1]_.

    References
    ----------
    .. [1] Sanchez-Rivas, D., & Rico-Ramirez, M. A. (2022). Calibration of
        radar differential reflectivity using quasi-vertical profiles.
        Atmospheric Measurement Techniques, 15(2), 503–520.
        https://doi.org/10.5194/amt-15-503-2022
    """
    # 1. Canonical variable mapping
    defaults = {"ZDR": "ZDR", "ZH": "DBZH", "RHOHV": "RHOHV",
                "height": "height"}
    names = {**defaults, **(inp_names or {})}
    height = ds[names["height"]]
    # 2. Determine melting-layer geometry
    # if detect_mlyr:
    #     mlyr_kwargs = mlyr_kwargs or {}
    #     mlyr = detect_mlyr_from_profiles(ds, **mlyr_kwargs)
    if mlyr is None:
        ml_top = 5.0
        ml_thk = 0.75
        ml_bottom = ml_top - ml_thk
    else:
        ml_top = float(mlyr["MLYRTOP"])
        ml_bottom = float(mlyr["MLYRBTM"])
        ml_thk = float(mlyr["MLYRTHK"])
    # 3. Height slicing using find_nearest_index
    hvals = height.values
    # Invalid MLyr → offset = 0
    if np.isnan(ml_bottom) or np.isnan(ml_top):
        offset = xr.DataArray(0.0, name="ZDR_OFFSET")
        stats = _empty_stats() if return_stats else None
    else:
        i0 = find_nearest_index(hvals, min_h)
        i1 = find_nearest_index(hvals, ml_bottom)
        if i1 <= i0:
            offset = xr.DataArray(0.0, name="ZDR_OFFSET")
            stats = _empty_stats() if return_stats else None
        else:
            ds_sel = ds.isel({names["height"]: slice(i0, i1)})
            zdr_sel = ds_sel[names["ZDR"]]
            zh_sel = ds_sel[names["ZH"]]
            rho_sel = ds_sel[names["RHOHV"]]
            h_sel = ds_sel[names["height"]]
            # 4. Apply light-rain filtering
            mask = ((zh_sel >= zhmin) & (zh_sel <= zhmax) & (rho_sel >= rhvmin)
                    & (h_sel <= max_h))
            zdr_filt = zdr_sel.where(mask)
            # 5. minbins
            valid_bins = zdr_filt.count(dim=names["height"])
            if valid_bins <= minbins:
                offset = xr.DataArray(0.0, name="ZDR_OFFSET")
                stats = _empty_stats() if return_stats else None
            else:
                # 6. Compute offset
                mean_zdr = zdr_filt.mean(dim=names["height"], skipna=True)
                offset = (mean_zdr - zdr_0).rename("ZDR_OFFSET")
                # 7. Compute stats
                if return_stats:
                    stats = xr.Dataset(
                        {"offset_max": zdr_filt.max(dim=names["height"],
                                                    skipna=True),
                         "offset_min": zdr_filt.min(dim=names["height"],
                                                    skipna=True),
                         "offset_std": zdr_filt.std(dim=names["height"],
                                                    skipna=True),
                         "offset_sem":
                             (zdr_filt.std(dim=names["height"], skipna=True)
                              / np.sqrt(valid_bins))})
                else:
                    stats = None
    # 8. Build output dataset
    coords = {name: coord for name, coord in ds.coords.items()
              if coord.dims == ()}  # keep scalar coords only
    data_vars = {"ZDR_OFFSET": offset}
    if return_stats and stats is not None:
        for k, v in stats.data_vars.items():
            data_vars[k] = v
    ds_out = xr.Dataset(data_vars, coords=coords)
    # 9. Record dataset-level provenance
    #TODO: add step_description
    extra = {'step_description': ('')}
    params = {"min_h": min_h, "max_h": max_h, "zhmin": zhmin, "zhmax": zhmax,
              "rhvmin": rhvmin, "minbins": minbins, "zdr_0": zdr_0,
              "ml_top": ml_top, "ml_bottom": ml_bottom, "ml_thickness": ml_thk}
    outputs = 'ZDR_OFFSET'
    ds_out = record_provenance(
        ds_out, step="zdr_offsetdetection_qvp",
        inputs=[names["ZDR"], names["ZH"], names["RHOHV"]], outputs=outputs,
        parameters=params, extra_attrs=extra,
        module_provenance="towerpy.calib.calib_zdr.zdr_offsetdetection_qvp")
    return ds_out