"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import copy
import warnings

import numpy as np
import xarray as xr

from ..datavis import rad_display
from ..io import modeltp as mdtp
from ..ml.mlyr import (_normalise_ml_input, attach_melting_layer,
                       mlyr_ppidelimitation)
from ..utils.radutilities import (add_correction_step, despike_isolated,
                                  fill_both, find_nearest, find_nearest_index,
                                  record_provenance, rolling_std_xr,
                                  rolling_window, safe_assign_variable,
                                  std_mask_isolated, std_mask_threshold)


class PhiDP_Calibration:
    r"""
    A class for calibrating the differential phase :math:`(\Phi_{DP})`.

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
    phidp_offset : dict
        Computed :math:`\Phi_{DP}` offset
    phidp_offset_stats : dict
        Stats calculated during the computation of the :math:`\Phi_{DP}`
        offset.
    vars : dict
        Offset-corrected :math:`(\Phi_{DP})` and user-defined radar variables.
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

    def offsetdetection_vps(self, pol_profs, mlyr=None, min_h=1.1, max_h=None,
                            zhmin=5, zhmax=30, rhvmin=0.98, minbins=2,
                            stats=False, plot_method=False, rad_georef=None,
                            rad_vars=None):
        r"""
        Calculate the offset on :math:`\Phi_{DP}` using vertical profiles.

        Parameters
        ----------
        pol_profs : dict
            Profiles of polarimetric variables.
        mlyr : class, optional
            Melting layer class containing the top and bottom boundaries of
            the ML.
        min_h : float, optional
            Minimum height of usable data within the polarimetric profiles.
            The default is 1.1.
        max_h : float, optional
            Maximum height of usable data within the polarimetric profiles.
            Use only if ML boundaries are not available.
            The default is 3.
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
            Consecutive bins of :math:`\Phi_{DP}` related to light rain.
            The default is 2.
        stats : dict, optional
            If True, the function returns stats related to the computation of
            the :math:`\Phi_{DP}` offset. The default is False.
        plot_method : Bool, optional
            Plot the offset detection method. The default is False.
        rad_georef : dict, optional
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others. The default is None.
        rad_vars : dict, optional
            Radar variables used for plotting the offset correction method.
            The default is None.
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
            boundaries_idx = [
                find_nearest(pol_profs.georef['profiles_height [km]'], min_h),
                find_nearest(pol_profs.georef['profiles_height [km]'],
                             mlvl-mlyr_thickness)]
        else:
            boundaries_idx = [
                find_nearest(pol_profs.georef['profiles_height [km]'], min_h),
                find_nearest(pol_profs.georef['profiles_height [km]'],
                             mlyr_bottom)]
        if boundaries_idx[1] <= boundaries_idx[0]:
            boundaries_idx = [np.nan]
            # boundaries_idx *= np.nan
        if np.isnan(mlvl) and np.isnan(mlyr_bottom):
            boundaries_idx = [np.nan]
        if max_h:
            maxheight = find_nearest(
                pol_profs.georef['profiles_height [km]'], max_h)

        if any(np.isnan(boundaries_idx)):
            self.phidp_offset = 0
        else:
            profs = copy.deepcopy(pol_profs.vps)
            calphidp_vps = {k: v[boundaries_idx[0]:boundaries_idx[1]]
                            for k, v in profs.items()}

            calphidp_vps['PhiDP [deg]'][calphidp_vps['ZH [dBZ]'] < zhmin] = np.nan
            calphidp_vps['PhiDP [deg]'][calphidp_vps['ZH [dBZ]'] > zhmax] = np.nan
            calphidp_vps['PhiDP [deg]'][calphidp_vps['rhoHV [-]'] < rhvmin] = np.nan
            if np.count_nonzero(~np.isnan(calphidp_vps['PhiDP [deg]'])) <= minbins:
                calphidp_vps['PhiDP [deg]'] *= np.nan

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                calphidpvps_mean = np.nanmean(calphidp_vps['PhiDP [deg]'])
                calphidpvps_max = np.nanmax(calphidp_vps['PhiDP [deg]'])
                calphidpvps_min = np.nanmin(calphidp_vps['PhiDP [deg]'])
                calphidpvps_std = np.nanstd(calphidp_vps['PhiDP [deg]'])
                calphidpvps_sem = np.nanstd(
                    calphidp_vps['PhiDP [deg]']) / np.sqrt(
                        len(calphidp_vps['PhiDP [deg]']))

            if not np.isnan(calphidpvps_mean):
                self.phidp_offset = calphidpvps_mean
            else:
                self.phidp_offset = 0
            if stats:
                self.phidp_offset_stats = {'offset_max': calphidpvps_max,
                                           'offset_min': calphidpvps_min,
                                           'offset_std': calphidpvps_std,
                                           'offset_sem': calphidpvps_sem,
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
                var = 'PhiDP [deg]'
                rad_var = np.array([i[boundaries_idx[0]:boundaries_idx[1]]
                                    for i in rad_vars[var]], dtype=np.float64)
                rad_display.plot_offsetcorrection(
                    rad_georef, rad_params, rad_var,
                    var_offset=self.phidp_offset, var_name=var)

    def offsetdetection_qvps(self, pol_profs, mlyr=None, min_h=0., max_h=3.,
                             zhmin=0, zhmax=20, rhvmin=0.985, minbins=4,
                             stats=False):
        r"""
        Calculate the offset on :math:`\Phi_{DP}` using QVPs.

        Parameters
        ----------
        pol_profs : dict
            Profiles of polarimetric variables.
        mlyr : class
            Melting layer class containing the top and bottom
            boundaries of the ML.
        min_h : float, optional
            Minimum height of usable data within the polarimetric
            profiles. The default is 0.
        max_h : float, optional
            Maximum height of usable data within the polarimetric
            profiles. The default is 3.
        zhmin : float, optional
            Threshold on :math:`Z_{H}` (in dBZ) related to light rain.
            The default is 0.
        zhmax : float, optional
            Threshold on :math:`Z_{H}` (in dBZ) related to light rain.
            The default is 20.
        rhvmin : float, optional
            Threshold on :math:`\rho_{HV}` (unitless) related to
            light rain. The default is 0.985.
        minbins : float, optional
            Consecutive bins of :math:`\Phi_{DP}` related to light
            rain. The default is 3.
        stats : dict, optional
            If True, the function returns stats related to the
            computation of the :math:`\Phi_{DP}` offset.
            The default is False.

        Notes
        -----
        1. Adapted from the method described in [1]

        References
        ----------
        .. [1] Sanchez-Rivas, D. and Rico-Ramirez, M. A. (2022):
            "Calibration of radar differential reflectivity using
            quasi-vertical profiles", Atmos. Meas. Tech., 15, 503–520,
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

        maxheight = find_nearest(pol_profs.georef['profiles_height [km]'],
                                 max_h)

        if any(np.isnan(boundaries_idx)):
            self.phidp_offset = 0
        else:
            profs = copy.deepcopy(pol_profs.qvps)
            calpdp_qvps = {k: v[boundaries_idx[0]:boundaries_idx[1]]
                           for k, v in profs.items()}

            calpdp_qvps['PhiDP [deg]'][calpdp_qvps['ZH [dBZ]']
                                       < zhmin] = np.nan
            calpdp_qvps['PhiDP [deg]'][calpdp_qvps['ZH [dBZ]']
                                       > zhmax] = np.nan
            calpdp_qvps['PhiDP [deg]'][calpdp_qvps['rhoHV [-]']
                                       < rhvmin] = np.nan
            if np.count_nonzero(
                    ~np.isnan(calpdp_qvps['PhiDP [deg]'])) <= minbins:
                calpdp_qvps['PhiDP [deg]'] *= np.nan
            calpdp_qvps['PhiDP [deg]'][calpdp_qvps['rhoHV [-]']
                                       > maxheight] = np.nan

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                calpdpqvps_mean = np.nanmean(calpdp_qvps['PhiDP [deg]'])
                calpdpqvps_max = np.nanmax(calpdp_qvps['PhiDP [deg]'])
                calpdpqvps_min = np.nanmin(calpdp_qvps['PhiDP [deg]'])
                calpdpqvps_std = np.nanstd(calpdp_qvps['PhiDP [deg]'])
                calpdpqvps_sem = (np.nanstd(
                    calpdp_qvps['PhiDP [deg]'])
                    / np.sqrt(len(calpdp_qvps['PhiDP [deg]'])))

            if not np.isnan(calpdpqvps_mean):
                self.phidp_offset = calpdpqvps_mean
            else:
                self.phidp_offset = 0

            if stats:
                self.phidp_offset_stats = {'offset_max': calpdpqvps_max,
                                           'offset_min': calpdpqvps_min,
                                           'offset_std': calpdpqvps_std,
                                           'offset_sem': calpdpqvps_sem}

    def offsetdetection_ppi(self, rad_vars, mov_avrgf_len=(1, 3), thr_spdp=10,
                            rhohv_min=0.9, zh_min=5., max_off=360, preset=None,
                            preset_tol=5, mode='median', plot_method=False):
        r"""
        Compute the :math:`\Phi_{DP}` offset using PPIs`.

        Parameters
        ----------
        rad_vars : dict
            Dict containing radar variables to plot. :math:`\Phi_{DP}`,
            :math:`\rho_{HV}` and :math:`Z_H` must be in the dict.
        mov_avrgf_len : 2-element tuple or list, optional
            Window size used to smooth :math:`\Phi_{DP}` by applying a
            moving average window. The default is (1, 3). It is
            recommended to average :math:`\Phi_{DP}` along the range,
            i.e. keep the window size in a (1, n) size.
        thr_spdp : int or float, optional
            Threshold used to discard bins with standard deviations of
            :math:`\Phi_{DP}` greater than the selected value.
            The default is 10 deg.
        rhohv_min : float, optional
            Threshold in :math:`\rho_{HV}` used to discard bins
            related to nonmeteorological signals. The default is 0.90
        zh_min : float, optional
            Threshold in :math:`Z_H` used to discard bins related to
            nonmeteorological signals. The default is 5.
        max_off : float or int, optional
            Maximum value allowed for :math:`\Phi_{DP}(0)`.
            The default is 360.
        preset : float or int, optional
            Preset :math:`\Phi_{DP}(0)`. The default is None.
        preset_tol : float or int, optional
            Maximum difference allowed between the preset and
            computed :math:`\Phi_{DP}(0)` values. Offset values that
            exceed this tolerance value are replaced with the preset
            value. The default tolerance is 5 deg.
        mode : str, optional
            Resulting :math:`\Phi_{DP}` offset. The string has to be
            one of 'median' or 'multiple'. If median, :math:`\Phi_{DP}`
            offset is computed as a single value
            (the median of all rays). Otherwise,
            the :math:`\Phi_{DP}` offset is calculated ray-wise.
            The default is 'median'.
        """
        rad_vars = copy.copy(rad_vars)

        if (mov_avrgf_len[1] % 2) == 0:
            print('Choose an odd number to apply the '
                  + 'moving average filter')
        phidp_O = {k: np.ones_like(rad_vars[k]) * rad_vars[k]
                   for k in list(rad_vars) if k.startswith('Phi')}
        # phidp_O['PhiDP [deg]'][:, 0] = np.nan
        # Filter isolated values
        phidp_pad = np.pad(phidp_O['PhiDP [deg]'],
                           ((0, 0), (mov_avrgf_len[1]//2,
                                     mov_avrgf_len[1]//2)),
                           mode='constant', constant_values=(np.nan))
        phidp_dspk = np.array(
            [[np.nan if ~np.isnan(vbin)
              and (np.isnan(phidp_pad[nr][nbin-1])
                   and np.isnan(phidp_pad[nr][nbin+1]))
              else 1 for nbin, vbin in enumerate(phidp_pad[nr])
              if nbin != 0
              and nbin < phidp_pad.shape[1] - mov_avrgf_len[1] + 2]
             for nr in range(phidp_pad.shape[0])], dtype=np.float64)
        # Filter using ZH
        phidp_dspk[rad_vars['ZH [dBZ]'] < zh_min] = np.nan
        # Filter using rhoHV
        phidp_dspk[rad_vars['rhoHV [-]'] < rhohv_min] = np.nan
        # Computes sPhiDP for each ray
        phidp_dspk_rhv = phidp_O['PhiDP [deg]'] * phidp_dspk
        phidp_s = np.nanstd(rolling_window(
            phidp_dspk_rhv, mov_avrgf_len), axis=-1, ddof=1)
        phidp_pad = np.pad(phidp_s, ((0, 0), (mov_avrgf_len[1]//2,
                                              mov_avrgf_len[1]//2)),
                           mode='constant', constant_values=(np.nan))
        # Filter values with std values greater than std threshold
        phidp_sfnv = np.array(
            [[np.nan if vbin >= thr_spdp
              and (phidp_pad[nr][nbin-1] >= thr_spdp
                   or phidp_pad[nr][nbin+1] >= thr_spdp)
              else 1 for nbin, vbin in enumerate(phidp_pad[nr])
              if nbin != 0
              and nbin < phidp_pad.shape[1] - mov_avrgf_len[1] + 2]
             for nr in range(phidp_pad.shape[0])], dtype=np.float64)
        # Filter isolated values
        phidp_sfnv2 = np.array(
            [[np.nan if ~np.isnan(vbin)
                and (np.isnan(phidp_pad[nr][nbin-1])
                     or np.isnan(phidp_pad[nr][nbin+1]))
              else 1 for nbin, vbin in enumerate(phidp_pad[nr])
              if nbin != 0
              and nbin < phidp_pad.shape[1] - mov_avrgf_len[1] + 2]
             for nr in range(phidp_pad.shape[0])], dtype=np.float64)
        phidp_sfnv = phidp_sfnv*phidp_sfnv2
        phidp_f = phidp_dspk_rhv * phidp_sfnv
        # Filter isolated values
        phidp_pad = np.pad(phidp_f, ((0, 0), (mov_avrgf_len[1]//2,
                                              mov_avrgf_len[1]//2)),
                           mode='constant', constant_values=(np.nan))
        phidp_f2 = np.array(
            [[np.nan if ~np.isnan(vbin)
              and (np.isnan(phidp_pad[nr][nbin-1])
                   or np.isnan(phidp_pad[nr][nbin+1]))
              else 1 for nbin, vbin in enumerate(phidp_pad[nr])
              if nbin != 0
              and nbin < phidp_pad.shape[1] - mov_avrgf_len[1] + 2]
             for nr in range(phidp_pad.shape[0])], dtype=np.float64)
        phidp_f = phidp_f * phidp_f2
        # Computes an initial PhiDP(0)
        phidp0 = np.array([[
            nr[np.isfinite(nr)][0] if ~np.isnan(nr).all() else 0
            for nr in phidp_f]], dtype=np.float64).transpose()
        phidp0[phidp0 == 0] = np.nanmedian(phidp0[phidp0 != 0])
        phidp0[abs(phidp0) > max_off] = 0
        phidp0 = np.nan_to_num(phidp0)
        if preset:
            phidp0[abs(phidp0 - preset) > preset_tol] = preset
        if mode == 'median':
            phidp_offset = np.nanmedian(phidp0)
            if abs(phidp_offset) > max_off or np.isnan(phidp_offset):
                phidp_offset = 0
            if preset and abs(phidp_offset - preset) > preset_tol:
                phidp_offset = preset
        elif mode == 'multiple':
            phidp_offset = phidp0.flatten()
        self.phidp_offset = phidp_offset
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
            var = 'PhiDP [deg]'
            phidp02plot = np.nanmedian(phidp0)
            rad_var = phidp0.flatten()
            azim = np.deg2rad(np.arange(len(rad_var)))
            rad_georef = {}
            rad_georef['azim [rad]'] = azim
            rad_display.plot_offsetcorrection(
                rad_georef, rad_params, rad_vars['PhiDP [deg]'], var_m=rad_var,
                var_offset=phidp02plot, var_name=var, mode='other')


    def offset_correction(self, phidp2calib, phidp_offset=0,
                          data2correct=None):
        r"""
        Correct the PhiDP offset using a given value.

        Parameters
        ----------
        phidp2calib : array of float
            Offset-affected differential phase :math:`(\Phi_{DP})`
            in deg.
        phidp_offset : float
            Differential phase offset in deg. The default is 0.
        data2correct : dict, optional
            Dictionary to update the offset-corrected
            :math:`(\Phi_{DP})`. The default is None.
        """
        if isinstance(phidp_offset, (int, float)):
            if np.isnan(phidp_offset):
                phidp_offset = 0
            phidp_oc = copy.deepcopy(phidp2calib) - phidp_offset
        elif isinstance(phidp_offset, (np.ndarray, list, tuple)):
            # TODO: raise error if lens are different
            phidp_oc = copy.deepcopy(phidp2calib)
            phidp_oc[:] = [ray - phidp_offset[cnt]
                           for cnt, ray in enumerate(phidp_oc)]
        if data2correct is None:
            self.vars = {'PhiDP [deg]': phidp_oc}
        else:
            data2cc = copy.deepcopy(data2correct)
            data2cc.update({'PhiDP [deg]': phidp_oc})
            self.vars = data2cc


# =============================================================================
# %% xarray implementation
# =============================================================================
def _empty_stats():
    return xr.Dataset({"offset_max": ((), np.nan), "offset_min": ((), np.nan),
                       "offset_std": ((), np.nan), "offset_sem": ((), np.nan)})


def _phidp_filtering(phidp, rhohv=None, zh=None, window=(1, 3), thr_spdp=10.,
                     minthr_pdp0=-np.inf, rhohv_min=0.9, dbz_min=5., dbz_max=60.,
                     range_dim="range", azimuth_dim="azimuth"):
    r"""
    Filter spurious values in :math:`\Phi_{DP}`.

    Parameters
    ----------
    phidp : xr.DataArray
        Input :math:`\Phi_{DP}` field (degrees).
    rhohv : xr.DataArray, optional
        :math:`\rho_{HV}` field used to remove non‑meteorological gates.
    zh : xr.DataArray, optional
        Reflectivity field (dBZ) used to remove non‑meteorological gates.
    window : tuple of int, default (1, 3)
        Rolling window size ``(m, n)`` where ``m`` is along azimuth and
        ``n`` along range. ``n`` must be odd. Using ``(1, n)`` is recommended.
    thr_spdp : float, default 10.0
        Maximum allowed standard deviation (deg) of :math:`\Phi_{DP}`. Gates
        exceeding this threshold are removed.
    minthr_pdp0 : float, default -inf
        Minimum allowed :math:`\Phi_{DP}` value. If finite, values below this
        threshold are clipped before filtering.
    rhohv_min : float, default 0.9
        Minimum :math:`\rho_{HV}` threshold for retaining meteorological gates.
    dbz_min : float, default 5.0
        Minimum reflectivity threshold (dBZ).
    dbz_max : float, default 60
        Maximum reflectivity threshold (dBZ).
    range_dim : str, default "range"
        Name of the range dimension.
    azimuth_dim : str, default "azimuth"
        Name of the azimuth dimension.

    Returns
    -------
    xr.DataArray
        Filtered :math:`\Phi_{DP}` field with spurious gates masked (NaN).

    Notes
    * NaNs propagate through all masking steps.
    * Only range‑dimension despiking is applied.
    """

    m, n = window
    if n % 2 == 0:
        raise ValueError("Range window length n must be odd.")
    # minthr_pdp0 clipping
    if np.isfinite(minthr_pdp0):
        phidp = xr.where(phidp < minthr_pdp0, minthr_pdp0, phidp)
    # Despike #1 (isolated PHIDP)
    mask1 = despike_isolated(phidp, n, range_dim)
    # ZH mask
    if zh is not None:
        mask1 = mask1 & (zh >= dbz_min) & (zh <= dbz_max)
    # rhoHV mask
    if rhohv is not None:
        mask1 = mask1 & (rhohv >= rhohv_min)
    # Apply mask to PHIDP
    phidp_dspk_rhv = phidp.where(mask1)
    # Rolling std
    phidp_s = rolling_std_xr(phidp_dspk_rhv, mov_avrgf_len=window,
                             azimuth_dim=azimuth_dim, range_dim=range_dim)
    # Std threshold mask
    mask2 = std_mask_threshold(phidp_s, thr_spdp, n, range_dim)
    # Std isolated mask
    mask3 = std_mask_isolated(phidp_s, n, range_dim)
    mask_std = mask2 & mask3
    # Apply std mask
    phidp_f = phidp_dspk_rhv.where(mask_std)
    # Final despike (isolated PHIDP)
    mask4 = despike_isolated(phidp_f, n, range_dim)
    phidp_f = phidp_f.where(mask4)
    return phidp_f


def phidp_offsetdetection_vp(ds, inp_names=None, mlyr=None, min_h=1.1,
                             minbins=2, dbz_min=20., dbz_max=60., rhv_min=0.98,
                             return_stats=False):
    r"""
    Compute the :math:`\Phi_{DP}` calibration offset using vertical profiles
    (VPS), following Frech, (2013).

    Parameters
    ----------
    ds : xarray.Dataset
        VPS dataset containing PHIDP, DBZ, RHOHV, and a height coordinate.
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{"height": "height", "PHIDP": "PHIDP", "DBZ": "DBZH",
        "RHOHV": "RHOHV"}``
    mlyr : xarray.Dataset or None, optional
        Dataset containing the melting-layer boundaries height.
        ``[MLYRTOP, MLYRBTM, MLYRTHK]``. Only gates below the melting layer
        bottom (i.e. the rain region below the melting layer) are included in
        the computation.
    min_h : float, default 1.1
        Minimum height (km) to include in the VPS analysis.
    minbins : int, default 2
        Minimum number of valid bins required to compute the offset.
    dbz_min : float, default 20.
        Minimum ZH threshold (dBZ) for selecting rain gates.
    dbz_max : float, default 60.
        Maximum ZH threshold (dBZ) for selecting rain gates.
    rhv_min : float, default 0.98
        Minimum RHOHV threshold for selecting rain gates.
    return_stats : bool, default False
        If True, return (offset, stats_dataset).

    Returns
    -------
    offset : xr.Dataset
        Scalar :math:`\Phi_{DP}` calibration offset, in deg.
    stats : xr.Dataset, optional
        Dataset with offset_max, offset_min, offset_std, offset_sem.

    Notes
    -----
    * The method estimates the differential phase shift offset following [2]_:
          .. math:: \Phi^{O_{VP}}_{DP} = \frac{1}{n} \sum_{i=1}^{n} \Phi_{{DP}_{i}}
    * Only gates between `min_h` and the melting-layer bottom are used.
    * Only rain gates are used, using threshold in RHOHV and ZH, according to
      [1]_

    References
    ----------
    .. [1] Frech, M., & Frech, M. (2013, September 17). Monitoring the data
        quality of the new polarimetric weather radar network of the German
        Meteorological Service.
        https://ams.confex.com/ams/36Radar/webprogram/Paper228472.html
    .. [2] Sanchez-Rivas, D., & Rico-Ramirez, M. A. (2023). Towerpy: An
        open-source toolbox for processing polarimetric weather radar data.
        Environmental Modelling & Software, 167, 105746.
        https://doi.org/10.1016/j.envsoft.2023.105746
    """
    # 1. Canonical variable mapping
    defaults = {"height": "height", "PHIDP": "PHIDP", "DBZ": "DBZH",
                "RHOHV": "RHOHV"}
    names = {**defaults, **(inp_names or {})}
    height = ds[names["height"]]
    # 2. Determine melting-layer geometry
    if mlyr is None:
        ml_top = 5.0
        ml_thk = 0.5
        ml_bottom = ml_top - ml_thk
    else:
        ml_top = float(mlyr["MLYRTOP"])
        ml_bottom = float(mlyr["MLYRBTM"])
        ml_thk = float(mlyr["MLYRTHK"])
    # 3. Height slicing
    hvals = height.values
    # Invalid MLyr -> offset = 0
    if np.isnan(ml_bottom) or np.isnan(ml_top):
        offset = xr.DataArray(0.0, name="PHIDP_OFFSET")
        stats = _empty_stats() if return_stats else None

    else:
        i0 = find_nearest_index(hvals, min_h)
        i1 = find_nearest_index(hvals, ml_bottom)
        if i1 <= i0:
            offset = xr.DataArray(0.0, name="PHIDP_OFFSET")
            stats = _empty_stats() if return_stats else None

        else:
            ds_sel = ds.isel({names["height"]: slice(i0, i1)})
            phidp_sel = ds_sel[names["PHIDP"]]
            zh_sel = ds_sel[names["DBZ"]]
            rho_sel = ds_sel[names["RHOHV"]]
            # 4. Light-rain filtering
            mask = ((zh_sel >= dbz_min) & (zh_sel <= dbz_max) & (rho_sel >= rhv_min))
            phidp_filt = phidp_sel.where(mask)
            # 5. minbins
            valid_bins = phidp_filt.count(dim=names["height"])
            if valid_bins <= minbins:
                offset = xr.DataArray(0.0, name="PHIDP_OFFSET")
                stats = _empty_stats() if return_stats else None
            else:
                # 6. Compute offset
                offset = phidp_filt.mean(dim=names["height"], skipna=True)
                offset = offset.rename("PHIDP_OFFSET")
                # 7. Stats
                if return_stats:
                    stats = xr.Dataset(
                        {"offset_max": phidp_filt.max(dim=names["height"],
                                                      skipna=True),
                         "offset_min": phidp_filt.min(dim=names["height"],
                                                      skipna=True),
                         "offset_std": phidp_filt.std(dim=names["height"],
                                                      skipna=True),
                         "offset_sem": (phidp_filt.std(dim=names["height"],
                                                       skipna=True) / np.sqrt(valid_bins)),
                        })
                else:
                    stats = None
    # 8. Build output dataset
    coords = {name: coord for name, coord in ds.coords.items()
              if coord.dims == ()}
    data_vars = {"PHIDP_OFFSET": offset}
    if return_stats and stats is not None:
        for k, v in stats.data_vars.items():
            data_vars[k] = v
    ds_out = xr.Dataset(data_vars, coords=coords)    
    # 9. Record provenance
    #TODO: add step_description
    extra = {'step_description': ('')}
    params = {"min_h": min_h, "dbz_min": dbz_min, "dbz_max": dbz_max,
              "rhv_min": rhv_min, "minbins": minbins, "ml_top": ml_top,
              "ml_bottom": ml_bottom, "ml_thickness": ml_thk}
    outputs = 'PHIDP_OFFSET'
    ds_out = record_provenance(
        ds_out, step="compute_phidp0_vps",
        inputs=[names["PHIDP"], names["DBZ"], names["RHOHV"]], outputs=outputs,
        parameters=params, extra_attrs=extra,
        module_provenance="towerpy.calib.calib_phidp.phidp_offsetdetection_vp")
    return ds_out



def phidp_offsetdetection_ppi(ds, inp_names=None, mode="median", rhohv_min=0.9,
                              dbz_min=5., dbz_max=60., mov_avrgf_len=(1, 3),
                              thr_spdp=10, max_off=180, preset=None,
                              preset_tol=5):
    r"""
    Estimate the differential phase offset :math:`\Phi_{DP}(0)` from PPI scans.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing PHIDP (deg) and polar coordinates (range, azimuth).
        Reflectivity (dBZ) and :math:`\rho_{HV}` are recommended for
        thresholding non-meteorological echoes.
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{'azi': 'azimuth', 'rng': 'range', 'ZH': 'DBZH',
        'RHOHV': 'RHOHV', 'PHIDP': 'PHIDP'}``.
    mode : {"median", "multiple"}, default "median"
        Output mode:

        - ``"median"`` - return a single scalar offset (median of all rays).
        - ``"multiple"`` - return ray-wise offsets.
    rhohv_min : float, default 0.9
        Minimum :math:`\rho_{HV}` threshold for retaining meteorological gates.
    dbz_min : float, default 5.
        Minimum reflectivity threshold (dBZ).
    dbz_max : float, default 60.
        Maximum reflectivity threshold (dBZ).
    mov_avrgf_len : tuple of int, default (1, 3)
        Window size for smoothing :math:`\Phi_{DP}` using a moving average.
        Recommended to smooth along range only, i.e. ``(1, n)``.
    thr_spdp : float, default 10
        Maximum allowed standard deviation (deg) of :math:`\Phi_{DP}` along
        range. Rays exceeding this threshold are discarded.
    max_off : float, default 180
        Maximum allowed :math:`\Phi_{DP}(0)` value (deg).
    preset : float or int or None, default None
        Preset :math:`\Phi_{DP}(0)` value. If provided, the computed offset
        is replaced by this value when the difference is within ``preset_tol``.
    preset_tol : float, default 5
        Maximum allowed difference (deg) between the preset and computed
        offsets before enforcing the preset value.

    Returns
    -------
    xr.Dataset
        Dataset containing the detected :math:`\Phi_{DP}` offset(s), with
        variable name ``PHIDP_OFFSET``. In ``"median"`` mode, this is a
        scalar field; in ``"multiple"`` mode, offsets are provided per ray.

    Notes
    -----
    * This function operates in native polar radar coordinates.
    * :math:`\Phi_{DP}` is smoothed before offset detection.
    * Only meteorological gates are used, based on ZH and :math:`\rho_{HV}`
      thresholds.
    * Rays with high :math:`\Phi_{DP}` variability are discarded.
    """
    # 1. Variable mapping
    defaults = {'azi': 'azimuth', 'rng': 'range', "PHIDP": "PHIDP",
                "DBZ": "DBZH", "RHOHV": "RHOHV"}
    names = {**defaults, **(inp_names or {})}
    phidp = ds[names["PHIDP"]]
    rhohv = ds[names["RHOHV"]]
    zh = ds[names["DBZ"]]
    range_dim = names["rng"]
    azimuth_dim = names["azi"]
    # 2. Clean phidp
    phidp_f = _phidp_filtering(
        phidp, rhohv, zh, window=mov_avrgf_len, thr_spdp=thr_spdp,
        rhohv_min=rhohv_min, dbz_min=dbz_min, dbz_max=dbz_max, minthr_pdp0=-np.inf,
        range_dim=range_dim, azimuth_dim=azimuth_dim)
    # 3. Extract PHIDP(0)
    finite = phidp_f.notnull()
    first_gate = finite.idxmax(range_dim)
    phidp0 = phidp_f.sel({range_dim: first_gate})
    # Rays with no finite values -> 0
    no_valid = ~finite.any(range_dim)
    phidp0 = phidp0.where(~no_valid, 0)
    # Replace zeros with median of non-zero
    valid = phidp0 != 0
    median_valid = phidp0.where(valid).median(dim=azimuth_dim, skipna=True)
    phidp0 = phidp0.where(valid, median_valid)
    # Clip > max_off
    phidp0 = xr.where(np.abs(phidp0) > max_off, 0, phidp0)
    # Replace NaNs with 0
    phidp0 = phidp0.fillna(0)
    # Set preset value if defined
    if preset is not None:
        phidp0 = xr.where(np.abs(phidp0 - preset) > preset_tol, preset, phidp0)
    # 4. Output
    if mode == "median":
        out = phidp0.median(dim=azimuth_dim, skipna=True)
        out = xr.where(np.abs(out) > max_off, 0, out)
        if preset is not None:
            out = xr.where(np.abs(out - preset) > preset_tol, preset, out)
        out_name = "PHIDP_OFFSET"
    elif mode == "multiple":
        out = phidp0
        out_name = "PHIDP_OFFSET"
    else:
        raise ValueError("mode must be 'median' or 'multiple'")
    # 5. Build minimal output dataset
    coords = {name: coord for name, coord in ds.coords.items()
              if coord.dims == ()}
    # If ray-wise output, keep azimuth coordinate
    if mode == "multiple":
        coords[names["azi"]] = ds[names["azi"]]
    ds_out = xr.Dataset({out_name: out}, coords=coords)
    # 6. Record dataset-level provenance
    #TODO: add step_description
    extra = {'step_description': ('')}
    params = {"mov_avrgf_len": mov_avrgf_len, "thr_spdp": thr_spdp,
              "rhohv_min": rhohv_min, "dbz_min": dbz_min, "dbz_max": dbz_max,
              "max_off": max_off, "preset": preset, "preset_tol": preset_tol,
              "mode": mode}
    ds_out = record_provenance(
        ds_out, step="offsetdetection_ppi",
        inputs=[names["PHIDP"], names["DBZ"], names["RHOHV"]],
        outputs=[out_name], parameters=params, extra_attrs=extra,
        module_provenance="towerpy.calib.calib_phidp.phidp_offsetdetection_ppi")
    return ds_out


def phidp_qc_processing(ds, inp_names=None, mov_avrgf_len=(1, 3), t_spdp=10,
                        minthr_pdp0=-5, rhohv_min=0.90, dbz_min=5, dbz_max=50,
                        phidp0_correction=False, mlyr_intp=False, mlyr_top=5.,
                        mlyr_thk=0.75, mlyr_btm=None, mask=True,
                        replace_vars=False):
    r"""
    Apply a full quality-control processing workflow to :math:`\Phi_{DP}`

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing PHIDP (in deg) along with the polar coordinates
        (range, azimuth). Reflectivity (in dBZ) and correlation coefficient are
        recommended for thresholding.
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset.
        Defaults to ``{'azi': 'azimuth', 'rng': 'range', 'ZH': 'DBZH',
        'RHOHV': 'RHOHV', 'PHIDP': 'PHIDP'}``.
    mov_avrgf_len : tuple of int, default (1, 3)
        Window size ``(m, n)`` for moving-average smoothing, where ``m`` is
        along azimuth and ``n`` along range. ``n`` must be odd. Using
        ``(1, n)`` is recommended.
    t_spdp : float, default 10
        Maximum allowed rolling standard deviation (deg) of :math:`\Phi_{DP}`.
        Gates exceeding this threshold are removed.
    minthr_pdp0 : float, default -5
        Minimum allowed :math:`\Phi_{DP}` value. Values below this threshold
        are clipped before filtering.
    rhohv_min : float, default 0.90
        Minimum :math:`\rho_{HV}` threshold for retaining meteorological gates.
    dbz_min : float, default 5
        Minimum reflectivity threshold (dBZ).
    dbz_max : float, default 50
        Maximum reflectivity threshold (dBZ).
    phidp0_correction : bool, default False
        If ``True``, compute :math:`\Phi_{DP}(0)` per ray from the first
        finite gate. If ``False``, use a constant value (1e-5) for rays with
        valid data. Zero offsets are replaced by the median of non-zero
        offsets.
    mlyr_intp : bool, default False
        If ``True``, mask the melting-layer region and interpolate
        :math:`\Phi_{DP}` through it using the geometric parameters.
    mlyr_top, mlyr_thk, mlyr_btm : float or array-like, optional
        Melting-layer geometric parameters (km). Each may be scalar or
        per-azimuth arrays. Any two of the three define the third. Used to
        identify and interpolate through the melting layer.
    mask : bool, list of str, dict of str to str, or None, default True
        Controls which variables receive the QC-processed :math:`\\Phi_{DP}`.

        * ``None`` or ``False`` -> classification only; no correction applied.
        * ``True`` -> correct the default PHIDP variable.
        * list of str -> correct only the listed variables.
        * dict -> map input variable names to explicit output names.
    replace_vars : bool, default False
        If True, overwrite selected variables.
        If False, corrected variables receive a ``_QC`` suffix unless
        explicit names are provided via ``mask`` (dict form).

    Returns
    -------
    xarray.Dataset
        Dataset containing the processed :math:`\Phi_{DP}` field and updated
        provenance metadata.

    Notes
    -----
    * This function operates in native polar radar coordinates.
    * :math:`\Phi_{DP}` must be unfolded and offset-corrected before calling
      this function.
    * The processing workflow includes:
        - clipping of low :math:`\Phi_{DP}` values,
        - despiking of isolated gates,
        - ZH and :math:`\rho_{HV}` thresholding,
        - :math:`\Phi_{DP}(0)` estimation and correction,
        - optional melting-layer masking and interpolation,
        - multi-stage moving-average smoothing and interpolation,
        - final NME mask.
    """
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # Resolve variable and coord names
    defaults = {'azi': 'azimuth', 'rng': 'range', "PHIDP": "PHIDP",
                "DBZ": "DBZH", "RHOHV": "RHOHV"}
    names = {**defaults, **(inp_names or {})}
    phidp = ds[names["PHIDP"]]
    rhohv = ds[names["RHOHV"]]
    zh = ds[names["DBZ"]]
    range_dim = names["rng"]
    azimuth_dim = names["azi"]
    # QC API: determine variables to correct
    default_var = names["PHIDP"]
    if mask is None or mask is False:
        # Classification only -> no correction applied
        return ds
    if mask is True:
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
    # =============================================================================
    # PHIDP QC pipeline
    # =============================================================================
    # Clean PHIDP
    phidp_f = _phidp_filtering(
        phidp, rhohv, zh, window=mov_avrgf_len, thr_spdp=t_spdp,
        rhohv_min=rhohv_min, dbz_min=dbz_min, dbz_max=dbz_max,
        minthr_pdp0=minthr_pdp0, range_dim=range_dim, azimuth_dim=azimuth_dim)
    # PHIDP(0) handling
    n = mov_avrgf_len[1]
    # Compute PHIDP(0)
    if phidp0_correction:
        # First finite gate per ray
        finite = phidp_f.notnull()
        first_idx = finite.argmax(range_dim)
        phidp0 = phidp_f.isel(range=first_idx)
        # Rays with no finite values -> 0
        no_valid = ~finite.any(range_dim)
        phidp0 = phidp0.where(~no_valid, 0)
    else:
        # Constant PHIDP(0) = 1e-5 for rays with any valid data, else 0
        has_valid = phidp_f.notnull().any(range_dim)
        phidp0 = xr.where(has_valid, 1e-5, 0)
    # Replace zeros with median of non-zero PHIDP(0)
    valid0 = phidp0 != 0
    median_valid0 = phidp0.where(valid0).median(azimuth_dim)
    phidp0 = phidp0.where(valid0, median_valid0)
    # Apply PHIDP(0) only at the first valid gate (preserve NaN padding)
    first_idx = phidp_f.notnull().argmax(range_dim)
    # Mask selecting only the first valid gate per ray
    first_gate_mask = xr.zeros_like(phidp_f, dtype=bool)
    first_gate_mask = first_gate_mask.isel(range=first_idx)
    # Set that gate to PHIDP(0)
    phidp_f = phidp_f.where(~first_gate_mask, phidp0)
    # Subtract PHIDP(0) from the whole ray
    phidp_f = phidp_f - phidp0
    # Melting-layer filtering and interpolation
    if mlyr_intp:
        # 1. Normalise ML inputs (scalar or per-azimuth arrays)
        ml_top = _normalise_ml_input(mlyr_top, ds, azimuth_dim=azimuth_dim)
        ml_bottom = _normalise_ml_input(mlyr_btm, ds, azimuth_dim=azimuth_dim)
        ml_thick = _normalise_ml_input(mlyr_thk, ds, azimuth_dim=azimuth_dim)
        # 2. Attach ML metadata to the dataset (MLYRTOP, MLYRBTM, MLYRTHK)
        ds_ml = attach_melting_layer(ds, mlyr_top=ml_top, mlyr_bottom=ml_bottom,
                                     mlyr_thickness=ml_thick, overwrite=True,
                                     source="phidp_qc_processing",
                                     method="user-specified")
        # 3. Compute ML classification (rain=1, ML=2, solid=3)
        ml_class = mlyr_ppidelimitation(
            ds_ml, beam_cone="centre", mlyr_top=ml_top, mlyr_bottom=ml_bottom,
            mlyr_thickness=ml_thick, azimuth_dim=azimuth_dim, range_dim=range_dim)
        # 4. mlyr_ppidelimitation
        # ml_region = ml_class  # (azimuth, range) DataArray named "PCP_REGION"
        if "ML_PCP_CLASS" in ml_class:
            ml_region = ml_class["ML_PCP_CLASS"]
        else:
            # beam_cone="all"
            ml_region = ml_class["ML_PCP_CLASS_beamc_height"]
        # 5. Mask PHIDP inside ML (region == 2)
        phidp_fml = phidp_f.where(ml_region != 2)
        # 6. Interpolate NaNs along range, ray by ray
        def _interp_1d_phidp(ray):
            # ray is a 1D numpy array
            if np.isnan(ray).all():
                return ray
            x = np.arange(ray.size, dtype=float)
            mask = ~np.isnan(ray)
            # If fewer than 2 valid points, nothing to interpolate
            if mask.sum() < 2:
                return ray
            ray_out = ray.copy()
            ray_out[~mask] = np.interp(x[~mask], x[mask], ray[mask])
            return ray_out
        phidp_fmli = xr.apply_ufunc(_interp_1d_phidp, phidp_fml,
                                    input_core_dims=[[range_dim]],
                                    output_core_dims=[[range_dim]],
                                    vectorize=True, dask="parallelized",
                                    output_dtypes=[phidp_f.dtype])
        # 7. Replace ML region with interpolated values
        phidp_f = xr.where(ml_region == 2, phidp_fmli, phidp_f)
    # First moving-average stage
    n = mov_avrgf_len[1]
    # Moving average along range, ignore NaNs, no edge extension
    phidp_m = (phidp_f.rolling({range_dim: n}, center=True).mean())
    # Despike isolated gates after MAV
    mask_iso = despike_isolated(phidp_m, n, range_dim=range_dim)
    phidp_m = phidp_m.where(mask_iso)
    # Interpolation + second MAV
    n = mov_avrgf_len[1]
    # Interpolate NaNs along range (no extrapolation)
    phidp_i = phidp_m.interpolate_na(dim=range_dim, method="linear")
    # Restore first n+1 gates from phidp_f
    first = phidp_f.isel(range=slice(0, n+1))
    rest = phidp_i.isel(range=slice(n+1, None))
    phidp_i = xr.concat([first, rest], dim=range_dim)
    # Forward-fill remaining NaNs
    phidp_i = phidp_i.interpolate_na(dim=range_dim, method="nearest", fill_value=None)
    # Second MAV: include NaNs, extend valid
    phidp_maf = (phidp_i.rolling({range_dim: n}, center=True).mean())
    phidp_maf = fill_both(phidp_maf, dim=range_dim)
    # Final ZH mask: where ZH is NaN, set PHIDP to NaN
    phidp_maf = phidp_maf.where(zh.notnull())
    # Determine output name
    ds_out = ds.copy()
    corrected_vars = []
    for var in vars_to_correct:
        if isinstance(mask, dict):
            out_var = rename_map[var]
        else:
            out_var = var if replace_vars else f"{var}_QC"
        # Build attrs
        # parent_attrs = ds[var].attrs.copy()
        parent_attrs = ds[var].attrs.copy() if var in ds else {}
        canonical_attrs = sweep_vars_attrs_f.get(out_var, {}).copy()
        merged_attrs = {**parent_attrs, **canonical_attrs}
        merged_attrs = add_correction_step(
            parent_attrs=merged_attrs,
            step="phidp_qc",
            parent=var,
            params={"mov_avrgf_len": mov_avrgf_len,
                    "t_spdp": t_spdp,
                    "minthr_pdp0": minthr_pdp0,
                    "rhohv_min": rhohv_min,
                    "dbz_min": dbz_min,
                    "dbz_max": dbz_max,
                    "phidp0_correction": phidp0_correction},
            outputs=[out_var],
            mode="overwrite" if replace_vars else "preserve",
            module_provenance="towerpy.calib.calib_phidp.phidp_qc_processing")
        ds_out = safe_assign_variable(ds_out, out_var, phidp_maf,
                                      new_attrs=merged_attrs)
        # If explicit rename requested, rename internal variable -> out_var
        if isinstance(mask, dict) and var != out_var:
            if var in ds_out:
                ds_out = ds_out.rename({var: out_var})
        if replace_vars and var != out_var:
            ds_out = ds_out.drop_vars(var)
        corrected_vars.append(out_var)
    # Dataset-level provenance
    extra = {'step_description': ('Quality-control processing workflow of'
                                  ' PHIDP')}
    ds_out = record_provenance(
        ds_out, step="phidp_qc_processing",
        inputs=[names["PHIDP"], names["DBZ"], names["RHOHV"]],
        outputs=corrected_vars,
        parameters={"mask": mask,
                    "replace_vars": replace_vars,
                    "mov_avrgf_len": mov_avrgf_len,
                    "t_spdp": t_spdp,
                    "dbz_min": dbz_min,
                    "dbz_max": dbz_max,
                    "rhohv_min": rhohv_min,
                    "minthr_pdp0": minthr_pdp0,
                    "phidp0_correction": phidp0_correction},
        extra_attrs=extra,
        module_provenance="towerpy.calib.calib_phidp.phidp_qc_processing")
    return ds_out

