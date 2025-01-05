"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import warnings
import numpy as np
import copy
from ..datavis import rad_display
from ..utils.radutilities import find_nearest
from ..utils.radutilities import rolling_window



class PhiDP_Calibration:
    r"""
    A class to calibrate the radar differential phase :math:`(\Phi_{DP})`.

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

    def __init__(self, radobj):
        self.elev_angle = radobj.elev_angle
        self.file_name = radobj.file_name
        self.scandatetime = radobj.scandatetime
        self.site_name = radobj.site_name

    def offsetdetection_vps(self, pol_profs, mlyr=None, min_h=1.1, max_h=None,
                            zhmin=5, zhmax=30, rhvmin=0.98, minbins=2,
                            stats=False, plot_method=False, rad_georef=None,
                            rad_params=None, rad_vars=None):
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
        rad_params : dict, optional
            Radar technical details. The default is None.
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
            maxheight = find_nearest(pol_profs.georef['profiles_height [km]'],
                                     max_h)

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
                var = 'PhiDP [deg]'
                rad_var = np.array([i[boundaries_idx[0]:boundaries_idx[1]]
                                    for i in rad_vars[var]])
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
            Consecutive bins of :math:`\Phi_{DP}` related to light rain.
            The default is 3.
        stats : dict, optional
            If True, the function returns stats related to the computation of
            the :math:`\Phi_{DP}` offset. The default is False.

        Notes
        -----
        1. Adapted from the method described in [1]

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
                            rhohv_min=0.9, zh_min=5., max_off=360,
                            mode='median'):
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
            :math:`\Phi_{DP}` greater than the selected value. The default is
            10 deg.
        rhohv_min : float, optional
            Threshold in :math:`\rho_{HV}` used to discard bins related to
            nonmeteorological signals. The default is 0.90
        zh_min : float, optional
            Threshold in :math:`Z_H` used to discard bins related to
            nonmeteorological signals. The default is 5.
        max_off : float or int, optional
            Maximum value allowed for :math:`\Phi_{DP}(0)`. The default is 360.
        mode : str, optional
            Resulting :math:`\Phi_{DP}` offset. The string has to be one of
            'median' or 'multiple'. If median, :math:`\Phi_{DP}` offset is
            computed as a single value (the median of all rays).
            Otherwise, the :math:`\Phi_{DP}` offset is calculated ray-wise.
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
             for nr in range(phidp_pad.shape[0])])
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
        # Computes and initial PhiDP(0)
        phidp0 = np.array([[np.nanmedian(
            nr[np.isfinite(nr)][:mov_avrgf_len[1]])
            if ~np.isnan(nr).all() else 0
            for nr in phidp_f]], dtype=np.float64).transpose()
        phidp0[phidp0 == 0] = np.nanmedian(phidp0[phidp0 != 0])
        if mode == 'median':
            phidp_offset = np.nanmedian(phidp0)
            if abs(phidp_offset) > max_off or np.isnan(phidp_offset):
                phidp_offset = 0
        elif mode == 'multiple':
            phidp0[abs(phidp0) > max_off] = 0
            phidp0 = np.nan_to_num(phidp0)
            phidp_offset = phidp0
        self.phidp_offset = phidp_offset
        # return phidp_offset

    def offset_correction(self, phidp2calib, phidp_offset=0,
                          data2correct=None):
        r"""
        Correct the PhiDP offset using a given value.

        Parameters
        ----------
        phidp2calib : array of float
            Offset-affected differential phase :math:`(\Phi_{DP})` in deg.
        phidp_offset : float
            Differential phase offset in deg. The default is 0.
        data2correct : dict, optional
            Dictionary to update the offset-corrected :math:`(\Phi_{DP})`.
            The default is None.
        """
        if np.isnan(phidp_offset):
            phidp_offset = 0
        phidp_oc = copy.deepcopy(phidp2calib) - phidp_offset

        if data2correct is None:
            self.vars = {'PhiDP [deg]': phidp_oc}
        else:
            data2cc = copy.deepcopy(data2correct)
            data2cc.update({'PhiDP [deg]': phidp_oc})
            self.vars = data2cc
