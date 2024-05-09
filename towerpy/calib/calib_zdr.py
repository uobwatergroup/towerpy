"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import warnings
import numpy as np
import copy
from ..utils.radutilities import find_nearest
from ..datavis import rad_display


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

    def __init__(self, radobj):
        self.elev_angle = radobj.elev_angle
        self.file_name = radobj.file_name
        self.scandatetime = radobj.scandatetime
        self.site_name = radobj.site_name

    def offsetdetection_vps(self, pol_profs, mlyr=None, min_h=1.1, zhmin=5,
                            zhmax=30, rhvmin=0.98, minbins=2, stats=False,
                            plot_method=False, rad_georef=None,
                            rad_params=None, rad_vars=None):
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
        rad_params : dict, optional
            Used only to depict the methodolgy. Radar technical details.
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
                var = 'ZDR [dB]'
                rad_var = np.array([i[boundaries_idx[0]:boundaries_idx[1]]
                                    for i in rad_vars[var]])
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
