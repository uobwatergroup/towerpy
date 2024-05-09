"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import warnings
import numpy as np
import copy
from ..datavis import rad_display
from ..utils.radutilities import find_nearest


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
            boundaries_idx = [find_nearest(pol_profs.georef['profiles_height [km]'], min_h),
                              find_nearest(pol_profs.georef['profiles_height [km]'],
                                           mlvl-mlyr_thickness)]
        else:
            boundaries_idx = [find_nearest(pol_profs.georef['profiles_height [km]'], min_h),
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
                calphidpvps_sem = np.nanstd(calphidp_vps['PhiDP [deg]']) / np.sqrt(len(calphidp_vps['PhiDP [deg]']))

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
