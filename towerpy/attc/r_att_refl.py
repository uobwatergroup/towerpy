"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import copy
import numpy as np
from scipy.interpolate import interp1d
from ..utils import unit_conversion as tpuc
from ..datavis import rad_display as rdd


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

    def __init__(self, radobj):
        self.elev_angle = radobj.elev_angle
        self.file_name = radobj.file_name
        self.scandatetime = radobj.scandatetime
        self.site_name = radobj.site_name

    def ah_zh(self, rad_vars, var2calc='ZH [dBZ]', rband='C', temp=20.,
              coeff_a=None, coeff_b=None, zh_lower_lim=20., zh_upper_lim=50.,
              copy_ofr=True, data2correct=None, plot_method=False):
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
            temps = np.array((0, 10, 20, 30))
            # Default values for C- and X-band radars
            coeffs_a = {'X': np.array((1.62e-4, 1.15e-4, 7.99e-5, 5.5e-5)),
                        'C': np.array((4.27e-5, 2.89e-5, 2.09e-5, 1.59e-5))}
            coeffs_b = {'X': np.array((0.74, 0.78, 0.82, 0.86)),
                        'C': np.array((0.73, 0.75, 0.76, 0.77))}
            # Interpolate the temp, and coeffs to set the coeffs
            icoeff_a = interp1d(temps, coeffs_a.get(rband))
            icoeff_b = interp1d(temps, coeffs_b.get(rband))
            coeff_a = icoeff_a(temp)
            coeff_b = icoeff_b(temp)
        if var2calc == 'ZH [dBZ]':
            # Copy the original dict to keep variables unchanged
            rvars = copy.deepcopy(rad_vars)
            # Computes Zh
            r_ahzhl = (rvars['AH [dB/km]'] / coeff_a) ** (1 / coeff_b)
            r_ahzh = {}
            # r_ahzh['ZH [mm^6m^-3]'] = r_ahzhl
            # Computes ZH
            r_ahzh['ZH [dBZ]'] = tpuc.x2xdb(r_ahzhl)
            # Filter values using a lower limit
            r_ahzh['ZH [dBZ]'][r_ahzh['ZH [dBZ]'] < zh_lower_lim] = np.nan
            # Filter values using an upper limit
            r_ahzh['ZH [dBZ]'][r_ahzh['ZH [dBZ]'] > zh_upper_lim] = np.nan
            # Filter invalid values
            if copy_ofr and 'ZH [dBZ]' in rad_vars.keys():
                # Filter invalid values
                # Use original values to populate with out-of-range values.
                ind = np.isneginf(r_ahzh['ZH [dBZ]'])
                r_ahzh['ZH [dBZ]'][ind] = rvars['ZH [dBZ]'][ind]
                ind = np.isnan(r_ahzh['ZH [dBZ]'])
                r_ahzh['ZH [dBZ]'][ind] = rvars['ZH [dBZ]'][ind]
        if var2calc == 'AH [dB/km]':
            # Copy the original dict to keep variables unchanged
            rvars = copy.deepcopy(rad_vars)
            # Filter values using a lower limit
            rvars['ZH [dBZ]'][rad_vars['ZH [dBZ]'] < zh_lower_lim] = np.nan
            # Filter values using an upper limit
            rvars['ZH [dBZ]'][rad_vars['ZH [dBZ]'] > zh_upper_lim] = np.nan
            r_ahzh = {}
            r_ahzh['AH [dB/km]'] = (
                coeff_a * (tpuc.xdb2x(rvars['ZH [dBZ]']) ** coeff_b))
            # Filter invalid values
            # ind = np.isneginf(r_ahzh['AH [dB/km]'])
            # r_ahzh['AH [dB/km]'][ind] = rad_vars['ZH [dBZ]'][ind]
            # # Filter invalid values
            # ind = np.isnan(r_ahzh['ZH [dBZ]'])
            # r_ahzh['ZH [dBZ]'][ind] = rad_vars['ZH [dBZ]'][ind]
        self.vars = r_ahzh
        self.coeff_a = coeff_a
        self.coeff_b = coeff_b
        if data2correct is not None:
            # Copy the original dict to keep variables unchanged
            data2cc = copy.deepcopy(data2correct)
            # data2cc = dict(data2correct)
            data2cc.update(r_ahzh)
            self.vars = data2cc

        if plot_method:
            rdd.plot_zhah(rad_vars, r_ahzh, temp, coeff_a, coeff_b,
                          coeffs_a.get(rband), coeffs_b.get(rband), temps,
                          zh_lower_lim, zh_upper_lim)

    def av_zv(self, rad_vars, var2calc='ZV [dBZ]', rband='C', temp=10.,
              coeff_a=None, coeff_b=None, zv_lower_lim=20., zv_upper_lim=50.,
              copy_ofr=True, data2correct=None, plot_method=False):
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
            temps = np.array((0, 10, 20, 30))
            # Default values for C- and X-band radars
            coeffs_a = {'X': np.array((1.35e-4, 9.47e-5, 6.5e-5, 4.46e-5)),
                        'C': np.array((3.87e-5, 2.67e-5, 1.97e-5, 1.53e-5))}
            coeffs_b = {'X': np.array((0.78, 0.82, 0.86, 0.89)),
                        'C': np.array((0.75, 0.77, 0.78, 0.78))}
            # Interpolate the temp, and coeffs to set the coeffs
            icoeff_a = interp1d(temps, coeffs_a.get(rband))
            icoeff_b = interp1d(temps, coeffs_b.get(rband))
            coeff_a = icoeff_a(temp)
            coeff_b = icoeff_b(temp)
        if var2calc == 'ZV [dBZ]':
            # Copy the original dict to keep variables unchanged
            rvars = copy.deepcopy(rad_vars)
            # Computes Zv
            r_avzvl = (rvars['AV [dB/km]'] / coeff_a) ** (1 / coeff_b)
            r_avzv = {}
            # r_avzv['ZV [mm^6m^-3]'] = r_avzvl
            # Computes ZV
            r_avzv['ZV [dBZ]'] = tpuc.x2xdb(r_avzvl)
            # Filter values using a lower limit
            r_avzv['ZV [dBZ]'][r_avzv['ZV [dBZ]'] < zv_lower_lim] = np.nan
            # Filter values using an upper limit
            r_avzv['ZV [dBZ]'][r_avzv['ZV [dBZ]'] > zv_upper_lim] = np.nan
            # Filter invalid values
            if copy_ofr and 'ZV [dBZ]' in rad_vars.keys():
                # Filter invalid values
                # Use original values to populate with out-of-range values.
                ind = np.isneginf(r_avzv['ZV [dBZ]'])
                r_avzv['ZV [dBZ]'][ind] = rvars['ZV [dBZ]'][ind]
                ind = np.isnan(r_avzv['ZV [dBZ]'])
                r_avzv['ZV [dBZ]'][ind] = rvars['ZV [dBZ]'][ind]
        if var2calc == 'AV [dB/km]':
            # Copy the original dict to keep variables unchanged
            rvars = copy.deepcopy(rad_vars)
            # Filter values using a lower limit
            rvars['ZV [dBZ]'][rad_vars['ZV [dBZ]'] < zv_lower_lim] = np.nan
            # Filter values using an upper limit
            rvars['ZV [dBZ]'][rad_vars['ZV [dBZ]'] > zv_upper_lim] = np.nan
            r_avzv = {}
            r_avzv['AV [dB/km]'] = (
                coeff_a * (tpuc.xdb2x(rvars['ZV [dBZ]']) ** coeff_b))
            # Filter invalid values
            # ind = np.isneginf(r_avzv['AV [dB/km]'])
            # r_avzv['AV [dB/km]'][ind] = rad_vars['ZV [dBZ]'][ind]
            # # Filter invalid values
            # ind = np.isnan(r_avzv['ZV [dBZ]'])
            # r_avzv['ZV [dBZ]'][ind] = rad_vars['ZV [dBZ]'][ind]
        self.vars = r_avzv
        self.coeff_a = coeff_a
        self.coeff_b = coeff_b
        if data2correct is not None:
            # Copy the original dict to keep variables unchanged
            data2cc = copy.deepcopy(data2correct)
            # data2cc = dict(data2correct)
            data2cc.update(r_avzv)
            self.vars = data2cc

        if plot_method:
            rdd.plot_zhah(rad_vars, temp, coeff_a, coeff_b,
                          coeffs_a.get(rband), coeffs_b.get(rband), temps)
