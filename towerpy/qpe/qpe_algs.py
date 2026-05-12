"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
# from __future__ import annotations
from ..utils import radutilities as rut
from ..utils import unit_conversion as tpuc
from ..utils.radutilities import record_provenance

class RadarQPE:
    """
    A class to calculate rain rates from radar variables.

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

    # def adp_to_r(self, adp, rband='C', temp=20., a=None, b=None, mlyr=None,
    #              beam_height=None):
    #     r"""
    #     Compute rain rates using the :math:`R(A_{DP})` estimator [Eq.1]_.

    #     Parameters
    #     ----------
    #     adp : float or array
    #         Floats that corresponds to the differential attenuation, in dB/km.
    #     rband: str
    #         Frequency band according to the wavelength of the radar.
    #         The string has to be one of 'S', 'C' or 'X'. The default is 'C'.
    #     temp: float
    #         Temperature, in :math:`^{\circ}C`, used to derive the coefficients
    #         a, b according to [1]_. The default is 20.
    #     a, b : floats
    #         Override the computed coefficients of the :math:`R(A_{DP})`
    #         relationship. The default are None.
    #     mlyr : MeltingLayer Class, optional
    #         Melting layer class containing the top of the melting layer, (i.e.,
    #         the melting level) and its thickness. Only gates below the melting
    #         layer bottom (i.e. the rain region below the melting layer) are
    #         included in the computation; ml_top and ml_thickness can be either
    #         a single value (float, int), or an array (or list) of values
    #         corresponding to each azimuth angle of the scan. If None, the
    #         rainfall estimator is applied to the whole PPI scan.
    #     beam_height : array, optional
    #         Height of the centre of the radar beam, in km.

    #     Returns
    #     -------
    #     R : dict
    #         Computed rain rates, in mm/h.

    #     Math
    #     ----
    #     .. [Eq.1]
    #     .. math::  R = aA_{DP}^b
    #     where R in mm/h, ADP in dB/km

    #     Notes
    #     -----
    #     Standard values according to [1]_.

    #     References
    #     ----------
    #     .. [1] Ryzhkov, A., Diederich, M., Zhang, P., & Simmer, C. (2014).
    #         "Potential Utilization of Specific Attenuation for Rainfall
    #         Estimation, Mitigation of Partial Beam Blockage, and Radar
    #         Networking" Journal of Atmospheric and Oceanic Technology, 31(3),
    #         599-619. https://doi.org/10.1175/JTECH-D-13-00038.1

    #     """
    #     if a is None and b is None:
    #         # Default values for the temp
    #         temps = np.array((0, 10, 20, 30))
    #         # Default values for S-C and X-band radars
    #         coeffs_a = {'X': np.array((57.8, 53.3, 51.1, 51.0)),
    #                     'C': np.array((281, 326, 393, 483)),
    #                     'S': np.array((3.02e3, 4.12e3, 5.51e3, 7.19e3))}
    #         coeffs_b = {'X': np.array((0.89, 0.85, 0.81, 0.78)),
    #                     'C': np.array((0.95, 0.94, 0.93, 0.93)),
    #                     'S': np.array((1.06, 1.06, 1.06, 1.06))}
    #         # Interpolate the temp and coeffs to set coeffs a and b
    #         icoeff_a = interp1d(temps, coeffs_a.get(rband))
    #         icoeff_b = interp1d(temps, coeffs_b.get(rband))
    #         coeff_a = icoeff_a(temp).item()
    #         coeff_b = icoeff_b(temp).item()
    #     else:
    #         coeff_a = a
    #         coeff_b = b
    #     # print(f'{coeff_a}, {coeff_b}')
    #     adpr = np.zeros_like(adp)+adp
    #     if mlyr is not None and beam_height is not None:
    #         mlyr_bottom = mlyr.ml_top - mlyr.ml_thickness
    #         if isinstance(mlyr_bottom, (int, float)):
    #             mlb_idx = [rut.find_nearest(nbh, mlyr_bottom)
    #                        for nbh in beam_height]
    #         elif isinstance(mlyr_bottom, (np.ndarray, list, tuple)):
    #             mlb_idx = [rut.find_nearest(nbh, mlyr_bottom[cnt])
    #                        for cnt, nbh in enumerate(beam_height)]
    #         for cnt, azi in enumerate(adpr):
    #             azi[mlb_idx[cnt]:] = 0
    #     nanidx = np.where(np.isnan(adp))
    #     adpr[nanidx] = np.nan
    #     r = {'Rainfall [mm/h]': coeff_a*adp**coeff_b}
    #     r['coeff_a'] = coeff_a
    #     r['coeff_b'] = coeff_b
    #     self.r_adp = r

    def ah_to_r(self, ah, rband='C', temp=20., a=None, b=None, mlyr=None,
                beam_height=None):
        r"""
        Compute rain rates using the :math:`R(A_{H})` estimator [Eq.1]_.

        Parameters
        ----------
        ah : float or array
            Floats that corresponds to the specific attenuation, in dB/km.
        rband: str
            Frequency band according to the wavelength of the radar.
            The string has to be one of 'S', 'C' or 'X'. The default is 'C'.
        temp: float
            Temperature, in :math:`^{\circ}C`, used to derive the coefficients
            a, b according to [1]_. The default is 20.
        a, b : floats
            Override the computed coefficients of the :math:`R(A_{H})`
            relationship. The default are None.
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness. Only gates below the melting
            layer bottom (i.e. the rain region below the melting layer) are
            included in the computation; ml_top and ml_thickness can be either
            a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            rainfall estimator is applied to the whole PPI scan.
        beam_height : array, optional
            Height of the centre of the radar beam, in km.

        Returns
        -------
        R : dict
            Computed rain rates, in mm/h.

        Math
        ----
        .. [Eq.1]
        .. math::  R = aA_H^b
        where R in mm/h, AH in dB/km

        Notes
        -----
        Standard values according to [1]_.

        References
        ----------
        .. [1] Ryzhkov, A., Diederich, M., Zhang, P., & Simmer, C. (2014).
            "Potential Utilization of Specific Attenuation for Rainfall
            Estimation, Mitigation of Partial Beam Blockage, and Radar
            Networking" Journal of Atmospheric and Oceanic Technology, 31(3),
            599-619. https://doi.org/10.1175/JTECH-D-13-00038.1

        """
        if a is None and b is None:
            # Default values for the temp
            temps = np.array((0, 10, 20, 30))
            # Default values for S-C and X-band radars
            coeffs_a = {'X': np.array((49.1, 45.5, 43.5, 43.0)),
                        'C': np.array((221, 250, 294, 352)),
                        'S': np.array((2.23e3, 3.10e3, 4.12e3, 5.33e3))}
            coeffs_b = {'X': np.array((0.87, 0.83, 0.79, 0.76)),
                        'C': np.array((0.92, 0.91, 0.89, 0.89)),
                        'S': np.array((1.03, 1.03, 1.03, 1.03))}
            # Interpolate the temp and coeffs to set coeffs a and b
            icoeff_a = interp1d(temps, coeffs_a.get(rband))
            icoeff_b = interp1d(temps, coeffs_b.get(rband))
            coeff_a = icoeff_a(temp).item()
            coeff_b = icoeff_b(temp).item()
        else:
            coeff_a = a
            coeff_b = b
        # print(f'{coeff_a}, {coeff_b}')
        ahr = np.zeros_like(ah)+ah
        if mlyr is not None and beam_height is not None:
            mlyr_bottom = mlyr.ml_top - mlyr.ml_thickness
            if isinstance(mlyr_bottom, (int, float)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom)
                           for nbh in beam_height]
            elif isinstance(mlyr_bottom, (np.ndarray, list, tuple)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom[cnt])
                           for cnt, nbh in enumerate(beam_height)]
            for cnt, azi in enumerate(ahr):
                azi[mlb_idx[cnt]:] = 0
        nanidx = np.where(np.isnan(ah))
        ahr[nanidx] = np.nan
        r = {'Rainfall [mm/h]': coeff_a*ahr**coeff_b}
        r['coeff_a'] = coeff_a
        r['coeff_b'] = coeff_b
        self.r_ah = r

    def kdp_to_r(self, kdp, a=24.68, b=0.81, beam_height=None, mlyr=None):
        r"""
        Compute rain rates using the :math:`R(K_{DP})` estimator [Eq.1]_.

        Parameters
        ----------
        kdp : float or array
            Floats that corresponds to specific differential phase, in deg/km.
        a, b : floats
            Parameters of the :math:`R(K_{DP})` relationship
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness. Only gates below the melting
            layer bottom (i.e. the rain region below the melting layer) are
            included in the computation; ml_top and ml_thickness can be either
            a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            rainfall estimator is applied to the whole PPI scan.

        Returns
        -------
        R : dict
            Computed rain rates, in mm/h.

        Math
        ----
        .. [Eq.1]
        .. math::  R(K_{DP}) = aK_{DP}^b

        where R in mm/h and :math:`K_{DP}` in deg/km.

        Notes
        -----
        Standard values according to [1]_.

        References
        ----------
        .. [1] Bringi, V.N., Rico-Ramirez, M.A., Thurai, M. (2011). "Rainfall
            estimation with an operational polarimetric C-band radar in the
            United Kingdom: Comparison with a gauge network and error
            analysis" Journal of Hydrometeorology 12, 935–954.
            https://doi.org/10.1175/JHM-D-10-05013.1
        """
        kdpr = np.zeros_like(kdp)+kdp
        if mlyr is not None and beam_height is not None:
            mlyr_bottom = mlyr.ml_top - mlyr.ml_thickness
            if isinstance(mlyr_bottom, (int, float)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom)
                           for nbh in beam_height]
            elif isinstance(mlyr_bottom, (np.ndarray, list, tuple)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom[cnt])
                           for cnt, nbh in enumerate(beam_height)]
            for cnt, azi in enumerate(kdpr):
                azi[mlb_idx[cnt]:] = 0
        nanidx = np.where(np.isnan(kdp))
        kdpr[nanidx] = np.nan
        r = {'Rainfall [mm/h]': a*abs(kdpr)**b*np.sign(kdpr)}
        r['coeff_a'] = a
        r['coeff_b'] = b
        self.r_kdp = r

    def kdp_zdr_to_r(self, kdp, zdr, a=37.9, b=0.89, c=-0.72, beam_height=None,
                     mlyr=None):
        r"""
        Compute rain rates using the :math:`R(K_{DP}, Z_{dr})` estimator [Eq.1]_.

        Parameters
        ----------
        kdp : float or array
            Floats that corresponds to specific differential phase,
            in deg/km.
        zdr : float or array
            Floats that corresponds to differential reflectivity, in dB.
        a, b, c : floats
            Parameters of the :math:`R(K_{DP}, Z_{dr})` relationship
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness. Only gates below the melting
            layer bottom (i.e. the rain region below the melting layer) are
            included in the computation; ml_top and ml_thickness can be either
            a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            rainfall estimator is applied to the whole PPI scan.

        Returns
        -------
        R : dict
            Computed rain rates, in mm/h.

        Math
        ----
        .. [Eq.1]
        .. math::  R = aK_{DP}^b Z_{dr}^c
        where R in mm/h, :math:`K_{DP}` in deg/km,
        :math:`Z_{dr} = 10^{0.1*Z_{DR}}` and :math:`Z_{DR}` in dB.

        Notes
        -----
        Standard values according to [1]_

        References
        ----------
        .. [1] Bringi, V.N., Chandrasekar, V., (2001). "Polarimetric Doppler
            Weather Radar" Cambridge University Press, Cambridge ; New York.
            https://doi.org/10.1017/CBO9780511541094

        """
        kdpr = np.zeros_like(kdp)+kdp
        zdrl = tpuc.xdb2x(zdr)
        if mlyr is not None and beam_height is not None:
            mlyr_bottom = mlyr.ml_top - mlyr.ml_thickness
            if isinstance(mlyr_bottom, (int, float)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom)
                           for nbh in beam_height]
            elif isinstance(mlyr_bottom, (np.ndarray, list, tuple)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom[cnt])
                           for cnt, nbh in enumerate(beam_height)]
            for cnt, azi in enumerate(kdpr):
                azi[mlb_idx[cnt]:] = 0
        nanidx = np.where(np.isnan(kdp))
        kdpr[nanidx] = np.nan
        r = {'Rainfall [mm/h]': (a*kdpr**b)*(zdrl**c)}
        r['coeff_a'] = a
        r['coeff_b'] = b
        r['coeff_c'] = c
        self.r_kdp_zdr = r

    def z_to_r(self, zh, a=200, b=1.6, beam_height=None, mlyr=None):
        r"""
        Compute rain rates using the :math:`R(Z_h)` estimator [Eq.1]_.

        Parameters
        ----------
        zh : float or array
             Floats that corresponds to reflectivity, in dBZ.
        a, b : float
            Parameters of the :math:`R(Z_h)` relationship.
        beam_height : array, optional
            Height of the centre of the radar beam, in km, corresponding to
            each azimuth angle of the scan.
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness. Only gates below the melting
            layer bottom (i.e. the rain region below the melting layer) are
            included in the computation; ml_top and ml_thickness can be either
            a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            rainfall estimator is applied to the whole PPI scan.

        Returns
        -------
        r_z : dict
            Computed rain rates, in mm/h.

        Math
        ----
        .. [Eq.1]
        .. math::  Z_h = aR^b
        where R in mm/h, :math:`Z_h = 10^{0.1*Z_H}`, :math:`Z_h` in
        :math:`mm^6 \cdot m^{-3}`

        Notes
        -----
        Standard values according to [1]_.

        References
        ----------
        .. [1] Marshall, J., Hitschfeld, W., & Gunn, K. (1955). Advances in
            radar weather. In Advances in geophysics (pp. 1–56).
            https://doi.org/10.1016/s0065-2687(08)60310-6

        """
        zhl = tpuc.xdb2x(zh)
        if mlyr is not None and beam_height is not None:
            mlyr_bottom = mlyr.ml_top - mlyr.ml_thickness
            if isinstance(mlyr_bottom, (int, float)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom)
                           for nbh in beam_height]
            elif isinstance(mlyr_bottom, (np.ndarray, list, tuple)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom[cnt])
                           for cnt, nbh in enumerate(beam_height)]
            for cnt, azi in enumerate(zhl):
                azi[mlb_idx[cnt]:] = 0
        nanidx = np.where(np.isnan(zh))
        zhl[nanidx] = np.nan
        r = {'Rainfall [mm/h]': (zhl/a)**(1/b)}
        r['coeff_a'] = a
        r['coeff_b'] = b
        self.r_z = r

    def z_zdr_to_r(self, zh, zdr, a=0.0058, b=0.91, c=-2.09,
                   beam_height=None, mlyr=None):
        r"""
        Compute rain rates using the :math:`R(Z_h, Z_{dr})` estimator [Eq.1]_.

        Parameters
        ----------
        zh : float or array
            Floats that corresponds to reflectivity, in dBZ.
        zdr : float or array
            Floats that corresponds to differential reflectivity, in dB.
        a, b, c : floats
            Parameters of the :math:`R(Z_h, Z_{dr})` relationship.
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness. Only gates below the melting
            layer bottom (i.e. the rain region below the melting layer) are
            included in the computation; ml_top and ml_thickness can be either
            a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            rainfall estimator is applied to the whole PPI scan.

        Returns
        -------
        r_z_zdr : dict
            Computed rain rates, in mm/h.

        Math
        ----
        .. [Eq.1]
        .. math:: R(Z_h, Z_{dr}) = aZ_h^b Z_{dr}^c
        where R in mm/h, :math:`Z_h = 10^{0.1*Z_H}`,
        :math:`Z_H` in dBZ, :math:`Z_h` in :math:`mm^6 m^{-3}`,
        :math:`Z_{dr} = 10^{0.1*Z_{DR}}` and :math:`Z_{DR}` in dB.

        Notes
        -----
        Standard values according to [1]_.

        References
        ----------
        .. [1] Bringi, V.N., Chandrasekar, V., 2001. Polarimetric Doppler
            Weather Radar. Cambridge University Press, Cambridge, New York,
            http://dx.doi.org/10.1017/cbo9780511541094.
        """
        zhl = tpuc.xdb2x(zh)
        zdrl = tpuc.xdb2x(zdr)
        if mlyr is not None and beam_height is not None:
            mlyr_bottom = mlyr.ml_top - mlyr.ml_thickness
            if isinstance(mlyr_bottom, (int, float)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom)
                           for nbh in beam_height]
            elif isinstance(mlyr_bottom, (np.ndarray, list, tuple)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom[cnt])
                           for cnt, nbh in enumerate(beam_height)]
            for cnt, azi in enumerate(zhl):
                azi[mlb_idx[cnt]:] = 0
        nanidx = np.where(np.isnan(zh))
        zhl[nanidx] = np.nan
        r = {'Rainfall [mm/h]': (a*zhl**b)*(zdrl**c)}
        r['coeff_a'] = a
        r['coeff_b'] = b
        r['coeff_c'] = c
        self.r_z_zdr = r

    def z_ah_to_r(self, zh, ah, rz_a=200, rz_b=1.6, rah_a=None, rah_b=None,
                  rband='C', temp=20., z_thld=40, beam_height=None, mlyr=None):
        r"""
        Compute rain rates using an hybrid estimator that combines :math:`R(Z_h)` [Eq.1]_ and :math:`R(A_H)` [Eq.2]_ for a given threshold in :math:`Z_H`.

        Parameters
        ----------
        zh : float or array
             Floats that corresponds to reflectivity, in dBZ.
        ah : float or array
            Floats that corresponds to specific attenuation, in dB/km.
        rz_a, rz_b : float
            Parameters of the :math:`R(Z_h)` relationship.
        rah_a, rah_b : floats
            Override the computed coefficients of the :math:`R(A_{H})`
            relationship. The default are None.
        rband: str
            Frequency band according to the wavelength of the radar.
            The string has to be one of 'S', 'C' or 'X'. The default is 'C'.
        temp: float
            Temperature, in :math:`^{\circ}C`, used to derive the coefficients
            rah_a, rah_b according to [1]_. The default is 20.
        z_thld : float, optional
            :math:`Z_H` threshold used for the transition to :math:`R(A_{H})`.
            The default is 40 dBZ.
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness. Only gates below the melting
            layer bottom (i.e. the rain region below the melting layer) are
            included in the computation; ml_top and ml_thickness can be either
            a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            rainfall estimator is applied to the whole PPI scan.

        Returns
        -------
        R : dict
            Computed rain rates, in mm/h.

        Math
        ----
        .. [Eq.1]
        .. math:: Z_H < 40 dBZ \rightarrow Z_h = aR^b
        .. [Eq.2]
        .. math:: Z_H \geq 40 dBZ \rightarrow R = aA_{H}^b
        where R in mm/h, :math:`Z_h = 10^{0.1*Z_H}`, :math:`Z_h` in
        :math:`mm^6 \cdot m^{-3}`, :math:`A_H` in dB/km

        Notes
        -----
        Standard values according to [1]_ and [2]_.

        References
        ----------
        .. [1] Marshall, J., Hitschfeld, W., & Gunn, K. (1955). Advances in
            radar weather. In Advances in geophysics (pp. 1–56).
            https://doi.org/10.1016/s0065-2687(08)60310-6
        .. [2] Ryzhkov, A., Diederich, M., Zhang, P., & Simmer, C. (2014).
            "Potential Utilization of Specific Attenuation for Rainfall
            Estimation, Mitigation of Partial Beam Blockage, and Radar
            Networking" Journal of Atmospheric and Oceanic Technology, 31(3),
            599-619. https://doi.org/10.1175/JTECH-D-13-00038.1
        """
        if rah_a is None and rah_b is None:
            # Default values for the temp
            temps = np.array((0, 10, 20, 30))
            # Default values for S-C and X-band radars
            coeffs_a = {'X': np.array((49.1, 45.5, 43.5, 43.0)),
                        'C': np.array((221, 250, 294, 352)),
                        'S': np.array((2.23e3, 3.10e3, 4.12e3, 5.33e3))}
            coeffs_b = {'X': np.array((0.87, 0.83, 0.79, 0.76)),
                        'C': np.array((0.92, 0.91, 0.89, 0.89)),
                        'S': np.array((1.03, 1.03, 1.03, 1.03))}
            # Interpolate the temp and coeffs to set coeffs a and b
            icoeff_a = interp1d(temps, coeffs_a.get(rband))
            icoeff_b = interp1d(temps, coeffs_b.get(rband))
            coeff_a = icoeff_a(temp).item()
            coeff_b = icoeff_b(temp).item()
        else:
            coeff_a = rah_a
            coeff_b = rah_b
        zh = np.array(zh)
        zhl = tpuc.xdb2x(zh)
        ahr = np.zeros_like(ah)+ah
        if mlyr is not None and beam_height is not None:
            mlyr_bottom = mlyr.ml_top - mlyr.ml_thickness
            if isinstance(mlyr_bottom, (int, float)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom)
                           for nbh in beam_height]
            elif isinstance(mlyr_bottom, (np.ndarray, list, tuple)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom[cnt])
                           for cnt, nbh in enumerate(beam_height)]
            for cnt, azi in enumerate(zhl):
                azi[mlb_idx[cnt]:] = 0
            for cnt, azi in enumerate(ahr):
                azi[mlb_idx[cnt]:] = 0
        nanidx = np.where(np.isnan(zh))
        zhl[nanidx] = np.nan
        nanidx = np.where(np.isnan(ah))
        ahr[nanidx] = np.nan
        rzh = (zhl/rz_a)**(1/rz_b)
        rah = coeff_a*ahr**coeff_b
        rzh[(zh >= z_thld)] = rah[(zh >= z_thld)]
        r = {'Rainfall [mm/h]': rzh}
        r['coeff_arz'] = rz_a
        r['coeff_brz'] = rz_b
        r['coeff_arah'] = coeff_a
        r['coeff_brah'] = coeff_b
        self.r_z_ah = r

    def z_kdp_to_r(self, zh, kdp, rz_a=200, rz_b=1.6, rkdp_a=24.68,
                   rkdp_b=0.81, z_thld=40, beam_height=None, mlyr=None):
        r"""
        Compute rain rates using an hybrid estimator that combines :math:`R(Z_h)` [Eq.1]_ and :math:`R(K_{DP})` [Eq.2]_ for a given threshold in :math:`Z_H`.

        Parameters
        ----------
        zh : float or array
             Floats that corresponds to reflectivity, in dBZ.
        kdp : float or array
            Floats that corresponds to specific differential phase,
            in deg/km.
        rz_a, rz_ab : float
            Parameters of the :math:`R(Z_h)` relationship.
        rkdp_a, rkdp_b : floats
            Parameters of the :math:`R(K_{DP})` relationship.
        z_thld : float, optional
            :math:`Z_H` threshold used for the transition to :math:`R(K_{DP})`.
            The default is 40 dBZ.
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness. Only gates below the melting
            layer bottom (i.e. the rain region below the melting layer) are
            included in the computation; ml_top and ml_thickness can be either
            a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            rainfall estimator is applied to the whole PPI scan.

        Returns
        -------
        R : dict
            Computed rain rates, in mm/h.

        Math
        ----
        .. [Eq.1]
        .. math:: Z_H < 40 dBZ \rightarrow Z_h = aR^b
        .. [Eq.2]
        .. math:: Z_H \geq 40 dBZ \rightarrow R = aK_{DP}^b
        where R in mm/h, :math:`Z_h = 10^{0.1*Z_H}`, :math:`Z_h` in
        :math:`mm^6 \cdot m^{-3}`, :math:`K_{DP}` in deg/km

        Notes
        -----
        Standard values according to [1]_ and [2]_.

        References
        ----------
        .. [1] Marshall, J., Hitschfeld, W., & Gunn, K. (1955). Advances in
            radar weather. In Advances in geophysics (pp. 1–56).
            https://doi.org/10.1016/s0065-2687(08)60310-6
        .. [2] Bringi, V.N., Rico-Ramirez, M.A., Thurai, M. (2011). "Rainfall
            estimation with an operational polarimetric C-band radar in the
            United Kingdom: Comparison with a gauge network and error
            analysis" Journal of Hydrometeorology 12, 935–954.
            https://doi.org/10.1175/JHM-D-10-05013.1
        """
        zh = np.array(zh)
        zhl = tpuc.xdb2x(zh)
        kdpr = np.zeros_like(kdp)+kdp
        if mlyr is not None and beam_height is not None:
            mlyr_bottom = mlyr.ml_top - mlyr.ml_thickness
            if isinstance(mlyr_bottom, (int, float)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom)
                           for nbh in beam_height]
            elif isinstance(mlyr_bottom, (np.ndarray, list, tuple)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom[cnt])
                           for cnt, nbh in enumerate(beam_height)]
            for cnt, azi in enumerate(zhl):
                azi[mlb_idx[cnt]:] = 0
            for cnt, azi in enumerate(kdpr):
                azi[mlb_idx[cnt]:] = 0
        nanidx = np.where(np.isnan(zh))
        zhl[nanidx] = np.nan
        nanidx = np.where(np.isnan(kdp))
        kdpr[nanidx] = np.nan
        rzh = (zhl/rz_a)**(1/rz_b)
        rkdp = rkdp_a*abs(kdpr)**rkdp_b*np.sign(kdpr)
        rzh[(zh >= z_thld)] = rkdp[(zh >= z_thld)]
        r = {'Rainfall [mm/h]': rzh}
        r['coeff_arz'] = rz_a
        r['coeff_brz'] = rz_b
        r['coeff_arkdp'] = rkdp_a
        r['coeff_brkdp'] = rkdp_b
        self.r_z_kdp = r

    def ah_kdp_to_r(self, zh, ah, kdp, rah_a=None, rah_b=None, rkdp_a=24.68,
                    rkdp_b=0.81, rband='C', temp=20., z_thld=40,
                    beam_height=None, mlyr=None):
        r"""
        Compute rain rates using an hybrid estimator that combines :math:`R(A_H)` [Eq.1]_ and :math:`R(K_{DP})` [Eq.2]_ for a given threshold in :math:`Z_H`.

        Parameters
        ----------
        zh : float or array
             Floats that corresponds to reflectivity, in dBZ.
        ah : float or array
            Floats that corresponds to specific attenuation, in dB/km.
        kdp : float or array
            Floats that corresponds to specific differential phase,
            in deg/km.
        rah_a, rah_b : floats
            Override the computed coefficients of the :math:`R(A_{H})`
            relationship. The default are None.
        rkdp_a, rkdp_b : floats
            Parameters of the :math:`R(K_{DP})` relationship.
        rband: str
            Frequency band according to the wavelength of the radar.
            The string has to be one of 'S', 'C' or 'X'. The default is 'C'.
        temp: float
            Temperature, in :math:`^{\circ}C`, used to derive the coefficients
            rah_a, rah_b according to [1]_. The default is 20.
        z_thld : float, optional
            :math:`Z_H` threshold used for the transition from :math:`R(A_{H})`
            to :math:`R(K_{DP})`.
            The default is 40 dBZ.
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness. Only gates below the melting
            layer bottom (i.e. the rain region below the melting layer) are
            included in the computation; ml_top and ml_thickness can be either
            a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            rainfall estimator is applied to the whole PPI scan.

        Returns
        -------
        R : dict
            Computed rain rates, in mm/h.

        Math
        ----
        .. [Eq.1]
        .. math:: Z_H < 40 dBZ \rightarrow R = aA_{H}^b
        .. [Eq.2]
        .. math:: Z_H \geq 40 dBZ \rightarrow R = aK_{DP}^b
        where R in mm/h, :math:`Z_H` in dBZ, :math:`A_H` in dB/km,
        :math:`K_{DP}` in deg/km

        Notes
        -----
        Standard values according to [1]_ and [2]_.

        References
        ----------
        .. [1] Ryzhkov, A., Diederich, M., Zhang, P., & Simmer, C. (2014).
            "Potential Utilization of Specific Attenuation for Rainfall
            Estimation, Mitigation of Partial Beam Blockage, and Radar
            Networking" Journal of Atmospheric and Oceanic Technology, 31(3),
            599-619. https://doi.org/10.1175/JTECH-D-13-00038.1
        .. [2] Bringi, V.N., Rico-Ramirez, M.A., Thurai, M. (2011). "Rainfall
            estimation with an operational polarimetric C-band radar in the
            United Kingdom: Comparison with a gauge network and error
            analysis" Journal of Hydrometeorology 12, 935–954.
            https://doi.org/10.1175/JHM-D-10-05013.1
        """
        if rah_a is None and rah_b is None:
            # Default values for the temp
            temps = np.array((0, 10, 20, 30))
            # Default values for S-C and X-band radars
            coeffs_a = {'X': np.array((49.1, 45.5, 43.5, 43.0)),
                        'C': np.array((221, 250, 294, 352)),
                        'S': np.array((2.23e3, 3.10e3, 4.12e3, 5.33e3))}
            coeffs_b = {'X': np.array((0.87, 0.83, 0.79, 0.76)),
                        'C': np.array((0.92, 0.91, 0.89, 0.89)),
                        'S': np.array((1.03, 1.03, 1.03, 1.03))}
            # Interpolate the temp and coeffs to set coeffs a and b
            icoeff_a = interp1d(temps, coeffs_a.get(rband))
            icoeff_b = interp1d(temps, coeffs_b.get(rband))
            coeff_a = icoeff_a(temp).item()
            coeff_b = icoeff_b(temp).item()
        else:
            coeff_a = rah_a
            coeff_b = rah_b
        zh = np.array(zh)
        ahr = np.zeros_like(ah)+ah
        kdpr = np.zeros_like(kdp)+kdp
        if mlyr is not None and beam_height is not None:
            mlyr_bottom = mlyr.ml_top - mlyr.ml_thickness
            if isinstance(mlyr_bottom, (int, float)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom)
                           for nbh in beam_height]
            elif isinstance(mlyr_bottom, (np.ndarray, list, tuple)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom[cnt])
                           for cnt, nbh in enumerate(beam_height)]
            for cnt, azi in enumerate(zh):
                azi[mlb_idx[cnt]:] = 0
        nanidx = np.where(np.isnan(ah))
        ahr[nanidx] = np.nan
        nanidx = np.where(np.isnan(kdp))
        kdpr[nanidx] = np.nan
        rah = coeff_a*ahr**coeff_b
        rkdp = rkdp_a*abs(kdpr)**rkdp_b*np.sign(kdpr)
        rah[(zh >= z_thld)] = rkdp[(zh >= z_thld)]
        # rkdp[(zh < z_thld)] = rah[(zh < z_thld)]
        r = {'Rainfall [mm/h]': rah}
        # r = {'Rainfall [mm/h]': rkdp}
        r['coeff_arah'] = coeff_a
        r['coeff_brah'] = coeff_b
        r['coeff_arkdp'] = rkdp_a
        r['coeff_brkdp'] = rkdp_b
        self.r_ah_kdp = r


# =============================================================================
# %% xarray implementation
# =============================================================================
# =============================================================================
# %%% QPE documentation
# =============================================================================

# %%%% Auto-generate compute_rqpe docstring

_COMPUTE_RQPE_DOC = """
Compute a radar-based QPE product.

Parameters
----------
qpe_based_on : str
    Name of the rainfall estimator to use. Available options:
    {available_estimators}

See Also
--------
ds.tp.qpe_based_on : detailed documentation for each estimator.
"""

# def _inject_compute_rqpe_doc(func):
#     return func


# %%%% QPE help dict
_QPE_DOCS = {
    # --- Base single-predictor estimators ------------------------------------
    "a": {
        "title": "A-based Rainfall Estimator",
        "math": r".. math:: R = a A^b",
        "units": "R in mm/h, A in dB/km.",
        "notes": "Standard values according to [RYZ14]_.",
        "refs": ["RYZ14"],
        "standard_name": "R(A_H)",
        "latex_name": r"$R(A_H)$",
    },

    "kdp": {
        "title": "KDP-based Rainfall Estimator",
        "math": r".. math:: R = a K_{DP}^b",
        "units": "R in mm/h, KDP in deg/km.",
        "notes": "Standard values according to [BRI11]_.",
        "refs": ["BRI11"],
        "standard_name": "R(K_{DP})",
        "latex_name": r"$R(K_{DP})$",
    },

    "z": {
        "title": "Z-based Rainfall Estimator",
        "math": r".. math:: z = a R^b",
        "units": "R in mm/h, z = 10^{0.1 Z} in mm^6 m^{-3}, Z in dBZ",
        "notes": "Standard values according to [MP55]_.",
        "refs": ["MP55"],
        "standard_name": "R(Z_H)",
        "latex_name": r"$R(Z_H)$",
    },

    # --- Dual-predictor estimators -------------------------------------------
    "kdp_zdr": {
        "title": "KDP–ZDR Dual-Predictor Rainfall Estimator",
        "math": r".. math:: R = a K_{DP}^b Z_{dr}^c",
        "units": (
            "R in mm/h, KDP in deg/km, "
            "Zdr = 10^{0.1 ZDR} (ZDR in dB)."
        ),
        "notes": "Standard values according to [BRI01]_.",
        "refs": ["BRI01"],
        "standard_name": "R(K_{DP}, Z_{DR})",
        "latex_name": r"$R(K_{DP}, Z_{DR})$",
    },

    "z_zdr": {
        "title": "Z–ZDR Dual-Predictor Rainfall Estimator",
        "math": r".. math:: R = a z^b Z_{dr}^c",
        "units": (
            "z = 10^{0.1 Z} (Z in dBZ), "
            "Zdr = 10^{0.1 ZDR} (ZDR in dB)."
        ),
        "notes": "Standard values according to [BRI01]_.",
        "refs": ["BRI01"],
        "standard_name": "R(Z_H, Z_{DR})",
        "latex_name": r"$R(Z_H, Z_{DR})$",
    },

    # --- Hybrid estimators ----------------------------------------------------
    "z_a": {
        "title": "Hybrid Z–A Rainfall Estimator",
        "math": (
            r".. math::\n"
            r"   Z < Z_{thld} \rightarrow z = a R^b \\\n"
            r"   Z \ge Z_{thld} \rightarrow R = a A^b"
        ),
        "units": (
            "R in mm/h, z = 10^{0.1 Z} in mm^6 m^{-3}, "
            "A in dB/km."
        ),
        "notes": "Standard values according to [MP55]_ and [RYZ14]_.",
        "refs": ["MP55", "RYZ14"],
        "standard_name": "R(Z_H)&R(A_H)",
        "latex_name": r"$R(Z_H) \& R(A_H)$",
    },

    "z_kdp": {
        "title": "Hybrid Z–KDP Rainfall Estimator",
        "math": (
            r".. math::\n"
            r"   Z < Z_{thld} \rightarrow z = a R^b \\\n"
            r"   Z \ge Z_{thld} \rightarrow R = a K_{DP}^b"
        ),
        "units": (
            "R in mm/h, z = 10^{0.1 Z} in mm^6 m^{-3}, "
            "KDP in deg/km."
        ),
        "notes": "Standard values according to [MP55]_ and [BRI11]_.",
        "refs": ["MP55", "BRI11"],
        "standard_name": "R(Z_H)&R(K_{DP})",
        "latex_name": r"$R(Z_H) \& R(K_{DP})$",
    },

    "a_kdp": {
        "title": "Hybrid A–KDP Rainfall Estimator",
        "math": (
            r".. math::\n"
            r"   Z < Z_{thld} \rightarrow R = a A^b \\\n"
            r"   Z \ge Z_{thld} \rightarrow R = a K_{DP}^b"
        ),
        "units": (
            "R in mm/h, A in dB/km, "
            "KDP in deg/km."
        ),
        "notes": "Standard values according to [RYZ14]_ and [BRI11]_.",
        "refs": ["RYZ14", "BRI11"],
        "standard_name": "R(A_H)&R(K_{DP})",
        "latex_name": r"$R(A_H) \& R(K_{DP})$",
    },

    # --- Optimised estimators -------------------------------------------------
    "z_opt": {
        "title": "Optimised Z-based Rainfall Estimator",
        "math": r".. math:: z = a R^b",
        "units": "R in mm/h, z = 10^{0.1 Z} in mm^6 m^{-3}, Z in dBZ",
        "notes": "Coefficients optimised adaptively.",
        "refs": ["SR26"],
        "standard_name": "R(Z_H)[opt]",
        "latex_name": r"$R(Z_H)[opt]$",
    },

    "kdp_opt": {
        "title": "Optimised KDP-based Rainfall Estimator",
        "math": r".. math:: R = a K_{DP}^b",
        "units": "R in mm/h, KDP in deg/km.",
        "notes": "Coefficients optimised adaptively.",
        "refs": ["SR26"],
        "standard_name": "R(K_{DP})[opt]",
        "latex_name": r"$R(K_{DP})[opt]$",
    },

    # --- Hybrid optimised -----------------------------------------------------
    # "z_opt_a": {
    #     "title": "Hybrid Z_opt–A Rainfall Estimator",
    #     "math": r".. math:: R = f(Z_{opt}, A)",
    #     "units": "R in mm/h.",
    #     "notes": "Optimised Z–R used in low-Z regime; A–R in high-Z regime.",
    #     "refs": ["MP55", "RYZ14"],
    #     "standard_name": "R(Z_H)[opt]&R(A_H)",
    #     "latex_name": r"$R(Z_H)[opt] \& R(A_H)$",
    # },

    # "z_opt_kdp": {
    #     "title": "Hybrid Z_opt–KDP Rainfall Estimator",
    #     "math": r".. math:: R = f(Z_{opt}, K_{DP})",
    #     "units": "R in mm/h.",
    #     "notes": "Optimised Z–R used in low-Z regime; KDP–R in high-Z regime.",
    #     "refs": ["MP55", "BRI11"],
    #     "standard_name": "R(Z_H)[opt]&R(K_{DP})",
    #     "latex_name": r"$R(Z_H)[opt] \& R(K_{DP})$",
    # },

    # "z_opt_kdp_opt": {
    #     "title": "Hybrid Z_opt–KDP_opt Rainfall Estimator",
    #     "math": r".. math:: R = f(Z_{opt}, K_{DP,opt})",
    #     "units": "R in mm/h.",
    #     "notes": "Both predictors use adaptive coefficients.",
    #     "refs": ["MP55", "BRI11"],
    #     "standard_name": "R(Z_H)[opt]&R(K_{DP})[opt]",
    #     "latex_name": r"$R(Z_H)[opt] \& R(K_{DP})[opt]$",
    # },

    # # --- DE (Germany-optimised) variants -------------------------------------
    # "z_de": {
    #     "title": "Z-based Rainfall Estimator [DE]",
    #     "math": r".. math:: z = a R^b",
    #     "units": "R in mm/h, z in mm^6 m^{-3}.",
    #     "notes": "Coefficients optimised for Germany (DE).",
    #     "refs": ["MP55"],
    #     "standard_name": "R(Z_H)[DE]",
    #     "latex_name": r"$R(Z_H)[DE]$",
    # },

    # "kdp_de": {
    #     "title": "KDP-based Rainfall Estimator [DE]",
    #     "math": r".. math:: R = a K_{DP}^b",
    #     "units": "R in mm/h, KDP in deg/km.",
    #     "notes": "Coefficients optimised for Germany (DE).",
    #     "refs": ["BRI11"],
    #     "standard_name": "R(K_{DP})[DE]",
    #     "latex_name": r"$R(K_{DP})[DE]$",
    # },

    # "a_kdp_de": {
    #     "title": "Hybrid A–KDP Rainfall Estimator [DE]",
    #     "math": r".. math:: R = f(A, K_{DP})",
    #     "units": "R in mm/h.",
    #     "notes": "Hybrid estimator with coefficients optimised for Germany (DE).",
    #     "refs": ["RYZ14", "BRI11"],
    #     "standard_name": "R(A_H)&R(K_{DP})[DE]",
    #     "latex_name": r"$R(A_H) \& R(K_{DP})[DE]$",
    # },
}


# %%%% Refs

_QPE_REFS = {
    "MP55": r"""
Marshall, J., Hitschfeld, W., & Gunn, K. (1955). Advances in radar weather.
In Advances in geophysics (pp. 1–56).
https://doi.org/10.1016/s0065-2687(08)60310-6
""",

    "RYZ14": r"""
Ryzhkov, A., Diederich, M., Zhang, P., & Simmer, C. (2014). Potential 
utilization of specific attenuation for rainfall estimation, mitigation of 
partial beam blockage, and radar networking. Journal of Atmospheric and 
Oceanic Technology, 31(3), 599–619. https://doi.org/10.1175/jtech-d-13-00038.1
""",

    "BRI11": r"""
Bringi, V. N., Rico-Ramirez, M. A., & Thurai, M. (2011). Rainfall Estimation 
with an Operational Polarimetric C-Band Radar in the United Kingdom: 
Comparison with a Gauge Network and Error Analysis. Journal of 
Hydrometeorology, 12(5), 935–954. https://doi.org/10.1175/jhm-d-10-05013.1
""",

    "BRI01": r"""
Bringi, V. N., & Chandrasekar, V. (2001). Polarimetric doppler 
weather radar. In Cambridge University Press. 
https://doi.org/10.1017/cbo9780511541094
"""
}



# =============================================================================
# %%% QPE engines
# =============================================================================


def _kernel_power_law(x, a, b):
    return a * np.power(x, b)

def _kernel_power_law_two(x, y, a, b, c):
    return a * np.power(x, b) * np.power(y, c)

def _kernel_kdp(x, a, b):
    return a * np.power(np.abs(x), b) * np.sign(x)

def _kernel_hybrid_threshold(var, r_low, r_high, threshold):
    """
    Generic hybrid threshold kernel.
    var: array used for thresholding (e.g. Z, ZV, KDP, A)
    r_low: rain rate for var < threshold
    r_high: rain rate for var >= threshold
    """
    out = r_low.copy()
    mask = var >= threshold
    out[mask] = r_high[mask]
    return out


# %%% xarray wrappers around the kernels

def rr_power(x, *, a, b):
    """R = a * x**b"""
    return xr.apply_ufunc(_kernel_power_law, x, a, b,
                          input_core_dims=[x.dims, [], []],
                          output_core_dims=[x.dims],
                          dask="parallelized", vectorize=True)


def rr_power2(x, y, *, a, b, c):
    """R = a * x**b * y**c"""
    return xr.apply_ufunc(_kernel_power_law_two, x, y, a, b, c,
                          input_core_dims=[x.dims, y.dims, [], [], []],
                          output_core_dims=[x.dims],
                          dask="parallelized", vectorize=True)


def rr_kdp(x, *, a, b):
    """R(KDP) = a * |KDP|**b * sign(KDP)"""
    return xr.apply_ufunc(_kernel_kdp, x, a, b,
                          input_core_dims=[x.dims, [], []],
                          output_core_dims=[x.dims],
                          dask="parallelized", vectorize=True)


def rr_hybrid(var, r_low, r_high, *, threshold):
    """Generic hybrid threshold: var < thr -> r_low, else r_high."""
    return xr.apply_ufunc(_kernel_hybrid_threshold, var, r_low, r_high,
                          threshold,
                          input_core_dims=[var.dims, var.dims, var.dims, []],
                          output_core_dims=[var.dims],
                          dask="parallelized", vectorize=True)

# =============================================================================
# %%% QPE engines (final-QPE masking)
# =============================================================================

def _apply_rain_mask(r, pred_nan, ml_mask):
    """
    Apply QPE masking
    """
    # 1) predictor NaN -> QPE = NaN
    r = r.where(~pred_nan)
    # 2) outside ML & predictor valid -> QPE = 0
    if ml_mask is not None:
        outside_and_valid = (~ml_mask) & (~pred_nan)
        r = xr.where(outside_and_valid, 0, r)
    return r


# 2) R(A)
def r_a(spcatt, *, a, b, ml_mask=None):
    r = rr_power(spcatt, a=a, b=b)
    pred_nan = xr.ufuncs.isnan(spcatt)
    return _apply_rain_mask(r, pred_nan, ml_mask)


# 3) R(KDP)
def r_kdp(kdp, *, a, b, ml_mask=None):
    r = rr_kdp(kdp, a=a, b=b)
    pred_nan = xr.ufuncs.isnan(kdp)
    return _apply_rain_mask(r, pred_nan, ml_mask)


# 4) R(KDP, ZDR)
def r_kdp_zdr(kdp, zdr, *, a, b, c, ml_mask=None):
    zdr_lin = tpuc.xdb2x(zdr)
    r = rr_power2(kdp, zdr_lin, a=a, b=b, c=c)
    pred_nan = xr.ufuncs.isnan(kdp) | xr.ufuncs.isnan(zdr_lin)
    return _apply_rain_mask(r, pred_nan, ml_mask)


# 5) R(Z)
def r_z(dbz, *, a, b, ml_mask=None):
    z_lin = tpuc.xdb2x(dbz)
    r = rr_power(z_lin / a, a=1.0, b=1.0 / b)
    pred_nan = xr.ufuncs.isnan(dbz)
    return _apply_rain_mask(r, pred_nan, ml_mask)


# 6) R(Z, ZDR)
def r_z_zdr(dbz, zdr, *, a, b, c, ml_mask=None):
    z_lin = tpuc.xdb2x(dbz)
    zdr_lin = tpuc.xdb2x(zdr)
    r = rr_power2(z_lin, zdr_lin, a=a, b=b, c=c)
    pred_nan = xr.ufuncs.isnan(z_lin) | xr.ufuncs.isnan(zdr_lin)
    return _apply_rain_mask(r, pred_nan, ml_mask)


# 7) Hybrid R(Z) / R(A)
def r_z_a(dbz, spcatt, *, rz_a, rz_b, ra_a, ra_b, thld_var_value,
          ml_mask=None, threshold_var=None):
    z_lin = tpuc.xdb2x(dbz)
    thr = threshold_var
    r_z  = rr_power(z_lin / rz_a, a=1.0, b=1.0 / rz_b)
    r_a = rr_power(spcatt, a=ra_a, b=ra_b)
    r = rr_hybrid(thr, r_low=r_z, r_high=r_a, threshold=thld_var_value)
    pred_nan = xr.ufuncs.isnan(z_lin) | xr.ufuncs.isnan(spcatt)
    return _apply_rain_mask(r, pred_nan, ml_mask)


# 8) Hybrid R(Z) / R(KDP)
def r_z_kdp(dbz, kdp, *, rz_a, rz_b, rkdp_a, rkdp_b, thld_var_value,
               ml_mask=None, threshold_var=None):
    z_lin = tpuc.xdb2x(dbz)
    thr = threshold_var
    r_z   = rr_power(z_lin / rz_a, a=1.0, b=1.0 / rz_b)
    r_kdp = rr_kdp(kdp, a=rkdp_a, b=rkdp_b)
    r = rr_hybrid(thr, r_low=r_z, r_high=r_kdp, threshold=thld_var_value)
    pred_nan = xr.ufuncs.isnan(z_lin) | xr.ufuncs.isnan(kdp)
    return _apply_rain_mask(r, pred_nan, ml_mask)


# 9) Hybrid R(A) / R(KDP)
def r_a_kdp(spcatt, kdp, *, ra_a, ra_b, rkdp_a, rkdp_b, thld_var_value,
            ml_mask=None, threshold_var=None):
    thr = threshold_var
    r_a  = rr_power(spcatt, a=ra_a, b=ra_b)
    r_kdp = rr_kdp(kdp, a=rkdp_a, b=rkdp_b)
    r = rr_hybrid(thr, r_low=r_a, r_high=r_kdp, threshold=thld_var_value)
    pred_nan = xr.ufuncs.isnan(spcatt) | xr.ufuncs.isnan(kdp)
    return _apply_rain_mask(r, pred_nan, ml_mask)


# 10) Future hybrids can reuse the same patterns

# =============================================================================
# %%% Coefficient‑lookup module
# =============================================================================
"""
Coefficient lookup tables for all QPE estimators.

This module centralises all scientific constants used in the rainfall
estimators, ensuring consistency, maintainability, and testability.

All functions return *scalars* (floats), never arrays.
"""


def _interp_coeff(temps, values, temp):
    """Interpolate coefficient table to the requested temperature."""
    f = interp1d(temps, values)
    return float(f(temp))


# AV coefficients (Ryzhkov et al. 2014)
_AV_COEFFS_A = {
    "X": np.array((57.8, 53.3, 51.1, 51.0)),
    "C": np.array((281, 326, 393, 483)),
    "S": np.array((3.02e3, 4.12e3, 5.51e3, 7.19e3)),
    }

_AV_COEFFS_B = {
    "X": np.array((0.89, 0.85, 0.81, 0.78)),
    "C": np.array((0.95, 0.94, 0.93, 0.93)),
    "S": np.array((1.06, 1.06, 1.06, 1.06)),
    }

_TEMPS = np.array((0, 10, 20, 30))


def get_coeffs_av(rband: str, temp: float) -> tuple[float, float]:
    """Return (a, b) for R(AV)."""
    a = _interp_coeff(_TEMPS, _AV_COEFFS_A[rband], temp)
    b = _interp_coeff(_TEMPS, _AV_COEFFS_B[rband], temp)
    return a, b


# AH coefficients (Ryzhkov et al. 2014)
_AH_COEFFS_A = {
    "X": np.array((49.1, 45.5, 43.5, 43.0)),
    "C": np.array((221, 250, 294, 352)),
    "S": np.array((2.23e3, 3.10e3, 4.12e3, 5.33e3)),
    }

_AH_COEFFS_B = {
    "X": np.array((0.87, 0.83, 0.79, 0.76)),
    "C": np.array((0.92, 0.91, 0.89, 0.89)),
    "S": np.array((1.03, 1.03, 1.03, 1.03)),
    }


def get_coeffs_ah(rband: str, temp: float) -> tuple[float, float]:
    """Return (a, b) for R(A)."""
    a = _interp_coeff(_TEMPS, _AH_COEFFS_A[rband], temp)
    b = _interp_coeff(_TEMPS, _AH_COEFFS_B[rband], temp)
    return a, b


# Z–R coefficients (Marshall–Palmer 1948)
def get_coeffs_zr() -> tuple[float, float]:
    """Return (a, b) for R(Z)."""
    return 200.0, 1.6


# KDP coefficients (Bringi et al. 2011)
def get_coeffs_kdp() -> tuple[float, float]:
    """Return (a, b) for R(KDP)."""
    return 24.68, 0.81


# KDP–ZDR coefficients (Bringi & Chandrasekar 2001)
def get_coeffs_kdp_zdr() -> tuple[float, float, float]:
    """Return (a, b, c) for R(KDP, ZDR)."""
    return 37.9, 0.89, -0.72


# Z–ZDR coefficients (Bringi & Chandrasekar 2001)
def get_coeffs_z_zdr() -> tuple[float, float, float]:
    """Return (a, b, c) for R(Z, ZDR)."""
    return 0.0058, 0.91, -2.09


# Hybrid coefficients (Z–R + A or Z–R + KDP)
# These simply reuse the above functions.
def get_coeffs_z_a(rband: str, temp: float):
    """Return ((rz_a, rz_b), (ra_a, ra_b))."""
    return get_coeffs_zr(), get_coeffs_ah(rband, temp)


def get_coeffs_z_kdp():
    """Return ((rz_a, rz_b), (rkdp_a, rkdp_b))."""
    return get_coeffs_zr(), get_coeffs_kdp()


def get_coeffs_a_kdp(rband: str, temp: float):
    """Return ((ra_a, ra_b), (rkdp_a, rkdp_b))."""
    return get_coeffs_ah(rband, temp), get_coeffs_kdp()

# =============================================================================
# %%% QPE Dispatcher
# =============================================================================

def qpe_dispatcher(qpe_based_on, *, spcatt=None, kdp=None, dbz=None,
                   zdr=None, ml_mask=None, rband='C', temp=20.0, pol='H',
                   threshold_var=None, thld_var_value=40.0, 
                   # single‑predictor overrides
                   rz_a=None, rz_b=None, rspcatt_a=None, rspcatt_b=None,
                   rkdp_a=None, rkdp_b=None,
                   # dual‑predictor overrides
                   rkdpzdr_a=None, rkdpzdr_b=None, rkdpzdr_c=None,
                   rzzdr_a=None, rzzdr_b=None, rzzdr_c=None):
    """Internal QPE dispatcher."""
    qpe_based_on = qpe_based_on.lower()
    # Single predictor
    if qpe_based_on == "a":
        if spcatt is None:
            raise ValueError("A field is required for qpe_based_on='a'.")
        if pol.upper() == "H":
            a_default, b_default = get_coeffs_ah(rband, temp)
        elif pol.upper() == "V":
            a_default, b_default = get_coeffs_av(rband, temp)
        else:
            raise ValueError("pol must be 'H' or 'V'")
        a_used = rspcatt_a if rspcatt_a is not None else a_default
        b_used = rspcatt_b if rspcatt_b is not None else b_default
        return r_a(spcatt, a=a_used, b=b_used, ml_mask=ml_mask)
    if qpe_based_on == "kdp":
        if kdp is None:
            raise ValueError("KDP field is required for qpe_based_on='kdp'.")
        if rkdp_a is None or rkdp_b is None:
            rkdp_a, rkdp_b = get_coeffs_kdp()
        return r_kdp(kdp, a=rkdp_a, b=rkdp_b, ml_mask=ml_mask)
    if qpe_based_on == "z":
        if dbz is None:
            raise ValueError("Z field is required for qpe_based_on='z'.")
        if rz_a is None or rz_b is None:
            rz_a, rz_b = get_coeffs_zr()
        return r_z(dbz, a=rz_a, b=rz_b, ml_mask=ml_mask)
    # Two predictors
    if qpe_based_on == "kdp_zdr":
        if kdp is None or zdr is None:
            raise ValueError("KDP and ZDR fields are required for qpe_based_on='kdp_zdr'.")
        if rkdpzdr_a is None or rkdpzdr_b is None or rkdpzdr_c is None:
            rkdpzdr_a, rkdpzdr_b, rkdpzdr_c = get_coeffs_kdp_zdr()
        return r_kdp_zdr(kdp, zdr, a=rkdpzdr_a, b=rkdpzdr_b, c=rkdpzdr_c,
                            ml_mask=ml_mask)
    if qpe_based_on == "z_zdr":
        if dbz is None or zdr is None:
            raise ValueError("Z and ZDR fields are required for qpe_based_on='z_zdr'.")
        if rzzdr_a is None or rzzdr_b is None or rzzdr_c is None:
            rzzdr_a, rzzdr_b, rzzdr_c = get_coeffs_z_zdr()
        return r_z_zdr(dbz, zdr, a=rzzdr_a, b=rzzdr_b, c=rzzdr_c,
                          ml_mask=ml_mask)
    # Hybrids
    if qpe_based_on == "z_a":
        if dbz is None or spcatt is None:
            raise ValueError("Z and A fields are required for qpe_based_on='z_a'.")
        (rz_a_default, rz_b_default), _ = get_coeffs_z_a(rband, temp)
        rz_a = rz_a if rz_a is not None else rz_a_default
        rz_b = rz_b if rz_b is not None else rz_b_default
        if pol.upper() == "H":
            a_default, b_default = get_coeffs_ah(rband, temp)
        elif pol.upper() == "V":
            a_default, b_default = get_coeffs_av(rband, temp)
        else:
            raise ValueError("pol must be 'H' or 'V'")
        a_used = rspcatt_a if rspcatt_a is not None else a_default
        b_used = rspcatt_b if rspcatt_b is not None else b_default
        return r_z_a(dbz, spcatt, rz_a=rz_a, rz_b=rz_b, ra_a=a_used, ra_b=b_used,
                         ml_mask=ml_mask, thld_var_value=thld_var_value,
                         threshold_var=threshold_var)
    if qpe_based_on == "z_kdp":
        if dbz is None or kdp is None:
            raise ValueError("Z and KDP fields are required for qpe_based_on='z_kdp'.")
        if rz_a is None or rz_b is None or rkdp_a is None or rkdp_b is None:
            (rz_a_default, rz_b_default), (rkdp_a_default, rkdp_b_default) = get_coeffs_z_kdp()
            rz_a = rz_a if rz_a is not None else rz_a_default
            rz_b = rz_b if rz_b is not None else rz_b_default
            rkdp_a = rkdp_a if rkdp_a is not None else rkdp_a_default
            rkdp_b = rkdp_b if rkdp_b is not None else rkdp_b_default
        return r_z_kdp(dbz, kdp, rz_a=rz_a, rz_b=rz_b, rkdp_a=rkdp_a,
                          rkdp_b=rkdp_b, thld_var_value=thld_var_value,
                          threshold_var=threshold_var, ml_mask=ml_mask)
    if qpe_based_on == "a_kdp":
        if spcatt is None or kdp is None:
            raise ValueError("A and KDP fields are required for qpe_based_on='a_kdp'.")

        (_, _), (rkdp_a_default, rkdp_b_default) = get_coeffs_a_kdp(rband, temp)
        rkdp_a = rkdp_a if rkdp_a is not None else rkdp_a_default
        rkdp_b = rkdp_b if rkdp_b is not None else rkdp_b_default

        if pol.upper() == "H":
            a_default, b_default = get_coeffs_ah(rband, temp)
        elif pol.upper() == "V":
            a_default, b_default = get_coeffs_av(rband, temp)
        else:
            raise ValueError("pol must be 'H' or 'V'")

        a_used = rspcatt_a if rspcatt_a is not None else a_default
        b_used = rspcatt_b if rspcatt_b is not None else b_default

        return r_a_kdp(spcatt, kdp, ra_a=a_used, ra_b=b_used,
                       rkdp_a=rkdp_a, rkdp_b=rkdp_b,
                       thld_var_value=thld_var_value,
                       threshold_var=threshold_var, ml_mask=ml_mask)

    raise ValueError(f"Unknown QPE qpe_based_on: {qpe_based_on!r}")


# =============================================================================
# %%% Compute Radar QPE
# =============================================================================
# @_inject_compute_rqpe_doc
def compute_rqpe(ds, qpe_based_on, predictor_names=None, rband="C", temp=20.,
                 pol='H', qpe_amlb=False, ml_mask=None,
                 ml_mask_name="ML_PCP_CLASS", threshold_var_name=None,
                 thld_var_value=40.0, out_name=None, append_to=None,
                 rz_a=None, rz_b=None, rspcatt_a=None, rspcatt_b=None,
                 rkdp_a=None, rkdp_b=None,
                 rkdpzdr_a=None, rkdpzdr_b=None, rkdpzdr_c=None,
                 rzzdr_a=None, rzzdr_b=None, rzzdr_c=None):
    r"""
    Compute a radar‑based Quantitative Precipitation Estimation (QPE) product.

    Parameters
    ----------
    ds : xarray.Dataset
        Input radar sweep containing the predictor variables (e.g. DBZ, KDP,
        ZDR, A). All predictors referenced through predictor_names must exist
        as variables in ds.
    qpe_based_on : str
        Name of the QPE relation to apply. Available estimators:
        {available_estimators}. These correspond to estimators implemented in
        the Towerpy QPE module.
    predictor_names : dict, optional
        Mapping for variable names in the dataset. Valid keys are:
        ``["dbz", "spcatt", "kdp", "zdr"]``.
        Missing keys default to ``None`` and are ignored in the QPE.
    rband : ['X', 'C', 'S'], default 'C'
        Radar frequency band used for temperature‑dependent coefficient lookup.
    temp : float, default 20.0
        Temperature (°C) used for temperature‑dependent relations.
    pol : ['H', 'V'], default 'H'
        Polarisation of the attenuation predictors. Coefficients for A‑based
        estimators are selected accordingly.
    qpe_amlb : bool, default False
        If ``False``, a mask outside the rain region is applied using either
        `ml_mask` or `ml_mask_name`. If ``True``, the estimator is evaluated
        over the entire PPI without masking.
    ml_mask : xarray.DataArray of bool, optional
        Boolean mask defining the rain region (``True`` = rain). If provided,
        this mask is used directly and overrides `ml_mask_name`.
    ml_mask_name : str, default 'ML_PCP_CLASS'
        Name of the melting‑layer classification variable in `ds`. The
        referenced variable must follow the Towerpy convention: ``1 = rain``.
        Ignored if `ml_mask` is provided.
    threshold_var_name : str, optional
        Name of the variable used for hybrid thresholding. If ``None``, the
        default threshold variable for hybrid estimators is DBZ (in dBZ).
        Thresholding is required for hybrid relations.
    thld_var_value : float, default 40.0
        Numeric threshold applied to the threshold variable for hybrid
        estimators. Ignored for non‑hybrid relations.
    out_name : str, optional
        Name of the output QPE variable. If omitted, a name of the form
        ``"R_<RELATION>"`` is used (e.g. ``"R_KDP"``).
    append_to : xarray.Dataset, optional
        Existing QPE dataset to which the new QPE variable should be appended.
        If ``None`` (default), a new dataset is created from `ds`.
    rz_a, rz_b : float, optional
        Override coefficients for the Z–R relation used in the 'z' and hybrid
        estimators. Standard values according to [1]_.
    rspcatt_a, rspcatt_b : float, optional
        Override coefficients for the A‑based estimator ('spcatt') and hybrid
        estimators involving A. Standard values according to [2]_.
    rkdp_a, rkdp_b : float, optional
        Override coefficients for the KDP‑based estimator ('kdp') and hybrid
        estimators involving KDP. Standard values according to [3]_.
    rkdpzdr_a, rkdpzdr_b, rkdpzdr_c : float, optional
        Override coefficients for the KDP–ZDR estimator ('kdp_zdr').
        Standard values according to [4]_.
    rzzdr_a, rzzdr_b, rzzdr_c : float, optional
        Override coefficients for the Z–ZDR estimator ('z_zdr').
        Standard values according to [4]_.

    Returns
    -------
    xarray.Dataset
        A dataset containing:
        - all coordinates and attributes from `ds`
        - the computed QPE variable

    Notes
    -----
    * See for instance, ``print(ds.tp.qpe_based_on_help('z'))`` for detailed
      documentation of each estimator.
    * Hybrid estimators always require a threshold variable. If
      `threshold_var_name` is None, DBZ is used by default but must be
      provided in `predictor_names`.

    
    References
    ----------
    .. [1] Marshall, J., Hitschfeld, W., & Gunn, K. (1955). Advances in radar
        weather. In Advances in geophysics (pp. 1–56).
        https://doi.org/10.1016/s0065-2687(08)60310-6
    .. [2] Ryzhkov, A., Diederich, M., Zhang, P., & Simmer, C. (2014).
        Potential utilization of specific attenuation for rainfall estimation,
        mitigation of partial beam blockage, and radar networking. Journal of
        Atmospheric and Oceanic Technology, 31(3), 599–619.
        https://doi.org/10.1175/jtech-d-13-00038.1
    .. [3] Bringi, V. N., Rico-Ramirez, M. A., & Thurai, M. (2011). Rainfall
        Estimation with an Operational Polarimetric C-Band Radar in the United
        Kingdom: Comparison with a Gauge Network and Error Analysis. Journal
        of Hydrometeorology, 12(5), 935–954.
        https://doi.org/10.1175/jhm-d-10-05013.1
    .. [4] Bringi, V. N., & Chandrasekar, V. (2001). Polarimetric doppler
        weather radar. In Cambridge University Press.
        https://doi.org/10.1017/cbo9780511541094

    Examples
    --------
    Compute a single QPE product:

    >>> rqpe = compute_rqpe(ds, qpe_based_on="kdp", predictor_names={{'kdp': 'KDP'}})
    # Append a second QPE product to the same container:
    >>> rqpe = compute_rqpe(ds, qpe_based_on="z", predictor_names={{'dbz': 'DBZH'}}, append_to=rqpe)
    """
    from ..io import modeltp as mdtp

    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # Resolve predictor DataArrays
    pred_defnames = {'dbz': None, 'spcatt': None, 'kdp': None, 'zdr': None}
    pred_names = {**pred_defnames, **(predictor_names or {})}
    predictors = {}
    for key, name in pred_names.items():
        if name is not None:
            if name not in ds:
                raise KeyError(f"predictor '{key}' refers to variable '{name}', "
                               f"which is not present in the dataset.")
            predictors[key] = ds[name]
        else:
            predictors[key] = None
    # Build ML mask
    if qpe_amlb:
        ml_mask_used = False
        ml_mask_name_used = None
        ml_mask_eff = None
    else:
        if ml_mask is not None:
            # User provided explicit boolean mask
            if ml_mask.dtype != bool:
                raise TypeError("ml_mask must be a boolean DataArray.")
            ml_mask_eff = ml_mask
            ml_mask_used = True
            ml_mask_name_used = None
        else:
            # Derive from ml_mask_name (flag variable or boolean)
            if ml_mask_name is not None and ml_mask_name in ds:
                var = ds[ml_mask_name]
                if var.dtype == bool:
                    ml_mask_eff = var
                else:
                    ml_mask_eff = (var == 1)
                ml_mask_used = True
                ml_mask_name_used = ml_mask_name
            else:
                ml_mask_eff = None
                ml_mask_used = False
                ml_mask_name_used = None
    # Threshold variable
    if threshold_var_name is None:
        threshold_var = predictors.get('dbz')   # default threshold variable
        threshold_var_used = pred_names.get('dbz')
    else:
        threshold_var = ds[threshold_var_name]
        threshold_var_used = threshold_var_name
    # Compute QPE
    r = qpe_dispatcher(qpe_based_on, pol=pol, dbz=predictors.get('dbz'),
                       spcatt=predictors.get('spcatt'),
                       kdp=predictors.get('kdp'), zdr=predictors.get('zdr'),
                       ml_mask=ml_mask_eff, threshold_var=threshold_var,
                       thld_var_value=thld_var_value, rband=rband, temp=temp,
                       rspcatt_a=rspcatt_a, rspcatt_b=rspcatt_b,
                       rkdp_a=rkdp_a, rkdp_b=rkdp_b, rz_a=rz_a, rz_b=rz_b,
                       rkdpzdr_a=rkdpzdr_a, rkdpzdr_b=rkdpzdr_b,
                       rkdpzdr_c=rkdpzdr_c, rzzdr_a=rzzdr_a,
                       rzzdr_b=rzzdr_b, rzzdr_c=rzzdr_c)
    # Determine output variable name
    if out_name is None:
        out_name = f"R_{qpe_based_on.upper()}"
    r = r.rename(out_name)
    # Determine coefficients actually used
    coeffs_used = {}
    # Single predictor
    if pol.upper() == "H":
        a_default, b_default = get_coeffs_ah(rband, temp)
    elif pol.upper() == "V":
        a_default, b_default = get_coeffs_av(rband, temp)
    else:
        raise ValueError("pol must be 'H' or 'V'")
    if qpe_based_on == "a":
            coeffs_used = {"rspcatt_a": rspcatt_a or a_default,
                           "rspcatt_b": rspcatt_b or b_default}
    elif qpe_based_on == "kdp":
        coeffs_used = {"rkdp_a": rkdp_a or get_coeffs_kdp()[0],
                       "rkdp_b": rkdp_b or get_coeffs_kdp()[1]}
    elif qpe_based_on == "z":
        coeffs_used = {"rz_a": rz_a or get_coeffs_zr()[0],
                       "rz_b": rz_b or get_coeffs_zr()[1]}
    # Dual predictor
    elif qpe_based_on == "kdp_zdr":
        if None in (rkdpzdr_a, rkdpzdr_b, rkdpzdr_c):
            rkdpzdr_a, rkdpzdr_b, rkdpzdr_c = get_coeffs_kdp_zdr()
        coeffs_used = {"rkdpzdr_a": rkdpzdr_a,
                       "rkdpzdr_b": rkdpzdr_b,
                       "rkdpzdr_c": rkdpzdr_c}
    elif qpe_based_on == "z_zdr":
        if None in (rzzdr_a, rzzdr_b, rzzdr_c):
            rzzdr_a, rzzdr_b, rzzdr_c = get_coeffs_z_zdr()
        coeffs_used = {"rzzdr_a": rzzdr_a,
                       "rzzdr_b": rzzdr_b,
                       "rzzdr_c": rzzdr_c}
    # Hybrids
    elif qpe_based_on == "z_a":
        if rz_a is None or rz_b is None:
            (rz_a, rz_b), _ = get_coeffs_z_a(rband, temp)
        coeffs_used = {"rz_a": rz_a, "rz_b": rz_b,
                       "rspcatt_a": rspcatt_a or a_default,
                       "rspcatt_b": rspcatt_b or b_default}
    elif qpe_based_on == "a_kdp":
        # ensure rkdp_a/b reflect what was actually used
        if rkdp_a is None or rkdp_b is None:
            (_, _), (rkdp_a, rkdp_b) = get_coeffs_a_kdp(rband, temp)
        coeffs_used = {"rspcatt_a": rspcatt_a or a_default,
                       "rspcatt_b": rspcatt_b or b_default,
                       "rkdp_a": rkdp_a, "rkdp_b": rkdp_b}
    elif qpe_based_on == "z_kdp":
        if rz_a is None or rz_b is None or rkdp_a is None or rkdp_b is None:
            (rz_a_default, rz_b_default), (rkdp_a_default, rkdp_b_default) = get_coeffs_z_kdp()
            if rz_a is None: rz_a = rz_a_default
            if rz_b is None: rz_b = rz_b_default
            if rkdp_a is None: rkdp_a = rkdp_a_default
            if rkdp_b is None: rkdp_b = rkdp_b_default
        coeffs_used = {"rz_a": rz_a, "rz_b": rz_b,
                       "rkdp_a": rkdp_a, "rkdp_b": rkdp_b}
    # Attach to metadata
    inputs = []
    for key, name in pred_names.items():
        if name is not None:
            if name in ds:
                inputs.append(name)
    r.attrs = sweep_vars_attrs_f.get('RAIN_RATE', {})
    r.attrs["estimator"] = f"R_{qpe_based_on.upper()}"
    r.attrs["coefficients"] = coeffs_used
    # r.attrs["long_name"] = "Rain rate from QPE estimator"
    # r.attrs["standard_name"] = "rainfall_rate"
    # r.attrs["units"] = "mm h-1"
    # r.attrs["predictors"] = list(predictors.keys())
    r.attrs["predictors"] = sorted(set(inputs or []))
    r.attrs["rband"] = rband
    # r.attrs["temperature"] = temp
    if ml_mask_used:
        if ml_mask_name_used is not None:
            r.attrs["ml_mask_variable"] = ml_mask_name_used
        else:
            r.attrs["ml_mask_variable"] = "user_supplied_boolean_mask"
    if threshold_var_used is not None:
        r.attrs["threshold_variable"] = threshold_var_used
        r.attrs["threshold_value"] = thld_var_value
    # Coefficients (only those explicitly set)
    # coeffs = {}
    # for key, val in {"rz_a": rz_a, "rz_b": rz_b,
    #                  "rspcatt_a": rspcatt_a, "rspcatt_b": rspcatt_b,
    #                  "rkdp_a": rkdp_a, "rkdp_b": rkdp_b,
    #                  "rkdpzdr_a": rkdpzdr_a, "rkdpzdr_b": rkdpzdr_b,
    #                  "rkdpzdr_c": rkdpzdr_c,
    #                  "rzzdr_a": rzzdr_a, "rzzdr_b": rzzdr_b,
    #                  "rzzdr_c": rzzdr_c,
    #                  }.items():
    #     if val is not None:
    #         coeffs[key] = val
    # if coeffs:
    #     r.attrs["coefficients"] = coeffs
    # r.attrs["coefficients"] = coeffs_used
    # Build output dataset, same coords
    if append_to is None:
        # fresh QPE-only dataset (coords only)
        out_ds = ds.coords.to_dataset().copy(deep=True)
        for coord in ds.coords:
            out_ds = out_ds.set_coords(coord)
    else:
        # append mode: copy existing QPE dataset
        out_ds = append_to.copy(deep=True)
    # Insert the new QPE variable
    out_ds[out_name] = r
    # Restore coordinate status and attributes after variable insertion
    for coord in ds.coords:
        if coord in out_ds:
            out_ds = out_ds.set_coords(coord)
            # restore attrs explicitly
            out_ds[coord].attrs = ds[coord].attrs.copy()
    # Record provenance
    #TODO: add step_description
    extra = {'step_description': ('')}
    inputs = []
    for name in [pred_names.get('dbz'), pred_names.get('spcatt'),
                 pred_names.get('kdp'), pred_names.get('zdr'),
                 ml_mask_name_used, threshold_var_used]:
        if name is not None:
            inputs.append(name)
    record_provenance(
        out_ds, step=f"qpe_based_on_{qpe_based_on}", inputs=inputs,
        outputs=[out_name],
        parameters={"estimator": f"R_{qpe_based_on.upper()}",
                    "ml_mask": ml_mask_name_used if ml_mask_used else None,
                    "threshold_variable": threshold_var_used,
                    "threshold_value": thld_var_value if threshold_var_used is not None else None,
                    "rband": rband,"temp": temp, "pol": pol,
                    "coefficients": coeffs_used}, extra_attrs=extra,
        module_provenance='towerpy.qpe.qpe_algs.compute_rqpe')
    return out_ds


def add_qpe_prod(base_ds, ds_rvars, **kwargs):
    """
    Add a QPE product to an existing QPE dataset.

    Parameters
    ----------
    base_ds : xarray.Dataset
        Existing QPE dataset to append to.
    ds_rvars : xarray.Dataset
        Dataset containing the predictor variables (Z, KDP, etc.).
    **kwargs :
        Passed directly to compute_rqpe.

    Returns
    -------
    xarray.Dataset
        Updated QPE dataset with the new product appended.
    """
    return compute_rqpe(
        ds_rvars,
        append_to=base_ds,
        **kwargs)



def _finalise_compute_rqpe_doc():
    available = ", ".join(sorted(_QPE_DOCS.keys()))
    compute_rqpe.__doc__ = compute_rqpe.__doc__.format(
        available_estimators=available)

_finalise_compute_rqpe_doc()
