"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np
from scipy.interpolate import interp1d
from ..utils import radutilities as rut
from ..utils import unit_conversion as tpuc


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

    def __init__(self, radobj):
        self.elev_angle = radobj.elev_angle
        self.file_name = radobj.file_name
        self.scandatetime = radobj.scandatetime
        self.site_name = radobj.site_name

    def adp_to_r(self, adp, rband='C', temp=20., a=None, b=None, mlyr=None,
                 beam_height=None):
        r"""
        Compute rain rates using the :math:`R(A_{DP})` estimator [Eq.1]_.

        Parameters
        ----------
        adp : float or array
            Floats that corresponds to the differential attenuation, in dB/km.
        rband: str
            Frequency band according to the wavelength of the radar.
            The string has to be one of 'S', 'C' or 'X'. The default is 'C'.
        temp: float
            Temperature, in :math:`^{\circ}C`, used to derive the coefficients
            a, b according to [1]_. The default is 20.
        a, b : floats
            Override the default coefficients of the :math:`R(A_{H})`
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
        .. math::  R = aA_{DP}^b
        where R in mm/h, ADP in dB/km

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
            coeffs_a = {'X': np.array((57.8, 53.3, 51.1, 51.0)),
                        'C': np.array((281, 326, 393, 483)),
                        'S': np.array((3.02e3, 4.12e3, 5.51e3, 7.19e3))}
            coeffs_b = {'X': np.array((0.89, 0.85, 0.81, 0.78)),
                        'C': np.array((0.95, 0.94, 0.93, 0.93)),
                        'S': np.array((1.06, 1.06, 1.06, 1.06))}
            # Interpolate the temp and coeffs to set coeffs a and b
            icoeff_a = interp1d(temps, coeffs_a.get(rband))
            icoeff_b = interp1d(temps, coeffs_b.get(rband))
            coeff_a = icoeff_a(temp)
            coeff_b = icoeff_b(temp)
        else:
            coeff_a = a
            coeff_b = b
        # print(f'{coeff_a}, {coeff_b}')
        adpr = np.zeros_like(adp)+adp
        if mlyr is not None and beam_height is not None:
            mlyr_bottom = mlyr.ml_top - mlyr.ml_thickness
            if isinstance(mlyr_bottom, (int, float)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom)
                           for nbh in beam_height]
            elif isinstance(mlyr_bottom, (np.ndarray, list, tuple)):
                mlb_idx = [rut.find_nearest(nbh, mlyr_bottom[cnt])
                           for cnt, nbh in enumerate(beam_height)]
            for cnt, azi in enumerate(adpr):
                azi[mlb_idx[cnt]:] = 0
        nanidx = np.where(np.isnan(adp))
        adpr[nanidx] = np.nan
        r = {'Rainfall [mm/h]': coeff_a*adp**coeff_b}
        r['coeff_a'] = coeff_a.item()
        r['coeff_b'] = coeff_b.item()
        self.r_adp = r

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
            Override the default coefficients of the :math:`R(A_{H})`
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
            coeff_a = icoeff_a(temp)
            coeff_b = icoeff_b(temp)
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
        r['coeff_a'] = coeff_a.item()
        r['coeff_b'] = coeff_b.item()
        self.r_ah = r

    def kdp_to_r(self, kdp, a=24.68, b=0.81, beam_height=None, mlyr=None):
        r"""
        Compute rain rates using the :math:`R(K_{DP})` estimator [Eq.1]_.

        Parameters
        ----------
        kdp : float or array
            Floats that corresponds to specific differential phase,
            in deg/km.
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
        .. [1] Marshall, J.S., Palmer, W.M.K., 1948. "The distribution of
            raindrops with size. Journal of Meteorology 5, 165–166.
            https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2

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
            Parameters of the :math:`R(A_{H})` relationship.
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
        .. [1] Marshall, J.S., Palmer, W.M.K., 1948. "The distribution of
            raindrops with size. Journal of Meteorology 5, 165–166.
            https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2
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
            coeff_a = icoeff_a(temp)
            coeff_b = icoeff_b(temp)
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
        .. [1] Marshall, J.S., Palmer, W.M.K., 1948. "The distribution of
            raindrops with size. Journal of Meteorology 5, 165–166.
            https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2
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
