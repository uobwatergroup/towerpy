"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np
from ..utils.radutilities import find_nearest


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

    def z_to_r(self, zh, a=200, b=1.6, beam_height=None, mlyr=None):
        r"""
        Calculate the rain rate (R) from Zh (mm^6/m^3).

        Parameters
        ----------
        zh : float or array
             Floats that corresponds to reflectivity, in dBZ.
        a, b : float
            Parameters of the :math:`R(Z_H)` relationship
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : class, optional
            Melting layer class containing the top and bottom boundaries of the
            ML. Only gates below the melting layer bottom (i.e. the rain region
            below the melting layer) are included in the correction. If None,
            the default values of the melting level and the thickness of the
            melting layer are set to 5 and 0.5, respectively.

        Returns
        -------
        r_z : dict
            Computed rain rates (in mm h^-1).

        Notes
        -----
        .. math::  Z = aR^b
        Standard values according to [1]_.

        References
        ----------
        .. [1] Marshall, J.S., Palmer, W.M.K., 1948. "The distribution of
            raindrops with size. Journal of Meteorology 5, 165–166.
            https://doi.org/10.1175/1520-0469(1948)005<0165:TDORWS>2.0.CO;2

        """
        zh = np.array(zh)
        if mlyr is None:
            mlvl = 5.
            mlyr_thickness = 0.5
            # mlyr_bottom = mlvl - mlyr_thickness
        else:
            mlvl = mlyr.ml_top
            mlyr_thickness = mlyr.ml_thickness
            # mlyr_bottom = mlyr.ml_bottom
        if beam_height is not None:
            mlidx = find_nearest(beam_height, mlvl-mlyr_thickness)
            nanidx = np.where(np.isnan(zh))
            zh[:, mlidx:] = 0
            zh[nanidx] = np.nan
        r = {'Rainfall [mm/hr]': ((10**(zh/10))/a)**(1/b)}
        self.r_z = r

    def z_zdr_to_r1(self, zh, zdr, a=0.01583, b=0.8349, c=-0.3732,
                    beam_height=None, mlyr=None):
        """
        Calculate the rain rate (R) from Zh (mm^6/m^3) and ZDR (dB).

        Parameters
        ----------
        zh : float or array
            Floats that corresponds to reflectivity, in dBZ.
        zdr : float or array
            Floats that corresponds to differential reflectivit, in dB.
        a, b, c : floats
            Parameters of the :math:`R(Z_H, Z_{DR})` relationship
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : class, optional
            Melting layer class containing the top and bottom boundaries of the
            ML. Only gates below the melting layer bottom (i.e. the rain region
            below the melting layer) are included in the correction. If None,
            the default values of the melting level and the thickness of the
            melting layer are set to 5 and 0.5, respectively.

        Returns
        -------
        r_z_zdr : dict
            Computed rain rates (in mm h^-1).

        Notes
        -----
        .. math::  R = aZ_h^b10^{cZ_{DR}}
        where:
        R in mm hr^{-1}, Zh in mm^6 m^{-3} and ZDR in dB.

        Standard values according to [1]_.

        References
        ----------
        .. [1] Rico-ramirez, M.A., Cluckie, I.D., 2006. "Assessment of
            polarimetric rain rate algorithms at C-band frequencies" in:
            Fourth European Conference on Radar in Meteorology and
            Hydrology, Universitat Politecnica de Catalunya, Barcelona,
            Spain. pp. 221 – 224.
        """
        zh = 10**(np.array(zh)/10)
        zdr = np.array(zdr)
        if mlyr is None:
            mlvl = 5.
            mlyr_thickness = 0.5
            # mlyr_bottom = mlvl - mlyr_thickness
        else:
            mlvl = mlyr.ml_top
            mlyr_thickness = mlyr.ml_thickness
            # mlyr_bottom = mlyr.ml_bottom
        if beam_height is not None:
            mlidx = find_nearest(beam_height, mlvl-mlyr_thickness)
            nanidx = np.where(np.isnan(zh))
            zh[:, mlidx:] = 0
            zh[nanidx] = np.nan
            # zdr[:, mlidx:] = 0
            # zdr[nanidx] = np.nan
        r = {'Rainfall [mm/hr]': a*zh**b*10**(c*zdr)}
        self.r_z_zdr = r

    def z_zdr_to_r2(self, zh, zdr, a=0.00403, b=0.8787, c=-0.8077,
                    beam_height=None, mlyr=None):
        """
        Calculate the rain rate (R) from Zh (mm^6/m^3) and Zdr (linear scale).

        Parameters
        ----------
        zh : float or array
            Floats that corresponds to reflectivity, in dBZ.
        zdr : float or array
            Floats that corresponds to differential reflectivity, in dB.
        a, b, c : floats
            Parameters of the :math:`R(Z_H, Z_{dr})` relationship
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : class, optional
            Melting layer class containing the top and bottom boundaries of the
            ML. Only gates below the melting layer bottom (i.e. the rain region
            below the melting layer) are included in the correction. If None,
            the default values of the melting level and the thickness of the
            melting layer are set to 5 and 0.5, respectively.

        Returns
        -------
        r_z_zdrl : dict
            Computed rain rates (in mm h^-1).

        Notes
        -----
        .. math::  R = aZ_h^bZ_{dr}^c
        where:
        R in mm hr^{-1}, Zh in mm^6 m^{-3}, Zdr in linear scale.

        Standard values according to [1]_.

        References
        ----------
        .. [1] Rico-ramirez, M.A., Cluckie, I.D., 2006.
            Assessment of polarimetric rain rate algorithms at C-band
            frequencies. In: Fourth European Conference on Radar in Meteorology
            and Hydrology, Universitat Politecnica de Catalunya, Barcelona,
            Spain. pp. 221 – 224.

        """
        zh = 10**(np.array(zh)/10)
        zdr = np.array(10**(0.1*zdr))
        if mlyr is None:
            mlvl = 5.
            mlyr_thickness = 0.5
            # mlyr_bottom = mlvl - mlyr_thickness
        else:
            mlvl = mlyr.ml_top
            mlyr_thickness = mlyr.ml_thickness
            # mlyr_bottom = mlyr.ml_bottom
        if beam_height is not None:
            mlidx = find_nearest(beam_height, mlvl-mlyr_thickness)
            nanidx = np.where(np.isnan(zh))
            zh[:, mlidx:] = 0
            zh[nanidx] = np.nan
        r = {'Rainfall [mm/hr]': a*zh**b*zdr**c}
        self.r_z_zdrl = r

    def kdp_to_r(self, kdp, a=24.68, b=0.81, beam_height=None, mlyr=None):
        """
        Calculate the rain rate (R) from KDP (deg km^-1).

        Parameters
        ----------
        kdp : float or array
            Floats that corresponds to specific differential phase,
            in (deg km^-1).
        a, b : floats
            Parameters of the :math:`R(K_{DP})` relationship
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : class, optional
            Melting layer class containing the top and bottom boundaries of the
            ML. Only gates below the melting layer bottom (i.e. the rain region
            below the melting layer) are included in the correction. If None,
            the default values of the melting level and the thickness of the
            melting layer are set to 5 and 0.5, respectively.

        Returns
        -------
        R : dict
            Computed rain rates (in mm h^-1).

        Notes
        -----
        .. math::  R = aK_{DP}^b
        where:
        R in mm hr^{-1} and KDP in deg km^-1.

        Standard values according to [1]_.

        References
        ----------
        .. [1] Bringi, V.N., Rico-Ramirez, M.A., Thurai, M. (2011). "Rainfall
            estimation with an operational polarimetric C-band radar in the
            United Kingdom: Comparison with a gauge network and error
            analysis" Journal of Hydrometeorology 12, 935–954.
            https://doi.org/10.1175/JHM-D-10-05013.1
        """
        kdp = np.array(kdp)
        if mlyr is None:
            mlvl = 5.
            mlyr_thickness = 0.5
            # mlyr_bottom = mlvl - mlyr_thickness
        else:
            mlvl = mlyr.ml_top
            mlyr_thickness = mlyr.ml_thickness
            # mlyr_bottom = mlyr.ml_bottom
        if beam_height is not None:
            mlidx = find_nearest(beam_height, mlvl-mlyr_thickness)
            nanidx = np.where(np.isnan(kdp))
            kdp[:, mlidx:] = 0
            kdp[nanidx] = np.nan
        r = {'Rainfall [mm/hr]': a*abs(kdp)**b*np.sign(kdp)}
        self.r_kdp = r

    def kdp_zdr_to_r(self, kdp, zdr, a=37.9, b=-0.72, c=0.89, beam_height=None,
                     mlyr=None):
        """
        Calculate the rain rate (R) from KDP (deg km^-1) and ZDR (dB).

        Parameters
        ----------
        kdp : float or array
            Floats that corresponds to specific differential phase,
            in deg km^-1.
        zdr : float or array
            Floats that corresponds to differential reflectivity, in dB.
        a, b, c : floats
            Parameters of the :math:`R(K_{DP})` relationship
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : class, optional
            Melting layer class containing the top and bottom boundaries of the
            ML. Only gates below the melting layer bottom (i.e. the rain region
            below the melting layer) are included in the correction. If None,
            the default values of the melting level and the thickness of the
            melting layer are set to 5 and 0.5, respectively.

        Returns
        -------
        R : dict
            Computed rain rates (in mm h^-1).

        Notes
        -----
        .. math::  R = aK_{DP}^b10^{-0.1cZ_{DR}}
        where:
        R in mm hr^{-1}, KDP in deg km^-1 and ZDR in dB.

        Standard values according to [1]_

        References
        ----------
        .. [1] Bringi, V.N., Chandrasekar, V., (2001). "Polarimetric Doppler
            Weather Radar" Cambridge University Press, Cambridge ; New York.
            https://doi.org/10.1017/CBO9780511541094

        """
        kdp = np.array(kdp)
        zdr = np.array(zdr)
        if mlyr is None:
            mlvl = 5.
            mlyr_thickness = 0.5
            # mlyr_bottom = mlvl - mlyr_thickness
        else:
            mlvl = mlyr.ml_top
            mlyr_thickness = mlyr.ml_thickness
            # mlyr_bottom = mlyr.ml_bottom
        if beam_height is not None:
            mlidx = find_nearest(beam_height, mlvl-mlyr_thickness)
            nanidx = np.where(np.isnan(kdp))
            kdp[:, mlidx:] = 0
            kdp[nanidx] = np.nan
        r = {'Rainfall [mm/hr]': a*kdp**b*10**(-0.1*c*zdr)}
        self.r_kdp_zdr = r

    def ah_to_r(self, ah, a=294, b=0.89, beam_height=None, mlyr=None):
        """
        Calculate the rain rate (R) from AH (in dB km^-1).

        Parameters
        ----------
        ah : float or array
            Floats that corresponds to specific attenuation, in dB/km.
        a, b : floats
            Parameters of the :math:`R(A_{H})` relationship
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : class, optional
            Melting layer class containing the top and bottom boundaries of the
            ML. Only gates below the melting layer bottom (i.e. the rain region
            below the melting layer) are included in the correction. If None,
            the default values of the melting level and the thickness of the
            melting layer are set to 5 and 0.5, respectively.

        Returns
        -------
        R : dict
            Computed rain rates (in mm h^-1).

        Notes
        -----
        .. math::  R = aA_H^b
        where
        R in mm hr^{-1}, AH in dBkm-1

        Standard values according to [1]_.

        References
        ----------
        .. [1] Ryzhkov, A., Diederich, M., Zhang, P., & Simmer, C. (2014).
            "Potential Utilization of Specific Attenuation for Rainfall
            Estimation, Mitigation of Partial Beam Blockage, and Radar
            Networking" Journal of Atmospheric and Oceanic Technology, 31(3),
            599-619. https://doi.org/10.1175/JTECH-D-13-00038.1

        """
        ah = np.array(ah)
        if mlyr is None:
            mlvl = 5.
            mlyr_thickness = 0.5
            # mlyr_bottom = mlvl - mlyr_thickness
        else:
            mlvl = mlyr.ml_top
            mlyr_thickness = mlyr.ml_thickness
            # mlyr_bottom = mlyr.ml_bottom
        if beam_height is not None:
            mlidx = find_nearest(beam_height, mlvl-mlyr_thickness)
            nanidx = np.where(np.isnan(ah))
            ah[:, mlidx:] = 0
            ah[nanidx] = np.nan
        r = {'Rainfall [mm/hr]': a*ah**b}
        self.r_ah = r

    def z_kdp_to_r(self, zh, kdp, a1=200, b1=1.6, a2=24.68, b2=0.81,
                   z_thld=40, beam_height=None, mlyr=None):
        r"""
        Calculate the rain rate (R) using an hybrid estimator Z-AH.

        Parameters
        ----------
        zh : float or array
             Floats that corresponds to reflectivity, in dBZ.
        kdp : float or array
            Floats that corresponds to specific differential phase,
            in deg km^-1.
        a1, b1 : float
            Parameters of the :math:`R(Z_H)` relationship.
        a2, b2 : floats
            Parameters of the :math:`R(K_{DP})` relationship.
        z_thld : float, optional
            :math:`Z_H` threshold used for the transition to :math:`K_{DP}`.
            The default is 40.
        beam_height : array, optional
            Height of the centre of the radar beam, in km.
        mlyr : class, optional
            Melting layer class containing the top and bottom boundaries of the
            ML. Only gates below the melting layer bottom (i.e. the rain region
            below the melting layer) are included in the correction. If None,
            the default values of the melting level and the thickness of the
            melting layer are set to 5 and 0.5, respectively.

        Returns
        -------
        R : dict
            Computed rain rates (in mm h^-1).

        Notes
        -----
        Standard values according to [1]_ and [2]_.

        .. math::  Z = aR^b \rightarrow Z_H <= 40 dBZ
        .. math::  R = aK_{DP}^b \rightarrow Z_H > 40 dBZ

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
        kdp = np.array(kdp)
        if mlyr is None:
            mlvl = 5.
            mlyr_thickness = 0.5
            # mlyr_bottom = mlvl - mlyr_thickness
        else:
            mlvl = mlyr.ml_top
            mlyr_thickness = mlyr.ml_thickness
            # mlyr_bottom = mlyr.ml_bottom
        if beam_height is not None:
            mlidx = find_nearest(beam_height, mlvl-mlyr_thickness)
            nanidx = np.where(np.isnan(zh))
            zh[:, mlidx:] = 0
            zh[nanidx] = np.nan
        rzh = ((10**(zh/10))/a1)**(1/b1)
        rkdp = a2*abs(kdp)**b2*np.sign(kdp)
        rzh[(zh >= z_thld)] = rkdp[(zh >= z_thld)]
        r = {'Rainfall [mm/hr]': rzh}
        self.r_z_kdp = r
