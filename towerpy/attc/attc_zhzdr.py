"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

from pathlib import Path
import ctypes as ctp
import platform
import time
import numpy as np
import numpy.ctypeslib as npct
from scipy import optimize
from ..base import TowerpyError
from ..utils.radutilities import find_nearest
from ..datavis import rad_display
from ..ml.mlyr import MeltingLayer


class AttenuationCorrection:
    r"""
    A class to calculate the attenuation of the radar signal power.

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
    res_zdrcorrection : dict, opt
        Descriptor of the :math:`Z_{DR}` attenuation correction process.
    vars : dict
        Output of the :math:`Z_{H}` and/or :math:`Z_{DR}` attenuation
        correction process.
    """

    def __init__(self, radobj):
        self.elev_angle = radobj.elev_angle
        self.file_name = radobj.file_name
        self.scandatetime = radobj.scandatetime
        self.site_name = radobj.site_name

    def zh_correction(self, rad_georef, rad_params, attvars, cclass, mlyr=None,
                      attc_method='ABRI', pdp_pxavr_rng=7, pdp_pxavr_azm=1,
                      pdp_dmin=20, coeff_a=[1e-5, 9e-5, 3e-5],
                      coeff_b=[0.65, 0.85, 0.78],
                      coeff_alpha=[0.020, 0.1, 0.073], niter=500,
                      plot_method=False):
        r"""
        Calculate the attenuation of :math:`Z_{H}`.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        attvars : dict
            Polarimetric variables used for the attenuation correction.
        cclass : array
            Clutter, noise and meteorological classification.
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness, in km. Only gates below the
            melting layer bottom (i.e. the rain region below the melting layer)
            are included in the computation; ml_top and ml_thickness can be
            either a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            function is applied to the whole PPI scan, assuming a ml_thickness
            of 0.5 km.
        attc_method : str
            Attenuation correction algorithm to be used. The default is 'ABRI':

                [ABRI] = Bringi (optimised).

                [AFV] = Final value (optimised).

                [AHB] = Hitschfeld and Bordan (optimised).

                [ZPHI] = Testud (constant parameters).

                [BRI] = Bringi (constant parameters).

                [FV] = Final value (constant parameters).

                [HB] = Hitschfeld and Bordan (constant parameters).
        pdp_pxavr_rng : int
            Pixels to average in :math:`\Phi_{DP}` along range: Odd number
            equivalent to about 4km, i.e. 4km/range_resolution. The default
            is 7.
        pdp_pxavr_azm : int
            Pixels to average in :math:`\Phi_{DP}` along azimuth. Must be an
            odd number. The default is 1.
        pdp_dmin : float
            Minimum total :math:`\Phi_{DP}` expected in a ray to perform
            attenuation correction (at least 20-30 degrees). The default is 20.
        coeff_a : 3-element tuple or list, optional
            [Min, max, fixed value] of coeff :math:`a`. These bounds are
            used to find the optimum value of :math:`a` from
            :math:`A_H = a Z_{H}^b`. Default values are [1e-5, 9e-5, 3e-5],
            derived for C-band.
        coeff_b : 3-element tuple or list, optional
            [Min, max, fixed value] of coeff :math:`b`. These bounds are used
            to find the optimum value of :math:`b` from
            :math:`A_H = a Z_{H}^b`. Default values are [0.65, 0.85, 0.78],
            derived for C-band.
        coeff_alpha : 3-element tuple or list, optional
            [Min, max, fixed value] of coeff :math:`\alpha`. These bounds are
            used to find the optimum value of :math:`\alpha` from
            :math:`A_H = \alpha K_{DP}`. Default values are
            [0.020, 0.1, 0.073], derived for C-band.
        niter : int
            Number of iterations to find the optimised values of
            the coeffs :math:`a, b, \alpha`. The default is 500.
        plot_method : Bool, optional
            Plot the ZH attenuation correction method. The default is False.

        Returns
        -------
        vars : dict
            ZH [dBZ]:
                Corrected reflectivity.
            AH [dB/km]:
                Specific attenuation
            PhiDP [deg]:
                Measured differential phase, but :math:`\Phi_{DP}(r0)`
                is adjusted.
            PhiDP* [deg]:
                Computed differential phase according to Bringi's method.
            alpha:
                parameter :math:`\alpha` optimised for each beam.

        Notes
        -----
        1. The attenuation is computed up to a user-defined melting level
        height.

        2. This function uses the shared object 'lnxlibattenuationcorrection'
        or the dynamic link library 'w64libattenuationcorrection' depending on
        the operating system (OS).

        3. Based on the method described in [1]_

        References
        ----------
        .. [1] M. A. Rico-Ramirez, "Adaptive Attenuation Correction Techniques
            for C-Band Polarimetric Weather Radars," in IEEE Transactions on
            Geoscience and Remote Sensing, vol. 50, no. 12, pp. 5061-5071,
            Dec. 2012. https://doi.org/10.1109/TGRS.2012.2195228
        """
        tic = time.time()

        array1d = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array2d = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
        if platform.system() == 'Linux':
            libac = npct.load_library("lnxlibattenuationcorrection.so",
                                      Path(__file__).parent.absolute())
        elif platform.system() == 'Windows':
            libac = ctp.cdll.LoadLibrary(f'{Path(__file__).parent.absolute()}'
                                         '/w64libattenuationcorrection.dll')
        else:
            libac = None
            raise TowerpyError(f'The {platform.system} OS is not currently'
                               'compatible with this version of Towerpy')
        libac.attenuationcorrection.restype = None
        libac.attenuationcorrection.argtypes = [ctp.c_int, ctp.c_int, array2d,
                                                array2d, array2d, array2d,
                                                array2d, array1d, array1d,
                                                array1d, array1d, array2d,
                                                array2d, array2d, array2d,
                                                array2d]
        if mlyr is None:
            mlyr = MeltingLayer(self)
            mlyr.ml_top = 5
            mlyr.ml_thickness = 0.5
            mlyr.ml_bottom = mlyr.ml_top - mlyr.ml_thickness
        else:
            mlyr.ml_bottom = mlyr.ml_top - mlyr.ml_thickness

        if isinstance(mlyr.ml_top, (int, float)):
            mlgrid = np.zeros_like(
                attvars['ZH [dBZ]']) + (mlyr.ml_top) * 1000
        elif isinstance(mlyr.ml_top, (np.ndarray, list, tuple)):
            mlgrid = (np.ones_like(
                attvars['ZH [dBZ]'].T) * mlyr.ml_top * 1000).T
        param_atc = np.zeros(15)
        nrays = len(rad_georef['azim [rad]'])
        ngates = len(rad_georef['range [m]'])

        if attc_method == 'ABRI':
            param_atc[0] = 0
        elif attc_method == 'AFV':
            param_atc[0] = 1
        elif attc_method == 'AHB':
            param_atc[0] = 2
        elif attc_method == 'ZPHI':
            param_atc[0] = 3
        elif attc_method == 'BRI':
            param_atc[0] = 4
        elif attc_method == 'FV':
            param_atc[0] = 5
        elif attc_method == 'HB':
            param_atc[0] = 6
        else:
            raise TowerpyError('Please select a valid attenuation correction'
                               ' method')
        param_atc[1] = pdp_pxavr_rng
        param_atc[2] = pdp_pxavr_azm
        param_atc[3] = pdp_dmin
        param_atc[4] = coeff_a[2]  # a_opt
        param_atc[5] = coeff_b[2]  # b_opt
        param_atc[6] = coeff_alpha[2]  # alpha_opt
        param_atc[7] = coeff_a[0]  # mina
        param_atc[8] = coeff_a[1]  # maxa
        param_atc[9] = coeff_b[0]  # minb
        param_atc[10] = coeff_b[1]  # maxb
        param_atc[11] = coeff_alpha[0]  # minalpha
        param_atc[12] = coeff_alpha[1]  # maxalpha
        param_atc[13] = niter  # number of iterations
        param_atc[14] = mlyr.ml_thickness * 1000  # BB thickness in meters
        zhh_Ac = np.full(attvars['ZH [dBZ]'].shape, np.nan)
        Ah = np.full(attvars['ZH [dBZ]'].shape, np.nan)
        phidp_m = np.full(attvars['ZH [dBZ]'].shape, np.nan)
        phidp_c = np.full(attvars['ZH [dBZ]'].shape, np.nan)
        alpha = np.full(attvars['ZH [dBZ]'].shape, np.nan)
        libac.attenuationcorrection(
            nrays, ngates, attvars['ZH [dBZ]'], attvars['PhiDP [deg]'],
            attvars['rhoHV [-]'], mlgrid, cclass, rad_georef['range [m]'],
            rad_georef['azim [rad]'], rad_georef['elev [rad]'],
            param_atc, zhh_Ac, Ah, phidp_m, phidp_c, alpha)
        attcorr = {'ZH [dBZ]': zhh_Ac, 'AH [dB/km]': Ah, 'alpha [-]': alpha,
                   'PhiDP [deg]': phidp_m, 'PhiDP* [deg]': phidp_c}
        attcorr['PIA [dB]'] = attcorr['PhiDP* [deg]'] * attcorr['alpha [-]']
        attcorr['KDP [deg/km]'] = np.nan_to_num(
            attcorr['AH [dB/km]'] / attcorr['alpha [-]'])

        alphacopy = np.zeros_like(attcorr['alpha [-]']) + attcorr['alpha [-]']
        for i in range(nrays):
            idmx = np.nancumsum(alphacopy[i]).argmax()
            if idmx != 0:
                attcorr['PIA [dB]'][i][idmx+1:] = attcorr['PIA [dB]'][i][idmx]

        # =====================================================================
        # Filter non met values
        # =====================================================================
        for key, values in attcorr.items():
            values[cclass != 0] = np.nan

        self.vars = attcorr
        toc = time.time()
        print(r'Z_H attenuation correction running time: '
              f'{toc-tic:.3f} sec.')
        if plot_method:
            rad_display.plot_zhattcorr(rad_georef, rad_params, attvars,
                                       attcorr)

    def zdr_correction(self, rad_georef, rad_params, attvars, attcorr_vars,
                       cclass, mlyr=None, coeff_beta=[0.002, 0.04, 0.02],
                       rhv_thld=0.98, mov_avrgf_len=5, minbins=10,
                       p2avrf=3, beta_alpha_ratio=0.265, method='linear',
                       rparams=None, descr=False, plot_method=False):
        r"""
        Calculate the attenuation of :math:`Z_{DR}`.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height.
        rad_params : dict
            Radar technical details.
        attvars : dict
            Polarimetric variables used the for attenuation correction.
        attcorr_vars : dict
            Radar object containing  attenuation corrected polarimetric
            variables used for calculations.
        cclass : array
            Clutter and meteorological classification.
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness, in km. Only gates below the
            melting layer bottom (i.e. the rain region below the melting layer)
            are included in the computation; ml_top and ml_thickness can be
            either a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            function is applied to the whole PPI scan, assuming a ml_thickness
            of 0.5 km.
        rhv_thld : float
            Minimum value of :math:`\rho_{HV}` expected in the rain medium.
            The default is 0.98.
        minbins : int
            Minimum number of bins related to the length of each rain cell
            along the beam. The default is 10.
        mov_avrgf_len : int
            Odd number used to apply a moving average filter to each beam and
            smooth the signal. The default is 5.
        p2avrf : int
            Number of bins to average on the far side of the rain cell.
            The default is 3.
        beta_alpha_ratio : float, opt
            Quotient between :math:`\alpha` and :math:`\beta` parameters from
            :math:`A_{DP} = ( \beta / \alpha )A_{H}`
        method : str
            Method used to compute the ZH-ZDR relationship. The "linear" model
            uses the relation provided in [2]_ and [4]_, whereas the "exp"
            model uses the relation proposed by [3]_.
        params: dict
            Additional parameters describing the ZH-ZDR relationship.

            .. math::
                Z_{DR} = \Biggl\{ 0 \rightarrow Z_H(r_m)<=20 dBZ \\
                         0.048*Z_H-0.774 \rightarrow 20<Z_H(r_m)<=45 dBZ \\
                         1.4 \rightarrow Z_H(r_m)>45 dBZ \Biggl\}
        descr : bool
            Controls if the statistics of the calculations are returned.
            The default is True.
        plot_method : Bool, optional
            Plot the ZDR attenuation correction method. The default is False.

        Returns
        -------
         vars : dict
            ZDR [dB]:
                Attenuation-corrected differential reflectivity.
            ADP [dB/km]':
                Specific differential attenuation.
            beta:
               parameter :math:`\beta` optimised for a given beam.

        Notes
        -----
        1. The attenuation is computed up to a user-defined melting level
        height.

        2. Based on the method described in [1]_

        References
        ----------
        .. [1] M. A. Rico-Ramirez, "Adaptive Attenuation Correction Techniques
            for C-Band Polarimetric Weather Radars," in IEEE Transactions on
            Geoscience and Remote Sensing, vol. 50, no. 12, pp. 5061-5071,
            Dec. 2012. https://doi.org/10.1109/TGRS.2012.2195228

        .. [2] V. N. Bringi, T. D. Keenan and V. Chandrasekar, "Correcting
            C-band radar reflectivity and differential reflectivity data for
            rain attenuation: a self-consistent method with constraints,"
            in IEEE Transactions on Geoscience and Remote Sensing, vol. 39,
            no. 9, pp. 1906-1915, Sept. 2001, https://doi.org/10.1109/36.951081

        .. [3] Gou, Y., Chen, H. and Zheng, J., 2019. "An improved
            self-consistent approach to attenuation correction for C-band
            polarimetric radar measurements and its impact on quantitative
            precipitation estimation", in Atmospheric Research, 226, pp.32-48.
            https://doi.org/10.1016/j.atmosres.2019.03.006

        .. [4] Park, S., Bringi, V. N., Chandrasekar, V., Maki, M.,
            & Iwanami, K. (2005). Correction of Radar Reflectivity and
            Differential Reflectivity for Rain Attenuation at X Band. Part I:
            Theoretical and Empirical Basis, Journal of Atmospheric and Oceanic
            Technology, 22(11), 1621-1632. https://doi.org/10.1175/JTECH1803.1

        """
        tic = time.time()
        params = {'ZH-ZDR relation': method, 'ZH_lower_lim': 20,
                  'ZH_upper_lim': 45, 'model': 'a1*ZH-b1', 'zdr_max': 1.4,
                  'a1': 0.048, 'b1': 0.774}
        if method == 'exp':
            params['ZH-ZDR relation'] = method
            params['model'] = 'a1*ZH^b1'
            params['a1'] = 0.00012
            params['b1'] = 2.5515
        if rparams is not None:
            params.update
        # if mlyr is None:
        #     mlyr = MeltingLayer(self)
        #     mlvl = 5
        #     mlyr_thickness = 0.5
        #     # mlyr_bottom = mlvl - mlyr_thickness
        # else:
        #     mlvl = mlyr.ml_top
        #     mlyr_thickness = mlyr.ml_thickness
        #     mlyr.ml_bottom = mlvl - mlyr_thickness
        if mlyr is None:
            mlyr = MeltingLayer(self)
            mlyr.ml_top = 5
            mlyr.ml_thickness = 0.5
            mlyr.ml_bottom = mlyr.ml_top - mlyr.ml_thickness
        else:
            mlyr.ml_bottom = mlyr.ml_top - mlyr.ml_thickness

        if isinstance(mlyr.ml_top, (int, float)):
            mlt_idx = [find_nearest(nbh, mlyr.ml_top)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_top, (np.ndarray, list, tuple)):
            mlt_idx = [find_nearest(nbh, mlyr.ml_top[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        if isinstance(mlyr.ml_bottom, (int, float)):
            mlb_idx = [find_nearest(nbh, mlyr.ml_bottom)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_bottom, (np.ndarray, list, tuple)):
            mlb_idx = [find_nearest(nbh, mlyr.ml_bottom[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        # mlt_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
        #                      for cnt, ix in enumerate(mlt_idx)])
        # mlt_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
        #                      for cnt, ix in enumerate(mlt_idx)])
        # mlb_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
        #                      for cnt, ix in enumerate(mlb_idx)])
        # mlb_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
        #                      for cnt, ix in enumerate(mlb_idx)])
        
        
        nrays = len(rad_georef['azim [rad]'])
        # TODO fix this so nonisotropic ML are an inpyt.
        # idxlimfl = [find_nearest(rad_georef['beam_height [km]'][i, :],
        #             mlvl-mlyr_thickness) for i in range(nrays)]
        m = np.zeros_like(attcorr_vars['ZH [dBZ]'])
        for n, rows in enumerate(m):
            rows[mlb_idx[n]:] = np.nan
        m[cclass != 0] = np.nan
        m[:, 0] = np.nan

        attcorrmask = {keys: np.ma.masked_array(values, m)
                       for keys, values in attcorr_vars.items()}
        attcorrmask['rhoHV [-]'] = np.ma.masked_array(attvars['rhoHV [-]'], m)
        attcorrmask['ZDR [dB]'] = np.ma.masked_array(attvars['ZDR [dB]'], m)

        idxzdrattcorr = [i for i in range(nrays)
                         if any(attcorr_vars['alpha [-]'][i, :] > 0)]

        zdrattcorr = []
        adpif = []
        betaof = []
        zdrstat = []

        alphacopy = (np.zeros_like(attcorr_vars['alpha [-]']
                                   + attcorr_vars['alpha [-]']))

        for i in range(nrays):
            if i in idxzdrattcorr:
                idmx = np.nancumsum(alphacopy[i]).argmax()
                if idmx != 0:
                    alphacopy[i][idmx+1:] = alphacopy[i][idmx]
                zdr_ibeam = (np.ones_like(attcorrmask['ZDR [dB]'][i, :])
                             * attcorrmask['ZDR [dB]'][i, :])
                zh_ibeam = (np.ones_like(attcorrmask['ZH [dBZ]'][i, :])
                            * attcorrmask['ZH [dBZ]'][i, :])

                zdr_fltr_rhv = np.ma.masked_where(
                    attcorrmask['rhoHV [-]'][i, :] < rhv_thld, zdr_ibeam)
                zh_fltr_rhv = np.ma.masked_where(
                    attcorrmask['rhoHV [-]'][i, :] < rhv_thld, zh_ibeam)

                zdr_mvavfl = (np.ma.convolve(zdr_fltr_rhv,
                                             np.ones(mov_avrgf_len)
                                             / mov_avrgf_len, mode='same'))
                zh_mvavfl = (np.ma.convolve(zh_fltr_rhv,
                                            np.ones(mov_avrgf_len)
                                            / mov_avrgf_len, mode='same'))

                rcells_idx = np.array(
                    sorted(
                        np.vstack((np.argwhere(
                            np.diff(np.isnan(zdr_mvavfl).mask)),
                            np.argwhere(np.diff(np.isnan(zdr_mvavfl).mask))+1,
                            np.array([[0], [len(zdr_mvavfl)-1]])))))
                rcells_raw = [rcells_idx[i:i + 2]
                              for i in range(0, len(rcells_idx), 2)][1::2]
                if rcells_raw:
                    rcells_dm1 = np.concatenate(
                        [np.array(range(int(i[0]), int(i[-1])+1))
                         for i in rcells_raw])
                    rcells_dm2 = np.split(
                        rcells_dm1, np.where(np.diff(rcells_dm1) > 2)[0] + 1)
                    rcells_crc = [np.array([i[0], i[-1]]).reshape((2, 1))
                                  for i in rcells_dm2]
                    rcells_crc_len = np.concatenate(
                        [np.ediff1d(i)+1 if np.ediff1d(i)+1 >= minbins
                         else [np.nan] for i in rcells_crc])
                    raincells = [j for i, j in enumerate(rcells_crc)
                                 if not np.isnan(rcells_crc_len[i])]
                    if raincells:
                        raincells_len = np.concatenate([np.ediff1d(i)+1
                                                        for i in raincells])
                        rcell = raincells[np.nanargmax(raincells_len)]
                        idxrs = int(rcell[0])
                        idxrf = int(rcell[1])
                        zhcrf = np.nanmean(zh_mvavfl[idxrf - p2avrf+1:idxrf+1])
                        zdrmrf = np.nanmean(
                            zdr_mvavfl[idxrf - p2avrf+1:idxrf+1])
                        if method == 'linear':
                            if zhcrf <= params['ZH_lower_lim']:
                                zdrerf = 0
                            elif (params['ZH_lower_lim']
                                  < zhcrf <= params['ZH_upper_lim']):
                                zdrerf = params['a1'] * zhcrf - params['b1']
                            elif zhcrf > params['ZH_upper_lim']:
                                zdrerf = params['zdr_max']
                            else:
                                zdrerf = np.nan
                        elif method == 'exp':
                            zdrerf = params['a1']*zhcrf**params['b1']
                        else:
                            raise TowerpyError('Please check the method '
                                               'selected for estimating the '
                                               'theoretical values of ZDR')
                        if zdrerf > zdrmrf:
                            try:
                                betai = (
                                    abs(zdrmrf - zdrerf)
                                    / (attcorrmask['PhiDP [deg]'][i, :][idxrf]
                                       - attcorrmask['PhiDP [deg]'][i, :][idxrs]))
                                zdrirfpia = (
                                    zdrmrf +
                                    (betai
                                     / attcorrmask['alpha [-]'][i, :][idxrf])
                                    * np.nanmean(attcorrmask
                                                 ['PIA [dB]'][i, :]
                                                 [idxrf-p2avrf+1:idxrf+1]))
                                if abs(zdrirfpia - zdrerf) > 0:
                                    sl1 = optimize.root_scalar(
                                        lambda betaif: (
                                            (zdrmrf) + (betaif) *
                                            ((1 / attcorrmask['alpha [-]'][i, :][idxrf])
                                             * np.nanmean(
                                                 attcorrmask['PIA [dB]'][i, :]
                                                 [idxrf-p2avrf+1:idxrf+1])))
                                        - zdrerf,
                                        bracket=[coeff_beta[:2]],
                                        x0=(np.nanmin(alphacopy[i])
                                            * beta_alpha_ratio),
                                        method='brentq')
                                    betao = sl1.root
                                else:
                                    betao = betai
                                if betao <= 0:
                                    betao = coeff_beta[0]
                                if betao >= coeff_beta[1]:
                                    betao = coeff_beta[1]
                                zdrcr = ((attvars['ZDR [dB]'][i, :])
                                         + ((betao/alphacopy[i, :])
                                            * attcorr_vars['PIA [dB]'][i, :]))
                                adpi = ((betao
                                         / attcorr_vars['alpha [-]'][i, :])
                                        * attcorr_vars['AH [dB/km]'][i, :])
                                betaopt = (np.zeros_like(
                                    attcorr_vars['alpha [-]'][i, :]) + betao)
                                statzdr = f'{i}: beta coeff optimised 1 iter'
                            except ValueError:
                                idxrs = int(raincells[0][0])
                                idxrf = int(raincells[-1][-1])
                                zhcrf = np.nanmean(zh_mvavfl[idxrf-p2avrf+1:
                                                             idxrf+1])
                                zdrmrf = np.nanmean(zdr_mvavfl[idxrf-p2avrf+1:
                                                               idxrf+1])
                                if method == 'linear':
                                    if zhcrf <= params['ZH_lower_lim']:
                                        zdrerf = 0
                                    elif (params['ZH_lower_lim']
                                          < zhcrf <= params['ZH_upper_lim']):
                                        zdrerf = (params['a1']
                                                  * zhcrf - params['b1'])
                                    elif zhcrf > params['ZH_upper_lim']:
                                        zdrerf = params['zdr_max']
                                    else:
                                        zdrerf = np.nan
                                elif method == 'exp':
                                    zdrerf = params['a1']*zhcrf**params['b1']
                                else:
                                    raise
                                    TowerpyError('Please check the method '
                                                 'selected for estimating the '
                                                 'theoretical values of ZDR')
                                try:
                                    betai = (
                                        abs(zdrmrf - zdrerf)
                                        / (attcorrmask['PhiDP [deg]'][i, :][idxrf]
                                           - attcorrmask['PhiDP [deg]'][i, :][idxrs]))
                                    zdrirfpia = (
                                        zdrmrf +
                                        (betai / attcorrmask['alpha [-]'][i, :][idxrf])
                                        * np.nanmean(
                                            attcorrmask['PIA [dB]'][i, :]
                                            [idxrf-p2avrf+1:idxrf+1]))
                                    if abs(zdrirfpia - zdrerf) > 0:
                                        sl1 = optimize.root_scalar(
                                            lambda
                                            betaif: ((zdrmrf) + (betaif) *
                                                     ((1 / attcorrmask['alpha [-]'][i, :][idxrf])
                                                      * np.nanmean(attcorrmask['PIA [dB]'][i, :][idxrf - p2avrf+1:idxrf+1]))) - zdrerf,
                                            bracket=[coeff_beta[:2]],
                                            x0=(np.nanmin(alphacopy[i])
                                                * beta_alpha_ratio),
                                            method='brentq')
                                        betao = sl1.root
                                    else:
                                        betao = betai
                                    if betao <= 0:
                                        betao = coeff_beta[0]
                                    if betao >= coeff_beta[1]:
                                        betao = coeff_beta[1]
                                    zdrcr = (
                                        (attvars['ZDR [dB]'][i, :])
                                        + ((betao/alphacopy[i, :])
                                           * attcorr_vars['PIA [dB]'][i, :]))
                                    adpi = ((betao
                                             / attcorr_vars['alpha [-]'][i, :])
                                            * attcorr_vars['AH [dB/km]'][i, :])
                                    betaopt = (np.zeros_like(
                                        attcorr_vars['alpha [-]'][i, :])
                                        + betao)
                                    statzdr = f'{i}: beta coeff optimised 2 iter'
                                except ValueError:
                                    zdrcr = (
                                        (attvars['ZDR [dB]'][i, :])
                                        + (beta_alpha_ratio
                                           * attcorr_vars['PIA [dB]'][i, :]))
                                    adpi = (beta_alpha_ratio
                                            * attcorr_vars['AH [dB/km]'][i, :])
                                    betaopt = (attcorr_vars['alpha [-]'][i, :]
                                               * beta_alpha_ratio)
                                    statzdr = f'{i}: beta/alpha: fixed value'
                        else:
                            zdrcr = ((attvars['ZDR [dB]'][i, :])
                                     + (beta_alpha_ratio
                                        * attcorr_vars['PIA [dB]'][i, :]))
                            adpi = (beta_alpha_ratio
                                    * attcorr_vars['AH [dB/km]'][i, :])
                            betaopt = (attcorr_vars['alpha [-]'][i, :]
                                       * beta_alpha_ratio)
                            statzdr = f'{i}: beta/alpha: fixed value'
                    else:
                        zdrcr = (
                            (attvars['ZDR [dB]'][i, :])
                            + (beta_alpha_ratio
                               * attcorr_vars['PIA [dB]'][i, :]))
                        adpi = (beta_alpha_ratio
                                * attcorr_vars['AH [dB/km]'][i, :])
                        betaopt = (attcorr_vars['alpha [-]'][i, :]
                                   * beta_alpha_ratio)
                        statzdr = f'{i}: beta/alpha: fixed value'
                else:
                    zdrcr = (
                        (attvars['ZDR [dB]'][i, :])
                        + (beta_alpha_ratio * attcorr_vars['PIA [dB]'][i, :]))
                    adpi = beta_alpha_ratio * attcorr_vars['AH [dB/km]'][i, :]
                    betaopt = (attcorr_vars['alpha [-]'][i, :]
                               * beta_alpha_ratio)
                    statzdr = f'{i}: beta/alpha: fixed value'
            else:
                zdrcr = (
                    (attvars['ZDR [dB]'][i, :])
                    + (beta_alpha_ratio * attcorr_vars['PIA [dB]'][i, :]))
                adpi = beta_alpha_ratio * attcorr_vars['AH [dB/km]'][i, :]
                betaopt = attcorr_vars['alpha [-]'][i, :] * beta_alpha_ratio
                statzdr = f'{i}: beta/alpha: fixed value'
            zdrattcorr.append(zdrcr)
            adpif.append(adpi)
            betaof.append(betaopt)
            zdrstat.append(statzdr)

        attcorr1 = {'ZDR [dB]': np.array(zdrattcorr),
                    'ADP [dB/km]': np.array(adpif),
                    'beta [-]': np.array(betaof)}

        for n, rows in enumerate(attcorr1['ADP [dB/km]']):
            rows[mlb_idx[n]:] = 0
        for n, rows in enumerate(attcorr1['beta [-]']):
            rows[mlb_idx[n]:] = 0

        # =====================================================================
        # Filter non met values
        # =====================================================================
        for key, values in attcorr1.items():
            values[cclass != 0] = np.nan
        zdr_calc = {}
        if descr is not False:
            zdr_calc['descriptor'] = [j for i, j in enumerate(zdrstat)
                                      if i in idxzdrattcorr]

        self.res_zdrcorrection = zdr_calc
        self.vars |= attcorr1

        toc = time.time()
        print(r'Z_{DR} attenuation correction running time: '
              f'{toc-tic:.3f} sec.')
        if plot_method:
            rad_display.plot_zdrattcorr(rad_georef, rad_params, attvars,
                                        attcorr1)
