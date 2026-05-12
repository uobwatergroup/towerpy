"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import copy
import ctypes as ctp
import platform
from pathlib import Path

import numpy as np
import numpy.ctypeslib as npct
import xarray as xr
from scipy import optimize

from ..base import TowerpyError
from ..datavis import rad_display
from ..io import modeltp as mdtp
from ..ml.mlyr import MeltingLayer, attach_melting_layer
from ..utils.radutilities import (
    add_correction_step,
    fillnan1d,
    find_nearest,
    interp_nan,
    maf_radial,
    record_provenance,
    rolling_window,
    safe_assign_variable,
)
from ..utils.unit_conversion import convert


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

    def __init__(self, radobj=None):
        self.elev_angle = getattr(radobj, 'elev_angle',
                                  None) if radobj else None
        self.file_name = getattr(radobj, 'file_name',
                                 None) if radobj else None
        self.scandatetime = getattr(radobj, 'scandatetime',
                                    None) if radobj else None
        self.site_name = getattr(radobj, 'site_name',
                                 None) if radobj else None

    def attc_phidp_prepro(self, rad_georef, rad_params, attvars,
                          mov_avrgf_len=(1, 3), t_spdp=10, minthr_pdp0=-5,
                          rhohv_min=0.90, phidp0_correction=False,
                          mlyr=None):
        r"""
        Prepare :math:`\Phi_{DP}` for attenuation correction.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        attvars : dict
            Polarimetric variables used for the attenuation correction.
        mov_avrgf_len : 2-element tuple or list, optional
            Window size used to smooth :math:`\Phi_{DP}` by applying a
            moving average window. The default is (1, 3). It is
            recommended to average :math:`\Phi_{DP}` along the range,
            i.e. keep the window size in a (1, n) size.
        t_spdp: int or float, optional
            Discard bins with standard deviations of :math:`\Phi_{DP}`
            greater than the selected value. The default is 10 deg.
        minthr_pdp0: int or float, optional
            Tolerance for the true value of :math:`\Phi_{DP}(r0)`.
            Values below this threshold are removed.
            The default is -5 deg.
        rhohv_min: float, optional
            Threshold in :math:`\rho_{HV}` used to discard bins
            related to nonmeteorological
            signals. The default is 0.90
        phidp0_correction : Bool, optional
            If True, adjust :math:`\Phi_{DP}(r0)` for each individual
            ray.
        mlyr : MeltingLayer Class, optional
            Filter and interpolate PhiDP within the melting layer.
            The ml_top (float, int, list or np.array) and ml_bottom
            (float, int, list or np.array) must be explicitly defined.
            The default is None.

        Notes
        -----
        1. This function smooths the total :math:`\Phi_{DP}` using
        the given window size, remove spurious values within the signal
        phase, etc.

        2. :math:`\Phi_{DP}` must have been previously unfolded and
        offset (:math:`\Phi_{DP}(r0)`) corrected.

        """
        ngates = rad_params['ngates']
        attvars = copy.copy(attvars)

        if (mov_avrgf_len[1] % 2) == 0:
            print('Choose an odd number to apply the '
                  + 'moving average filter')
        phidp_O = {k: np.ones_like(attvars[k]) * attvars[k]
                   for k in list(attvars) if k.startswith('Phi')}
        # Removes low-noisy values of PhiDP below the given PhiDP(0)
        phidp_O['PhiDP [deg]'][phidp_O['PhiDP [deg]']
                               < minthr_pdp0] = minthr_pdp0
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
        # Filter using rhoHV
        phidp_dspk[attvars['rhoHV [-]'] < rhohv_min] = np.nan
        # Computes sPhiDP for each ray
        phidp_dspk_rhv = phidp_O['PhiDP [deg]'] * phidp_dspk
        phidp_s = np.nanstd(rolling_window(
            phidp_dspk_rhv, mov_avrgf_len), axis=-1, ddof=1)
        phidp_pad = np.pad(phidp_s, ((0, 0), (mov_avrgf_len[1]//2,
                                              mov_avrgf_len[1]//2)),
                           mode='constant', constant_values=(np.nan))
        # Filter values with std values greater than std threshold
        phidp_sfnv = np.array(
            [[np.nan if vbin >= t_spdp
              and (phidp_pad[nr][nbin-1] >= t_spdp
                   or phidp_pad[nr][nbin+1] >= t_spdp)
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
        # Computes initial PhiDP(0)
        if phidp0_correction:
            phidp0 = np.array([[
                nr[np.isfinite(nr)][0] if ~np.isnan(nr).all() else 0
                for nr in phidp_f]], dtype=np.float64).transpose()
        else:
            phidp0 = np.array([[1e-5 if ~np.isnan(nr).all() else 0
                                for nr in phidp_f]],
                              dtype=np.float64).transpose()
        phidp0[phidp0 == 0] = np.nanmedian(phidp0[phidp0 != 0])
        phidp0t = np.tile(phidp0, (1, mov_avrgf_len[1] + 1))
        phidp_f[:, : mov_avrgf_len[1] + 1] = phidp0t
        # Add phidp0
        phidp_f = np.array([nr - phidp0[cr]
                            for cr, nr in enumerate(phidp_f)],
                           dtype=np.float64)
        # Filter and interpolate ML
        if mlyr is not None:
            if not getattr(mlyr, 'mlyr_limits', None):
                mlyr.ml_ppidelimitation(rad_georef, rad_params)
            phidp_fml = np.where(
                mlyr.mlyr_limits['pcp_region [HC]'] == 2, np.nan, phidp_f)
            itprng = np.array(range(ngates), dtype=np.float64)
            phidp_fmli = np.array(
                [interp_nan(itprng, nr, nan_type='nan') if ~np.isnan(nr).all()
                 else nr for nr in phidp_fml], dtype=np.float64)
            phidp_f = np.where(mlyr.mlyr_limits['pcp_region [HC]'] == 2,
                               phidp_fmli, phidp_f)
        # Computes a MAV
        phidp_m = maf_radial(
            {'PhiDPi [deg]': phidp_f}, maf_len=mov_avrgf_len[1],
            maf_ignorenan=True, maf_extendvalid=False)['PhiDPi [deg]']
        # Filter isolated values
        phidp_pad = np.pad(phidp_m, ((0, 0), (mov_avrgf_len[1]//2,
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
        phidp_m = phidp_m * phidp_f2
        # Interpolation and final filtering
        itprng = np.array(range(ngates), dtype=np.float64)
        phidp_i = {'PhiDPi [deg]': np.array(
            [interp_nan(itprng, nr, nan_type='nan') if ~np.isnan(nr).all()
             else nr for nr in phidp_m], dtype=np.float64)}
        phidp_i['PhiDPi [deg]'][:, : mov_avrgf_len[
            1] + 1] = phidp_f[:, : mov_avrgf_len[1] + 1]
        phidp_i = {'PhiDPi [deg]': np.array(
            [fillnan1d(i) for i in phidp_i['PhiDPi [deg]']], dtype=np.float64)}
        # Apply a moving average filter to the whole PPI
        phidp_maf = maf_radial(
            phidp_i, maf_len=mov_avrgf_len[1], maf_ignorenan=False,
            maf_extendvalid=True)
        # Filter values using ZH
        phidp_maf['PhiDPi [deg]'][np.isnan(
            attvars['ZH [dBZ]'])] = np.nan
        attvars['PhiDP [deg]'] = phidp_maf['PhiDPi [deg]']
        self.vars = attvars

    def zh_correction(self, rad_georef, attvars, cclass, mlyr=None,
                      attc_method='ABRI', pdp_pxavr_rng=7, pdp_pxavr_azm=1,
                      pdp_dmin=20, coeff_alpha=[0.020, 0.1, 0.073],
                      coeff_a=[1e-5, 9e-5, 3e-5], coeff_b=[0.65, 0.85, 0.78],
                      phidp0=0, niter=500, plot_method=False):
        r"""
        Calculate the attenuation of :math:`Z_{H}`.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        attvars : dict
            Polarimetric variables used for the attenuation correction.
        cclass : array
            Clutter, noise and meteorological echoes classification:
            'pcpn' = 0, 'noise' = 3, 'clutter' = 5
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness, in km. Only gates below the
            melting layer bottom (i.e. the rain region below the melting layer)
            are included in the computation; ml_top and ml_thickness can be
            either a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            function is applied assuming ml_top=5km and ml_thickness=0.5 km.
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
            Minimum total :math:`\Delta\Phi_{DP}` expected in a ray to perform
            attenuation correction (at least 10-20 degrees). The default is 20.
        coeff_alpha : 3-element tuple or list, optional
            [Min, max, fixed value] of coeff :math:`\alpha`. These bounds are
            used to find the optimum value of :math:`\alpha` from
            :math:`A_H = \alpha K_{DP}`. Default values are
            [0.020, 0.1, 0.073], derived for C-band.
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
        niter : int
            Number of iterations to find the optimised values of
            the coeffs :math:`a, b, \alpha`. The default is 500.
        phidp0 : int , float or None
            Adjusts the value (in deg) of :math:`\Phi_{DP}(r0)` for the whole
            scan. If None, the function computes the value of the offset by
            averaging the value of :math:`\Phi_{DP}` in the first ten
            consecutive bins classified as rain, according to the cclass. The
            default is 0.
        plot_method : Bool, optional
            Plot the ZH attenuation correction method. The default is False.

        Returns
        -------
        vars : dict
            ZH [dBZ]:
                Corrected horizontal reflectivity
            AH [dB/km]:
                Specific horizontal attenuation
            PhiDP [deg]:
                Processed and adjusted differential phase
            PhiDP* [deg]:
                Computed differential phase: Its availability depends on the
                selected method.
            KDP [deg/km]:
                Specific differential phase, calculated using the equation
                :math:`K_{DP}=A_H/\alpha`
            PIA [dB]:
                Path-Integrated Attenuation, calculated using the equation
                :math:`PIA=\Phi_{DP}*\alpha`
            alpha:
                parameter :math:`\alpha` that represents the ratio
                :math:`A_H/K_{DP}`

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

        if isinstance(mlyr.ml_top, (int, float)):
            mlgrid = np.zeros_like(
                attvars['ZH [dBZ]']) + (mlyr.ml_top) * 1000
        elif isinstance(mlyr.ml_top, (np.ndarray, list, tuple)):
            mlgrid = (np.ones_like(
                attvars['ZH [dBZ]'].T) * mlyr.ml_top * 1000).T
        param_atc = np.zeros(16)
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
        if isinstance(phidp0, (int, float)):
            param_atc[15] = phidp0
        else:
            param_atc[15] = -999  # PhiDP offset
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
        attcorr['PIA [dB]'] = attcorr['PhiDP [deg]'] * attcorr['alpha [-]']
        attcorr['PIA [dB]'][attcorr['PIA [dB]'] < 0] = 0
        attcorr['AH [dB/km]'][attcorr['AH [dB/km]'] < 0] = 0
        attcorr['KDP [deg/km]'] = np.nan_to_num(
            attcorr['AH [dB/km]'] / attcorr['alpha [-]'])
        piacopy = np.zeros_like(attcorr['PIA [dB]']) + attcorr['PIA [dB]']
        for i in range(nrays):
            idmx = np.nancumsum(piacopy[i]).argmax()
            if idmx != 0:
                attcorr['PIA [dB]'][i][idmx+1:] = attcorr['PIA [dB]'][i][idmx]
        # =====================================================================
        # Filter non met values
        # =====================================================================
        for key, values in attcorr.items():
            if np.array(values).ndim == 2:
                values[cclass != 0] = np.nan

        self.vars = attcorr
        self.phidp_offset = param_atc[15]

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
            rad_display.plot_zhattcorr(
                rad_georef, rad_params, attvars, attcorr, mlyr=mlyr,
                vars_bounds={'alpha [-]':
                             [0,
                              round((coeff_alpha[1]+(coeff_alpha[1]*.1))*100,
                                    -1)/100, 11]})

    def zdr_correction(self, rad_georef, attvars, attcorr_vars,
                       cclass, mlyr=None, attc_method='BRI',
                       coeff_beta=[0.008, 0.1, 0.04], beta_alpha_ratio=0.265,
                       rhv_thld=0.985, mov_avrgf_len=9, minbins=10, p2avrf=5,
                       zh_zdr_model='linear', rparams=None, descr=False,
                       plot_method=False):
        r"""
        Calculate the attenuation of :math:`Z_{DR}`.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height.
        attvars : dict
            Polarimetric variables to be corrected for attenuation.
        attcorr_vars : dict
            Attenuation-corrected polarimetric variables used for calculations.
        cclass : array
            Clutter, noise and meteorological echoes classification:
            'pcpn' = 0, 'noise' = 3, 'clutter' = 5
        mlyr : MeltingLayer Class, optional
            Melting layer class containing the top of the melting layer, (i.e.,
            the melting level) and its thickness, in km. Only gates below the
            melting layer bottom (i.e. the rain region below the melting layer)
            are included in the computation; ml_top and ml_thickness can be
            either a single value (float, int), or an array (or list) of values
            corresponding to each azimuth angle of the scan. If None, the
            function is applied assuming ml_top=5km and ml_thickness=0.5 km.
        attc_method : str
            Attenuation correction algorithm to be used. The default is 'BRI':

                [ABRI] = Bringi (optimised beta parameter).

                [BRI] = Bringi (constant beta/alpha ratio).
        coeff_beta : 3-element tuple or list, optional
            [Min, max, fixed value] of coeff :math:`\beta`. These bounds are
            used to find the optimum value of :math:`\beta`.
            Default values are [0.002, 0.04, 0.02], derived for C-band.
        beta_alpha_ratio : float, opt
            Quotient between :math:`\alpha` and :math:`\beta` parameters from
            :math:`A_{DP} = ( \beta / \alpha )A_{H}`. The default is 0.265 for
            C-band.
        rhv_thld : float
            Minimum value of :math:`\rho_{HV}` expected in the rain medium.
            The default is 0.98.
        mov_avrgf_len : int
            Odd number used to apply a moving average filter to each beam and
            smooth the signal. The default is 5.
        minbins : int
            Minimum number of bins related to the length of each rain cell
            along the beam. The default is 10.
        p2avrf : int
            Number of bins to average on the far side of the rain cell.
            The default is 3.
        zh_zdr_model : str
            Method used to compute the ZH-ZDR relationship. The "linear" model
            uses the relation provided in [2]_ and [4]_, whereas the "exp"
            model uses the relation proposed by [3]_.
        rparams: dict
            Additional parameters describing the ZH-ZDR relationship:
                For the linear model: ZH_lower_lim: 20 dBZ, ZH_upper_lim:
                45 dBZ, coeff_a: 0.048, coeff_b: 0.774, zdr_max: 1.4

                .. math::
                    \overline{Z}_{DR} =
                    \Biggl\{ 0 \rightarrow  \overline{Z_H}(r_m)<=Z_H(lowerlim) \\
                            a*Z_H-b \rightarrow Z_H(lowerlim)<Z_H(r_m)<=Z_H(upperlim) \\
                                Z_{DR}(max) \rightarrow Z_H(r_m)>Z_H(upperlim) \Biggl\}

                For the exp model: coeff_a: 0.00012, coeff_b: 2.5515

                .. math::
                    \overline{Z}_{DR} =
                    \Biggl\{ a*Z_H^{b} \Biggl\}
        descr : bool
            Controls if the statistics of the calculations are returned.
            The default is False.
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
               parameter :math:`\beta` optimised for each beam.

        Notes
        -----
        1. The attenuation is computed up to a user-defined melting level
        height.

        2. The ZDR attenuation correction method assumes that ZH has first been
        corrected for attenuation, e.g., using the methods described in [1]_

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
        # =====================================================================
        # Parameter setup
        # =====================================================================
        params = {'ZH-ZDR model': zh_zdr_model, 'ZH_lower_lim': 20,
                  'ZH_upper_lim': 45, 'model': 'coeff_a*ZH-coeff_b',
                  'zdr_max': 1.4, 'coeff_a': 0.048, 'coeff_b': 0.774}
        if zh_zdr_model == 'exp':
            params['ZH-ZDR model'] = zh_zdr_model
            params['model'] = 'coeff_a*ZH^coeff_b'
            params['coeff_a'] = 0.00012
            params['coeff_b'] = 2.5515
        if rparams is not None:
            params.update(rparams)
        if mlyr is None:
            mlyr = MeltingLayer(self)
            mlyr.ml_top = 5
            mlyr.ml_thickness = 0.5
            mlyr.ml_bottom = mlyr.ml_top - mlyr.ml_thickness
        else:
            mlyr.ml_bottom = mlyr.ml_top - mlyr.ml_thickness

        if isinstance(mlyr.ml_bottom, (int, float)):
            mlb_idx = [find_nearest(nbh, mlyr.ml_bottom)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_bottom, (np.ndarray, list, tuple)):
            mlb_idx = [find_nearest(nbh, mlyr.ml_bottom[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]

        nrays = len(rad_georef['azim [rad]'])

        mlb_mask = np.full_like(attcorr_vars['ZH [dBZ]'], 0.0, dtype=float)
        for n, rows in enumerate(mlb_mask):
            rows[mlb_idx[n]+1:] = np.nan
        mlb_mask[cclass != 0] = np.nan
        mlb_mask[:, 0] = np.nan

        attcorrmask = {keys: np.ma.masked_array(values, mlb_mask)
                       for keys, values in attcorr_vars.items()}
        attcorrmask['rhoHV [-]'] = np.ma.masked_array(attvars['rhoHV [-]'],
                                                      mlb_mask)
        attcorrmask['ZDR [dB]'] = np.ma.masked_array(attvars['ZDR [dB]'],
                                                     mlb_mask)

        idxzdrattcorr = [i for i in range(nrays)
                         if any(attcorr_vars['alpha [-]'][i, :] > 0)]

        zdrattcorr, adpif, betaof, zdrstat = [], [], [], []

        alphacopy = np.array(attcorr_vars['alpha [-]'], copy=True)

        if attc_method == 'ABRI':
            bracket = coeff_beta[:2]
        elif attc_method == 'BRI':
            bracket = []
        # =====================================================================
        # Ray loop for attc
        # =====================================================================
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
                        [np.array(range(int(i[0].item()), int(i[-1].item())+1))
                         for i in rcells_raw])
                    rcells_dm2 = np.split(
                        rcells_dm1, np.where(np.diff(rcells_dm1) > 1)[0] + 1)
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
                        idxrs = int(rcell[0].item())
                        idxrf = int(rcell[1].item())
                        zhcrf = np.nanmean(zh_mvavfl[idxrf - p2avrf+1:idxrf+1])
                        zdrmrf = np.nanmean(
                            zdr_mvavfl[idxrf - p2avrf+1:idxrf+1])
                        if params['ZH-ZDR model'] == 'linear':
                            if zhcrf <= params['ZH_lower_lim']:
                                zdrerf = 0
                            elif (params['ZH_lower_lim']
                                  < zhcrf <= params['ZH_upper_lim']):
                                zdrerf = params['coeff_a'] * zhcrf - params['coeff_b']
                            elif zhcrf > params['ZH_upper_lim']:
                                zdrerf = params['zdr_max']
                            else:
                                zdrerf = np.nan
                        elif params['ZH-ZDR model'] == 'exp':
                            zdrerf = params['coeff_a']*zhcrf**params['coeff_b']
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
                                        - zdrerf, bracket=bracket,
                                        x0=(np.nanmin(alphacopy[i])
                                            * beta_alpha_ratio),
                                        method='brentq')
                                    betao = sl1.root
                                else:
                                    betao = betai
                                if betao <= coeff_beta[0]:
                                    betao = coeff_beta[0]
                                if betao >= coeff_beta[1]:
                                    betao = coeff_beta[1]
                                if np.isnan(betao):
                                    betao = coeff_beta[2]
                                betaopt = (np.zeros_like(
                                    attcorr_vars['alpha [-]'][i, :]) + betao)
                                zdrcr = ((attvars['ZDR [dB]'][i, :])
                                         + ((betao/alphacopy[i, :])
                                            * attcorr_vars['PIA [dB]'][i, :]))
                                adpi = ((betao
                                         / attcorr_vars['alpha [-]'][i, :])
                                        * attcorr_vars['AH [dB/km]'][i, :])
                                statzdr = f'{i}: beta coeff optimised 1 iter'
                            except ValueError:
                                idxrs = int(raincells[0][0].item())
                                idxrf = int(raincells[-1][-1].item())
                                zhcrf = np.nanmean(zh_mvavfl[idxrf-p2avrf+1:
                                                             idxrf+1])
                                zdrmrf = np.nanmean(zdr_mvavfl[idxrf-p2avrf+1:
                                                               idxrf+1])
                                if params['ZH-ZDR model'] == 'linear':
                                    if zhcrf <= params['ZH_lower_lim']:
                                        zdrerf = 0
                                    elif (params['ZH_lower_lim']
                                          < zhcrf <= params['ZH_upper_lim']):
                                        zdrerf = (params['coeff_a']
                                                  * zhcrf - params['coeff_b'])
                                    elif zhcrf > params['ZH_upper_lim']:
                                        zdrerf = params['zdr_max']
                                    else:
                                        zdrerf = np.nan
                                elif params['ZH-ZDR model'] == 'exp':
                                    zdrerf = params['coeff_a']*zhcrf**params['coeff_b']
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
                                            bracket=bracket,
                                            x0=(np.nanmin(alphacopy[i])
                                                * beta_alpha_ratio),
                                            method='brentq')
                                        betao = sl1.root
                                    else:
                                        betao = betai
                                    if betao <= coeff_beta[0]:
                                        betao = coeff_beta[0]
                                    if betao >= coeff_beta[1]:
                                        betao = coeff_beta[1]
                                    if np.isnan(betao):
                                        betao = coeff_beta[2]
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
                    # No segments found
                    zdrcr = (
                        (attvars['ZDR [dB]'][i, :])
                        + (beta_alpha_ratio * attcorr_vars['PIA [dB]'][i, :]))
                    adpi = beta_alpha_ratio * attcorr_vars['AH [dB/km]'][i, :]
                    betaopt = (attcorr_vars['alpha [-]'][i, :]
                               * beta_alpha_ratio)
                    statzdr = f'{i}: beta/alpha: fixed value'
            else:
                # Ray with no positive alpha
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
            rows[mlb_idx[n]+1:] = 0
        for n, rows in enumerate(attcorr1['beta [-]']):
            rows[mlb_idx[n]+1:] = 0

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
            rad_display.plot_zdrattcorr(rad_georef, rad_params, attvars,
                                        attcorr1, mlyr=mlyr)


# =============================================================================
# %% xarray implementation
# =============================================================================

def attenuation_correction_zh(dsattvars, cclass, inp_names=None,
                              attc_method='ABRI', mlyr_top=5., mlyr_thk=0.75,
                              mlyr_btm=None, phidp0=0., pdp_dmin=20,
                              pdp_pxavr_rng=7, pdp_pxavr_azm=1, niter=500,
                              coeff_alpha=[0.020, 0.1, 0.073],
                              coeff_a=[1e-5, 9e-5, 3e-5],
                              coeff_b=[0.65, 0.85, 0.78], merge_into_ds=False,
                              replace_vars=False, modify_output=None):
    r"""
    Perform attenuation correction of :math:`Z_H` using one of the algorithms
    described in Rico‑Ramirez (2012).

    Parameters
    ----------
    dsattvars : xarray.Dataset
        Dataset containing polarimetric variables filtered by noise, along
        with the polar coordinates (range, azimuth, elevation).
    cclass : xarray.Dataarray
        Clutter, noise and weather echoes classification. Attributes must
        describe the flags used for classification, e.g.:
        {'pcpn': 0, 'noise': 3, 'clutter': 5}
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{'azi': 'azimuth', 'rng': 'range', 'elv': 'elevation', "ZH": "DBZH",
        "RHOHV": "RHOHV", "PHIDP": "PHIDP"}``
    attc_method : str, default 'ABRI'
        Attenuation correction algorithm to be used:

            ``'ABRI'`` = Bringi (optimised).

            ``'AFV'`` = Final value (optimised).

            ``'AHB'`` = Hitschfeld and Bordan (optimised).

            ``'ZPHI'`` = Testud (constant parameters).

            ``'BRI'`` = Bringi (constant parameters).

            ``'FV'`` = Final value (constant parameters).

            ``'HB'`` = Hitschfeld and Bordan (constant parameters).
    mlyr_top, mlyr_thk, mlyr_btm : float or array, optional
        Heights of the melting layer boundaries, in km. Only gates below the
        melting layer bottom (i.e. the rain region below the melting layer)
        are included in the computation.
    pdp_dmin : float, default 20
        Minimum total :math:`\Delta\Phi_{DP}` expected in a ray to perform
        attenuation correction (at least 10–20 degrees).
    pdp_pxavr_rng : int, default 7
        Pixels to average in :math:`\Phi_{DP}` along range: odd number
        equivalent to about 4 km, i.e. 4 km / range_resolution.
    pdp_pxavr_azm : int, default 1
        Pixels to average in :math:`\Phi_{DP}` along azimuth. Must be an
        odd number.
    coeff_alpha : 3-element tuple or list, optional
        [Min, max, fixed value] of coeff :math:`\alpha`. These bounds are
        used to find the optimum value of :math:`\alpha` from
        :math:`A_H = \alpha K_{DP}`. Default values are
        [0.020, 0.1, 0.073], derived for C-band.
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
    niter : int, default 500
        Number of iterations to find the optimised values of the coeffs
        :math:`a, b, \alpha`.
    phidp0 : int or float or None, default 0
        Adjusts the value (in deg) of :math:`\Phi_{DP}(r0)` for the whole
        scan. If None, the function computes the value of the offset by
        averaging the value of :math:`\Phi_{DP}` in the first ten
        consecutive bins classified as rain.
    merge_into_ds : bool, default False
        If True, corrected variables are merged into the full dataset.
        If False, return a dataset containing only the corrected outputs.
    replace_vars : bool, default False
        If True, overwrite existing variables (ZH, PHIDP, etc.).
        If False, corrected variables receive an "_ATTC" suffix unless
        explicit names are provided via modify_output.
    modify_output : bool | list[str] | dict[str, str] | None
        Controls which variables receive the corrected outputs and how they
        are named:

        - None: apply correction only to the primary variables
          (ZH, PHIDP, AH, ALPHA, KDP, PIA).
        - True: apply correction to all primary variables.
        - list: apply correction only to the listed variables.
        - dict: map input variable names to explicit output names.

    Returns
    -------
    xarray.Dataset
        Dataset containing the attenuation‑corrected variables:

        ZH : dBZ
            Corrected horizontal reflectivity.
        AH : dB/km
            Specific horizontal attenuation.
        PHIDP : deg
            Processed and adjusted differential phase.
        PHIDP_CALC : deg
            Computed differential phase: Its availability depends on the
            selected method.
        KDP : deg/km
            Specific differential phase, calculated using the equation
            :math:`K_{DP}=A_H/\alpha`.
        PIA : dB
            Path-Integrated Attenuation, calculated using the equation
            :math:`PIA=\Phi_{DP}*\alpha`.
        ALPHA :
            parameter :math:`\alpha` that represents the ratio
            :math:`A_H/K_{DP}`.

    Notes
    -----
    * This function operates in native polar radar coordinates.
    * The attenuation is computed up to a user-defined melting level
      height.
    * This function uses the shared object *'lnxlibattenuationcorrection'*
      or the dynamic link library *'w64libattenuationcorrection'* depending on
      the operating system (OS).
    * Units for range, azimuth and elevation are inspected and converted to
      the appropriate units (m, rad) when necessary.

    References
    ----------
    .. [1] Rico-Ramirez, M. A. (2012). Adaptive attenuation correction
        techniques for C-Band polarimetric weather radars. IEEE Transactions
        on Geoscience and Remote Sensing, 50(12), 5061–5071.
        https://doi.org/10.1109/tgrs.2012.2195228
    """
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # =============================================================================
    # Resolve variable names
    # =============================================================================
    defaults = {'azi': 'azimuth', 'rng': 'range', 'elv': 'elevation',
                "ZH": "DBZH", "RHOHV": "RHOHV", "PHIDP": "PHIDP"}
    names = {**defaults, **(inp_names or {})}
    # =============================================================================
    # Prepare ctypes interface
    # =============================================================================
    array1d = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
    array2d = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
    if platform.system() == 'Linux':
        libattc = 'lnxlibattenuationcorrection.so'
        load_libattc = npct.load_library(
            libattc, Path(__file__).parent.absolute())
        # load_libattc = npct.load_library(libattc, Path.cwd())
    elif platform.system() == 'Windows':
        libattc = 'w64libattenuationcorrection.dll'
        load_libattc = ctp.cdll.LoadLibrary(f'{Path(__file__).parent.absolute()}/'
                                     + libattc)
    else:
        load_libattc = None
        libattc = 'no_libraryOS'
        raise ValueError(f'The {platform.system()} OS is not currently'
                         'compatible with this version of Towerpy')
    load_libattc.attenuationcorrection.restype = None
    load_libattc.attenuationcorrection.argtypes = [
        ctp.c_int, ctp.c_int, array2d, array2d, array2d, array2d, array2d,
        array1d, array1d, array1d, array1d, array2d, array2d, array2d, array2d,
        array2d]
    # =============================================================================
    #  Prepare dataset, melting-layer grid and cclass
    # =============================================================================
    # Convert the ML height into a 2‑D grid
    ds = dsattvars.copy(deep=False)
    ds = attach_melting_layer(ds, mlyr_top=mlyr_top, mlyr_bottom=mlyr_btm,
                              mlyr_thickness=mlyr_thk, units="km",
                              source="user-defined", method="zh_attc",
                              overwrite=True)
    # Convert km -> m
    ml_top_m = convert(ds.MLYRTOP, 'm')
    # Expand to (azimuth, range)
    mlgrid = ml_top_m.broadcast_like(ds[names["ZH"]])
    # Extract raw contiguous numpy array for the C library
    mlgrid = np.ascontiguousarray(mlgrid.values, dtype=np.float64)
    # Remap classification IDs
    echoesID = {'pcpn': 0, 'noise': 3, 'clutter': 5}
    flagsID = {'pcpn': cclass.attrs['flags']['pcpn'],
               'noise': cclass.attrs['flags']['noise'],
               'clutter': cclass.attrs['flags']['clutter']}
    cclass_arr = cclass.copy()
    cclass_arr.values[cclass_arr.values == flagsID['pcpn']] = echoesID['pcpn']
    cclass_arr.values[cclass_arr.values == flagsID['noise']] = echoesID['noise']
    cclass_arr.values[cclass_arr.values == flagsID['clutter']] = echoesID['clutter']
    cclass_arr = np.ascontiguousarray(cclass_arr.values, dtype=np.float64)
    # =============================================================================
    # Build parameter vector for C routine
    # =============================================================================
    param_atc = np.zeros(16, dtype=np.float64)

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
        raise ValueError('Please select a valid attenuation correction'
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
    param_atc[14] = mlyr_thk * 1000  # BB thickness in meters
    if isinstance(phidp0, (int, float)):
        param_atc[15] = phidp0
    else:
        param_atc[15] = -999  # PhiDP offset
    # =============================================================================
    # Allocate output arrays and prepare inputs
    # =============================================================================
    nrays = ds.sizes[names["azi"]]
    nbins = ds.sizes[names["rng"]]
    zhh_Ac = np.full((nrays, nbins), np.nan)
    Ah = np.full((nrays, nbins), np.nan)
    phidp_m = np.full((nrays, nbins), np.nan)
    phidp_c = np.full((nrays, nbins), np.nan)
    alpha = np.full((nrays, nbins), np.nan)
    # Z = ds[names["rng"]].fillna(-50.0)
    Z = np.ascontiguousarray(ds[names["ZH"]].values, dtype=np.float64)
    PHIDP = np.ascontiguousarray(ds[names["PHIDP"]].values, dtype=np.float64)
    RHOHV = np.ascontiguousarray(ds[names["RHOHV"]].values, dtype=np.float64)
    # Geometry normalisation
    rng_m = convert(ds[names["rng"]], "m")
    azi_rad = convert(ds[names["azi"]], "rad")
    elv_rad = convert(ds[names["elv"]], "rad")
    rng_m = np.ascontiguousarray(rng_m.values, dtype=np.float64)
    azi_rad = np.ascontiguousarray(azi_rad.values, dtype=np.float64)
    elv_rad = np.ascontiguousarray(elv_rad.values, dtype=np.float64)
    # =============================================================================
    # Call C routine for attc
    # =============================================================================
    load_libattc.attenuationcorrection(nrays, nbins, Z, PHIDP, RHOHV, mlgrid,
                                       cclass_arr, rng_m, azi_rad, elv_rad,
                                       param_atc, zhh_Ac, Ah, phidp_m, phidp_c,
                                       alpha)
    # =============================================================================
    # Post-processing: PIA, KDP, clipping, cumulative max, masking
    # =============================================================================
    attcorr = {"ZH_ATT": zhh_Ac, "AH": Ah, "ALPHA": alpha, "PHIDP_m": phidp_m,
               "PHIDP_CALC": phidp_c}
    # PIA = PhiDP_m * alpha
    pia = attcorr["PHIDP_m"] * attcorr["ALPHA"]
    pia[pia < 0] = 0.0
    # Cumulative max of PIA along range
    pia_cum = pia.copy()
    for i in range(nrays):
        idmx = np.nancumsum(pia_cum[i]).argmax()
        if idmx != 0:
            pia_cum[i, idmx + 1 :] = pia_cum[i, idmx]
    attcorr["PIA"] = pia_cum
    # AH must be non-negative
    Ah[Ah < 0] = 0.0
    attcorr["AH"] = Ah
    # KDP = AH / alpha (avoid NaNs blowing up)
    with np.errstate(divide="ignore", invalid="ignore"):
        kdp = np.nan_to_num(attcorr["AH"] / attcorr["ALPHA"])
    attcorr["KDP"] = kdp
    # Mask non-meteorological gates (cclass != 0)
    for key, values in attcorr.items():
        if values.ndim == 2:
            values[cclass_arr != echoesID["pcpn"]] = np.nan
    # =============================================================================
    # Build xarray.Dataset output
    # =============================================================================
    # Determine which outputs to include (minimal new logic)
    primary_outputs = {key: key for key in attcorr.keys()}
    # Determine selected outputs
    if modify_output is None or modify_output is True:
        selected = list(primary_outputs.keys())
    elif isinstance(modify_output, list):
        selected = modify_output
    elif isinstance(modify_output, dict):
        selected = list(modify_output.keys())
    else:
        raise TypeError("modify_output must be None, bool, list, or dict")
    # Start from original dataset or empty dataset WITH coords + attrs
    if merge_into_ds:
        ds_out = ds.copy()
    else:
        ds_out = xr.Dataset(coords=ds.coords, attrs=ds.attrs.copy())
    created_vars = []
    for logical_name in selected:
        arr = attcorr[logical_name]
        # Determine parent variable
        if logical_name == "ZH_ATT":
            parent_name = names["ZH"]
        elif logical_name in ("PHIDP_m", "PHIDP_CALC"):
            parent_name = names["PHIDP"]
        else:
            parent_name = logical_name
        # Determine output name
        if isinstance(modify_output, dict):
            out_name = modify_output.get(logical_name, logical_name)
        else:
            if logical_name == "PHIDP_CALC":
                # PHIDP_CALC is always a separate diagnostic field
                out_name = f"{names['PHIDP']}_CALC"
            elif replace_vars:
                out_name = parent_name
            else:
                out_name = f"{parent_name}_ATTC"
        # Build attrs
        parent_attrs = ds[parent_name].attrs.copy() if parent_name in ds else {}
        canonical_attrs = sweep_vars_attrs_f.get(out_name, {}).copy()
        attrs = {**parent_attrs, **canonical_attrs}
        attrs = add_correction_step(
            parent_attrs=attrs, step="attc_dbz",
            parent=parent_name,
            params={"method": attc_method,
                    "phidp0_input": phidp0,
                    "phidp0_calc": float(param_atc[15]),
                    "coeff_a": coeff_a,
                    "coeff_b": coeff_b,
                    "coeff_alpha": coeff_alpha,
                    "niter": niter,
                    "mlyr_top_km (mean)": float(mlyr_top.mean()),
                    "mlyr_bottom_km (mean)": (
                        float(mlyr_btm.mean()) if mlyr_btm is not None else None),
                    "mlyr_thickness_km (mean)": float(mlyr_thk.mean()),
                    },
            outputs=[out_name],
            mode="overwrite" if replace_vars else "preserve",
            module_provenance="towerpy.attc.attc.attenuation_correction_zh")
        da = xr.DataArray(arr, dims=(names["azi"], names["rng"]),
                          coords={names["azi"]: ds[names["azi"]],
                                  names["rng"]: ds[names["rng"]]},
                          attrs=attrs,)
        ds_out = safe_assign_variable(ds_out, out_name, da)
        created_vars.append(out_name)
    # Dataset-level provenance
    #TODO: add step_description
    extra = {"step_description": ''}
    ds_out = record_provenance(
        ds_out, step="attenuation_correction_dbz",
        inputs=[names["ZH"], names["PHIDP"], names["RHOHV"]],
        outputs=created_vars,
        parameters={
            "method": attc_method,
            "phidp0_input": phidp0,
            "phidp0_calc": float(param_atc[15]),
            "coeff_a": coeff_a,
            "coeff_b": coeff_b,
            "coeff_alpha": coeff_alpha,
            "niter": niter,
            "mlyr_top_km (mean)": float(mlyr_top.mean()),
            "mlyr_bottom_km (mean)": (
                float(mlyr_btm.mean()) if mlyr_btm is not None else None),
            "mlyr_thickness_km (mean)": float(mlyr_thk.mean()),
            "merge_into_ds": bool(merge_into_ds),
            "replace_vars": bool(replace_vars),
            "modify_output": modify_output}, extra_attrs=extra,
        module_provenance="towerpy.attc.attc.attenuation_correction_zh")
    return ds_out


def _theoretical_zdr(zh, model, params):
    """
    Compute theoretical ZDR corresponding to a given ZH, using either the
    linear or exponential ZH–ZDR model.

    Parameters
    ----------
    zh : float
        Mean reflectivity (dBZ) at the far side of the rain cell.
    model : {"linear", "exp"}
        ZH–ZDR relationship type.
    params : dict
        Dictionary containing the required coefficients:
            For linear:
                "ZH_lower_lim"
                "ZH_upper_lim"
                "coeff_a"
                "coeff_b"
                "zdr_max"
            For exp:
                "coeff_a"
                "coeff_b"

    Returns
    -------
    zdr_theoretical : float
    """

    if np.isnan(zh):
        return np.nan

    if model == "linear":
        zh_low  = params["ZH_lower_lim"]
        zh_high = params["ZH_upper_lim"]
        a = params["coeff_a"]
        b = params["coeff_b"]
        zdr_max = params["zdr_max"]

        if zh <= zh_low:
            return 0.0
        elif zh_low < zh <= zh_high:
            return a * zh - b
        elif zh > zh_high:
            return zdr_max
        else:
            return np.nan

    elif model == "exp":
        a = params["coeff_a"]
        b = params["coeff_b"]
        return a * (zh ** b)

    else:
        raise ValueError(f"Unknown ZH–ZDR model: {model}")


def _find_rain_cells(zdr_mvav, minbins, mask):
    """
    Identify rain cells using the mask from np.ma.convolve

    Parameters
    ----------
    zdr_mvav : 1D ndarray
        Moving-average filtered ZDR values (data only).
    mask : 1D boolean array
        Mask from masked convolution (True = invalid).
    minbins : int
        Minimum contiguous length required.

    Returns
    -------
    longest : (idxrs, idxrf) or None
    full_span : (idxrs_full, idxrf_full) or None
    """
    isnan = mask

    if np.all(isnan):
        return None, None

    segments = []
    in_seg = False
    start = None
    n = len(zdr_mvav)
    for i, flag in enumerate(isnan):
        if not flag and not in_seg:
            in_seg = True
            start = i
        elif flag and in_seg:
            in_seg = False
            end = i - 1
            if end - start + 1 >= minbins:
                segments.append((start, end))

    if in_seg:
        end = n - 1
        if end - start + 1 >= minbins:
            segments.append((start, end))

    if not segments:
        return None, None

    lengths = [end - start + 1 for start, end in segments]
    i_max = int(np.argmax(lengths))
    longest = segments[i_max]
    full_span = (segments[0][0], segments[-1][1])

    return longest, full_span


def _optimise_beta(zdrmrf, zdrerf, phidp_rf, phidp_rs, pia_segment, alpha_rf,
                   alpha_min_ray, coeff_beta, beta_alpha_ratio, bracket,
                   second_attempt=False):
    """
    beta optimisation.

    Returns
    -------
    beta : float
    status : int
        1 = optimised (primary)
        2 = optimised (secondary)
        0 = fixed ratio fallback
    """
    # 0. BRI -> no optimisation at all
    if bracket is None:
        return beta_alpha_ratio, 0
    # 1. Initial betai
    dphidp = phidp_rf - phidp_rs
    if dphidp == 0 or np.isnan(dphidp):
        return beta_alpha_ratio, 0
    betai = abs(zdrmrf - zdrerf) / dphidp
    # 2. ZDR corrected using betai
    pia_mean = np.nanmean(pia_segment)
    if alpha_rf == 0 or np.isnan(alpha_rf):
        return beta_alpha_ratio, 0
    zdrirfpia = zdrmrf + (betai / alpha_rf) * pia_mean
    # If perfect match -> no root solving, just clamp betai
    if abs(zdrirfpia - zdrerf) == 0:
        beta = betai
        beta = max(coeff_beta[0], min(coeff_beta[1], beta))
        if np.isnan(beta):
            beta = coeff_beta[2]
        return beta, 1 if not second_attempt else 2
    # 3. Root-solving for β
    def f(betaif):
        return zdrmrf + (betaif / alpha_rf) * pia_mean - zdrerf
    # x0: np.nanmin(alphacopy[i]) * beta_alpha_ratio
    x0 = alpha_min_ray * beta_alpha_ratio
    try:
        sol = optimize.root_scalar(f, bracket=bracket, x0=x0,
                                   method="brentq")
        beta = sol.root
    except Exception:
        # caller decides whether to retry with full-span or fall back
        raise
    # 4. Clamp and finalise
    beta = max(coeff_beta[0], min(coeff_beta[1], beta))
    if np.isnan(beta):
        beta = coeff_beta[2]
    return beta, 1 if not second_attempt else 2


def _attc_zdr_1d(zh, zdr, rhohv, phidp, pia, ah, alpha, ml_bottom_gate, cclass,
                 coeff_beta, beta_alpha_ratio, rhv_thld, mov_avrgf_len,
                 minbins, p2avrf, params, attc_method):
    """
    Per‑ray ZDR attenuation correction.

    Returns
    -------
    zdr_corr : 1D array
    adp      : 1D array
    beta_arr : 1D array
    status   : int (0=fixed, 1=optimised_1, 2=optimised_2)
    """
    ng = len(zdr)
    gates = np.arange(ng)

    below_ml = gates <= ml_bottom_gate
    met = (cclass == 0)
    valid_alpha = (alpha > 0)
    valid = below_ml & met & valid_alpha
    # 1. fixed β/α branch (BRI or fallback)
    def _fixed_beta_branch():
        # ZDR correction along the whole ray
        zdr_corr_loc = zdr + beta_alpha_ratio * pia
        # ADP and β along the whole ray, then restricted
        adp_loc = beta_alpha_ratio * ah
        beta_arr_loc = alpha * beta_alpha_ratio
        # Above ML -> 0 for ADP and β
        adp_loc[~below_ml] = 0.0
        beta_arr_loc[~below_ml] = 0.0
        # Non‑met -> NaN
        zdr_corr_loc[~met] = np.nan
        adp_loc[~met] = np.nan
        beta_arr_loc[~met] = np.nan
        return zdr_corr_loc, adp_loc, beta_arr_loc
    # 1a. No α below ML -> fixed ratio
    if not np.any(valid):
        return *_fixed_beta_branch(), 0
    # 1b. BRI -> always fixed ratio
    if attc_method == "BRI":
        return *_fixed_beta_branch(), 0
    # 2. ρHV filtering + moving average
    mask_rhv = (rhohv < rhv_thld)
    zdr_ma = np.ma.array(zdr, mask=mask_rhv)
    zh_ma = np.ma.array(zh,  mask=mask_rhv)
    kernel = np.ones(mov_avrgf_len) / mov_avrgf_len
    zdr_mv_ma = np.ma.convolve(zdr_ma, kernel, mode="same")
    zh_mv_ma = np.ma.convolve(zh_ma,  kernel, mode="same")
    zdr_mv = zdr_mv_ma.data
    zh_mv = zh_mv_ma.data
    zdr_mv_mask = zdr_mv_ma.mask
    # 3. Rain cells
    longest, full_span = _find_rain_cells(zdr_mv, minbins, mask=zdr_mv_mask)
    if longest is None:
        return *_fixed_beta_branch(), 0
    idxrs, idxrf = longest
    # 4. Means at far side
    zhcrf = np.nanmean(zh_mv[idxrf - p2avrf + 1 : idxrf + 1])
    zdrmrf = np.nanmean(zdr_mv[idxrf - p2avrf + 1 : idxrf + 1])
    # 5. Theoretical ZDR
    zdrerf = _theoretical_zdr(zhcrf, params["ZH-ZDR model"], params)
    if zdrerf <= zdrmrf:
        return *_fixed_beta_branch(), 0
    # 6. ABRI optimisation
    alpha_min_ray = np.nanmin(alpha[valid])
    beta = beta_alpha_ratio
    status = 0
    try:
        beta, status = _optimise_beta(
            zdrmrf, zdrerf, phidp[idxrf], phidp[idxrs],
            pia[idxrf - p2avrf + 1 : idxrf + 1], alpha[idxrf], alpha_min_ray,
            coeff_beta, beta_alpha_ratio, bracket=coeff_beta[:2])
    except ValueError:
        if full_span is not None:
            idxrs2, idxrf2 = full_span
            zhcrf2 = np.nanmean(zh_mv[idxrf2 - p2avrf + 1 : idxrf2 + 1])
            zdrmrf2 = np.nanmean(zdr_mv[idxrf2 - p2avrf + 1 : idxrf2 + 1])
            zdrerf2 = _theoretical_zdr(zhcrf2, params["ZH-ZDR model"], params)
            try:
                beta, status = _optimise_beta(
                    zdrmrf2, zdrerf2, phidp[idxrf2], phidp[idxrs2],
                    pia[idxrf2 - p2avrf + 1 : idxrf2 + 1], alpha[idxrf2],
                    alpha_min_ray, coeff_beta, beta_alpha_ratio,
                    bracket=coeff_beta[:2], second_attempt=True)
            except ValueError:
                return *_fixed_beta_branch(), 0
        else:
            return *_fixed_beta_branch(), 0
    if status == 0:
        return *_fixed_beta_branch(), 0
    # 7. Apply optimised correction (ABRI)
    # ZDR correction along the whole ray
    corr_term = np.zeros_like(zdr)
    mask_alpha = (alpha > 0)
    corr_term[mask_alpha] = (beta / alpha[mask_alpha]) * pia[mask_alpha]
    zdr_corr = zdr + corr_term
    # ADP along the whole ray, then restricted
    adp = np.zeros_like(zdr)
    adp[mask_alpha] = (beta / alpha[mask_alpha]) * ah[mask_alpha]
    # β along the whole ray (scalar per ray), then restricted
    beta_arr = np.zeros_like(alpha)
    beta_arr[mask_alpha] = beta
    # Above ML -> 0 for ADP and β
    adp[~below_ml] = 0.0
    beta_arr[~below_ml] = 0.0
    # Non‑met -> NaN
    zdr_corr[~met] = np.nan
    adp[~met] = np.nan
    beta_arr[~met] = np.nan

    return zdr_corr, adp, beta_arr, status


def attenuation_correction_zdr(dsattvars, cclass, inp_names=None,
                               attc_method="BRI", mlyr_top=5., mlyr_thk=0.75,
                               mlyr_btm=None, coeff_beta=[0.008, 0.1, 0.04],
                               beta_alpha_ratio=0.265, rhv_thld=0.985,
                               mov_avrgf_len=9, minbins=10, p2avrf=5,
                               zh_zdr_model="linear", rparams=None,
                               merge_into_ds=False, replace_vars=False,
                               modify_output=None,):
    r"""
    Perform attenuation correction of :math:`Z_{DR}` using the algorithm
    described in Bringi et al. (2001).
    
    Parameters
    ----------
    dsattvars : xarray.Dataset
        Dataset containing polarimetric variables filtered by noise, along
        with the polar coordinates (range, azimuth, elevation).
    cclass : xarray.Dataarray
        Clutter, noise and weather echoes classification. Attributes must
        describe the flags used for classification, e.g.:
        {'pcpn': 0, 'noise': 3, 'clutter': 5}
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{"azi": "azimuth", "rng": "range", "elv": "elevation", "ZH": "DBZH",
        "ZDR": "ZDR", "RHOHV": "RHOHV", "PHIDP": "PHIDP", "PIA": "PIA",
        "AH": "AH", "ALPHA": "ALPHA"}``
    attc_method : str, default "BRI"
        Attenuation‑correction method. Supported options are:

            'ABRI': Bringi method with optimised β.

            'BRI': Bringi method with fixed β/α ratio.
    mlyr_top, mlyr_thk, mlyr_btm : float or array, optional
        Heights of the melting layer boundaries, in km. Only gates below the
        melting layer bottom (i.e. the rain region below the melting layer)
        are included in the computation.
    coeff_beta : list or tuple, default [0.008, 0.1, 0.04]
        [Min, max, fixed value] of coeff β. These bounds are used to find the
        optimum value of β. Default values are derived for C‑band.
    beta_alpha_ratio : float, default 0.265
        Ratio β / α used in A_DP = (β / α) A_H.
    rhv_thld : float, default 0.985
        Minimum value of ρ_HV expected in the rain medium.
    mov_avrgf_len : int, default 9
        Odd number used to apply a moving average filter to each beam and
        smooth the signal.
    minbins : int, default 10
        Minimum number of bins related to the length of each rain cell
        along the beam.
    p2avrf : int, default 5
        Number of bins to average on the far side of the rain cell.
    zh_zdr_model : {"linear", "exp"}, default "linear"
        Model used to estimate the intrinsic (unattenuated) Z_DR–Z_H
        relationship.
    rparams : dict, optional
        Additional parameters controlling the Z_H–Z_DR relationship.
        Defaults depend on the selected model. See Notes for details.
    merge_into_ds : bool, default False
        If True, corrected variables are merged into a full copy of the input
        dataset. If False, return a dataset containing only the corrected
        outputs (same coords and attrs).
    replace_vars : bool, default False
        If True, overwrite existing variables (e.g. ZDR, ADP, BETA) where
        applicable. If False, corrected variables receive an "_ATTC" suffix
        unless explicit names are provided via ``modify_output``.
    modify_output : bool | list[str] | dict[str, str] | None
        Controls which logical outputs are written and how they are named.
        Logical outputs are: ``"ZDR_ATTC"``, ``"ADP"``, ``"BETA"``.
        
        - None: write all logical outputs with default naming rules.
        - True: same as None (all outputs).
        - list[str]: only write the listed logical outputs.
        - dict[str, str]: map logical output names to explicit output names.

    Returns
    -------
    xarray.Dataset
        Dataset containing the attenuation‑corrected variables:

        ZDR : dB
            Attenuation-corrected differential reflectivity.
        ADP : dB/km
            Specific differential attenuation.
        BETA :
           parameter :math:`\beta` optimised for each beam.

    Notes
    -----
    * This function operates in native polar radar coordinates.
    * This method assumes that :math:`Z_H` has already been corrected for
      attenuation, e.g. using the methods described in [1]_.
    * The attenuation is computed up to a user-defined melting level
      height.
    * ZH–ZDR relationship - Linear model:
        .. math::
            \overline{Z}_{DR} =
            \Biggl\{ 0 \rightarrow  \overline{Z_H}(r_m)<=Z_H(lowerlim) \\
                    a*Z_H-b \rightarrow Z_H(lowerlim)<Z_H(r_m)<=Z_H(upperlim) \\
                        Z_{DR}(max) \rightarrow Z_H(r_m)>Z_H(upperlim) \Biggl\}
        where:
            - ZH_lower_lim: 20 dBZ
            - ZH_upper_lim: 45 dBZ
            - coeff_a: 0.048
            - coeff_b: 0.774
            - zdr_max: 1.4

    * ZH–ZDR relationship - Exponential model:
        .. math::
            \overline{Z}_{DR} = \Biggl\{ a*Z_H^{b} \Biggl\}
        where:
            - coeff_a: 0.00012
            - coeff_b: 2.5515

    References
    ----------
    .. [1] Rico-Ramirez, M. A. (2012). Adaptive attenuation correction
        techniques for C-Band polarimetric weather radars. IEEE Transactions
        on Geoscience and Remote Sensing, 50(12), 5061–5071.
        https://doi.org/10.1109/tgrs.2012.2195228

    .. [2] Bringi, V., Keenan, T., & Chandrasekar, V. (2001). Correcting C-band
        radar reflectivity and differential reflectivity data for rain
        attenuation: a self-consistent method with constraints. IEEE
        Transactions on Geoscience and Remote Sensing, 39(9), 1906–1915.
        https://doi.org/10.1109/36.951081

    .. [3] Gou, Y., Chen, H., & Zheng, J. (2019). An improved self-consistent
        approach to attenuation correction for C-band polarimetric radar
        measurements and its impact on quantitative precipitation estimation.
        Atmospheric Research, 226, 32–48.
        https://doi.org/10.1016/j.atmosres.2019.03.006

    .. [4] Park, S., Bringi, V. N., Chandrasekar, V., Maki, M., & Iwanami, K.
        (2005). Correction of radar reflectivity and differential reflectivity
        for rain attenuation at X Band. Part I: Theoretical and Empirical
        basis. Journal of Atmospheric and Oceanic Technology, 22(11), 1621–1632.
        https://doi.org/10.1175/jtech1803.1
    """
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # 1. Resolve variable names
    defaults = {"azi": "azimuth", "rng": "range", "elv": "elevation",
                "ZH": "DBZH", "ZDR": "ZDR", "RHOHV": "RHOHV",
                "PHIDP": "PHIDP", "PIA": "PIA", "AH": "AH", "ALPHA": "ALPHA"}
    names = {**defaults, **(inp_names or {})}
    # 2. Remap classification to {0,3,5}
    echoesID = {"pcpn": 0, "noise": 3, "clutter": 5}
    flagsID = cclass.attrs["flags"]
    cclass_arr = cclass.copy()
    cclass_arr = xr.where(cclass_arr == flagsID["pcpn"], echoesID["pcpn"],
                          cclass_arr)
    cclass_arr = xr.where(cclass_arr == flagsID["noise"], echoesID["noise"],
                          cclass_arr)
    cclass_arr = xr.where(cclass_arr == flagsID["clutter"], echoesID["clutter"],
                          cclass_arr)
    # 3. Attach melting layer and compute bottom gate index per ray
    ds = attach_melting_layer(dsattvars, mlyr_top=mlyr_top,
                              mlyr_bottom=mlyr_btm,
                              mlyr_thickness=mlyr_thk, units="km",
                              source="user-defined", method="zdr_attc",
                              overwrite=True)
    ml_bottom_km = ds.MLYRTOP - ds.MLYRTHK
    ml_bottom_m = convert(ml_bottom_km, "m")
    mlb_grid = ml_bottom_m.broadcast_like(ds[names["ZH"]])
    beam_height = ds.beamc_height  # dims: (azimuth, range)
    # Correct per-ray ML bottom gate index
    ml_bottom_gate = np.abs(beam_height - mlb_grid).argmin(dim=names["rng"])
    # 4. Build ZH–ZDR model params dict
    params = {"ZH-ZDR model": zh_zdr_model,
              "ZH_lower_lim": 20.0,
              "ZH_upper_lim": 45.0,
              "coeff_a": 0.048,
              "coeff_b": 0.774,
              "zdr_max": 1.4}
    if zh_zdr_model == "exp":
        params["coeff_a"] = 0.00012
        params["coeff_b"] = 2.5515
    if rparams is not None:
        params.update(rparams)
    # 5. Apply per-ray ZDR attenuation correction
    zdr_corr, adp, beta, status = xr.apply_ufunc(
        _attc_zdr_1d, ds[names["ZH"]], ds[names["ZDR"]], ds[names["RHOHV"]],
        ds[names["PHIDP"]], ds[names["PIA"]], ds[names["AH"]],
        ds[names["ALPHA"]],
        ml_bottom_gate,          # shape (azimuth,)
        cclass_arr,
        kwargs=dict(
            coeff_beta=coeff_beta,
            beta_alpha_ratio=beta_alpha_ratio,
            rhv_thld=rhv_thld,
            mov_avrgf_len=mov_avrgf_len,
            minbins=minbins,
            p2avrf=p2avrf,
            params=params,
            attc_method=attc_method),
        input_core_dims=[
            [names["rng"]],      # ZH
            [names["rng"]],      # ZDR
            [names["rng"]],      # RHOHV
            [names["rng"]],      # PHIDP
            [names["rng"]],      # PIA
            [names["rng"]],      # AH
            [names["rng"]],      # ALPHA
            [],                  # <-- ml_bottom_gate is a scalar per ray
            [names["rng"]],      # cclass_arr
        ],
        output_core_dims=[[names["rng"]], [names["rng"]], [names["rng"]], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, int])
    # 6. Build output dataset and attrs
    outputs = {"ZDR_ATTC": zdr_corr, "ADP": adp, "BETA": beta}
    # Determine which logical outputs to include
    logical_keys = list(outputs.keys())
    if modify_output is None or modify_output is True:
        selected = logical_keys
    elif isinstance(modify_output, list):
        selected = modify_output
    elif isinstance(modify_output, dict):
        selected = list(modify_output.keys())
    else:
        raise TypeError("modify_output must be None, bool, list, or dict")
    # Start from original dataset or empty dataset WITH coords + attrs
    if merge_into_ds:
        ds_out = ds.copy()
    else:
        ds_out = xr.Dataset(coords=ds.coords, attrs=ds.attrs.copy())
    created_vars = []
    for logical_name in selected:
        arr = outputs[logical_name]
        # Determine parent variable
        if logical_name == "ZDR_ATTC":
            parent_name = names["ZDR"]
        else:
            parent_name = logical_name
        # Determine output name
        if isinstance(modify_output, dict):
            out_name = modify_output.get(logical_name, logical_name)
        else:
            if logical_name == "ZDR_ATTC":
                if replace_vars:
                    out_name = names["ZDR"]
                else:
                    out_name = f"{names['ZDR']}_ATTC"
            else:
                if replace_vars:
                    out_name = parent_name
                else:
                    # Only suffix if there is a collision
                    if parent_name in ds_out:
                        out_name = f"{parent_name}_ATTC"
                    else:
                        out_name = parent_name
        created_vars.append(out_name)
        # Build attrs: parent + canonical
        parent_attrs = ds[parent_name].attrs.copy() if parent_name in ds else {}
        canonical_attrs = sweep_vars_attrs_f.get(out_name, {}).copy()
        attrs = {**parent_attrs, **canonical_attrs}
        if logical_name == "ZDR_ATTC":
            attrs["description"] = (
                "Corrected differential reflectivity for attenuation in rain.")
        elif logical_name == "ADP":
            attrs["description"] = (
                "Specific differential attenuation derived from attenuation correction.")
        elif logical_name == "BETA":
            attrs["description"] = (
                "Optimised β parameter for ZDR attenuation correction.")
        # Add correction step provenance
        attrs = add_correction_step(
            parent_attrs=attrs, step="attc_zdr",
            parent=parent_name,
            params={"method": attc_method, "coeff_beta": coeff_beta,
                    "beta_alpha_ratio": beta_alpha_ratio, "rhv_thld": rhv_thld,
                    "mov_avrgf_len": mov_avrgf_len, "minbins": minbins,
                    "p2avrf": p2avrf, "zh_zdr_model": zh_zdr_model,
                    "rparams": rparams,
                    "mlyr_top_km (mean)": float(mlyr_top.mean()),
                    "mlyr_bottom_km (mean)": (
                        float(mlyr_btm.mean()) if mlyr_btm is not None else None),
                    "mlyr_thickness_km (mean)": float(mlyr_thk.mean())},
            outputs=[out_name],
            mode="overwrite" if replace_vars else "preserve",
            module_provenance="towerpy.attc.attc.attenuation_correction_zdr")
        # Create DataArray and assign
        da = xr.DataArray(arr, dims=(names["azi"], names["rng"]),
                          coords={names["azi"]: ds[names["azi"]],
                                  names["rng"]: ds[names["rng"]]},
                          attrs=attrs)
        ds_out = safe_assign_variable(ds_out, out_name, da)
    # Dataset-level provenance
    #TODO: add step_description
    extra = {"step_description": ""}
    ds_out = record_provenance(
        ds_out, step="attenuation_correction_zdr",
        inputs=[names["ZH"], names["ZDR"], names["PHIDP"],
                names["RHOHV"], names["PIA"], names["AH"], names["ALPHA"]],
        outputs=created_vars,
        parameters={"method": attc_method,
                    "coeff_beta": coeff_beta,
                    "beta_alpha_ratio": beta_alpha_ratio,
                    "rhv_thld": rhv_thld,
                    "mov_avrgf_len": mov_avrgf_len,
                    "minbins": minbins,
                    "p2avrf": p2avrf,
                    "zh_zdr_model": zh_zdr_model,
                    "rparams": rparams,
                    "mlyr_top_km (mean)": float(mlyr_top.mean()),
                    "mlyr_bottom_km (mean)": (
                        float(mlyr_btm.mean()) if mlyr_btm is not None else None),
                    "mlyr_thickness_km (mean)": float(mlyr_thk.mean()),
                    "merge_into_ds": bool(merge_into_ds),
                    "replace_vars": bool(replace_vars),
                    "modify_output": modify_output}, extra_attrs=extra,
        module_provenance="towerpy.attc.attc.attenuation_correction_zdr")
    return ds_out
