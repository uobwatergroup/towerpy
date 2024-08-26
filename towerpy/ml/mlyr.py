"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

from functools import reduce
from itertools import product
import numpy as np
from ..base import TowerpyError
from ..utils import radutilities as rut
from ..datavis import rad_display
from ..datavis import rad_interactive


class MeltingLayer:
    """
    A class to determine the melting layer using weather radar data.

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
    ml_top : float
        Top of the detected melting layer, in km.
    ml_bottom : float
        Bottom of the detected melting layer, in km.
    ml_thickness : float
        Thickness of the detected melting layer, in km.
    """

    def __init__(self, radobj):
        self.elev_angle = radobj.elev_angle
        self.file_name = radobj.file_name
        self.scandatetime = radobj.scandatetime
        self.site_name = radobj.site_name
        if hasattr(radobj, "profs_type"):
            self.profs_type = radobj.profs_type
        else:
            self.profs_type = 'user-defined'

    def findpeaksboundaries(profile, pheight, param_w=0):
        """
        Find peaks inside a profile using signal processing.

        Parameters
        ----------
        profile : array
            Radar profile built from high elevation scans.
        pheight : array
            Heights correspondig to the radar profiles, in km.

        Returns
        -------
        peaks_idx : dict
            Index values of the peaks found within the profiles.

        """
        import scipy.signal as scs
        peaks = {'prfpks': scs.find_peaks(profile, height=(None, None),
                                          width=(None, None), rel_height=1,
                                          prominence=(None, None)),
                 'prfbnd': scs.find_peaks(-profile, height=(None, None),
                                          width=(None, None), rel_height=1,
                                          prominence=(None, None)),
                 }
        peaks['prfpks'][1]['left_ips'] = np.interp(
            peaks['prfpks'][1]['left_ips'], np.arange(0, len(profile)),
            pheight)
        peaks['prfpks'][1]['right_ips'] = np.interp(
            peaks['prfpks'][1]['right_ips'], np.arange(0, len(profile)),
            pheight)
        if not peaks['prfpks'][0].size:
            pkprf = np.nan
            maxpeakh = np.nan
        else:
            # Locate peaks maxima
            pkprf = (peaks['prfpks']
                     [0][peaks['prfpks'][1]['peak_heights'].argmax()])
            maxpeakh = peaks['prfpks'][1]['peak_heights'].max()
        if not peaks['prfbnd'][0].size or not peaks['prfpks'][0].size:
            buprf = np.nan
            btprf = np.nan
            flprf = np.nan
            bottp = np.nan
            maxpeakh = np.nan
            mlpeak = np.nan
        else:
            # Locate peak top boundarie
            if pkprf >= peaks['prfbnd'][0][-1]:
                buprf = peaks['prfbnd'][0][-1]
                btprf = peaks['prfbnd'][0][-1]
            else:
                idx_peak = peaks['prfpks'][1]['peak_heights'].argmax()
                if param_w == 0:
                    param_w = 0.05
                elif param_w >= 0.3:
                    param_w = 0.3
                param_w2 = maxpeakh * param_w
                aux = [-i if -i <= param_w2 else np.nan
                       for i in peaks['prfbnd'][1]['peak_heights']]
                if len(aux[idx_peak+1:]) < 1:
                    buprf = peaks['prfbnd'][0][np.searchsorted(
                        peaks['prfbnd'][0], pkprf)]
                else:
                    buprf = peaks['prfbnd'][0][idx_peak + (
                        np.isnan(aux[idx_peak+1:]).argmin(axis=0))+1]
                if len(aux[:idx_peak-1]) < 1:
                    btprf = peaks['prfbnd'][0][np.searchsorted(
                        peaks['prfbnd'][0], pkprf)-1]
                else:
                    btprf = peaks['prfbnd'][0][idx_peak-1-np.isnan(
                        np.flip(aux[:idx_peak-1])).argmin(axis=0)-1]
            if buprf <= pkprf:
                buprf = np.nan  # check
                flprf = np.nan
            else:
                flprf = pheight[buprf]
            if btprf >= pkprf:
                btprf = np.nan
                bottp = np.nan  # check
            if not np.isnan(btprf):
                bottp = pheight[btprf]
            if not np.isnan(maxpeakh):
                mlpeak = pheight[pkprf]
            else:
                mlpeak = np.nan
        peaks_idx = {'idxmax': pkprf, 'idxtop': buprf, 'idxbot': btprf,
                     'mltop': flprf, 'mlbtm': bottp, 'peakmaxvalue': maxpeakh,
                     'mlpeak': mlpeak}
        return peaks_idx

    def ml_detection(self, pol_profs, min_h=0, max_h=5, zhnorm_min=5.,
                     zhnorm_max=60., rhvnorm_min=0.85, rhvnorm_max=1.,
                     phidp_peak='left', gradv_peak='left', param_k=0.05,
                     param_w=0.75, comb_id=None, plot_method=False):
        r"""
        Detect melting layer signatures within polarimetric VPs/QVPs.

        Parameters
        ----------
        pol_profs : dict
            Polarimetric profiles of radar variables.
        min_h : float, optional
            Minimum height of usable data within the polarimetric profiles.
            The default is 0.
        max_h : float, optional
            Maximum height to search for the bright band peak.
            The default is 5.
        zhnorm_min : float, optional
            Min value of :math:`Z_{H}` to use for the min-max normalisation.
            The default is 5.
        zhnorm_max : float, optional
            Max value of :math:`Z_{H}` to use for the min-max normalisation.
            The default is 60.
        rhvnorm_min : float, optional
            Min value of :math:`\rho_{HV}` to use for the min-max
            normalisation. The default is 0.85.
        rhvnorm_max : float, optional
            Max value of :math:`\rho_{HV}` to use for the min-max
            normalisation. The default is 1.
        phidp_peak : str, optional
            Direction of the peak in :math:`\Phi_{DP}` related to the ML. The
            method described in [1]_ assumes that the peak points to the left,
            (see Figure 3 in the paper) but this can be changed using this
            argument.
        gradv_peak : str, optional
            Direction of the peak in :math:`gradV` related to the ML. The
            method described in [1]_ assumes that the peak points to the left,
            (see Figure 3 in the paper) but this can be changed using this
            argument.
        param_k : float, optional
            Threshold related to the magnitude of the peak used to detect the
            ML. The default is 0.05.
        param_w : float, optional
            Weighting factor used to sharpen the peak within the profile.
            The default is 0.75.
        comb_id : int, optional
            Identifier of the combination selected for the ML detection.
            If None, the method provides all the possible combinations of
            polarimetric variables for VPs/QVPs. The default is None.
        plot_method : bool, optional
            Plot the ML detection method. The default is False.

        Notes
        -----
        1. Based on the methodology described in [1]_

        References
        ----------
        .. [1] Sanchez-Rivas, D. and Rico-Ramirez, M. A. (2021)
            "Detection of the melting level with polarimetric weather radar"
            in Atmospheric Measurement Techniques Journal, Volume 14, issue 4,
            pp. 2873–2890, 13 Apr 2021 https://doi.org/10.5194/amt-14-2873-2021

        """
        min_hidx = rut.find_nearest(pol_profs.georef['profiles_height [km]'],
                                    min_h)
        max_hidx = rut.find_nearest(pol_profs.georef['profiles_height [km]'],
                                    max_h)

        # The user shall use the combID described in the paper, thus it is
        # necessary to adjust to python indexing.
        if comb_id is not None:
            comb_idpy = comb_id-1
        else:
            comb_idpy = None

        if self.profs_type == 'VPs':
            if 'ZH [dBZ]' and 'rhoHV [-]' in pol_profs.vps:
                profzh = pol_profs.vps['ZH [dBZ]'].copy()
                profrhv = pol_profs.vps['rhoHV [-]'].copy()
            else:
                raise TowerpyError(r'Profiles of $Z_H$ and $\rho_{HV}$ are '
                                   'required to run this function')
            if 'ZDR [dB]' in pol_profs.vps:
                profzdr = pol_profs.vps['ZDR [dB]'].copy()
            else:
                print(r'$Z_{DR}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profzdr = np.ones_like(profzh)
            if 'PhiDP [deg]' in pol_profs.vps:
                if phidp_peak == 'left':
                    profpdp = pol_profs.vps['PhiDP [deg]'].copy()
                elif phidp_peak == 'right':
                    profpdp = pol_profs.vps['PhiDP [deg]'].copy()
                    profpdp *= -1
            else:
                print(r'$Phi_{DP}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profpdp = np.ones_like(profzh)
            if 'gradV [dV/dh]' in pol_profs.vps:
                if gradv_peak == 'left':
                    profdvel = pol_profs.vps['gradV [dV/dh]'].copy()
                elif gradv_peak == 'right':
                    profdvel = pol_profs.vps['gradV [dV/dh]'].copy()
                    profdvel *= -1
            else:
                print('gradV [dV/dh] profile was not found. A dummy one was '
                      'built to run the method.')
                profdvel = np.ones_like(profzh)
        elif self.profs_type == 'QVPs':
            if 'ZH [dBZ]' and 'rhoHV [-]' in pol_profs.qvps:
                profzh = pol_profs.qvps['ZH [dBZ]'].copy()
                profrhv = pol_profs.qvps['rhoHV [-]'].copy()
            else:
                raise TowerpyError(r'Profiles of $Z_H$ and $\rho_{HV}$ are '
                                   'required to run this function')
            if 'ZDR [dB]' in pol_profs.qvps:
                profzdr = pol_profs.qvps['ZDR [dB]'].copy()
            else:
                print(r'$Z_{DR}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profzdr = np.ones_like(profzh)
            if 'PhiDP [deg]' in pol_profs.qvps:
                if phidp_peak == 'left':
                    profpdp = pol_profs.qvps['PhiDP [deg]'].copy()
                elif phidp_peak == 'right':
                    profpdp = pol_profs.qvps['PhiDP [deg]'].copy()
                    profpdp *= -1
            else:
                print(r'$Phi_{DP}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profpdp = np.ones_like(profzh)
        elif self.profs_type == 'RD-QVPs':
            if 'ZH [dBZ]' and 'rhoHV [-]' in pol_profs.rd_qvps:
                profzh = pol_profs.rd_qvps['ZH [dBZ]'].copy()
                profrhv = pol_profs.rd_qvps['rhoHV [-]'].copy()
            else:
                raise TowerpyError(r'At least $Z_H$ and $\rho_{HV}$ are '
                                   + 'required to run this function')
            if 'ZDR [dB]' in pol_profs.rd_qvps:
                profzdr = pol_profs.rd_qvps['ZDR [dB]'].copy()
            else:
                print(r'$Z_{DR}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profzdr = np.ones_like(profzh)
            if 'PhiDP [deg]' in pol_profs.rd_qvps:
                if phidp_peak == 'left':
                    profpdp = pol_profs.rd_qvps['PhiDP [deg]'].copy()
                elif phidp_peak == 'right':
                    profpdp = pol_profs.rd_qvps['PhiDP [deg]'].copy()
                    profpdp *= -1
            else:
                print(r'$Phi_{DP}$ profile was not found. A dummy one was '
                      'built to run the method.')
                profpdp = np.ones_like(profzh)

        # Normalise ZH and rhoHV
        profzh[profzh < zhnorm_min] = zhnorm_min
        profzh[profzh > zhnorm_max] = zhnorm_max
        profzh_norm = rut.normalisenanvalues(profzh, zhnorm_min, zhnorm_max)
        profrhv[profrhv < rhvnorm_min] = rhvnorm_min
        profrhv[profrhv > rhvnorm_max] = rhvnorm_max
        profrhv_norm = rut.normalisenanvalues(profrhv, rhvnorm_min,
                                              rhvnorm_max)

        # Combine ZH and rhoHV (norm) to create a new profile
        profcombzh_rhv = profzh_norm[min_hidx:max_hidx]*(
            1-profrhv_norm[min_hidx:max_hidx])

        # Detect peaks within the new profile
        pkscombzh_rhv = MeltingLayer.findpeaksboundaries(
            profcombzh_rhv,
            pol_profs.georef['profiles_height [km]'][min_hidx:max_hidx],
            param_w=param_w)

        # If no peaks were found, the profile is classified as No ML signatures
        if (all(value is np.nan for value in pkscombzh_rhv.values())
           or pkscombzh_rhv['peakmaxvalue'] < param_k):
            ttxt_dt = f"{self.scandatetime:%Y-%m-%d %H:%M:%S}"
            print(f'ML signatures could not be found at {ttxt_dt}')
            mlyr = np.nan
            # mlrand = np.nan
            # combin = np.nan
            # idxml_top_it1 = 0
        else:
            peakcombzh_rhv = (pol_profs.georef['profiles_height [km]']
                              [min_hidx:max_hidx][pkscombzh_rhv['idxmax']])
            idxml_btm_it1 = rut.find_nearest(
                pol_profs.georef['profiles_height [km]'], peakcombzh_rhv-.75)+1
            idxml_top_it1 = rut.find_nearest(
                pol_profs.georef['profiles_height [km]'], peakcombzh_rhv+.75)+1
            if peakcombzh_rhv > 4.25:
                idxml_top_it1 = 0
            if idxml_btm_it1 < min_hidx:
                idxml_btm_it1 = min_hidx
            # else:
                # min_hidx = idxml_btm_it1
            if idxml_top_it1 > min_hidx:
                if self.profs_type == 'VPs':
                    n = 5
                    ncomb = [1-rut.normalisenan(
                        profdvel[idxml_btm_it1:idxml_top_it1]),
                             profzh_norm[idxml_btm_it1:idxml_top_it1],
                             rut.normalisenan(
                                 profzdr[idxml_btm_it1:idxml_top_it1]),
                             1-profrhv_norm[idxml_btm_it1:idxml_top_it1],
                             1-rut.normalisenan(
                                 profpdp[idxml_btm_it1:idxml_top_it1])]
                else:
                    n = 4
                    ncomb = [profzh_norm[idxml_btm_it1:idxml_top_it1],
                             rut.normalisenan(
                                 profzdr[idxml_btm_it1:idxml_top_it1]),
                             1-profrhv_norm[idxml_btm_it1:idxml_top_it1],
                             rut.normalisenan(
                                 profpdp[idxml_btm_it1:idxml_top_it1])]
                combin = np.array(list(map(list, product([0, 1],
                                                         repeat=n)))[1:])
                comb_mult = []
                for i, j in enumerate(combin):
                    nfin4 = []
                    [idx] = np.where(combin[i] == 1)
                    for idxcomb in idx:
                        nfin = ncomb[idxcomb]
                        nfin4.append(nfin)
                    nfin5 = reduce(lambda x, y: x*y, nfin4)
                    comb_mult.append(nfin5)

                comb_mult_w = [i-(param_w*(np.gradient(np.gradient(i))))
                               for i in comb_mult]
                mlrand = [MeltingLayer.findpeaksboundaries(
                        i, pol_profs.georef['profiles_height [km]'][idxml_btm_it1:idxml_top_it1],
                        param_w=param_w)
                          for i in comb_mult_w]
                for i, j in enumerate(mlrand):
                    if mlrand[i]['peakmaxvalue'] < param_k:
                        mlrand[i]['mltop'] = np.nan
                mlrandf = [{'ml_top': n['mltop'],
                            'ml_bottom': n['mlbtm'],
                            'ml_peak': n['mlpeak'],
                            'ml_thickness': n['mltop']-n['mlbtm'],
                            'ml_peakv': n['peakmaxvalue']}
                           for n in mlrand]
                if comb_idpy is None:
                    mlyr = mlrandf
                else:
                    mlyr = mlrandf[comb_idpy]
            else:
                mlrand = np.nan
                combin = np.nan
                mlyr = np.nan
                comb_mult = []

            if plot_method:
                rad_interactive.ml_detectionvis(
                    pol_profs.georef['profiles_height [km]'], profzh_norm,
                    profrhv_norm, profcombzh_rhv, pkscombzh_rhv, comb_mult,
                    comb_mult_w, comb_idpy, mlrand, min_hidx, max_hidx,
                    param_k, idxml_btm_it1, idxml_top_it1)

        if isinstance(mlyr, dict):
            self.ml_top = mlyr['ml_top']
            self.ml_bottom = mlyr['ml_bottom']
            self.ml_thickness = mlyr['ml_thickness']
            self.profpeakv = mlyr['ml_peakv']
        else:
            self.ml_top = np.nan
            self.ml_bottom = np.nan
            self.ml_thickness = np.nan
            self.profpeakv = np.nan

    def ml_ppidelimitation(self, rad_georef, rad_params, rad_vars,
                           classid=None, plot_method=False):
        """
        Create a PPI depicting the limits of the melting layer.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used to identify the clutter echoes.
        classid : dict, optional
            Modifies the key/values of the melting layer delimitation
            (regionID). The default are the same as in regionID.
        plot_method : bool, optional
            Plot the results of the ML delimitation. The default is False.

        Attributes
        ----------
        regionID : dict
            Key/values of the rain limits:
                'rain' = 1

                'mlyr' = 2

                'solid_pcp' = 3
        """
        ml_top = self.ml_top
        ml_thickness = self.ml_thickness
        ml_bottom = self.ml_bottom
        self.regionID = {'rain': 1.,
                         'mlyr': 2.,
                         'solid_pcp': 3.}
        if classid is not None:
            self.regionID.update(classid)
        if np.isnan(ml_bottom):
            ml_bottom = ml_top - ml_thickness
        if isinstance(ml_top, (int, float)):
            mlt_idx = [rut.find_nearest(nbh, ml_top)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(ml_top, (np.ndarray, list, tuple)):
            mlt_idx = [rut.find_nearest(nbh, ml_top[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        if isinstance(ml_bottom, (int, float)):
            mlb_idx = [rut.find_nearest(nbh, ml_bottom)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(ml_bottom, (np.ndarray, list, tuple)):
            mlb_idx = [rut.find_nearest(nbh, ml_bottom[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        ashape = np.zeros_like(rad_vars[[i for i in rad_vars.keys()][0]])
        for cnt, azi in enumerate(ashape):
            azi[:mlb_idx[cnt]] = self.regionID['rain']
            azi[mlt_idx[cnt]:] = self.regionID['solid_pcp']
        ashape[ashape == 0] = self.regionID['mlyr']
        self.mlyr_limits = {'pcp_region [cc]': ashape}
        if plot_method:
            rad_display.plot_ppi(rad_georef, rad_params, self.mlyr_limits,
                                 ucmap='tpylc_div_yw_gy_bu')
