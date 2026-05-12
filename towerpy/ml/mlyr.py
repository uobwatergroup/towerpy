"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import copy
from functools import reduce
from itertools import product
import numpy as np
import xarray as xr
import scipy.signal as scs
from ..base import TowerpyError
from ..utils import radutilities as rut
from ..datavis import rad_display
from ..datavis import rad_interactive
from ..io import modeltp as mdtp
from ..utils.radutilities import (safe_assign_variable, find_nearest_index,
                                  record_provenance, add_correction_step)


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

    def __init__(self, radobj=None):
        self.elev_angle = getattr(radobj, 'elev_angle',
                                  None) if radobj else None
        self.file_name = getattr(radobj, 'file_name',
                                 None) if radobj else None
        self.scandatetime = getattr(radobj, 'scandatetime',
                                    None) if radobj else None
        self.site_name = getattr(radobj, 'site_name',
                                 None) if radobj else None
        self.ml_source = getattr(radobj, 'profs_type',
                                 'user-defined') if radobj else 'user-defined'
        self.profs_type = getattr(radobj, 'profs_type',
                                  'user-defined') if radobj else 'user-defined'

    def findpeaksboundaries(profile_v, profile_h, param_k=None):
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
        peaks, peaks_props = scs.find_peaks(profile_v, height=(None, None),
                                            threshold=(None, None),
                                            prominence=(param_k*.7, 2),
                                            width=(None, None),
                                            plateau_size=(None, None),
                                            # rel_height=0.985
                                            rel_height=0.99
                                            )
        peaks_props['left_ips'] = np.interp(
            peaks_props['left_ips'], np.arange(0, len(profile_v)),
            profile_h)
        peaks_props['right_ips'] = np.interp(
            peaks_props['right_ips'], np.arange(0, len(profile_v)),
            profile_h)

        if peaks.size == 0 or np.all(np.isnan(peaks)):
            peaks_idx = {'idxmax': np.nan, 'idxtop': np.nan, 'idxbot': np.nan,
                         'mltop': np.nan, 'mlbtm': np.nan, 'peakmaxvalue': np.nan,
                         'mlpeak': np.nan}
        else:
            bbpeak_naiveidx = np.nanargmax(peaks_props['peak_heights'])
            peakmax_props = {
                (k1 if k1.endswith('ips') else k1[:-1]): v1[bbpeak_naiveidx] 
                for k1, v1 in peaks_props.items()}
            idx_top = rut.find_nearest(profile_h, peakmax_props['right_ips'],
                                       mode='major')
            idx_bot = rut.find_nearest(profile_h, peakmax_props['left_ips'],
                                       mode='minor')
            peaks_idx = {
                'idxmax': peaks[bbpeak_naiveidx],
                # 'idxtop': peakmax_props['right_base'],
                # 'idxbot': peakmax_props['left_base'],
                'idxtop': idx_top,
                'idxbot': idx_bot,
                'peakmaxvalue': peakmax_props['peak_height'],
                'peakmaxits': peakmax_props['prominence'],
                'mlpeak': profile_h[peaks[bbpeak_naiveidx]],
                # 'mltop': profile_h[peakmax_props['right_base']],
                # 'mlbtm': profile_h[peakmax_props['left_base']],
                'mltop': profile_h[idx_top],
                'mlbtm': profile_h[idx_bot]
                }
            peaks_idx['mlthk'] = peaks_idx['mltop'] - peaks_idx['mlbtm']

        return peaks_idx

    def ml_detection(self, pol_profs, min_h=0., max_h=5., zhnorm_min=5.,
                     zhnorm_max=60., rhvnorm_min=0.85, rhvnorm_max=1.,
                     upplim_thr=0.75, param_k=0.05, param_w=0.75, comb_id=None,
                     phidp_peak='left', gradv_peak='left', plot_method=False):
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

        if self.profs_type.lower() == 'vps':
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
        elif self.profs_type.lower() == 'qvps':
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
        elif self.profs_type.lower() == 'rd-qvps':
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
        profrhv_norm = rut.normalisenanvalues(
            profrhv, rhvnorm_min, rhvnorm_max)
        # Combine ZH and rhoHV (norm) to create a new profile
        profcombzh_rhv = profzh_norm[min_hidx:max_hidx]*(
            1-profrhv_norm[min_hidx:max_hidx])
        # Detect peaks within the new profile
        pkscombzh_rhv = MeltingLayer.findpeaksboundaries(
            profcombzh_rhv,
            pol_profs.georef['profiles_height [km]'][min_hidx:max_hidx],
            param_k=param_k)
        # If no peaks were found, the profile is classified as No ML signatures
        if (all(value is np.nan for value in pkscombzh_rhv.values())
           or pkscombzh_rhv['peakmaxvalue'] < param_k):
            mlyr = np.nan
            mlrand = np.nan
            combin = np.nan
            idxml_top_it1 = 0
            comb_mult = []
            comb_mult_w = []
            idxml_btm_it1 = 0
        else:
            peakcombzh_rhv = (pol_profs.georef['profiles_height [km]']
                              [min_hidx:max_hidx][pkscombzh_rhv['idxmax']])
            idxml_btm_it1 = rut.find_nearest(
                pol_profs.georef['profiles_height [km]'],
                peakcombzh_rhv-upplim_thr)
            idxml_top_it1 = rut.find_nearest(
                pol_profs.georef['profiles_height [km]'],
                peakcombzh_rhv+upplim_thr)
            if idxml_top_it1 > min_hidx:
                if self.profs_type.lower() == 'vps':
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
                        param_k=param_k)
                          for i in comb_mult_w]
                for i, j in enumerate(mlrand):
                    if mlrand[i]['peakmaxvalue'] <= param_k:
                        mlrand[i] = {k: np.nan for k in mlrand[i]}
                    if (mlrand[i]['mltop'] < min_h
                        or mlrand[i]['mltop'] > max_h
                        # or mlrand[i]['mltop'] <= 0
                        ):
                        mlrand[i] = {k: np.nan for k in mlrand[i]}
                    # if mlrand[i]['mlbtm'] <= 0:
                    #     mlrand[i]['mlbtm'] = 0
                mlrandf = [{'ml_top': n['mltop'],
                            'ml_bottom': n['mlbtm'],
                            'ml_thickness': n['mltop']-n['mlbtm'],
                            'ml_peakh': n['mlpeak'],
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
        if isinstance(mlrand, list) and mlrand and ~np.isnan(mlrand[7]['idxmax']):
            bb_intensity = profzh[idxml_btm_it1:idxml_top_it1][mlrand[7]['idxmax']]
            bb_peakh = pol_profs.georef['profiles_height [km]'][idxml_btm_it1:idxml_top_it1][mlrand[7]['idxmax']]
        else:
            bb_intensity = np.nan
            bb_peakh = np.nan

        if plot_method:
            rad_interactive.ml_detectionvis(
                pol_profs.georef['profiles_height [km]'], profzh_norm,
                profrhv_norm, profcombzh_rhv, pkscombzh_rhv, comb_mult,
                comb_mult_w, comb_idpy, mlrand, min_hidx, max_hidx,
                param_k, idxml_btm_it1, idxml_top_it1)
        if comb_idpy is None:
            self.ml_id = mlyr
        elif comb_idpy is not None and isinstance(mlyr, dict):
            self.ml_top = mlyr['ml_top']
            self.ml_bottom = mlyr['ml_bottom']
            self.ml_thickness = mlyr['ml_thickness']
            self.profpeakv = mlyr['ml_peakv']
            self.profpeakh = mlyr['ml_peakh']
            self.bbpeakvalue = bb_intensity
            self.bb_peakh = bb_peakh
        else:
            self.ml_top = np.nan
            self.ml_bottom = np.nan
            self.ml_thickness = np.nan
            self.profpeakv = np.nan
            self.profpeakh = np.nan
            self.bbpeakvalue = bb_intensity
            self.bb_peakh = bb_peakh
    
    def ml_ppidelimitation(self, rad_georef, rad_params, beam_cone='centre',
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
        beam_cone : {'centre', 'top', 'bottom', 'all'}, optional
            Beam‑height descriptor to use for ML gating.
            'centre' uses ``beam_height [km]`` (default),
            'top' uses ``beamtop_height [km]``,
            'bottom' uses ``beambottom_height [km]``,
            'all' returns results for all three.
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
        gbeam2use = (['beam_height [km]'] if beam_cone == 'centre' else 
                     ['beamtop_height [km]'] if beam_cone == 'top' else 
                     ['beambottom_height [km]'] if beam_cone == 'bottom' else 
                     ['beam_height [km]', 'beamtop_height [km]',
                      'beambottom_height [km]'])
        # =============================================================================
        mlylims = {}
        for gb2d in gbeam2use:
            if isinstance(ml_top, (int, float)):
                mlt_idx = [rut.find_nearest(nbh, ml_top)
                           for nbh in rad_georef[gb2d]]
            elif isinstance(ml_top, (np.ndarray, list, tuple)):
                mlt_idx = [rut.find_nearest(nbh, ml_top[cnt])
                           for cnt, nbh in
                           enumerate(rad_georef[gb2d])]
            if isinstance(ml_bottom, (int, float)):
                if np.isnan(ml_bottom):
                    ml_bottom = ml_top - ml_thickness
                mlb_idx = [rut.find_nearest(nbh, ml_bottom)
                           for nbh in rad_georef[gb2d]]
            elif isinstance(ml_bottom, (np.ndarray, list, tuple)):
                ml_bottom = [mlt - ml_thickness[c1]
                             if np.isnan(ml_bottom[c1]) else ml_bottom[c1]
                             for c1, mlt in enumerate(ml_top)]
                mlb_idx = [rut.find_nearest(nbh, ml_bottom[cnt])
                           for cnt, nbh in
                           enumerate(rad_georef[gb2d])]
            ashape = np.zeros((rad_params['nrays'], rad_params['ngates']))
            for cnt, azi in enumerate(ashape):
                azi[:mlb_idx[cnt]] = self.regionID['rain']
                azi[mlt_idx[cnt]:] = self.regionID['solid_pcp']
            ashape[ashape == 0] = self.regionID['mlyr']
            if gb2d == 'beam_height [km]':
                mlylims['pcp_region [HC]'] = ashape
            elif gb2d == 'beamtop_height [km]':
                mlylims['pcp_region_tb [HC]'] = ashape
            elif gb2d == 'beambottom_height [km]':
                mlylims['pcp_region_bb [HC]'] = ashape
        # =============================================================================
        # self.mlyr_limits = {'pcp_region [HC]': ashape}
        self.mlyr_limits = mlylims
        if plot_method:
            rad_display.plot_ppi(rad_georef, rad_params, self.mlyr_limits,
                                 cbticks=self.regionID,
                                 ucmap='tpylc_div_yw_gy_bu')
    
    # def ml_ppidelimitation(self, rad_georef, rad_params, beam_cone='centre',
    #                        classid=None, plot_method=False):
    #     """
    #     Create a PPI depicting the limits of the melting layer.

    #     Parameters
    #     ----------
    #     rad_georef : dict
    #         Georeferenced data containing descriptors of the azimuth, gates
    #         and beam height, amongst others.
    #     rad_params : dict
    #         Radar technical details.
    #     beam_cone : {'centre', 'top', 'bottom', 'all'}, optional
    #         Beam‑height descriptor to use for ML delimitation.
    #         'centre' uses ``beam_height [km]`` (default),
    #         'top' uses ``beamtop_height [km]``,
    #         'bottom' uses ``beambottom_height [km]``,
    #         'all' returns results for all three.
    #     classid : dict, optional
    #         Modifies the key/values of the melting layer delimitation
    #         (regionID). The default are the same as in regionID.
    #     plot_method : bool, optional
    #         Plot the results of the ML delimitation. The default is False.

    #     Attributes
    #     ----------
    #     regionID : dict
    #         Key/values of the rain limits:
    #             'rain' = 1

    #             'mlyr' = 2

    #             'solid_pcp' = 3
    #     """
    #     ml_top = self.ml_top
    #     ml_thickness = self.ml_thickness
    #     ml_bottom = self.ml_bottom
        
    #     self.regionID = {'rain': 1.,
    #                      'mlyr': 2.,
    #                      'solid_pcp': 3.}
    #     if classid is not None:
    #         self.regionID.update(classid)
        
    #     beam_map = {
    #         'centre':  ['beam_height [km]'],
    #         'top':     ['beamtop_height [km]'],
    #         'bottom':  ['beambottom_height [km]'],
    #         'all':     ['beam_height [km]', 'beamtop_height [km]',
    #                     'beambottom_height [km]']}
    #     gbeam2use = beam_map.get(beam_cone, beam_map['centre'])
        
    #     # =============================================================================
    #     nrays = rad_params['nrays']
    #     ngates = rad_params['ngates']
        
    #     # --- normalise ml_top ---
    #     if np.isscalar(ml_top):
    #         ml_top_arr = np.full(nrays, ml_top, dtype=float)
    #     else:
    #         ml_top_arr = np.asarray(ml_top, dtype=float)
    #         if ml_top_arr.shape[0] != nrays:
    #             raise ValueError("ml_top must have length nrays")
        
    #     # --- handle ml_bottom according to your rules ---
    #     if np.isscalar(ml_bottom):
    #         if np.isnan(ml_bottom):
    #             ml_bottom_arr = ml_top_arr - ml_thickness
    #         else:
    #             ml_bottom_arr = np.full(nrays, ml_bottom, dtype=float)
    #     else:
    #         ml_bottom_arr = np.asarray(ml_bottom, dtype=float)
    #         if ml_bottom_arr.shape[0] != nrays:
    #             raise ValueError("ml_bottom must have length nrays")
    #         if np.isnan(ml_bottom_arr).any():
    #             raise ValueError("ml_bottom array must not contain NaN")
        
    #     mlylims = {}
    #     rain_id  = self.regionID['rain']
    #     solid_id = self.regionID['solid_pcp']
    #     mlyr_id  = self.regionID['mlyr']
        
    #     key_map = {'beam_height [km]':      'pcp_region [HC]',
    #                'beamtop_height [km]':   'pcp_region_tb [HC]',
    #                'beambottom_height [km]':'pcp_region_bb [HC]'}
    #     gate_idx = np.arange(ngates)[None, :]
    #     for gb2d in gbeam2use:
    #         ashape = np.full((nrays, ngates), mlyr_id, dtype=float)
    #         rain_mask  = gate_idx <  ml_bottom_arr[:, None]
    #         solid_mask = gate_idx >= ml_top_arr[:, None]
    #         ashape[rain_mask]  = rain_id
    #         ashape[solid_mask] = solid_id
    #         # store result
    #         mlylims[key_map[gb2d]] = ashape
    #     # =============================================================================
    #     self.mlyr_limits = mlylims
    #     if plot_method:
    #         rad_display.plot_ppi(rad_georef, rad_params, self.mlyr_limits,
    #                              cbticks=self.regionID,
    #                              ucmap='tpylc_div_yw_gy_bu')


# =============================================================================
# %% xarray implementation
# =============================================================================


def _to_da(value, ds, name, units="km", azimuth_dim="azimuth"):
    """
    Convert scalar or array-like ML parameter into a DataArray.
    """
    # Determine dims
    if np.isscalar(value):
        da = xr.DataArray(float(value))
    else:
        value = np.asarray(value)
        if value.shape != (ds.sizes[azimuth_dim],):
            raise ValueError(
                f"{name} must be scalar or have shape (azimuth,), "
                f"got {value.shape}")
        # da = xr.DataArray(value, dims=(azimuth_dim,), dtype=float)
        da = xr.DataArray(value.astype(float), dims=(azimuth_dim,))
    # Metadata dictionary
    meta = {"units": units,
            "long_name": {
                "mlyr_top": "melting-layer top height",
                "mlyr_bottom": "melting-layer bottom height",
                "mlyr_thickness": "melting-layer thickness"}.get(name, name),
            "standard_name": {"mlyr_top": "mlyr_top",
                              "mlyr_bottom": "mlyr_bottom",
                              "mlyr_thickness": "mlyr_thickness"}.get(name, name),
            "short_name": {"mlyr_top": "MLYRTOP",
                           "mlyr_bottom": "MLYRBTM",
                           "mlyr_thickness": "MLYRTHK",}.get(name, name),
        "description": {
            "mlyr_top": "Height of the upper boundary of the melting layer",
            "mlyr_bottom": "Height of the lower boundary of the melting layer",
            "mlyr_thickness": "Vertical thickness of the melting layer"}.get(name, "")
        }
    da.attrs = meta
    return da


def _ensure_azimuth_da(val, ds, name, azimuth_dim="azimuth"):
    """
    Normalise melting-layer inputs into a 1D DataArray over azimuth.

    Accepts:
        - scalar
        - NumPy array
        - DataArray (scalar or 1D)
    Returns:
        xr.DataArray with dims ('azimuth',)
    """
    # Case 1 — already a DataArray
    if isinstance(val, xr.DataArray):
        if val.ndim == 0:
            # Broadcast scalar DA to azimuth
            out = xr.full_like(ds.azimuth, float(val), dtype=float).rename(name)
            out.attrs.update(val.attrs)
            return out
        if val.ndim == 1:
            # Must be azimuth-aligned
            if val.dims != (azimuth_dim,):
                raise ValueError(
                    f"{name} DataArray must have dims ('azimuth',), got {val.dims}"
                )
            return val.rename(name)

        raise ValueError(
            f"{name} DataArray must be scalar or 1D over azimuth, got dims {val.dims}"
        )

    # Case 2 — scalar
    if np.isscalar(val) or isinstance(val, (np.generic,)):
        return xr.DataArray(
            np.full(ds.sizes[azimuth_dim], float(val)),
            dims=(azimuth_dim,),
            name=name,
        )


    # Case 3 — NumPy array
    arr = np.asarray(val)
    if arr.ndim == 1 and arr.size == ds.sizes[azimuth_dim]:
        return xr.DataArray(arr, dims=(azimuth_dim,), name=name)

    raise ValueError(
        f"{name} must be scalar, a 1D array of length azimuth, "
        f"or a DataArray over azimuth"
    )


def _normalise_ml_input(x, ds, azimuth_dim="azimuth"):
    if x is None:
        return None

    arr = np.asarray(x)

    # Scalar or 0‑D array → convert to float
    if arr.shape == ():
        return float(arr)

    # 1‑D per‑azimuth array
    if arr.shape == (ds.sizes[azimuth_dim],):
        return arr.astype(float)

    raise ValueError(f"ML parameter must be scalar or shape (azimuth,),"
                     f" got shape {arr.shape}")


def attach_melting_layer(ds, mlyr_top=None, mlyr_bottom=None, mlyr_thickness=None,
                         units="km", source="user-defined", method=None,
                         overwrite=False, delimit_mlyrinppi=False, classid=None,
                         beam_cone="centre"):
    """
    Attach melting-layer metadata to a radar sweep dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Input radar sweep dataset.
    mlyr_top, mlyr_bottom, mlyr_thickness : float, array-like, or DataArray
        Melting-layer top height, bottom height, and thickness. Scalars,
        NumPy arrays, and 1D DataArrays over ``azimuth`` are accepted. Any two
        must be provided; the third is derived.
    units : str, optional
        Units assigned to the melting-layer metadata variables (default: "km").
    source : str, optional
        Provenance tag describing the origin of the melting-layer estimate.
    method : str, optional
        Additional provenance tag describing the estimation method.
    overwrite : bool, optional
        If False (default), an error is raised when melting-layer metadata
        already exist in the dataset.
    delimit_mlyrinppi : bool, optional
        If True, compute melting-layer precipitation-region classification
        using the attached metadata and add the resulting fields to the dataset.
    beam_cone : {'centre', 'top', 'bottom', 'all'}, optional
        Beam-height descriptor(s) to use when performing classification.
        Passed directly to `mlyr_ppidelimitation`.
    classid : dict, optional
        Optional mapping overriding the default region identifiers used in
        classification.
    
    Notes
    -----
    1. Any two of (mlyr_top, mlyr_bottom, mlyr_thickness) must be provided. The third
    is computed automatically. All three are validated for consistency.
    """
    # Avoid overwriting
    existing = {"mlyr_top", "mlyr_bottom", "mlyr_thickness"} & set(ds.data_vars)
    if existing and not overwrite:
        raise ValueError(f"Dataset already contains ML metadata: {existing}. "
                         "Use overwrite=True to replace.")
    # Count provided parameters 
    provided = {"mlyr_top": mlyr_top is not None,
                "mlyr_bottom": mlyr_bottom is not None,
                "mlyr_thickness": mlyr_thickness is not None}
    n_provided = sum(provided.values())
    if n_provided < 2:
        raise ValueError(
            "Provide at least two of: mlyr_top, mlyr_bottom, mlyr_thickness.")
    # Convert to arrays
    mlyr_top_arr = _normalise_ml_input(mlyr_top, ds)
    mlyr_bottom_arr = _normalise_ml_input(mlyr_bottom, ds)
    mlyr_th_arr = _normalise_ml_input(mlyr_thickness, ds)
    # Compute missing parameter
    if mlyr_top_arr is None:
        mlyr_top_arr = mlyr_bottom_arr + mlyr_th_arr
    if mlyr_bottom_arr is None:
        mlyr_bottom_arr = mlyr_top_arr - mlyr_th_arr
    if mlyr_th_arr is None:
        mlyr_th_arr = mlyr_top_arr - mlyr_bottom_arr
    # Validate consistency
    if not np.allclose(mlyr_bottom_arr, mlyr_top_arr - mlyr_th_arr, equal_nan=True):
        raise ValueError("Inconsistent ML: mlyr_bottom != mlyr_top - mlyr_thickness")
    if not np.allclose(mlyr_th_arr, mlyr_top_arr - mlyr_bottom_arr, equal_nan=True):
        raise ValueError("Inconsistent ML: mlyr_thickness != mlyr_top - mlyr_bottom")
    # Convert to DataArrays
    mlyr_top_da = _to_da(mlyr_top_arr, ds, "mlyr_top", units=units)
    mlyr_bottom_da = _to_da(mlyr_bottom_arr, ds, "mlyr_bottom", units=units)
    mlyr_thickness_da = _to_da(mlyr_th_arr, ds, "mlyr_thickness", units=units)
    # Attach variables
    ds2 = ds.copy()
    ds2 = safe_assign_variable(ds2, "MLYRTOP", mlyr_top_da)
    ds2 = safe_assign_variable(ds2, "MLYRBTM", mlyr_bottom_da)
    ds2 = safe_assign_variable(ds2, "MLYRTHK", mlyr_thickness_da)
    # Optional: melting-layer delimitation in PPI
    if delimit_mlyrinppi:
        classif = mlyr_ppidelimitation(ds2, mlyr_top=mlyr_top_da,
                                       mlyr_bottom=mlyr_bottom_da,
                                       mlyr_thickness=mlyr_thickness_da,
                                       beam_cone=beam_cone,
                                       classid=classid)
        # Attach each classification field
        ds2 = ds2.assign(classif)
    # Dataset-level provenance
    outputs = ["MLYRTOP", "MLYRBTM", "MLYRTHK"]
    # Figure out which ML inputs were explicitly provided
    ml_inputs = []
    if mlyr_top is not None:
        ml_inputs.append("mlyr_top")
    if mlyr_bottom is not None:
        ml_inputs.append("mlyr_bottom")
    if mlyr_thickness is not None:
        ml_inputs.append("mlyr_thickness")
    # Geometry inputs (coords actually used)
    geom_inputs = []
    for name in ("azimuth", "range", "elevation"):
        if name in ds.coords:
            geom_inputs.append(name)
    inputs = geom_inputs + ml_inputs
    # if classif is not None:
    #     classid = list(classif.flags())
    if method is None:
        method = 'towerpy.ml.mlyr.detect_mlyr_from_profiles'
    params = {"units": units, "source": source, "method": method,
              "beam_cone": beam_cone, "classid": classid,
              "overwrite": bool(overwrite),
              "delimit_mlyrinppi": bool(delimit_mlyrinppi)}
    if delimit_mlyrinppi:
        extra = {"step_description":
                 ("Attach melting-layer metadata (top, bottom, thickness) and "
                  "classify precipitation regions in PPI.")}
    else:
        extra = {"step_description":
                 ("Attach melting-layer metadata (top, bottom, thickness.")}
    ds2 = record_provenance(
        ds2, step="attach_melting_layer",
        inputs=inputs,
        outputs=outputs, parameters=params, extra_attrs=extra,
        module_provenance="towerpy.ml.mlyr.attach_melting_layer")
    return ds2


def mlyr_ppidelimitation(ds, mlyr_top, mlyr_bottom, mlyr_thickness,
                         beam_cone="centre", classid=None,
                         azimuth_dim="azimuth", range_dim="range"):
    """
    Classify radar bins into rain, mixed-phase, and solid-precipitation
    regions using melting-layer top and bottom heights.

    Parameters
    ----------
    ds : xr.Dataset
        Radar dataset containing beam‑height coordinates:
        ``beamc_height``, ``beamb_height``, ``beamt_height`` with dimensions
        ``(azimuth, range)``.
    mlyr_top, mlyr_bottom, mlyr_thickness : float, array-like, or DataArray
        Melting-layer top height, bottom height, and thickness. Scalars,
        NumPy arrays, and 1D DataArrays over ``azimuth`` are accepted. NaN
        values in ``mlyr_bottom`` are replaced by ``mlyr_top - mlyr_thickness``.
    beam_cone : {'centre', 'top', 'bottom', 'all'}, optional
        Selects which beam‑height descriptor(s) to use:
            - 'centre' → ``beamc_height`` (default)
            - 'top'    → ``beamt_height``
            - 'bottom' → ``beamb_height``
            - 'all'    → all three
    classid : dict, optional
        Optional mapping overriding the default region identifiers:
            {'rain': 1.0, 'mlyr': 2.0, 'solid_pcp': 3.0}

    Returns
    -------
    dict[str, xr.DataArray]
       A dictionary mapping each selected beam-height field to a classification
       array with dimensions ``(azimuth, range)``. Values correspond to the
       region identifiers.
    """
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # Region identifiers
    regionID = {"rain": 1.0, "mlyr": 2.0, "solid_pcp": 3.0}
    if classid is not None:
        regionID.update(classid)
    # Beam-height selector
    cone_map = {"centre": ["beamc_height"],
                "top": ["beamt_height"],
                "bottom": ["beamb_height"],
                "all": ["beamc_height", "beamt_height", "beamb_height"]}
    beams_to_use = cone_map[beam_cone]
    # Normalise melting-layer inputs
    mlyr_top_da = _ensure_azimuth_da(mlyr_top, ds, "mlyr_top")
    mlyr_bottom_da = _ensure_azimuth_da(mlyr_bottom, ds, "mlyr_bottom")
    mlyr_thickness_da = _ensure_azimuth_da(mlyr_thickness, ds, "mlyr_thickness")
    # Replace NaN bottom heights with top - thickness
    mlyr_bottom_da = xr.where(xr.ufuncs.isnan(mlyr_bottom_da),
                              mlyr_top_da - mlyr_thickness_da,
                              mlyr_bottom_da)
    # Precompute range indices for broadcasting
    range_idx = xr.DataArray(np.arange(ds.sizes[range_dim]), dims=(range_dim,))
    out = {}
    # Vectorised classification for each beam-height field
    for beam_name in beams_to_use:
        bh = ds[beam_name]  # (azimuth, range)
        # Nearest-range indices for ML top and bottom
        mlyr_top_idx = abs(bh - mlyr_top_da).argmin(dim=range_dim)
        mlyr_bottom_idx = abs(bh - mlyr_bottom_da).argmin(dim=range_dim)
        # Region masks (broadcasted)
        rain_mask = range_idx < mlyr_bottom_idx
        solid_mask = range_idx >= mlyr_top_idx
        mlyr_mask = ~(rain_mask | solid_mask)
        da = (rain_mask * regionID["rain"]
              + solid_mask * regionID["solid_pcp"]
              + mlyr_mask * regionID["mlyr"])
        da = da.transpose(azimuth_dim, range_dim)
        da = da.assign_coords(azimuth=ds.azimuth, range=ds.range).rename(
            "ML_PCP_CLASS")
        # Attach attributes
        da.attrs = sweep_vars_attrs_f.get("ML_PCP_CLASS", {})
        da.attrs.update({"units": f"flags [{len(regionID)}]",
                         "flags": regionID, "cone_map": beam_cone})
        if beam_cone == "all":
            varname = f"ML_PCP_CLASS_{beam_name}"
        else:
            varname = "ML_PCP_CLASS"
        # Variable-level provenance
        da.attrs = add_correction_step(
            parent_attrs=da.attrs, step="mlyr_ppi",
            parent=beam_name,
            params={"beam_cone": beam_cone, "classid": classid},
            outputs=[varname], mode="preserve",
            module_provenance="towerpy.ml.mlyr.attach_melting_layer")
        out[varname] = da
    return out


def compute_profile_peaks_props(profile_v, profile_h, param_k):
    """
    Detect peaks in a 1D radar profile and derive height‑based peak properties.

    Parameters
    ----------
    profile_v : array_like of float
        1D radar profile values.
    profile_h : array_like of float
        Heights corresponding to ``profile_v`` (km). Must have the same
        length as ``profile_v``.
    param_k : float
        Scaling factor used to set the minimum peak prominence.

    Returns
    -------
    props : dict
        Dictionary containing scalar peak properties. All fields are
        returned as ``numpy.nan`` if no peak is detected.

    Notes
    -----
    * Peak detection is performed using :func:`scipy.signal.find_peaks`.
    """
    profile_v = np.asarray(profile_v, dtype=float)
    profile_h = np.asarray(profile_h, dtype=float)
    peaks, props = scs.find_peaks(profile_v, height=(None, None),
                                  width=(None, None), threshold=(None, None),
                                  rel_height=0.99, plateau_size=(None, None),
                                  prominence=(param_k * 0.7, 2))
    if peaks.size == 0 or np.all(np.isnan(peaks)):
        return {"idxmax": np.nan, "idxtop": np.nan, "idxbot": np.nan,
                "mltop": np.nan, "mlbtm": np.nan, "mlthk": np.nan,
                "peakmaxvalue": np.nan, "peakmaxits": np.nan, "mlpeak": np.nan}
    # interpolate left/right ips to height
    x = np.arange(len(profile_v))
    left_h  = np.interp(props["left_ips"],  x, profile_h)
    right_h = np.interp(props["right_ips"], x, profile_h)
    bb_idx = np.nanargmax(props["peak_heights"])
    peak_height = props["peak_heights"][bb_idx]
    prom = props["prominences"][bb_idx]
    peak_idx = peaks[bb_idx]
    idx_top = find_nearest_index(profile_h, right_h[bb_idx], mode="major")
    idx_bot = find_nearest_index(profile_h, left_h[bb_idx],  mode="minor")
    mltop = profile_h[idx_top]
    mlbtm = profile_h[idx_bot]
    mlpeak = profile_h[peak_idx]
    return {"idxmax": peak_idx, "idxtop": idx_top, "idxbot": idx_bot,
            "mltop": mltop, "mlbtm": mlbtm, "mlthk": mltop - mlbtm,
            "peakmaxvalue": peak_height, "peakmaxits": prom, "mlpeak": mlpeak}


def _mlyr_detection_engine(height, dbz, rhohv, zdr, phidp, gradv, profile_type,
                           min_h, max_h, dbznorm_min, dbznorm_max,
                           rhvnorm_min, rhvnorm_max, upplim_thr, param_k,
                           param_w, comb_id, phidp_peak, gradv_peak,
                           return_diagnostics=False, return_internal=False):
    """
    Core melting-layer detection engine on 1D NumPy profiles.
    """
    # 1. ensure NumPy arrays
    height = np.asarray(height, dtype=float)
    dbz    = np.asarray(dbz,    dtype=float)
    rhohv  = np.asarray(rhohv,  dtype=float)
    zdr    = None if zdr   is None else np.asarray(zdr,   dtype=float)
    phidp  = None if phidp is None else np.asarray(phidp, dtype=float)
    gradv  = None if gradv is None else np.asarray(gradv, dtype=float)
    # 2. indices for global min/max height search
    min_hidx = find_nearest_index(height, min_h, mode="any")
    max_hidx = find_nearest_index(height, max_h, mode="any")
    # 3. normalise ZH and RHOHV (clip + fixed-range normalisation)
    dbz_clipped = np.clip(dbz, dbznorm_min, dbznorm_max)
    dbz_norm = (dbz_clipped - dbznorm_min) / (dbznorm_max - dbznorm_min)
    rho_clipped = np.clip(rhohv, rhvnorm_min, rhvnorm_max)
    rho_norm = (rho_clipped - rhvnorm_min) / (rhvnorm_max - rhvnorm_min)
    # 4. pre-detection profile
    profcombdbz_rhv = dbz_norm[min_hidx:max_hidx] * (
        1.0 - rho_norm[min_hidx:max_hidx])
    pks_pre = compute_profile_peaks_props(
        profcombdbz_rhv, height[min_hidx:max_hidx], param_k=param_k)
    if (np.isnan(pks_pre["peakmaxvalue"]) or pks_pre["peakmaxvalue"] < param_k):
        # No ML detected
        mlyr = np.nan
        mlrand = np.nan
        combin = np.nan
        comb_mult = []
        comb_mult_w = []
        idxml_btm_it1 = 0
        idxml_top_it1 = 0
        bb_intensity = np.nan
        bb_peakh = np.nan
        diagnostics = {"bb_intensity": np.nan, "bb_peakh": np.nan}
        intermediate = {"ml_peakh_all": [], "ml_peakv_all": [],
                        "dbz_norm": dbz_norm, "rho_norm": rho_norm,
                        "profcombdbz_rhv": profcombdbz_rhv,
                        "pks_pre": pks_pre, "comb_mult": [], "comb_mult_w": [],
                        "mlrand": np.nan, "combin": np.nan,
                        "min_hidx": min_hidx, "max_hidx": max_hidx,
                        "idxml_btm_it1": 0, "idxml_top_it1": 0}
        if return_diagnostics and return_internal:
            return mlyr, diagnostics, intermediate
        elif return_diagnostics:
            return mlyr, diagnostics
        elif return_internal:
            return mlyr, intermediate
        else:
            return mlyr
    # 5. ML window around the pre-detection peak
    peakcomb_h = pks_pre["mlpeak"]
    idxml_btm_it1 = find_nearest_index(height, peakcomb_h - upplim_thr,
                                       mode="any")
    idxml_top_it1 = find_nearest_index(height, peakcomb_h + upplim_thr,
                                       mode="any")
    if idxml_top_it1 <= min_hidx:
        # No ML window
        mlyr = np.nan
        mlrand = np.nan
        combin = np.nan
        comb_mult = []
        comb_mult_w = []
        bb_intensity = np.nan
        bb_peakh = np.nan
        diagnostics = {"bb_intensity": np.nan, "bb_peakh": np.nan}
        intermediate = {"ml_peakh_all": [], "ml_peakv_all": [],
                        "dbz_norm": dbz_norm, "rho_norm": rho_norm,
                        "profcombdbz_rhv": profcombdbz_rhv, "pks_pre": pks_pre,
                        "comb_mult": [], "comb_mult_w": [], "mlrand": np.nan,
                        "combin": np.nan, "min_hidx": min_hidx,
                        "max_hidx": max_hidx, "idxml_btm_it1": idxml_btm_it1,
                        "idxml_top_it1": idxml_top_it1}
        if return_diagnostics and return_internal:
            return mlyr, diagnostics, intermediate
        elif return_diagnostics:
            return mlyr, diagnostics
        elif return_internal:
            return mlyr, intermediate
        else:
            return mlyr
    # 6. Build component profiles in window
    sl = slice(idxml_btm_it1, idxml_top_it1)
    h_win = height[sl]
    comps = []
    # gradV only for VP
    if profile_type.lower() == "vp":
        if gradv is not None:
            gv = gradv[sl].astype(float)
            gv_min = np.nanmin(gv)
            gv_max = np.nanmax(gv)
            if np.isfinite(gv_min) and np.isfinite(gv_max) and gv_max > gv_min:
                gv_norm = (gv - gv_min) / (gv_max - gv_min)
            else:
                gv_norm = np.zeros_like(gv)
            if gradv_peak == "right":
                gv_norm = -gv_norm
            comps.append(1.0 - gv_norm)
        else:
            comps.append(np.ones_like(dbz_norm[sl]))
    # ZH_norm
    comps.append(dbz_norm[sl])
    # ZDR_norm
    if zdr is not None:
        z = zdr[sl].astype(float)
        z_min = np.nanmin(z)
        z_max = np.nanmax(z)
        if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
            z_norm = (z - z_min) / (z_max - z_min)
        else:
            z_norm = np.zeros_like(z)
        comps.append(z_norm)
    else:
        comps.append(np.ones_like(dbz_norm[sl]))
    # 1 - RHOHV_norm
    comps.append(1.0 - rho_norm[sl])
    # PHIDP_norm
    if phidp is not None:
        ph = phidp[sl].astype(float)
        ph_min = np.nanmin(ph)
        ph_max = np.nanmax(ph)
        if np.isfinite(ph_min) and np.isfinite(ph_max) and ph_max > ph_min:
            ph_norm = (ph - ph_min) / (ph_max - ph_min)
        else:
            ph_norm = np.zeros_like(ph)
        if phidp_peak == "right":
            ph_norm = -ph_norm
        comps.append(ph_norm)
    else:
        comps.append(np.ones_like(dbz_norm[sl]))
    
    # 7. All non-empty combinations
    
    n = len(comps)
    combin = np.array(list(map(list, product([0, 1], repeat=n)))[1:])
    comps_arr = np.vstack(comps)
    mask = combin[:, :, None] == 1
    selected = np.where(mask, comps_arr, 1)
    comb_mult = np.prod(selected, axis=1)
    # second derivative sharpening
    comb_mult_w = [c - param_w * np.gradient(np.gradient(c))
                   for c in comb_mult]
    # 8. Peak detection per combination
    mlrand = []
    for prof in comb_mult_w:
        res_pk = compute_profile_peaks_props(prof, h_win, param_k=param_k)
        if (np.isnan(res_pk["peakmaxvalue"])
            or res_pk["peakmaxvalue"] <= param_k
            or res_pk["mltop"] < min_h
            or res_pk["mltop"] > max_h):
            mlrand.append({k: np.nan for k in res_pk})
        else:
            mlrand.append(res_pk)
    mlrandf = [{"MLYRTOP": r["mltop"], "MLYRBTM": r["mlbtm"],
                "MLYRTHK": r["mlthk"], "mlyr_peakh": r["mlpeak"],
                "mlyr_peakv": r["peakmaxvalue"]} for r in mlrand]
    if comb_id is None:
        mlyr = mlrandf
    else:
        comb_idpy = comb_id - 1
        mlyr = mlrandf[comb_idpy] if 0 <= comb_idpy < len(mlrandf) else np.nan
    # 9. Bright band intensity for combination 7
    if (isinstance(mlrand, list) and len(mlrand) > 7
        and not np.isnan(mlrand[7].get("idxmax", np.nan))):
        idx7 = mlrand[7]["idxmax"]
        bb_intensity = dbz[sl][idx7]
        bb_peakh = h_win[idx7]
    else:
        bb_intensity = np.nan
        bb_peakh = np.nan
    # 10. Build diagnostics
    diagnostics = {"bb_intensity": bb_intensity, "bb_peakh": bb_peakh}
    intermediate = {"ml_peakh_all": [r.get("mlpeak", np.nan) for r in mlrand],
                    "ml_peakv_all": [r.get("peakmaxvalue", np.nan)
                                     for r in mlrand],
                    "dbz_norm": dbz_norm, "rho_norm": rho_norm,
                    "profcombdbz_rhv": profcombdbz_rhv, "pks_pre": pks_pre,
                    "comb_mult": comb_mult, "comb_mult_w": comb_mult_w,
                    "mlrand": mlrand, "combin": combin, "min_hidx": min_hidx,
                    "max_hidx": max_hidx, "idxml_btm_it1": idxml_btm_it1,
                    "idxml_top_it1": idxml_top_it1}
    # 11. Return according to flags
    if return_diagnostics and return_internal:
        return mlyr, diagnostics, intermediate
    elif return_diagnostics:
        return mlyr, diagnostics
    elif return_internal:
        return mlyr, intermediate
    else:
        return mlyr


def detect_mlyr_from_profiles(ds, *, inp_names=None, profile_type='qvp',
                              min_h=0.0, max_h=5.0, dbznorm_min=5.0,
                              dbznorm_max=60.0, rhvnorm_min=0.85,
                              rhvnorm_max=1.0, param_k=0.08, param_w=3/4,
                              comb_id=None, upplim_thr=0.75, phidp_peak="left",
                              gradv_peak="left", return_diagnostics=True,
                              return_internal=False):
    """	
    Detect melting layer signatures from polarimetric radar profiles appliyng
    the MLyr detection methodology of Sanchez-Rivas and Rico-Ramirez (2021).

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing height and polarimetric profile variables.
    inp_names : dict or None, default None
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{'height': 'height', "DBZ": "DBZH", "ZDR": "ZDR", "RHOHV": "RHOHV",
        "PHIDP": "PHIDP", 'GRADV': "GRAD_VRADV"}``
    profile_type : {'vp', 'qvp', 'rdqvp'}, default 'qvp'
        Type of input profile used to configure the detection engine.
    min_h : float, default 0.0
        Minimum height (km) considered in the detection.
    max_h : float, default 5.0
        Maximum height (km) considered in the detection.
    dbznorm_min : float, default 5.0
        Min value of :math:`Z_{H}` to use for the min-max normalisation.
    dbznorm_max : float, default 60.0
        Max value of :math:`Z_{H}` to use for the min-max normalisation.
    rhvnorm_min : float, default 0.85
        Min value of :math:`\rho_{HV}` to use for the min-max normalisation.
    rhvnorm_max : float, default 1.0
        Max value of :math:`\rho_{HV}` to use for the min-max normalisation.
    param_k : float, default 0.08
        Threshold related to the magnitude of the peak used to detect the ML.
    param_w : float, default 3/4
        Weighting factor used to sharpen the peak within the profile.
    comb_id : int or None, default None
        Index identifier of the combination selected for the MLyr detection.
        If None, the method provides all the possible combinations of
        available profiles.
    upplim_thr : float, default 0.75
        Maximum upward search distance (km) from the detected peak, according
        to [1]_.
    phidp_peak : {'left', 'right'}, default 'left'
        Side of the PHIDP peak used for boundary selection.
    gradv_peak : {'left', 'right'}, default 'left'
        Side of the GRADV peak used for boundary selection.
    return_diagnostics : bool, default True
        If ``True``, include bright‑band diagnostics in the output.
    return_internal : bool, default False
        If ``True``, also return intermediate fields from the detection engine.

    Returns
    -------
    out : xarray.Dataset
        Dataset containing melting‑layer boundaries:
            - ``MLYRTOP`` : float  
                Melting‑layer top height (km).
            - ``MLYRBTM`` : float  
                Melting‑layer bottom height (km).
            - ``MLYRTHK`` : float  
                Melting‑layer thickness (km).
        If ``return_diagnostics=True``:
            - ``BB_INTENSITY`` : float
                Bright‑band peak reflectivity (dBZ).
            - ``BB_PEAKH`` : float  
                Bright‑band peak height (km).

    Notes
    -----
    * The detection engine evaluates combinations of normalised DBZ, RHOHV,
      ZDR, PHIDP and GRADV profiles depending on availability and
      configuration.
    * Internal peak detection uses :func:`scipy.signal.find_peaks`.
    * ``return_internal`` produces a dictionary exposing internal normalised
      profiles, peak‑finding results, and combination scores used during the
      detection process. They are intended for debugging and algorithm
      development.

    References
    ----------
    .. [1] Sanchez-Rivas, D., & Rico-Ramirez, M. A. (2021). Detection of the
        melting level with polarimetric weather radar. Atmospheric Measurement
        Techniques, 14(4), 2873–2890. https://doi.org/10.5194/amt-14-2873-2021

    """
    defaults = {'height': 'height', "DBZ": "DBZH", "ZDR": "ZDR",
                "RHOHV": "RHOHV", "PHIDP": "PHIDP", 'GRADV': "GRAD_VRADV"}
    names = {**defaults, **(inp_names or {})}
    # Extract 1D NumPy arrays
    height = np.asarray(ds[names["height"]].values)
    dbz_arr = np.asarray(ds[names["DBZ"]].values)
    rho_arr = np.asarray(ds[names["RHOHV"]].values)
    zdr_arr = np.asarray(ds[names["ZDR"]].values) if names["ZDR"] in ds else None
    ph_arr  = np.asarray(ds[names["PHIDP"]].values) if names["PHIDP"] in ds else None
    gv_arr  = np.asarray(ds[names["GRADV"]].values) if names["GRADV"] in ds else None
    # Apply MLyr detection engine
    result = _mlyr_detection_engine(
        height, dbz_arr, rho_arr, zdr_arr, ph_arr, gv_arr,
        profile_type=profile_type,
        min_h=min_h, max_h=max_h,
        dbznorm_min=dbznorm_min, dbznorm_max=dbznorm_max,
        rhvnorm_min=rhvnorm_min, rhvnorm_max=rhvnorm_max,
        upplim_thr=upplim_thr, param_k=param_k, param_w=param_w,
        comb_id=comb_id, phidp_peak=phidp_peak, gradv_peak=gradv_peak,
        return_diagnostics=return_diagnostics,
        return_internal=return_internal)
    # unpack
    if return_diagnostics and return_internal:
        mlyr, diagnostics, intermediate = result
    elif return_diagnostics:
        mlyr, diagnostics = result
    elif return_internal:
        mlyr, intermediate = result
    else:
        mlyr = result
    # Construct xarray output
    if comb_id is None:
        # all combinations
        if not isinstance(mlyr, list):
            out = xr.Dataset(coords={"combination": []})
        else:
            keys = list(mlyr[0].keys())
            arr = {k: np.empty(len(mlyr), dtype=float) for k in keys}
            
            for i, d in enumerate(mlyr):
                for k in keys:
                    arr[k][i] = d[k]
            out = xr.Dataset({k: ("combination", arr[k]) for k in keys},
                             coords={"combination": np.arange(len(mlyr))})
    else:
        # single combination
        if isinstance(mlyr, dict):
            geometry_keys = ["MLYRTOP", "MLYRBTM", "MLYRTHK"]
            out = xr.Dataset({k: ((), v) for k, v in mlyr.items()
                              if k in geometry_keys})
        else:
            out = xr.Dataset({"MLYRTOP": ((), np.nan),
                              "MLYRBTM": ((), np.nan),
                              "MLYRTHK": ((), np.nan)})
    # Attach BB diagnostics only if requested
    if return_diagnostics:
        out["BB_INTENSITY"] = ((), diagnostics["bb_intensity"])
        out["BB_PEAKH"] = ((), diagnostics["bb_peakh"])
        out["BB_INTENSITY"].attrs.update({
            "long_name": "bright band peak reflectivity",
            "units": ds[names["DBZ"]].attrs.get("units", "dBZ")})
        out["BB_PEAKH"].attrs.update({"long_name": "bright band peak height",
                                      "units": "km"})
    # add scalar / non-height coords from the input dataset
    scalar_coords = {name: coord for name, coord in ds.coords.items()
                     if coord.dims == ()
                     and name not in ("height", "range", "azimuth")}
    out = out.assign_coords(scalar_coords)
    # Add metadata
    if "MLYRTOP" in out:
        hvar = ds[names["height"]]
        units = hvar.attrs.get("units", "km")
        out["MLYRTOP"].attrs.update({
            "units": units,
            "long_name": "melting-layer top height",
            "standard_name": "melting_layer_top_height",
            "short_name": "MLYRTOP",
            "description": "Height of the upper boundary of the melting layer"})
        out["MLYRBTM"].attrs.update({
            "units": units,
            "long_name": "melting-layer bottom height",
            "standard_name": "melting_layer_bottom_height",
            "short_name": "MLYRBTM",
            "description": "Height of the lower boundary of the melting layer"})
        out["MLYRTHK"].attrs.update({
            "units": units,
            "long_name": "melting-layer thickness",
            "standard_name": "melting_layer_thickness",
            "short_name": "MLYRTHK",
            "description": "Vertical thickness of the melting layer"})
    # Provenance
    vars_used = {"DBZ": names["DBZ"], "RHOHV": names["RHOHV"]}
    if names["ZDR"] in ds:
        vars_used["ZDR"] = names["ZDR"]
    if names["PHIDP"] in ds:
        vars_used["PHIDP"] = names["PHIDP"]
    if names["GRADV"] in ds:
        vars_used["GRADV"] = names["GRADV"]
    #TODO: add step_description
    extra = {'step_description': ('')}
    profs_type = ds.attrs.get("profs_type", profile_type)
    out = record_provenance(
        out, step=f"mlyr_detection_from_{profs_type}",
        inputs=list(vars_used.values()), outputs=list(out.data_vars),
        parameters={"source_profile_type": profile_type,
                    "min_h": min_h, "max_h": max_h,
                    "dbznorm_min": dbznorm_min, "dbznorm_max": dbznorm_max,
                    "rhvnorm_min": rhvnorm_min, "rhvnorm_max": rhvnorm_max,
                    "param_k": param_k, "param_w": param_w, "comb_id": comb_id,
                    "upplim_thr": upplim_thr,  "phidp_peak": phidp_peak,
                    "gradv_peak": gradv_peak, "inp_names": names},
        extra_attrs=extra,
        module_provenance="towerpy.ml.mlyr.detect_mlyr_from_profiles")
    out.attrs["profs_type"] = profs_type
    if "processing_chain" in ds.attrs:
        out.attrs["input_processing_chain"] = copy.deepcopy(
            ds.attrs["processing_chain"])
    if return_internal:
        return out, intermediate
    else:
        return out
