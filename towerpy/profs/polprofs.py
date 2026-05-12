"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import copy
from datetime import datetime
import warnings
import numpy as np
import xarray as xr
from ..io import modeltp as mdtp
from ..utils.radutilities import record_provenance, scan_midtime, get_attrval
from ..utils.unit_conversion import convert

from ..base import TowerpyError
from ..datavis import rad_display
from ..utils.radutilities import find_nearest

# TODO: Add KDP to the building process.
# TODO: Add printing message warning of KDP and Vel? instead of raising error.
class PolarimetricProfiles:
    """
    A class to generate profiles of polarimetric variables.

    Attributes
    ----------
        elev_angle : float or list
            Elevation angle at which the scan was taken, in deg.
        file_name : str or list
            Name of the file containing radar data.
        scandatetime : datetime or list
            Date and time of scan.
        site_name : str
            Name of the radar site.
        georef : dict, optional
            Descriptor of the computed profiles height.
        vps : dict, optional
            Profiles generated from a birdbath scan.
        vps_stats : dict, optional
            Statistics of the VPs generation.
        qvps : dict, optional
            Quasi-Vertical Profiles generated from the PPI scan.
        qvps_stats : dict, optional
            Statistics of the QVPs generation.
        rd_qvps : dict, optional
            Range-defined Quasi-Vertical Profiles generated from PPI scans
            taken at different elevation angles.
        qvps_itp : dict, optional
            QVPs generated from each elevation angle.
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
        self.profs_type = getattr(radobj, 'profs_type',
                                  None) if radobj else None

    def pol_vps(self, rad_georef, rad_params, rad_vars, thlds=None,
                valid_gates=0, stats=False):
        """
        Generate profiles of polarimetric variables from a birdbath scan.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used to generate the VPs.
        thlds : dict containing key and 2-element tuple or list, optional
            Thresholds [min, max]  of radar variables used to discard gates
            in the azimuthal averaging. The default is None.
        valid_gates : int, optional
            Number of valid gates (or azimuths) along the radial.
            The default is 0.
        stats : Bool, optional
            Statistics of the VPs generation:
                'std_dev': Standard Deviation

                'min': Min values

                'max': Max values

                'sem': Standard Error of the Mean
        """
        ri, rf = 0, rad_params['ngates']

        if thlds is not None:
            thlds_vps = {'ZH [dBZ]': None, 'ZDR [dB]': None, 'rhoHV [-]': None,
                         'PhiDP [deg]': None, 'V [m/s]': None,
                         'KDP [deg/km]': None}
            thlds_vps.update(thlds)
            rvars_idx = {k: np.where((kv >= thlds_vps[k][0])
                                     & (kv <= thlds_vps[k][1]),
                                     True, False)
                         for k, kv in rad_vars.items()
                         if thlds_vps[k] is not None}
            valid_idx = True
            for i in rvars_idx:
                valid_idx = valid_idx*rvars_idx[i]

            rad_vars = {k: np.where(valid_idx, kv, np.nan)
                        for k, kv in rad_vars.items()}

        # if self.elev_angle < 89:
        #     raise TowerpyError('The elevation angle must be around 90 deg')
        # if self.elev_angle > 89:
        vpdata = {key: values for key, values in rad_vars.items()}
        validgates = 0
        vppol = {
            key:
                np.array([np.nanmean(values[0:rad_params['nrays'], i:i+1])
                          if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1]))>validgates
                          else np.nan for i in range(ri, rf)])
                 for key, values in vpdata.items()}
        if stats:
            vpsstd = {key: np.array([np.nanstd(values[0:rad_params['nrays'], i:i+1])
                                     if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1]))>validgates
                                     else np.nan for i in range(ri, rf)])
                      for key, values in vpdata.items()}
            vpsmin = {key: np.array([np.nanmin(values[0:rad_params['nrays'], i:i+1])
                                     if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1]))>validgates
                                     else np.nan for i in range(ri, rf)])
                      for key, values in vpdata.items()}
            vpsmax = {key: np.array([np.nanmax(values[0:rad_params['nrays'], i:i+1])
                                     if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1]))>validgates
                                     else np.nan for i in range(ri, rf)])
                      for key, values in vpdata.items()}
            vpssem = {key: np.array([np.nanstd(values[0:rad_params['nrays'], i:i+1])/np.sqrt(np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1])))
                                     if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1]))>validgates
                                     else np.nan for i in range(ri, rf)])
                      for key, values in vpdata.items()}
            self.vps_stats = {'std_dev': vpsstd,
                              'min': vpsmin,
                              'max': vpsmax,
                              'sem': vpssem}
        if 'V [m/s]' in rad_vars.keys() and isinstance(vppol['V [m/s]'],
                                                       np.ndarray):
            vppol['gradV [dV/dh]'] = np.array(np.gradient(vppol['V [m/s]'])).T
        if 'gradV [dV/dh]' in vppol.keys() and stats:
            self.vps_stats['std_dev']['gradV [dV/dh]'] = np.empty_like(self.vps_stats['std_dev']['V [m/s]'])
            self.vps_stats['min']['gradV [dV/dh]'] = np.empty_like(self.vps_stats['min']['V [m/s]'])
            self.vps_stats['max']['gradV [dV/dh]'] = np.empty_like(self.vps_stats['max']['V [m/s]'])
            self.vps_stats['sem']['gradV [dV/dh]'] = np.empty_like(self.vps_stats['sem']['V [m/s]'])
            self.vps_stats['std_dev']['gradV [dV/dh]'][:] = np.nan
            self.vps_stats['min']['gradV [dV/dh]'][:] = np.nan
            self.vps_stats['max']['gradV [dV/dh]'][:] = np.nan
            self.vps_stats['sem']['gradV [dV/dh]'][:] = np.nan
        self.vps = vppol
        self.profs_type = 'VPs'
        self.georef = {}
        profh = np.array([np.mean(rays)
                          for rays in rad_georef['beam_height [km]'].T])
        self.georef['profiles_height [km]'] = profh

    def pol_qvps(self, rad_georef, rad_params, rad_vars, thlds='default',
                 valid_gates=30, stats=False, exclude_vars=['V [m/s]'],
                 qvps_height_method='bh'):
        r"""
        Generate QVPs of polarimetric variables.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used to generate the QVPs.
        thlds : dict containing 2-element tuple, optional
            Thresholds [min, max] of radar variables used to discard gates
            in the azimuthal averaging. The default are: ZH [dBZ] > -10 and
            rhoHV > 0.6, according to [1]_.
        valid_gates : int, optional
            Number of valid gates (or azimuths) along the radial.
            The default is 30, according to [1]_.
        stats : Bool, optional
            Statistics of the QVPs generation:
                'std_dev': Standard Deviation

                'min': Min values

                'max': Max values

                'sem': Standard Error of the Mean
        exclude_vars : list, optional
            Name of the variables that will not be used to compute the QVPs.
            The default is ['V [m/s]'].

        Notes
        -----
        1. It is recommended to follow the routine described in [2]_ to
        preprocess :math:`\Phi_{DP}` and compute :math:`K_{DP}`.

        References
        ----------
        .. [1] Ryzhkov, A. V. et al. (2016) "Quasi-vertical profiles-A new way
            to look at polarimetric radar data"", Journal of Atmospheric and
            Oceanic Technology, 33(3), pp. 551–562.
            https://doi.org/10.1175/JTECH-D-15-0020.1
        .. [2] Griffin, E. M., Schuur, T. J., & Ryzhkov, A. V. (2018).
            "A Polarimetric Analysis of Ice Microphysical Processes in Snow,
            Using Quasi-Vertical Profiles", Journal of Applied Meteorology and
            Climatology, 57(1), 31-50. https://doi.org/10.1175/JAMC-D-17-0033.1
        """
        ri, rf = 0, rad_params['ngates']

        dh1 = (rad_georef['beam_height [km]'][0]
               + np.diff(rad_georef['range [m]']/1000, prepend=0)
               * np.sin(np.deg2rad(rad_params['elev_ang [deg]'])))

        dh2 = (rad_georef['beam_height [km]'][0]
               + rad_georef['beam_height [km]'][0]
               * np.deg2rad(rad_params['beamwidth [deg]'])
               * (1/np.tan(np.deg2rad(rad_params['elev_ang [deg]']))))

        dh = np.array([dhe if dhe > dh2[c1] else dh2[c1]
                       for c1, dhe in enumerate(dh1)])
        if qvps_height_method == 'vr':
            qvps_h = dh
        elif qvps_height_method == 'bh':
            qvps_h = rad_georef['beam_height [km]'][0]
        else:
            raise TowerpyError('Choose a method to compute the height of the'
                               'Quasi-Vertical Profiles.')
        thlds_qvps = {'ZH [dBZ]': [-10, np.inf], 'ZDR [dB]': None,
                      'rhoHV [-]': [0.6, np.inf], 'PhiDP [deg]': None,
                      'V [m/s]': None}
        thlds_qvps = {k: thlds_qvps[k] if k in thlds_qvps.keys()
                      else None for k in rad_vars.keys()}
        if thlds != 'default':
            thlds_qvps.update(thlds)
        rvars_idx = {k: np.where((kv >= thlds_qvps[k][0])
                                 & (kv <= thlds_qvps[k][1]), True, False)
                     for k, kv in rad_vars.items()
                     if thlds_qvps[k] is not None}
        valid_idx = True
        for i in rvars_idx:
            valid_idx = valid_idx*rvars_idx[i]

        rad_vars = {k: np.where(valid_idx, kv, np.nan)
                    for k, kv in rad_vars.items()}

        validgates = valid_gates
        # vars_nu = ['V [m/s]']  # Modify to compute QVPs of V.
        qvpvar = sorted(list(set([k for k in rad_vars.keys()
                                  if k not in exclude_vars])),
                        reverse=True)

        qvpdata = {key: values
                   for key, values in rad_vars.items() if key in qvpvar}
        qvppol = {key: np.array([np.nanmean(values[0:rad_params['nrays'],
                                                   i:i+1])
                                 if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i: i+1])) > validgates
                                 else np.nan
                                 for i in range(ri, rf)])
                  for key, values in qvpdata.items()}
        if stats:
            qvpsstd = {key: np.array([np.nanstd(values[0:rad_params['nrays'], i:i+1])
                                      if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1]))>validgates
                                      else np.nan for i in range(ri, rf)])
                       for key, values in qvpdata.items()}
            qvpsmin = {key: np.array([np.nanmin(values[0:rad_params['nrays'], i:i+1])
                                      if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1]))>validgates
                                      else np.nan for i in range(ri, rf)])
                       for key, values in qvpdata.items()}
            qvpsmax = {key: np.array([np.nanmax(values[0:rad_params['nrays'], i:i+1])
                                      if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1]))>validgates
                                      else np.nan for i in range(ri, rf)])
                       for key, values in qvpdata.items()}
            qvpssem = {key: np.array([np.nanstd(values[0:rad_params['nrays'], i:i+1])/np.sqrt(np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1])))
                                      if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1]))>validgates
                                      else np.nan for i in range(ri, rf)])
                       for key, values in qvpdata.items()}
            self.qvps_stats = {'std_dev': qvpsstd,
                               'min': qvpsmin,
                               'max': qvpsmax,
                               'sem': qvpssem}
        # qvppol['V [m/s]'] = qvppol['ZH [dBZ]']*np.nan
        # qvppol['gradV [dV/dh]'] = qvppol['ZH [dBZ]']*np.nan

        self.qvps = qvppol
        self.profs_type = 'QVPs'
        self.georef = {}
        self.georef['profiles_height [km]'] = qvps_h

    def pol_rdqvps(self, rscans_georef, rscans_params, rscans_vars, r0=None,
                   valid_gates=30, thlds='default', power_param1=0, vert_res=2,
                   power_param2=2, spec_range=50, all_desc=True,
                   exclude_vars=['V [m/s]'], qvps_height_method='bh',
                   plot_method=False):
        r"""
        Generate RD-QVPs of polarimetric variables.

        Parameters
        ----------
        rscans_georef : list
            List of dicts containing the georeference of the PPI scans.
        rscans_params : list
            List of dicts containing Radar technical details.
        rscans_vars : list
            List of dicts containing radar variables used to generate the
            RD-QVPs.
        r0 : float or list of floats, optional
            Initial range within the PPI scans to build the QVPS, in km.
            The default is None.
        valid_gates : int, optional
            Number of valid gates (or azimuths) along the radial.
            The default is 30, according to [1]_.
        thlds : dict containing 2-element tuple, optional
            Thresholds [min, max] of radar variables used to discard gates
            in the azimuthal averaging. The default are: ZH [dBZ] > -10 and
            rhoHV > 0.6, according to [1]_.
        power_param1 : float, optional
            Power parameter for :math:`r_i \leq d-1`. The default is 0,
            according to [2]_.
        vert_res : float, optional
            Resolution of the common vertical axis, in m. The default is 2.
        power_param2 : float, optional
            Power parameter for :math:`r_i > d-1`. The default is 2,
            according to [2]_.
        spec_range : int, optional
            Range from the radar within which the data will be used.
            The default is 50.
        all_desc : bool, optional
            If False, the function provides descriptors using an average of
            datetime and elevations and will not give the initial QVPs used
            to compute the RD-QPVs. The default is True.
        exclude_vars : list, optional
            Name of the variables that will not be used to compute the QVPs.
            The default is ['V [m/s]'].
        qvps_height_method : str, optional
            'bh' or 'vr'
        plot_method : bool, optional
            Plot the RD-QVPS. The default is False.

        Returns
        -------
        None.

        References
        ----------
        .. [1] Ryzhkov, A. V. et al. (2016)
            ‘Quasi-vertical profiles-A new way to look at polarimetric
            radar data’,
            Journal of Atmospheric and Oceanic Technology, 33(3), pp. 551–562.
            https://doi.org/10.1175/JTECH-D-15-0020.1
        .. [2] Tobin, D. M., & Kumjian, M. R. (2017). Polarimetric Radar and
            Surface-Based Precipitation-Type Observations of Ice Pellet to
            Freezing Rain Transitions, Weather and Forecasting, 32(6),
            2065-2082. https://doi.org/10.1175/WAF-D-17-0054.1
        .. [3] Griffin, E. M., Schuur, T. J., & Ryzhkov, A. V. (2018).
            A Polarimetric Analysis of Ice Microphysical Processes in Snow,
            Using Quasi-Vertical Profiles, Journal of Applied Meteorology and
            Climatology, 57(1), 31-50. https://doi.org/10.1175/JAMC-D-17-0033.1

        """
        if r0 is None:
            r0 = [0 for i in rscans_params]
        else:
            if isinstance(r0, (int, float)):
                r0 = [find_nearest(i['range [m]'], r0*1000)
                      for i in rscans_georef]
            elif isinstance(r0, list):
                if len(r0) == len(rscans_vars):
                    r0 = [find_nearest(i['range [m]'], r0[c]*1000)
                          for c, i in enumerate(rscans_georef)]
                else:
                    raise TowerpyError('Length of values r0 does not match'
                                       ' length of elevation index'
                                       ' (rscans_vars)')
        # if rf is None:
        #     rf = [i['ngates'] for i in rscans_params]
        # else:
        #     if isinstance(rf, (int, float)):
        #         rf = [find_nearest(i['range [m]'], rf*1000)
        #               for i in rscans_georef]
        #     elif isinstance(r0, list):
        #         if len(rf) == len(rscans_vars):
        #             rf = [find_nearest(i['range [m]'], rf[c]*1000)
        #                   for c, i in enumerate(rscans_georef)]
        #         else:
        #             raise TowerpyError('Length of values rf does not match'
        #                                ' length of elevation index'
        #                                ' (rscans_vars)')
        # rf = [find_nearest(i['range [m]'], spec_range*1000)
        #       for i in rscans_georef]
        rf = [i['ngates'] for i in rscans_params]

        dh1 = [v['beam_height [km]'][0]+np.diff(v['range [m]']/1000, prepend=0)
               * np.sin(np.deg2rad(rscans_params[c]['elev_ang [deg]']))
               for c, v in enumerate(rscans_georef)]

        dh2 = [v['beam_height [km]'][0]+v['beam_height [km]'][0]
               * np.deg2rad(rscans_params[c]['beamwidth [deg]'])
               * (1/np.tan(np.deg2rad(rscans_params[c]['elev_ang [deg]'])))
               for c, v in enumerate(rscans_georef)]

        dh = [np.array([dhi if dhi > dh2[c1][c2] else dh2[c1][c2]
                        for c2, dhi in enumerate(dhe)])
              for c1, dhe in enumerate(dh1)]
        if qvps_height_method == 'vr':
            qvps_h = [dh[c] for c, v in enumerate(rscans_georef)]
        elif qvps_height_method == 'bh':
            qvps_h = [v['beam_height [km]'][0]
                      for n, v in enumerate(rscans_georef)]
        else:
            raise TowerpyError('Choose a method to compute the height of the'
                               'Quasi-Vertical Profiles.')
        qvps_hr = [hb[r0[c]:rf[c]] for c, hb in enumerate(qvps_h)]

        vg = valid_gates

        thlds_qvps = {'ZH [dBZ]': [-10, 100], 'ZDR [dB]': None,
                      'rhoHV [-]': [0.6, 10], 'PhiDP [deg]': None,
                      'V [m/s]': None, 'KDP [deg/km]': None}
        if thlds != 'default':
            thlds_qvps.update(thlds)
        rvars_idx = [{k: np.where((kv >= thlds_qvps[k][0])
                                  & (kv <= thlds_qvps[k][1]), True, False)
                     for k, kv in rad_vars.items()
                     if thlds_qvps[k] is not None} for rad_vars in rscans_vars]
        valid_idx = []
        for elevsc in rvars_idx:
            validxs = True
            for i in elevsc:
                validxs = validxs * elevsc[i]
            valid_idx.append(validxs)

        # vars_nu = ['V [m/s]']  # Modify to compute QVPs of V.
        qvpvar = sorted(list(set([k for robj in rscans_vars
                                  for k in robj.keys()
                                  if k not in exclude_vars])),
                        reverse=True)

        rscans_vc = [{k: np.where(valid_idx[c], kv, np.nan)
                      for k, kv in rad_vars.items() if k in qvpvar}
                     for c, rad_vars in enumerate(rscans_vars)]

        qvppol = [{key: np.array([np.nanmean(values[0:
                                                    rscans_params[c]['nrays'],
                                                    i:i+1])
                                 if np.count_nonzero(~np.isnan(values[0: rscans_params[c]['nrays'],
                                                                      i: i+1])) > vg
                                 else np.nan
                                  for i in range(r0[c], rf[c])])
                  for key, values in qvpdata.items()}
                  for c, qvpdata in enumerate(rscans_vc)]

        # qvppol['V [m/s]'] = qvppol['ZH [dBZ]']*np.nan
        # qvppol['gradV [dV/dh]'] = qvppol['ZH [dBZ]']*np.nan

        yaxis = np.arange(0, np.ceil(max([np.nanmax(hb) for hb in qvps_hr])),
                          vert_res/1000)

        qvps_itp = [{nv: np.interp(yaxis, qvps_hr[c], pvars,
                                   left=np.nan, right=np.nan
                                   )
                    for nv, pvars in qvps.items()}
                    for c, qvps in enumerate(qvppol)]

        rng_d = [rngs['range [m]']/1000
                 for c, rngs in enumerate(rscans_georef)]
        rng_itp = [np.linspace(rng[0], rng[-1], len(yaxis))
                   for c, rng in enumerate(rng_d)]

        w_func = np.array([np.array([1 if spec_range-1 < rngi <= spec_range
                                     else
                                     1/(abs(rngi-(spec_range-1))**power_param1)
                                     if rngi <= spec_range-1 else
                                     1/(abs(rngi-(spec_range-1))**power_param2)
                                     if rngi > spec_range-1 else
                                     np.nan
                                     for rngi in rngelv])
                           for rngelv in rng_itp]).T

        rdqvps_vidx = {pvar: np.array([~np.isnan([e[pvar][i]
                                                  for e in qvps_itp])
                                       for i in range(len(yaxis))])
                       for pvar in qvpvar}

        rdqvps_val = {pvar: np.array([[e[pvar][i] for e in qvps_itp]
                                      for i in range(len(yaxis))])
                      for pvar in qvpvar}

        rdqvps = {pvar: np.array([np.nansum(rdqvps_val[pvar][row]
                                            * (w_func[row]
                                            * rdqvps_vidx[pvar][row]))
                                  / np.nansum((w_func[row]
                                              * rdqvps_vidx[pvar][row]))
                                  if
                                  np.count_nonzero(rdqvps_vidx[pvar][row]) >= 1
                                  else np.nan for row in range(len(yaxis))])
                  for pvar in qvpvar}
        self.rd_qvps = rdqvps
        self.profs_type = 'RD-QVPs'
        self.georef = {}
        self.georef['profiles_height [km]'] = yaxis
        if all_desc:
            self.qvps_itp = qvps_itp
            self.elev_angle = np.array([i['elev_ang [deg]']
                                        for i in rscans_params])
            self.scandatetime = [i['datetime'] for i in rscans_params]
        else:
            self.elev_angle = np.average(np.array([i['elev_ang [deg]']
                                                   for i in rscans_params]))
            dmmydt = [i['datetime'] for i in rscans_params]
            self.scandatetime = datetime.fromtimestamp(sum(d.timestamp()
                                                           for d in dmmydt)
                                                       / len(dmmydt))
        self.file_name = 'RD-QVPs'
        # snames_list = [i['site_name'] for i in rscans_params]
        # if snames_list.count(snames_list[0]) == len(snames_list):
        #     self.site_name = snames_list[0]
        # else:
        #     self.site_name = [i['site_name'] for i in rscans_params]
        if plot_method:
            rad_display.plot_rdqvps(rscans_georef, rscans_params, self,
                                    spec_range=spec_range, all_desc=all_desc)


# =============================================================================
# %% xarray implementation
# =============================================================================

def _resolve_inp_names(ds, inp_names):
    """Resolve canonical-to-dataset variable name mapping."""
    canonical_vars = ("DBZ", "ZDR", "RHOHV", "PHIDP")
    # Case 1: preset
    if isinstance(inp_names, str):
        if inp_names == "standard":
            present = {c: c for c in canonical_vars if c in ds.data_vars}
            if not present:
                raise ValueError(
                    "No standard QVP variables found. "
                    f"Expected any of {canonical_vars}."
                )
            return present
        raise ValueError(f"Unknown inp_names preset: {inp_names!r}")
    # Case 2: explicit mapping
    if isinstance(inp_names, dict):
        missing = [v for v in inp_names.values() if v not in ds.data_vars]
        if missing:
            raise KeyError(f"Dataset variables not found: {missing}")
        return dict(inp_names)

    raise TypeError("inp_names must be a dict or 'standard'.")


def _resolve_thresholds(inp_map, thresholds):
    """Resolve per-variable threshold definitions."""
    default_thr = {"DBZ": (-10., np.inf),
                   "RHOHV": (0.6, np.inf),
                   "ZDR": None,
                   "PHIDP": None,
                   "KDP": None,
                   "AH": None,
                   }
    # Disable thresholding
    if thresholds is None:
        return {canon: None for canon in inp_map}
    # Default thresholds
    if thresholds == "default":
        return {canon: default_thr.get(canon, None)
                for canon in inp_map}
    # User-provided thresholds
    thr = {}
    for canon in inp_map:
        if canon in thresholds:
            thr[canon] = thresholds[canon]
        else:
            thr[canon] = default_thr.get(canon, None)
    return thr


def _build_joint_mask(ds_sel, thresholds, inp_map, azimuth_dim, range_dim):
    """Build a joint validity mask from threshold definitions."""
    mask = xr.ones_like(next(iter(ds_sel.data_vars.values())), dtype=bool)
    for canon, ds_name in inp_map.items():
        v = ds_sel[ds_name]
        thr = thresholds.get(canon)
        if thr is None:
            continue
        if isinstance(thr, tuple):
            lo, hi = thr
            mask &= (v >= lo) & (v <= hi)
        else:
            mask &= v >= thr
    return mask


def build_vp(ds, inp_names=None, thresholds=None, valid_gates=0,
             method="mean", stats=False):
    """
    Compute a Vertical Profile (VP) from a birdbath scan.

    Parameters
    ----------
    ds : xarray.Dataset
        Birdbath scan containing the required polarimetric variables. Must
        include azimuth, range, elevation and beam height coordinates.
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{'azi': 'azimuth', 'rng': 'range', 'elv': 'elevation',
        'beam_height': 'beamc_height', "DBZ": "DBZH", "ZDR": "ZDR",
        "RHOHV": "RHOHV", "PHIDP": "PHIDP", "VRADV": "VRADV"}``
    thresholds : dict, "default", or None, optional
        Thresholds applied to variables prior to azimuthal averaging.
        ``"default"`` applies DBZ and RHOHV thresholds; ``None`` disables 
        thresholding.
    valid_gates : int, default 0
        Minimum number of valid (non‑NaN, thresholded) azimuthal samples
        required for a gate to contribute to the VP.
    method : {"mean", "median"}, default "mean"
        Aggregation method used for azimuthal averaging.
    stats : bool, default False
        If ``True``, compute additional statistics for each variable:
        standard deviation (``std_<var>``), minimum (``min_<var>``),
        maximum (``max_<var>``), and standard error of the mean
        (``sem_<var>``).

    Returns
    -------
    vp : xarray.Dataset
        Dataset containing the VP variables on a height coordinate,
        including the vertical gradient of Doppler radial velocity
        (``GRAD_VRADV``) when available, and optional statistics.

    Notes
    -----
    * The range dimension is replaced by the height coordinate.
    * Thresholding and masking are applied before aggregation.
    * _resolve_thresholds is used to apply thresholding.
    * Elevation (in degrees) and mid‑scan time are added as coordinates
      when available.
    """
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # 1. Canonical + coordinate defaults
    defaults = {'azi': 'azimuth', 'rng': 'range', 'elv': 'elevation',
                'beam_height': 'beamc_height', "DBZ": "DBZH", "ZDR": "ZDR",
                "RHOHV": "RHOHV", "PHIDP": "PHIDP", "VRADV": "VRADV"}
    names = {**defaults, **(inp_names or {})}
    coord_keys = {"azi", "rng", "elv", 'beam_height'}
    var_names = {k: v for k, v in names.items() if k not in coord_keys}
    # 2. Resolve canonical → dataset mapping
    inp_map = _resolve_inp_names(ds, var_names or "standard")
    # 3. Select variables
    ds_sel = ds[list(inp_map.values())]
    # 4. Resolve thresholds
    thr = _resolve_thresholds(inp_map, thresholds)
    # 5. Build joint mask
    mask = _build_joint_mask(ds_sel, thr, inp_map, azimuth_dim=names["azi"],
                             range_dim=names["rng"])
    ds_masked = ds_sel.where(mask)
    # 6. Azimuthal aggregation
    count = ds_masked.notnull().sum(dim=names["azi"])
    if method == "mean":
        agg = ds_masked.mean(dim=names["azi"], skipna=True)
    elif method == "median":
        agg = ds_masked.median(dim=names["azi"], skipna=True)
    agg = agg.where(count > valid_gates)
    # 7. Height coordinate
    height = ds[names["beam_height"]].mean(dim=names["azi"])
    height.attrs.setdefault("units",
                            ds[names["beam_height"]].attrs.get("units", "km"))
    agg = agg.assign_coords(height=height).swap_dims({names["rng"]: "height"})
    if "range" in agg.coords:
        agg = agg.drop_vars("range")
    # 8. Assemble VP dataset
    data_vars = {}
    for canon, ds_name in inp_map.items():
        data_vars[ds_name] = agg[ds_name]   # keep dataset name
    vp = xr.Dataset(data_vars=data_vars, coords={"height": height})
    # Clean up leftover range dimension/coordinate
    if "range" in vp.coords:
        vp = vp.drop_vars("range")
    if "range" in vp.dims:
        vp = vp.drop_dims("range")
    # 9. Optional statistics
    if stats:
        std = ds_masked.std(dim=names['azi'], skipna=True)
        min_ = ds_masked.min(dim=names['azi'], skipna=True)
        max_ = ds_masked.max(dim=names['azi'], skipna=True)
        sem = std / np.sqrt(count)
        std = std.where(count > valid_gates)
        min_ = min_.where(count > valid_gates)
        max_ = max_.where(count > valid_gates)
        sem = sem.where(count > valid_gates)
        std = std.assign_coords(height=height).swap_dims(
            {names["rng"]: "height"})
        min_ = min_.assign_coords(height=height).swap_dims(
            {names["rng"]: "height"})
        max_ = max_.assign_coords(height=height).swap_dims(
            {names["rng"]: "height"})
        sem = sem.assign_coords(height=height).swap_dims(
            {names["rng"]: "height"})
        if "range" in std.coords:
            std = std.drop_vars("range")
        if "range" in min_.coords:
            min_ = min_.drop_vars("range")
        if "range" in max_.coords:
            max_ = max_.drop_vars("range")
        if "range" in sem.coords:
            sem = sem.drop_vars("range")
        for canon, ds_name in inp_map.items():
            vp[f"std_{ds_name}"] = std[ds_name]
            vp[f"min_{ds_name}"] = min_[ds_name]
            vp[f"max_{ds_name}"] = max_[ds_name]
            vp[f"sem_{ds_name}"] = sem[ds_name]
    # 10. Time + elevation
    if "time" in ds.coords:
        mid_np, mid_py = scan_midtime(ds["time"].values)
        vp = vp.assign_coords(time=mid_np)
        vp.attrs["scan_datetime"] = mid_py
    if "elevation" in ds.coords:
        elev_deg = float(convert(ds.elevation, 'deg').mean())
        vp = vp.assign_coords(elevation=elev_deg)
    # 11. Vertical velocity gradient
    if "VRADV" in inp_map:
        ds_name = inp_map["VRADV"]
        vp[f"GRAD_{ds_name}"] = vp[ds_name].differentiate("height")
    # 12. Assign metadata to ALL variables
    for var in vp.data_vars:
        if var in sweep_vars_attrs_f:
            vp[var].attrs.update(sweep_vars_attrs_f[var])
    # Extract site name
    if "where" in ds.attrs and isinstance(ds.attrs["where"], dict):
        rname_out = {"site_name": ds.attrs["where"].get("site_name", "Radar")}
    elif "site_name" in ds.attrs:
        rname_out = {"site_name": ds.attrs["site_name"]}
    else:
        rname_out = {"site_name": "Radar"}
    # 13. Provenance
    #TODO: add step_description
    extra = {'step_description': ('')}
    vp = record_provenance(
        vp, step="build_vp", inputs=list(inp_map.values()),
        outputs=list(vp.data_vars),
        parameters={"aggregation_method": method,
                    "inp_names": dict(inp_map),
                    "thresholds": thr,
                    "valid_gates": valid_gates,
                    "elevation_deg": float(vp.coords.get("elevation", np.nan))},
        extra_attrs=extra, module_provenance='towerpy.profs.polprofs.build_vp')
    vp.attrs["profs_type"] = "Vertical Profiles"
    vp.attrs["where"] = rname_out   # Python dict in memory
    # Pass dataset‑level processing chain
    if "processing_chain" in ds.attrs:
        vp.attrs["input_processing_chain"] = copy.deepcopy(
            ds.attrs["processing_chain"])
    return vp


def build_qvp(ds, inp_names=None, beamwidth=None, thresholds="default",
              valid_gates=30, method="mean", stats=False, resolution=False):
    r"""
    Compute a Quasi‑Vertical Profile (QVP) from a single‑elevation PPI, 
    according to the methodology described in Ryzhkov et al. (2016)

    Parameters
    ----------
    ds : xarray.Dataset
        A single‑elevation PPI containing the required polarimetric variables
        Must include azimuth, range, elevation and beam height coordinates.
    inp_names : dict, default None
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{'azi': 'azimuth', 'rng': 'range', 'elv': 'elevation',
        'beam_height': 'beamc_height', "DBZ": "DBZH", "ZDR": "ZDR",
        "RHOHV": "RHOHV", "PHIDP": "PHIDP"}``
    beamwidth : float, default None
        Radar beamwidth in degrees. Required only when
        ``resolution=True`` for vertical‑resolution diagnostics.
    thresholds : dict or {"default", None}, default "default"
        Thresholds applied to variables prior to azimuthal averaging.
        ``"default"`` applies DBZ > -10 and RHOHV > 0.6; ``None`` disables
        thresholding.
    valid_gates : int, default 30
        Minimum number of valid (non‑NaN, thresholded) azimuthal samples
        required for a gate to contribute to the QVP.
    method : {"mean", "median"}, default "mean"
        Aggregation method used for azimuthal averaging.
    stats : bool, default False
        If ``True``, compute additional statistics for each variable:
        standard deviation (``std_<var>``), minimum (``min_<var>``),
        maximum (``max_<var>``), and standard error of the mean
        (``sem_<var>``).
    resolution : bool, optional
        If ``True``, compute the effective vertical resolution following [1]_.
        The result is stored as the variable ``VRES``.

    Returns
    -------
    qvp : xarray.Dataset
        Dataset containing the QVP variables on a height coordinate,
        with optional statistics and vertical‑resolution diagnostics.
    
    Notes
    -----
    * The range dimension is replaced by the height coordinate.
    * Thresholding and masking are applied before azimuthal aggregation.
    * Elevation (in degrees) and mid‑scan time are added as coordinates when
      available.
    * It is recommended to follow the routine described in [2]_ to
      preprocess :math:`\Phi_{DP}` and compute :math:`K_{DP}`.

    References
    ----------
    .. [1] Ryzhkov, A., Zhang, P., Reeves, H., Kumjian, M., Tschallener, T.,
        Trömel, S., & Simmer, C. (2016). Quasi-Vertical Profiles—A new way to
        look at polarimetric radar data. Journal of Atmospheric and Oceanic
        Technology, 33(3), 551–562. https://doi.org/10.1175/jtech-d-15-0020.1
    .. [2] Griffin, E. M., Schuur, T. J., & Ryzhkov, A. V. (2018). A 
        polarimetric analysis of Ice microphysical processes in snow, using
        Quasi-Vertical profiles. Journal of Applied Meteorology and
        Climatology, 57(1), 31–50. https://doi.org/10.1175/jamc-d-17-0033.1
    """
    # =============================================================================
    # Resolve variable names
    # =============================================================================
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    defaults = {'azi': 'azimuth', 'rng': 'range', 'elv': 'elevation',
                'beam_height': 'beamc_height', "DBZ": "DBZH", "ZDR": "ZDR",
                "RHOHV": "RHOHV", "PHIDP": "PHIDP"}
    names = {**defaults, **(inp_names or {})}
    coord_keys = {"azi", "rng", "elv", 'beam_height'}
    var_names   = {k: v for k, v in names.items() if k not in coord_keys}
    # 1. Resolve canonical → dataset variable mapping
    inp_map = _resolve_inp_names(ds, var_names or "standard")
    # 2. Select only the needed variables
    ds_sel = ds[list(inp_map.values())]
    rng_km = convert(ds[names["rng"]], "km")
    # azi_rad = convert(ds[names["azi"]], "rad")
    elv_rad = convert(ds[names["elv"]], "rad")[0]
    # 3. Resolve thresholds (canonical → threshold)
    thr = _resolve_thresholds(inp_map, thresholds)
    # 4. Build joint mask
    mask = _build_joint_mask(ds_sel, thr, inp_map, azimuth_dim=names["azi"],
                             range_dim=names["rng"])
    ds_masked = ds_sel.where(mask)
    # 5. Azimuthal aggregation
    count = ds_masked.notnull().sum(dim=names['azi'])
    if method == "mean":
        agg = ds_masked.mean(dim=names['azi'], skipna=True)
    elif method == "median":
        agg = ds_masked.median(dim=names['azi'], skipna=True)
    else:
        raise ValueError("method must be 'mean' or 'median'.")
    agg = agg.where(count > valid_gates)
    # 6. Height coordinate
    bh = ds[names["beam_height"]]
    # Allow 2D (azimuth, range) or 1D (range)
    if names['azi'] in bh.dims and names["rng"] in bh.dims:
        height = bh.mean(dim=names['azi'])
    else:
        height = bh
    # IMPORTANT: keep dimension as "range"
    height.attrs.setdefault("units", bh.attrs.get('units'))
    agg = agg.assign_coords(height=height).swap_dims({names["rng"]: "height"})
    # Remove the old range coordinate from all variables
    if "range" in agg.coords:
        agg = agg.drop_vars("range")
    # 7. Assemble QVP dataset
    data_vars = {}
    for canon, ds_name in inp_map.items():
        da = agg[ds_name]          # no rename
        data_vars[ds_name] = da    # use dataset name
    qvp = xr.Dataset(data_vars=data_vars, coords={"height": height})
    if "range" in qvp.coords:
        qvp = qvp.drop_vars("range")
    if "range" in qvp.dims:
        qvp = qvp.drop_dims("range")
    # 8. Optional statistics
    if stats:
        std = ds_masked.std(dim=names['azi'], skipna=True)
        min_ = ds_masked.min(dim=names['azi'], skipna=True)
        max_ = ds_masked.max(dim=names['azi'], skipna=True)
        sem = std / np.sqrt(count)
        std = std.where(count > valid_gates)
        min_ = min_.where(count > valid_gates)
        max_ = max_.where(count > valid_gates)
        sem = sem.where(count > valid_gates)
        std = std.assign_coords(height=height).swap_dims(
            {names["rng"]: "height"})
        min_ = min_.assign_coords(height=height).swap_dims(
            {names["rng"]: "height"})
        max_ = max_.assign_coords(height=height).swap_dims(
            {names["rng"]: "height"})
        sem = sem.assign_coords(height=height).swap_dims(
            {names["rng"]: "height"})
        if "range" in std.coords:
            std = std.drop_vars("range")
        if "range" in min_.coords:
            min_ = min_.drop_vars("range")
        if "range" in max_.coords:
            max_ = max_.drop_vars("range")
        if "range" in sem.coords:
            sem = sem.drop_vars("range")
        for canon, ds_name in inp_map.items():
            qvp[f"std_{ds_name}"] = std[ds_name]
            qvp[f"min_{ds_name}"] = min_[ds_name]
            qvp[f"max_{ds_name}"] = max_[ds_name]
            qvp[f"sem_{ds_name}"] = sem[ds_name]
    # 9. Compute mid-scan time
    if "time" in ds.coords:
        mid_np, mid_py = scan_midtime(ds["time"].values)
        qvp = qvp.assign_coords(time=mid_np)
        # qvp.attrs["scan_datetime"] = mid_py
        # qvp.attrs["scan_datetime"] = mid_np
        ts_ns = int(mid_np.astype("datetime64[ns]").astype("int"))
        qvp.attrs["scan_datetime_unix_ns"] = ts_ns
        qvp.attrs["scan_datetime_iso"] = np.datetime_as_string(
            np.datetime64(ts_ns, "ns"), unit="ms")
        qvp.attrs["scan_datetime_unit"] = "ns since 1970-01-01"
    # 10. Compute vertical resolution diagnostics
    if resolution:
        # Beamwidth (radians)
        if beamwidth is not None:
            try:
                bw = float(beamwidth)
            except Exception:
                raise ValueError(f"beamwidth must be numeric, got {beamwidth!r}")
        else:
            # Metadata lookup
            bw = get_attrval("beamwidth", ds, required=True)
            try:
                bw = float(bw)
            except Exception:
                raise ValueError(f"Beamwidth attribute is not numeric: {bw!r}")
        bw_rad = np.deg2rad(bw)
        # Range gate spacing Δr (km)
        dr = (rng_km.diff(names["rng"]).median()).item()   # scalar km
        # Height array
        h = agg["height"]  # dims: ("height",)
        # Δh1 = Δr * sin(α)
        delta_h1 = dr * np.sin(elv_rad)
        delta_h1 = xr.ones_like(h, dtype=float) * delta_h1
        delta_h1.attrs.update({"long_name": "vertical_resolution_range_gate",
                               "description": "Δh1 = Δr * sin(elev)",
                               "units": height.attrs.get("units", "")})
        # Δh2 = h * θ * cot(α)
        delta_h2 = h * bw_rad * (np.cos(elv_rad) / np.sin(elv_rad))
        delta_h2.attrs.update({"long_name": "vertical_resolution_beam_broadening",
                               "description": "Δh2 = h * beamwidth * cot(elev)",
                               "units": height.attrs.get("units", "")})
        # Δh = max(Δh1, Δh2)
        delta_h = xr.ufuncs.maximum(delta_h1, delta_h2)
        delta_h.attrs.update({"long_name": "effective_vertical_resolution",
                              "description": "Δh = max(Δh1, Δh2)",
                              "units": height.attrs.get("units", "")})
        # Attach to QVP dataset
        qvp["VRES"] = delta_h
        # Add to provenance
        
    # 11. Assign metadata to ALL variables
    for var in qvp.data_vars:
        if var in sweep_vars_attrs_f:
            qvp[var].attrs.update(sweep_vars_attrs_f[var])
    # Extract site name
    if "where" in ds.attrs and isinstance(ds.attrs["where"], dict):
        rname_out = {"site_name": ds.attrs["where"].get("site_name", "Radar")}
    elif "site_name" in ds.attrs:
        rname_out = {"site_name": ds.attrs["site_name"]}
    else:
        rname_out = {"site_name": "Radar"}
    # 12. Provenance
    #TODO: add step_description
    extra = {'step_description': ('')}
    qvp = record_provenance(
        qvp, step="build_qvp", inputs=list(inp_map.values()),
        outputs=list(qvp.data_vars),
        parameters={"aggregation_method": method,
                    "inp_names": dict(inp_map),
                    "thresholds": thr,
                    "valid_gates": valid_gates,
                    "elevation_deg": float(qvp.coords.get("elevation", np.nan)),
                    "beam_geometry": names["beam_height"]},
        extra_attrs=extra, module_provenance='towerpy.profs.polprofs.build_qvp')
    #TODO: maybe should not record provenance but use add_correction_step
    if resolution:
        qvp = record_provenance(
            qvp, step="vertical_resolution",
            inputs=[names["rng"], names["elv"]], outputs=["VRES"],
            parameters={
                "beamwidth_deg": float(bw), "range_gate_km": float(dr),
                "elevation_deg": float(convert(ds[names["elv"]], "deg")[0])},
            extra_attrs=extra,
            module_provenance='towerpy.profs.polprofs.build_qvp')
    qvp.attrs["profs_type"] = "Quasi-Vertical Profiles"
    qvp.attrs["where"] = rname_out   # Python dict in memory
    # Pass dataset‑level processing chain
    if "processing_chain" in ds.attrs:
        qvp.attrs["input_processing_chain"] = copy.deepcopy(
            ds.attrs["processing_chain"])
    return qvp


def build_rdqvp(dss, qvp_kwargs=None, height_res=0.002, spec_range=50.,
                power_param=2., stats=False):
    r"""
    Compute Range‑Defined Quasi‑Vertical Profiles (RD‑QVP) from a set of
    single‑elevation PPIs, following the methodology of Tobin & Kumjian (2017).

    Parameters
    ----------
    dss : list of xarray.Dataset
        List of single‑elevation PPIs with polarimetric variables. Each
        dataset must contain include azimuth, range, elevation and beam height
        coordinates.
    qvp_kwargs : dict, default None
        Optional keyword arguments forwarded directly to :func:`build_qvp`
        when constructing each QVP.
    height_res : float, default 0.002
        Vertical resolution (km) of the common height grid used for
        interpolation and RD‑QVP combination.
    spec_range : float, default 50.0
        Desired range :math:`d` (km) used in the RD‑QVP weighting
        formulation.
    power_param : float, default 2.0
        Exponent :math:`p` used in the weighting function for
        :math:`r_i(h) > d - 1`.
    stats : bool, default False
        If ``True``, propagate QVP statistics (``std_*``, ``min_*``,
        ``max_*``, ``sem_*``) through the RD‑QVP combination.

    Returns
    -------
    rdqvp : xarray.Dataset
        RD‑QVP dataset with height coordinate, RD‑QVP variables, interpolated
        QVPs, elevation metadata, and provenance.
        
    Notes
    -----
    * Each elevation is first converted into a QVP using :func:`build_qvp`,
      following [1]_.
    * These QVPs are interpolated onto a common height grid and then combined
      using the range‑dependent weighting scheme of [2]_:
        .. math::

         w_i(h) =
         \\begin{cases}
             1, & r_i(h) \\le d \\\\
             1 / |r_i(h) - (d - 1)|^p, & r_i(h) > d - 1
         \\end{cases}
    * It is recommended to follow the routine described in [3]_ to
      preprocess :math:`\Phi_{DP}` and compute :math:`K_{DP}`.

    References
    ----------
    .. [1] Ryzhkov, A., Zhang, P., Reeves, H., Kumjian, M., Tschallener, T.,
        Trömel, S., & Simmer, C. (2016). Quasi-Vertical Profiles—A new way to
        look at polarimetric radar data. Journal of Atmospheric and Oceanic
        Technology, 33(3), 551–562. https://doi.org/10.1175/jtech-d-15-0020.1
    .. [2] Tobin, D. M., & Kumjian, M. R. (2017). Polarimetric radar and
        Surface-Based Precipitation-Type observations of ice pellet to freezing
        rain transitions. Weather and Forecasting, 32(6), 2065–2082.
        https://doi.org/10.1175/waf-d-17-0054.1
    .. [3] Griffin, E. M., Schuur, T. J., & Ryzhkov, A. V. (2018). A 
        polarimetric analysis of Ice microphysical processes in snow, using
        Quasi-Vertical profiles. Journal of Applied Meteorology and
        Climatology, 57(1), 31–50. https://doi.org/10.1175/jamc-d-17-0033.1
    """
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # 1. Build QVPs for each elevation
    qvp_kwargs = qvp_kwargs or {}
    qvps = []
    for ds in dss:
        qvp = build_qvp(ds, **qvp_kwargs)   # RD‑QVP does not use Δh
        qvps.append(qvp)
    # 2. Common height grid
    max_height = max(float(qvp.height.max()) for qvp in qvps)
    common_height = np.arange(0.0, max_height, height_res)
    # 3. Interpolate QVPs onto common height grid
    qvps_interp = [qvp.interp(height=common_height) for qvp in qvps]
    # Stack into (elevation, variable, height)
    # dims: elevation, variable, height
    qvps_interp_ds = xr.concat(
        [qvp.to_array("variable") for qvp in qvps_interp], dim="elevation")
    # 4. Build range profile r_i(h) for each elevation
    range_profiles = []
    # Extract beam-height variable name from qvp_kwargs
    inp = qvp_kwargs.get("inp_names", {})
    beam_height_name = inp.get("beam_height", "beamc_height")
    azi_name = inp.get("azi", "azimuth")
    rng_name = inp.get("rng", "range")
    for ds in dss:
        # Range in km
        rng_km = convert(ds[rng_name], "km")
        # Beam height vs range (mean over azimuth)
        bh = ds[beam_height_name]
        if azi_name in bh.dims:
            bh = bh.mean(dim=azi_name)
        # Invert height->range: create DA with dimension "height"
        rng_vs_h = xr.DataArray(rng_km.values, dims=("height",),
                                coords={"height": bh.values})
        # Interpolate to common height grid
        rng_profile = rng_vs_h.interp(height=common_height)
        range_profiles.append(rng_profile)
    # Stack into (elevation, height)
    range_profile = xr.concat(range_profiles, dim="elevation")
    # 5. Range‑dependent weights w_i(h)
    r = range_profile  # dims: (elevation, height)
    d = float(spec_range)
    # Weighting:
    w = xr.where(r <= d, 1., 1. / xr.ufuncs.fabs(r - (d - 1.))**power_param)
    # 6. Weighted combination for each variable
    vars_to_combine = [str(v) for v in qvps_interp_ds["variable"].values
                       if not str(v).startswith(("std_", "min_", "max_", "sem_"))]
    rdqvp_vars = {}
    for var in vars_to_combine:
        da = qvps_interp_ds.sel(variable=var)  # dims: (elevation, height)
        num = (da * w).sum(dim="elevation", skipna=True)
        den = w.where(~xr.ufuncs.isnan(da)).sum(dim="elevation")
        rdqvp_vars[var] = num / den
    # 7. Build RD‑QVP dataset
    rdqvp = xr.Dataset({var: (("height",), rdqvp_vars[var].values)
                        for var in vars_to_combine},
                       coords={"height": common_height})
    rdqvp.height.attrs["units"] = "km"
    # 8. Attach interpolated QVPs + elevation metadata
    rdqvp["qvp_interp"] = qvps_interp_ds  # dims: elevation, variable, height
    elev_angles = []
    scan_datetimes = []
    for qvp in qvps:
        elev = qvp.coords.get("elevation", None)
        elev_angles.append(float(elev.values) if elev is not None else np.nan)
        scan_datetimes.append(qvp.attrs.get("scan_datetime", None))
    rdqvp = rdqvp.assign_coords(
        elevation=("elevation", np.arange(len(qvps_interp))),
        elevation_angle=("elevation", elev_angles),
        scan_datetime=("elevation", scan_datetimes))
    # 9. Metadata + provenance
    for var in rdqvp.data_vars:
        if var in sweep_vars_attrs_f:
            rdqvp[var].attrs.update(sweep_vars_attrs_f[var])
    # 10. Collect QVP variable names (union across elevations)
    all_qvp_vars = sorted( set().union(*[set(qvp.data_vars) for qvp in qvps]))
    # 11. Extract site name
    if "where" in ds.attrs and isinstance(ds.attrs["where"], dict):
        rname_out = {"site_name": ds.attrs["where"].get("site_name", "Radar")}
    elif "site_name" in ds.attrs:
        rname_out = {"site_name": ds.attrs["site_name"]}
    else:
        rname_out = {"site_name": "Radar"}
    # 12. Provenance
    #TODO: add step_description
    extra = {'step_description': ('')}
    rdqvp = record_provenance(
        rdqvp, step="build_rdqvp", inputs=all_qvp_vars,
        outputs=list(rdqvp.data_vars),
        parameters={
            "n_elevations": len(dss),
            "qvp_kwargs": qvp_kwargs or {},
            "height_resolution_km": height_res,
            "spec_range_km": spec_range,
            "power_param": power_param,
            "elevation_angles_deg": elev_angles,
            "scan_datetimes": scan_datetimes,},
        extra_attrs=extra, module_provenance='towerpy.profs.polprofs.build_rdqvp')
    rdqvp.attrs["profs_type"] = "Range‑Defined Quasi-Vertical Profiles"
    rdqvp.attrs["where"] = rname_out   # Python dict in memory
    return rdqvp
