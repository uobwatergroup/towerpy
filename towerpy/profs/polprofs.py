"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np
import time
# import copy
from datetime import datetime
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

    def __init__(self, radobj):
        if not isinstance(radobj, list):
            self.elev_angle = radobj.elev_angle
            self.file_name = radobj.file_name
            self.scandatetime = radobj.scandatetime
            self.site_name = radobj.site_name

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

        if self.elev_angle < 89:
            raise TowerpyError('The elevation angle must be around 90 deg')
        if self.elev_angle > 89:
            vpdata = {key: values for key, values in rad_vars.items()}
            validgates = 0
            vppol = {key: np.array([np.nanmean(values[0:rad_params['nrays'], i:i+1])
                                    if np.count_nonzero(~np.isnan(values[0:rad_params['nrays'], i:i+1]))>validgates
                                    else np.nan
                                    for i in range(ri, rf)])
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

        dh1 = rad_georef['beam_height [km]'][0]+np.diff(rad_georef['range [m]']/1000, prepend=0) * np.sin(np.deg2rad(rad_params['elev_ang [deg]']))

        dh2 = rad_georef['beam_height [km]'][0]+rad_georef['beam_height [km]'][0] * np.deg2rad(rad_params['beamwidth [deg]']) * (1/np.tan(np.deg2rad(rad_params['elev_ang [deg]'])))

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
                      'V [m/s]': None, 'KDP [deg/km]': None}
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
