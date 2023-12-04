"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import netCDF4 as nc
from scipy import constants as sc
from ..utils import radutilities as rut
from ..georad import georef_rdata as geo
from ..base import TowerpyError


class Rad_scan:
    """A Towerpy class to store radar scans data."""

    def __init__(self, filename):
        """
        Init Rad_scan Class with a filename.

        Parameters
        ----------
        fname : string
        """
        self.file_name = filename

    def ppi_ncasx(self, ncfilename, elevangle=None, bvars=False,
                  tz='Europe/London'):
        """
        Read-in NCAS NetCDF radar data.

        Parameters
        ----------
        ncfilename : dict
            NetCDF radar file.
        elevangle : float, optional
            Elevation angle to work with, in deg. The default is None.

        Returns
        -------
        None.

        """
        ncdat = {k: v[:] for k, v in nc.Dataset(ncfilename).variables.items()}
        print(f'This file contains data taken at the following elevation'
              f" angles: \n {ncdat['fixed_angle']}")
        if elevangle is None:
            elev = ncdat['fixed_angle'][0]
        else:
            if elevangle in ncdat['fixed_angle']:
                elev = elevangle
            else:
                raise TowerpyError('Oops!... Choose one of the following'
                                   f" elevations: \n {ncdat['fixed_angle']}")
        idxelev = np.where(ncdat['elevation'].data == elev)
        nrays = ncdat['azimuth'][idxelev].shape[0]
        ngates = ncdat['range [m]'].shape[0]
        # ======================================================================
        # Reads in the netcdf parameters and store them in a dict
        # ======================================================================
        rparams = {k: v for k, v in ncdat.items()}
        for k, v in rparams.items():
            if v.size == ncdat['azimuth'].size:
                rparams[k] = ncdat[k][idxelev]
        rparams['ngates'] = ngates
        rparams['nrays'] = nrays
        rparams['gateres [m]'] = ncdat['ray_gate_spacing'][idxelev][0]
        rparams['frequency [GHz]'] = rparams.pop('frequency',
                                                 np.nan).data/1000000000
        rparams['rpm'] = rparams.pop('rpm', np.nan)
        rparams['prf [Hz]'] = 1/ncdat['prt'][idxelev]
        rparams['pulselength [ns]'] = rparams.pop('r_calib_pulse_width',
                                                          np.nan)
        rparams['avsamples'] = rparams.pop('n_samples', np.nan)
        rparams['wavelength [cm]'] = (sc.c / rparams['Frequency [GHz]'])/10000000
        rparams['latitude [dd]'] = rparams.pop('latitude', np.nan)
        rparams['longitude [dd]'] = rparams.pop('longitude', np.nan)
        rparams['altitude [m]'] = rparams.pop('altitude', np.nan)
        rparams['easting [km]'] = rparams.pop('easting', np.nan)
        rparams['northing [km]'] = rparams.pop('northing', np.nan)
        rparams['radar constant [dB]'] = rparams.pop('r_calib_radar_constant_h',
                                               np.nan)
        rparams['elev_ang [deg]'] = rparams.pop('elevation',
                                                np.nan)[0]
        date_string = ''.join([str(a, encoding='UTF-8')
                               for a in ncdat['time_coverage_start'].data])
        rparams['datetime'] = dt.datetime.strptime(date_string,
                                                   "%Y-%m-%dT%H:%M:%SZ",
                                                   ).replace(tzinfo=ZoneInfo(tz))
        rparams['datetimearray'] = list(rparams['datetime'].timetuple())[: -3]
        rparams['beamwidth [deg]'] = rparams.pop('radar_beam_width_v',
                                                 np.nan)
        rparams['status_xml'] = str(rparams.pop('status_xml', np.nan),
                                    encoding='UTF-8')
        rvars = {k: rparams.pop(k, np.nan)[idxelev].data
                 for k, v in ncdat.items()
                 if v.shape == (ncdat['azimuth'].size,
                                ncdat['range [m]'].size)}
        for k, v in rvars.items():
            rvars[k][ncdat[k][idxelev].mask == 1] = np.nan
        rvars['ZH [dBZ]'] = rvars.pop('dBZ', np.nan)
        rvars['ZDR [dB]'] = rvars.pop('ZDR', np.nan)
        rvars['rhoHV [-]'] = rvars.pop('rhoHV', np.nan)
        rvars['PhiDP [deg]'] = rvars.pop('PhiDP', np.nan)
        rvars['V [m/s]'] = rvars.pop('V', np.nan)
        rvars['KDP [deg/km]'] = rvars.pop('KDP', np.nan)
        # rvars['W [m/s]'] = rvars.pop('W', np.nan)
        # rvars['SQI [-]'] = rvars.pop('SQI', np.nan)
        if bvars:
            rvars = {k: v for k, v in rvars.items() if '[' in k}
        # ======================================================================
        # Rolls the array, so the azimuth 0 matches the row 0
        # ======================================================================
        azim = np.deg2rad(ncdat['azimuth'][idxelev])
        idx0 = abs(rut.find_nearest(azim, 0) - ncdat['azimuth'].size)
        azim = np.roll(azim, idx0)
        for k, v in rvars.items():
            rvars[k] = np.roll(v, idx0, axis=0)
        elevrad = np.deg2rad(ncdat['elevation'][idxelev])
        gatei = ncdat['ray_start_range'][0]
        rng = np.arange(gatei, rparams['ngates']*rparams['gateres [m]'],
                        rparams['gateres [m]'], dtype=float)
        rh, th = np.meshgrid(rng/1000, azim)
        xgrid, ygrid = geo.pol2cart(rh, np.pi/2-th)
        geogrid = {'azim [rad]': azim,
                   'elev [rad]': elevrad,
                   'range [m]': rng,
                   'rho': rh,
                   'theta': th,
                   'grid_rectx': xgrid,
                   'grid_recty': ygrid}
        geogrid['beam_height [km]'] = np.array([geo.height_beamc(ray, rng/1000)
                                                for ray in np.rad2deg(elevrad)])
        geogrid['beambottom_height [km]'] = np.array([geo.height_beamc(ray-rparams['beamwidth [deg]']/2,
                                                                       rng/1000)
                                                      for ray in np.rad2deg(elevrad)])
        geogrid['beamtop_height [km]'] = np.array([geo.height_beamc(ray+rparams['beamwidth [deg]']/2,
                                                                    rng/1000)
                                                   for ray in np.rad2deg(elevrad)])

        self.georef = geogrid
        self.params = rparams
        self.vars = rvars
