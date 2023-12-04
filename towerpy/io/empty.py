"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
from ..georad import georef_rdata as geo


class Rad_scan:
    """
    A Towerpy class to store radar scan data.

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
        georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        params : dict
            Radar technical details.
        vars : dict
            Radar variables.
    """

    def __init__(self, filename, site_name=None):
        self.file_name = filename
        self.site_name = site_name

    def ppi_emptylike(self, nrays=360, ngates=425, elev=0.5,
                      rad_vars='default', scandt=None, tz='Europe/London'):
        r"""
        Create an empty object listing different radar parameters.

        Parameters
        ----------
        nrays : int
            Number of rays on the radar sweep. The default is 360.
        ngates : int, optional
            Number of bins comprising the radar rays. The default is 425.
        elev : float, optional
            Elevation angle of the radar scan. The default is 0.5.
        rad_vars : list, optional
            Polarimetric variables to add to the object. The default are:
                :math:`Z_{H} [dBZ]`, :math:`Z_{DR} [dB]`,
                :math:`\rho_{HV} [-]`, :math:`\Phi_{DP} [deg]`, :math:`V [m/s]`
                The default is 'default'.
        scandt : datetime, optional
            Date and time of the scan. If not provided, datetime.now is used.
            The default is None.
        tz : str
            Key/name of the radar data timezone. The given tz string is then
            retrieved from the ZoneInfo module. Default is 'Europe/London'
        """
        #  add aditional pol vars defined by the user
        if rad_vars == 'default':
            radvars = ['ZH [dBZ]', 'ZDR [dB]', 'PhiDP [deg]', 'rhoHV [-]',
                       'V [m/s]']
        else:
            radvars = rad_vars

        # create dicts to store the empty arrays
        # poldata = {i: np.empty([nrays, ngates],dtype=float) for i in radvars}
        poldata = {i: np.nan for i in radvars}
        parameters = {'nvars': len(radvars), 'ngates': int(ngates),
                      'nrays': int(nrays), 'gateres [m]': np.nan,
                      'rpm': np.nan, 'prf [Hz]': np.nan,
                      'pulselength [ns]': np.nan, 'avsamples': np.nan,
                      'wavelength [cm]': np.nan,
                      'latitude [dd]': np.nan, 'longitude [dd]': np.nan,
                      'altitude [m]': 0, 'easting [km]': np.nan,
                      'northing [km]': np.nan, 'radar constant [dB]': 0,
                      'elev_ang [deg]': elev,
                      'beamwidth [deg]': 1.}
        if scandt is None:
            parameters['datetime'] = dt.datetime.now(tz=ZoneInfo(tz))
            nowdt = [dt.datetime.now(tz=ZoneInfo(tz)).year,
                     dt.datetime.now(tz=ZoneInfo(tz)).month,
                     dt.datetime.now(tz=ZoneInfo(tz)).day,
                     dt.datetime.now(tz=ZoneInfo(tz)).hour,
                     dt.datetime.now(tz=ZoneInfo(tz)).minute,
                     dt.datetime.now(tz=ZoneInfo(tz)).second]
            parameters['datetimearray'] = nowdt
        else:
            parameters['datetime'] = scandt.replace(tzinfo=ZoneInfo(tz))
            udt = list(parameters['datetime'].timetuple())[: -3]
            parameters['datetimearray'] = udt
        self.vars = poldata
        self.params = parameters
        self.elev_angle = parameters['elev_ang [deg]']
        self.scandatetime = parameters['datetime']

    def ppi_emptygeoref(self, azim=None, gate0=0, gateres=250):
        """
        Create a georeferenced grid for the empty object.

        Parameters
        ----------
        azim : array
            Azimuth angles of the scan, in radians. If None, the azimuths are
            computed from 0 to the number of rays.
        gate0 : float
            Distance from the radar to the first bin.
        gateres : int
            Bin resolution of the scan, in metres.
        """
        self.params['gateres [m]'] = gateres
        elev = np.deg2rad(np.full(self.params['nrays'], self.elev_angle))
        azim = np.deg2rad(np.arange(self.params['nrays']))
        rng = np.arange(gate0, self.params['ngates']*gateres,
                        self.params['gateres [m]'], dtype=float)
        bhkm = np.array([geo.height_beamc(ray, rng/1000)
                         for ray in np.rad2deg(elev)])
        bhbkm = np.array([geo.height_beamc(ray
                                           - self.params['beamwidth [deg]']/2,
                                           rng/1000)
                          for ray in np.rad2deg(elev)])
        bhtkm = np.array([geo.height_beamc(ray
                                           + self.params['beamwidth [deg]']/2,
                                           rng/1000)
                          for ray in np.rad2deg(elev)])

        rh, th = np.meshgrid(rng/1000, azim)

        s = np.array([geo.cartesian_distance(ray, rng/1000, bhkm[0])
                      for i, ray in enumerate(np.rad2deg(elev))])
        a = [geo.pol2cart(arcl, azim) for arcl in s.T]
        xgrid = np.array([i[1] for i in a]).T
        ygrid = np.array([i[0] for i in a]).T
        geogrid = {'azim [rad]': azim,
                   'elev [rad]': elev,
                   'range [m]': rng,
                   'rho': rh,
                   'theta': th,
                   'grid_rectx': xgrid,
                   'grid_recty': ygrid}
        self.params['gateres [m]'] = gateres
        # alt = self.params['altitude [m]']
        geogrid['beam_height [km]'] = bhkm
        geogrid['beambottom_height [km]'] = bhbkm
        geogrid['beamtop_height [km]'] = bhtkm
        self.georef = geogrid
