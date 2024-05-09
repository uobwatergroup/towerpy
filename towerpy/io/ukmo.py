"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import ctypes as ctp
import platform
from pathlib import Path
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np
import numpy.ctypeslib as npct
from ..georad import georef_rdata as geo
from ..base import TowerpyError


# TODO: add xarray compatibility
class Rad_scan:
    """
    A Towerpy class to store radar data.

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
            Radar variables retrieved from the file.
    """

    def __init__(self, filename, radar_site):
        self.file_name = filename
        self.site_name = radar_site

    def ppi_ukmoraw(self, get_polvar='all', exclude_vars=None,
                    tz='Europe/London'):
        """
        Read raw polarimetric variables from current UKMO PPI binary files.

        Parameters
        ----------
        get_polvar : str, optional
            Define variables to read by the function. The default is 'all'.
        exclude_vars : list, optional
            Define variables to discard. The default is None.
        tz : str
            Key/name of the radar data timezone. The given tz value is then
            retrieved from the ZoneInfo module. Default is 'Europe/London'

        Notes
        -----
        1. This function uses the shared object 'lnxlibreadpolarradardata'
        or the dynamic link library 'w64libreadpolarradardata' depending on the
        operating system (OS).

        Examples
        --------
        >>> rdata = io.ukmo.Rad_scan('metoffice-c-band-rain-radar_chenies_201804090938_raw-dual-polar-augldr-lp-el0.dat')
        >>> rdata.ukmo_rawpol()
        """
        # Define Ctypes parameters
        array1d = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array2d = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')

        if platform.system() == 'Linux':
            librp = npct.load_library('lnxlibreadpolarradardata.so',
                                      Path(__file__).parent.absolute())
        elif platform.system() == 'Windows':
            librp = ctp.cdll.LoadLibrary(f'{Path(__file__).parent.absolute()}'
                                         + '/w64libreadpolarradardata.dll')
        else:
            librp = None
            raise TowerpyError(f'Oops!... The {platform.system} OS '
                               'is not currently '
                               'compatible with this version of Towerpy')
        librp.readpolarradardata.restype = None
        librp.readpolarradardata.argtypes = [ctp.c_char_p, array1d, array2d,
                                             array1d, array1d, array1d,
                                             array1d, array1d, ctp.c_char_p]
        fname = str.encode(self.file_name)

        # Create empty arrays to read nrays/ngates
        emptyarr1 = [np.empty(20) for i in range(8)]
        emptyarr1[1] = np.empty((1, 1))
        emptyarr1[7] = bytes(16)
        librp.readpolarradardata(ctp.c_char_p(fname), emptyarr1[0],
                                 emptyarr1[1], emptyarr1[2], emptyarr1[3],
                                 emptyarr1[4], emptyarr1[5], emptyarr1[6],
                                 ctp.c_char_p(emptyarr1[7]))
        nrays, ngates = int(emptyarr1[6][2]), int(emptyarr1[6][1])

        # read all radar variables
        if get_polvar == 'all' or get_polvar is None:
            emptyarr2 = [np.empty(20) for i in range(8)]
            emptyarr2[0] = np.array([0, nrays, ngates], dtype=float)
            emptyarr2[1] = np.empty((nrays, ngates))
            emptyarr2[2] = np.empty((nrays))
            emptyarr2[3] = np.empty((nrays))
            emptyarr2[4] = np.empty((ngates))
            emptyarr2[5] = np.empty(6)
            emptyarr2[7] = bytes(16)
            librp.readpolarradardata(ctp.c_char_p(fname), emptyarr2[0],
                                     emptyarr2[1], emptyarr2[2], emptyarr2[3],
                                     emptyarr2[4], emptyarr2[5], emptyarr2[6],
                                     ctp.c_char_p(emptyarr2[7]))
            nvar = int(emptyarr2[6][0])
            emptyarr = [np.empty(20) for i in range(8)]
            emptyarr[0] = np.array([0, nrays, ngates], dtype=float)
            emptyarr[1] = np.empty((nrays, ngates))
            emptyarr[2] = np.empty((nrays))
            emptyarr[3] = np.empty((nrays))
            emptyarr[4] = np.empty((ngates))
            emptyarr[5] = np.empty(6)
            emptyarr[7] = bytes(16)
            vardat = {}
            varnam = {}
            dicaxs = {}
            for i in range(nvar):
                emptyarr[0][0] = i
                librp.readpolarradardata(ctp.c_char_p(fname), emptyarr[0],
                                         emptyarr[1], emptyarr[2], emptyarr[3],
                                         emptyarr[4], emptyarr[5], emptyarr[6],
                                         ctp.c_char_p(emptyarr[7]))
                varname = emptyarr[7].decode()
                varnam[i] = varname[0:varname.find(']')+1]
                vardat[i] = np.array(emptyarr[1])
                dicaxs[i] = [emptyarr[6][4], emptyarr[6][5]]
                if emptyarr[0][0] == 0:
                    outpar = np.array(emptyarr[6])
        else:
            # read rad variable defined by user
            emptyarr = [np.empty(20) for i in range(8)]
            emptyarr[0] = np.array([0, nrays, ngates], dtype=float)
            emptyarr[1] = np.empty((nrays, ngates))
            emptyarr[2] = np.empty((nrays))
            emptyarr[3] = np.empty((nrays))
            emptyarr[4] = np.empty((ngates))
            emptyarr[5] = np.empty(6)
            emptyarr[7] = bytes(16)

            vardat = {}
            varnam = {}
            dicaxs = {}
            if get_polvar == 'ZH [dBZ]':
                nvar = 0
            elif get_polvar == 'ZDR [dB]':
                nvar = 1
            elif get_polvar == 'PhiDP [deg]':
                nvar = 2
            elif get_polvar == 'rhoHV [-]':
                nvar = 3
            elif get_polvar == 'V [m/s]':
                nvar = 4
            elif get_polvar == 'W [m/s]':
                nvar = 5
            elif get_polvar == 'CI [dB]':
                nvar = 6
            elif get_polvar == 'SQI [-]':
                nvar = 7
            else:
                raise TowerpyError(f'Oops!... The variable {nvar}'
                                   'cannot be retreived')
            emptyarr[0][0] = nvar
            librp.readpolarradardata(ctp.c_char_p(fname), emptyarr[0],
                                     emptyarr[1], emptyarr[2], emptyarr[3],
                                     emptyarr[4], emptyarr[5], emptyarr[6],
                                     ctp.c_char_p(emptyarr[7]))
            varname = emptyarr[7].decode()
            varnam[nvar] = varname[0:varname.find(']')+1]
            vardat[nvar] = np.array(emptyarr[1])
            dicaxs[nvar] = [emptyarr[6][4], emptyarr[6][5]]
            outpar = np.array(emptyarr[6])
        poldata = {varnam[i]: j for (i, j) in vardat.items()}
        if any(v.startswith('Zh') for k, v in varnam.items()):
            poldata['ZH [dBZ]'] = poldata.pop('Zh [dBZ]')
        if any(v.startswith('Zdr') for k, v in varnam.items()):
            poldata['ZDR [dB]'] = poldata.pop('Zdr [dB ]')
        if any(v.startswith('RhoHV') for k, v in varnam.items()):
            poldata['rhoHV [-]'] = poldata.pop('RhoHV [   ]')
        if any(v.startswith('LDR') for k, v in varnam.items()):
            poldata['LDR [dB]'] = poldata.pop('LDR [dB ]')
        if any(v.startswith('Phi') for k, v in varnam.items()):
            poldata['PhiDP [deg]'] = poldata.pop('Phidp [deg]')
        if any(not v for k, v in varnam.items()):
            poldata['Absphase_V [ ]'] = poldata.pop('')
        poldata = dict(sorted(poldata.items(), reverse=True))
        if exclude_vars is not None:
            evars = exclude_vars
            poldata = {k: val for k, val in poldata.items() if k not in evars}

        # Create dict to store radparameters
        dttime = dt.datetime(int(emptyarr[5][0]), int(emptyarr[5][1]),
                             int(emptyarr[5][2]), int(emptyarr[5][3]),
                             int(emptyarr[5][4]), int(emptyarr[5][5]),
                             tzinfo=ZoneInfo(tz))
        outpar[17] = np.rad2deg(emptyarr[3][0])
        parameters = {'nvars': int(outpar[0]),
                      'ngates': int(outpar[1]),
                      'nrays': int(outpar[2]),
                      'gateres [m]': outpar[3],
                      'rpm': outpar[6],
                      'prf [Hz]': outpar[7],
                      'pulselength [ns]': outpar[8],
                      'avsamples': outpar[9],
                      'wavelength [cm]': outpar[10]*100,
                      'latitude [dd]': outpar[11],
                      'longitude [dd]': outpar[12],
                      'altitude [m]': outpar[13],
                      'easting [km]': outpar[14]/1000,
                      'northing [km]': outpar[15]/1000,
                      'radar constant [dB]': outpar[16],
                      'elev_ang [deg]': outpar[17],
                      'datetime': dttime,
                      'datetimearray': emptyarr[5]}
        if 'metoffice' in self.file_name:
            parameters['beamwidth [deg]'] = 1.
        parameters['site_name'] = self.site_name
        parameters['range_start [m]'] = emptyarr[4][0]

        # Create dict to store geospatial data
        rh, th = np.meshgrid(emptyarr[4]/1000, emptyarr[2])
        geogrid = {'range [m]': emptyarr[4], 'elev [rad]': emptyarr[3],
                   'azim [rad]': emptyarr[2], 'rho': rh, 'theta': th}

        self.elev_angle = parameters['elev_ang [deg]']
        self.scandatetime = parameters['datetime']
        self.georef = geogrid
        self.params = parameters
        self.vars = poldata

    def ppi_ukmogeoref(self):
        """Create georeferenced data from the UKMO PPI scan."""
        # xgrid, ygrid = geo.pol2cart(self.georef['rho'],
        #                             np.pi/2-self.georef['theta'])
        bhkm = np.array([geo.height_beamc(ray, self.georef['range [m]']/1000)
                         for ray in np.rad2deg(self.georef['elev [rad]'])])
        bhbkm = np.array([geo.height_beamc(ray
                                           - self.params['beamwidth [deg]']/2,
                                           self.georef['range [m]']/1000)
                          for ray in np.rad2deg(self.georef['elev [rad]'])])
        bhtkm = np.array([geo.height_beamc(ray
                                           + self.params['beamwidth [deg]']/2,
                                           self.georef['range [m]']/1000)
                          for ray in np.rad2deg(self.georef['elev [rad]'])])
        s = np.array([geo.cartesian_distance(ray,
                                             self.georef['range [m]']/1000,
                                             bhkm[0])
                      for i,
                      ray in enumerate(np.rad2deg(self.georef['elev [rad]']))])
        a = [geo.pol2cart(arcl, self.georef['azim [rad]']) for arcl in s.T]

        grid_rectx = np.array([i[1] for i in a]).T
        grid_recty = np.array([i[0] for i in a]).T

        grid_osgbx = (grid_rectx + self.params['easting [km]'])*1000
        grid_osgby = (grid_recty + self.params['northing [km]'])*1000

        self.georef['grid_rectx'] = grid_rectx
        self.georef['grid_recty'] = grid_recty
        self.georef['beam_height [km]'] = bhkm
        self.georef['beambottom_height [km]'] = bhbkm
        self.georef['beamtop_height [km]'] = bhtkm
        self.georef['grid_osgbx'] = grid_osgbx
        self.georef['grid_osgby'] = grid_osgby
