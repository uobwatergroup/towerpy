"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import ctypes as ctp
import datetime as dt
from pathlib import Path
import platform
import re
from zoneinfo import ZoneInfo
import numpy as np
import numpy.ctypeslib as npct
from scipy import constants as sc
import xarray as xr
import xradar as xrd
from ..io import modeltp as mdtp
from ..georad import georef_rdata as geo
from ..utils.radutilities import scan_midtime


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
            raise UserWarning(f'Oops!... The {platform.system} OS '
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
            elif get_polvar == 'LDR [dB]':
                nvar = 1
            else:
                raise UserWarning(f'Oops!... The variable {nvar}'
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
        if any(v.startswith('Phi') for k, v in varnam.items()):
            poldata['PhiDP [deg]'] = poldata.pop('Phidp [deg]')
        if any(v.startswith('RhoHV') for k, v in varnam.items()):
            poldata['rhoHV [-]'] = poldata.pop('RhoHV [   ]')
        if any(v.startswith('LDR') for k, v in varnam.items()):
            poldata['LDR [dB]'] = poldata.pop('LDR [dB ]')        
        if any(not v for k, v in varnam.items()):
            poldata['Absphase_V [ ]'] = poldata.pop('')
        if any(v.startswith('CI [-') for k, v in varnam.items()):
            poldata['CI [dB]'] = poldata.pop('CI [- ]')
        if any(v.startswith('V') for k, v in varnam.items()):
            poldata['V [m/s]'] = poldata.pop('V [m/s]')
        # poldata = dict(sorted(poldata.items(), reverse=True))
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
        # rh, th = np.meshgrid(emptyarr[4]/1000, emptyarr[2])
        geogrid = {'range [m]': emptyarr[4], 'elev [rad]': emptyarr[3],
                   'azim [rad]': emptyarr[2]}

        self.elev_angle = parameters['elev_ang [deg]']
        self.scandatetime = parameters['datetime']
        self.georef = geogrid
        self.params = parameters
        self.vars = poldata

    def ppi_ukmogeoref(self):
        """
        Create georeferenced data from the UKMO PPI scan.
        
        Notes
        -----
        1. This method wraps :func:`geo.ppi_georef` and adds OSGB coordinates based on radar easting and northing offsets.
        2. OSGB coordinates are computed by shifting the relative Cartesian grid ('grid_rectx', 'grid_recty') by the radar easting/northing (in kilometres) and converting to metres.
        """
        geogrid, _ = geo.ppi_georef(self.params, georef=self.georef)
        geogrid['grid_osgbx'] = (
            geogrid['grid_rectx'] + self.params['easting [km]'])*1000
        geogrid['grid_osgby'] = (
            geogrid['grid_recty'] + self.params['northing [km]'])*1000
        self.georef.update(geogrid)


# =============================================================================
# %% xarray implementation
# =============================================================================

def _parse_ukmo_filename(filename: str):
    name = Path(filename).stem  # remove .dat
    parts = name.split("_")[-1].split("-")

    pulse = None
    pol_mode = None
    elevation = None

    for p in parts:
        if p in ("lp", "sp"):
            pulse = "long_pulse" if p == "lp" else "short_pulse"
        elif p in ("augzdr", "augldr"):
            pol_mode = ("simultaneous-dual" if p == "augzdr"
                        else "alternate-dual")
        elif p.startswith("el"):
            m = re.match(r"el(\d+)", p)
            if m:
                elevation = int(m.group(1))
    return pulse, pol_mode, elevation


def ppi_ukmoraw(file_name, site_name, get_polvar='all', exclude_vars=None,
                tz='UTC'):
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
        retrieved from the ZoneInfo module. Default is 'UTC'

    Notes
    -----
    1. This function uses the shared object 'lnxlibreadpolarradardata'
    or the dynamic link library 'w64libreadpolarradardata' depending on the
    operating system (OS).

    Examples
    --------
    >>> fname = 'metoffice-c-band-rain-radar_chenies_201804090938_raw-dual-polar-augldr-lp-el0.dat'
    >>> rdata = rdata.ukmo_rawpol(fname)
    """
    # Define Ctypes parameters
    array1d = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
    array2d = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
    # Set shared library
    if platform.system() == 'Linux':
        librp = npct.load_library('lnxlibreadpolarradardata.so',
                                  Path(__file__).parent.absolute())
        # librp = npct.load_library('lnxlibreadpolarradardata.so', Path.cwd())
    elif platform.system() == 'Windows':
        librp = ctp.cdll.LoadLibrary(f'{Path(__file__).parent.absolute()}'
                                     + '/w64libreadpolarradardata.dll')
    else:
        librp = None
        raise ValueError(f'The {platform.system} OS is not compatible'
                         ' with this version of Towerpy')
    librp.readpolarradardata.restype = None
    librp.readpolarradardata.argtypes = [ctp.c_char_p, array1d, array2d,
                                         array1d, array1d, array1d,
                                         array1d, array1d, ctp.c_char_p]
    # Encode file name
    file_name = str(file_name)
    ukmo_modes = _parse_ukmo_filename(file_name)
    fname = str.encode(file_name)
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
        if get_polvar == 'ZH [dBZ]': nvar = 0
        elif get_polvar == 'ZDR [dB]': nvar = 1
        elif get_polvar == 'PhiDP [deg]': nvar = 2
        elif get_polvar == 'rhoHV [-]': nvar = 3
        elif get_polvar == 'V [m/s]': nvar = 4
        elif get_polvar == 'W [m/s]': nvar = 5
        elif get_polvar == 'CI [dB]': nvar = 6
        elif get_polvar == 'SQI [-]': nvar = 7
        elif get_polvar == 'LDR [dB]': nvar = 1
        else:
            raise ValueError(f'Oops!... The variable {nvar}'
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
        # poldata['Absphase_V [ ]'] = poldata.pop('')
        poldata.pop('')
    if any(v.startswith('CI [-') for k, v in varnam.items()):
        poldata['CI [dB]'] = poldata.pop('CI [- ]')
    if any(v.startswith('V') for k, v in varnam.items()):
        poldata['V [m/s]'] = poldata.pop('V [m/s]')
    # Detect elevation
    is_birdbath = np.nanmean(np.rad2deg(emptyarr[3])) > 85.0
    # print(is_birdbath)
    sweep_vars_mapping = {'DBTH': 'ZH [dBZ]', 'UZDR': 'ZDR [dB]',
                          'UPHIDP': 'PhiDP [deg]', 'URHOHV': 'rhoHV [-]',
                          'CI':'CI [dB]' , 'SQIH': 'SQI [-]',
                          'LDR': 'LDR [dB]'}
    # Doppler velocity mapping depends on elevation
    if is_birdbath:
        sweep_vars_mapping['VRADV'] = 'V [m/s]'
        sweep_vars_mapping['UWRADV'] = 'W [m/s]'
    else:
        sweep_vars_mapping['VRADH'] = 'V [m/s]'
        sweep_vars_mapping['UWRADH'] = 'W [m/s]'
    if exclude_vars is not None:
        # evars = exclude_vars
        evars = [val for k, val in sweep_vars_mapping.items()
                 if k not in exclude_vars]
        poldata = {k: val for k, val in poldata.items() if k in evars}

    # Create dict to store radparameters
    dttime = dt.datetime(int(emptyarr[5][0]), int(emptyarr[5][1]),
                         int(emptyarr[5][2]), int(emptyarr[5][3]),
                         int(emptyarr[5][4]), int(emptyarr[5][5]),
                         tzinfo=ZoneInfo(tz))
    outpar[17] = np.rad2deg(emptyarr[3][0])
    # Use xradar to set attributes
    azimuth = np.rad2deg(emptyarr[2])
    elevation = np.rad2deg(emptyarr[3])
    elevation_attrs = xrd.model.get_elevation_attrs()
    rng = emptyarr[4]
    azimuth_attrs = xrd.model.get_azimuth_attrs(azimuth)
    range_attrs = xrd.model.get_range_attrs(rng)
    # Handle time coordinate
    time_coord = np.full(len(azimuth), np.datetime64(dttime),
                         dtype='datetime64[ns]')
    # Initialise an xarray dataset
    sweep = xr.Dataset(coords=dict(azimuth=(["azimuth"], azimuth, azimuth_attrs), 
                       elevation=(["azimuth"], elevation, elevation_attrs),
                       range=(["range"], rng, range_attrs)))
    sweep = sweep.assign_coords(sweep_mode="azimuth_surveillance",
                                sweep_number=int(ukmo_modes[2]),
                                prt_mode='not_set',
                                follow_mode='not_set',
                                sweep_fixed_angle=np.nanmedian(elevation),
                                longitude=outpar[12], latitude=outpar[11],
                                easting=outpar[14], northing=outpar[15],
                                altitude=outpar[13],
                                time=("azimuth", time_coord))
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    for k1, v1 in sweep_vars_mapping.items():
        if v1 in poldata.keys():
            sweep = sweep.assign({k1: (['azimuth', 'range'],
                                       poldata[v1])})
        if k1 in sweep_vars_attrs_f and v1 in poldata.keys():
            sweep[k1].attrs = sweep_vars_attrs_f[k1]
    sweep["longitude"].attrs = xrd.model.get_longitude_attrs()
    sweep["latitude"].attrs = xrd.model.get_latitude_attrs()
    sweep["altitude"].attrs = xrd.model.get_altitude_attrs()
    sweep["easting"].attrs = {
        "units": "m",
        "long_name": "Ordnance Survey National Grid (OSGB) easting"}
    sweep["northing"].attrs = {
        "units": "m",
        "long_name": "Ordnance Survey National Grid (OSGB) northing"}
    sweep.attrs["crs"] = "EPSG:27700"  # OSGB36 example
    # =============================================================================
    # Set parameters
    # =============================================================================
    rswp_metadata = {
        "file_name": file_name,
        "how":{'RXlossH': np.nan,
               'RXlossV': np.nan,
               'antgainH': np.nan,
               'antgainV': np.nan,
               'beamwH': np.nan,
               'beamwV': np.nan,
               'beamwidth': 1.,
               'beamwidth_units': 'deg',
               'endepochs': np.nan,
               'extensions': np.nan,
               'poltype': ukmo_modes[1],
               'scan_count': np.nan,
               'simulated': 'False',
               'software': 'Cyclops',
               'startepochs': np.nan,
               'sw_version': 'CEDA',
               'system': np.nan,
               'task': np.nan,
               'wavelength': outpar[10]*100,
               'wavelength_units': 'cm',
               'frequency': np.nan,
               'frequency_units': 'GHz',
               },
        "how_monitor": {'zdr-offset_90deg_pw0': np.nan,
                        'zdr-offset_90deg_pw2': np.nan},
        "how_radar_system": {'dBZ0_H_pw0': np.nan,
                             'dBZ0_H_pw2': np.nan,
                             'dBZ0_V_pw0': np.nan,
                             'dBZ0_V_pw2': np.nan,
                             'noise_H_pw0': np.nan,
                             'noise_H_pw2': np.nan,
                             'noise_V_pw0': np.nan,
                             'noise_V_pw2': np.nan,
                             'phidp-offset_system': np.nan,
                             'zdr-offset_system': np.nan},
        "what": {'date': dttime.strftime('%Y%m%d'),
                 'object': 'SCAN',
                 'source': f'NOD:gb{site_name.lower()}',
                 'time': dttime.strftime('%H%M%S'),
                 'version': 'CEDA',
                 'nvars': int(outpar[0]),
                 'scan_type': 'volume_scan',
                 },
        "where": {'height': outpar[13],
                  'lat': outpar[11],
                  'lon': outpar[12],
                  'site_name': site_name,
                  },
        "dataset1_how": {'NI': np.nan,
                         'afc_status': np.nan,
                         'angle_step': np.nan,
                         'bpwr': np.nan,
                         'highprf': np.nan,
                         'lowprf': np.nan,
                         'polmode': ukmo_modes[1],
                         'prf': outpar[7],
                         "prf_units": "Hz",
                         'pulse': ukmo_modes[0],
                         # 'pulsewidth': np.nan,
                         'pulselength': outpar[8],
                         "pulselength_units": "ns",
                         'radconstH': outpar[16],
                         'radconstV': np.nan,
                         "radconst_units": "dB",
                         'range': np.nan,
                         'rpm': outpar[6],
                         'scan_index': np.nan,
                         'startazA': np.nan,
                         'startazT': np.nan,
                         'startelA': np.nan,
                         'stopazA': np.nan,
                         'stopazT': np.nan,
                         'stopelA': np.nan,
                         'task': 'SCAN',
                         'wavelength': outpar[10]*100,
                         'wavelength_units': 'cm',
                         'avsamples': outpar[9],
                         },
        "dataset1_where": {'a1gate': np.nan,
                           'elangle': outpar[17],
                           'nbins': int(outpar[1]),
                           'nrays': int(outpar[2]),
                           'rscale': np.nan,
                           'rstart': np.nan},
        "dataset1_what": {'enddate': np.nan,
                          'endtime': np.nan,
                          'product': 'SCAN',
                          'startdate': np.nan,
                          'starttime': np.nan,
                          # 'sweep_avrg_datetime64': np.nan,
                          # 'sweep_avrg_datetime': np.nan
                          }
                     }
    rswp_metadata['how']['frequency'] = (sc.c / (rswp_metadata['how']['wavelength'] * 1e-2) / 1e9)
    scandt_av, scandt_avpy = scan_midtime(sweep['time'].values)
    ts_ns = int(scandt_av.astype("datetime64[ns]").astype("int"))
    rswp_metadata['what']['sweep_avrg_datetime_unix_ns'] = ts_ns
    rswp_metadata['what']['sweep_avrg_datetime_iso'] = np.datetime_as_string(
        np.datetime64(ts_ns, "ns"), unit="ms")
    rswp_metadata['what']['sweep_avrg_datetime_unit'] = "ns since 1970-01-01"
    sweep.attrs = rswp_metadata
    return sweep