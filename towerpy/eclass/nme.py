"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

from pathlib import Path
import platform
import warnings
import copy
import time
import ctypes as ctp
import numpy as np
import numpy.ctypeslib as npct
from ..datavis import rad_display
from ..base import TowerpyError


warnings.filterwarnings("ignore", category=RuntimeWarning)


class NME_ID:
    r"""
    A class to identify non-meteorlogical echoes within radar data.

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
        echoesID : dict
            Key/values of the clutter classification:
                'meteorological_echoes' = 0

                'noise' = 3

                'clutter' = 5
        nme_classif : dict
            Results of the clutter classification.
        vars : dict
            Radar variables with clutter echoes removed.
    """

    def __init__(self, radobj):
        self.elev_angle = radobj.elev_angle
        self.file_name = radobj.file_name
        self.scandatetime = radobj.scandatetime
        self.site_name = radobj.site_name

    def clutter_id(self, rad_georef, rad_params, rad_vars, path_mfs=None,
                   min_snr=0, binary_class=0, clmap=None, classid=None,
                   data2correct=None, plot_method=False):
        r"""
        Classify between weather and clutter echoes.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth, gates
            and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used to identify the clutter echoes.
        path_mfs : str, optional
            Location of the membership function files.
        min_snr : float, optional
            Reference noise value. The default is 0.
        binary_class : int
            Binary code used for clutter classification:
                :math:`\rho_{HV} = 128`

                :math:`CM = 64`

                :math:`LDR = 32`

                :math:`V = 16`

                :math:`\sigma(\rho_{HV}) = 8`

                :math:`\sigma(\Phi_{DP}) = 4`

                :math:`\sigma(Z_{DR}) = 2`

                :math:`\sigma(Z_{H}) = 1`
            The default is 0.
        clmap : array, optional
            Clutter frequency map in the interval [0-1]. The default is None.
        classid : dict, optional
            Modifies the key/values of the clutter classification results
            (echoesID). The default are the same as in echoesID.
        data2correct : dict, optional
            Variables into which clutter echoes are removed.
            The default is None.
        plot_method : bool, optional
            Plot the clutter classification method. The default is False.

        Notes
        -----
        1. Make sure to define which radar variables are used in the
        classification by setting up the parameter 'binary_class'.

        2. This function uses the shared object 'lnxlibclutterclassifier'
        or the dynamic link library 'w64libclutterclassifier' depending on the
        operating system (OS).

        3. Based on the method described in [1]_

        References
        ----------
        .. [1] Rico-Ramirez, M. A., & Cluckie, I. D. (2008). Classification of
            ground clutter and anomalous propagation using dual-polarization
            weather radar. IEEE Transactions on Geoscience and Remote Sensing,
            46(7), 1892-1904. https://doi.org/10.1109/TGRS.2008.916979

        """
        self.echoesID = {'meteorological_echoes': 0,
                         'noise': 3,
                         'clutter': 5}
        if classid is not None:
            self.echoesID.update(classid)
        if path_mfs is None:
            pathmfs = str.encode(str(Path(__file__).parent.absolute())
                                 + '/mfs_cband/')
        else:
            pathmfs = str.encode(path_mfs)
        array1d = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array2d = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
        if platform.system() == 'Linux':
            libcc = npct.load_library('lnxlibclutterclassifier.so',
                                      Path(__file__).parent.absolute())
        elif platform.system() == 'Windows':
            libcc = ctp.cdll.LoadLibrary(f'{Path(__file__).parent.absolute()}'
                                         + '/w64libclutterclassifier.dll')
        else:
            libcc = None
            raise TowerpyError(f'The {platform.system} OS is not currently'
                               'compatible with this version of Towerpy')
        libcc.clutterclassifier.restype = None
        libcc.clutterclassifier.argtypes = [ctp.c_char_p, ctp.c_int, ctp.c_int,
                                            array2d, array2d, array2d, array2d,
                                            array2d, array2d, array2d, array1d,
                                            array1d, array1d, array1d, array2d]
        param_clc = np.zeros(5)
        param_clc[0] = rad_params['radar constant [dB]']
        param_clc[1] = min_snr
        param_clc[2] = binary_class
        rdatv = copy.deepcopy(rad_vars)
        rdatp = copy.deepcopy(rad_params)
        rdatg = copy.deepcopy(rad_georef)
        clc = np.full(rdatv['ZH [dBZ]'].shape, 0.)
        if 'LDR [dB]' in rad_vars.keys():
            ldr = rdatv['LDR [dB]']
        else:
            ldr = np.full(rdatv['ZH [dBZ]'].shape, 0.)-35
        if clmap is None:
            pc = np.full(rdatv['ZH [dBZ]'].shape, 1.)
        else:
            pc = clmap
        if 'ZDR [dB]' not in rdatv.keys():
            rdatv['ZDR [dB]'] = ldr
        if 'PhiDP [deg]' not in rdatv.keys():
            rdatv['PhiDP [deg]'] = ldr
        if 'rhoHV [-]' not in rdatv.keys():
            rdatv['rhoHV [-]'] = ldr
        np.nan_to_num(rdatv['ZH [dBZ]'], copy=False, nan=-50.)
        # if rdatp['elev_ang [deg]'] < 89:
        libcc.clutterclassifier(pathmfs, rdatp['nrays'],
                                rdatp['ngates'],
                                rdatv['ZH [dBZ]'],
                                rdatv['ZDR [dB]'],
                                rdatv['PhiDP [deg]'],
                                rdatv['rhoHV [-]'],
                                rdatv['V [m/s]'], ldr, pc,
                                rdatg['range [m]'],
                                rdatg['azim [rad]'],
                                rdatg['elev [rad]'],
                                param_clc, clc)
        if classid is not None:
            clc[clc == 0] = self.echoesID['meteorological_echoes']
            clc[clc == 3] = self.echoesID['noise']
            clc[clc == 5] = self.echoesID['clutter']
        # clc[self.snr_class['snrclass'] != 1] = 3  # ###
        # if data2correct is None:
        ccpoldata = {'classif': clc, 'clutter_map': clmap}
        if data2correct is not None:
            data2cc = copy.deepcopy(data2correct)
            # if rdatp['elev_ang [deg]'] > 89:
            #     for key, values in data2cc.items():
            #         values[self.snr_class['snrclass'] != 1] = np.nan
            #     ccpoldata = {'cclass': clc, 'vars': data2cc}
            # else:
            for key, values in data2cc.items():
                values[clc != self.echoesID['meteorological_echoes']] = np.nan
            self.vars = data2cc
            # ccpoldata = {'classif': clc,
            #              'clutter_map': clmap}
        self.nme_classif = ccpoldata

        if plot_method:
            if clmap is not None:
                rad_display.plot_nmeclassif(rad_georef, rad_params, clc, clmap)
            else:
                rad_display.plot_nmeclassif(rad_georef, rad_params, clc)
