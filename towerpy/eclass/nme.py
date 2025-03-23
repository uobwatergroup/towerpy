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
            Key/values of the ME/NME classification:
                'pcpn' = 0

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

    def lsinterference_filter(self, rad_georef, rad_params, rad_vars,
                              rhv_min=0.3, classid=None, data2correct=None,
                              plot_method=False):
        """
        Filter linear signatures and speckles.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth,
            gates and beam height, amongst others.
        rad_params : dict
            Radar technical details.
        rad_vars : dict
            Radar variables used to identify the LS and speckles.
        rhv_min : float, optional
            Minimal threshold in rhoHV [-] used to discard
            non-meteorological scatterers. The default is 0.3
        classid : dict, optional
            Modifies the key/values of the LS/Despeckling results
            (echoesID). The default are the same as in echoesID
            (see class definition).
        data2correct :  dict, optional
            Variables into which LS ans speckles are removed.
            The default is None.
        plot_method : TYPE, optional
            Plot the LS/speckles classification method.
            The default is False.

        Notes
        -----
        1. Radar variables should already be (at least) filtered for
        noise to ensure accurate and reliable results.

        """
        self.echoesID = {'pcpn': 0,
                         'noise': 3,
                         'clutter': 5}
        if classid is not None:
            self.echoesID.update(classid)

        window = (3, 3)
        mode = 'constant'
        arr_rhohv = rad_vars['rhoHV [-]'].copy()
        constant_values = np.nan
        # Create a padded array
        if mode == 'edge':
            apad = np.pad(arr_rhohv, ((0, 0), (window[1]//2, window[1]//2)),
                          mode='edge')
        elif mode == 'constant':
            apad = np.pad(arr_rhohv, ((0, 0), (window[1]//2, window[1]//2)),
                          mode='constant', constant_values=(constant_values))
        if window[0] > 1:
            apad = np.pad(apad, ((window[0]//2, window[0]//2), (0, 0)),
                          mode='wrap')
        # Check that all sorrounding values of pixel are nan to remove speckles
        spckl1 = np.array([[np.nan if ~np.isnan(vbin)
                            and np.isnan(apad[nray-1][nbin-1])
                            and np.isnan(apad[nray-1][nbin])
                            and np.isnan(apad[nray-1][nbin+1])
                            and np.isnan(apad[nray][nbin-1])
                            and np.isnan(apad[nray][nbin+1])
                            and np.isnan(apad[nray+1][nbin-1])
                            and np.isnan(apad[nray+1][nbin])
                            and np.isnan(apad[nray+1][nbin+1])
                            else 1
                            for nbin, vbin in enumerate(apad[nray])
                            if nbin != 0 and nbin != apad.shape[1]-1]
                           for nray in range(apad.shape[0])
                           if nray != 0 and nray != apad.shape[0]-1])
        spckl1[:, 0] = np.nan
        # Filter using rhohv threshold.
        spckl1[rad_vars['rhoHV [-]'] <= rhv_min] = np.nan
        # Detect linear signatures.
        spckl2 = np.array([[np.nan if ~np.isnan(vbin)
                            and np.isnan(apad[nray-1][nbin])
                            and np.isnan(apad[nray+1][nbin]) else 1
                            for nbin, vbin in enumerate(apad[nray])
                            if nbin != 0 and nbin != apad.shape[1]-1]
                           for nray in range(apad.shape[0])
                           if nray != 0 and nray != apad.shape[0]-1])
        spckl1[:, 0] = 1
        # Classifies the pixels according to echoesID
        fclass = np.where(np.isnan(rad_vars['ZH [dBZ]']), 3, 0)
        fclass2 = np.where(np.isnan(spckl1 * spckl2), 5, 0)

        fclass = np.where(fclass2 == 5, 5, fclass)

        if classid is not None:
            fclass[fclass == 0] = self.echoesID['pcpn']
            fclass[fclass == 3] = self.echoesID['noise']
            fclass[fclass == 5] = self.echoesID['clutter']

        lsc_data = {'classif': fclass}

        if data2correct is not None:
            data2cc = copy.deepcopy(data2correct)
            for key, values in data2cc.items():
                values[fclass != self.echoesID['pcpn']] = np.nan
            self.vars = data2cc
        self.ls_dsp_class = lsc_data

        if plot_method:
            rad_display.plot_ppi(rad_georef, rad_params, lsc_data,
                                 cbticks=self.echoesID,
                                 ucmap='tpylc_div_yw_gy_bu')

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
            (echoesID). The default are the same as in echoesID
            (see class definition).
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

        Examples
        --------
        >>> rnme = tp.eclass.nme.NME_ID(rdata)
        >>> rnme.clutter_id(rdata.georef, rdata.params, rsnr.vars,
                            binary_class=159, min_snr=rsnr.min_snr)

        binary_class = 159 -> (128+16+8+4+2+1) i.e.
        :math:`\rho_{HV} + V + \sigma(\rho_{HV}) + \sigma(\Phi_{DP})
        + \sigma(Z_{DR}) + \sigma(Z_{H})`

        """
        self.echoesID = {'pcpn': 0,
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
            clc[clc == 0] = self.echoesID['pcpn']
            clc[clc == 3] = self.echoesID['noise']
            clc[clc == 5] = self.echoesID['clutter']
        ccpoldata = {'classif': clc, 'clutter_map': clmap}
        if data2correct is not None:
            data2cc = copy.deepcopy(data2correct)
            for key, values in data2cc.items():
                values[clc != self.echoesID['pcpn']] = np.nan
            self.vars = data2cc
        self.nme_classif = ccpoldata

        if plot_method:
            if clmap is not None:
                rad_display.plot_nmeclassif(rad_georef, rad_params, clc, clmap)
            else:
                rad_display.plot_nmeclassif(rad_georef, rad_params, clc)
