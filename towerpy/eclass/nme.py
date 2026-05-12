"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

from pathlib import Path
import platform
import warnings
import copy
import ctypes as ctp
from scipy.ndimage import convolve
import xarray as xr
import numpy as np
import numpy.ctypeslib as npct
from ..io import modeltp as mdtp
from ..datavis import rad_display
# from ..base import TowerpyError
from ..utils.radutilities import get_attrval, safe_assign_variable
from ..utils.radutilities import record_provenance, apply_correction_chain
from ..utils.unit_conversion import convert


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

    def __init__(self, radobj=None):
        self.elev_angle = getattr(radobj, 'elev_angle',
                                  None) if radobj else None
        self.file_name = getattr(radobj, 'file_name',
                                 None) if radobj else None
        self.scandatetime = getattr(radobj, 'scandatetime',
                                    None) if radobj else None
        self.site_name = getattr(radobj, 'site_name',
                                 None) if radobj else None

    def lsinterference_filter(self, rad_georef, rad_vars, data2correct=None,
                              rhv_min=0.3, classid=None, plot_method=False):
        """
        Filter linear signatures and speckles.

        Parameters
        ----------
        rad_georef : dict
            Georeferenced data containing descriptors of the azimuth,
            gates and beam height, amongst others.
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
        plot_method : bool, optional
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
                           if nray != 0 and nray != apad.shape[0]-1],
                          dtype=np.float64)
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
                           if nray != 0 and nray != apad.shape[0]-1],
                          dtype=np.float64)
        spckl1[:, 0] = 1
        # Classifies the pixels according to echoesID
        fclass = np.where(np.isnan(rad_vars['ZH [dBZ]']), 3., 0.)
        fclass2 = np.where(np.isnan(spckl1 * spckl2), 5., 0.)

        fclass = np.where(fclass2 == 5., 5., fclass)
        fclass[:, :5] = 0

        if classid is not None:
            fclass[fclass == 0] = self.echoesID['pcpn']
            fclass[fclass == 3] = self.echoesID['noise']
            fclass[fclass == 5] = self.echoesID['clutter']

        lsc_data = {'classif [EC]': fclass}

        if data2correct is not None:
            data2cc = copy.deepcopy(data2correct)
            for key, values in data2cc.items():
                values[fclass != self.echoesID['pcpn']] = np.nan
            self.vars = data2cc
        self.ls_dsp_class = lsc_data

        if plot_method:
            rad_params = {}
            if self.elev_angle:
                rad_params['elev_ang [deg]'] = self.elev_angle
            else:
                rad_params['elev_ang [deg]'] = 'surveillance scan'
            if self.scandatetime:
                rad_params['datetime'] = self.scandatetime
            else:
                rad_params['datetime'] = None

            rad_display.plot_ppi(rad_georef, rad_params, lsc_data,
                                 cbticks=self.echoesID,
                                 ucmap='tpylc_div_yw_gy_bu')

    def clutter_id(self, rad_georef, rad_params, rad_vars, path_mfs=None,
                   min_snr=0, binary_class=255, clmap=None, classid=None,
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
            The default is 255, i.e. all the variables are used.
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
            raise ValueError(f'The {platform.system} OS is not currently'
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
        ccpoldata = {'classif [EC]': clc, 'clutter_map': clmap}
        if data2correct is not None:
            data2cc = copy.deepcopy(data2correct)
            for key, values in data2cc.items():
                values[clc != self.echoesID['pcpn']] = np.nan
            self.vars = data2cc
        self.nme_classif = ccpoldata

        if plot_method:
            if clmap is not None:
                rad_display.plot_nmeclassif(rad_georef, rad_params, clc,
                                            self.echoesID, clmap)
            else:
                rad_display.plot_nmeclassif(rad_georef, rad_params, clc,
                                            self.echoesID)

# =============================================================================
# %% xarray implementation
# =============================================================================


def _conv2_8(mask_nan):
    """Return 8‑neighbour 3×3 convolution using constant‑zero padding."""
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=int)
    return convolve(mask_nan, kernel, mode="constant", cval=0)


def _conv1_vert(mask):
    """Return ray-wise 3‑point convolution using constant‑zero padding."""
    kernel = np.array([1, 0, 1], dtype=int)
    # convolve along axis 0 (azimuth)
    return convolve(mask, kernel[:, None], mode="constant", cval=0)


def lsinterference_filter(ds, inp_names=None, rhv_min=0.3, classid=None,
                          mask=None, replace_vars=False):
    """
    Detect and filter linear signatures (LS) related to interference in PPIs.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing polarimetric variables (DBZ, RHOHV) and polar
        coordinates (range, azimuth). Variables should already be
        noise-filtered.
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{'azi': 'azimuth', 'rng': 'range', 'DBZ': 'DBZH', 'RHOHV': 'RHOHV'}``.
    rhv_min : float, default 0.3
        Minimum :math:`\rho_{HV}` threshold used to identify
        non-meteorological scatterers and linear interference.
    classid : dict, optional
        Mapping of LS/despeckling class identifiers. If ``None``, defaults
        from the internal ``echoesID`` class are used.    
    mask : bool, list of str, dict of str to str, or None, optional
        Controls which variables receive LS/speckle masking.
    
        * ``None`` or ``False``: classification only; no masking applied.
        * ``True``: mask all 2‑D data variables in the dataset.
        * list of str: mask only the listed variables.
        * dict: map input variable names to explicit output names.
    replace_vars : bool, default False
        If True, overwrite selected variables.
        If False, masked variables receive a ``_QC`` suffix unless explicit
        names are provided via ``mask`` (dict form).

    Returns
    -------
    xarray.Dataset
        Dataset containing:
        
        - LS/speckle classification fields,
        - masked variables (if ``mask`` is True, a list, or a dict)
        - updated provenance metadata.

    Notes
    -----
    * Radar variables should already be filtered for noise to ensure
      accurate LS detection.
    """
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # Resolve variable names
    defaults = {'azi': 'azimuth', 'rng': 'range', "DBZ": "DBZH",
                "RHOHV": "RHOHV"}
    names = {**defaults, **(inp_names or {})}
    azi = names["azi"]
    rng = names["rng"]
    dbz = ds[names["DBZ"]]
    rhohv = ds[names["RHOHV"]]
    # Classification IDs
    echoesID = {"pcpn": 0, "noise": 3, "clutter": 5}
    if classid is not None:
        echoesID.update(classid)
    # Padding
    window = (3, 3)
    arr_rhohv = rhohv.values
    apad = np.pad(arr_rhohv, ((0, 0), (window[1] // 2, window[1] // 2)),
                  mode="constant", constant_values=np.nan)
    apad = np.pad(apad, ((window[0] // 2, window[0] // 2), (0, 0)),
                  mode="wrap")
    # Interior (same shape as original)
    center = apad[1:-1, 1:-1]
    naz, nrg = center.shape
    # Precompute masks
    center_non_nan = ~np.isnan(center)
    mask_nan = np.isnan(apad).astype(int)
    rhohv_low = rhohv.values <= rhv_min
    # Pre-allocate working arrays
    spckl1 = np.ones((naz, nrg), dtype=np.float64)
    spckl2 = np.ones((naz, nrg), dtype=np.float64)
    fclass = np.zeros((naz, nrg), dtype=np.float64)
    fclass2 = np.zeros((naz, nrg), dtype=np.float64)
    # Detect speckles via conv2_8
    conv8 = _conv2_8(mask_nan)
    conv8_int = conv8[1:-1, 1:-1]
    all_neigh_nan = (conv8_int == 8)
    # Mask speckles
    mask_spckl = center_non_nan & all_neigh_nan
    spckl1[mask_spckl] = np.nan
    # First column NaN
    spckl1[:, 0] = np.nan
    # RHOHV threshold
    spckl1[rhohv_low] = np.nan
    # First column forced to 1
    spckl1[:, 0] = 1.0
    # Linear signatures via vertical convolution
    conv_vert = _conv1_vert(mask_nan)
    conv_vert_int = conv_vert[1:-1, 1:-1]
    ls_cond = center_non_nan & (conv_vert_int == 2)
    spckl2[ls_cond] = np.nan
    # Assign classification
    fclass[np.isnan(dbz.values)] = 3.0
    fclass2[np.isnan(spckl1 * spckl2)] = 5.0
    fclass[fclass2 == 5.0] = 5.0
    fclass[:, :5] = 0.0
    if classid is not None:
        fclass[fclass == 0.0] = echoesID["pcpn"]
        fclass[fclass == 3.0] = echoesID["noise"]
        fclass[fclass == 5.0] = echoesID["clutter"]
    # # Prepare output and attach classification
    dims = (azi, rng)
    coords = {azi: ds[azi], rng: ds[rng]}
    cl_da = xr.DataArray(
        fclass, dims=dims, coords=coords,
        attrs=sweep_vars_attrs_f.get('CL_CLASS', ''))
    cl_da.attrs.update({"units": f"flags [{len(echoesID)}]",
                        "flags": echoesID})
    ds_out = xr.Dataset({"CL_CLASS": cl_da}, coords=ds.coords,
                        attrs=ds.attrs.copy())
    outputs = ["CL_CLASS"]
    # Apply masking to selected variables
    # QC API: determine variables to mask
    if mask is None or mask is False:
        return ds_out
    if mask is True:
        vars_to_correct = [v for v, da in ds.data_vars.items() if da.ndim == 2]
        rename_map = {}
    elif isinstance(mask, (list, tuple, set)):
        vars_to_correct = list(mask)
        rename_map = {}
    elif isinstance(mask, dict):
        vars_to_correct = list(mask.keys())
        rename_map = mask
    else:
        raise TypeError("mask must be None, bool, list, or dict")
    missing = [v for v in vars_to_correct if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Variables not found in dataset: {missing}")
    ds_out2 = ds.copy()
    # Attach classification output
    ds_out2 = safe_assign_variable(ds_out2, "CL_CLASS", ds_out["CL_CLASS"])
    corrected_vars = []
    # Mask: keep precipitation, mask noise + clutter
    mask_lsf = (cl_da != echoesID["pcpn"]).astype(bool)
    for var in vars_to_correct:
        # Determine output name
        if isinstance(mask, dict):
            out_var = rename_map[var]
        else:
            out_var = var if replace_vars else f"{var}_QC"
        # Apply masking
        ds_out2 = apply_correction_chain(
            ds_out2, varname=var, mask=mask_lsf, step="ls_filter",
            suffix="" if replace_vars or isinstance(mask, dict) else "_QC",
            params={"rhv_min": rhv_min, "class_ids": echoesID},
            module_provenance="towerpy.eclass.nme.lsinterference_filter")
        # Rename if needed
        internal_name = var if replace_vars else f"{var}_QC"
        if isinstance(mask, dict) and internal_name != out_var:
            if internal_name in ds_out2:
                ds_out2 = ds_out2.rename({internal_name: out_var})
        # Merge canonical attrs
        old_attrs = ds_out2[out_var].attrs.copy()
        new_attrs = sweep_vars_attrs_f.get(out_var, {})
        merged = {**old_attrs, **new_attrs}
        ds_out2 = safe_assign_variable(ds_out2, out_var, ds_out2[out_var],
                                       new_attrs=merged)
        corrected_vars.append(out_var)
    # Redcord provenance
    #TODO: add step_description
    extra = {'step_description': ('')}
    params = {"rhv_min": rhv_min, "class_ids": echoesID, "mask": mask,
              "replace_vars": replace_vars, "corrected_vars": corrected_vars}
    ds_out2 = record_provenance(
        ds_out2, step="lsinterference_filter",
        inputs=[names["DBZ"], names["RHOHV"], names["azi"], names["rng"]],
        outputs=outputs + corrected_vars, parameters=params, extra_attrs=extra,
        module_provenance="towerpy.eclass.nme.lsinterference_filter")
    return ds_out2


def _resolve_binary_class_vars(ds: xr.Dataset, binary_class: int,
                               inp_names: dict = None) -> dict:
    """
    Resolve variables required by the clutter classifier.
    """
    default_inp_names = {"ZH": "DBTH", "RHOHV": "URHOHV", "LDR": "LDR",
                         "PHIDP": "UPHIDP", "ZDR": "UZDR", 'CMAP': 'CMAP'}
    default_inp_names["V"] = "VRADH"
    if "elevation" in ds.coords:
        is_birdbath = np.rad2deg(ds.elevation.mean()) > 85.0
        if is_birdbath:
            default_inp_names["V"] = "VRADV"
    if inp_names is None:
        inp_names = default_inp_names
    else:
        inp_names = {**default_inp_names, **inp_names}

    bitmask_map = {128: "RHOHV",
                   64: 'CMAP',
                   32: "LDR",
                   16: "V",
                   8: "RHOHV",   # sigma(rhoHV)
                   4: "PHIDP",   # sigma(PhiDP)
                   2: "ZDR",     # sigma(ZDR)
                   1: "ZH",      # sigma(ZH)
                   }

    vars_dict = {}
    dims = list(ds.sizes.keys())
    shape_ref = tuple(ds.sizes[d] for d in dims)

    for bit, canonical_name in bitmask_map.items():
        mapped_name = inp_names[canonical_name]
        if binary_class & bit:  # required
            if mapped_name not in ds:
                raise ValueError(
                    f"Variable '{mapped_name}' must be present in dataset "
                    f"when binary_class includes '{canonical_name}' (bit {bit})."
                )
            vars_dict[canonical_name] = ds[mapped_name]
        else:
            # # if not required, then dummy array
            vars_dict[canonical_name] = xr.DataArray(
                np.ones(shape_ref), dims=dims,
                coords={d: ds[d] for d in dims}, attrs={"dummy": True})
    return vars_dict


def clutter_classif(ds, inp_names=None, min_snr=None, rcst_dB=None, cmap=None,
                    binary_class=255, path_nds=None, classid=None,
                    mask=None, replace_vars=False):
    r"""
    Classify clutter, noise, and precipitation echoes using the
    clutter-classification method described in Rico-Ramirez & Cluckie (2008).
    Optionally apply CL-based masking to selected variables.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing polarimetric variables and polar coordinates
        (range, azimuth, elevation). Variables must already be noise-filtered.
    inp_names : dict, optional
        Mapping for variable/attribute names in the dataset. Defaults:
        ``{'azi': 'azimuth', 'rng': 'range', 'ZH': 'DBZH', 'RHOHV': 'RHOHV',
        'LDR': 'LDR', 'ZDR': 'ZDR', 'PHIDP': 'PHIDP', 'V': 'VRADH',
        'CMAP': 'CMAP'}``
    min_snr : float, default None
        Minimum signal‑to‑noise ratio threshold (in dB). Values below
        this threshold are classified as noise. If None, the function attempts
        to retrieve the radar constant from metadata. If missing, default is 0.
    rcst_dB : float, default None
        Radar constant (in dB). If None, the function attempts to retrieve
        the radar constant from metadata. If missing, default is 0.
    cmap : array-like or xarray.DataArray, optional
        Clutter frequency map (CMAP) in the range [0, 1]. If provided, it is
        inserted into the dataset under the variable name defined by
        ``inp_names["CMAP"]``.
    binary_class : int, default 255
        Bitmask specifying which polarimetric variables are used in the
        classifier. Each bit corresponds to a variable:
            - 128 → :math:`\rho_{HV}`
            - 64  → CMAP
            - 32  → LDR
            - 16  → Doppler velocity
            - 8   → texture of :math:`\rho_{HV}`
            - 4   → texture of :math:`\Phi_{DP}`
            - 2   → texture of :math:`Z_{DR}`
            - 1   → texture of :math:`Z_{H}`
        Required variables must exist in the dataset when their bit is set.
    path_nds : str or pathlib.Path, default None
        Path to the directory containing the normalised distribution (NDS)
        files used by the classifier. If ``None``, built-in C-band NDS files
        are used.
    classid : dict, default None
        Override classification IDs for ``'pcpn'``, ``'noise'``, and
        ``'clutter'``. Defaults are ``{'pcpn': 0, 'noise': 3, 'clutter': 5}``.
    mask : bool, list of str, dict of str to str, or None, optional
        Controls which variables receive clutter-based masking.
    
        * ``None`` or ``False``: classification only; no masking applied.
        * ``True``: mask all 2‑D data variables in the dataset.
        * list of str: mask only the listed variables.
        * dict: map input variable names to explicit output names.    
    replace_vars : bool, default False
        If True, overwrite selected variables.
        If False, masked variables receive a ``_QC`` suffix unless explicit
        names are provided via ``mask`` (dict form).

    Returns
    -------
    xarray.Dataset
        Dataset containing:
    
        - ``CL_CLASS`` : 2‑D classification field (pcpn / noise / clutter)
        - ``CMAP``     : clutter map (if provided or present in input)
        - masked variables (if ``mask`` is True, a list, or a dict)
        - updated variable-level and dataset-level provenance
    
    Notes
    -----
    * The normalised frequency distributions describe the empirical
      distributions of the polarimetric variables, as shown in [1]_ or [2]_.
      When providing a custom path, it must point to a directory containing a
      complete and compatible set of NDS files.
    * Variables required by the classifier must be present if their
      corresponding bit is set in ``binary_class``. For example, if the
      bitmask includes ``s(rhoHV)`` (8), then the dataset must contain a
      valid ``rhoHV`` variable. Missing variables raise a ``ValueError`` during
      internal validation.
    * Texture fields required by the classifier are computed internally.
    * If the CMAP bit (64) is set in binary_class and a CMAP variable already
      exists in ds, it is used automatically. If CMAP is required but missing,
      the user must provide it via cmap. Passing cmap always overrides any
      existing CMAP.
    * This function uses the shared object 'lnxlibclutterclassifier'
      or the dynamic link library 'w64libclutterclassifier' depending on the
      operating system (OS).
    * Units for range, azimuth and elevation are inspected and converted to
      the appropriate units (m, rad) when necessary.

    References
    ----------
    .. [1] Rico-Ramirez, M., & Cluckie, I. (2008). Classification of ground
        clutter and anomalous propagation using Dual-Polarization Weather
        Radar. IEEE Transactions on Geoscience and Remote Sensing, 46(7),
        1892–1904. https://doi.org/10.1109/tgrs.2008.916979
    .. [2] Sanchez-Rivas, D., & Rico-Ramirez, M. A. (2023). Towerpy: An
        open-source toolbox for processing polarimetric weather radar data.
        Environmental Modelling & Software, 167, 105746.
        https://doi.org/10.1016/j.envsoft.2023.105746
    """
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # Resolve variable names
    defaults = {'azi': 'azimuth', 'rng': 'range', 'elv': 'elevation',
                'minSNR': 'min_snr_threshold', "ZH": "DBZH", "RHOHV": "RHOHV",
                "LDR": "LDR", "ZDR": "ZDR", "PHIDP": "PHIDP", 'CMAP': 'CMAP'}
    defaults["V"] = "VRADH"
    if "elevation" in ds.coords:
        is_birdbath = convert(ds.elevation, 'deg').mean() > 85.0
        if is_birdbath:
            defaults["V"] = "VRADV"
    names = {**defaults, **(inp_names or {})}
    # Classification IDs
    echoesID = {'pcpn': 0, 'noise': 3, 'clutter': 5}
    if classid is not None:
        echoesID.update(classid)
    # Set path of the NDS
    if path_nds is None:
        pathnds = str.encode(str(Path(__file__).parent.absolute())
                             + '/mfs_cband/')
    else:
        pathnds = str.encode(path_nds.as_posix().rstrip("/") + "/")
    # Prepare inputs for ctypes
    array1d = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
    array2d = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
    if platform.system() == 'Linux':
        libclc = 'lnxlibclutterclassifier.so'
        load_libclc = npct.load_library(libclc, Path(__file__).parent.absolute())
        # load_libclc = npct.load_library(libclc, Path.cwd())
    elif platform.system() == 'Windows':
        libclc = 'w64libclutterclassifier.dll'
        load_libclc = ctp.cdll.LoadLibrary(f'{Path(__file__).parent.absolute()}/'
                                     + libclc)
    else:
        load_libclc = None
        libclc = 'no_libraryOS'
        raise ValueError(f'The {platform.system()} OS is not compatible'
                         ' with this version of Towerpy')
    load_libclc.clutterclassifier.restype = None
    load_libclc.clutterclassifier.argtypes = [
        ctp.c_char_p, ctp.c_int, ctp.c_int, array2d, array2d, array2d, array2d,
        array2d, array2d, array2d, array1d, array1d, array1d, array1d, array2d]
    ds = ds.copy()
    # Inject CMAP if provided
    if cmap is not None:
        # Resolve the expected variable name
        cmap_name = names["CMAP"]
        if isinstance(cmap, xr.DataArray):
            # Validate dims
            if cmap.dims != (names["azi"], names["rng"]):
                raise ValueError(
                    f"CMAP DataArray dims {cmap.dims} do not match expected "
                    f"({names['azi']}, {names['rng']})")
            ds[cmap_name] = cmap
        else:
            # Assume numpy-like; enforce dims explicitly
            ds[cmap_name] = ((names["azi"], names["rng"]), np.asarray(cmap))
    rng_m = convert(ds[names["rng"]], "m")
    azi_rad = convert(ds[names["azi"]], "rad")
    elv_rad = convert(ds[names["elv"]], "rad")
    rng_m = np.ascontiguousarray(rng_m.values, dtype=np.float64)
    azi_rad = np.ascontiguousarray(azi_rad.values, dtype=np.float64)
    elv_rad = np.ascontiguousarray(elv_rad.values, dtype=np.float64)
    
    param_clc = np.zeros(5)
    # Resolve radar constant
    rc = float(rcst_dB) if rcst_dB is not None else get_attrval(
        "radconstH", ds, default=None)
    param_clc[0] = rc
    # Resolve minSNR
    min_snr = float(min_snr) if min_snr is not None else get_attrval(
        "minSNR", ds, default=None)
    param_clc[1] = min_snr
    param_clc[2] = binary_class
    clc = np.zeros((ds.sizes[names["azi"]], ds.sizes[names["rng"]]))
    # Resolve variables
    vars_dict = _resolve_binary_class_vars(ds, binary_class, names)
    # Z = vars_dict["ZH"].fillna(-50.0)
    Z = np.nan_to_num(vars_dict["ZH"], nan=-50.0)
    ZDR = vars_dict["ZDR"]
    PhiDP = vars_dict["PHIDP"]
    rhoHV = vars_dict["RHOHV"]
    V = vars_dict["V"]
    LDR = vars_dict["LDR"]
    CM = vars_dict['CMAP']
    # Call C library
    load_libclc.clutterclassifier(
        pathnds, ds.sizes[names["azi"]], ds.sizes[names["rng"]],
        np.ascontiguousarray(Z, dtype=np.float64),
        np.ascontiguousarray(ZDR, dtype=np.float64),
        np.ascontiguousarray(PhiDP, dtype=np.float64),
        np.ascontiguousarray(rhoHV, dtype=np.float64),
        np.ascontiguousarray(V, dtype=np.float64),
        np.ascontiguousarray(LDR, dtype=np.float64),
        np.ascontiguousarray(CM, dtype=np.float64),
        rng_m, azi_rad, elv_rad, param_clc, clc)
    # Remap classification IDs
    if classid is not None:
       clc.values[clc.values == 0] = echoesID["pcpn"]
       clc.values[clc.values == 3] = echoesID["noise"]
       clc.values[clc.values == 5] = echoesID["clutter"]
    # Prepare output dataset
    dims = (names["azi"], names["rng"])
    coords = {names["azi"]: ds[names["azi"]], names["rng"]: ds[names["rng"]]}  
    # Attach classification
    ds_out = ds.copy()
    ds_out = safe_assign_variable(
        ds_out, "CL_CLASS",
        xr.DataArray(clc, dims=dims, coords=coords,
                     attrs=sweep_vars_attrs_f.get("CL_CLASS", {})))
    ds_out.CL_CLASS.attrs.update({"units": f"flags [{len(echoesID)}]",
                                  "flags": echoesID})
    # Attach clutter map only if it's a real variable, not dummy
    if not CM.attrs.get("dummy", False):
        # Build DataArray with attrs explicitly
        ds_out = safe_assign_variable(
            ds_out, "CMAP",
            xr.DataArray(CM.values, dims=CM.dims, coords=CM.coords,
                         attrs=sweep_vars_attrs_f.get("CMAP", {})))
        ds_out.CMAP.attrs.update({"units": 'relative_frequency'})
    # QC API: determine variables to mask
    if mask is None or mask is False:
        return ds_out
    if mask is True:
        vars_to_correct = [v for v, da in ds.data_vars.items() if da.ndim == 2]
        rename_map = {}
    elif isinstance(mask, (list, tuple, set)):
        vars_to_correct = list(mask)
        rename_map = {}
    elif isinstance(mask, dict):
        vars_to_correct = list(mask.keys())
        rename_map = mask
    else:
        raise TypeError("mask must be None, bool, list, or dict")
    # Validate variables
    missing = [v for v in vars_to_correct if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Variables not found in dataset: {missing}")
    ds_out2 = ds.copy()
    # Attach classification output
    ds_out2 = safe_assign_variable(ds_out2, "CL_CLASS", ds_out["CL_CLASS"])
    if "CMAP" in ds_out:
        ds_out2 = safe_assign_variable(ds_out2, "CMAP", ds_out["CMAP"])
    corrected_vars = []
    # Mask: keep precipitation, mask noise + clutter
    mask_cl = (clc != echoesID["pcpn"]).astype(bool)
    for var in vars_to_correct:
        # Determine output name
        if isinstance(mask, dict):
            out_var = rename_map[var]
        else:
            out_var = var if replace_vars else f"{var}_QC"
        # Apply masking
        ds_out2 = apply_correction_chain(
            ds_out2, varname=var, mask=mask_cl, step="nme_classif",
            suffix="" if replace_vars or isinstance(mask, dict) else "_QC",
            params={"class_ids": echoesID},
            module_provenance="towerpy.eclass.nme.clutter_classif")
        # Rename if needed
        internal_name = var if replace_vars else f"{var}_QC"
        if isinstance(mask, dict) and internal_name != out_var:
            if internal_name in ds_out2:
                ds_out2 = ds_out2.rename({internal_name: out_var})
        # Merge canonical attrs
        old_attrs = ds_out2[out_var].attrs.copy()
        new_attrs = sweep_vars_attrs_f.get(out_var, {})
        merged = {**old_attrs, **new_attrs}
        ds_out2 = safe_assign_variable(ds_out2, out_var, ds_out2[out_var],
                                       new_attrs=merged)
        corrected_vars.append(out_var)
    # Provenance
    outputs = ["CL_CLASS"]
    if not CM.attrs.get("dummy", False):
        outputs.append("CMAP")
    # Build provenance inputs from actually used variables
    inputs = [names["rng"], names["azi"], names["elv"]]
    for canonical_name, da in vars_dict.items():
        # Skip dummy placeholders (not real inputs)
        if da.attrs.get("dummy", False):
            continue
        # Use the mapped dataset name, not the canonical key
        mapped_name = names.get(canonical_name, canonical_name)
        if mapped_name in ds:
            inputs.append(mapped_name)
    #TODO: add step_description
    params = {"min_snr": min_snr,
              "radar_constant_value_dB": rc,
              "range_var": names["rng"],
              "azimuth_var": names["azi"],
              "class_ids": echoesID,
              "binary_class": int(binary_class),
              "mask": mask,
              "replace_vars": replace_vars,
              "corrected_vars": corrected_vars,
              "library": libclc,}
    extra = {'step_description': ('')}
    ds_out2 = record_provenance(
        ds_out2, step="clutter_classif", inputs=inputs,
        outputs=outputs + corrected_vars, parameters=params, extra_attrs=extra,
        module_provenance="towerpy.eclass.nme.clutter_classif")
    return ds_out2
