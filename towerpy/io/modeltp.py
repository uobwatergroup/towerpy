"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

from xradar import model

"""
Use XRadar Data Model
=================

Import the data model from xradar, adding some non-standard sweep dataset
variable names realted to function outputs.
"""

# towerpy/io/modeltp.py

sweep_vars_attrs_xrd = model.sweep_vars_mapping

sweep_vars_attrs_ukmo = {
    'CI': {'standard_name': 'clutter_indicator',
           'long_name': 'pulse-to-pulse variation of power',
           'short_name': 'CI',
           'units': 'dB'},
    }

sweep_vars_attrs_snr = {
    "SNR_CLASS": {'standard_name': 'signal_noise_class',
                  'long_name': 'Signal and Noise Classification',
                  'short_name': 'SNRCLASS',
                  'units': 'flags', 'flags': 'echoesID'},
    "DBSNR": {'standard_name': 'signal_noise_ratio',
              'long_name': 'Signal Noise Ratio',
              'short_name': 'DBSNR',
              'units': 'dB'},
    }

sweep_vars_attrs_clc = {
    "CL_CLASS": {'standard_name': 'clutter_class',
                 'long_name': 'clutter, noise and weather echoes classification',
                 'short_name': 'CLCLASS',
                 'units': 'flags',
                 'flags': 'echoesID'},
    }

sweep_vars_attrs_mlpcp = {
    "ML_PCP_CLASS": {'standard_name': 'mlpcp_delimitation',
                     'long_name': 'precipitation region delimitation based'
                     ' on the melting layer boundaries ',
                     'short_name': 'MLPCLASS',
                     'units': 'flags',
                     'flags': 'echoesID'},
    }

sweep_vars_attrs_attc = {
    "AH": {'standard_name': 'specific_attenuation_h',
           'long_name': 'Specific horizontal attenuation',
           'short_name': 'AH',
           'units': 'dB per kilometer'},
    "AV": {'standard_name': 'specific_attenuation_v',
           'long_name': 'Specific vertical attenuation',
           'short_name': 'AV',
           'units': 'dB per kilometer'},
    "ADP": {'standard_name': 'specific_differential_attenuation',
            'long_name': 'Specific differential attenuation',
            'short_name': 'ADP',
            'units': 'dB per kilometer'},
    "PIA": {'standard_name': 'path_integrated_attenuation',
            'long_name': 'Path-Integrated Attenuation',
            'short_name': 'PIA',
            'units': 'dB'},
    "ALPHA": {'standard_name': 'alpha_attenuation',
              'long_name': 'ratio between A_H and K_{DP}',
              'short_name': 'ALPHA',
              'units': 'unitless'},
    "BETA": {'standard_name': 'beta_attenuation',
             'long_name': 'ratio between A_{DP} and A_H',
             'short_name': 'BETA',
             'units': 'unitless'},
    "GAMMA": {'standard_name': 'gamma_attenuation',
              'long_name': 'ratio between ALPHA and BETA',
              'short_name': 'GAMMA',
              'units': 'unitless'},
    }

sweep_vars_attrs_qpe = {
    "RAIN_RATE": {"standard_name": "rainfall_rate",
                  "long_name": "rainfall_rate",
                  'short_name': 'RATE',
                  "units": "mm h-1"},
    "RAIN_ACCUM": {"standard_name": "rain_accumulation",
                   "long_name": "rainfall_amount",
                   'short_name': 'MLPCLASS',
                   "units": "mm"},
    }

sweep_vars_attrs_profs = {
    "GRAD_VRADV": {
        "standard_name": "vertical_gradient_of_radial_velocity",
        "long_name": "Vertical gradient of Doppler radial velocity",
        "short_name": "GRAD_VRADV",
        "units": "∂V/∂h"},
    'BIN_CLASS': {
        "standard_name": "precipitation_classification_in_height_bins",
        "long_name": "Per-height-bin precipitation classification",
        "short_name": "BIN_CLASS",
        "units": "flags [6]",
        "flags": {"no_rain": 0, "light_rain": 1, "modrt_rain": 2,
                  "heavy_rain": 3, "mixed_pcpn": 4, "solid_pcpn": 5}},
    'PROF_TYPE': {
        "standard_name": "precipitation_type_of_profile",
        "long_name": "Profile precipitation type (0–6 scheme)",
        "short_name": "PROF_TYPE",
        "units": "flags [7]",
        "flags": {"NR": 0, "LR [STR]": 1, "MR [STR]": 2, "HR [STR]": 3,
                  "LR [CNV]": 4, "MR [CNV]": 5, "HR [CNV]": 6}},
    'PCP_TYPE': {
        "standard_name": "scalar_precipitation_type_of_profile",
        "long_name": "Scalar precipitation type",
        "short_name": "PCP_TYPE",
        "units": "flags [7]",
        "flags": {"NR": 0, "LR [STR]": 1, "MR [STR]": 2, "HR [STR]": 3,
                  "LR [CNV]": 4, "MR [CNV]": 5, "HR [CNV]": 6}}
    }

sweep_vars_attrs_f = (sweep_vars_attrs_xrd     |
                      sweep_vars_attrs_ukmo    |
                      sweep_vars_attrs_snr     |
                      sweep_vars_attrs_clc     |
                      sweep_vars_attrs_mlpcp   |
                      sweep_vars_attrs_attc    |
                      sweep_vars_attrs_qpe     |
                      sweep_vars_attrs_profs
                      )
