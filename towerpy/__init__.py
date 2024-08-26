"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

# import subpackages
from . import attc
from . import calib
from . import datavis
from . import eclass
from . import georad
from . import io
from . import ml
from . import profs
from . import qpe
from . import utils

_welcome_text = """
You are using the Towerpy framework, an open source library for
working with polarimetric weather radar data.

If you find our work useful for your research, please consider citing our
following publication:

https://doi.org/10.1016/j.envsoft.105746.2023
"""

print(_welcome_text)

name = 'towerpy'
__version__ = "v1.0.5"
