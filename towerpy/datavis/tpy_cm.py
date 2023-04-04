"""Towerpy: an open-source toolbox for processing polarimetric radar data."""


import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from .tpy_colors import towerpy_colours

"""
This file create colour maps from a dictionary listing colour gradients.
"""

warnings.filterwarnings("ignore", category=UserWarning)

tpy_LSC = {cname: mcolors.LinearSegmentedColormap.from_list(cname, colrs)
           for cname, colrs in towerpy_colours.items()}

tpy_LC = {cname: mcolors.ListedColormap(colrs)
          for cname, colrs in towerpy_colours.items()}

comb_tpycms = np.vstack((mpl.colormaps.get_cmap(tpy_LSC['grey'])(np.linspace(0, 1,
                                                                             64)),
                         mpl.colormaps.get_cmap(tpy_LSC['pvars'])(np.linspace(0, 1,
                                                                              192))))

tpy_LSC['2slope'] = mcolors.LinearSegmentedColormap.from_list('2slope',
                                                              comb_tpycms)

tpy_LSC_r = {cname+'_r': clmap.reversed()
             for cname, clmap in tpy_LSC.items()}

tpy_LC_r = {cname+'_r': clmap.reversed()
            for cname, clmap in tpy_LC.items()}

locals().update(tpy_LSC)
locals().update(tpy_LC)
locals().update(tpy_LSC_r)
locals().update(tpy_LC_r)

for name, cmap in tpy_LSC.items():
    fname = 'tpylsc_' + name
    mpl.colormaps.register(name=fname, cmap=cmap, force=True)

for name, cmap in tpy_LC.items():
    fname = 'tpylc_' + name
    mpl.colormaps.register(name=fname, cmap=cmap, force=True)

for name, cmap in tpy_LSC_r.items():
    fname = 'tpylsc_' + name
    mpl.colormaps.register(name=fname, cmap=cmap, force=True)

for name, cmap in tpy_LC_r.items():
    fname = 'tpylc_' + name
    mpl.colormaps.register(name=fname, cmap=cmap, force=True)
