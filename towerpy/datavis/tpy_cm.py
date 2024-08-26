"""Towerpy: an open-source toolbox for processing polarimetric radar data."""


import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
from .tpy_colors import towerpy_colours
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


def plot_color_gradients(category, cmap_list):
    """
    Plot data with the associated colormaps.

    Parameters
    ----------
    category : TYPE
        DESCRIPTION.
    cmap_list : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    cmaps = {}
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(9.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list


def plot_examples(colormaps, vmin=-5, vmax=5):
    """
    Plot random data to depict the colormaps.

    Parameters
    ----------
    colormaps : TYPE
        DESCRIPTION.
    vmin : TYPE, optional
        DESCRIPTION. The default is -5.
    vmax : TYPE, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    None.

    """
    np.random.seed(19680801)
    data = np.random.randn(50, 50)
    # n = len(colormaps)
    nitems = len(colormaps)
    if nitems > 5:
        ncols = int(nitems**0.5)
        nrows = nitems // ncols
        #     EDIT for correct number of rows:
        #     If one additional row is necessary -> add one:
        if nitems % ncols != 0:
            nrows += 1
        fsize = (9, 9)
    else:
        ncols = nitems
        nrows = 1
        fsize = (nitems * 2 + 2, 3)

    fig, axs = plt.subplots(nrows, ncols, figsize=fsize,
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True,
                            vmin=vmin, vmax=vmax)
        fig.colorbar(psm, ax=ax)
    plt.show()


def plot_color_gradients_grayscale(cmap_list):
    """
    Plot random data to depict the colormaps in grayscale.

    Parameters
    ----------
    cmap_list : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    from colorspacious import cspace_converter

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    x = np.linspace(0.0, 1.0, 100)

    fig, axs = plt.subplots(nrows=len(cmap_list), ncols=2)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99,
                        wspace=0.05)
    fig.suptitle(' colormaps', fontsize=14, y=1.0, x=0.6)

    for ax, name in zip(axs, cmap_list):

        # Get RGB values for colormap.
        rgb = mpl.colormaps[name](x)[np.newaxis, :, :3]

        # Get colormap in CAM02-UCS colorspace. We want the lightness.
        lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
        L = lab[0, :, 0]
        L = np.float32(np.vstack((L, L, L)))

        ax[0].imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
        ax[1].imshow(L, aspect='auto', cmap='binary_r', vmin=0., vmax=100.)
        pos = list(ax[0].get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs.flat:
        ax.set_axis_off()

    plt.show()


def plot_color_gradients_cvd(cmap_list, cvd_type='protanomaly',
                             severity=100):
    """
    Plot random data to depict the colormaps for CVD.

    Parameters
    ----------
    cmap_list : TYPE
        DESCRIPTION.
    cvd_type : TYPE, optional
        DESCRIPTION. The default is 'protanomaly'.
    severity : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    None.

    """
    from colorspacious import cspace_convert

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    cvd_space = {'name': 'sRGB1+CVD', 'cvd_type': cvd_type,
                 'severity': severity}

    x = np.linspace(0.0, 1.0, 100)

    fig, axs = plt.subplots(nrows=len(cmap_list), ncols=2, figsize=(10, 6))
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.11, right=0.99,
                        wspace=0.05)
    fig.suptitle(f'Color vision deficiency simulation [{cvd_type}'
                 + f' - severity:{severity}]',
                 fontsize=14, y=1.0, x=0.6)

    for ax, name in zip(axs, cmap_list):

        # Get RGB values for colormap.
        rgb = mpl.colormaps[name](x)[np.newaxis, :, :3]

        # Get colormap in CAM02-UCS colorspace. We want the lightness.
        # lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
        # hopper_deuteranomaly_sRGB

        lab = cspace_convert(rgb, cvd_space, "sRGB1")
        L = np.clip(lab, 0, 1)
        # L = lab[0, :, 0]
        # L = np.float32(np.vstack((L, L, L)))

        ax[0].imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
        ax[1].imshow(L, aspect='auto', cmap='binary_r', vmin=0., vmax=100.)
        pos = list(ax[0].get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs.flat:
        ax.set_axis_off()

    plt.show()


# This part creates colour maps from a dictionary listing colour gradients.
tpy_LSC = {cname: mcolors.LinearSegmentedColormap.from_list(cname, colrs)
           for cname, colrs in towerpy_colours.items()}

tpy_LC = {cname: mcolors.ListedColormap(colrs)
          for cname, colrs in towerpy_colours.items()}

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
