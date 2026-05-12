"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import warnings
import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import pickle
import xarray as xr
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from matplotlib.widgets import RadioButtons, Slider
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from scipy import spatial
from ..utils import radutilities as rut
from ..base import TowerpyError
from ..datavis.rad_display import pltparams
from ..datavis.rad_display import plot_params, plot_ppi_xr
from ..datavis.rad_display import _add_colorbar, _resolve_var2plot
from ..utils.unit_conversion import _safe_units, convert, np64_to_dtm
from ..utils.radutilities import resolve_rect_coords, _safe_metadata


def format_coord(x, y):
    """
    Format the coordinates used in plots.

    Parameters
    ----------
    x : float
        x-coordinates.
    y : float
        y-coordinates.

    Returns
    -------
    z: str
        Value of a given pixel.
    [q, r] : list
        angle and range of a given pixel.

    """
    if gcoord_sys == 'rect':
        xy = [(x, y)]
        distance, index = spatial.KDTree(gflat_coords).query(xy)
        id1 = np.unravel_index(index, (intradparams['nrays'],
                                       intradparams['ngates']))
        q = id1[0][0]
        r = id1[1][0]
        z = mrv[id1[0][0], id1[1][0]]
        if r < intradparams['ngates']:
            z = mrv[q, r]
            return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f} [{q},{r}]'
        else:
            return f'x={x:1.4f}, y={y:1.4f}'


# def format_coord2(x, y):
#     """
#     Format the coordinates used in plots.

#     Parameters
#     ----------
#     x : float
#         x-coordinates.
#     y : float
#         y-coordinates.

#     Returns
#     -------
#     z: str
#         Value of a given pixel.
#     [q, r] : list
#         angle and range of a given pixel.

#     """
#     gres_m = intradparams['gateres [m]']
#     ngates_m = intradparams['ngates']
#     if gcoord_sys == 'rect':
#         q, r = quadangle(x, y, gres_m)
#         if q >= 360:
#             q = 0
#         if r < ngates_m:
#             z = mrv[q, r]
#             return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f} [{q},{r}]'
#         else:
#             return f'x={x:1.4f}, y={y:1.4f}'


# def quadangle(x, y, gater):
#     """
#     Compute the range and angle of a given pixel.

#     Parameters
#     ----------
#     x : float
#         x-coordinates.
#     y : float
#         y-coordinates.
#     gater : float
#         Gate resolution, in m.

#     Returns
#     -------
#     theta : int
#         Angle of a given pixel.
#     nrange : int
#         Range of a given pixel

#     """
#     if intradparams['range_start [m]'] == 0:
#         quaddmmy = 0.5
#     else:
#         quaddmmy = 0
#     if x > 0 and y > 0:
#         theta = 89 - int(np.degrees(np.arctan(y/x)))
#     elif x < 0 and y > 0:
#         theta = abs(int(np.degrees(np.arctan(y/x))))+270
#     elif x < 0 and y < 0:
#         theta = abs(int(np.degrees(np.arctan(x/y))))+180
#     else:
#         theta = abs(int(np.degrees(np.arctan(y/x))))+90
#     nrange = abs(int(math.hypot(x, y) / (gater/1000)+quaddmmy))
#     return theta, nrange


class PPI_Int:
    """A class to create an interactive PPI plot."""

    def __init__(self):
        figradint.canvas.mpl_connect('button_press_event', self.on_pick)
        figradint.canvas.mpl_connect('key_press_event', self.on_press)
        figradint.canvas.mpl_connect('key_press_event', self.on_key)
        self.lastind = [0, 0]
        self.keycoords = []
        self.current_line = [None]
        self.text = f3_axvar2plot.text(0.01, 0.03, 'selected: none',
                                       transform=f3_axvar2plot.transAxes,
                                       va='top')

    def on_pick(self, event):
        """
        Get the click locations.

        Parameters
        ----------
        event : Mouse click
            Right or left click from the mouse.

        """
        # gres_m = intradparams['gateres [m]']
        if gcoord_sys == 'polar':
            if event.button is MouseButton.LEFT:
                if event.inaxes != f3_axvar2plot:
                    return True
                if event.xdata >= 0:
                    nangle = int(np.round(np.rad2deg(event.xdata)))
                else:
                    nangle = int(np.round(np.rad2deg(event.xdata)+359))
                nrange = event.ydata
                cdt = [nangle, nrange]
                print(f'azimuth {abs(nangle-359)}',
                      f'range {int(np.round(nrange))}')
                self.lastind = cdt
                self.update()
        if gcoord_sys == 'rect':
            if event.button is MouseButton.RIGHT:
                if event.inaxes != f3_axvar2plot:
                    return True
                xy = [(event.xdata, event.ydata)]
                distance, index = spatial.KDTree(gflat_coords).query(xy)
                id1 = np.unravel_index(index, (intradparams['nrays'],
                                               intradparams['ngates']))
                nangle = id1[0][0]
                nrange = id1[1][0]
                cdt = [nangle, nrange]
                print(f'azimuth {abs(nangle)}',
                      f'gate {int(np.round(nrange))}')
                if self.current_line[0] is not None:
                    self.current_line[0].remove()
                # Define specific target point
                x_target, y_target = xy[0]
                # Draw the line
                f3line, = f3_axvar2plot.plot([x_center, x_target],
                                             [y_center, y_target],
                                             'ok:', lw=2,  mfc='none')

                self.current_line[0] = f3line
                self.lastind = cdt
                self.update()

    def on_press(self, event):
        """
        Browse through the next and previous azimuth.

        Parameters
        ----------
        event : key-press, 'n' or 'm'
            Keyboard event.

        """
        if self.lastind is None:
            return
        if event.key not in ('n', 'm'):
            return
        if event.key == 'n':
            inc = -1
        else:
            inc = 1

        if self.current_line[0] is not None:
            self.current_line[0].remove()
        x_ray = intradgeoref['grid_rectx'][self.lastind[0] + inc, :]
        y_ray = intradgeoref['grid_recty'][self.lastind[0] + inc, :]
        if gcoord_sys == 'rect':
            f3line, = f3_axvar2plot.plot(x_ray, y_ray, 'ok:', lw=2,  mfc='none',
                                         markevery=[0,-1])
            self.current_line[0] = f3line
        self.lastind[0] += inc
        self.lastind[0] = np.clip(self.lastind[0], 0, intradparams['nrays']-1)
        self.update()

    def on_key(self, event):
        """
        Record keyboard presses.

        Parameters
        ----------
        event : key-press, 0-9
            Record the coordinates when user press any number from 0 to 9.

        """
        # gres_m = intradparams['gateres [m]']
        keynum = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        if event.key in keynum and event.inaxes == f3_axvar2plot:
            # nangle, nrange = quadangle(event.xdata, event.ydata, gres_m)
            xy = [(event.xdata, event.ydata)]
            distance, index = spatial.KDTree(gflat_coords).query(xy)
            id1 = np.unravel_index(index, (intradparams['nrays'],
                                           intradparams['ngates']))
            nangle = id1[0][0]
            nrange = id1[1][0]
            print('you pressed', event.key, nangle, nrange)
            self.keycoords.append((nangle, nrange, event.key))

    def update(self):
        """Update the interactive plot."""
        if self.lastind is None:
            return

        nangle = self.lastind[0]
        nrange = self.lastind[1]
        heightbeamint = intradgeoref['beam_height [km]']
        intradarrange = intradgeoref['range [m]']
        # gres_m = intradparams['gateres [m]']
        ngates_m = intradparams['ngates']

        for i in intradaxs:
            intradaxs[i].cla()
        f3_axhbeam.cla()

        f3_axhbeam.plot(intradarrange/1000, heightbeamint[nangle], ls=':')
        f3_axhbeam.plot(intradarrange/1000, rbeamh_t[nangle], c='k')
        f3_axhbeam.plot(intradarrange/1000, rbeamh_b[nangle], c='k')
        f3_axhbeam.set_xlabel('Range [Km]', fontsize=14)
        f3_axhbeam.set_ylabel('Beam height [Km]', fontsize=14)
        # f3_axhbeam.set_ylabel('Beam height [Km]', fontsize=14)
        f3_axhbeam.grid()

        if gcoord_sys == 'polar':
            nrange2 = rut.find_nearest(nrange, intradarrange/1000)
            f3_axhbeam.axhline(heightbeamint[nangle][nrange2], alpha=0.5,
                               c='tab:red')
            f3_axhbeam.axvline(nrange, alpha=0.5, c='tab:red')
        if gcoord_sys == 'rect' and nrange < ngates_m:
            f3_axhbeam.axhline(heightbeamint[nangle][int(np.round(nrange))],
                               alpha=0.5, c='tab:red')
            f3_axhbeam.axvline(intradarrange[int(np.round(nrange))]/1000,
                               alpha=0.5, c='tab:red')

        if gcoord_sys == 'polar':
            f3_axvar2plot.set_thetagrids(np.arange(nangle, nangle+2))
            f3_axvar2plot.set_xticklabels([])
            for i, j in enumerate(intradvars):
                intradaxs[f'f3_ax{i+2}'].plot(intradarrange/1000,
                                              np.flipud(intradvars[j])[nangle,
                                                                       :],
                                              marker='.', markersize=3)
                if '[dB]' in j:
                    intradaxs[f'f3_ax{i+2}'].axhline(0, alpha=.2, c='gray')
                intradaxs[f'f3_ax{i+2}'].set_title(j)
                intradaxs[f'f3_ax{i+2}'].axvline(nrange, alpha=.2)
        if gcoord_sys == 'rect' and nrange < ngates_m:
            for i, j in enumerate(intradvars):
                intradaxs[f'f3_ax{i+2}'].plot(intradarrange/1000,
                                              intradvars[j][nangle, :],
                                              marker='.', markersize=3)
                intradaxs[f'f3_ax{i+2}'].axvline(
                    intradarrange[int(np.round(nrange))]/1000, alpha=.2)
                if j == 'ZDR [dB]':
                    intradaxs[f'f3_ax{i+2}'].axhline(0, alpha=.8, c='gray')
                if j == 'rhoHV [-]':
                    intradaxs[f'f3_ax{i+2}'].axhline(1., alpha=.8, c='thistle')
                intradaxs[f'f3_ax{i+2}'].set_title(j)
        if vars_ylim is not None:
            for i, j in enumerate(intradvars):
                if j in vars_ylim:
                    intradaxs[f'f3_ax{i+2}'].set_ylim(vars_ylim[j])
        else:
            for i in intradaxs:
                intradaxs[i].autoscale()
        intradaxs[list(intradaxs)[-1]].set_xlabel('Range [Km]', fontsize=14)

        for i in intradaxs:
            if vars_xlim is not None:
                intradaxs[i].set_xlim(vars_xlim)
            else:
                intradaxs[i].set_xlim(0, max(intradarrange/1000))
        # intradaxs[list(intradaxs)[-1]].set_xlim(0, 250)
        if gcoord_sys == 'polar':
            self.text.set_text(f'selected: {np.abs(nangle-359)}')
        elif gcoord_sys == 'rect' and nrange < ngates_m:
            self.text.set_text(f'selected: azim={np.abs(nangle)},'
                               + f' bin_range={nrange}')

        figradint.canvas.draw()

    def savearray2binfile(self, file_name, dir2save, min_snr=None, rsite=None):
        """
        Save the coordinates and pixel values of key-mouse events in a binfile.

        Parameters
        ----------
        file_name : str
            Name of the file to be saved.
        dir2save : str
            Directory of the file to be saved.

        """
        coord_lst = self.keycoords
        b = [[pcoords[0], pcoords[1], int(pcoords[2])]
             for pcoords in coord_lst]
        coord_lstnd = list(set([i for i in [tuple(i) for i in b]]))
        # gres_m = intradparams['gateres [m]']
        ngates1 = intradparams['ngates']
        nrays1 = intradparams['nrays']
        fgates = [i for i, j in enumerate(coord_lstnd) if j[1] >= ngates1]
        frays = [i for i, j in enumerate(coord_lstnd) if j[0] >= nrays1]
        if len(fgates) > 0 or len(frays) > 0:
            print('Some selected pixels were out of the PPI scan, these'
                  + ' pixels will be removed from the coordinates list.')
        coord_lstnd[:] = [j for i, j in enumerate(coord_lstnd)
                          if i not in fgates]
        coord_lstnd[:] = [j for i, j in enumerate(coord_lstnd)
                          if i not in frays]
        # Creates a shapelike array to store the manual classification
        dict1occ = list(intradvars.values())[0]

        nanarr = np.full(dict1occ.shape, np.nan)
        for i in coord_lstnd:
            nanarr[i[0], i[1]] = i[2]
        rdataobj = {}
        rdataobj['manual_class'] = nanarr
        rdataobj['coord_list'] = coord_lstnd
        if min_snr is not None:
            rdataobj['min_snr'] = min_snr
        if rsite is not None:
            rdataobj['rsite'] = rsite

        fname = file_name[file_name.rfind('/')+1:]
        fnamec = file_name
        rdataobj['file_name'] = fnamec

        if dir2save:
            with open(dir2save+fname+'.tpmc', 'wb') as handle:
                pickle.dump(rdataobj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('A binary file was created at '+dir2save+fname+'.tpmc')


def ppi_base(rad_georef, rad_params, rad_vars, var2plot=None, coord_sys='polar',
             mlyr=None, vars_bounds=None, ucmap=None, unorm=None, cbticks=None,
             cb_ext=None, ppi_xlims=None, ppi_ylims=None, radial_xlims=None,
             radial_ylims=None, fig_size=None):
    """
    Create the base display for the interactive PPI explorer.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_vars : dict
        Dict containing radar variables to plot.
    var2plot : str, optional
        Key of the radar variable to plot. The default is None. This option
        will plot ZH or the 'first' element in the rad_vars dict.
    coord_sys : 'rect' or 'polar', optional
        Coordinates system (polar or rectangular). The default is 'polar'.
    mlyr : MeltingLayer Class, optional
        Plot the melting layer height. ml_top (float, int, list or np.array)
        and ml_bottom (float, int, list or np.array) must be explicitly
        defined. The default is None.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    cbticks : dict, optional
        Modifies the default ticks' location (dict values) and labels
        (dict keys) in the colour bar.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    ppi_xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max] in the PPI. The default is None.
    ppi_ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max] in the PPI. The default is None.
    radial_xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max] in the PPI. The default is None.
    radial_ylims : dict containing key and 2-element tuple or list, optional
        Set the y-axis view limits [min, max] in the radial variables. Key must
        be in rad_vars dict. The default is None.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']}"
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{2}} deg."
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    dtdes0 = f"{rad_params['site_name']}"

    nangle = 1
    nrange = 1
    # global figradint, intradgs, f3_axvar2plot, f3_axhbeam, heightbeamint
    global figradint, intradgs, f3_axvar2plot, f3_axhbeam, gcoord_sys
    global intradgeoref, intradparams, intradvars, intradaxs
    global mrv, vars_xlim, vars_ylim, rbeamh_b, rbeamh_t, gflat_coords

    if 'beambottom_height [km]' in rad_georef and isinstance(rad_georef['beambottom_height [km]'], np.ndarray):
        rbeamh_b = rad_georef['beambottom_height [km]']
    if 'beamtop_height [km]' in rad_georef and isinstance(rad_georef['beamtop_height [km]'], np.ndarray):
        rbeamh_t = rad_georef['beamtop_height [km]']
    if mlyr is not None:
        if isinstance(mlyr.ml_top, (int, float)):
            mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_top, (np.ndarray, list, tuple)):
            mlt_idx = [rut.find_nearest(nbh, mlyr.ml_top[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        if isinstance(mlyr.ml_bottom, (int, float)):
            mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom)
                       for nbh in rad_georef['beam_height [km]']]
        elif isinstance(mlyr.ml_bottom, (np.ndarray, list, tuple)):
            mlb_idx = [rut.find_nearest(nbh, mlyr.ml_bottom[cnt])
                       for cnt, nbh in
                       enumerate(rad_georef['beam_height [km]'])]
        mlt_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                             for cnt, ix in enumerate(mlt_idx)])
        mlt_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                             for cnt, ix in enumerate(mlt_idx)])
        mlb_idxx = np.array([rad_georef['grid_rectx'][cnt, ix]
                             for cnt, ix in enumerate(mlb_idx)])
        mlb_idxy = np.array([rad_georef['grid_recty'][cnt, ix]
                             for cnt, ix in enumerate(mlb_idx)])

    gcoord_sys = coord_sys
    intradgeoref, intradparams, intradvars = rad_georef, rad_params, rad_vars

    if radial_xlims is not None:
        vars_xlim = radial_xlims
    else:
        vars_xlim = None
    if radial_ylims is not None:
        vars_ylim = radial_ylims
    else:
        vars_ylim = None
    if fig_size is None:
        fig_size = (16, 9)
    figradint = plt.figure(figsize=fig_size)
    # plt.tight_layout()
    if len(rad_vars) > 3:
        intradgs = figradint.add_gridspec(len(rad_vars), 4)
    else:
        intradgs = figradint.add_gridspec(len(rad_vars)+3, 4)
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        var2plot, rad_vars.keys(), vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    if var2plot is None:
        var2plot = v2p
    polradv = var2plot
    cmapp = cmaph.get(polradv[polradv.find('['):],
                      mpl.colormaps['tpylsc_rad_pvars'])
    mrv = rad_vars[polradv]
    plotunits = [i[i.find('['):] for i in rad_vars.keys() if polradv == i][0]
    vnamettle = [i[:i.find('[')-1] for i in rad_vars.keys() if polradv == i][0]
    ptitle = f"{dtdes0} -- {dtdes2} \n" + dtdes1 + f' [{vnamettle}]'

    if gcoord_sys == 'rect':
        print('\n \n \n'
              ' ============================================================\n'
              '  Right-click on a pixel within the PPI to select its \n'
              '  azimuth or use the n and m keys to browse through the next \n'
              '  and previous azimuth.                                      \n'
              '  Radial profiles of polarimetric variables will be shown at \n'
              '  the axes on the right.                                     \n'
              '  Press a number (0-9) to store the coordinates and value    \n'
              '  of the current position of the mouse pointer.              \n'
              '  These coordinate can be retrieved at                       \n'
              '  ppiexplorer.keycoords                                    \n'
              ' =============================================================')
        # if coastl is not True:
        global x_center, y_center
        x_center = rad_georef['grid_rectx'].mean()
        y_center = rad_georef['grid_recty'].mean()
        gflat_coords = [[j for j in i]
                        for i in zip(rad_georef['grid_rectx'].flat,
                                     rad_georef['grid_recty'].flat)]
        f3_axvar2plot = figradint.add_subplot(intradgs[0:-1, 0:2])
        f3rec = f3_axvar2plot.pcolormesh(rad_georef['grid_rectx'],
                                         rad_georef['grid_recty'],
                                         mrv, shading='auto',
                                         cmap=cmapp, norm=normp)
        if cbticks is not None:
            clb = plt.colorbar(
                f3rec, ax=f3_axvar2plot, ticks=list(cbticks.values()),
                format=mticker.FixedFormatter(list(cbticks.keys())))
            clb.ax.tick_params(direction='in', labelsize=12, rotation=90)
        else:
            clb = plt.colorbar(
                mpl.cm.ScalarMappable(norm=normp, cmap=cmapp),
                ax=f3_axvar2plot, format=f'%.{cbtks_fmt}f', ticks=tcks)
            clb.ax.tick_params(direction='in', labelsize=12)
        f3_axvar2plot.axes.set_aspect('equal')
        clb.ax.set_title(plotunits, fontsize=14)
        f3_axvar2plot.format_coord = format_coord
        if mlyr is not None:
            f3_axvar2plot.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                               path_effects=[pe.Stroke(linewidth=5,
                                                       foreground='w'),
                                             pe.Normal()],
                               label=r'$MLyr_{(T)}$')
            f3_axvar2plot.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                               path_effects=[pe.Stroke(linewidth=5,
                                                       foreground='w'),
                                             pe.Normal()],
                               label=r'$MLyr_{(B)}$')
            f3_axvar2plot.legend(loc='upper right')
        # else:
        #     prs = ccrs.PlateCarree()
        #     f3_axvar2plot = figradint.add_subplot(intradgs[0:-1, 0:2],
        #                                           projection=prs)
        #     f3_axvar2plot.set_extent([-10.5, 3.5, 60, 49],
        #                              crs=ccrs.PlateCarree())
        #     f3_axvar2plot.coastlines()
        #     f3_axvar2plot.pcolormesh(rad_georef['grid_rectx']*1000,
        #                              rad_georef['grid_recty']*1000,
        #                              mrv, shading='auto',
        #                              cmap=cmaph, norm=normp,
        #                              transform=ccrs.OSGB(approx=False))
        #     f3_axvar2plot.gridlines(draw_labels=True, dms=False,
        #                             x_inline=False, y_inline=False)
        #     plt.colorbar(mpl.cm.ScalarMappable(norm=normp, cmap=cmaph),
        #                  ax=f3_axvar2plot)

        # 050822
        if ppi_xlims is not None:
            f3_axvar2plot.set_xlim(ppi_xlims)
        if ppi_ylims is not None:
            f3_axvar2plot.set_ylim(ppi_ylims)
        f3_axvar2plot.tick_params(axis='both', labelsize=12)
        f3_axvar2plot.set_xlabel('Distance from the radar [km]', fontsize=14)
        f3_axvar2plot.set_ylabel('Distance from the radar [km]', fontsize=14)

    if gcoord_sys == 'polar':
        # TODO add ML visualisation
        print('\n \n \n'
              ' ============================================================\n'
              '  Left-click on a pixel within the PPI to select its azimuth \n'
              '  or use the n and m keys to browse through the next and     \n'
              '  previous azimuth. \n'
              '  Radial profiles of polarimetric variables will be shown at \n'
              '  the axes on the right. \n'
              '  Press a number (0-9) to store the coordinates and value    \n'
              '  of the current position of the mouse pointer.              \n'
              '  These coordinate can be retrieved at                       \n'
              '  ppiexplorer.keycoords                                    \n'
              ' =============================================================')
        f3_axvar2plot = figradint.add_subplot(intradgs[0:-1, 0:2],
                                              projection='polar')
        f3pol = f3_axvar2plot.pcolormesh(
            *np.meshgrid(rad_georef['azim [rad]'],
                         rad_georef['range [m]'] / 1000, indexing='ij'),
            np.flipud(mrv), shading='auto', cmap=cmapp, norm=normp)
        if cbticks is not None:
            clb = plt.colorbar(
                f3pol, ax=f3_axvar2plot, ticks=list(cbticks.values()),
                format=mticker.FixedFormatter(list(cbticks.keys())))
        else:
            clb = plt.colorbar(mpl.cm.ScalarMappable(norm=normp, cmap=cmapp),
                               ax=f3_axvar2plot)
        f3_axvar2plot.grid(color='gray', linestyle=':')
        f3_axvar2plot.set_theta_zero_location('N')
        f3_axvar2plot.set_thetagrids(np.arange(nangle, nangle+2))
        f3_axvar2plot.xaxis.grid(ls='-')
        if not isinstance(rad_params['elev_ang [deg]'],
                          str) and rad_params['elev_ang [deg]'] < 89:
            plt.rgrids(np.arange(0, (rad_georef['range [m]'][-1]/1000)*1.1,
                                 5 * round((rad_georef['range [m]'][-1]/1000)
                                           / 25)),
                       angle=90)
        f3_axvar2plot.set_xticklabels([])
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # fc, ec = 'w', 'k'
    # f3_axvar2plot.annotate('| Created using Towerpy |', xy=(0.175, .03),
    #                          fontsize=8, xycoords='axes fraction',
    #                          va='center', ha='center',
    #                          bbox=dict(boxstyle=txtboxs,
    #                                    fc=fc, ec=ec))

    f3_axvar2plot.set_title(ptitle, fontsize=16)
    f3_axvar2plot.grid(True)

    f3_axhbeam = figradint.add_subplot(intradgs[-1:, 0:2])
    f3_axhbeam.set_xlabel('Range [Km]', fontsize=14)
    f3_axhbeam.set_ylabel('Beam height [Km]', fontsize=14)
    f3_axhbeam.tick_params(axis='both', labelsize=12)

    intradaxs = {f'f3_ax{i+2}': figradint.add_subplot(intradgs[i:i+1, 2:],
                                                      sharex=f3_axhbeam)
                 for i, j in enumerate(rad_vars)}

    for i in intradaxs:
        if i != list(intradaxs)[-1]:
            intradaxs[i].get_xaxis().set_visible(False)
            intradaxs[i].tick_params(axis='y', labelsize=12)

    intradaxs[list(intradaxs)[-1]].set_xlabel('Range [Km]', fontsize=14)
    intradaxs[list(intradaxs)[-1]].tick_params(axis='both', labelsize=12)
    plt.tight_layout()


class HTI_Int:
    """A class to create an interactive HTI plot."""

    def __init__(self):
        figprofsint.canvas.mpl_connect('button_press_event', self.on_pick)
        # figprofsint.canvas.mpl_connect('hzfunc', self.hzfunc)
        self.lastind = 0

    def hzfunc(self, label):
        """
        Update the right panel of the interactive HTI plot.

        Parameters
        ----------
        label : str
            Name of the radar variable.

        """
        if isinstance(self.lastind, int):
            return
        hviax.cla()
        idxdt = self.lastind[0]
        hviax.plot(intpvars[label][:, idxdt], intheight[:, idxdt], lw=5,
                   label='Profile')
        if intstats is not None and label in intstats:
            hviax.fill_betweenx(intheight[:, idxdt], intpvars[label][:, idxdt]
                                + intstats[label][:, idxdt],
                                intpvars[label][:, idxdt]
                                - intstats[label][:, idxdt], alpha=0.4,
                                label='std')
        if mlyrt is not None:
            hviax.axhline(mlyrt[idxdt], c='k', ls='dashed', lw=2, alpha=.75,
                          label='$MLyr_{(T)}$')
        if mlyrb is not None:
            hviax.axhline(mlyrb[idxdt], c='gray', ls='dashed', lw=2, alpha=.75,
                          label='$MLyr_{(B)}$')
        handles, labels = hviax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        hviax.legend(by_label.values(), by_label.keys(), fontsize=16,
                     loc='upper left')
        hviax.grid(axis='both')
        hviax.set_title(f"{intscdt[idxdt]:%Y-%m-%d %H:%M:%S}", fontsize=24)
        hviax.set_xlabel(label, fontsize=24, labelpad=15)
        figprofsint.canvas.draw()

    def on_pick(self, event):
        """
        Get the click locations.

        Parameters
        ----------
        event : Mouse click
            Right click from the mouse.

        """
        if event.button is MouseButton.RIGHT:
            if event.inaxes != htiplt:
                return True
            tz = ZoneInfo(tzi)
            # tms = mdates.num2date(event.xdata).replace(tzinfo=tz).timestamp()
            tms = mdates.num2date(event.xdata).timestamp()
            idxdate = rut.find_nearest(profsdtn, tms)
            yheight = event.ydata
            cdt = [idxdate, yheight]
            print(f'{intscdt[idxdate]:%Y-%m-%d %H:%M:%S}',
                  f'height {yheight:.3f}')
            self.lastind = cdt
            self.update()

    def update(self):
        """Update the HTI plot."""
        if self.lastind is None:
            return
        idxdt = self.lastind[0]

        hviax.cla()
        hviax.plot(intpvars[ppvar][:, idxdt], intheight[:, idxdt], lw=5,
                   label='Profile')
        if statsname == 'std_dev':
            hviax.fill_betweenx(intheight[:, idxdt], intpvars[ppvar][:, idxdt]
                                + intstats[ppvar][:, idxdt],
                                intpvars[ppvar][:, idxdt]
                                - intstats[ppvar][:, idxdt], alpha=0.4,
                                label='std')
        elif statsname == 'sem':
            hviax.fill_betweenx(intheight[:, idxdt], intpvars[ppvar][:, idxdt]
                                + intstats[ppvar][:, idxdt],
                                intpvars[ppvar][:, idxdt]
                                - intstats[ppvar][:, idxdt], alpha=0.4,
                                label='SEM')
        elif statsname == 'min' or statsname == 'max':
            hviax.fill_betweenx(intheight[:, idxdt], intpvars[ppvar][:, idxdt]
                                + intstats[ppvar][:, idxdt],
                                intpvars[ppvar][:, idxdt]
                                - intstats[ppvar][:, idxdt], alpha=0.4,
                                label='min/max')
        if mlyrt is not None:
            hviax.axhline(mlyrt[idxdt], c='k', ls='dashed', lw=2, alpha=.75,
                          label='$MLyr_{(T)}$')
        if mlyrb is not None:
            hviax.axhline(mlyrb[idxdt], c='gray', ls='dashed', lw=2, alpha=.75,
                          label='$MLyr_{(B)}$')
        handles, labels = hviax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        hviax.legend(by_label.values(), by_label.keys(), fontsize=16,
                     loc='upper left')
        hviax.grid(axis='both')
        hviax.set_title(f"{intscdt[idxdt]:%Y-%m-%d %H:%M:%S}", fontsize=24)
        hviax.set_xlabel(ppvar, fontsize=24, labelpad=15)
        # hviax.set_xlim(35, 55)
        figprofsint.canvas.draw()


def hti_base(pol_profs, mlyrs=None, stats=None, var2plot=None, ptype='pseudo',
             vars_bounds=None, contourl=None, ucmap=None, unorm=None,
             htixlim=None, htiylim=None, cbticks=None, cb_ext=None,
             fig_size=None, tz='Europe/London'):
    """
    Create the base display for the HTI.

    Parameters
    ----------
    pol_profs : list
        List of PolarimetricProfiles objects.
    mlyrs : list, optional
        List of MeltingLayer objects. The default is None.
    stats : str, optional
        Profiles statistic to plot in the right panel of the HTI plot.
        The default is None.
    var2plot : str, optional
        Key of the radar variable to plot. The default is None. This option
        will plot ZH or the 'first' element in the rad_vars dict.
    ptype : str, 'pseudo' or 'fcontour'
        Create a pseudocolor or filled contours plot.
        The default is 'pseudo'.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped.
    contourl : str, optional
        Draw contour lines of the specified radar variable.
        The default is None.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    htixlim : 2-element tuple or list of datetime objects, optional
        Set the x-axis view limits [min, max]. The default is None.
    htiylim : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    cbticks : dict, optional
        Modifies the default ticks' location (dict values) and labels
        (dict keys) in the colour bar.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    tz : str
        Key/name of the radar data timezone. The given tz string is then
        retrieved from the ZoneInfo module. Default is 'Europe/London'

    Returns
    -------
    radio : widget
        A MPL radio button.

    """
    profsheight = np.array([nprof.georef['profiles_height [km]']
                            for nprof in pol_profs]).T
    if pol_profs[0].profs_type.lower() == 'rd-qvps':
        profsdt = [nprof.scandatetime[0] for nprof in pol_profs]
    else:
        profsdt = [nprof.scandatetime for nprof in pol_profs]
    if pol_profs[0].profs_type.lower() == 'vps':
        profsvars = {k: np.array([nprof.vps[k] for nprof in pol_profs]).T
                     for k in pol_profs[0].vps.keys()}
        if stats == 'std_dev' or stats == 'sem':
            profsstat = {k: np.array([nprof.vps_stats[stats][k]
                                      for nprof in pol_profs]).T
                         for k in pol_profs[0].vps_stats[stats].keys()}
        else:
            profsstat = None
    elif pol_profs[0].profs_type.lower() == 'qvps':
        profsvars = {k: np.array([nprof.qvps[k] for nprof in pol_profs]).T
                     for k in pol_profs[0].qvps.keys()}
        # TODO add max/min visualisation
        if stats == 'std_dev' or stats == 'sem':
            profsstat = {k: np.array([nprof.qvps_stats[stats][k]
                                      for nprof in pol_profs]).T
                         for k in pol_profs[0].qvps_stats[stats].keys()}
        else:
            profsstat = None
    elif pol_profs[0].profs_type.lower() == 'rd-qvps':
        profsvars = {k: np.array([nprof.rd_qvps[k] for nprof in pol_profs]).T
                     for k in pol_profs[0].rd_qvps.keys()}
        profsstat = None
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        var2plot, profsvars, vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    if var2plot is None:
        var2plot = v2p
    prflv = var2plot
    cmapp = cmaph.get(prflv[prflv.find('['):],
                      mpl.colormaps['tpylsc_rad_pvars'])
    if mlyrs:
        mlyrtop = [mlyr.ml_top if isinstance(mlyr.ml_top, float) else np.nan
                   for mlyr in mlyrs]
        mlyrbot = [mlyr.ml_bottom if isinstance(mlyr.ml_bottom, float)
                   else np.nan for mlyr in mlyrs]
    else:
        mlyrtop = None
        mlyrbot = None
    # plotunits = [i[i.find('['):]
    #              for i in profsvars.keys() if prflv == i][0]
    plotunits = prflv[prflv.find('['):]
    plotvname = [i[:i.find('[')-1]
                 for i in profsvars.keys() if prflv == i][0]

    # -------------------------------------------------------------------------
    fontsizelabels = 24
    fontsizetick = 20
    linec, lwid = 'k', 3
    ptitle = f"{profsdt[0]:%Y-%m-%d %H:%M:%S}"
    # -------------------------------------------------------------------------

    global figprofsint, htiplt, hviax, profsdtn, intpvars, intheight, intscdt
    global intstats, ppvar, statsname, radio, mlyrt, mlyrb, tzi

    profsdtn = [dt.datetime.timestamp(dtp) for dtp in profsdt]

    intpvars, intheight, intscdt = profsvars, profsheight, profsdt
    intstats, ppvar, statsname = profsstat, prflv, stats
    # if mlyrs:
    mlyrt, mlyrb = mlyrtop, mlyrbot

    tzi = tz
    if fig_size is None:
        fig_size = (16, 9)
    figprofsint, axd = plt.subplot_mosaic(
        """
        AAAB
        """,
        figsize=fig_size)

    htiplt = axd['A']
    if ptype is None or ptype == 'pseudo':
        htiplt.pcolormesh(profsdt, profsheight, profsvars[prflv],
                          shading='auto', cmap=cmapp, norm=normp)
    elif ptype == 'fcontour':
        htiplt.contourf(profsdt, profsheight[:, 0], profsvars[prflv],
                        shading='auto', cmap=cmapp, norm=normp,
                        levels=normp.boundaries)
    else:
        raise TowerpyError('Oops!... Check the selected plot type')
    if contourl is not None:
        contourlp = htiplt.contour(
            profsdt, profsheight[:, 0], profsvars[contourl], colors='k',
            levels=bnd.get(contourl[contourl.find('['):]), alpha=0.4,
            zorder=10)
        htiplt.clabel(contourlp, inline=True, fontsize=8)
    if mlyrt is not None:
        htiplt.plot(profsdt, mlyrt, lw=lwid, c=linec, ls='--',
                    path_effects=[pe.Stroke(linewidth=7, foreground='w'),
                                  pe.Normal()], label=r'$MLyr_{(T)}$')
        htiplt.scatter(profsdt, mlyrt, lw=2, s=3, c=linec)
    if mlyrb is not None:
        htiplt.plot(profsdt, mlyrb, lw=lwid, c='grey', ls='--',
                    path_effects=[pe.Stroke(linewidth=7, foreground='w'),
                                  pe.Normal()], label=r'$MLyr_{(B)}$')
        htiplt.scatter(profsdt, mlyrb, lw=2, s=3, c='grey')
    # plot_bbh = True
    # if plot_bbh:
    #     htiplt.plot(profsdt, [i.bb_peakh for i in mlyrs])
    if htixlim is not None:
        htiplt.set_xlim(htixlim)
    if htiylim is not None:
        htiplt.set_ylim(htiylim)
    htiplt.grid(True)

    ax1_divider = make_axes_locatable(htiplt)
    cax1 = ax1_divider.append_axes("top", size="10%", pad="7%")
    if cbticks is not None:
        cb1 = figprofsint.colorbar(
            mpl.cm.ScalarMappable(norm=normp, cmap=cmapp), ax=htiplt, cax=cax1,
            orientation="horizontal", ticks=list(cbticks.values()),
            format=mticker.FixedFormatter(list(cbticks.keys())),
            extend='neither')
        cb1.ax.tick_params(length=0, direction='in', labelsize=20)
    else:
        if '[-]' in prflv:
            cb1 = figprofsint.colorbar(
                mpl.cm.ScalarMappable(norm=normp, cmap=cmapp), ax=htiplt,
                ticks=tcks, format=f'%.{cbtks_fmt}f', cax=cax1,
                orientation="horizontal")
        else:
            cb1 = figprofsint.colorbar(
                mpl.cm.ScalarMappable(norm=normp, cmap=cmapp), ax=htiplt,
                cax=cax1, format=f'%.{cbtks_fmt}f',
                orientation="horizontal")
        cb1.ax.tick_params(direction='in', labelsize=20)
    cb1.ax.set_ylabel(f'{plotvname} \n {plotunits}', fontsize=15, labelpad=50)
    cax1.xaxis.set_ticks_position("top")
    htiplt.tick_params(axis='both', direction='in', labelsize=fontsizetick,
                       pad=10)
    htiplt.set_xlabel('Date and Time', fontsize=fontsizelabels, labelpad=15)
    htiplt.set_ylabel('Height [km]', fontsize=fontsizelabels, labelpad=15)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=13)
    formatter = mdates.ConciseDateFormatter(locator, tz=tzi)
    htiplt.xaxis.set_major_locator(locator)
    htiplt.xaxis.set_major_formatter(formatter)
    htiplt.xaxis.get_offset_text().set_size(20)
    # mpl.rcParams['timezone'] = tz
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # fc, ec = 'w', 'k'
    # htiplt.annotate('| Created using Towerpy |', xy=(0.02, -.1), fontsize=8,
    #                 xycoords='axes fraction', va='center', ha='center',
    #                 bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))

    hviax = axd['B']
    hviax.grid(axis='both')
    hviax.sharey(htiplt)
    hviax.tick_params(axis='both', direction='in', labelsize=fontsizetick,
                      pad=10)
    hviax.yaxis.set_tick_params(labelbottom=False)
    hviax.set_title(ptitle, fontsize=fontsizelabels)
    hviax.set_xlabel(prflv, fontsize=fontsizelabels, labelpad=15)

    ax2_divider = make_axes_locatable(hviax)
    cax2 = ax2_divider.append_axes("top", size="10%", pad="7%",
                                   facecolor='lightsteelblue')
    # cax2.remove()
    radio = RadioButtons(cax2, tuple(intpvars.keys()), activecolor='gold',
                         active=list(intpvars.keys()).index(f'{prflv}'),
                         radio_props={'s': [45]})
    for txtr in radio.labels:
        txtr.set_fontsize(9)
    plt.tight_layout()
    plt.show()

    return radio


def ml_detectionvis(hbeam, profzh_norm, profrhv_norm, profcombzh_rhv,
                    pkscombzh_rhv, comb_mult, comb_mult_w, comb_idpy, mlrand,
                    min_hidx, max_hidx, param_k, idxml_btm_it1, idxml_top_it1):
    """Create an interactive plot for the ml_detection function."""
    lbl_fs = 24
    lgn_fs = 16
    tks_fs = 20
    lw = 4

    heightbeam = hbeam
    init_comb = comb_idpy
    hb_lim_it1 = heightbeam[idxml_btm_it1:idxml_top_it1]
    if comb_mult_w:
        # resimp1d = np.gradient(comb_mult_w[comb_idpy])
        resimp2d = np.gradient(np.gradient(comb_mult_w[comb_idpy]))

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(14, 10))
    plt.subplots_adjust(left=0.096, right=0.934, top=0.986, bottom=0.091)

    # =============================================================================
    ax1 = axs[0]
    # =============================================================================
    ax1.plot(profzh_norm[min_hidx:max_hidx], heightbeam[min_hidx:max_hidx],
             label=r'$Z^{*}_{H}$', lw=1.5, c='tab:purple')
    ax1.plot(profrhv_norm[min_hidx:max_hidx], heightbeam[min_hidx:max_hidx],
             label=r'$1- \rho^{*}_{HV}$', lw=1.5, c='tab:red')
    ax1.plot(profcombzh_rhv, heightbeam[min_hidx:max_hidx],
             label=r'$P_{comb}$', lw=3, c='tab:blue')
    if ~np.isnan(pkscombzh_rhv['idxmax']):
        ax1.scatter(profcombzh_rhv[pkscombzh_rhv['idxmax']],
                    heightbeam[min_hidx:max_hidx][pkscombzh_rhv['idxmax']],
                    s=300, marker="X", c='tab:orange', label='$P_{{peak}}$')
    ax1.axvline(param_k, c='k', ls=':', lw=2.5, label='k')
    ax1.set_xlim([-0.05, 1.05])
    # ax1.set_ylim([heightbeam[min_hidx], heightbeam[max_hidx]])
    ax1.set_ylim([0, heightbeam[max_hidx]])
    ax1.tick_params(axis='both', labelsize=tks_fs)
    ax1.set_xlabel('(norm)', fontsize=lbl_fs, labelpad=10)
    ax1.set_ylabel('Height [km]', fontsize=lbl_fs, labelpad=10)
    ax1.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    ax1.legend(fontsize=lgn_fs, loc='upper right')
    ax1.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(True)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("top", size="10%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.set_facecolor('slategrey')
    at = AnchoredText('Initial identification of the \n' +
                      'Melting Layer signatures \n'
                      'combining the normalised \n' +
                      r'profiles of $Z_H$ and $\rho_{HV}$', loc='center',
                      prop=dict(size=12, color='white'), frameon=False)
    cax.add_artist(at)

    # =============================================================================
    ax2 = axs[1]
    # =============================================================================
    ac = 0.7
    if comb_mult:
        ax2.plot(comb_mult[comb_idpy], hb_lim_it1,
                 label=f'$P^*_{{{comb_idpy+1}}}$', lw=1.5, c='tab:blue')
        # ax2.plot(resimp1d, hb_lim_it1,
        #           label=f"$P_{{{comb_idpy+1}}}^*'$", lw=3., c='tab:gray',
        #           alpha=ac)
        ax2.plot(-resimp2d, hb_lim_it1,
                 label=f"$-P_{{{comb_idpy+1}}}^*''$", lw=3., c='gold', alpha=ac)
        ax2.plot(comb_mult_w[comb_idpy], hb_lim_it1,
                 # label=(f'$P^*_{{{comb_idpy+1}}}$-'+r'(w $\cdot$'
                 #        + f"$P_{{{comb_idpy+1}}}^*''$)"),
                 label=f'$P_{{{comb_idpy+1}}}$',
                 lw=3., c='tab:green', alpha=ac)
    if comb_mult_w:        
        if ~np.isnan(mlrand[comb_idpy]['idxtop']):
            ax2.scatter(comb_mult_w[comb_idpy][mlrand[comb_idpy]['idxtop']],
                        hb_lim_it1[mlrand[comb_idpy]['idxtop']],
                        s=300, marker='*', c='deeppink', alpha=0.5,
                        label=f"$P_{{{comb_idpy+1}(top)}}$")
        if ~np.isnan(mlrand[comb_idpy]['idxmax']):
            ax2.scatter(comb_mult_w[comb_idpy][mlrand[comb_idpy]['idxmax']],
                        hb_lim_it1[mlrand[comb_idpy]['idxmax']],
                        s=300, marker="X", c='tab:orange', alpha=0.5,
                        label=f"$P_{{{comb_idpy+1}(peak)}}$")
        if ~np.isnan(mlrand[comb_idpy]['idxbot']):
            ax2.scatter(comb_mult_w[comb_idpy][mlrand[comb_idpy]['idxbot']],
                        hb_lim_it1[mlrand[comb_idpy]['idxbot']],
                        s=300, marker='o', c='deeppink', alpha=0.5,
                        label=f'$P_{{{comb_idpy+1}(bottom)}}$')
    ax2.tick_params(axis='x', labelsize=tks_fs)
    ax2.set_xlabel('(norm)', fontsize=lbl_fs, labelpad=10)
    ax2.tick_params(axis='both', labelsize=tks_fs)
    ax2.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    ax2.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.legend(fontsize=lgn_fs)
    ax2.grid(True)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("top", size="10%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.set_facecolor('slategrey')
    at = AnchoredText('Detection of the ML boundaries \n' +
                      'for a given combination of\n' +
                      'polarimetric profiles.\n', loc='center',
                      # 'The 1st and 2nd derivative are \n' +
                      # ' also shown 10.5194/amt-14-2873-2021',
                      prop=dict(size=12, color='white'), frameon=False)
    cax.add_artist(at)

    # =============================================================================
    ax3 = axs[2]
    # =============================================================================
    ax3.tick_params(axis='x', labelsize=tks_fs)
    ax3.set_xlabel('(norm)', fontsize=lbl_fs, labelpad=10)
    if comb_mult_w:        
        # Create the figure and the line that we will manipulate
        for i in range(0, len(comb_mult)):
            ax3.plot(comb_mult_w[i], hb_lim_it1, c='silver',
                     lw=2, alpha=.4, zorder=0)
            if ~np.isnan(mlrand[i]['idxtop']):
                ax3.scatter(comb_mult_w[i][mlrand[i]['idxtop']],
                            hb_lim_it1[mlrand[i]['idxtop']],
                            s=100, marker='*', c='silver')
            if ~np.isnan(mlrand[i]['idxbot']):
                ax3.scatter(comb_mult_w[i][mlrand[i]['idxbot']],
                            hb_lim_it1[mlrand[i]['idxbot']],
                            s=100, marker='o', c='silver')
    
        line, = ax3.plot(comb_mult_w[init_comb], hb_lim_it1, lw=lw, c='tab:green')
        if ~np.isnan(mlrand[init_comb]['idxtop']):
            mlts = ax3.axhline(hb_lim_it1[mlrand[init_comb]['idxtop']],
                               c='slateblue', ls='dashed', lw=lw, alpha=0.5,
                               label=r'$MLyr_{(T)}$')
        if ~np.isnan(mlrand[init_comb]['idxbot']):
            mlbs = ax3.axhline(hb_lim_it1[mlrand[init_comb]['idxbot']],
                               c='steelblue', ls='dashed', lw=lw, alpha=0.5,
                               label=r'$MLyr_{(B)}$')
    ax3.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    ax3.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax3.legend(fontsize=lgn_fs)
    ax3.grid(True)
    # ax3.set_xlim([-0.3, 1.5])

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("top", size="10%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.set_facecolor('slategrey')
    at = AnchoredText('Use the slider to assess the \n' +
                      'performance of each profile \n' +
                      'combination for detecting the ML.',
                      # 'The 1st and 2nd derivative are \n' +
                      # ' also shown 10.5194/amt-14-2873-2021',
                      loc='center',
                      prop=dict(size=12, color='white'), frameon=False)
    cax.add_artist(at)
    
    if comb_mult_w:
        ax_amp = plt.axes([0.95, 0.15, 0.0225, 0.63])
        # define the values to use for snapping
        allowed_combs = np.linspace(1, len(comb_mult_w),
                                    len(comb_mult_w)).astype(int)
        # create the sliders
        samp = Slider(ax_amp, "Comb", 1, len(comb_mult_w), valinit=init_comb+1,
                      valstep=allowed_combs, color="green", orientation="vertical")
    
        def comb_slider(val):
            amp = samp.val-1
            line.set_xdata(comb_mult_w[amp])
            if np.isfinite(mlrand[amp]['idxtop']):
                mlts.set_ydata((hb_lim_it1[mlrand[amp]['idxtop']],
                               hb_lim_it1[mlrand[amp]['idxtop']]))
            if np.isfinite(mlrand[amp]['idxbot']):
                mlbs.set_ydata((hb_lim_it1[mlrand[amp]['idxbot']],
                               hb_lim_it1[mlrand[amp]['idxbot']]))
            fig.canvas.draw_idle()
    
        samp.on_changed(comb_slider)
    plt.tight_layout()
    plt.show()


# =============================================================================
# %% xarray implementation
# =============================================================================

class PPI_Exp:
    """A class to create an interactive PPI plot."""

    # def __init__(self):
    def __init__(self, xrds, polarplot=False, coord_sys="polar"):
        self.xrds = xrds
        self.polarplot = polarplot
        self.coord_sys = coord_sys
        self.lastind = [0, 0]
        self.keycoords = []
        self.current_line = [None]
        self._ax_ppi = None
        self.text = None

    def _on_close(self, event):
        if not hasattr(self, "_cids"):
            return
        for cid in self._cids:
            self._fig.canvas.mpl_disconnect(cid)


    def format_coord(self, x, y):
        """Format coordinates for the PPI axis."""
        if self.coord_sys == "rect":
            # fall back if ppi_base hasn’t built the tree yet
            if not hasattr(self, "_tree"):
                return f"x={x:1.4f}, y={y:1.4f}"
    
            xy = np.array([[x, y]])
            _, index = self._tree.query(xy)
            index = int(np.ravel(index)[0])
            q, r = np.unravel_index(index, self._shape)

            da = self.xrds[self._var2plot]
            # bounds check
            if r < 0 or r >= da.shape[1]:
                return f"x={x:1.4f}, y={y:1.4f}"
            z = da.values[q, r]
            return f"x={x:1.4f}, y={y:1.4f}, z={z:1.4f} [{q},{r}]"
        if self.coord_sys == "polar":
            # fall back if ppi_base hasn’t built the arrays yet
            if not hasattr(self, "_az") or not hasattr(self, "_rng_km"):
                #  use the original formatter (polar axis)
                if hasattr(self, "_orig_format_coord"):
                    return self._orig_format_coord(x, y)
                return f"x={x:1.4f}, y={y:1.4f}"
            da = self.xrds[self._var2plot]
            # Convert x to degrees depending on polarplot mode
            if self.polarplot:
                # x is theta (rad) on a Matplotlib polar axis.
                angle_deg = np.rad2deg(x) % 360.
            else:
                # x is already azimuth in degrees on a Cartesian axis
                angle_deg = x
            # Nearest azimuth index using actual azimuth grid (deg)
            q = int(np.argmin(np.abs(self._az - angle_deg)))
            # Nearest range index using actual range grid (km)
            r = int(np.argmin(np.abs(self._rng_km - y)))
            if q < 0 or q >= da.shape[0] or r < 0 or r >= da.shape[1]:
                if hasattr(self, "_orig_format_coord") and self.polarplot:
                    return self._orig_format_coord(x, y)
                return f"x={x:1.4f}, y={y:1.4f}"
            z = da.values[q, r]
            # For polarplot=True, preserve Matplotlib's θ, r string and append z
            if self.polarplot and hasattr(self, "_orig_format_coord"):
                base = self._orig_format_coord(x, y)  # e.g. "θ=0.75π, r=50"
                return f"{base}, z={z:1.3f} [{q},{r}]"
            # For polar coord_sys but non-polar axis, give a simple text
            return (f"azi={self._az[q]:.2f}°, r={self._rng_km[r]:.2f} km, "
                    f"z={z:1.3f} [{q},{r}]")        
        # # for now, no special handling for cartopy
    
    def _nearest_gate(self, x, y):
        """
        Map a point in display coordinates to the nearest (azimuth_bin, range_bin).
    
        Parameters
        ----------
        x, y : float
            Coordinates in the PPI axis coordinate system. In rectangular mode these
            are Cartesian (x, y) values. In polar mode they represent either
            (theta, r) for polarplot=True or (azimuth_deg, range_km) for
            polarplot=False.
    
        Returns
        -------
        (q, r) : tuple of ints or (None, None)
            The nearest azimuth and range bin indices. Returns (None, None) if the
            lookup cannot be performed (e.g. missing lookup arrays).
        """
        # Rectangular mode
        if self.coord_sys == "rect":
            if not hasattr(self, "_tree"):
                return None, None
            xy = np.array([[x, y]])
            _, index = self._tree.query(xy)
            index = int(np.ravel(index)[0])
            q, r = np.unravel_index(index, self._shape)
            return int(q), int(r)
        # Polar mode
        if self.coord_sys == "polar":
            if not hasattr(self, "_az") or not hasattr(self, "_rng_km"):
                return None, None
            if self.polarplot:
                angle_deg = np.rad2deg(x) % 360.0
            else:
                angle_deg = x
            q = int(np.argmin(np.abs(self._az - angle_deg)))
            r = int(np.argmin(np.abs(self._rng_km - y)))
            return q, r
        return None, None

    def on_click(self, event):
        """
        Handle mouse clicks on the PPI axis.
    
        Behaviour depends on the active coordinate system:
        - Rectangular mode: right-click selects the nearest (ray, gate).
        - Polar mode, polarplot=True: left-click selects the nearest bin.
        - Polar mode, polarplot=False: right-click selects the nearest bin.
    
        The selected bin is stored in `self.lastind` and the display is updated.
        """
        if event.inaxes is not self._ax_ppi:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Rectangular mode
        if self.coord_sys == "rect":
            if event.button is not MouseButton.RIGHT:
                return
            q, r = self._nearest_gate(event.xdata, event.ydata)
            if q is None:
                return
            # print(f"azimuth index {q}, gate {r}")
            if self.current_line[0] is not None:
                self.current_line[0].remove()
            f3line, = self._ax_ppi.plot([self._x_center, event.xdata],
                                        [self._y_center, event.ydata],
                                        "ok:", lw=2, mfc="none", alpha=0.5)
            self.current_line[0] = f3line
            self.lastind = [q, r]
            self.update()
            return
        # Polar mode
        if self.coord_sys == "polar":
            if self.polarplot:
                if event.button is not MouseButton.LEFT:
                    return
            else:
                if event.button is not MouseButton.RIGHT:
                    return
            q, r = self._nearest_gate(event.xdata, event.ydata)
            if q is None:
                return
            # print(f"azimuth {self._az[q]:.2f}°, range {self._rng_km[r]:.2f} km")
            self.lastind = [q, r]
            self.update()
            return

    def on_numkey(self, event):
        """
        Handle numeric key presses (0–9) inside the PPI axis.
    
        Records the nearest (azimuth_bin, range_bin) at the cursor position and
        stores it in `self.keycoords` together with the pressed digit.
        """
        if not event.key.isdigit():
            return
        if event.inaxes is not self._ax_ppi:
            return
        if event.xdata is None or event.ydata is None:
            return
        q, r = self._nearest_gate(event.xdata, event.ydata)
        if q is None:
            return
        print(f"you pressed {event.key}: [angle={q}, range_bin={r}]")
        self.keycoords.append((q, r, int(event.key)))
    
    def browse_azimuth(self, event):
        """
        Step through azimuth bins using keyboard navigation.
    
        Keys
        ----
        'n' : move to the previous azimuth bin
        'm' : move to the next azimuth bin
    
        The selected azimuth index is updated in `self.lastind` and the display is
        refreshed.
        """
        if self.lastind is None:
            return
        if event.key not in ("n", "m"):
            return
        # Determine increment
        inc = -1 if event.key == "n" else 1
        # Current azimuth index
        q, r = self.lastind
        # New azimuth index
        q_new = q + inc
        # Clip to valid range
        q_new = int(np.clip(q_new, 0, len(self._az) - 1))
        # Remove previous indicator line if present
        if self.current_line[0] is not None:
            self.current_line[0].remove()
            self.current_line[0] = None
        # Rectangular mode
        if self.coord_sys == "rect":
            # Extract the ray in rect coordinates
            x_ray = self.xrds[self._coord_namex].values[q_new, :].ravel()
            y_ray = self.xrds[self._coord_namey].values[q_new, :].ravel()    
            # Draw the ray indicator
            f3line, = self._ax_ppi.plot(x_ray, y_ray, "ok:", lw=2, mfc="none",
                                        markevery=[0, -1], alpha=0.5)
            self.current_line[0] = f3line
        # Polar mode
        elif self.coord_sys == "polar":
            # Nothing to draw here — update() handles both polarplot modes
            pass
        # Store updated index
        self.lastind = [q_new, r]
        # Trigger full redraw
        self.update()

    def update(self):
        """
        Refresh all dependent panels after a new (azimuth_bin, range_bin) selection.
    
        This updates:
        - the PPI indicator (ray line, theta grid, or selection rectangle)
        - the beam-height panel
        - all radial profile panels
        - the status text

        The method assumes `self.lastind` contains a valid (q, r) pair.
        """
        if self.lastind is None:
            return
        if not hasattr(self, "_ax_beam") or not hasattr(self, "_ax_radials"):
            return
        if not hasattr(self, "_rng_km"):
            return

        q, r = self.lastind  # azimuth index, range index
        # =============================================================================
        # Update PPI azimuth indicator
        # =============================================================================
        if self.coord_sys == "polar" and self.polarplot:
            # Show thin azimuth gridlines around the selected angle
            azi_deg = self._az[q]
            # self._ax_ppi.set_thetagrids([azi_deg])
            self._ax_ppi.set_thetagrids(np.arange(azi_deg, azi_deg+2))
            self._ax_ppi.set_xticklabels([])
        # =============================================================================
        # Polar coord_sys, azimuth–range PPI (polarplot=False):
        # draw a rectangle enclosing the selected azimuth and range
        # =============================================================================
        if self.coord_sys == "polar" and not self.polarplot:
            # Remove previous lines
            if hasattr(self, "_ppi_lines"):
                for ln in self._ppi_lines:
                    ln.remove()
            self._ppi_lines = []
        
            azi_deg = self._az
            rng_km = self._rng_km
            # Compute bounds safely
            q0 = max(q - 1, 0)
            q1 = min(q + 1, len(azi_deg) - 1)
            r0 = max(r - 1, 0)
            r1 = min(r + 1, len(rng_km) - 1)
            # Vertical lines (azimuth bounds)
            ln1 = self._ax_ppi.axvline(azi_deg[q0], color="k", ls=":", lw=1.5,
                                       alpha=0.5)
            ln2 = self._ax_ppi.axvline(azi_deg[q1], color="k", ls=":", lw=1.5,
                                       alpha=0.5)
            # Horizontal lines (range bounds)
            ln3 = self._ax_ppi.axhline(rng_km[r0], color="k", ls=":", lw=1.5,
                                       alpha=0.5)
            ln4 = self._ax_ppi.axhline(rng_km[r1], color="k", ls=":", lw=1.5,
                                       alpha=0.5)
            # Store for removal next update
            self._ppi_lines = [ln1, ln2, ln3, ln4]
        # =============================================================================
        # Plot beam-height fields (all in km)
        # =============================================================================
        # Range coordinate (km)
        rng_km = self._rng_km    
        # =============================================================================
        # Clear beam-height axis
        # =============================================================================
        axb = self._ax_beam
        axb.cla()
        # Plot beam centre / top / bottom for selected azimuth q
        axb.plot(rng_km, self.beam_center.isel(azimuth=q), ls=":")
        axb.plot(rng_km, self.beam_top.isel(azimuth=q),    c="k")
        axb.plot(rng_km, self.beam_bottom.isel(azimuth=q), c="k")
        axb.set_xlabel(f"{self.xrds.range.attrs.get('standard_name', 'range')} [km]")
        axb.set_ylabel(f'{self.beam_center.attrs.get("short_name", "beam_height").lower()}'
                       f' [{self.beam_center.attrs.get("units", "km")}]')
        axb.grid()
        # Highlight selected gate
        if 0 <= r < self.beam_center.sizes["range"]:
            axb.axhline(self.beam_center.values[q, r], alpha=0.5, c="tab:red")
            axb.axvline(rng_km[r], alpha=0.5, c="tab:red")
        # =============================================================================
        # Update radial profile panels
        # =============================================================================
        for vname, ax in self._ax_radials.items():
            ax.cla()
            da = self.xrds[vname]  # (azimuth, range)
            # Skip variables that are not 2‑D polar fields
            if da.ndim != 2 or set(da.dims) != {"azimuth", "range"}:
                print(f"Variable {da.name} has no azimuth dimension — skipping profile extraction.")
                ax.set_title(f"{vname} [not 2‑D]")
                continue
            # Extract radial profile at azimuth q
            prof = da.isel(azimuth=q)
            ax.plot(rng_km, prof, marker=".", markersize=3)
            rpunits = _safe_units(self.xrds[vname])
            ax.set_title(f'{vname} [{rpunits}]')
            # Highlight selected gate
            if 0 <= r < prof.sizes["range"]:
                ax.axvline(rng_km[r], alpha=0.2)
            # Optional variable-specific lines
            if "zdr" in vname.lower():
                ax.axhline(0, alpha=0.2, c="gray")
            if vname.lower().startswith("rho"):
                ax.axhline(1.0, alpha=0.2, c="thistle")
        # Label last radial axis
        list(self._ax_radials.values())[-1].set_xlabel(
            f"{self.xrds.range.attrs.get('standard_name', 'range')} [km]")
        # =============================================================================
        # Update status text
        # =============================================================================
        if self.coord_sys == "polar":
            azi_deg = self._az[q]
            self.text.set_text(f"selected: az={azi_deg:.2f}°, gate={r}")
        else:
            self.text.set_text(f"selected: az_idx={q}, gate={r}")
        # =============================================================================
        # Redraw
        # =============================================================================
        self._fig.canvas.draw_idle()

    def ppi_base(self, var2plot=None, varnames=None, plot_colorbar=False,
                 polarcoord_names={"azi": "azimuth", "rng": "range"},
                 rectcoord_names={"x": "grid_rectx", "y": "grid_recty"},
                 ppi_xlims=None, ppi_ylims=None,
                 radial_xlims=None, radial_ylims=None, fig_size=None,
                 fig_title=None, **ppi_kwargs):
        """
        Constructs the interactive PPI display.

        Parameters
        ----------
        var2plot : str, optional
            Name of the variable to display in the PPI panel. If None, a suitable
            default is chosen (preferentially a reflectivity-like field).
        varnames : list of str, optional
            Variables to display in the radial-profile panels. If None, all data
            variables in the dataset are used.
        plot_colorbar : bool, optional
            Whether to add a colourbar to the PPI panel.
        polarcoord_names : dict, optional
            Mapping specifying the azimuth and range coordinate names in polar mode.
            Defaults to {"azi": "azimuth", "rng": "range"}.
        rectcoord_names : dict, optional
            Mapping specifying the x and y coordinate names in rectangular mode.
            Defaults to {"x": "grid_rectx", "y": "grid_recty"}.
        ppi_xlims, ppi_ylims : tuple, optional
            Axis limits for the PPI panel.
        radial_xlims, radial_ylims : tuple, optional
            Axis limits for the radial-profile panels.
        fig_size : tuple, optional
            Figure size in inches. Defaults to (16, 9).
        fig_title : str, optional
            Optional title for the PPI panel. If None, a default title is generated.
        **ppi_kwargs :
            Additional keyword arguments forwarded to
            :func:`towerpy.datavis.rad_display.plot_ppi_xr`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax_ppi : matplotlib.axes.Axes
            The PPI axis.
        ax_beam : matplotlib.axes.Axes
            The beam-height axis.
        ax_radials : dict of matplotlib.axes.Axes
            Dictionary mapping variable names to their radial-profile axes.

        Notes
        -----
        - This method attaches all interactive event handlers:
            * mouse clicks (`on_click`)
            * numeric key presses (`on_numkey`)
            * azimuth browsing (`browse_azimuth`)
        - Coordinate names are resolved dynamically, allowing datasets with
          non-standard naming conventions.
        - Lookup structures are initialised depending on the coordinate system:
            * rectangular mode: KDTree on (x, y)
            * polar mode: azimuth and range arrays
        """
        # =============================================================================
        # Figure + gridspec layout (PPI + beam + radials)
        # =============================================================================
        if fig_size is None:
            fig_size = (16, 9)
        fig = plt.figure(figsize=fig_size)
    
        if varnames is None:
            varnames = list(self.xrds.data_vars.keys())
        nvars = len(varnames)
        # Resolve variable2plot
        var2plot = _resolve_var2plot(self.xrds, var2plot)
        if nvars > 3:
            gs = fig.add_gridspec(nvars, 4)
        else:
            gs = fig.add_gridspec(nvars + 3, 4)
        # =============================================================================
        # Main PPI axis occupies left block
        # =============================================================================
        if self.coord_sys == "rect" and self.polarplot:
            warnings.warn("polarplot=True is incompatible with coord_sys='rect'. "
                          "Falling back to polarplot=False.")
            self.polarplot = False
        if self.polarplot:
            ax_ppi = fig.add_subplot(gs[0:-1, 0:2],
                                     projection='polar')
        else:
            ax_ppi = fig.add_subplot(gs[0:-1, 0:2])
        # Call plot_ppi to draw the PPI into ax_ppi
        mappable, ax_ppi = plot_ppi_xr(self.xrds, var2plot=var2plot,
                                    coord_sys=self.coord_sys, xlims=ppi_xlims,
                                    ylims=ppi_ylims, fig=fig, ax1=ax_ppi,
                                    fig_size=None, fig_title='', 
                                    polarcoord_names=polarcoord_names,
                                    rectcoord_names=rectcoord_names,
                                    polarplot=self.polarplot, 
                                    add_colorbar=False, **ppi_kwargs)
        # =============================================================================
        # Resolve coordinates
        # =============================================================================
        coord_names = rectcoord_names if self.coord_sys == "rect" else polarcoord_names
        coord_namex, coord_namey = resolve_rect_coords(self.xrds, coord_names)
        has_polar = {polarcoord_names["rng"], polarcoord_names["azi"]} <= set(self.xrds.coords)
        has_rect = (coord_namex is not None and coord_namey is not None
                    and {coord_namex, coord_namey} <= set(self.xrds.coords))
        self._has_polar = has_polar
        self._has_rect = has_rect
        self._coord_namex = coord_namex
        self._coord_namey = coord_namey
        if has_rect and self.coord_sys == "rect":
            rect_x = self.xrds[coord_namex].values.ravel()
            rect_y = self.xrds[coord_namey].values.ravel()
            self._tree = spatial.KDTree(np.column_stack([rect_x, rect_y]))
            self._shape = self.xrds[var2plot].shape
            self._var2plot = var2plot
            self._x_center = rect_x.mean()
            self._y_center = rect_y.mean()
        if has_polar:
            az_da = self.xrds[polarcoord_names["azi"]]
            az_units = az_da.attrs.get("units", "").lower()
            if az_units.startswith("rad"):
                self._az = np.rad2deg(az_da.values)
            else:
                self._az = az_da.values  # assume degrees
            # rng_da = _to_kilometers(self.xrds[polarcoord_names["rng"]])
            rng_da = convert(self.xrds[polarcoord_names["rng"]], "km")
            self._rng_km = rng_da.values
            self._var2plot = var2plot
        if not has_rect and not has_polar:
            raise ValueError("Dataset does not contain required coordinate variables.")
        # =============================================================================
        # Attach interactive coordinate formatter (xarray-aware version)
        # =============================================================================
        self._orig_format_coord = ax_ppi.format_coord
        # Save original formatter
        ax_ppi.format_coord = self.format_coord
        # =============================================================================
        # Beam-height axis (bottom left)
        # =============================================================================
        self.beam_center = self.xrds["beamc_height"] # (azimuth, range)
        self.beam_top = self.xrds["beamt_height"]
        self.beam_bottom = self.xrds["beamb_height"]
        ax_beam = fig.add_subplot(gs[-1:, 0:2])
        ax_beam.set_xlabel(f"{rng_da.attrs.get('standard_name', 'range')} [{rng_da.units}]")
        ax_beam.set_ylabel(
            f'{self.beam_center.attrs.get("short_name", "beam_height").lower()}'
            f' [{self.beam_center.attrs.get("units", "km")}]')
        # =============================================================================
        # Radial-profile axes (right column)
        # =============================================================================
        ax_radials = {}
        for i, vname in enumerate(varnames):
            ax = fig.add_subplot(gs[i:i+1, 2:], sharex=ax_beam)
            ax_radials[vname] = ax
        # =============================================================================
        # Store PPI axis for interaction
        # =============================================================================
        self._ax_ppi = ax_ppi    
        # Connect events
        cid1 = fig.canvas.mpl_connect("button_press_event", self.on_click)
        cid2 = fig.canvas.mpl_connect("key_press_event", self.browse_azimuth)
        cid3 = fig.canvas.mpl_connect("key_press_event", self.on_numkey)
        # Store IDs to disconnect later
        self._cids = (cid1, cid2, cid3)
        # Add status text in the PPI axis
        self.text = ax_ppi.text(0.01, 0.03, "selected: none",
                                transform=ax_ppi.transAxes, va="top")
        self._ax_beam = ax_beam
        self._ax_radials = ax_radials
        self._fig = fig
        self._fig.canvas.mpl_connect("close_event", self._on_close)
        # =============================================================================
        # Format gridspec layout
        # =============================================================================
        # Hide x-axis for all but last radial axis
        for name, ax in list(ax_radials.items())[:-1]:
            ax.get_xaxis().set_visible(False)
        # Configure last radial axis
        last_ax = list(ax_radials.values())[-1]
        last_ax.set_xlabel(f"{rng_da.attrs.get('standard_name', 'range')} [{rng_da.units}]")
        if radial_xlims is not None:
            ax_beam.set_xlim(radial_xlims)
            for ax in ax_radials.values():
                ax.set_xlim(radial_xlims)
        if radial_ylims is not None:
            for ax in ax_radials.values():
                ax.set_ylim(radial_ylims)
        # =============================================================================
        # Title metadata 
        # =============================================================================
        # Extract metadata safely
        meta = _safe_metadata(self.xrds)
        # elev_str = meta["elev_str"]
        dt_str = meta["dt_str"]
        rname = meta["rname"]
        swp_mode = meta["swp_mode"]
        vunits = _safe_units(self.xrds[var2plot])
        ptitle = fig_title or (f"{swp_mode} -- {var2plot} [{vunits}]")
        ax_ppi.set_title(ptitle, fontsize=14)
        fig.suptitle(f"{rname.title()} -- {dt_str}", fontsize=16)
        # =============================================================================
        # Add colourbar
        # =============================================================================
        if plot_colorbar:
            pltprms = mappable._pltprms
            if 'rhohv' in var2plot.lower():
                clb = plt.colorbar(mappable, ax=ax_ppi,
                                   ticks=pltprms.norm_boundaries)
            else:
                clb = plt.colorbar(mappable, ax=ax_ppi)
            clb.ax.tick_params(direction='out', labelsize=12, rotation=0)
            # clb.ax.set_title(f'[{vunits}]', fontsize=14)
            if pltprms.ticklabels is not None:
                clb.set_ticks(pltprms.norm_boundaries)
                clb.set_ticklabels(pltprms.ticklabels)
                clb.ax.tick_params(direction="in")
        fig.tight_layout()

        return fig, ax_ppi, ax_beam, ax_radials


#TODO: add keyboard navigation for time browsing (like browse_azimuth)
class HTI_Exp:
    """A class to create an interactive HTI plot."""

    # ------------------------------------------------------------------
    # Constructor: store dataset + configuration only
    # ------------------------------------------------------------------
    def __init__(self, ds, stats=None, ptype="pseudo", contour_var=None,
                 vars_bounds=None, cb_ext=None, unorm=None, custom_rules=None,
                 ucmap=None, plot_mlyr=False, mlyr_ds=None,
                 mlyr_top="MLYRTOP", mlyr_bottom="MLYRBTM", add_colorbar=True,
                 add_grid=False, tz="UTC"):
        self.ds = ds
        self.stats = stats
        self.ptype = ptype
        self.contour_var = contour_var
        self.vars_bounds = vars_bounds
        self.cb_ext = cb_ext
        self.unorm = unorm
        self.ucmap = ucmap
        self.custom_rules = custom_rules
        self.plot_mlyr = plot_mlyr
        self.mlyr_ds = mlyr_ds
        self.mlyr_top = mlyr_top
        self.mlyr_bottom = mlyr_bottom
        self.tz = tz
        self.add_grid = add_grid
        self.add_colorbar = add_colorbar

        # interactive state
        self.current_var = None
        self.current_time = None
        self._time_indicator = None
        # self._time_highlight = None

        # figure elements (set in hti_base)
        self._fig = None
        self._ax_hti = None
        self._ax_profile = None
        self._radio = None
        self._cids = ()
        self.text = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def hti_base(self, *, var2plot=None, fig_size=None, fig_title=None,
                 htixlim=None, htiylim=None, add_grid=False):
        """
        Build an interactive Height–Time Indicator (HTI) figure from a time–height
        radar profile dataset.

        The figure consists of:
        - a time–height panel (left),
        - a height–value profile panel (right),
        - radio buttons for selecting the displayed variable,
        - an interactive controller that updates the profile panel when the user
          right‑clicks on the HTI panel or selects a new variable.

        Parameters
        ----------
        ds : xarray.Dataset
            Time–height dataset, typically produced by :func:`merge_qvps_in_time`.
            Must contain:
            - a 1D ``height`` coordinate,
            - a ``time`` coordinate,
            - one or more variables with dimensions ``(time, height)``.
        var2plot : str or None, default None
            Initial variable to display. If None, the first primary HTI variable
            (as determined by ``_primary_hti_vars``) is used.
        stats : {"std_dev", "sem", "min", "max", None}, default None
            Statistic to visualise in the profile panel. If provided, the
            corresponding statistic variable (e.g., ``"std_var"`` or ``"min_var"``)
            must exist in ``ds``.
        ptype : {"pseudo", "fcontour"}, default "pseudo"
            Plot type for the HTI panel. ``"pseudo"`` uses ``pcolormesh``;
            ``"fcontour"`` uses filled contours.
        vars_bounds : dict or None, default None
            Optional per‑variable bounds passed to :func:`plot_params`.
        cb_ext : dict or None, default None
            Optional per‑variable colourbar extension settings.
        unorm : dict or None, default None
            Optional per‑variable normalisation objects for colour mapping.
        ucmap : dict or None, default None
            Optional per‑variable colormap overrides.
        custom_rules : dict or None, default None
            Optional custom plotting rules passed to :func:`plot_params`.
        fig_size : tuple of float or None, default None
            Figure size in inches. If None, defaults to ``(16, 9)``.
        fig_title : str or None, default None
            Custom figure title. If None, a title is constructed from dataset
            metadata and the selected variable.
        plot_mlyr : bool, default False
            If True, overlay melting‑layer boundaries on both panels.
        mlyr_ds : xarray.Dataset or None, default None
            Dataset containing melting‑layer geometry. If None, ``ds`` is used.
        mlyr_top : str, default "MLYRTOP"
            Name of the melting‑layer top variable in ``mlyr_ds``.
        mlyr_bottom : str, default "MLYRBTM"
            Name of the melting‑layer bottom variable in ``mlyr_ds``.
        plot_axislabels : bool, default True
            Whether to draw axis labels on the HTI panel.
        htixlim : tuple of datetime-like or None, default None
            X‑axis limits for the HTI panel.
        htiylim : tuple of float or None, default None
            Y‑axis limits for the HTI panel.
        tz : str, default "UTC"
            Timezone used for datetime formatting and click handling.

        Returns
        -------
        hti : HTIPlot
            Dataclass containing:
            - ``fig`` : the created figure,
            - ``hti_ax`` : the HTI (time–height) axis,
            - ``profile_ax`` : the profile (value–height) axis,
            - ``radio`` : the variable‑selection radio buttons,
            - ``controller`` : the interactive controller handling clicks and updates.

        Notes
        -----
        * Primary HTI variables are determined by :func:`_primary_hti_vars`, which
          excludes statistical variables and vertical‑resolution diagnostics.
        * Right‑clicking on the HTI panel updates the profile panel to the nearest
          time slice.
        * Variable‑specific plotting parameters (colormap, normalisation, bounds,
          colourbar settings) are obtained from :func:`plot_params`.
        """

        if fig_size is None:
            fig_size = (16, 9)
        prim_vars = _primary_hti_vars(self.ds)
        if not prim_vars:
            raise ValueError("No primary HTI variables found in dataset.")
        if var2plot is None:
            var = _resolve_var2plot(self.ds, None)
        else:
            var = var2plot
        # Resolve title
        fsizes = {"fsz_cb": fontsizetick}  # or whatever you use elsewhere
        vunits = _safe_units(self.ds[var])
        if fig_title is not None:
            auto_title = fig_title
        else:
            meta = _safe_metadata(self.ds)
            elev_str = meta["elev_str"]
            # dt_str = meta["dt_str"]
            rname = meta["rname"]
            # swp_mode = meta["swp_mode"]
            # ptitle = f"HTI Plot ({rname.title()}): Elevation angle scan: {elev_str}"
            # auto_title = f"{ptitle} -- {var} [{vunits}]"
            ptitle1 = "HTI Plot"
            ptitle2 = f"{rname.title()}"
            ptitle3 = f"{elev_str}"  # elevation angle scan
            ptitle4 = f'{var} [{vunits}]'
            auto_title = f"{ptitle1}: {ptitle4} -- {ptitle2} [{ptitle3}]"
        # Layout: AAAB
        fig, axd = plt.subplot_mosaic("""
                                      AAAB
                                      """, figsize=fig_size)
        # fig.suptitle(auto_title, fontsize=fsizes["fsz_cb"])
        hti_ax = axd["A"]
        profile_ax = axd["B"]
        # Store figure elements
        self._fig = fig
        self._ax_hti = hti_ax
        self._ax_profile = profile_ax
        self.ax_title = auto_title
        self.text = self._ax_hti.text(0.008, .02, "selected: none", va='top',
                                      transform=hti_ax.transAxes)
        # Draw HTI panel using the class method
        mappable, pltprms = self._plot_hti_panel(var)
        #TODO: reserve space even if no colorbar is requested
        # Colorbar on top of HTI panel
        if self.add_colorbar:
            _add_colorbar(fig, hti_ax, mappable=mappable, pltprms=pltprms,
                          fsizes=fsizes, vunits=vunits, rotangle=0, size="10%",
                          pad="7%", coord_sys="rect", cartopy_enabled=False,
                          # label=f'{var}\n[{vunits}]',
                          )
        # Right panel: shared y with HTI
        profile_ax.sharey(hti_ax)
        profile_ax.grid(self.add_grid)
        profile_ax.tick_params(labelleft=False)
        # Axis limits
        if htixlim is not None:
            hti_ax.set_xlim(htixlim)
        if htiylim is not None:
            hti_ax.set_ylim(htiylim)
        # Radio buttons above profile panel
        ax_div2 = make_axes_locatable(profile_ax)
        cax2 = ax_div2.append_axes("top", size="10%", pad="7%",
                                   facecolor="lightsteelblue")
        radio = mpl.widgets.RadioButtons(cax2, labels=tuple(prim_vars),
                                         active=prim_vars.index(var),
                                         activecolor="gold")
        for txt in radio.labels:
            txt.set_fontsize(9)
        
        # Controller
        self._radio = radio
        plt.tight_layout()
        plt.show()
        # Connect events
        cid1 = self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        cid2 = self._radio.on_clicked(self._on_var_change)
        self._cids = (cid1, cid2)
        
        # Initial interactive state
        self.current_var = var
        self.current_time = self.ds["time"].isel(time=0).item()
        
        # Draw initial profile
        self._update_profile()

    # ------------------------------------------------------------------
    # Drawing helpers (panel creation)
    # ------------------------------------------------------------------
    def _plot_hti_panel(self, var):
        """
        Draw the left HTI panel (time–height field) into self._ax_hti.
        Returns (mappable, pltprms).
        """
        pltprms = plot_params(
            varname=var, xrds=self.ds, vars_bounds=self.vars_bounds,
            unorm=self.unorm, cb_ext=self.cb_ext,
            custom_rules=self.custom_rules, ucmap=self.ucmap)
        if self.ptype in (None, "pseudo"):
            h = self.ds[var].plot.pcolormesh(
                ax=self._ax_hti, x="time", y="height", cmap=pltprms.cmap,
                norm=pltprms.norm, add_colorbar=False)
        elif self.ptype == "fcontour":
            h = self.ds[var].plot.contourf(
                ax=self._ax_hti, x="time", y="height", cmap=pltprms.cmap,
                norm=pltprms.norm, levels=pltprms.norm_boundaries,
                add_colorbar=False)
        else:
            raise ValueError(f"Unknown ptype: {self.ptype!r}")
        if self.contour_var is not None:
            if self.contour_var == var:
                _overlay_contour(self._ax_hti, self.ds, self.contour_var,
                                 vars_bounds=self.vars_bounds, pltprms=pltprms,
                                 levels=pltprms.norm_boundaries, zorder=10)
            else:
                contour_prms = plot_params(
                    varname=self.contour_var, xrds=self.ds,
                    vars_bounds=self.vars_bounds, unorm=self.unorm,
                    cb_ext=self.cb_ext, custom_rules=self.custom_rules,
                    ucmap=self.ucmap)
                _overlay_contour(self._ax_hti, self.ds, self.contour_var,
                                 vars_bounds=self.vars_bounds,
                                 pltprms=contour_prms, zorder=10)
        # Melting layer overlays
        src = self.mlyr_ds if self.mlyr_ds is not None else self.ds
        if self.plot_mlyr and self.mlyr_top in src:
            self._ax_hti.plot(src["time"], src[self.mlyr_top], lw=2, c="k", ls="--",
                    path_effects=[pe.Stroke(linewidth=7, foreground='w'),
                                  pe.Normal()], label=r"$MLyr_{(T)}$")
            self._ax_hti.scatter(src["time"], src[self.mlyr_top], s=6, c="k", lw=0.5)
        if self.plot_mlyr and self.mlyr_bottom in src:
            self._ax_hti.plot(src["time"], src[self.mlyr_bottom], lw=2, c="grey",
                         ls="--",
                         path_effects=[pe.Stroke(linewidth=7, foreground='w'),
                                       pe.Normal()], label=r"$MLyr_{(B)}$")
            self._ax_hti.scatter(src["time"], src[self.mlyr_bottom], s=6, c="grey",
                            lw=0.5)
        self._ax_hti.set_xlabel("Date and Time", fontsize=fontsizelabels, labelpad=15)
        #TODO: do not hardcode height units
        self._ax_hti.set_ylabel("Height [km]", fontsize=fontsizelabels, labelpad=15)
        self._ax_hti.grid(self.add_grid)
        self._ax_hti.tick_params(axis='both', direction='in', labelsize=fontsizetick, pad=10)
        # using "" removes xarray’s auto-title
        self._ax_hti.set_title(self.ax_title, fontsize=fontsizetitles)
        # locator = mdates.AutoDateLocator(minticks=3, maxticks=13)
        # formatter = mdates.ConciseDateFormatter(locator, tz=ZoneInfo(tz))
        # ax.xaxis.set_major_locator(locator)
        # ax.xaxis.set_major_formatter(formatter)
        self._ax_hti.xaxis.get_offset_text().set_size(fontsizelabels)
        # Legend
        handles, labels = self._ax_hti.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            self._ax_hti.legend(by_label.values(), by_label.keys(), loc="upper left")
        return h, pltprms

    def _update_profile(self):
        """
        Redraw the right profile panel for (self.current_var, self.current_time).
        """
        # Convert numpy.datetime64 -> Python datetime
        t = np64_to_dtm(self.current_time)
        # print(t)
        # Remove previous indicator
        if self._time_indicator is not None:
            self._time_indicator.remove()
            self._time_indicator = None    
        # Draw new vertical line
        self._time_indicator = self._ax_hti.axvline(t, color="k", lw=2.5, 
                                                    ls=":", alpha=0.5)
        # Draw profile panel
        _plot_profile_panel(self.ds, self.current_var, self.current_time,
                            self._ax_profile, stats=self.stats,
                            plot_mlyr=self.plot_mlyr, mlyr_ds=self.mlyr_ds,
                            mlyr_top=self.mlyr_top, mlyr_bottom=self.mlyr_bottom,
                            add_grid=self.add_grid)
        self._ax_profile.figure.canvas.draw_idle()

    def _on_click(self, event):
        # Only respond to right-clicks inside the HTI axis
        if event.button != mpl.backend_bases.MouseButton.RIGHT:
            return
        if event.inaxes is not self._ax_hti:
            return
        # Map xdata (datetime64)
        dt_np = np.datetime64(mdates.num2date(event.xdata), "ns")
        # Nearest time in dataset
        sel_time = self.ds["time"].sel(time=dt_np, method="nearest")
        # self.current_time = sel_time
        self.current_time = sel_time.item()
        hti_seldt = mdates.num2date(event.xdata)
        self.text.set_text(f"selected: {hti_seldt.strftime('%Y-%m-%dT%H:%M:%S')}")
        # Update profile panel
        self._update_profile()

    def _on_var_change(self, label):
        self.current_var = str(label)
        self._update_profile()

    #TODO: add keyboard navigation (simialr to browse_azimuth) bot for time.
    def _on_close(self, event):
        """
        Disconnect callbacks when the figure is closed.
        """
        if not hasattr(self, "_cids"):
            return
        for cid in self._cids:
            self._fig.canvas.mpl_disconnect(cid)


fontsizelabels = 14
fontsizetick = 12
linec, lwid = 'k', 3
fontsizetitles=16


def _primary_hti_vars(ds):
    """Return the list of radar variables to use in the HTI radio buttons."""
    prim = []
    for name, da in ds.data_vars.items():
        if set(da.dims) != {"time", "height"}:
            continue
        if name.startswith(("std_", "min_", "max_", "sem_")):
            continue
        if name in {"VRES"}:
            continue
        prim.append(name)
    return prim


def _stats_var_name(var, stats):
    """ Map stats variable name in the dataset."""
    if stats is None:
        return None
    if stats == "std_dev":
        prefix = "std_"
    elif stats == "sem":
        prefix = "sem_"
    elif stats in {"min", "max"}:
        return None  # handled specially in controller
    else:
        raise ValueError(f"Unknown stats option: {stats!r}")
    return f"{prefix}{var}"


def _overlay_contour(ax, ds, var, *, vars_bounds=None, levels=None,
                     pltprms=None, **kwargs):
    if var not in ds:
        raise KeyError(f"Contour variable {var!r} not found in dataset.")
    # Priority 1: explicit bounds
    if levels is None and vars_bounds and var in vars_bounds:
        levels = vars_bounds[var]
    # Priority 2: boundaries from plot_params for this variable
    if levels is None and pltprms is not None:
        nb = getattr(pltprms, "norm_boundaries", None)
        if nb is not None:
            levels = nb
    # Priority 3: fallback linspace
    if levels is None:
        vmin = float(ds[var].min())
        vmax = float(ds[var].max())
        levels = np.linspace(vmin, vmax, 11)
    c = ds[var].plot.contour(ax=ax, x="time", y="height",
                             levels=levels, colors="k", alpha=0.4,
                             add_colorbar=False,
                             **kwargs)
    ax.clabel(c, inline=True, fontsize=8)
    return c


def _plot_profile_panel(ds, var, time_val, ax, *, plot_mlyr=None, stats=None,
                        mlyr_ds=None, mlyr_top=None, mlyr_bottom=None,
                        add_grid=False):
    """Draw the rightHTI panel (values–height field)."""
    # Normalize time_val to numpy.datetime64
    if isinstance(time_val, xr.DataArray):
        time_val = time_val.item()
    elif hasattr(time_val, "item") and not isinstance(time_val, (np.datetime64,)):
        time_val = time_val.item()
    time_val = np.datetime64(time_val, "ns")
    ax.clear()
    # Select profile lazily, compute only this slice
    prof = ds[var].sel(time=time_val).compute()
    height = ds["height"]
    ax.plot(prof, height, lw=2, label="Profile")
    # Stats shading
    if stats in {"std_dev", "sem"}:
        stats_name = _stats_var_name(var, stats)
        if stats_name in ds:
            s = ds[stats_name].sel(time=time_val).compute()
            ax.fill_betweenx(height, prof - s, prof + s, alpha=0.4,
                             label="std" if stats == "std_dev" else "SEM")
    elif stats in {"min", "max"}:
        min_name = f"min_{var}"
        max_name = f"max_{var}"
        if min_name in ds and max_name in ds:
            vmin = ds[min_name].sel(time=time_val).compute()
            vmax = ds[max_name].sel(time=time_val).compute()
            ax.fill_betweenx(height, vmin, vmax, alpha=0.4, label="min/max")
    # Melting layer overlays (if present)
    src = mlyr_ds if mlyr_ds is not None else ds
    if plot_mlyr and mlyr_top in src:
        mlt = src[mlyr_top].sel(time=time_val, method="nearest").item()
        ax.axhline(mlt, c="k", ls="--", lw=2,
                   path_effects=[pe.Stroke(linewidth=7, foreground='w'),
                                 pe.Normal()], label=r"$MLyr_{(T)}$")
    if plot_mlyr and mlyr_bottom in src:
        mlb = src[mlyr_bottom].sel(time=time_val, method="nearest").item()
        ax.axhline(mlb, c="grey", ls="--", lw=2,
                   path_effects=[pe.Stroke(linewidth=7, foreground='w'),
                                 pe.Normal()], label=r"$MLyr_{(B)}$")
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")
    # Labels, title
    vunits = _safe_units(ds[var])
    label = f'{var} [{vunits}]'
    ax.set_xlabel(label, fontsize=fontsizelabels, labelpad=15)
    # ax.set_ylabel("Height [km]")
    ax.grid(add_grid)
    dt = np.datetime64(time_val, "ns")
    ax.set_title(
        f"{dt.astype('datetime64[ms]').astype(object):%Y-%m-%d %H:%M:%S}",
        fontsize=fontsizetitles)
    ax.tick_params(axis='both', direction='in', labelsize=fontsizetick, pad=10)


def plot_mlyr_detection_from_profiles(ds, rmlyr, diags, *, comb_id,
                                      param_k=None):
    """
    Plot diagnostic panels illustrating the melting-layer detection process
    for a selected combination of polarimetric profiles.

    This function reproduces the legacy visualisation used in the original
    melting-layer detection workflow. It displays:
    (1) normalised DBZ, 1–RHOHV, and their combined profile,
    (2) combination-specific peak-detection diagnostics, and
    (3) all available combinations with an interactive slider to inspect
        their behaviour.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing a 1D ``height`` coordinate (km). Only the height
        array is used.
    rmlyr : Any
        Unused legacy parameter kept for backward compatibility.
    diags : dict-like
        Dictionary containing diagnostic arrays produced by the melting-layer
        detection engine. Must include the following keys:

        - ``"dbz_norm"`` : 1D array of normalised reflectivity
        - ``"rho_norm"`` : 1D array of normalised RHOHV
        - ``"profcombdbz_rhv"`` : 1D combined-profile array
        - ``"pks_pre"`` : dict with peak indices (``"idxmax"``, etc.)
        - ``"comb_mult"`` : list of 1D arrays (per-combination scores)
        - ``"comb_mult_w"`` : list of 1D arrays (weighted scores)
        - ``"mlrand"`` : list of dicts with ML boundary indices
        - ``"min_hidx"`` : int, lower index of valid height range
        - ``"max_hidx"`` : int, upper index of valid height range
        - ``"idxml_btm_it1"`` : int, initial bottom index
        - ``"idxml_top_it1"`` : int, initial top index

        Optionally:
        - ``"param_k"`` : float, default threshold for peak detection
    comb_id : int
        1-based index of the profile combination to visualise. Converted
        internally to 0-based indexing.
    param_k : float or None, default None
        Threshold marker drawn in the first panel. If None, falls back to
        ``diags.get("param_k", 0.05)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure containing the three diagnostic panels and the
        interactive combination slider.

    Notes
    -----
    * This function is intended for diagnostic and algorithm-development
      purposes. It mirrors the plotting logic of the original legacy
      implementation.
    * All arrays are accessed via ``.values`` and must be NumPy-compatible.
    * ``comb_id`` must be 1-based to match the legacy combination numbering.
    """
   
    # Map new diagnostics to legacy variable names
    hbeam = ds["height"].values
    profzh_norm = diags["dbz_norm"]
    profrhv_norm = 1.0 - diags["rho_norm"]  # legacy used 1 - rho_norm
    profcombzh_rhv = diags["profcombdbz_rhv"]
    pkscombzh_rhv = diags["pks_pre"]
    comb_mult = diags["comb_mult"]
    comb_mult_w = diags["comb_mult_w"]
    mlrand = diags["mlrand"]
    min_hidx = diags["min_hidx"]
    max_hidx = diags["max_hidx"]
    idxml_btm_it1 = diags["idxml_btm_it1"]
    idxml_top_it1 = diags["idxml_top_it1"]
    if param_k is None:
        param_k = diags.get("param_k", 0.05)
    comb_idpy = int(comb_id) - 1
    lbl_fs = 24
    lgn_fs = 16
    tks_fs = 20
    lw = 4
    heightbeam = hbeam
    init_comb = comb_idpy
    hb_lim_it1 = heightbeam[idxml_btm_it1:idxml_top_it1]
    if comb_mult_w:
        resimp2d = np.gradient(np.gradient(comb_mult_w[comb_idpy]))
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(14, 10))
    plt.subplots_adjust(left=0.096, right=0.934, top=0.986, bottom=0.091)
    # =============================================================================
    ax1 = axs[0]
    # =============================================================================
    ax1.plot(profzh_norm[min_hidx:max_hidx], heightbeam[min_hidx:max_hidx],
             label=r"$Z^{*}_{H}$", lw=1.5, c="tab:purple")
    ax1.plot(profrhv_norm[min_hidx:max_hidx], heightbeam[min_hidx:max_hidx],
             label=r"$1- \rho^{*}_{HV}$", lw=1.5, c="tab:red")
    ax1.plot(profcombzh_rhv, heightbeam[min_hidx:max_hidx], lw=3, c="tab:blue",
             label=r"$P_{comb}$")
    if ~np.isnan(pkscombzh_rhv["idxmax"]):
        ax1.scatter(profcombzh_rhv[pkscombzh_rhv["idxmax"]],
                    heightbeam[min_hidx:max_hidx][pkscombzh_rhv["idxmax"]],
                    s=300, marker="X", c="tab:orange", label="$P_{{peak}}$")
    ax1.axvline(param_k, c="k", ls=":", lw=2.5, label="k")
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([0, heightbeam[max_hidx]])
    ax1.tick_params(axis="both", labelsize=tks_fs)
    ax1.set_xlabel("(norm)", fontsize=lbl_fs, labelpad=10)
    ax1.set_ylabel("Height [km]", fontsize=lbl_fs, labelpad=10)
    ax1.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    ax1.legend(fontsize=lgn_fs, loc="upper right")
    ax1.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax1.grid(True)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("top", size="10%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.set_facecolor("slategrey")
    at = AnchoredText("Initial identification of the \n"
                      "Melting Layer signatures \n"
                      "combining the normalised \n"
                      r"profiles of $Z_H$ and $\rho_{HV}$", loc="center",
                      prop=dict(size=12, color="white"), frameon=False)
    cax.add_artist(at)
    # =============================================================================
    ax2 = axs[1]
    # =============================================================================
    ac = 0.7
    if comb_mult:
        ax2.plot(comb_mult[comb_idpy], hb_lim_it1, lw=1.5, c="tab:blue",
                 label=f"$P^*_{{{comb_idpy+1}}}$")
        ax2.plot(-resimp2d, hb_lim_it1, lw=3.0, c="gold", alpha=ac,
                 label=f"$-P_{{{comb_idpy+1}}}^*''$")
        ax2.plot(comb_mult_w[comb_idpy], hb_lim_it1, lw=3.0, c="tab:green",
                 alpha=ac, label=f"$P_{{{comb_idpy+1}}}$")
    if comb_mult_w:
        if ~np.isnan(mlrand[comb_idpy]["idxtop"]):
            ax2.scatter(
                comb_mult_w[comb_idpy][mlrand[comb_idpy]["idxtop"]],
                hb_lim_it1[mlrand[comb_idpy]["idxtop"]], s=300, marker="*",
                c="deeppink", alpha=0.5, label=f"$P_{{{comb_idpy+1}(top)}}$")
        if ~np.isnan(mlrand[comb_idpy]["idxmax"]):
            ax2.scatter(
                comb_mult_w[comb_idpy][mlrand[comb_idpy]["idxmax"]],
                hb_lim_it1[mlrand[comb_idpy]["idxmax"]], s=300, marker="X",
                c="tab:orange", alpha=0.5, label=f"$P_{{{comb_idpy+1}(peak)}}$")
        if ~np.isnan(mlrand[comb_idpy]["idxbot"]):
            ax2.scatter(
                comb_mult_w[comb_idpy][mlrand[comb_idpy]["idxbot"]],
                hb_lim_it1[mlrand[comb_idpy]["idxbot"]], s=300, marker="o",
                c="deeppink", alpha=0.5, label=f"$P_{{{comb_idpy+1}(bottom)}}$")
    ax2.tick_params(axis="x", labelsize=tks_fs)
    ax2.set_xlabel("(norm)", fontsize=lbl_fs, labelpad=10)
    ax2.tick_params(axis="both", labelsize=tks_fs)
    ax2.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    ax2.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax2.legend(fontsize=lgn_fs)
    ax2.grid(True)
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("top", size="10%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.set_facecolor("slategrey")
    at = AnchoredText("Detection of the ML boundaries \n"
                      "for a given combination of\n"
                      "polarimetric profiles.\n", loc="center",
                      prop=dict(size=12, color="white"), frameon=False)
    cax.add_artist(at)
    # =============================================================================
    ax3 = axs[2]
    # =============================================================================
    ax3.tick_params(axis="x", labelsize=tks_fs)
    ax3.set_xlabel("(norm)", fontsize=lbl_fs, labelpad=10)
    if comb_mult_w:
        for i in range(0, len(comb_mult)):
            ax3.plot(comb_mult_w[i], hb_lim_it1, c="silver", lw=2, alpha=0.4,
                zorder=0)
            if ~np.isnan(mlrand[i]["idxtop"]):
                ax3.scatter(comb_mult_w[i][mlrand[i]["idxtop"]],
                            hb_lim_it1[mlrand[i]["idxtop"]], s=100, marker="*",
                            c="silver")
            if ~np.isnan(mlrand[i]["idxbot"]):
                ax3.scatter(comb_mult_w[i][mlrand[i]["idxbot"]],
                            hb_lim_it1[mlrand[i]["idxbot"]], s=100, marker="o",
                            c="silver")
        line, = ax3.plot(comb_mult_w[init_comb], hb_lim_it1, lw=lw,
                         c="tab:green")
        mlts = None
        mlbs = None
        if ~np.isnan(mlrand[init_comb]["idxtop"]):
            mlts = ax3.axhline(hb_lim_it1[mlrand[init_comb]["idxtop"]],
                               c="slateblue", ls="dashed", lw=lw, alpha=0.5,
                               label=r"$MLyr_{(T)}$")
        if ~np.isnan(mlrand[init_comb]["idxbot"]):
            mlbs = ax3.axhline(hb_lim_it1[mlrand[init_comb]["idxbot"]],
                               c="steelblue", ls="dashed", lw=lw, alpha=0.5,
                               label=r"$MLyr_{(B)}$")
    ax3.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
    ax3.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax3.legend(fontsize=lgn_fs)
    ax3.grid(True)
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("top", size="10%", pad=0)
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.set_facecolor("slategrey")
    at = AnchoredText("Use the slider to assess the \n"
                      "performance of each profile \n"
                      "combination for detecting the ML.", loc="center",
                      prop=dict(size=12, color="white"), frameon=False)
    cax.add_artist(at)
    # plt.tight_layout()
    # Slider
    if comb_mult_w:
        ax_amp = plt.axes([0.95, 0.15, 0.0225, 0.63])
        allowed_combs = np.linspace(
            1, len(comb_mult_w), len(comb_mult_w)).astype(int)
        samp = Slider(ax_amp, "Comb", 1, len(comb_mult_w),
                      valinit=init_comb + 1, valstep=allowed_combs,
                      color="green", orientation="vertical")
        def comb_slider(val):
            amp = int(samp.val) - 1
            line.set_xdata(comb_mult_w[amp])
            if mlts is not None and np.isfinite(mlrand[amp]["idxtop"]):
                mlts.set_ydata((hb_lim_it1[mlrand[amp]["idxtop"]],
                                hb_lim_it1[mlrand[amp]["idxtop"]]))
            if mlbs is not None and np.isfinite(mlrand[amp]["idxbot"]):
                mlbs.set_ydata((hb_lim_it1[mlrand[amp]["idxbot"]],
                                hb_lim_it1[mlrand[amp]["idxbot"]]))
            fig.canvas.draw_idle()
        samp.on_changed(comb_slider)
    plt.show()
    return fig