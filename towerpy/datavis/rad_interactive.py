"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import warnings
import datetime as dt
from zoneinfo import ZoneInfo
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from matplotlib.widgets import RadioButtons, Slider
from scipy import spatial
from ..utils import radutilities as rut
from ..base import TowerpyError

# warnings.filterwarnings("ignore", category=UserWarning)

tpycm_ref = mpl.colormaps['tpylsc_ref']
tpycm_plv = mpl.colormaps['tpylsc_pvars']
tpycm_rnr = mpl.colormaps['tpylsc_rainrt']
tpycm_2slope = mpl.colormaps['tpylsc_2slope']
tpycm_dv = mpl.colormaps['tpylsc_dbu_rd']
tpycm_3c = mpl.colormaps['tpylc_yw_gy_bu']


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
        if r < intradparams['ngates']-1:
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
        self.lastind = 0
        self.clickcoords = []
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
                # nangle, nrange = quadangle(event.xdata, event.ydata,
                #                            gres_m)
                # if nangle >= 360:
                #     nangle = 0
                xy = [(event.xdata, event.ydata)]
                distance, index = spatial.KDTree(gflat_coords).query(xy)
                id1 = np.unravel_index(index, (intradparams['nrays'],
                                               intradparams['ngates']))
                nangle = id1[0][0]
                nrange = id1[1][0]
                cdt = [nangle, nrange]
                print(f'azimuth {abs(nangle)}',
                      f'gate {int(np.round(nrange))}')
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
            inc = 1
        else:
            inc = -1

        self.lastind[0] += inc
        self.lastind[0] = np.clip(self.lastind[0], 1, 359)
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
            self.clickcoords.append((nangle, nrange, event.key))

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
        f3_axhbeam.set_ylabel('Beam height [Km]', fontsize=14)
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
                intradaxs[f'f3_ax{i+2}'].axvline(intradarrange[int(np.round(nrange))]/1000,
                                                 alpha=.2)
                if '[dB]' in j:
                    intradaxs[f'f3_ax{i+2}'].axhline(0, alpha=.8, c='gray')
                if '[-]' in j:
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
            self.text.set_text(f'selected: {np.abs(nangle), nrange}')

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
        coord_lst = self.clickcoords
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


def ppi_base(rad_georef, rad_params, rad_vars, var2plot=None, proj='rect',
             vars_bounds=None, ppi_xlims=None, ppi_ylims=None, ucmap=None,
             radial_xlims=None, radial_ylims=None, mlyr=None, fig_size=None):
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
    proj : 'rect' or 'polar', optional
        Coordinates projection (polar or rectangular). The default is 'rect'.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped. The default are:
            {'ZH [dBZ]': [-10, 60, 15],
             'ZDR [dB]': [-2, 6, 17],
             'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
             'rhoHV [-]': [0.3, .9, 1],
             'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
             'LDR [dB]': [-35, 0, 11],
             'Rainfall [mm/hr]': [0.1, 64, 11]}
    ppi_xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max] in the PPI. The default is None.
    ppi_ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max] in the PPI. The default is None.
    ucmap : colormap, optional
        User-defined colormap.
    radial_xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max] in the PPI. The default is None.
    radial_ylims : dict containing key and 2-element tuple or list, optional
        Set the y-axis view limits [min, max] in the radial variables. Key must
        be in rad_vars dict. The default is None.
    mlyr : MeltingLayer Class, optional
        Plots an isotropic melting layer. The default is None.

    Returns
    -------
    None.

    """
    ttxt1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    ttxt2 = f" {rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ttxt = ttxt1 + ttxt2
    nangle = 1
    nrange = 1
    lpv = {'ZH [dBZ]': [-10, 60, 15], 'ZV [dBZ]': [-10, 60, 15],
           'ZDR [dB]': [-2, 6, 17],
           'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
           'rhoHV [-]': [0.3, .9, 1],
           'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
           'LDR [dB]': [-35, 0, 11],
           'Rainfall [mm/hr]': [0.1, 64, 11]}
    if vars_bounds is not None:
        lpv.update(vars_bounds)

    # global figradint, intradgs, f3_axvar2plot, f3_axhbeam, heightbeamint
    global figradint, intradgs, f3_axvar2plot, f3_axhbeam, gcoord_sys
    global intradgeoref, intradparams, intradvars, intradaxs
    global mrv, vars_xlim, vars_ylim, rbeamh_b, rbeamh_t, gflat_coords

    if isinstance(rad_georef['beambottom_height [km]'], np.ndarray):
        rbeamh_b = rad_georef['beambottom_height [km]']
    if isinstance(rad_georef['beamtop_height [km]'], np.ndarray):
        rbeamh_t = rad_georef['beamtop_height [km]']

    if mlyr is not None:
        d1 = rad_georef['beam_height [km]']
        d2 = mlyr.ml_bottom
        d3 = mlyr.ml_top
        dx_mlb = rad_georef['xgrid'][:, rut.find_nearest(d1, d2)]
        dy_mlb = rad_georef['ygrid'][:, rut.find_nearest(d1, d2)]
        dz_mlb = np.ones(dx_mlb.shape)
        dx_mlt = rad_georef['xgrid'][:, rut.find_nearest(d1, d3)]
        dy_mlt = rad_georef['ygrid'][:, rut.find_nearest(d1, d3)]
        dz_mlt = np.ones(dx_mlt.shape)

    gcoord_sys = proj
    intradgeoref, intradparams, intradvars = rad_georef, rad_params, rad_vars
    # heightbeamint = rad_georef['beam_height [km]']
    # intradarrange = rad_georef['range [m]']
    # gres_m = rad_params['gateres [m]']
    # ngates_m = rad_params['ngates']
    # ['range [m]']
    # gres_m = intradparams['gateres [m]']

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

    bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
    if vars_bounds is None:
        bnd['bRainfall [mm/hr]'] = np.array((0.01, 0.5, 1, 2, 4, 8, 12, 20,
                                             28, 36, 48, 64, 80, 100))

    dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value, tpycm_plv.N,
                                               extend='both')
             for key, value in bnd.items()}
    if 'bZH [dBZ]' in bnd.keys():
        dnorm['nZH [dBZ]'] = mcolors.BoundaryNorm(bnd['bZH [dBZ]'],
                                                  tpycm_ref.N,
                                                  extend='both')
    if 'bZV [dBZ]' in bnd.keys():
        dnorm['nZV [dBZ]'] = mcolors.BoundaryNorm(bnd['bZV [dBZ]'],
                                                  tpycm_ref.N,
                                                  extend='both')
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(bnd['brhoHV [-]'],
                                                   tpycm_plv.N, extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        dnormrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/hr]'],
                                       tpycm_rnr.N, extend='max')
        dnorm['nRainfall [mm/hr]'] = dnormrr
    if 'bZDR [dB]' in bnd.keys():
        dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(bnd['bZDR [dB]'],
                                                  tpycm_2slope.N,
                                                  extend='both')
        # dnorm['nZDR [dB]'] = mcolors.TwoSlopeNorm(vmin=lpv['ZDR [dB]'][0],
        #                                           vcenter=0,
        #                                           vmax=lpv['ZDR [dB]'][1],
    if 'bKDP [deg/km]' in bnd.keys():
        dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(bnd['bKDP [deg/km]'],
                                                      tpycm_2slope.N,
                                                      extend='both')
    if 'bV [m/s]' in bnd.keys():
        dnorm['nV [m/s]'] = mcolors.BoundaryNorm(bnd['bV [m/s]'],
                                                 tpycm_dv.N,
                                                 extend='both')
    if var2plot is None or var2plot == 'ZH [dBZ]':
        if 'ZH [dBZ]' in rad_vars.keys():
            cmaph, normp = tpycm_ref, dnorm['nZH [dBZ]']
            polradv = 'ZH [dBZ]'
            mrv = rad_vars[polradv]
            tcks = bnd['bZH [dBZ]']
            fcb = 0
        else:
            polradv = list(rad_vars.keys())[0]
            cmaph = tpycm_plv
            normp = dnorm.get('n'+polradv)
            mrv = rad_vars[polradv]
            # fcb = 0
            if '[-]' in polradv:
                cbtks_fmt = 2
                cmaph = tpycm_plv
            if '[dB]' in polradv:
                cmaph = tpycm_2slope
                cbtks_fmt = 1
            if '[deg/km]' in polradv:
                cmaph = tpycm_2slope
                # cbtks_fmt = 1
            if '[m/s]' in polradv:
                cmaph = tpycm_dv
            if '[mm/hr]' in polradv:
                cmaph = tpycm_rnr
                # tpycm.set_under(color='#D2ECFA', alpha=0)
                # tpycm_rnr.set_bad(color='#D2ECFA', alpha=0)
                fcb = 2
    else:
        polradv = var2plot
        mrv = rad_vars[polradv]
        cmaph = tpycm_plv
        normp = dnorm.get('n'+polradv)
        fcb = 0
        if '[-]' in polradv:
            fcb = 2
        if '[dB]' in polradv or '[deg/km]' in polradv:
            cmaph = tpycm_2slope
            fcb = 2
        if '[m/s]' in polradv:
            cmaph = tpycm_dv
        if '[dV/dh]' in polradv:
            cmaph = tpycm_dv
            fcb = 2
        if '[mm/hr]' in polradv:
            cmaph = tpycm_rnr
            # tpycm.set_under(color='#D2ECFA', alpha=0)
            # tpycm_rnr.set_bad(color='#D2ECFA', alpha=0)
            fcb = 2
        if polradv in lpv:
            if lpv.get(polradv)[0] > -1 and lpv.get(polradv)[1] < 1:
                fcb = 2
    if ucmap is not None:
        cmaph = ucmap
    plotunits = [i[i.find('['):]
                 for i in rad_vars.keys() if polradv == i][0]
    tcks = bnd.get('b'+polradv)

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
              '  ppiexplorer.clickcoords                                    \n'
              ' =============================================================')
        # if coastl is not True:
        gflat_coords = [[j for j in i]
                        for i in zip(rad_georef['xgrid'].flat,
                                     rad_georef['ygrid'].flat)]
        f3_axvar2plot = figradint.add_subplot(intradgs[0:-1, 0:2])
        f3_axvar2plot.pcolormesh(rad_georef['xgrid'],
                                 rad_georef['ygrid'],
                                 mrv, shading='auto',
                                 cmap=cmaph, norm=normp)
        clb = plt.colorbar(mpl.cm.ScalarMappable(norm=normp, cmap=cmaph),
                           ax=f3_axvar2plot, format=f'%.{fcb}f',
                           ticks=tcks)
        clb.ax.tick_params(direction='in', labelsize=12)
        f3_axvar2plot.axes.set_aspect('equal')
        clb.ax.set_title(plotunits, fontsize=14)
        f3_axvar2plot.format_coord = format_coord
        if mlyr is not None:
            f3_axvar2plot.scatter(dx_mlb, dy_mlb, dz_mlb,
                                  lw=3, c='tab:purple')
            f3_axvar2plot.scatter(dx_mlt, dy_mlt, dz_mlt,
                                  lw=3, c='tab:orange')
        # else:
        #     prs = ccrs.PlateCarree()
        #     f3_axvar2plot = figradint.add_subplot(intradgs[0:-1, 0:2],
        #                                           projection=prs)
        #     f3_axvar2plot.set_extent([-10.5, 3.5, 60, 49],
        #                              crs=ccrs.PlateCarree())
        #     f3_axvar2plot.coastlines()
        #     f3_axvar2plot.pcolormesh(rad_georef['xgrid']*1000,
        #                              rad_georef['ygrid']*1000,
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
              '  ppiexplorer.clickcoords                                    \n'
              ' =============================================================')
        f3_axvar2plot = figradint.add_subplot(intradgs[0:-1, 0:2],
                                              projection='polar')
        f3_axvar2plot.pcolormesh(rad_georef['theta'],
                                 rad_georef['rho'],
                                 np.flipud(mrv), shading='auto',
                                 cmap=cmaph, norm=normp)
        plt.colorbar(mpl.cm.ScalarMappable(norm=normp, cmap=cmaph),
                     ax=f3_axvar2plot)
        f3_axvar2plot.grid(color='gray', linestyle=':')
        f3_axvar2plot.set_theta_zero_location('N')
        f3_axvar2plot.set_thetagrids(np.arange(nangle, nangle+2))
        f3_axvar2plot.xaxis.grid(ls='-')
        if rad_params['elev_ang [deg]'] < 89:
            plt.rgrids(np.arange(0, (rad_georef['range [m]'][-1]/1000)*1.1,
                                 5 * round((rad_georef['range [m]'][-1]/1000)
                                           / 25)),
                       angle=90)
        f3_axvar2plot.set_xticklabels([])
    txtboxs = 'round, rounding_size=0.5, pad=0.5'
    fc, ec = 'w', 'k'
    # f3_axvar2plot.annotate('| Created using Towerpy |', xy=(0.175, .03),
    #                          fontsize=8, xycoords='axes fraction',
    #                          va='center', ha='center',
    #                          bbox=dict(boxstyle=txtboxs,
    #                                    fc=fc, ec=ec))

    f3_axvar2plot.set_title(ttxt, fontsize=16)
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

    if len(intradaxs) == 2:
        intradaxs['f3_ax2'].get_shared_x_axes().join(intradaxs['f3_ax2'],
                                                     intradaxs['f3_ax3'])

    if len(intradaxs) == 3:
        intradaxs['f3_ax2'].get_shared_x_axes().join(intradaxs['f3_ax2'],
                                                     intradaxs['f3_ax3'],
                                                     intradaxs['f3_ax4'])

    if len(intradaxs) == 4:
        intradaxs['f3_ax2'].get_shared_x_axes().join(intradaxs['f3_ax2'],
                                                     intradaxs['f3_ax3'],
                                                     intradaxs['f3_ax4'],
                                                     intradaxs['f3_ax5'])

    if len(intradaxs) == 5:
        intradaxs['f3_ax2'].get_shared_x_axes().join(intradaxs['f3_ax2'],
                                                     intradaxs['f3_ax3'],
                                                     intradaxs['f3_ax4'],
                                                     intradaxs['f3_ax5'],
                                                     intradaxs['f3_ax6'])

    intradaxs[list(intradaxs)[-1]].set_xlabel('Range [Km]', fontsize=14)
    intradaxs[list(intradaxs)[-1]].tick_params(axis='both', labelsize=12)


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
        hviax.legend(by_label.values(), by_label.keys(), fontsize=16, loc=2)
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
            tms = mdates.num2date(event.xdata).replace(tzinfo=tz).timestamp()
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
        hviax.legend(by_label.values(), by_label.keys(), fontsize=16, loc=2)
        hviax.grid(axis='both')
        hviax.set_title(f"{intscdt[idxdt]:%Y-%m-%d %H:%M:%S}", fontsize=24)
        hviax.set_xlabel(ppvar, fontsize=24, labelpad=15)
        figprofsint.canvas.draw()


def hti_base(pol_profs, mlyrs=None, stats=None, var2plot=None, ucmap=None,
             vars_bounds=None, ptype='pseudo', contourl=None, htixlim=None,
             htiylim=None, tz='Europe/London', fig_size=None):
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
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped. The default are:
            {'ZH [dBZ]': [-10, 60, 15],
             'ZDR [dB]': [-2, 6, 17],
             'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
             'rhoHV [-]': [0.3, .9, 1],
             'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
             'LDR [dB]': [-35, 0, 11],
             'Rainfall [mm/hr]': [0.1, 64, 11]}
    ucmap : colormap, optional
        User-defined colormap.
    ptype : str, 'pseudo' or 'fcontour'
        Create a pseudocolor or filled contours plot.
        The default is 'pseudo'.
    contourl : str, optional
        Draw contour lines of the specified radar variable.
        The default is None.
    htixlim : 2-element tuple or list of datetime objects, optional
        Set the x-axis view limits [min, max]. The default is None.
    htiylim : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    tz : str
        Key/name of the radar data timezone. The given tz string is then
        retrieved from the ZoneInfo module. Default is 'Europe/London'

    Returns
    -------
    radio : widget
        A MPL radio button.

    """
    lpv = {'ZH [dBZ]': [-10, 60, 15], 'ZV [dBZ]': [-10, 60, 15],
           'ZDR [dB]': [-2, 6, 17],
           'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
           'rhoHV [-]': [0.3, .9, 1],
           'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
           'LDR [dB]': [-35, 0, 11],
           'Rainfall [mm/hr]': [0.1, 64, 11]}
    if vars_bounds is not None:
        lpv.update(vars_bounds)
    bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
    bnd['bRainfall [mm/hr]'] = np.array((0.01, 0.5, 1, 2, 4, 8, 12, 20,
                                         28, 36, 48, 64, 80, 100))

    dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value, tpycm_plv.N,
                                               extend='both')
             for key, value in bnd.items()}
    if 'bZH [dBZ]' in bnd.keys():
        dnorm['nZH [dBZ]'] = mcolors.BoundaryNorm(bnd['bZH [dBZ]'],
                                                  tpycm_ref.N,
                                                  extend='both')
    if 'bZV [dBZ]' in bnd.keys():
        dnorm['nZV [dBZ]'] = mcolors.BoundaryNorm(bnd['bZV [dBZ]'],
                                                  tpycm_ref.N,
                                                  extend='both')
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(bnd['brhoHV [-]'],
                                                   tpycm_plv.N, extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        bnrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/hr]'], tpycm_rnr.N,
                                    extend='max')
        dnorm['nRainfall [mm/hr]'] = bnrr
    if 'bZDR [dB]' in bnd.keys():
        dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(bnd['bZDR [dB]'],
                                                  tpycm_2slope.N,
                                                  extend='both')
    if 'bKDP [deg/km]' in bnd.keys():
        dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(bnd['bKDP [deg/km]'],
                                                      tpycm_2slope.N,
                                                      extend='both')
    if 'bV [m/s]' in bnd.keys():
        dnorm['nV [m/s]'] = mcolors.BoundaryNorm(bnd['bV [m/s]'],
                                                 tpycm_dv.N,
                                                 extend='both')
    if var2plot is None or var2plot == 'ZH [dBZ]':
        var2plot = 'ZH [dBZ]'
        prflv = var2plot
        cmaph, normp = tpycm_ref, dnorm['nZH [dBZ]']
        tcks = bnd['bZH [dBZ]']
        fcb = 0
    else:
        prflv = var2plot
        cmaph = tpycm_plv
        normp = dnorm.get('n'+var2plot)
        fcb = 2
        if '[dB]' in var2plot or '[deg/km]' in var2plot:
            cmaph = tpycm_2slope
        if '[m/s]' in var2plot:
            cmaph = tpycm_dv.reversed()
        if '[dV/dh]' in var2plot:
            cmaph = tpycm_dv
        if lpv.get(var2plot)[0] > -1 and lpv.get(var2plot)[1] < 1:
            fcb = 2
    if ucmap is not None:
        cmaph = ucmap
    tcks = bnd.get('b'+var2plot)

    profsheight = np.array([nprof.georef['profiles_height [km]']
                            for nprof in pol_profs]).T
    profsdt = [nprof.scandatetime for nprof in pol_profs]
    if pol_profs[0].profs_type == 'birdbath_scan':
        profsvars = {k: np.array([nprof.vps[k] for nprof in pol_profs]).T
                     for k in pol_profs[0].vps.keys()}
        if stats == 'std_dev' or stats == 'sem':
            profsstat = {k: np.array([nprof.vps_stats[stats][k]
                                      for nprof in pol_profs]).T
                         for k in pol_profs[0].vps_stats[stats].keys()}
        else:
            profsstat = None
        # elif:

    elif pol_profs[0].profs_type == 'quasi-vertical profiles':
        profsvars = {k: np.array([nprof.qvps[k] for nprof in pol_profs]).T
                     for k in pol_profs[0].qvps.keys()}
        # TODO add max/min visualisation
        if stats == 'std_dev' or stats == 'sem':
            profsstat = {k: np.array([nprof.qvps_stats[stats][k]
                                      for nprof in pol_profs]).T
                         for k in pol_profs[0].qvps_stats[stats].keys()}
        else:
            profsstat = None
    if mlyrs:
        mlyrtop = [mlyr.ml_top if isinstance(mlyr.ml_top, float) else np.nan
                   for mlyr in mlyrs]
        mlyrbot = [mlyr.ml_bottom if isinstance(mlyr.ml_bottom, float)
                   else np.nan for mlyr in mlyrs]
    else:
        mlyrtop = None
        mlyrbot = None
    plotunits = [i[i.find('['):]
                 for i in profsvars.keys() if var2plot == i][0]
    plotvname = [i[:i.find('[')-1]
                 for i in profsvars.keys() if var2plot == i][0]

    # -------------------------------------------------------------------------
    fontsizelabels = 24
    fontsizetick = 20
    linec, lwid = 'k', 3
    ttxt = f"{profsdt[0]:%Y-%m-%d %H:%M:%S}"
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
                          shading='auto', cmap=cmaph, norm=normp)
    elif ptype == 'fcontour':
        htiplt.contourf(profsdt, profsheight[:, 0], profsvars[prflv],
                        shading='auto', cmap=cmaph, norm=normp,
                        levels=bnd.get('b'+var2plot))
    else:
        raise TowerpyError('Oops!... Check the selected plot type')
    if contourl is not None:
        contourlp = htiplt.contour(profsdt, profsheight[:, 0],
                                   profsvars[contourl],
                                   levels=bnd.get('b'+contourl), colors='k',
                                   alpha=0.4, zorder=10)
        htiplt.clabel(contourlp, inline=True, fontsize=8)
    if mlyrtop is not None:
        htiplt.plot(profsdt, mlyrtop, label='ML', lw=lwid, c=linec, ls='--')
        htiplt.scatter(profsdt, mlyrtop, lw=2, s=3, c=linec)
    if mlyrbot is not None:
        htiplt.plot(profsdt, mlyrbot, label='bottom BB', lw=lwid, c='grey',
                    ls='--')
        htiplt.scatter(profsdt, mlyrbot, lw=2, s=3, c='grey')
    if htixlim is not None:
        htiplt.set_xlim(htixlim)
    if htiylim is not None:
        htiplt.set_ylim(htiylim)
    htiplt.grid(True)

    ax1_divider = make_axes_locatable(htiplt)
    cax1 = ax1_divider.append_axes("top", size="10%", pad="7%")
    if '[-]' in var2plot:
        cb1 = figprofsint.colorbar(mpl.cm.ScalarMappable(norm=normp,
                                                         cmap=cmaph),
                                   ax=htiplt, format=f'%.{fcb}f', cax=cax1,
                                   orientation="horizontal", ticks=tcks)
    else:
        cb1 = figprofsint.colorbar(mpl.cm.ScalarMappable(norm=normp,
                                                         cmap=cmaph),
                                   ax=htiplt, format=f'%.{fcb}f', cax=cax1,
                                   orientation="horizontal")
    cb1.ax.tick_params(direction='in', labelsize=20)
    cb1.ax.set_ylabel(f'{plotvname} \n'
                      + f'{plotunits}', fontsize=15, labelpad=50)
    cax1.xaxis.set_ticks_position("top")
    htiplt.tick_params(axis='both', direction='in', labelsize=fontsizetick,
                       pad=10)
    htiplt.set_xlabel('Date and Time', fontsize=fontsizelabels, labelpad=15)
    htiplt.set_ylabel('Height [km]', fontsize=fontsizelabels, labelpad=15)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=13)
    formatter = mdates.ConciseDateFormatter(locator)
    # formatter.offset_formats = ['',
    #                             '%Y',
    #                             '%b %Y',
    #                             '%d %b %Y'
    #                             ]
    htiplt.xaxis.set_major_locator(locator)
    htiplt.xaxis.set_major_formatter(formatter)
    mpl.rcParams['xtick.labelsize'] = 26
    mpl.rcParams['timezone'] = tz
    txtboxs = 'round, rounding_size=0.5, pad=0.5'
    fc, ec = 'w', 'k'
    # htiplt.annotate('| Created using Towerpy |', xy=(0.02, -.1), fontsize=8,
    #                 xycoords='axes fraction', va='center', ha='center',
    #                 bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))

    hviax = axd['B']
    hviax.grid(axis='both')
    hviax.sharey(htiplt)
    hviax.tick_params(axis='both', direction='in', labelsize=fontsizetick,
                      pad=10)
    hviax.yaxis.set_tick_params(labelbottom=False)
    hviax.set_title(ttxt, fontsize=fontsizelabels)
    hviax.set_xlabel(prflv, fontsize=fontsizelabels, labelpad=15)

    ax2_divider = make_axes_locatable(hviax)
    cax2 = ax2_divider.append_axes("top", size="10%", pad="7%",
                                   facecolor='lightsteelblue')
    # cax2.remove()
    radio = RadioButtons(cax2, tuple(intpvars.keys()), activecolor='gold')
    # for circle in radio.circles:
    #     circle.set_radius(0.05)
    for txtr in radio.labels:
        txtr.set_fontsize(8)

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

    resimp1d = np.gradient(comb_mult_w[comb_idpy])
    resimp2d = np.gradient(np.gradient(comb_mult_w[comb_idpy]))

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(12, 10))
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
    ax1.scatter(profcombzh_rhv[pkscombzh_rhv['idxmax']],
                heightbeam[min_hidx:max_hidx][pkscombzh_rhv['idxmax']], s=300,
                marker="X", c='tab:orange', label='$P_{{peak}}$')
    # ax1.axhline(peakcombzh_rhv+.75, c='gray', ls='dashed', lw=lw, alpha=.5,
    #             label=r'$U_{{L}}$')
    ax1.axvline(param_k, c='k', ls=':', lw=2.5, label='k')
    ax1.set_xlim([-0.05, 1.05])
    # ax1.set_ylim([heightbeam[min_hidx], heightbeam[max_hidx]])
    ax1.set_ylim([0, heightbeam[max_hidx]])
    ax1.tick_params(axis='both', labelsize=tks_fs)
    ax1.set_xlabel('(norm)', fontsize=lbl_fs, labelpad=10)
    ax1.set_ylabel('Height [km]', fontsize=lbl_fs, labelpad=10)
    ax1.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
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
                      r'profiles of $Z_H$ and $\rho_{HV}$',
                      loc=10, prop=dict(size=12, color='white'), frameon=False)
    cax.add_artist(at)

    # =============================================================================
    ax2 = axs[1]
    # =============================================================================
    ac = 0.7

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
    ax2.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
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
                      'polarimetric profiles.\n',
                      # 'The 1st and 2nd derivative are \n' +
                      # ' also shown 10.5194/amt-14-2873-2021',
                      loc=10, prop=dict(size=12, color='white'), frameon=False)
    cax.add_artist(at)

    # =============================================================================
    ax3 = axs[2]
    # =============================================================================
    ax3.tick_params(axis='x', labelsize=tks_fs)
    ax3.set_xlabel('(norm)', fontsize=lbl_fs, labelpad=10)

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
    ax3.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
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
                      loc=10, prop=dict(size=12, color='white'), frameon=False)
    cax.add_artist(at)

    # ax_amp = plt.axes([0.25, 0.15, 0.65, 0.03])
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
            mlts.set_ydata(hb_lim_it1[mlrand[amp]['idxtop']])
        if np.isfinite(mlrand[amp]['idxbot']):
            mlbs.set_ydata(hb_lim_it1[mlrand[amp]['idxbot']])
        fig.canvas.draw_idle()

    samp.on_changed(comb_slider)

    plt.show()
