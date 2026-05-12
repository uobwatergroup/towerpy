"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from ..utils import radutilities as rut
from ..base import TowerpyError
from ..utils import unit_conversion as tpuc
from ..utils.radutilities import resolve_rect_coords, find_nearest
from ..utils.radutilities import _safe_metadata, getcoordunits
from ..utils.radutilities import _as_dataset, _deep_update
from ..utils.unit_conversion import _safe_units, convert, _normalise_units
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def pltparams(var2plot, rad_varskeys, vars_bounds, ucmap=None, unorm=None,
              cb_ext=None):
    """
    Create parameters for plots.

    Parameters
    ----------
    var2plot : str
        Key of the radar variable to plot. The default is None.
        This option will plot ZH or the 'first' element in the
        rad_vars dict.
    rad_varskeys : list
        List of radar variables names.
    vars_bounds : dict containing key and 3-element tuple or list
        The default are:
           {'ZH [dBZ]': [-10, 60, 15], 'ZDR [dB]': [-2, 6, 17],
            'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
            'rhoHV [-]': [0.3, .9, 1], 'AH [dB/km]': [0, 0.5, 11],
            'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
            'LDR [dB]': [-30, 10, 17], }, 'Rainfall [mm/h]': [0, 64, 11],
            'Rainfall [mm]': [0, 200, 14], 'SQI [0-1]': [0, 1, 11]
            'beam_height [km]': [0, 7, 36]}
    ucmap : colormap, optional
        User-defined colormap, either a mpl.colors.ListedColormap,
        or string from matplotlib.colormaps.
    unorm : dict containing mpl.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto
        radar data. The default is None.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    """
    lpv = {'ZH [dBZ]': [-10, 60, 15], 'ZDR [dB]': [-2, 6, 17],
           'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
           'rhoHV [-]': [0.3, .9, 1], 'AH [dB/km]': [0, 0.5, 11],
           'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1.8, 0.6, 13],
           'Rainfall [mm/h]': [0, 64, 11], 'Rainfall [mm]': [0, 200, 14],
           'beam_height [km]': [0, 7, 36], 'SQI [0-1]': [0, 1, 11]}
    if var2plot is not None:
        if var2plot != 'ZDR [dB]':
            if var2plot == 'LDR [dB]' or 'LDR [dB]' in rad_varskeys:
                lpv['LDR [dB]'] = [-30, 10, 17]
            if var2plot == 'PIA [dB]' or 'PIA [dB]' in rad_varskeys:
                lpv['PIA [dB]'] = [0, 20, 17]
    if vars_bounds is not None:
        lpv.update(vars_bounds)
    if unorm is not None:
        lpv2 = {key: [value.vmin, value.vmax, value.N]
                for key, value in unorm.items()}
        lpv.update(lpv2)
    #
    bnd = {key[key.find('['):]: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
    if vars_bounds is None or 'Rainfall [mm/h]' not in vars_bounds.keys():
        bnd['[mm/h]'] = np.array((0.1, 1, 2, 4, 8, 12, 16, 20, 24, 30, 36, 48,
                                 56, 64))
    if vars_bounds is None or 'Rainfall [mm]' not in vars_bounds.keys():
        bnd['[mm]'] = np.array((0.1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                75, 100, 125, 150, 200))
    #
    cmaph = {'[dBZ]': mpl.colormaps['tpylsc_rad_ref'],
             '[-]': mpl.colormaps['tpylsc_rad_pvars'],
             '[dB]': mpl.colormaps['tpylsc_rad_2slope'],
             '[deg/km]': mpl.colormaps['tpylsc_rad_2slope'],
             '[dB/km]': mpl.colormaps['tpylsc_rad_pvars'],
             '[m/s]': mpl.colormaps['tpylsc_div_dbu_rd'],
             '[mm/h]': mpl.colormaps['tpylsc_rad_rainrt'],
             '[mm]': mpl.colormaps['tpylsc_rad_rainrt'],
             '[km]': mpl.colormaps['gist_earth'],
             '[dV/dh]': mpl.colormaps['tpylsc_rad_2slope_r'],
             }
    cmaph['[mm/h]'].set_under('whitesmoke')
    cmaph['[mm]'].set_under('whitesmoke')
    if var2plot and '[dB]' in var2plot and var2plot != 'ZDR [dB]':
        cmaph['[dB]'] = mpl.colormaps['tpylsc_rad_pvars']
    if var2plot == 'LDR [dB]':
        cmaph['[dB]'] = mpl.colormaps['tpylsc_rad_2slope_r']
    if var2plot == 'PIA [dB]':
        cmaph['[dB]'] = mpl.colormaps['tpylsc_useq_fiery']
    # elif len(rad_varskeys) == 1 and 'LDR [dB]' in rad_varskeys:
    #     cmaph['[dB]'] = mpl.colormaps['tpylsc_rad_2slope_r']
    if ucmap is not None:
        if var2plot:
            if isinstance(ucmap, str):
                cmaph[var2plot[var2plot.find('['):]] = mpl.colormaps[ucmap]
            else:
                cmaph[var2plot[var2plot.find('['):]] = ucmap
        else:
            if 'ZH [dBZ]' in rad_varskeys:
                v2pdmmy = 'ZH [dBZ]'
            else:
                v2pdmmy = list(rad_varskeys)[0]
            if isinstance(ucmap, str):
                cmaph[v2pdmmy[v2pdmmy.find('['):]] = mpl.colormaps[ucmap]
            else:
                cmaph[v2pdmmy[v2pdmmy.find('['):]] = ucmap
    #
    cmapext = {'[dBZ]': 'both', '[-]': 'both', '[dB]': 'both',
               '[deg/km]': 'both', '[m/s]': 'both', '[mm/h]': 'max',
               '[mm]': 'max', '[km]': 'max', '[dB/km]': 'max',
               '[0-1]': 'both', '[dV/dh]': 'both'}
    if var2plot == 'rhoHV [-]':
        cmapext['[-]'] = 'min'
    if cb_ext:
        cb_ext2 = {key[key.find('['):]: value for key, value in cb_ext.items()}
        cmapext.update(cb_ext2)
    #
    dnorm = {key: [value2 for key2, value2 in unorm.items()
                   if key2[key2.find('['):] == key][0]
             if unorm and key in [key2[key2.find('['):]
                                  for key2, value2 in unorm.items()]
             else
             mpc.BoundaryNorm(value, cmaph.get(
                 key[key.find('['):], mpl.colormaps['tpylsc_rad_pvars']).N,
                 extend=cmapext.get(key[key.find('['):], 'both'))
             for key, value in bnd.items()}
    #
    cbtks_fmt = 0
    tcks = None
    if var2plot is None or var2plot == 'ZH [dBZ]':
        if 'ZH [dBZ]' in rad_varskeys:
            var2plot = 'ZH [dBZ]'
            normp = dnorm['[dBZ]']
            if dnorm['[dBZ]']:
                tcks = dnorm['[dBZ]'].boundaries
            else:
                tcks = bnd['[dBZ]']
        else:
            var2plot = list(rad_varskeys)[0]
            normp = dnorm.get(var2plot[var2plot.find('['):])
            if dnorm.get(var2plot[var2plot.find('['):]):
                tcks = dnorm.get(var2plot[var2plot.find('['):]).boundaries
            else:
                tcks = bnd.get(var2plot[var2plot.find('['):])
    else:
        normp = dnorm.get(var2plot[var2plot.find('['):])
        # tcks = bnd.get(var2plot[var2plot.find('['):])
        if dnorm.get(var2plot[var2plot.find('['):]):
            tcks = dnorm.get(var2plot[var2plot.find('['):]).boundaries
        else:
            tcks = bnd.get(var2plot[var2plot.find('['):])
    if var2plot == 'rhoHV [-]':
        cbtks_fmt = 2
    if '[dB]' in var2plot:
        cbtks_fmt = 1
    if '[mm/h]' in var2plot:
        cbtks_fmt = 1
        # tickLabels = map(str, tcks)
    if '[mm]' in var2plot:
        cbtks_fmt = 1
    if '[km]' in var2plot:
        cbtks_fmt = 2
    if '[dB/km]' in var2plot:
        cbtks_fmt = 2
    if '[deg/km]' in var2plot:
        cbtks_fmt = 1
    if '[dV/dh]' in var2plot:
        cbtks_fmt = 2
    if tcks is not None and len(tcks) > 20:
        tcks = None

    return lpv, bnd, cmaph, cmapext, dnorm, var2plot, normp, cbtks_fmt, tcks


def plot_ppi(rad_georef, rad_params, rad_vars, var2plot=None, mlyr=None,
             vars_bounds=None, ucmap=None, unorm=None, plot_contourl=None,
             contour_kw=None, coord_sys='polar', cpy_feats=None, data_proj=None,
             proj_suffix='osgb', xlims=None, ylims=None, ring=None,
             range_rings=None, rd_maxrange=False, pixel_midp=False,
             points2plot=None, ptsvar2plot=None, cbticks=None, cb_ext=None,
             fig_title=None, fig_size=None, font_sizes='regular'):
    """
    Display a radar PPI scan.

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
        will plot ZH or look for the 'first' element in the rad_vars dict.
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
    plot_contourl: str, optional
        Key of the variable (within rad_vars) used to plot contour lines.
        Levels and normalisation are retrieved from vars_bounds, but
        these and other parameters can be overridden using the contour_kw
        parameter.
    contour_kw:
       Additional keyword arguments passed to matplotlib.pyplot.contour.
    coord_sys : 'rect' or 'polar', optional
        Coordinates system (polar or rectangular). The default is 'polar'.
    cpy_feats : dict, optional
        Cartopy attributes to add to the map. The default are:
         {'status': False, 'add_land': False, 'add_ocean': False,
         'add_coastline': False, 'add_borders': False, 'add_countries': True,
         'add_provinces': True, 'borders_ls': ':', 'add_lakes': False,
         'lakes_transparency': 0.5, 'add_rivers': False, 'tiles': False,
         'tiles_source': None, 'tiles_style': None}
    data_proj : Cartopy Coordinate Reference System object, optional
        Cartopy projection used to plot the data in a map e.g.,
        ccrs.OSGB(approx=False).
    proj_suffix : str, optional
        Suffix of the georeferenced grids used to display the data.
        The X/Y grids must exist in the rad_georef dictionary, e.g.
        'grid_osgbx--grid_osgby', 'grid_utmx--grid_utmy',
        'grid_wgs84x--grid_wgs84y', etc. The default is 'osgb'.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    ring : int or float, optional
        Plot a circle in the given distance, in km.
    range_rings : int, float, list or tuple, optional
        If int or float, plot circles at a fixed range, in km.
        If list or tuple, plot circles at the given ranges, in km.
    rd_maxrange : Bool, optional
        If True, plot the radar's maximum range coverage. Note that this arg
        won't work if a polar coordinates system is used. The default is False.
    pixel_midp : Bool, optional
        If True, mark the mid-point of all radar pixels. Note that this arg
        won't work if a polar coordinates system is used. The default is False.
    points2plot : dict, optional
        Plot a given set of points. Dict must contain the x-coord and y-coord
        in the same format as coord_sys or proj_suffix. A third element inside
        the dict can be used as the z-coord.
    ptsvar2plot : str, optional
        Key of the variable to plot. The default is None. This option
        will looks for the 'first' element in the points2plot dict.
    cbticks : dict, optional
        Modifies the default ticks' location (dict values) and labels
        (dict keys) in the colour bar.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    fig_title : str, optional
        String to show in the plot title.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    font_sizes : str, optional
        Modifies the size of the fonts in the plot. The string has to
        be one of 'regular' or 'large'.
    """
    fsizes = {'fsz_cb': 10, 'fsz_cbt': 12, 'fsz_pt': 14, 'fsz_axlb': 12,
              'fsz_axtk': 10}
    if font_sizes == 'large':
        fsizes = {k1: v1 + 4 for k1, v1 in fsizes.items()}
    # szpnts = 25
    szpnts = None
    #
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        var2plot, rad_vars.keys(), vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    if var2plot is None:
        var2plot = v2p
    cmapp = cmaph.get(var2plot[var2plot.find('['):],
                      mpl.colormaps['tpylsc_rad_pvars'])
# =============================================================================
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # txtboxc = (0, -.09)
    # fc, ec = 'w', 'k'
# =============================================================================
    cpy_features = {'status': False,
                    # 'coastresolution': '10m',
                    'add_land': False,
                    'add_ocean': False,
                    'add_coastline': False,
                    'add_borders': False,
                    'add_countries': True,
                    'add_provinces': True,
                    'borders_ls': ':',
                    'add_lakes': False,
                    'lakes_transparency': 0.5,
                    'add_rivers': False,
                    'tiles': False,
                    'tiles_source': None,
                    'tiles_style': None,
                    'tiles_res': 8, 'alpha_tiles': 0.5, 'alpha_rad': 1
                    }
    if cpy_feats:
        cpy_features.update(cpy_feats)
    if cpy_features['status']:
        coord_sys = 'rect'
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='10m',
            facecolor='none')
        countries = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='10m',
            facecolor='none')
# =============================================================================
    if fig_title is None:
        if isinstance(rad_params['elev_ang [deg]'], str):
            dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
        else:
            dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{2}f} deg. -- "
        if rad_params['datetime']:
            dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
        else:
            dtdes2 = ''
        ptitle = dtdes1 + dtdes2
    else:
        ptitle = fig_title
    plotunits = [i[i.find('['):]
                 for i in rad_vars.keys() if var2plot == i][0]
# =============================================================================
    if coord_sys == 'polar':
        if fig_size is None:
            fig_size = (6, 6.15)
        fig, ax1 = plt.subplots(figsize=fig_size,
                                subplot_kw=dict(projection='polar'))
        mappable = ax1.pcolormesh(
            *np.meshgrid(rad_georef['azim [rad]'],
                         rad_georef['range [m]'] / 1000, indexing='ij'),
            np.flipud(rad_vars[var2plot]), shading='auto', cmap=cmapp,
            norm=normp)
        ax1.set_title(f'{ptitle} \n' + f'PPI {var2plot}',
                      fontsize=fsizes['fsz_pt'])
        ax1.grid(color='gray', linestyle=':')
        ax1.set_theta_zero_location('N')
        ax1.tick_params(axis='both', labelsize=fsizes['fsz_axlb'])
        ax1.set_yticklabels([])
        ax1.set_thetagrids(np.arange(0, 360, 90))
        ax1.axes.set_aspect('equal')
        if (var2plot == 'rhoHV [-]' or '[mm]' in var2plot
           or '[mm/h]' in var2plot):
            cb1 = plt.colorbar(mappable, ax=ax1, aspect=8, shrink=0.65,
                               pad=.1, norm=normp, ticks=tcks,
                               format=f'%.{cbtks_fmt}f')
            cb1.ax.tick_params(direction='in', axis='both',
                               labelsize=fsizes['fsz_cb'])
        else:
            cb1 = plt.colorbar(mappable, ax=ax1, aspect=8, shrink=0.65,
                               pad=.1, norm=normp)
            cb1.ax.tick_params(direction='in', axis='both',
                               labelsize=fsizes['fsz_cb'])
        cb1.ax.set_title(f'{plotunits}', fontsize=fsizes['fsz_cbt'])
        if cbticks is not None:
            cb1.set_ticks(ticks=list(cbticks.values()),
                          labels=list(cbticks.keys()))
        # ax1.annotate('| Created using Towerpy |', xy=txtboxc,
        #              fontsize=8, xycoords='axes fraction',
        #              va='center', ha='center',
        #              bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
        plt.tight_layout()
        # plt.show()

    elif coord_sys == 'rect' and cpy_features['status'] is False:
        # =====================================================================
        # ptitle = dtdes1 + dtdes2
        # =====================================================================
        if fig_size is None:
            fig_size = (6, 6.75)
        fig, ax1 = plt.subplots(figsize=fig_size)
        mappable = ax1.pcolormesh(rad_georef['grid_rectx'],
                                  rad_georef['grid_recty'],
                                  rad_vars[var2plot], shading='auto',
                                  cmap=cmapp, norm=normp)
        if rd_maxrange:
            ax1.plot(rad_georef['grid_rectx'][:, -1],
                     rad_georef['grid_recty'][:, -1], 'gray')
        if pixel_midp:
            binx = rad_georef['grid_rectx'].ravel()
            biny = rad_georef['grid_recty'].ravel()
            ax1.scatter(binx, biny, c='grey', marker='+', alpha=0.2)
# =============================================================================
        if points2plot is not None:
            if len(points2plot) == 2:
                ax1.scatter(
                    points2plot['grid_rectx'], points2plot['grid_recty'],
                    color='k', marker='o', s=szpnts)
            elif len(points2plot) >= 3:
                ax1.scatter(
                    points2plot['grid_rectx'], points2plot['grid_recty'],
                    marker='o', norm=normp, edgecolors='k', cmap=cmapp,
                    c=[points2plot[ptsvar2plot]], s=szpnts)
# =============================================================================
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
            ax1.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(T)}$')
            ax1.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(B)}$')
            first_legend = ax1.legend(loc='upper left')
            # Add the legend manually to the Axes.
            ax1.add_artist(first_legend)
# =============================================================================
        if range_rings is not None:
            if isinstance(range_rings, range):
                range_rings = list(range_rings)
            if isinstance(range_rings, (int, float)):
                nrings = np.arange(range_rings*1000,
                                   rad_georef['range [m]'][-1],
                                   range_rings*1000)
            elif isinstance(range_rings, (np.ndarray, list, tuple)):
                nrings = np.array(range_rings) * 1000
            idx_rs = [rut.find_nearest(rad_georef['range [m]'], r)
                      for r in nrings]
            dmmy_rsx = np.array([rad_georef['grid_rectx'][:, i]
                                 for i in idx_rs])
            dmmy_rsy = np.array([rad_georef['grid_recty'][:, i]
                                 for i in idx_rs])
            dmmy_rsz = np.array([np.ones(i.shape) for i in dmmy_rsx])
            ax1.scatter(dmmy_rsx, dmmy_rsy, dmmy_rsz, c='grey', ls='--',
                        alpha=3/4)
            ax1.axhline(0, c='grey', ls='--', alpha=3/4)
            ax1.axvline(0, c='grey', ls='--', alpha=3/4)
            ax1.grid(True)
# =============================================================================
        if ring is not None:
            idx_rr = rut.find_nearest(rad_georef['range [m]'],
                                      ring*1000)
            dmmy_rx = rad_georef['grid_rectx'][:, idx_rr]
            dmmy_ry = rad_georef['grid_recty'][:, idx_rr]
            dmmy_rz = np.ones(dmmy_rx.shape)
            ax1.scatter(dmmy_rx, dmmy_ry, dmmy_rz, c='k', ls='--', alpha=3/4)
# =============================================================================
        if plot_contourl:
            ckw = {'alpha': 0.5, 'zorder': 2, 'colors': None,
                   'levels': bnd.get(plot_contourl[plot_contourl.find('['):]),
                   'norm': dnorm.get(plot_contourl[plot_contourl.find('['):]),
                   'cmap': cmaph.get(plot_contourl[plot_contourl.find('['):]),
                   'legend': False,
                   }
            if contour_kw is not None:
                ckw.update(contour_kw)
            contourlp = ax1.contour(
                rad_georef['grid_rectx'], rad_georef['grid_recty'],
                rad_vars[plot_contourl],
                **ckw)
            ax1.clabel(contourlp, inline=True, fontsize=fsizes['fsz_cbt'])
            if ckw['legend']:
                cspl, labels = contourlp.legend_elements()
                labels = [lb.replace('x = ', '') for lb in labels]
                ax1.legend(cspl, labels, title=plot_contourl,
                           loc='upper right').set_zorder(5)
# =============================================================================
        ax1_divider = make_axes_locatable(ax1)
        cax1 = ax1_divider.append_axes('top', size="7%", pad="2%")
        if (var2plot == 'rhoHV [-]' or '[mm]' in var2plot
           or '[mm/h]' in var2plot):
            cb1 = fig.colorbar(mappable, cax=cax1, orientation='horizontal',
                               ticks=tcks, format=f'%.{cbtks_fmt}f')
            cb1.ax.tick_params(direction='in', labelsize=fsizes['fsz_cb'],
                               rotation=(45 if font_sizes == 'large' else 0))
        else:
            cb1 = fig.colorbar(mappable, cax=cax1, orientation='horizontal')
            cb1.ax.tick_params(direction='in', labelsize=fsizes['fsz_cb'])
        fig.suptitle(f'{ptitle} \n' + f'PPI {var2plot}',
                     fontsize=fsizes['fsz_pt'])
        cax1.xaxis.set_ticks_position('top')
        if xlims is not None:
            ax1.set_xlim(xlims)
        if ylims is not None:
            ax1.set_ylim(ylims)
        ax1.set_xlabel('Distance from the radar [km]',
                       fontsize=fsizes['fsz_axlb'], labelpad=10)
        ax1.set_ylabel('Distance from the radar [km]',
                       fontsize=fsizes['fsz_axlb'], labelpad=10)
        ax1.tick_params(direction='in', axis='both',
                        labelsize=fsizes['fsz_axtk'])
        if cbticks is not None:
            cb1.set_ticks(ticks=list(cbticks.values()),
                          labels=list(cbticks.keys()), ha='right')
        ax1.axes.set_aspect('equal')
        # ax1.annotate('| Created using Towerpy |', xy=txtboxc,
        #              fontsize=8, xycoords='axes fraction',
        #              va='center', ha='center',
        #              bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
        # ax1.grid(True)
        plt.tight_layout()
        # plt.show()

    elif cpy_features['status']:
        # ptitle = dtdes1 + dtdes2
        proj = ccrs.PlateCarree()
        if fig_size is None:
            fig_size = (12, 6)
        if data_proj:
            proj2 = data_proj
        else:
            raise TowerpyError('User must specify the projected coordinate'
                               ' system of the radar data e.g.'
                               ' ccrs.OSGB(approx=False) or ccrs.UTM(zone=32)')
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        plt.subplots_adjust(left=0.05, right=0.99, top=0.981, bottom=0.019,
                            wspace=0, hspace=1)
        ax1 = fig.add_subplot(projection=proj)
        if xlims and ylims:
            extx = xlims
            exty = ylims
            ax1.set_extent(extx+exty, crs=proj)
        if cpy_features['tiles']:
            if (cpy_features['tiles_source'] is None
               or cpy_features['tiles_source'] == 'OSM'):
                imtiles = cimgt.OSM()
                ax1.add_image(imtiles, cpy_features['tiles_res'],
                              interpolation='spline36',
                              alpha=cpy_features['alpha_tiles'])
            elif cpy_features['tiles_source'] == 'GoogleTiles':
                if cpy_features['tiles_style'] is None:
                    imtiles = cimgt.GoogleTiles(style='street')
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
                else:
                    imtiles = cimgt.GoogleTiles(
                        style=cpy_features['tiles_style'])
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  # interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
            elif cpy_features['tiles_source'] == 'QuadtreeTiles':
                if cpy_features['tiles_style'] is None:
                    imtiles = cimgt.QuadtreeTiles()
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
            elif cpy_features['tiles_source'] == 'Stamen':
                if cpy_features['tiles_style'] is None:
                    imtiles = cimgt.Stamen(style='toner')
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
                else:
                    imtiles = cimgt.Stamen(style=cpy_features['tiles_style'])
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
        if cpy_features['add_land']:
            ax1.add_feature(cfeature.LAND)
        if cpy_features['add_ocean']:
            ax1.add_feature(cfeature.OCEAN)
        if cpy_features['add_coastline']:
            ax1.add_feature(cfeature.COASTLINE)
        if cpy_features['add_borders']:
            ax1.add_feature(cfeature.BORDERS,
                            linestyle=cpy_features['borders_ls'])
        if cpy_features['add_lakes']:
            ax1.add_feature(cfeature.LAKES,
                            alpha=cpy_features['lakes_transparency'])
        if cpy_features['add_rivers']:
            ax1.add_feature(cfeature.RIVERS)
        if cpy_features['add_countries']:
            ax1.add_feature(states_provinces, edgecolor='black', ls=":")
        if cpy_features['add_provinces']:
            ax1.add_feature(countries, edgecolor='black', )

        data_source = 'Natural Earth'
        data_license = 'public domain'
        # Add a text annotation for the license information to the
        # the bottom right corner.
        # text = AnchoredText(r'$\copyright$ {}; license: {}'
        #                     ''.format(SOURCE, LICENSE),
        #                     loc=4, prop={'size': 12}, frameon=True)
        # ax1.add_artist(text)
        print('\N{COPYRIGHT SIGN}' + f'{data_source}; license: {data_license}')
        if cpy_features['tiles_source'] == 'Stamen':
            print('\N{COPYRIGHT SIGN}' + 'Map tiles by Stamen Design, '
                  + 'under CC BY 3.0. Data by OpenStreetMap, under ODbL.')
        gl = ax1.gridlines(draw_labels=True, dms=False,
                           x_inline=False, y_inline=False)
        gl.xlabel_style = {'size': fsizes['fsz_axlb']}
        gl.ylabel_style = {'size': fsizes['fsz_axlb']}
        ax1.set_title(f'{ptitle} \n' + f'PPI {var2plot}',
                      fontsize=fsizes['fsz_pt'])
        # lon_formatter = LongitudeFormatter(number_format='.4f',
        #                                 degree_symbol='',
        #                                dateline_direction_label=True)
        # lat_formatter = LatitudeFormatter(number_format='.0f',
        #                                    degree_symbol=''
        #                                   )
        mappable = ax1.pcolormesh(rad_georef[f'grid_{proj_suffix}x'],
                                  rad_georef[f'grid_{proj_suffix}y'],
                                  rad_vars[var2plot], transform=proj2,
                                  shading='auto', cmap=cmapp, norm=normp,
                                  alpha=cpy_features['alpha_rad'])
        # ax1.xaxis.set_major_formatter(lon_formatter)
        # ax1.yaxis.set_major_formatter(lat_formatter)
        if pixel_midp:
            binx = rad_georef[f'grid_{proj_suffix}x'].ravel()
            biny = rad_georef[f'grid_{proj_suffix}y'].ravel()
            ax1.scatter(binx, biny, c='grey', marker='+', transform=proj2,
                        alpha=0.2)
        if rd_maxrange:
            ax1.plot(rad_georef[f'grid_{proj_suffix}x'][:, -1],
                     rad_georef[f'grid_{proj_suffix}y'][:, -1],
                     'gray', transform=proj2)
# =============================================================================
        if points2plot is not None:
            if len(points2plot) == 2:
                ax1.scatter(points2plot[f'grid_{proj_suffix}x'],
                            points2plot[f'grid_{proj_suffix}y'], color='k',
                            marker='o', s=szpnts)
            elif len(points2plot) >= 3:
                ax1.scatter(points2plot[f'grid_{proj_suffix}x'],
                            points2plot[f'grid_{proj_suffix}y'],
                            marker='o', norm=normp, edgecolors='k', s=szpnts,
                            c=points2plot[ptsvar2plot], cmap=cmapp)

# =============================================================================
        def make_colorbar(ax1, mappable, **kwargs):
            ax1_divider = make_axes_locatable(ax1)
            orientation = kwargs.pop('orientation', 'vertical')
            if orientation == 'vertical':
                loc = 'right'
            elif orientation == 'horizontal':
                loc = 'top'
# =============================================================================
            ticks = tcks
            if var2plot in lpv.keys():
                if ticks is not None and len(tcks) > 20:
                    ticks = tcks[::5]
            else:
                None
# =============================================================================
            cax = ax1_divider.append_axes(loc, '7%', pad='15%',
                                          axes_class=plt.Axes)
            if cbticks is not None:
                ax1.get_figure().colorbar(
                    mappable, cax=cax, orientation=orientation,
                    ticks=list(cbticks.values()),
                    format=mticker.FixedFormatter(list(cbticks.keys())))
            else:
                ax1.get_figure().colorbar(mappable, cax=cax,
                                          orientation=orientation,
                                          ticks=ticks,
                                          format=f'%.{cbtks_fmt}f')
            cax.tick_params(direction='in', labelsize=fsizes['fsz_cb'])
            cax.xaxis.set_ticks_position('top')
            cax.set_title(plotunits, fontsize=fsizes['fsz_cbt'])
        make_colorbar(ax1, mappable, orientation='vertical')


def plot_setppi(rad_georef, rad_params, rad_vars, mlyr=None, vars_bounds=None,
                ucmap=None, unorm=None, cb_ext=None, xlims=None, ylims=None,
                ncols=None, nrows=None, fig_title=None, fig_size=None):
    """
    Plot a set of PPIs of polarimetric variables.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_vars : dict
        Radar variables to be plotted.
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
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    ncols : int, optional
        Number of columns used to build the grid. The default is None.
    nrows : int, optional
        Number of rows used to build the grid. The default is None.
    fig_title : str, optional
        Modify the default plot title.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
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
    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = rad_params['elev_ang [deg]']
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{2}f} deg"
    if rad_params['datetime']:
        dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    else:
        dtdes2 = ''
    if fig_title is None:
        ptitle = (f"{rad_params['site_name'].title()} "
                  + f"[{dtdes1}] -- {dtdes2}")
    else:
        ptitle = fig_title
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # fc, ec = 'w', 'k'
    if nrows is None and ncols is None:
        if len(rad_vars) <= 3:
            nrw = 1
            ncl = int(len(rad_vars))
        elif len(rad_vars) > 3 and len(rad_vars) < 10 and len(rad_vars) % 2:
            ncl = int(np.ceil(len(rad_vars)/2))
            nrw = int(np.ceil(len(rad_vars)/ncl))
        elif len(rad_vars) >= 10 and len(rad_vars) % 2:
            ncl = int(np.ceil(len(rad_vars)/4))
            nrw = int(np.ceil(len(rad_vars)/ncl))
        else:
            ncl = int(np.ceil(len(rad_vars)/2))
            nrw = int(np.ceil(len(rad_vars)/ncl))
    elif nrows is not None and ncols is None:
        if len(rad_vars) <= 3:
            nrw = nrows
            ncl = int(len(rad_vars))
        else:
            nrw = nrows
            ncl = int(np.ceil(len(rad_vars)/nrw))
    elif ncols is not None and nrows is None:
        if len(rad_vars) <= 3:
            nrw = 1
            ncl = ncols
        else:
            ncl = ncols
            nrw = int(np.ceil(len(rad_vars)/ncl))
    else:
        nrw = nrows
        ncl = ncols
        if nrw * ncl < len(rad_vars):
            print('Warning: Due to the selected grid, some variables may not '
                  + 'be displayed. Please adjust your settings to view all '
                  + 'available variables.')
    if fig_size is None and nrw != 1:
        fig_size = (16, 9)
    if fig_size is None and nrw == 1:
        fig_size = (16, 4.5)
    f, ax = plt.subplots(nrw, ncl, sharex=True, sharey=True, figsize=fig_size)
    f.suptitle(f'{ptitle}', fontsize=16)
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        None, rad_vars.keys(), vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    lpv_vars = [rkey[:rkey.find('[')-1] for rkey in lpv.keys()]
    for a, (rkey, var2plot) in zip(ax.flatten(), rad_vars.items()):
        rkey_units = rkey[rkey.find('['):]
        rkey_var = rkey[:rkey.find('[')-1]
        if rkey in lpv or rkey_var in lpv_vars or [rk for rk in lpv_vars
                                                   if rkey_var.startswith(rk)]:
            lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
                rkey, rad_vars.keys(), vars_bounds, ucmap=ucmap, unorm=unorm,
                cb_ext=cb_ext)
        else:
            lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
                rkey, rad_vars.keys(), vars_bounds, cb_ext=cb_ext)
            b1 = np.linspace(np.nanmin(var2plot), np.nanmax(var2plot), 11)
            normp = mpc.BoundaryNorm(
                b1, mpl.colormaps['tpylsc_rad_pvars'].N,
                extend=cmapext.get(rkey[rkey.find('['):], 'both'))
        cmapp = cmaph.get(rkey[rkey.find('['):],
                          mpl.colormaps['tpylsc_rad_pvars'])
        # if rkey.lower().startswith('z') and '[dBZ]' in rkey:
        #     normp = dnorm.get('[dBZ]')
        #     cmapp = mpl.colormaps['tpylsc_rad_ref']
        # if rkey.lower().startswith('zdr') and '[dB]' in rkey:
        #     normp = dnorm.get('[dB]')
        #     cmapp = mpl.colormaps['tpylsc_rad_2slope']
        # if '[0-1]' in rkey:
        #     normp = dnorm.get('[0-1]')
        # if rkey == 'rhoHV [-]':
        #     norm = [mpc.BoundaryNorm(
        #         value, cmaph.get(key[key.find('['):],
        #                          mpl.colormaps['tpylsc_rad_pvars']).N,
        #         extend='min')
        #         for key, value in bnd.items() if key == '[-]'][0]
        # if rkey == 'PIA [dB]':
        #     cmapp = mpl.colormaps['tpylsc_useq_fiery']
        f1 = a.pcolormesh(rad_georef['grid_rectx'], rad_georef['grid_recty'],
                          var2plot, shading='auto', cmap=cmapp, norm=normp)
        if mlyr is not None:
            a.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                   path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                 pe.Normal()], label=r'$MLyr_{(T)}$')
            a.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                   path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                 pe.Normal()], label=r'$MLyr_{(B)}$')
            a.legend(loc='upper left')
        if xlims is not None:
            a.set_xlim(xlims)
        if ylims is not None:
            a.set_ylim(ylims)
        # a.set_title(f'{dtdes}' "\n" f'{key}')
        a.set_title(f'{rkey}', fontsize=12)
        if nrw == 1:
            a.set_xlabel('Distance from the radar [km]', fontsize=12)
        elif ncl == 1:
            a.set_ylabel('Distance from the radar [km]', fontsize=12)
        else:
            a.set_xlabel(None, size=12)
            a.set_ylabel(None, size=12)
        a.grid(True)
        a.axes.set_aspect('equal')
        a.tick_params(axis='both', which='major', labelsize=10)
        if rkey.startswith('rhoHV'):
            f.colorbar(f1, ax=a, ticks=tcks, format=f'%.{cbtks_fmt}f')
        else:
            f.colorbar(f1, ax=a)
    if nrw*ncl > len(rad_vars):
        for empax in range(nrw*ncl-len(rad_vars)):
            f.delaxes(ax.flatten()[-empax-1])
    if ax.ndim > 1:
        plt.setp(ax[-1, :], xlabel='Distance from the radar [km]')
        plt.setp(ax[:, 0], ylabel='Distance from the radar [km]')
    if nrw == 1:
        ax[0].set_ylabel('Distance from the radar [km]', fontsize=12)
    elif ncl == 1:
        ax[-1].set_xlabel('Distance from the radar [km]', fontsize=12)
    # txtboxc = (1.025, -.10)
    # txtboxc = (-3., -.10)
    # a.annotate('| Created using Towerpy |', xy=txtboxc, fontsize=8,
    #            xycoords='axes fraction', va='center', ha='center',
    #            bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    plt.tight_layout()
    # plt.show()


def plot_mgrid(rscans_georef, rscans_params, rscans_vars, var2plot=None,
               vars_bounds=None, ucmap=None, unorm=None, cb_ext=None,
               cpy_feats=None, proj_suffix='osgb', data_proj=None,
               xlims=None, ylims=None, ncols=None, nrows=None, fig_size=None):
    """
    Graph multiple PPI scans into a grid.

    Parameters
    ----------
    rscans_georef : list
        List of georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others, corresponding to each PPI scan.
    rscans_params : list
        List of radar technical details corresponding to each PPI scan.
    rscans_vars : list
        List of Dicts containing radar variables to plot corresponding to each
        PPI scan.
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
             'Rainfall [mm/h]': [0.1, 64, 11]}
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    coord_sys : 'rect' or 'polar', optional
        Coordinates system (polar or rectangular). The default is 'polar'.
    cpy_feats : dict, optional
        Cartopy attributes to add to the map. The default are:
        {
        'status': False,
        'add_land': False,
        'add_ocean': False,
        'add_coastline': False,
        'add_borders': False,
        'add_countries': True,
        'add_provinces': True,
        'borders_ls': ':',
        'add_lakes': False,
        'lakes_transparency': 0.5,
        'add_rivers': False,
        'tiles': False,
        'tiles_source': None,
        'tiles_style': None,
        }
    proj_suffix : str
        Suffix of the georeferenced grids used to display the data.
        The X/Y grids must exist in the rad_georef dictionary, e.g.
        'grid_osgbx--grid_osgby', 'grid_utmx--grid_utmy',
        'grid_wgs84x--grid_wgs84y', etc. The default is 'osgb'.
    data_proj : Cartopy Coordinate Reference System object, optional
        Cartopy projection used to plot the data in a map e.g.,
        ccrs.OSGB(approx=False).
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    ncols : int, optional
        Set the number of columns used to build the grid. The default is None.
    nrows : int, optional
        Set the number of rows used to build the grid. The default is None.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    from mpl_toolkits.axes_grid1 import ImageGrid
    from cartopy.mpl.geoaxes import GeoAxes

    dskeys = [k for i in rscans_vars for k in i.keys()]
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        var2plot,
        ('ZH [dBZ]' if all('ZH [dBZ]' in i.keys() for i in rscans_vars)
         else list(set([x for x in dskeys
                        if dskeys.count(x) >= len(rscans_vars)
                        and '[' in x]))),
        vars_bounds, ucmap=ucmap, unorm=unorm, cb_ext=cb_ext)
    if var2plot is None:
        var2plot = v2p
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # txtboxc = (0, -.09)
    # fc, ec = 'w', 'k'
    cmapp = cmaph.get(var2plot[var2plot.find('['):],
                      mpl.colormaps['tpylsc_rad_pvars'])
    cpy_features = {'status': False,
                    # 'coastresolution': '10m',
                    'add_land': False,
                    'add_ocean': False,
                    'add_coastline': False,
                    'add_borders': False,
                    'add_countries': True,
                    'add_provinces': True,
                    'borders_ls': ':',
                    'add_lakes': False,
                    'lakes_transparency': 0.5,
                    'add_rivers': False,
                    'tiles': False,
                    'tiles_source': None,
                    'tiles_style': None,
                    'tiles_res': 8, 'alpha_tiles': 0.5, 'alpha_rad': 1
                    }
    if cpy_feats:
        cpy_features.update(cpy_feats)
    if cpy_features['status']:
        states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_1_states_provinces_lines',
            scale='10m',
            facecolor='none')
        countries = cfeature.NaturalEarthFeature(
            category='cultural',
            name='admin_0_countries',
            scale='10m',
            facecolor='none')
    # TODO add fig_title
    pttl = [f"{p['elev_ang [deg]']} -- "
            + f"{p['datetime']:%Y-%m-%d %H:%M:%S}"
            if isinstance(p['elev_ang [deg]'], str)
            else
            f"{p['elev_ang [deg]']:{2}.{2}f} deg. -- "
            + f"{p['datetime']:%Y-%m-%d %H:%M:%S}"
            for p in rscans_params]
    if nrows is None and ncols is None:
        if len(rscans_vars) <= 3:
            nrw = 1
            ncl = int(len(rscans_vars))
        elif len(rscans_vars) > 3 and len(rscans_vars) < 10 and len(rscans_vars) % 2:
            ncl = int(np.ceil(len(rscans_vars)/2))
            nrw = int(np.ceil(len(rscans_vars)/ncl))
        elif len(rscans_vars) >= 10 and len(rscans_vars) % 2:
            ncl = int(np.ceil(len(rscans_vars)/4))
            nrw = int(np.ceil(len(rscans_vars)/ncl))
        else:
            ncl = int(np.ceil(len(rscans_vars)/2))
            nrw = int(np.ceil(len(rscans_vars)/ncl))
    elif nrows is not None and ncols is None:
        if len(rscans_vars) <= 3:
            nrw = nrows
            ncl = int(len(rscans_vars))
        else:
            nrw = nrows
            ncl = int(np.ceil(len(rscans_vars)/nrw))
    elif ncols is not None and nrows is None:
        if len(rscans_vars) <= 3:
            nrw = 1
            ncl = ncols
        else:
            ncl = ncols
            nrw = int(np.ceil(len(rscans_vars)/ncl))
    else:
        nrw = nrows
        ncl = ncols
        if nrw * ncl < len(rscans_vars):
            print('Warning: Due to the selected grid, some variables may not '
                  + 'be displayed. Please adjust your settings to view all '
                  + 'available variables.')
    if cpy_features['status'] is False:
        if fig_size is None:
            fig_size = (15, 5)
        fig = plt.figure(figsize=fig_size)
        grgeor = [[i['grid_rectx'], i['grid_recty']] for i in rscans_georef]
        grid2 = ImageGrid(fig, 111, nrows_ncols=(nrw, ncl), label_mode="L",
                          cbar_location="right", cbar_mode="single",
                          cbar_size="10%", cbar_pad=0.25, axes_pad=(0.5, 0.75),
                          share_all=True)
        for ax, z, g, pr, pt in zip(grid2, [i[var2plot] for i in rscans_vars],
                                    grgeor, rscans_params, pttl):
            f1 = ax.pcolormesh(g[0], g[1], z, shading='auto', cmap=cmapp,
                               norm=normp)
            ax.set_title(f"{pt} \n {pr['site_name']} - PPI {var2plot}",
                         fontsize=12)
            ax.set_xlabel('Distance from the radar [km]', fontsize=12)
            ax.set_ylabel('Distance from the radar [km]', fontsize=12)
            ax.grid(True)
            ax.axes.set_aspect('equal')
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
        if (var2plot == 'rhoHV [-]' or '[mm]' in var2plot
           or '[mm/h]' in var2plot):
            ax.cax.colorbar(f1, ticks=tcks, format=f'%.{cbtks_fmt}f')
        else:
            ax.cax.colorbar(f1)
        ax.cax.tick_params(direction='in', which='both', labelsize=12)
        ax.cax.set_title(var2plot[var2plot .find('['):], fontsize=12)
        # for ax, im_title in zip(grid2, ["(a)", "(b)", "(c)"]):
        #     t = add_inner_title(ax, im_title, loc='upper left')
        #     t.patch.set_ec("none")
        #     t.patch.set_alpha(0.5)
        if nrw*ncl > len(rscans_vars):
            for empax in range(nrw*ncl-len(rscans_vars)):
                grid2[-empax-1].remove()
        plt.tight_layout()
        plt.show()
    elif cpy_features['status']:
        if fig_size is None:
            fig_size = (16, 6)
        fig = plt.figure(figsize=fig_size)
        projection = ccrs.PlateCarree()
        axes_class = (GeoAxes, dict(map_projection=projection))
        grgeor = [[i[f'grid_{proj_suffix}x'], i[f'grid_{proj_suffix}y']]
                  for i in rscans_georef]
        grid2 = ImageGrid(fig, 111, nrows_ncols=(nrw, ncl), axes_pad=(.6, .9),
                          label_mode="L", cbar_mode="single", cbar_size="10%",
                          cbar_pad=0.75, cbar_location="right", share_all=True,
                          axes_class=axes_class)
        if data_proj:
            proj2 = data_proj
        else:
            raise TowerpyError('User must specify the projected coordinate'
                               ' system of the radar data e.g.'
                               ' ccrs.OSGB(approx=False) or ccrs.UTM(zone=32)')
        for ax1, z, g, pr, pt in zip(grid2, [i[var2plot] for i in rscans_vars],
                                     grgeor, rscans_params, pttl):
            ax1.set_title(f"{pt} \n {pr['site_name']} - PPI {var2plot}",
                          fontsize=12)
            if xlims and ylims:
                extx = xlims
                exty = ylims
                ax1.set_extent(extx+exty, crs=projection)
            if cpy_features['tiles']:
                if (cpy_features['tiles_source'] is None
                        or cpy_features['tiles_source'] == 'OSM'):
                    imtiles = cimgt.OSM()
                    ax1.add_image(imtiles, cpy_features['tiles_res'],
                                  interpolation='spline36',
                                  alpha=cpy_features['alpha_tiles'])
                elif cpy_features['tiles_source'] == 'GoogleTiles':
                    if cpy_features['tiles_style'] is None:
                        imtiles = cimgt.GoogleTiles(style='street')
                        ax1.add_image(imtiles, cpy_features['tiles_res'],
                                      interpolation='spline36',
                                      alpha=cpy_features['alpha_tiles'])
                    else:
                        imtiles = cimgt.GoogleTiles(
                            style=cpy_features['tiles_style'])
                        ax1.add_image(imtiles, cpy_features['tiles_res'],
                                      # interpolation='spline36',
                                      alpha=cpy_features['alpha_tiles'])
                elif cpy_features['tiles_source'] == 'Stamen':
                    if cpy_features['tiles_style'] is None:
                        imtiles = cimgt.Stamen(style='toner')
                        ax1.add_image(imtiles, cpy_features['tiles_res'],
                                      interpolation='spline36',
                                      alpha=cpy_features['alpha_tiles'])
                    else:
                        imtiles = cimgt.Stamen(
                            style=cpy_features['tiles_style'])
                        ax1.add_image(imtiles, cpy_features['tiles_res'],
                                      interpolation='spline36',
                                      alpha=cpy_features['alpha_tiles'])
            if cpy_features['add_land']:
                ax1.add_feature(cfeature.LAND)
            if cpy_features['add_ocean']:
                ax1.add_feature(cfeature.OCEAN)
            if cpy_features['add_coastline']:
                ax1.add_feature(cfeature.COASTLINE)
            if cpy_features['add_borders']:
                ax1.add_feature(cfeature.BORDERS,
                                linestyle=cpy_features['borders_ls'])
            if cpy_features['add_lakes']:
                ax1.add_feature(cfeature.LAKES,
                                alpha=cpy_features['lakes_transparency'])
            if cpy_features['add_rivers']:
                ax1.add_feature(cfeature.RIVERS)
            if cpy_features['add_countries']:
                ax1.add_feature(states_provinces, edgecolor='black', ls=":")
            if cpy_features['add_provinces']:
                ax1.add_feature(countries, edgecolor='black')
            data_source = 'Natural Earth'
            data_license = 'public domain'
            # Add a text annotation for the license information to the
            # the bottom right corner.
            # text = AnchoredText(r'$\copyright$ {}; license: {}'
            #                     ''.format(SOURCE, LICENSE),
            #                     loc=4, prop={'size': 12}, frameon=True)
            # ax1.add_artist(text)
            print('\N{COPYRIGHT SIGN}'
                  + f'{data_source}; license: {data_license}')
            if cpy_features['tiles_source'] == 'Stamen':
                print('\N{COPYRIGHT SIGN}' + 'Map tiles by Stamen Design, '
                      + 'under CC BY 3.0. Data by OpenStreetMap, under ODbL.')
            gl = ax1.gridlines(draw_labels=True, dms=False,
                               x_inline=False, y_inline=False)
            gl.xlabel_style = {'size': 11}
            gl.ylabel_style = {'size': 11}
            gl.top_labels = False
            gl.right_labels = False
            # ax1.set_title(f'{ptitle} \n' + f'PPI {var2plot}', fontsize=14)
            # lon_formatter = LongitudeFormatter(number_format='.4f',
            #                                 degree_symbol='',
            #                                dateline_direction_label=True)
            # lat_formatter = LatitudeFormatter(number_format='.0f',
            #                                    degree_symbol=''
            #                                   )
            # ax1.xaxis.set_major_formatter(lon_formatter)
            # ax1.yaxis.set_major_formatter(lat_formatter)
            # plotunits = [i[i.find('['):]
            #              for i in rad_vars.keys() if var2plot == i][0]
            mappable = ax1.pcolormesh(g[0], g[1], z, transform=proj2,
                                      shading='auto', cmap=cmapp, norm=normp,
                                      alpha=cpy_features['alpha_rad'])
            if (var2plot == 'rhoHV [-]' or '[mm]' in var2plot
               or '[mm/h]' in var2plot):
                # ticks = bnd.get(var2plot[var2plot.find('['):])
                if len(tcks) > 20:
                    tcks = tcks[::5]
                grid2.cbar_axes[0].colorbar(mappable, ticks=tcks,
                                            format=f'%.{cbtks_fmt}f')
            else:
                grid2.cbar_axes[0].colorbar(mappable)
            ax1.cax.set_title(var2plot[var2plot .find('['):], fontsize=12)
            # ax1.axes.set_aspect('equal')
        if nrw*ncl > len(rscans_vars):
            for empax in range(nrw*ncl-len(rscans_vars)):
                grid2[-empax-1].remove()
        plt.tight_layout()
        plt.show()


def plot_cone_coverage(rad_georef, rad_params, rad_vars, var2plot=None,
                       vars_bounds=None, xlims=None, ylims=None, zlims=[0, 8],
                       limh=8, ucmap=None, unorm=None, cbticks=None,
                       cb_ext=None, fig_size=None):
    """
    Display a 3-D representation of the radar cone coverage.

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
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    zlims : 2-element tuple or list, optional
        Set the z-axis view limits [min, max]. The default is None.
    limh : int or float, optional
        Set a height limit to the plot. The default is None.
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
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    from matplotlib.colors import LightSource

    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        var2plot, rad_vars.keys(), vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    #
    if var2plot is None:
        var2plot = v2p
    #
    cmapp = cmaph.get(var2plot[var2plot.find('['):],
                      mpl.colormaps['tpylsc_rad_pvars'])
    # dtdes0 = f"[{rad_params['site_name']}]"
    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{2}} deg. -- "
    if rad_params['datetime']:
        dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    else:
        dtdes2 = ''
    ptitle = dtdes1 + dtdes2
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # txtboxc = (0, -.09)
    # fc, ec = 'w', 'k'

    limidx = [rut.find_nearest(row, limh)
              for row in rad_georef['beam_height [km]']]

    m = np.ma.masked_invalid(rad_vars[var2plot]).mask
    for n, rows in enumerate(m):
        rows[limidx[n]:] = 1
    R = rad_vars[var2plot]

    X, Y = rad_georef['grid_rectx'], rad_georef['grid_recty']
    Z = np.resize(rad_georef['beam_height [km]'], R.shape)
    Z = np.resize(rad_georef['beam_height [km]'], R.shape)

    X = np.ma.array(X, mask=m)
    Y = np.ma.array(Y, mask=m)
    Z = np.ma.array(Z, mask=m)
    R = np.ma.array(R, mask=m)

    ls = LightSource(0, 0)

    rgb = ls.shade(R, cmap=cmapp, norm=normp, vert_exag=0.1, blend_mode='soft')
    if fig_size is None:
        fig_size = (12, 8)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=fig_size)

    # Plot the surface.
    ax.plot_surface(X, Y, Z, cmap=cmapp, norm=normp, facecolors=rgb,
                    rstride=1, cstride=8,
                    # rcount=360, ccount=600,
                    # rcount=360, ccount=150,
                    linewidth=0, antialiased=True, shade=False,)
    if cbticks is not None:
        mappable2 = ax.contourf(X, Y, R, zdir='z', offset=0, levels=tcks,
                                cmap=cmapp, norm=normp, antialiased=True)
    else:
        mappable2 = ax.contourf(X, Y, R, zdir='z', offset=0,
                                levels=normp.boundaries,
                                cmap=cmapp, norm=normp, extend=normp.extend,
                                antialiased=True)
    # Customize the axis.
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)
    ax.view_init(elev=10)
    ax.tick_params(axis='both', labelsize=14)

    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    if cbticks is not None:
        cb1 = fig.colorbar(mappable2, shrink=0.4, aspect=5, norm=normp,
                           cmap=cmapp, ticks=list(cbticks.values()),
                           format=mticker.FixedFormatter(list(cbticks.keys())))
    else:
        if (var2plot == 'rhoHV [-]' or '[mm]' in var2plot
           or '[mm/h]' in var2plot):
            cb1 = fig.colorbar(
                mappable2, shrink=0.4, aspect=5, norm=normp, cmap=cmapp,
                ticks=tcks, format=f'%.{cbtks_fmt}f')
        else:
            cb1 = fig.colorbar(
                mappable2, shrink=0.4, aspect=5, norm=normp, cmap=cmapp)
    cb1.ax.tick_params(direction='in', axis='both', labelsize=14)
    cb1.ax.set_title(var2plot[var2plot .find('['):], fontsize=14)

    ax.set_title(f'{ptitle} \n' + f'PPI {var2plot}', fontsize=16)
    ax.set_xlabel('Distance from the radar [km]', fontsize=14, labelpad=15)
    ax.set_ylabel('Distance from the radar [km]', fontsize=14, labelpad=15)
    ax.set_zlabel('Height [km]', fontsize=14, labelpad=15)
    plt.tight_layout()
    plt.show()


def plot_snr(rad_georef, rad_params, snr_data, min_snr, coord_sys='polar',
             ucmap_snr=None, fig_size=None):
    """
    Display the results of the SNR classification.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    snr_data : dict
        Results of the SNR_Classif method.
    proj : 'rect' or 'polar', optional
        Coordinates system (polar or rectangular). The default is 'polar'.
    """
    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{2}} deg. -- "
    if rad_params['datetime']:
        dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    else:
        dtdes2 = ''
    
    ptitle = dtdes1 + dtdes2
    if fig_size is None:
        fig_size = (10.5, 6.5)
    if coord_sys == 'polar':
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=fig_size,
                                       subplot_kw=dict(projection='polar'))
        f2 = ax2.pcolormesh(
            *np.meshgrid(rad_georef['azim [rad]'],
                         rad_georef['range [m]'] / 1000, indexing='ij'),
            np.flipud(snr_data['snr [dB]']), shading='auto',
            cmap='tpylsc_rad_ref')
        # ax2.axes.set_aspect('equal')
        ax2.grid(color='gray', linestyle=':')
        ax2.set_theta_zero_location('N')
        ax2.set_thetagrids(np.arange(0, 360, 90))
        # ax2.set_yticklabels([])
        cb2 = plt.colorbar(f2, ax=ax2, extend='both', orientation='horizontal',
                           # shrink=0.5,
                           )
        cb2.ax.tick_params(direction='in', axis='both', labelsize=14)
        cb2.ax.set_title('SNR [dB]', fontsize=14, y=-2.5)

        ax3.set_title(f'Signal (SNR > minSNR [{min_snr:.2f}])', fontsize=14,
                      y=-0.15)
        ax3.pcolormesh(
            *np.meshgrid(rad_georef['azim [rad]'],
                         rad_georef['range [m]'] / 1000, indexing='ij'),
            np.flipud(snr_data['snrclass']), shading='auto',
            cmap=mpl.colormaps['tpylc_div_yw_gy_bu'])
        # ax3.axes.set_aspect('equal')
        ax3.grid(color='w', linestyle=':')
        ax3.set_theta_zero_location('N')
        ax3.set_thetagrids(np.arange(0, 360, 90))
        # ax3.set_yticklabels([])
        mpl.colormaps['tpylc_div_yw_gy_bu'].set_bad(color='#505050')
        plt.show()

    elif coord_sys == 'rect':
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=fig_size,
                                       sharex=True, sharey=True)
        fig.suptitle(f'{ptitle}', fontsize=16)
        # Plots the SNR
        f2 = ax2.pcolormesh(rad_georef['grid_rectx'], rad_georef['grid_recty'],
                            snr_data['snr [dB]'], shading='auto',
                            cmap='tpylsc_rad_ref')
        ax2_divider = make_axes_locatable(ax2)
        cax2 = ax2_divider.append_axes("top", size="7%", pad="2%")
        cb2 = fig.colorbar(f2, cax=cax2, extend='max',
                           orientation='horizontal')
        cb2.ax.tick_params(direction='in', labelsize=10)
        # cb2.ax.set_xticklabels(cb2.ax.get_xticklabels(), rotation=90)
        cb2.ax.set_title('SNR [dB]', fontsize=14)
        # cb2.ax.set_ylabel('[dB]', fontsize=12, labelpad=0)
        cax2.xaxis.set_ticks_position("top")
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.set_ylabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax2.set_xlabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax2.axes.set_aspect('equal')
        # Plots the Signal detection
        f3 = ax3.pcolormesh(rad_georef['grid_rectx'], rad_georef['grid_recty'],
                            snr_data['snrclass'],
                            cmap=mpl.colormaps['tpylc_div_yw_gy_bu'],
                            vmin=0, vmax=6,
                            )
        ax3_divider = make_axes_locatable(ax3)
        cax3 = ax3_divider.append_axes("top", size="7%", pad="2%")
        cb3 = fig.colorbar(f3, cax=cax3, orientation='horizontal')
        cb3.ax.tick_params(direction='in', labelsize=10)
        cb3.ax.set_title(f'Signal detection - SNR >= minSNR [{min_snr:.2f}]',
                         fontsize=14)
        cax3.xaxis.set_ticks_position("top")
        cb3.set_ticks(ticks=[1., 3., 6.],
                      labels=['Signal', 'Noise', ''])
        ax3.set_xlabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax3.tick_params(axis='both', which='major', labelsize=10)
        ax3.axes.set_aspect('equal')
        plt.tight_layout()
        plt.show()


def plot_nmeclassif(rad_georef, rad_params, nme_classif, echoesID,
                    clutter_map=None, xlims=None, ylims=None, fig_size=None):
    """
    Plot a set of PPIs of polarimetric variables.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    nme_classif : dict
        Results of the NME_ID method.
    clutter_map : array, optional
        Clutter map used for the NME_ID method. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    """
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # fc, ec = 'w', 'k'
    # =========================================================================
    # Plot the Clutter classification
    # =========================================================================
    if fig_size is None:
        fig_size = (6, 6.15)
    clcdummy = nme_classif[nme_classif == echoesID['clutter']]
    if not clcdummy.size:
        nme_classif[0, 0] = echoesID['clutter']
    plot_ppi(rad_georef, rad_params, {'classif [EC]': nme_classif},
             cbticks=echoesID, ucmap='tpylc_div_yw_gy_bu')
    plt.tight_layout()
    # # txtboxc = (0, -.09)
    # # ax.annotate('| Created using Towerpy |', xy=txtboxc, fontsize=8,
    # #             xycoords='axes fraction', va='center', ha='center',
    # #             bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
    # =========================================================================
    # Plot the Clutter Map
    # =========================================================================
    if clutter_map is not None:
        norm = mpc.BoundaryNorm(boundaries=np.linspace(0, 100, 11),
                                ncolors=256)
        plot_ppi(rad_georef, rad_params,
                 {'Clutter probability [%]': clutter_map*100},
                 unorm={'Clutter probability [%]': norm},
                 ucmap='tpylsc_useq_bupkyw')
        # txtboxc = (0, -.09)
        # ax.annotate('| Created using Towerpy |', xy=txtboxc, fontsize=8,
        #             xycoords='axes fraction', va='center', ha='center',
        #             bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
        plt.tight_layout()
    plt.show()


def plot_zhattcorr(rad_georef, rad_params, rad_vars_att, rad_vars_attcorr,
                   vars_bounds=None, mlyr=None, xlims=None, ylims=None,
                   fig_size1=None, fig_size2=None):
    """
    Plot the results of the ZH attenuation correction method.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_vars_att : dict
        Radar variables not corrected for attenuation.
    rad_vars_attcorr : dict
        Results of the AttenuationCorection method.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped. The default are:
            {'ZH [dBZ]': [-10, 60, 15],
             'ZDR [dB]': [-2, 6, 17],
             'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
             'rhoHV [-]': [0.3, .9, 1],
             'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
             'LDR [dB]': [-35, 0, 11],
             'Rainfall [mm/h]': [0.1, 64, 11]}
    mlyr : MeltingLayer Class, optional
        Plot the melting layer height. ml_top (float, int, list or np.array)
        and ml_bottom (float, int, list or np.array) must be explicitly
        defined. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    """
    lpv = {'ZH [dBZ]': [-10, 60, 15], 'PhiDP [deg]': [0, 180, 10],
           'KDP [deg/km]': [-2, 6, 17], 'AH [dB/km]': [0, .1, 11],
           'alpha [-]': [0, 0.2, 11]}
    if vars_bounds is not None:
        lpv.update(vars_bounds)
# =============================================================================
    bnd = {key[key.find('['):]: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
# =============================================================================
    dnorm = {key: mpc.BoundaryNorm(
        value, mpl.colormaps['tpylsc_rad_pvars'].N, extend='both')
             for key, value in bnd.items()}
    if '[dBZ]' in bnd.keys():
        dnorm['[dBZ]'] = mpc.BoundaryNorm(
            bnd['[dBZ]'], mpl.colormaps['tpylsc_rad_ref'].N, extend='both')
    if '[dB]' in bnd.keys():
        dnorm['[dB]'] = mpc.BoundaryNorm(
            bnd['[dB]'], mpl.colormaps['tpylsc_rad_2slope'].N, extend='both')
    if '[dB/km]' in bnd.keys():
        dnorm['[dB/km]'] = mpc.BoundaryNorm(
            bnd['[dB/km]'], mpl.colormaps['tpylsc_rad_pvars'].N,
            extend='max')
    if '[-]' in bnd.keys():
        dnorm['[-]'] = mpc.BoundaryNorm(
            bnd['[-]'], mpl.colormaps['tpylsc_useq_fiery'].N,
            extend='neither')
# =============================================================================
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

    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{2}} deg. -- "
    if rad_params['datetime']:
        dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    else:
        dtdes2 = ''
    ptitle = dtdes1 + dtdes2

    # =========================================================================
    # Creates plots for ZH attenuation correction results.
    # =========================================================================
    mosaic = 'ABC'
    if fig_size1 is None:
        fig_size1 = (16, 5)
    if fig_size2 is None:
        fig_size2 = (6, 5)

    fig_mos1 = plt.figure(figsize=fig_size1, constrained_layout=True)
    ax_idx = fig_mos1.subplot_mosaic(mosaic, sharex=True, sharey=True)
    for key, value in rad_vars_att.items():
        if '[dBZ]' in key:
            cmap = mpl.colormaps['tpylsc_rad_ref']
            # norm = dnorm.get('n'+key)
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['A'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['A'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx['A'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['A'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['A'].legend(loc='upper left')
    if xlims is not None:
        ax_idx['A'].set_xlim(xlims)
    if ylims is not None:
        ax_idx['A'].set_ylim(ylims)
    plt.colorbar(fzhna, ax=ax_idx['A']).ax.tick_params(labelsize=10)
    ax_idx['A'].grid(True)
    ax_idx['A'].axes.set_aspect('equal')
    ax_idx['A'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if '[dBZ]' in key:
            cmap = mpl.colormaps['tpylsc_rad_ref']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['B'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['B'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx['B'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['B'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['B'].legend(loc='upper left')
    if xlims is not None:
        ax_idx['B'].set_xlim(xlims)
    if ylims is not None:
        ax_idx['B'].set_ylim(ylims)
    plt.colorbar(fzhna, ax=ax_idx['B']).ax.tick_params(labelsize=10)
    ax_idx['B'].grid(True)
    ax_idx['B'].axes.set_aspect('equal')
    ax_idx['B'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if 'AH' in key:
            cmap = mpl.colormaps['tpylsc_rad_pvars']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['C'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm
                                           )
            ax_idx['C'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['C'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['C'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['C'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx['C']).ax.tick_params(labelsize=10)
    ax_idx['C'].grid(True)
    ax_idx['C'].axes.set_aspect('equal')
    ax_idx['C'].tick_params(axis='both', labelsize=10)

    # =========================================================================
    # Creates plots for PHIDP attenuation correction results.
    # =========================================================================
    fig_mos2 = plt.figure(figsize=fig_size1, constrained_layout=True)
    ax_idx2 = fig_mos2.subplot_mosaic(mosaic, sharex=True, sharey=True)
    for key, value in rad_vars_att.items():
        if '[deg]' in key:
            cmap = mpl.colormaps['tpylsc_rad_pvars']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx2['A'].pcolormesh(rad_georef['grid_rectx'],
                                            rad_georef['grid_recty'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['A'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx2['A'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx2['A'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx2['A'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx2['A']).ax.tick_params(labelsize=10)
    ax_idx2['A'].grid(True)
    ax_idx2['A'].axes.set_aspect('equal')
    ax_idx2['A'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if key == 'PhiDP [deg]':
            cmap = mpl.colormaps['tpylsc_rad_pvars']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx2['B'].pcolormesh(rad_georef['grid_rectx'],
                                            rad_georef['grid_recty'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['B'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx2['B'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx2['B'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx2['B'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx2['B']).ax.tick_params(labelsize=10)
    ax_idx2['B'].grid(True)
    ax_idx2['B'].axes.set_aspect('equal')
    ax_idx2['B'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if key == 'PhiDP* [deg]':
            cmap = mpl.colormaps['tpylsc_rad_pvars']
            # norm = dnorm.get('n'+key.replace('*', ''))
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx2['C'].pcolormesh(rad_georef['grid_rectx'],
                                            rad_georef['grid_recty'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['C'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx2['C'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx2['C'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                          path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                        pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx2['C'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx2['C']).ax.tick_params(labelsize=10)
    ax_idx2['C'].grid(True)
    ax_idx2['C'].axes.set_aspect('equal')
    ax_idx2['C'].tick_params(axis='both', labelsize=10)

    # =========================================================================
    # Creates plots for attenuation correction vars.
    # =========================================================================
    fig_mos3, ax_idx3 = plt.subplots(figsize=fig_size2)
    for key, value in rad_vars_attcorr.items():
        if key == 'alpha [-]':
            # cmap = 'tpylsc_rad_pvars'
            cmap = 'tpylsc_useq_fiery'
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx3.pcolormesh(rad_georef['grid_rectx'],
                                       rad_georef['grid_recty'], value,
                                       shading='auto', cmap=cmap,
                                       norm=norm
                                       )
            ax_idx3.set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx3.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx3.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx3.legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx3).ax.tick_params(labelsize=10)
    ax_idx3.grid(True)
    ax_idx3.axes.set_aspect('equal')
    ax_idx3.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    fig_mos4, ax_idx4 = plt.subplots(figsize=fig_size2)
    for key, value in rad_vars_attcorr.items():
        if 'PIA' in key:
            # cmap = 'tpylsc_rad_pvars'
            cmap = 'tpylsc_useq_fiery'
            fzhna = ax_idx4.pcolormesh(rad_georef['grid_rectx'],
                                       rad_georef['grid_recty'], value,
                                       shading='auto', cmap=cmap,
                                       # norm=norm
                                       )
            ax_idx4.set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx4.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx4.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx4.legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx4).ax.tick_params(labelsize=10)
    ax_idx4.grid(True)
    ax_idx4.axes.set_aspect('equal')
    ax_idx4.tick_params(axis='both', labelsize=10)
    plt.tight_layout()


def plot_zdrattcorr(rad_georef, rad_params, rad_vars_att, rad_vars_attcorr,
                    vars_bounds=None, mlyr=None, xlims=None, ylims=None,
                    fig_size1=None, fig_size2=None):
    """
    Plot the results of the ZDR attenuation correction method.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_vars_att : dict
        Radar variables not corrected for attenuation.
    rad_vars_attcorr : dict
        Results of the AttenuationCorection method.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped. The default are:
            {'ZH [dBZ]': [-10, 60, 15],
             'ZDR [dB]': [-2, 6, 17],
             'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
             'rhoHV [-]': [0.3, .9, 1],
             'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
             'LDR [dB]': [-35, 0, 11],
             'Rainfall [mm/h]': [0.1, 64, 11]}
    mlyr : MeltingLayer Class, optional
        Plot the melting layer height. ml_top (float, int, list or np.array)
        and ml_bottom (float, int, list or np.array) must be explicitly
        defined. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    """
    lpv = {'ZDR [dB]': [-2, 6, 17], 'ADP [dB/km]': [0, 2.5, 20],
           'beta [-]': [0, 0.1, 11]}
    if vars_bounds is not None:
        lpv.update(vars_bounds)
# =============================================================================
    bnd = {key[key.find('['):]: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
# =============================================================================
    dnorm = {key: mpc.BoundaryNorm(
        value, mpl.colormaps['tpylsc_rad_pvars'].N, extend='both')
             for key, value in bnd.items()}
    if '[dBZ]' in bnd.keys():
        dnorm['[dBZ]'] = mpc.BoundaryNorm(
            bnd['[dBZ]'], mpl.colormaps['tpylsc_rad_ref'].N, extend='both')
    if '[dB]' in bnd.keys():
        dnorm['[dB]'] = mpc.BoundaryNorm(
            bnd['[dB]'], mpl.colormaps['tpylsc_rad_2slope'].N, extend='both')
    if '[dB/km]' in bnd.keys():
        dnorm['[dB/km]'] = mpc.BoundaryNorm(
            bnd['[dB/km]'], mpl.colormaps['tpylsc_rad_pvars'].N,
            extend='max')
    if '[-]' in bnd.keys():
        dnorm['[-]'] = mpc.BoundaryNorm(
            bnd['[-]'], mpl.colormaps['tpylsc_useq_fiery'].N,
            extend='max')
# =============================================================================
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

    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{2}} deg. -- "
    if rad_params['datetime']:
        dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    else:
        dtdes2 = ''
    ptitle = dtdes1 + dtdes2

    # =========================================================================
    # Creates plots for ZDR attenuation correction results.
    # =========================================================================
    mosaic = 'DEF'
    if fig_size1 is None:
        fig_size1 = (16, 5)
    if fig_size2 is None:
        fig_size2 = (6, 5)

    fig_mos1 = plt.figure(figsize=fig_size1, constrained_layout=True)
    ax_idx = fig_mos1.subplot_mosaic(mosaic, sharex=True, sharey=True)
    for key, value in rad_vars_att.items():
        if '[dB]' in key:
            cmap = mpl.colormaps['tpylsc_rad_2slope']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['D'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['D'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx['D'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['D'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['D'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx['D']).ax.tick_params(labelsize=10)
    ax_idx['D'].grid(True)
    ax_idx['D'].axes.set_aspect('equal')
    ax_idx['D'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if '[dB]' in key:
            cmap = mpl.colormaps['tpylsc_rad_2slope']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['E'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['E'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx['E'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['E'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['E'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx['E']).ax.tick_params(labelsize=10)
    ax_idx['E'].grid(True)
    ax_idx['E'].axes.set_aspect('equal')
    ax_idx['E'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if 'ADP' in key:
            cmap = mpl.colormaps['tpylsc_rad_pvars']
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx['F'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['F'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['F'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['F'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['F'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx['F']).ax.tick_params(labelsize=10)
    ax_idx['F'].grid(True)
    ax_idx['F'].axes.set_aspect('equal')
    ax_idx['F'].tick_params(axis='both', labelsize=10)

    # =========================================================================
    # Creates plots for attenuation correction vars.
    # =========================================================================
    fig_mos3, ax_idx3 = plt.subplots(figsize=fig_size2)
    for key, value in rad_vars_attcorr.items():
        if key == 'beta [-]':
            cmap = 'tpylsc_useq_fiery'
            norm = dnorm.get(key[key.find('['):])
            fzhna = ax_idx3.pcolormesh(rad_georef['grid_rectx'],
                                       rad_georef['grid_recty'], value,
                                       shading='auto', cmap=cmap, norm=norm)
            ax_idx3.set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx3.plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx3.plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                     path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                   pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx3.legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx3).ax.tick_params(labelsize=10)
    ax_idx3.grid(True)
    ax_idx3.axes.set_aspect('equal')
    ax_idx3.tick_params(axis='both', labelsize=10)
    plt.tight_layout()


def plot_radprofiles(rad_profs, beam_height, mlyr=None, stats=None, ylims=None,
                     vars_bounds=None, colours=False, unorm=None, ucmap=None,
                     cb_ext=None, fig_size=None):
    """
    Display a set of profiles of polarimetric variables.

    Parameters
    ----------
    rad_profs : dict
        Profiles generated by the PolarimetricProfiles class.
    beam_height : array
        The beam height.
    mlyr : MeltingLayer Class, optional
        Plots the melting layer within the polarimetric profiles.
        The default is None.
    stats : dict, optional
        Statistics of the profiles generation computed by the
        PolarimetricProfiles class. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max] between which radar variables are
        to be plotted.
    colours : Bool, optional
        Creates coloured profiles using norm to map colormaps.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    fontsizelabels = 20
    fontsizetitle = 25
    fontsizetick = 18
    prftype = getattr(rad_profs, 'profs_type').lower()

    # ttxt_elev = f"{rad_profs.elev_angle:{2}.{2}} Deg."
    # ttxt_dt = f"{rad_profs.scandatetime:%Y-%m-%d %H:%M:%S}"
    # ttxt = dtdes1+ttxt_dt
    if isinstance(rad_profs.elev_angle, str):
        dtdes1 = f"{rad_profs.elev_angle} -- "
    else:
        dtdes1 = f"{rad_profs.elev_angle:{2}.{2}} deg. -- "
    dtdes2 = f"{rad_profs.scandatetime:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2

    if fig_size is None:
        fig_size = (14, 10)

    def make_colorbar(ax1, mappable, **kwargs):
        ax1_divider = make_axes_locatable(ax1)
        orientation = kwargs.pop('orientation', 'vertical')
        if orientation == 'vertical':
            loc = 'right'
        elif orientation == 'horizontal':
            loc = 'top'
        cax = ax1_divider.append_axes(loc, '7%', pad='2.5%',
                                      axes_class=plt.Axes)
        ax1.get_figure().colorbar(mappable, cax=cax,
                                  orientation=orientation,
                                  ticks=ticks,
                                  format=f'%.{cbtks_fmt}f')
        cax.tick_params(direction='in', labelsize=10, rotation=90)
        cax.xaxis.set_ticks_position('top')
    if rad_profs.profs_type.lower() == 'vps':
        rprofs = rad_profs.vps
        fig, ax = plt.subplots(1, len(rad_profs.vps), figsize=fig_size,
                               sharey=True)
        fig.suptitle(f'Vertical profiles of polarimetric variables'
                     '\n' f'{ptitle}',
                     fontsize=fontsizetitle)
    elif rad_profs.profs_type.lower() == 'qvps':
        rprofs = rad_profs.qvps
        fig, ax = plt.subplots(1, len(rad_profs.qvps), figsize=fig_size,
                               sharey=True)
        fig.suptitle('Quasi-Vertical profiles of polarimetric variables \n'
                     f'{ptitle}',
                     fontsize=fontsizetitle)
    for n, (a, (key, value)) in enumerate(zip(ax.flatten(), rprofs.items())):
        lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, ticks = pltparams(
            key, getattr(rad_profs, prftype).keys(), vars_bounds, ucmap=ucmap,
            unorm=unorm, cb_ext=cb_ext)
        if key == 'rhoHV [-]':
            ticks = ticks
        else:
            ticks = None
        if colours is False:
            a.plot(value, beam_height, 'k')
        elif colours:
            if unorm is not None:
                dnorm.update(unorm)
            cmapp = cmaph.get(key[key.find('['):],
                              mpl.colormaps['tpylsc_rad_pvars'])
            points = np.array([value, beam_height]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmapp,
                                norm=dnorm.get(key[key.find('['):]))
            # Set the values used for colormapping
            lc.set_array(value)
            lc.set_linewidth(2)
            # line = a.add_collection(lc)
            a.add_collection(lc)
            make_colorbar(a, lc, orientation='horizontal')
            if np.isfinite(np.nanmin(value)) and np.isfinite(np.nanmax(value)):
                a.set_xlim([np.nanmin(value), np.nanmax(value)])
        # if stats:
        #     a.fill_betweenx(beam_height,
        #                     value + stats.get(key, value*np.nan),
        #                     value - stats.get(key, value*np.nan),
        #                     alpha=0.4, color='gray', label='std')
        if stats == 'std_dev' or stats == 'sem':
            if rad_profs.profs_type.lower() == 'vps':
                a.fill_betweenx(beam_height,
                                value + rad_profs.vps_stats[stats][key],
                                value - rad_profs.vps_stats[stats][key],
                                alpha=0.4, label=f'{stats}')
            if rad_profs.profs_type.lower() == 'qvps':
                a.fill_betweenx(beam_height,
                                value + rad_profs.qvps_stats[stats][key],
                                value - rad_profs.qvps_stats[stats][key],
                                alpha=0.4, label=f'{stats}')
            # a.fill_betweenx(beam_height,
            #                 value + stats.get(key, value*np.nan),
            #                 value - stats.get(key, value*np.nan),
            #                 alpha=0.4, color='gray', label='std')
        if n == 0:
            a.set_ylabel('Height [km]', fontsize=fontsizelabels, labelpad=15)
        a.tick_params(axis='both', labelsize=fontsizetick)
        a.grid(True)
        if vars_bounds:
            if key in lpv:
                if key == 'rhoHV [-]':
                    a.set_xlim(lpv.get(key)[0], lpv.get(key)[2])
                else:
                    a.set_xlim(lpv.get(key)[:2])
        if mlyr:
            a.axhline(mlyr.ml_top, c='tab:blue', ls='dashed', lw=5,
                      alpha=.5, label='$ML_{top}$')
            a.axhline(mlyr.ml_bottom, c='tab:purple', ls='dashed', lw=5,
                      alpha=.5, label='$ML_{bottom}$')
            a.legend(loc='upper right', fontsize=fontsizetick)
        if key == 'ZH [dBZ]':
            a.set_xlabel('$Z_{H}$ [dBZ]', fontsize=fontsizelabels)
        elif key == 'ZDR [dB]':
            a.set_xlabel('$Z_{DR}$ [dB]', fontsize=fontsizelabels)
        elif key == 'rhoHV [-]':
            a.set_xlabel(r'$ \rho_{HV}$ [-]', fontsize=fontsizelabels)
        elif key == 'PhiDP [deg]':
            a.set_xlabel(r'$ \Phi_{DP}$ [deg]', fontsize=fontsizelabels)
        elif key == 'V [m/s]':
            a.set_xlabel('V [m/s]', fontsize=fontsizelabels)
        elif key == 'gradV [dV/dh]' and rad_profs.profs_type.lower() == 'vps':
            a.set_xlabel('grad V [dV/dh]', fontsize=fontsizelabels)
        elif key == 'KDP [deg/km]':
            a.set_xlabel('$K_{DP}$'+r'$\left [\frac{deg}{km}\right ]$',
                         fontsize=fontsizelabels)
        else:
            a.set_xlabel(key, fontsize=fontsizelabels)
        if ylims:
            a.set_ylim(ylims)
        else:
            a.set_ylim(0, 10)
    plt.show()
    plt.tight_layout()


def plot_rdqvps(rscans_georef, rscans_params, tp_rdqvp, spec_range=None,
                mlyr=None, ylims=None, vars_bounds=None, ucmap=None,
                cb_ext=None, all_desc=False, fig_size=None):
    """
    Display a set of RD-QVPS of polarimetric variables.

    Parameters
    ----------
    rscans_georef : List
        List of georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others, corresponding to each QVP.
    rscans_params : List
        List of radar technical details corresponding to each QVP.
    tp_rdqvp : PolarimetricProfiles Class
        Outputs of the RD-QVPs function.
    spec_range : int, optional
        Range from the radar within which the RD-QVPS were built.
    mlyr : MeltingLayer Class, optional
        Plots the melting layer within the polarimetric profiles.
        The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    vars_bounds : dict containing key and 2-element tuple or list, optional
        Boundaries [min, max] between which radar variables are
        to be plotted.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    all_desc : bool, optional
        If True, plots the initial QVPs used to compute the RD-QPVs.
        The default is True.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    if fig_size is None:
        fig_size = (14, 10)

    fontsizelabels = 20
    fontsizetitle = 25
    fontsizetick = 18
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        None, tp_rdqvp.rd_qvps.keys(), vars_bounds, ucmap=ucmap, cb_ext=cb_ext)
    if vars_bounds:
        lpv.update(vars_bounds)
    cmaph = mpl.colormaps['Spectral'](
        np.linspace(0, 1, len(rscans_params)))
    if ucmap is not None:
        if isinstance(ucmap, str):
            cmaph = mpl.colormaps[ucmap](np.linspace(0, 1, len(rscans_params)))
        else:
            cmaph = ucmap(np.linspace(0, 1, len(rscans_params)))
    # ttxt = f"{rscans_params[0]['datetime']:%Y-%m-%d %H:%M:%S}"
    dt1 = min([i['datetime'] for i in rscans_params])
    dt2 = max([i['datetime'] for i in rscans_params])
    ttxt = (f"{dt1:%Y-%m-%d %H:%M:%S} - {dt2:%H:%M:%S}")

    mosaic = [chr(ord('@')+c+1) for c in range(len(tp_rdqvp.rd_qvps)+1)]
    mosaic = f'{"".join(mosaic)}'

    fig = plt.figure(layout="constrained", figsize=fig_size)
    fig.suptitle('RD-QVPs of polarimetric variables \n' f'{ttxt}',
                 fontsize=fontsizetitle)
    axd = fig.subplot_mosaic(mosaic, sharey=True, height_ratios=[5])

    if all_desc:
        for c, i in enumerate(tp_rdqvp.qvps_itp):
            for n, (a, (key, value)) in enumerate(zip(axd, i.items())):
                axd[a].plot(value, tp_rdqvp.georef['profiles_height [km]'],
                            color=cmaph[c], ls='--',
                            label=(f"{rscans_params[c]['elev_ang [deg]']:.1f}"
                                   + r"$^{\circ}$"))
                # axd[a].legend(loc='upper right')
    if not all_desc:
        i = tp_rdqvp.rd_qvps
    for n, (a, (key, value)) in enumerate(zip(axd, i.items())):
        axd[a].plot(tp_rdqvp.rd_qvps[key],
                    tp_rdqvp.georef['profiles_height [km]'], 'k', lw=3,
                    label='RD-QVP')
        axd[a].legend(loc='upper right')
        if vars_bounds:
            if key in lpv:
                axd[a].set_xlim(lpv.get(key))
            else:
                axd[a].set_xlim([np.nanmin(value), np.nanmax(value)])
        if mlyr:
            axd[a].axhline(mlyr.ml_top, c='tab:blue', ls='dashed', lw=5,
                           alpha=.5, label='$ML_{top}$')
            axd[a].axhline(mlyr.ml_bottom, c='tab:purple', ls='dashed', lw=5,
                           alpha=.5, label='$ML_{bottom}$')
        if ylims:
            axd[a].set_ylim(ylims)
        axd[a].set_xlabel(f'{key}', fontsize=fontsizelabels)
        if n == 0:
            axd[a].set_ylabel('Height [km]', fontsize=fontsizelabels,
                              labelpad=10)
        axd[a].tick_params(axis='both', labelsize=fontsizetick)
        axd[a].grid(True)

    scan_st = axd[mosaic[-1]]
    for c, i in enumerate(rscans_georef):
        scan_st.plot(i['range [m]']/1000, i['beam_height [km]'][0],
                     color=cmaph[c], ls='--',
                     label=(f"{rscans_params[c]['elev_ang [deg]']:.1f}"
                            + r"$^{\circ}$"))
        # scan_st.plot(i['range [m]']/-1000, i['beam_height [km]'][0],
        #               color=cmaph[c], ls='--')
    if spec_range:
        scan_st.axvline(spec_range, c='k', lw=3, label=f'RD={spec_range}')
    scan_st.set_xlabel('Range [km]', fontsize=fontsizelabels)
    scan_st.tick_params(axis='both', labelsize=fontsizetick)
    scan_st.grid(True)
    scan_st.legend(loc='upper right')


def plot_offsetcorrection(rad_georef, rad_params, rad_var, var_m=None,
                          var_offset=0, fig_size=None, var_name='PhiDP [deg]',
                          mode='mean', cmap='tpylsc_div_lbu_w_rd'):
    """
    Plot the offset detection method from ZDR/PhiDP_Calibration Class.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_var : dict
        PPI scan of the radar variable used to detect the offset.
    var_name : str
        Key of the radar variable used to detect the offset.
    cmap : colormap, optional
        User-defined colormap. The default is 'tpylsc_div_lbu_w_rd'.
    """
    if var_m is None:
        if mode == 'mean':
            var_m = np.array([np.nanmean(i) for i in rad_var])
        elif mode == 'median':
            var_m = np.array([np.nanmedian(i) for i in rad_var])
    else:
        var_m = var_m
    if var_name == 'PhiDP [deg]':
        label1 = r'$\Phi_{DP}$'
        labelm = r'$\overline{\Phi_{DP}}$'
        labelo = r'$\Phi_{DP}$ offset'
        dof = 90
        dval = dof // 3
    elif var_name == 'ZDR [dB]':
        label1 = '$Z_{DR}$'
        labelm = r'$\overline{Z_{DR}}$'
        labelo = r'$Z_{DR}$ offset'
        dval = 0.1
        dof = 1
    if fig_size is None:
        fig_size = (8, 8)

    fig, ax = plt.subplots(figsize=fig_size,
                           subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)
    if isinstance(rad_params['elev_ang [deg]'], str):
        dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
    else:
        dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{2}} deg. -- "
    if rad_params['datetime']:
        dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    else:
        dtdes2 = ''
    ptitle = dtdes1 + dtdes2
    ax.set_title(ptitle, fontsize=16)
    ax.grid(color='gray', linestyle=':')
    ax.set_theta_zero_location('N', offset=0)
    # =========================================================================
    # Plot the radar variable values at each azimuth
    # =========================================================================
    ax.scatter((np.ones_like(rad_var.T) * [rad_georef['azim [rad]']]).T,
               rad_var, s=5, c=rad_var, cmap=cmap, label=label1,
               norm=mpc.SymLogNorm(
                   linthresh=.01, linscale=.01, base=2,
                   vmin=(var_m.mean()-dval if mode == 'mean'
                         else np.median(var_m)-dval if mode == 'median'
                         else var_m.min()),
                   vmax=(var_m.mean()+dval if mode == 'mean'
                         else np.median(var_m)+dval if mode == 'median'
                         else var_m.max())))
    # =========================================================================
    # Plot the radar variable mean/median value of each azimuth
    # =========================================================================
    ax.plot(rad_georef['azim [rad]'], var_m, c='grey', linewidth=2,
            ls='', marker='s', markeredgecolor='k', alpha=0.4, label=labelm)
    # =========================================================================
    # Plot the radar variable offset
    # =========================================================================
    if var_offset != 0:
        ax.plot(rad_georef['azim [rad]'],
                np.full(rad_georef['azim [rad]'].shape, var_offset),
                c='k', linewidth=2.5,
                label=f'{labelo} \n [{var_offset:0.2f}]')

    ax.set_thetagrids(np.arange(0, 360, 90))
    ax.xaxis.grid(ls='-')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_rlabel_position(-45)
    # if var_name == 'PhiDP [deg]':
    #     ax.set_ylim([var_m.mean()-dof, var_m.mean()+dof])
    #     ax.set_yticks(np.arange(round(var_m.mean()/dval)*dval-dof,
    #                             round(var_m.mean()/dval)*dval+dof+1,
    #                             dval))
    # else:
    #     ax.set_ylim([var_m.mean()-dof, var_m.mean()+dof])
    #     # ax.set_yticks(np.arange(round(var_m.mean()/dval)*dval-dof,
    #     #                         round(var_m.mean()/dval)*dval+dof+.1,
    #     #                         dval))
    angle = np.deg2rad(67.5)
    ax.legend(fontsize=15, loc="lower left",
              bbox_to_anchor=(.58 + np.cos(angle)/2, .4 + np.sin(angle)/2))
    ax.axes.set_aspect('equal')
    plt.tight_layout()


def plot_mfs(path_mfs, norm=True, vars_bounds=None, fig_size=None):
    """
    Plot the membership functions used in clutter classification.

    Parameters
    ----------
    path_mfs : str
        Location of the membership function files..
    norm : bool, optional
        Determines if the variables are normalised for a more comprehensive
        visualisation of the MFS. The default is True.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, LaTeX Varnames] between which radar variables are
        to be mapped.
    fig_size : list or tuple containing 2-element numbers, optional
        Width, height in inches. The default is None.
    """
    import os
    mfspk = {
        'ZHH': [[-10, 60], '$Z_H$ [dBZ]'],
        'sZhh': [[0, 20], r'$\sigma(Z_{H})$ [dBZ]'],
        'ZDR': [[-6, 6], '$Z_{DR}$ [dB]'],
        'sZdr': [[0, 5], r'$\sigma(Z_{DR}$) [dB]'],
        'Rhv': [[0, 1], r'$\rho_{HV}$ [-]'],
        'sRhv': [[0, .4], r'$\sigma(\rho_{HV})$ [-]'],
        'Pdp': [[0, 180], r'$\Phi_{DP}$ [deg]'],
        'sPdp': [[0, 180], r'$\sigma(\Phi_{DP})$ [deg]'],
        'Vel': [[-3, 3], 'V [m/s]'],
        'sVel': [[0, 5], r'$\sigma(V)$ [m/s]'],
        'LDR': [[-40, 10], 'LDR [dB]'],
        }
    if vars_bounds is not None:
        mfspk.update(vars_bounds)

    mfsp = {f[f.find('mf_')+3: f.find('_preci')]: np.loadtxt(f'{path_mfs}{f}')
            for f in sorted(os.listdir(path_mfs))
            if f.endswith('_precipi.dat')}
    mfsp = {k: v for k, v in sorted(mfsp.items()) if k in mfspk}
    mfsc = {f[f.find('mf_')+3: f.find('_clu')]: np.loadtxt(f'{path_mfs}{f}')
            for f in sorted(os.listdir(path_mfs))
            if f.endswith('_clutter.dat')}
    mfsc = {k: v for k, v in sorted(mfsc.items()) if k in mfspk}

    varsp = {k for k in mfsp.keys()}
    varsc = {k for k in mfsc.keys()}

    if len(varsp) % 2 == 0:
        ncols = int(len(varsp) / 2)
        nrows = len(varsp) // ncols
        if fig_size is None:
            fig_size = (18, 5)
    else:
        ncols = 3
        if len(varsp) % 3 == 0:
            nrows = (len(varsp) // ncols)
        else:
            nrows = (len(varsp) // ncols)+1
        if fig_size is None:
            fig_size = (18, 7.5)

    if varsp != varsc:
        raise TowerpyError('Oops!... The number of membership functions for'
                           + 'clutter and precipitation do not correspond.'
                           + 'Please check before continue.')

    if norm is True:
        mfs_prnorm = {k: np.array([val[:, 0], rut.normalisenan(val[:, 1])]).T
                      for k, val in mfsp.items()}
        mfs_clnorm = {k: np.array([val[:, 0], rut.normalisenan(val[:, 1])]).T
                      for k, val in mfsc.items()}

    f, ax = plt.subplots(nrows, ncols, sharey=True, figsize=fig_size)
    for a, (key, value) in zip(ax.flatten(), mfs_prnorm.items()):
        a.plot(value[:, 0], value[:, 1], c='tab:blue', label='PR')
        a.plot(mfs_clnorm[key][:, 0], mfs_clnorm[key][:, 1], label='CL',
               ls='dashed', c='tab:orange')
        # a.set_xlim(left=0)
        a.set_xlim(mfspk[key][0])
        a.tick_params(axis='both', labelsize=16)

        divider = make_axes_locatable(a)
        cax = divider.append_axes("top", size="15%", pad=0)
        cax.get_xaxis().set_visible(False)
        cax.get_yaxis().set_visible(False)
        cax.set_facecolor('slategrey')

        at = AnchoredText(mfspk[key][1], loc='center',
                          prop=dict(size=18, color='white'), frameon=False)
        cax.add_artist(at)
        a.legend(fontsize=14)
    f.tight_layout()


def plot_zhah(rad_vars, r_ahzh, temp, coeff_a, coeff_b, coeffs_a, coeffs_b,
              temps, zh_lower_lim, zh_upper_lim, var2calc='ZH [dBZ]'):
    r"""
    Display the AH-ZH relation.

    Parameters
    ----------
    rad_vars : dict
        Dict containing radar variables to plot.
    r_ahzh : obj
        Results of the Attn_Refl_Relation class.
    temp: float
        Temperature, in :math:`^{\circ}C`, used to derive the coefficients
        according to [1]_. The default is 10.
    coeff_a, coeff_b: float
        Computed coefficients of the :math:`A_H(Z_H)` relationship.
    coeffs_a, coeffs_b: list or array
        Default coefficients of the :math:`A_H(Z_H)` relationship..
    temps : list or array
        Default values for the temperature.
    var2calc : str, optional
        Radar variable to be computed. The string has to be one of
        'AH [dB/km]' or 'ZH [dBZ]'. The default is 'ZH [dBZ]'.
    """
    tcksize = 14
    cmap = 'Spectral_r'
    n1 = mpc.LogNorm(vmin=1, vmax=1000)
    gridsize = 200
    ahzhii = np.arange(zh_lower_lim, zh_upper_lim, 0.05)
    ahzhlii = tpuc.xdb2x(ahzhii)
    ahzhi = coeff_a * ahzhlii ** coeff_b
    if var2calc == 'AH [dB/km]' and 'ZH [dBZ]' in rad_vars.keys():
        zh_all = rad_vars['ZH [dBZ]'].ravel()
        ah_all = rad_vars['AH [dB/km]'].ravel()
    elif 'AH [dB/km]' in r_ahzh.keys():
        zh_all = rad_vars['ZH [dBZ]'].ravel()
        ah_all = r_ahzh['AH [dB/km]'].ravel()
    if var2calc == 'ZH [dBZ]' and 'ZH [dBZ]' in rad_vars.keys():
        zh_all = rad_vars['ZH [dBZ]'].ravel()
        ah_all = rad_vars['AH [dB/km]'].ravel()
    # Plot the AH-ZH values
    fig, ax = plt.subplots()
    ax.plot(ahzhii, ahzhi, c='k', ls='--',
            label=f'$A_H={coeff_a:,.2e}Z_H^{{{coeff_b:,.2f}}}$')
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("top", size="7%", pad="2%")
    hxb = ax.hexbin(np.ma.masked_invalid(zh_all),
                    np.ma.masked_invalid(ah_all), gridsize=gridsize,
                    mincnt=1, cmap=cmap, norm=n1)
    cb = fig.colorbar(hxb, cax=cax, extend='max',
                      orientation='horizontal')
    ax.set_xlim([-10, 60])
    ax.set_ylim([0, 2])
    cb.ax.tick_params(direction='in', labelsize=tcksize,)
    cb.ax.set_title('Counts', fontsize=tcksize)
    cax.xaxis.set_ticks_position("top")
    ax.set_xlabel('$Z_H$ [dBZ]')
    ax.set_ylabel('$A_H$ [dB/km]')
    ax.legend(loc='upper left')
    ax.grid()
    # Plot the linar interpolation of temp.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle(rf'Linear Interpolation at T = {temp}$\degree$')
    ax = axs[0]
    ax.plot(coeffs_a, temps, '-ob')
    ax.plot(coeff_a, temp, 'ro')
    ax.set_xlabel('coeff a')
    ax.set_ylabel(r'Temp ${\degree}$C')
    ax = axs[1]
    ax.plot(coeffs_b, temps, '-ob')
    ax.plot(coeff_b, temp, 'ro')
    ax.set_xlabel('coeff b')
    fig.tight_layout()


def plot_ppidiff(rad_georef, rad_params, rad_var1, rad_var2, var2plot1=None,
                 var2plot2=None, diff_lims=[-10, 10, 1], mlyr=None, xlims=None,
                 ylims=None, vars_bounds=None, unorm=None, ucmap=None,
                 ucmap_diff=None, cb_ext=None, fig_title=None, fig_size=None):
    """
    Plot the difference between a radar variable from different dicts.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_var1 : dict
        Dict containing radar variables to plot.
    rad_var2 : dict
        Dict containing radar variables to plot.
    vars2plot : str, optional
        Keys of the radar variables to plot. Variables must have the same
        units. The default is None. This option will plot ZH or look for the
        'first' element in the rad_vars dict.
    diff_lims : 3-element tuple or list, optional
        Boundaries [min, max, step] used for mapping the difference plot.
        The default is [-10, 10, 1].
    mlyr : MeltingLayer Class, optional
        Plot the melting layer height. ml_top (float, int, list or np.array)
        and ml_bottom (float, int, list or np.array) must be explicitly
        defined. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max, nvals] between which radar variables are
        to be mapped.
    unorm : dict containing matplotlib.colors normalisation objects, optional
        User-defined normalisation methods to map colormaps onto radar data.
        The default is None.
    ucmap : colormap, optional
        User-defined colormap, either a matplotlib.colors.ListedColormap,
        or string from matplotlib.colormaps.
    ucmap_diff : str of colormap, optional
        User-defined colormap used in the difference plot.
    cb_ext : dict containing key and str, optional
        The str modifies the end(s) for out-of-range values for a
        given key (radar variable). The str has to be one of 'neither',
        'both', 'min' or 'max'.
    fig_title : str, optional
        String to show in the plot title.
    fig_size : 2-element tuple or list, optional
        Modify the default plot size.
    """
    lpv, bnd, cmaph, cmapext, dnorm, v2p, normp, cbtks_fmt, tcks = pltparams(
        var2plot1, rad_var1, vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    if var2plot1 is None:
        var2plot1 = v2p
    lpv2, bnd2, cmaph2, cmapext2, dnorm2, v2p2, normp2, cbtks_fmt2, tcks2 = pltparams(
        var2plot2, rad_var2, vars_bounds, ucmap=ucmap, unorm=unorm,
        cb_ext=cb_ext)
    if var2plot2 is None:
        var2plot2 = v2p2
    cmapp = cmaph.get(var2plot1[var2plot1.find('['):],
                      mpl.colormaps['tpylsc_rad_pvars'])
    if fig_title is None:
        if isinstance(rad_params['elev_ang [deg]'], str):
            dtdes1 = f"{rad_params['elev_ang [deg]']} -- "
        else:
            dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{2}} deg. -- "
        if rad_params['datetime']:
            dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
        else:
            dtdes2 = ''
        ptitle = dtdes1 + dtdes2
    else:
        ptitle = fig_title
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
    # =========================================================================
    # Creates plots to visualise difference
    # =========================================================================
    mosaic = 'ABC'
    if fig_size is None:
        fig_size = (16, 5)
    fig_mos1 = plt.figure(figsize=fig_size, constrained_layout=True)
    ax_idx = fig_mos1.subplot_mosaic(mosaic, sharex=True, sharey=True)
    for key, value in rad_var1.items():
        if key == var2plot1:
            fzhna = ax_idx['A'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmapp,
                                           norm=normp)
            ax_idx['A'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['A'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['A'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['A'].legend(loc='upper left')
    if xlims is not None:
        ax_idx['A'].set_xlim(xlims)
    if ylims is not None:
        ax_idx['A'].set_ylim(ylims)
    # plt.colorbar(fzhna, ax=ax_idx['A']).ax.tick_params(labelsize=10)
    ax_idx['A'].grid(True)
    ax_idx['A'].axes.set_aspect('equal')
    ax_idx['A'].tick_params(axis='both', labelsize=10)
    for key, value in rad_var2.items():
        if key == var2plot2:
            fzhna = ax_idx['B'].pcolormesh(rad_georef['grid_rectx'],
                                           rad_georef['grid_recty'], value,
                                           shading='auto', cmap=cmapp,
                                           norm=normp)
            ax_idx['B'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['B'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['B'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['B'].legend(loc='upper left')
    if xlims is not None:
        ax_idx['B'].set_xlim(xlims)
    if ylims is not None:
        ax_idx['B'].set_ylim(ylims)
    # plt.colorbar(fzhna, ax=ax_idx['B']).ax.tick_params(labelsize=10)
    if (var2plot1 == 'rhoHV [-]' or '[mm]' in var2plot1
       or '[mm/h]' in var2plot1):
        plt.colorbar(fzhna, ax=ax_idx['B'], ticks=tcks,
                     format=f'%.{cbtks_fmt}f')
    else:
        plt.colorbar(fzhna, ax=ax_idx['B'])
    ax_idx['B'].grid(True)
    ax_idx['B'].axes.set_aspect('equal')
    ax_idx['B'].tick_params(axis='both', labelsize=10)

    cmaph = 'tpylsc_div_dbu_rd'
    if ucmap_diff is not None:
        cmaph = ucmap_diff
    divnorm = mpl.colors.BoundaryNorm(
        rut.linspace_step(diff_lims[0], diff_lims[1], diff_lims[2]),
        mpl.colormaps[cmaph].N, extend='both')
    fzhna = ax_idx['C'].pcolormesh(rad_georef['grid_rectx'],
                                   rad_georef['grid_recty'],
                                   rad_var1[var2plot1]-rad_var2[var2plot2],
                                   shading='auto', cmap=cmaph, norm=divnorm)
    ax_idx['C'].set_title(f"{ptitle}" "\n"
                          + f"diff {var2plot1[var2plot1.find('['):]}")
    if mlyr is not None:
        ax_idx['C'].plot(mlt_idxx, mlt_idxy, c='k', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(T)}$')
        ax_idx['C'].plot(mlb_idxx, mlb_idxy, c='grey', ls='-', alpha=3/4,
                         path_effects=[pe.Stroke(linewidth=5, foreground='w'),
                                       pe.Normal()], label=r'$MLyr_{(B)}$')
        ax_idx['C'].legend(loc='upper left')
    plt.colorbar(fzhna, ax=ax_idx['C']).ax.tick_params(labelsize=10)
    ax_idx['C'].grid(True)
    ax_idx['C'].axes.set_aspect('equal')
    ax_idx['C'].tick_params(axis='both', labelsize=10)


# =============================================================================
# %% xarray implementation
# =============================================================================

# =============================================================================
# %%% Plotting parameters
# =============================================================================
from dataclasses import dataclass

@dataclass
class PlotParams:
    range_spec: list
    norm_boundaries: list
    cmap: object
    extend: str
    norm: object
    ticklabels: list | None
    force_all_ticks: bool = False

@dataclass
class CmapSpec:
    name: str
    extend: str = "both"
    set_under: str | None = None
    set_over: str | None = None


def _lookup_params_override(dct, varname, units):
    """Return override from dict using varname first, then units.
    Override keys are normalised so users can pass raw unit strings."""
    if not dct:
        return None

    # Normalise all keys in the override dict
    normed = {}
    for k, v in dct.items():
        # If the key is a variable name, keep it as-is
        if k in (varname,):
            normed[k] = v
        else:
            # Otherwise treat it as a unit string and normalise it
            normed[_normalise_units(k)] = v

    # Priority 1: variable name
    if varname in normed:
        return normed[varname]

    # Priority 2: canonical units
    if units in normed:
        return normed[units]

    return None


def _nice_continuous_ticks(boundaries, max_ticks=10):
    """
    Return a reduced, nicely spaced set of ticks for continuous colorbars
    using MaxNLocator, clipped to the actual boundaries.
    """
    bmin, bmax = float(boundaries[0]), float(boundaries[-1])
    locator = MaxNLocator(nbins=max_ticks)
    ticks = locator.tick_values(bmin, bmax)
    # Clip to [bmin, bmax] to avoid "empty" extensions
    ticks = ticks[(ticks >= bmin) & (ticks <= bmax)]
    # Ensure at least two ticks
    if ticks.size < 2:
        ticks = np.linspace(bmin, bmax, min(max_ticks, len(boundaries)))
    return ticks


def _add_colorbar(fig, ax, mappable, pltprms, fsizes, vunits, rotangle=0,
                  coord_sys="polar", cartopy_enabled=False, pos='top', pad="2%",
                  size="7%", label=None, labelpad=40):
    """Add a colorbar for plots."""
    bounds = pltprms.norm_boundaries
    # Skip colorbars with no data
    if np.allclose(bounds, bounds[0]):
        print("Colorbar skipped: collapsed boundaries")
        return None
    
    # =============================================================================
    # POLAR or CARTOPY COLORBAR
    # =============================================================================
    if coord_sys == "polar" or cartopy_enabled:
        rotangle = 90
        shrink = 0.65 if coord_sys == "polar" else 1.0
        cb = fig.colorbar(mappable, ax=ax, extend=pltprms.extend,
                          shrink=shrink, aspect=8)
        cb.ax.tick_params(direction="in", axis="both",
                          labelsize=fsizes["fsz_cb"])
        # Categorical case
        if pltprms.ticklabels is not None:
            cb.set_ticks(bounds)
            cb.set_ticklabels(pltprms.ticklabels)
            plt.setp(cb.ax.get_yticklabels(),
                     rotation=rotangle, ha="left", va="center")
        # RHOHV
        elif pltprms.force_all_ticks:
            cb.set_ticks(bounds)
        # Continuous case
        else:
            # If too many boundaries, reduce ticks manually
            # if len(bounds) > 20 or len(bounds) > mappable.cmap.N:
            threshold = max(20, mappable.cmap.N)
            if len(bounds) > threshold:
                nice_ticks = _nice_continuous_ticks(bounds)
                cb.set_ticks(nice_ticks)
            else:
                # Let Matplotlib choose ticks automatically
                pass
        # Optional label
        if label:
            cb.ax.set_ylabel(label, fontsize=fsizes["fsz_cb"], labelpad=labelpad)
            cb.ax.yaxis.set_label_position("right")
        return cb
    # =============================================================================
    # RECTANGULAR COLORBAR (TOP AXIS)
    # =============================================================================
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes(pos, size=size, pad=pad, axes_class=plt.Axes)
    sm = mpl.cm.ScalarMappable(cmap=pltprms.cmap, norm=pltprms.norm)
    # cb = fig.colorbar(mappable, cax=cax, orientation="horizontal",
    #                   extend=pltprms.extend)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal",
                      extend=pltprms.extend )
    cax.tick_params(direction="out", labelsize=fsizes["fsz_cb"],
                    rotation=rotangle)
    cax.xaxis.set_ticks_position("top")
    # Categorical case
    if pltprms.ticklabels is not None:
        cb.set_ticks(bounds)
        cb.set_ticklabels(pltprms.ticklabels)
    # RHOHV special case
    elif pltprms.force_all_ticks:
        cb.set_ticks(bounds)
    # Continuous case
    else:
        # If too many boundaries, reduce ticks manually
        threshold = max(20, mappable.cmap.N)
        if len(bounds) > threshold:
            nice_ticks = _nice_continuous_ticks(bounds)
            cb.set_ticks(nice_ticks)
        else:
            # Let Matplotlib choose ticks automatically
            pass
    if label:
        cb.ax.set_ylabel(label, fontsize=fsizes["fsz_cb"], labelpad=labelpad)
        cb.ax.yaxis.set_label_position("left")
    cb.ax.tick_params(direction="out")
    return cax


def _discrete_cmap_with_labels(values, cmap_name="tpylc_div_yw_gy_bu",
                              labels=None):
    """Generate discrete colormaps and normalisation for categorical data."""
    unique_vals = np.unique(values)

    # Bin edges halfway between successive values
    edges = np.concatenate([[unique_vals[0] - 0.5],
                            (unique_vals[:-1] + unique_vals[1:]) / 2,
                            [unique_vals[-1] + 0.5]])

    # Build a ListedColormap with exactly N colours
    base_cmap = plt.get_cmap(cmap_name)
    if hasattr(base_cmap, "colors"):
        colors = base_cmap.colors[:len(unique_vals)]
    else:
        colors = base_cmap(np.linspace(0, 1, len(unique_vals)))
    cmap = mpc.ListedColormap(colors)

    # BoundaryNorm ensures discrete mapping
    norm = mpc.BoundaryNorm(edges, cmap.N)

    # Labels
    ticklabels = [labels.get(v, str(v)) for v in unique_vals] if labels else [str(v) for v in unique_vals]

    return cmap, norm, unique_vals.tolist(), ticklabels



def _resolve_cmap(units, varname, ucmap, cb_ext=None):
    """
    Return a colormap and extension mode resolved from units, overrides,
    and defaults."""
    CMAP_DEFAULTS = {
        "dBZ":      CmapSpec("tpylsc_rad_ref", extend="both"),
        "unitless": CmapSpec("tpylsc_rad_pvars", extend="both"),
        "dB":       CmapSpec("tpylsc_rad_2slope", extend="both"),
        "deg/km":   CmapSpec("tpylsc_rad_2slope", extend="both"),
        "dB/km":    CmapSpec("tpylsc_rad_pvars", extend="max"),
        "m/s":      CmapSpec("tpylsc_div_dbu_rd", extend="both"),
        "mm/h":     CmapSpec("tpylsc_rad_rainrt", extend="max",
                             set_under="whitesmoke"),
        "mm":       CmapSpec("tpylsc_rad_rainrt", extend="max",
                             set_under="whitesmoke"),
        "km":       CmapSpec("gist_earth", extend="max"),
        "dV/dh":    CmapSpec("tpylsc_rad_2slope_r", extend="both"),
        "dBm":      CmapSpec("tpylsc_rad_pvars", extend="both"),
        "unorm":    CmapSpec("tpylsc_useq_bupkyw", extend="neither"),
        "0-1":      CmapSpec("tpylsc_useq_bupkyw", extend="neither"),
        }

    # Override by varname
    # if ucmap and varname in ucmap:
    #     spec = CmapSpec(ucmap[varname])
    cmap_name = _lookup_params_override(ucmap, varname, units)
    if cmap_name is not None:
        spec = CmapSpec(cmap_name)
    elif varname.lower().startswith('pia'):
        spec = CmapSpec("tpylsc_useq_fiery", extend="max")
    # elif varname.lower() == "alpha":
    elif varname.lower().startswith('alpha'):
        spec = CmapSpec("tpylsc_useq_fiery", extend="max")
    # elif varname.lower() == "beta":
    elif varname.lower().startswith('beta'):
        spec = CmapSpec("tpylsc_useq_fiery", extend="max")
    # here im not sure because is DWD special case with no units
    elif varname.lower().startswith('uvrad'):
        spec = CmapSpec("tpylsc_div_dbu_rd", extend="both")
    # Override by units
    elif ucmap and units in ucmap:
        spec = CmapSpec(ucmap[units])
    else:
        spec = CMAP_DEFAULTS.get(units, CmapSpec("tpylsc_rad_pvars"))

    # Apply cb_ext override (if provided)
    # if cb_ext and units in cb_ext:
    #     spec.extend = cb_ext[units]
    ext_override = _lookup_params_override(cb_ext, varname, units)
    if ext_override is not None:
        spec.extend = ext_override
    # Instantiate colormap safely
    cmap = mpl.colormaps[spec.name].copy()

    if spec.set_under:
        cmap.set_under(spec.set_under)
    if spec.set_over:
        cmap.set_over(spec.set_over)

    return cmap, spec.extend


def _resolve_var2plot(ds, var2plot):
    """Select a variable to plot, preferring reflectivity if available."""
    if var2plot is not None:
        return var2plot
    # Auto-select reflectivity if present
    for name, da in ds.data_vars.items():
        if da.attrs.get("units") == "dBZ":
            return name
    # Fallback: first variable
    return list(ds.data_vars)[0]


def plot_params(varname, xrds, vars_bounds=None, unorm=None, cb_ext=None,
                custom_rules=None, ucmap=None):
    """
    Generate plotting parameters for a given radar variable.

    Parameters
    ----------
    varname : str
        Variable name to plot (must be present in xrds.data_vars).
    xrds : xarray.Dataset
        Dataset containing radar variables and coordinates.
    vars_bounds : dict, optional
        Explicit bounds overrides keyed by variable name.
        Format: {"DBZH": [-20, 70, 19]}.
    unorm : dict, optional
        User normalisation overrides keyed by variable name or unit.
        Values should be matplotlib.colors.Normalize or BoundaryNorm objects.
    cb_ext : dict, optional
        Colourbar extension overrides keyed by unit.
    custom_rules : dict, optional
        Metadata-driven rules keyed by 'units', 'standard_name', or 'short_name'.
        Values can be lists (bounds), arrays (boundaries), or callables
        (functions returning boundaries).
    ucmap : dict, optional
        Colormap overrides keyed by unit or variable name.
        Example: {"dBZ": "viridis", "PHIDP": "twilight"}.

    Returns
    -------
    range_spec : list
        [vmin, vmax, N] used to generate boundaries (unless overridden).
    norm_boundaries : ndarray
        Array of boundaries for colour normalisation.
    cmap : dict
        Colormap dictionary keyed by unit.
    extend : dict
        Colourbar extension dictionary keyed by unit.
    normp : matplotlib.colors.Normalize
        Normalisation object for plotting.
    varname : str
        Variable name (echoed back).

    Override Hierarchy
    ------------------
    1. vars_bounds[varname] → explicit bounds override
    2. unorm[varname] or unorm[units] → explicit normalisation override
    3. ucmap[varname] or ucmap[units] → explicit colormap override
    4. custom_rules (units → standard_name → short_name)
    5. Built-in defaults (lpv_units, lpv_standard, lpv_short)
    6. Fallback: auto-scale from data min/max

    Notes
    -----
    - This function normalises obvious unit inconsistencies (e.g. "meters per seconds" → "m/s", "mm h-1" → "mm/h") and applies a hierarchy of rules to determine bounds, colormaps, and normalisation for plotting radar variables.
    - Unit strings are normalised internally (e.g. "meters per seconds" → "m/s").
    """
    da = xrds[varname]
    units_raw = da.attrs.get("units", "")
    stdname = da.attrs.get("standard_name", "")
    short = da.attrs.get("short_name", varname)

    # Normalise some unit inconsistencies 
    units = _normalise_units(units_raw)

    # Defaults grouped inside function 
    lpv_units = {"dBZ": [-10, 60, 15],
                 "dB": [-2, 6, 17],
                 "deg": [0, 180, 19],
                 "deg/km": [-2, 6, 17],
                 "unitless": [0.3, 0.9, 1.0],
                 "dB/km": [0, 0.5, 11],
                 "m/s": [-5, 5, 11],
                 "dV/dh": [-1.8, 0.6, 13],
                 "mm/h": [0, 64, 14],
                 "mm": [0, 200, 14],
                 "km": [0, 7, 36],
                 "0-1": [0, 1, 11],
                 "dBm": [-120, 0, 25],
                 "unorm": [0, 1, 11]
                 }
    # Remove unitless unless rhohv
    if "rhohv" not in short.lower():
        lpv_units.pop("unitless", None)
    # Remove dB default unless ZDR
    if "zdr" not in short.lower():
        lpv_units.pop("dB", None)

    lpv_standard = {
        "radar_linear_depolarization_ratio": [-30, 10, 17],
        # "radar_specific_differential_phase_hv": [-2, 6, 17],
        # "radar_differential_phase_hv": [0, 180, 19],
        "radar_doppler_spectrum_width_h": [0, 5, 11],
        "radar_doppler_spectrum_width_v": [0, 5, 11],
        "signal_noise_ratio": [-30, 130, 17],
        "linear_signal_noise_ratio": [0, 1e6, 20],
        "signal_noise_ratio_h": [0, 30, 17],
        "signal_noise_ratio_v": [0, 30, 17],
        # "signal_quality_index_h": [0, 1, 11],
        # "signal_quality_index_v": [0, 1, 11],
        # "clutter_indicator": [0, 1, 11],
        "clutter_map": [0, 1, 11],
        "radar_linear_equivalent_reflectivity_factor_h": [0, 1e6, 20],
        "radar_linear_equivalent_reflectivity_factor_v": [0, 1e6, 20]
        }

    lpv_short = {"PIA": [0, 20, 21], "alpha": [0, 0.2, 21],
                 "beta": [0, 0.1, 21]
                 }
    
    # Continuous variables
    if "flags" not in units.lower():
        bounds = _lookup_params_override(vars_bounds, varname, units)
        if bounds is not None:
            pass  # use this override
        elif custom_rules and "units" in custom_rules and units in custom_rules["units"]:
            bounds = custom_rules["units"][units]
        elif stdname in lpv_standard:
            bounds = lpv_standard[stdname]
        elif short in lpv_short:
            bounds = lpv_short[short]
        elif units in lpv_units:
            bounds = lpv_units[units]
        else:
            bounds = [float(da.min()), float(da.max()), 13]
        # Boundaries
        # Custom non-uniform rainfall boundaries
        # if units == "mm/h" and not (vars_bounds and varname in vars_bounds):
        if units == "mm/h" and not (
                vars_bounds and (varname in vars_bounds
                                 or _normalise_units("mm/h") in vars_bounds)):
            n = 13
            arr = np.geomspace(1, 64, num=n)  # exact 1 → 64 range
            arr = np.concatenate(([0.1, 0.5], arr))  # drizzle resolution
            arr = np.array((0.1, 1, 2, 4, 8, 12, 16, 20, 24, 30, 36, 48, 56,
                            64))
            custom_bnd = arr
            bounds = [arr.min(), arr.max(), len(arr)]
        elif units == "mm" and not (vars_bounds and varname in vars_bounds):
            arr = np.array([0.1, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
                            96, 128, 256])
            arr = np.array((0.1, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75,
                            100, 125, 150, 200))
            bounds = [arr.min(), arr.max(), len(arr)]
            custom_bnd = arr
        else:
            custom_bnd = None

        # Colormap resolution (with overrides) 
        cmap, ext = _resolve_cmap(units, varname, ucmap, cb_ext)
        if custom_bnd is not None:
            bnd = custom_bnd
        elif "rhohv" in short.lower():
            bnd = np.hstack((np.linspace(bounds[0], bounds[1], 4)[:-1],
                             np.linspace(bounds[1], bounds[2], 11)))
            ext = "min"
        else:
            bnd = np.linspace(bounds[0], bounds[1], bounds[2])
        # Normalisation hierarchy 
        # if unorm and varname in unorm:
        #     normp = unorm[varname]
        # elif unorm and units in unorm:
        #     normp = unorm[units]
        norm_override = _lookup_params_override(unorm, varname, units)
        if norm_override is not None:
            normp = norm_override
        else:
            normp = mpc.BoundaryNorm(bnd, cmap.N, extend=ext)
        ticklabels = None
    else:
        # Handle categorical flags 
        flags = da.attrs.get("flags", {})
        values = list(flags.values())
        labels = {v: k for k, v in flags.items()}  # invert mapping
        # Decide colormap source
        if ucmap and (varname in ucmap or units in ucmap):
            cmap_name = ucmap.get(varname, ucmap.get(units))
        else:
            if "mlpclass" in short.lower():
                cmap_name = "Paired"
            else:
                cmap_name = "tpylc_div_yw_gy_bu"  # default fallback
        cmap, normp, ticks, ticklabels = _discrete_cmap_with_labels(
            values, labels=labels, cmap_name=cmap_name)
        bounds = [min(values), max(values), len(values)]
        bnd = ticks
        ext = "neither"  # flags don’t need extensions
    force_all_ticks = False
    if "rhohv" in short.lower() or "rate" in short.lower():
        force_all_ticks = True

    return PlotParams(range_spec=bounds, norm_boundaries=bnd, cmap=cmap,
                      extend=ext, norm=normp, ticklabels=ticklabels,
                      force_all_ticks=force_all_ticks)


def default_cartopy_config():
    """
    Return the default configuration dictionary for Cartopy-based plotting.

    This dictionary defines all tunable Cartopy options used by
    ``plot_ppi_xr``. Users may override any subset of keys by passing a
    ``cartopy_cfg`` dictionary to ``plot_ppi_xr``; missing keys fall back to
    the defaults defined here.

    Returns
    -------
    cfg : dict
        Dictionary with the following structure:

        enable_cartopy : bool, default False
            Whether to activate Cartopy plotting. If True, ``coord_sys`` is
            forced to ``"rect"`` and ``polarplot`` is ignored.

        projection : cartopy.crs.CRS, default PlateCarree()
            Map projection used for the Matplotlib axes.

        data_crs : cartopy.crs.CRS or None, default None
            CRS of the input data coordinates. Required when
            ``enable_cartopy=True``. For lon/lat grids, use
            ``ccrs.PlateCarree()``; for projected grids, use the appropriate
            CRS (e.g. ``ccrs.UTM(32)``).

        extent : tuple of float or None, default None
            Optional map extent ``(xmin, xmax, ymin, ymax)`` in the projection
            CRS. If None, the extent is derived from the data.

        tiles : dict
            Configuration for background map tiles:

                enabled : bool, default False
                    Whether to draw background tiles.

                class : cartopy.io.img_tiles.* or None, default None
                    Tile provider class (e.g. ``OSM()``, ``StamenTerrain()``).

                kwargs : dict, default {}
                    Additional keyword arguments passed to the tile provider.

                resolution : int, default 8
                    Tile resolution level.

                alpha : float, default 1.0
                    Tile transparency.

        features : list
            List of Cartopy features to add (e.g. coastlines, borders, rivers).
            Defaults to ``default_cartopy_features``.

        gridlines : dict
            Configuration for gridlines:

                enabled : bool, default True
                    Whether to draw gridlines.

                draw_labels : bool, default True
                    Whether to label gridlines.

                label_size : int, default 10
                    Font size for gridline labels.

        alpha_rad : float, default 1.0
            Transparency applied to the radar field.
    """

    return {"enable_cartopy": False,
            "projection": ccrs.PlateCarree(),
            "data_crs": None,
            "extent": None,
            "tiles": {"enabled": False, "class": None, "kwargs": {},
                      "resolution": 8, "alpha": 1.0},
            "features": default_cartopy_features,
            "gridlines": {"enabled": True, "draw_labels": True,
                          "label_size": 10},
            "tick_spacing": {"dx": 1.0, "dy": 1.0},
            "alpha_rad": 1.0}

# =============================================================================
# %%% PPI plots
# =============================================================================

def _resolve_mlyr(ds, mlyr_bnames):
    """
    Resolve melting-layer top/bottom DataArrays from either:
    - dataset defaults (MLYRTOP, MLYRBTM), or
    - user-provided variable names via mlyr_bnames={'top': ..., 'bottom': ...}.
    """
    if mlyr_bnames is None:
        top_name = "MLYRTOP"
        bot_name = "MLYRBTM"
    else:
        top_name = mlyr_bnames.get("top")
        bot_name = mlyr_bnames.get("bottom")

    if top_name not in ds or bot_name not in ds:
        return None, None

    return ds[top_name], ds[bot_name]


def _addML2plot(ax, coord_sys, ds, mlyr_bnames=None, coord_names=None,
                polarplot=True, cartopy_enabled=False, data_crs=None,
                projcoord_names=None):
    """
    Overlay melting-layer boundaries (top and bottom) on a PPI axis.
    """
    # Resolve ML boundaries
    mlyr_top, mlyr_bottom = _resolve_mlyr(ds, mlyr_bnames)
    if mlyr_top is None:
        return

    # Beam height field
    if "beamc_height" not in ds:
        return
    bh = ds["beamc_height"]

    # Vectorised nearest-range lookup
    mlyr_top_idx = abs(bh - mlyr_top).argmin(dim="range")
    mlyr_bottom_idx = abs(bh - mlyr_bottom).argmin(dim="range")
    # =============================================================================
    # CARTOPY MODE
    # =============================================================================
    if cartopy_enabled:
        if projcoord_names is None:
            return

        xname = projcoord_names.get("x")
        yname = projcoord_names.get("y")

        if xname not in ds.coords or yname not in ds.coords:
            return

        proj_x = ds[xname]      # (azimuth, range)
        proj_y = ds[yname]

        # Ensure shapes match beam height
        if proj_x.shape != bh.shape:
            return

        # Extract ML boundary coordinates
        mlt_x = proj_x.isel(range=mlyr_top_idx).values
        mlt_y = proj_y.isel(range=mlyr_top_idx).values
        mlb_x = proj_x.isel(range=mlyr_bottom_idx).values
        mlb_y = proj_y.isel(range=mlyr_bottom_idx).values

        # Plot using data CRS
        line_top = ax.plot(
            mlt_x, mlt_y, c="k", ls="-", alpha=0.85,
            transform=data_crs,
            path_effects=[pe.Stroke(linewidth=4, foreground="w"), pe.Normal()],
            label=r"$MLyr_{(T)}$")[0]
        line_bot = ax.plot(
            mlb_x, mlb_y, c="grey", ls="-", alpha=0.85,
            transform=data_crs,
            path_effects=[pe.Stroke(linewidth=4, foreground="w"), pe.Normal()],
            label=r"$MLyr_{(B)}$")[0]
        legend = ax.legend(handles=[line_top, line_bot], loc="upper left")
        ax.add_artist(legend)
        return
    # =============================================================================
    # POLAR MODE
    # =============================================================================
    if coord_sys == "polar" and {"range", "azimuth"} <= set(ds.coords):

        r_km = convert(ds["range"], "km")
        az = ds["azimuth"]
        if polarplot:
            az = np.deg2rad(az)

        ml_top = r_km.isel(range=mlyr_top_idx).values
        ml_bot = r_km.isel(range=mlyr_bottom_idx).values

        line_top = ax.plot(
            az.values, ml_top, c="k", ls="-", alpha=0.85,
            path_effects=[pe.Stroke(linewidth=4, foreground="w"), pe.Normal()],
            label=r"$MLyr_{(T)}$")[0]
        line_bot = ax.plot(
            az.values, ml_bot, c="grey", ls="-", alpha=0.85,
            path_effects=[pe.Stroke(linewidth=4, foreground="w"), pe.Normal()],
            label=r"$MLyr_{(B)}$")[0]
        legend = ax.legend(handles=[line_top, line_bot], loc="upper left")
        ax.add_artist(legend)
        return
    # =============================================================================
    #     RECTANGULAR MODE
    # =============================================================================
    coord_namex, coord_namey = resolve_rect_coords(ds, coord_names)
    if coord_sys == "rect" and coord_namex in ds.coords and coord_namey in ds.coords:

        rect_x = ds[coord_namex]
        rect_y = ds[coord_namey]

        if rect_x.shape != bh.shape:
            return

        mlt_x = rect_x.isel(range=mlyr_top_idx).values
        mlt_y = rect_y.isel(range=mlyr_top_idx).values
        mlb_x = rect_x.isel(range=mlyr_bottom_idx).values
        mlb_y = rect_y.isel(range=mlyr_bottom_idx).values

        line_top = ax.plot(
            mlt_x, mlt_y, c="k", ls="-", alpha=0.85,
            path_effects=[pe.Stroke(linewidth=4, foreground="w"), pe.Normal()],
            label=r"$MLyr_{(T)}$")[0]

        line_bot = ax.plot(
            mlb_x, mlb_y, c="grey", ls="-", alpha=0.85,
            path_effects=[pe.Stroke(linewidth=4, foreground="w"), pe.Normal()],
            label=r"$MLyr_{(B)}$")[0]

        legend = ax.legend(handles=[line_top, line_bot], loc="upper left")
        ax.add_artist(legend)
        return


def _plot_max_range(ax, ds, *, coord_names=None, proj_x=None, proj_y=None,
                    data_crs=None, color="gray", linewidth=1.0, **kwargs):
    """
    Plot the maximum range boundary in either rectangular or projected coordinates.
    """
    # Cartopy / projected branch
    if proj_x is not None and proj_y is not None and data_crs is not None:
        # Last column is the max range boundary
        ax.plot(proj_x[:, -1], proj_y[:, -1], color=color,
                linewidth=linewidth, transform=data_crs, **kwargs)
        return

    # Rectangular branch
    xname, yname = resolve_rect_coords(ds, coord_names)
    if xname is None or yname is None:
        return
    x = ds[xname].values
    y = ds[yname].values
    # Max range is the last row or last column depending on orientation
    # I assume y increases with range
    ax.plot(x, np.full_like(x, y[-1]), color=color, linewidth=linewidth,
            **kwargs)


def _plot_pixel_midpoints(ax, ds, *, coord_names=None, proj_x=None, proj_y=None,
                          data_crs=None, color="grey", alpha=0.2, marker="+",
                          **kwargs):
    """
    Plot pixel midpoints in either rectangular or projected coordinates.
    """
    # Cartopy / projected branch
    if proj_x is not None and proj_y is not None and data_crs is not None:
        ax.scatter(proj_x.ravel(), proj_y.ravel(), color=color,
                   alpha=alpha, marker=marker, transform=data_crs, **kwargs)
        return

    # Rectangular branch
    xname, yname = resolve_rect_coords(ds, coord_names)
    if xname is None or yname is None:
        return

    xx = ds[xname].values.reshape(-1)
    yy = ds[yname].values.reshape(-1)

    ax.scatter(xx, yy, color=color, alpha=alpha, marker=marker, **kwargs)


def _plot_range_rings(ax, ds, range_rings, *, coord_sys, polarplot=True,
                      coord_names=None, color="grey", alpha=0.75, **kwargs):
    """
    Plot range rings in either rectangular or polar coordinates.
    """
    # Resolve range coordinate
    if "range" not in ds.coords:
        return
    # hardcode range
    r_km = convert(ds["range"], "km")

    # Normalise range_rings input
    if isinstance(range_rings, range):
        range_rings = list(range_rings)

    if isinstance(range_rings, (int, float)):
        # spacing in km → generate rings up to max range
        nrings_km = np.arange(range_rings, float(r_km.max()), range_rings)
    else:
        # explicit radii in km
        nrings_km = np.asarray(range_rings, dtype=float)

    # Find nearest range indices
    idx_rs = [find_nearest(r_km.values, rr) for rr in nrings_km]
    # =========================================================================
    # RECTANGULAR MODE
    # =========================================================================
    if coord_sys == "rect":
        xname, yname = resolve_rect_coords(ds, coord_names)
        if xname is None:
            return

        rect_x = ds[xname]
        rect_y = ds[yname]

        rings_x = np.stack([rect_x.isel(range=i).values for i in idx_rs])
        rings_y = np.stack([rect_y.isel(range=i).values for i in idx_rs])

        ax.scatter(rings_x, rings_y, color=color, alpha=alpha, marker=".",
                   **kwargs)
        # Crosshairs
        ax.axhline(0, color=color, ls="--", alpha=alpha)
        ax.axvline(0, color=color, ls="--", alpha=alpha)
        ax.grid(True)
        return
    # =========================================================================
    # POLAR MODE
    # =========================================================================
    if coord_sys == "polar":
        if "azimuth" not in ds.coords:
            return

        az = ds["azimuth"]
        if polarplot:
            az = np.deg2rad(az)

        # For each ring, plot a constant radius vs azimuth
        for i in idx_rs:
            rr_km = float(r_km.isel(range=i))
            ax.plot(az.values, np.full_like(az.values, rr_km),
                    color=color, alpha=alpha, ls="--", **kwargs)
        ax.grid(True)
        return


def _plot_contours(ax, ds, varname, *, coord_sys, polarplot=True,
                   coord_names=None, contour_kw, xrdsvar_sorted=None,
                   az_grid=None, r_grid_km=None, fs_label=10):
    """
    Plot contour overlays in either rectangular or polar coordinates.
    """
    # Resolve the data variable
    if varname not in ds:
        return
    da = ds[varname]

    # Merge default and user contour kwargs
    ckw = {"alpha": 0.5, "zorder": 2, "colors": None, "legend": False}
    if contour_kw:
        ckw.update(contour_kw)
    # =========================================================================
    # RECTANGULAR MODE
    # =========================================================================
    if coord_sys == "rect":
        xname, yname = resolve_rect_coords(ds, coord_names)
        if xname is None:
            return
        rect_x = ds[xname]
        rect_y = ds[yname]
        contour = ax.contour(rect_x.values, rect_y.values, da.values,
                             **contour_kw,)
        ax.clabel(contour, inline=True, fontsize=fs_label)
        if ckw["legend"]:
            handles, labels = contour.legend_elements()
            labels = [lb.replace("x = ", "") for lb in labels]
            ax.legend(handles, labels, title=varname, loc="upper right").set_zorder(5)
        return
    # =========================================================================
    # POLAR MODE
    # =========================================================================
    if coord_sys == "polar":
        if "azimuth" not in ds.coords or "range" not in ds.coords:
            return
        # da must be 2D: (azimuth, range)
        contour = ax.contour(az_grid.values, r_grid_km.values,
                             da.values, **contour_kw)
        ax.clabel(contour, inline=True, fontsize=fs_label)
        if ckw["legend"]:
            handles, labels = contour.legend_elements()
            labels = [lb.replace("x = ", "") for lb in labels]
            ax.legend(handles, labels, title=varname, loc="upper right").set_zorder(5)
        return


def _plot_points(ax, points, coord_names,
                 ptsvar=None, norm=None, cmap=None, size=20):

    # Validate coordinate names
    required = {coord_names["x"], coord_names["y"]}
    missing = required - set(points.keys())
    if missing:
        raise ValueError(
            f"points2plot is missing required coordinates: {missing}")
    # Extract coordinates
    x = points[coord_names["x"]]
    y = points[coord_names["y"]]
    # Scatter logic
    if ptsvar is None or ptsvar not in points:
        ax.scatter(x, y, color="k", marker="o", s=size)
    else:
        ax.scatter(x, y, c=points[ptsvar], cmap=cmap, norm=norm, marker="o",
                   edgecolors="k", s=size)

# =============================================================================
# Cartopy features
# =============================================================================
default_cartopy_features = [
    {"feature": cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "10m",
        facecolor="none"), "kwargs": {"edgecolor": "black", "ls": ":"}},
    {"feature": cfeature.NaturalEarthFeature(
        "cultural", "admin_0_countries", "10m", facecolor="none"),
        "kwargs": {"edgecolor": "black"}}]

def _print_cartopy_licenses(cfg):
    print("© Natural Earth; license: public domain")

    tiles = cfg.get("tiles", {})
    tile_class = tiles.get("class")

    if tile_class is None:
        return

    # Stamen
    if tile_class.__name__ == "Stamen":
        print("© Map tiles by Stamen Design (CC BY 3.0). "
              "Data by OpenStreetMap (ODbL).")

    # Mapbox
    if tile_class.__name__ == "MapboxTiles":
        print("© Mapbox; see https://www.mapbox.com/legal/tos/")

    # GoogleTiles
    if tile_class.__name__ == "GoogleTiles":
        print("© Google Maps tiles; usage subject to Google Terms of Service")


def _add_cartopy_tiles(ax, cfg):
    """Add tile imagery to a Cartopy GeoAxes."""
    tcfg = cfg.get("tiles", {})
    if not tcfg.get("enabled", False):
        return

    tile_class = tcfg.get("class")
    if tile_class is None:
        raise ValueError("tiles.enabled=True but no tile class provided")

    tile_kwargs = tcfg.get("kwargs", {})
    resolution = tcfg.get("resolution", 8)
    alpha = tcfg.get("alpha", 1.0)

    tile_source = tile_class(**tile_kwargs)
    ax.add_image(tile_source, resolution, alpha=alpha)


def _add_cartopy_features(ax, cfg):
    """Add arbitrary Cartopy features to the GeoAxes."""
    for item in cfg.get("features", []):
        feat = item["feature"]
        kwargs = item.get("kwargs", {})
        ax.add_feature(feat, **kwargs)


def _add_cartopy_gridlines(ax, cfg):
    """Add gridlines with optional labels."""
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    gl_cfg = cfg.get("gridlines", {})
    if not gl_cfg.get("enabled", True):
        return

    gl = ax.gridlines(draw_labels=gl_cfg.get("draw_labels", True),
                      dms=False, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlabel_style = {"size": gl_cfg.get("label_size", 10)}
    gl.ylabel_style = {"size": gl_cfg.get("label_size", 10)}


def plot_ppi_xr(xrds, var2plot=None, coord_sys='polar', polarplot=False,
             polarcoord_names={"azi": "azimuth", "rng": "range"},
             rectcoord_names={"x": "grid_rectx", "y": "grid_recty"},
             projcoord_names={"x": "grid_osgbx", "y": "grid_osgby"},
             cartopy_cfg=None, vars_bounds=None, cb_ext=None, unorm=None,
             ucmap=None, cbticks=None, font_sizes='regular',
             fig_size=None, fig_title=None, fig=None, ax1=None, plot_grid=False,
             plot_axislabels=True, plot_mlyr=False, mlyr_bnames=None,
             plot_contourl=None, contour_kw=None, range_rings=None,
             rd_maxrange=False, points2plot=None, ptsvar2plot=None,
             pixel_midp=False, add_colorbar=False, custom_rules=None,
             xlims=None, ylims=None):
    """
    Plot a radar Plan Position Indicator (PPI) scan from an xarray Dataset.

    This function provides a flexible and extensible interface for visualising
    radar sweep data in polar, rectangular, or projected (Cartopy) coordinate
    systems. It supports colour‑mapped fields, contours, range rings, point
    overlays, melting‑layer boundaries, and various customisation options.

    Parameters
    ----------
    xrds : xarray.Dataset
        Radar sweep dataset containing the variable to plot and the required
        coordinate fields. Must include azimuth/range for polar mode, or
        rectangular/projected coordinates for rect/cartopy modes.
    var2plot : str or None, optional
        Name of the variable to display. If None, the function attempts to
        infer a suitable default from the dataset.
    coord_sys : {'polar', 'rect'}, default 'polar'
        Coordinate system used for plotting. If Cartopy is enabled, this is
        overridden to 'rect'.
    polarplot : bool, default False
        If True, use a polar projection (radians) for azimuth. If False, plot
        azimuth on the x‑axis in degrees.
    polarcoord_names : dict, default {'azi': 'azimuth', 'rng': 'range'}
        Names of azimuth and range coordinates in polar mode.
    rectcoord_names : dict, default {'x': 'grid_rectx', 'y': 'grid_recty'}
        Names of rectangular coordinates for non‑Cartopy rect mode.
    projcoord_names : dict, default {'x': 'grid_osgbx', 'y': 'grid_osgby'}
        Names of projected coordinates for Cartopy mode.
    cartopy_cfg : dict or None
        Optional configuration dictionary controlling Cartopy-based plotting.
        Any keys provided here override the defaults returned by
        :func:`default_cartopy_config`. See ``help(default_cartopy_config)``
        for the full configuration structure and available options.
    vars_bounds, cb_ext, unorm, ucmap: optional
        Parameters forwarded to the colour‑mapping logic via `plot_params`.
    cbticks : list or None
        Custom tick locations for the colourbar.
    font_sizes : {'regular', 'large'}, default 'regular'
        Controls font sizes for labels, ticks, and titles.
    fig_size : tuple or None
        Figure size in inches. If None, a sensible default is chosen based on
        the coordinate system.
    fig_title : str or None
        Custom figure title. If None, a metadata‑derived title is used.
    fig, ax1 : matplotlib Figure and Axes or None
        Existing figure/axes to draw on. If None, new ones are created.
    plot_grid : bool, default True
        Whether to draw gridlines on the axes.
    plot_axislabels : bool, default True
        Whether to label the axes.
    plot_mlyr : bool, default False
        If True, overlay melting‑layer boundaries (top and bottom). Requires
        melting‑layer metadata in the dataset or user‑specified variable names.
    mlyr_bnames : dict or None
        Optional mapping specifying which dataset variables contain the melting
        layer boundaries, e.g.:
            {'top': 'MLT', 'bottom': 'MLB'}
        If None, defaults to 'MLYRTOP' and 'MLYRBTM'.
    plot_contourl : str or None
        Name of a variable to overlay as contours.
    contour_kw : dict or None
        Additional keyword arguments passed to `ax.contour`.
    range_rings : list or None
        Multiple range rings to draw (in km).
    rd_maxrange : bool, default False
        If True, draw a circle marking the radar's maximum range.
    points2plot : array-like or None
        Optional set of points to overlay on the PPI.
    ptsvar2plot : str or None
        Variable used to colour the points in `points2plot`.
    pixel_midp : bool, default False
        If True, overlay pixel midpoints.
    add_colorbar : bool, default False
        Whether to add a colourbar to the plot.
    custom_rules : dict or None
        Optional custom colour‑mapping rules for
        :func:`towerpy.datavis.rad_display.plot_params`.
    xlims, ylims : tuple or None
        Axis limits for rectangular or projected plots.

    Returns
    -------
    mappable : matplotlib.cm.ScalarMappable
        The object associated with the colour‑mapped field.
    ax1 : matplotlib Axes
        The axes on which the PPI is drawn.

    Notes
    -----
    - When Cartopy is enabled, `coord_sys` is forced to 'rect' and `polarplot`
      is ignored.
    - Melting‑layer plotting uses vectorised nearest‑range lookup based on the
      dataset's beam‑height fields.
    """
    # Normalise input
    xrds = xrds.squeeze(drop=True)
    xrds, da_name = _as_dataset(xrds)
    # Resolve variable
    var2plot = _resolve_var2plot(xrds, var2plot)
    # Extract metadata safely
    meta = _safe_metadata(xrds)
    elev_str = meta["elev_str"]
    dt_str = meta["dt_str"]
    rname = meta["rname"]
    swp_mode = meta["swp_mode"]
    mappable = None
    # =============================================================================
    # Creates plotting parameters 
    # =============================================================================
    fsizes = {'fsz_cb': 10, 'fsz_cbt': 12, 'fsz_pt': 14, 'fsz_axlb': 12,
              'fsz_axtk': 10}
    if font_sizes == 'large':
        fsizes = {k1: v1 + 4 for k1, v1 in fsizes.items()}
    # szpnts = 25
    szpnts = None
    pltprms = plot_params(var2plot, xrds, vars_bounds=vars_bounds,
                          unorm=unorm, cb_ext=cb_ext,
                          custom_rules=custom_rules,
                          ucmap=ucmap)
    # =============================================================================
    # Title metadata 
    # =============================================================================
    # ptitle = fig_title or f"{rname.title()} [{swp_mode}{elev_str}] -- {dt_str}"
    # =============================================================================
    # Cartopy features
    # =============================================================================
    default_cartopy_cfg = default_cartopy_config()
    if cartopy_cfg is not None:
        default_cartopy_cfg = _deep_update(default_cartopy_cfg, cartopy_cfg)
    if default_cartopy_cfg["enable_cartopy"] and default_cartopy_cfg["data_crs"] is None:
        raise ValueError("Cartopy plotting enabled but no data CRS provided. "
                         "Set cartopy_cfg={'data_crs': <your CRS>}.")
    if default_cartopy_cfg["enable_cartopy"]:
        coord_names = projcoord_names
    elif coord_sys == "rect":
        coord_names = rectcoord_names
    elif coord_sys == "polar":
        coord_names = polarcoord_names
    # =============================================================================
    # Coordinates
    # =============================================================================
    # Safeguard plot mode combinations
    # Cartopy + polar is invalid → force rect
    if default_cartopy_cfg["enable_cartopy"]:
        if coord_sys == "polar":
            coord_sys = "rect"
        polarplot = False
    # Rectangular mode cannot use polarplot
    if coord_sys == "rect":
        polarplot = False
    # Polar mode cannot use Cartopy
    if coord_sys == "polar":
        default_cartopy_cfg["enable_cartopy"] = False
    coord_namex, coord_namey = resolve_rect_coords(xrds, coord_names)
    has_polar = {polarcoord_names.get('rng'), polarcoord_names.get('azi')} <= set(xrds.coords)
    has_rect = (coord_namex is not None and coord_namey is not None
                and {coord_namex, coord_namey} <= set(xrds.coords))
    rect_x = xrds.coords.get(coord_namex) if has_rect else None
    rect_y = xrds.coords.get(coord_namey) if has_rect else None
    if default_cartopy_cfg["enable_cartopy"]:
        proj_x = xrds.coords.get(projcoord_names.get('x'))
        proj_y = xrds.coords.get(projcoord_names.get('y'))
        if proj_x is None or proj_y is None:
            raise ValueError(
                f"Cartopy plotting requested (enable_cartopy=True) but projected "
                f"coordinates {projcoord_names.get('x')}' / '{projcoord_names.get('y')}' "
                f"are missing in xrds.coords. Either disable Cartopy or provide "
                f"projected coordinates.")
    # =============================================================================
    # Init figure    
    # =============================================================================
    if fig_size is None:
        if coord_sys == "polar":
            if polarplot:
                fig_size = (6, 6.15)
            else:
                fig_size = (10, 6)
        elif coord_sys == "rect":
            if default_cartopy_cfg["enable_cartopy"]:
                fig_size = (10, 6)
            else:
                fig_size = (6, 7)

    if default_cartopy_cfg["enable_cartopy"]:
        proj = default_cartopy_cfg["projection"]
        subplot_kw = {"projection": proj}
        use_constrained = True
    elif polarplot:
        subplot_kw = {"projection": "polar"}
        use_constrained = False
    else:
        subplot_kw = {}
        use_constrained = False
    if fig is None or ax1 is None:
        fig, ax1 = plt.subplots(figsize=fig_size, subplot_kw=subplot_kw,
                                constrained_layout=use_constrained)
    # =============================================================================
    # Plot polar plot
    # =============================================================================
    if coord_sys == "polar":
        if has_polar:
            az = xrds[polarcoord_names.get('azi')]
            if polarplot:
                az = np.deg2rad(az)
            r = convert(xrds[polarcoord_names["rng"]], "km")
            # Broadcast to 2D grid
            az_grid, r_grid_km = xr.broadcast(az, r)
        else:
            az_grid = r_grid_km = None
        if has_polar and az_grid is not None:
            # xrdsvar_sorted = xrds[var2plot].sortby("azimuth", ascending=True)
            xrdsvar_sorted = xrds[var2plot]
            mappable = ax1.pcolormesh(
                az_grid, r_grid_km, xrdsvar_sorted, shading="auto",
                cmap=pltprms.cmap, norm=pltprms.norm)
            if polarplot:
                ax1.set_theta_zero_location('N')
                ax1.set_theta_direction(-1)
                ax1.set_thetagrids(np.arange(0, 360, 90))
                ax1.set_yticklabels([])
                ax1.axes.set_aspect('equal')
                if plot_grid:
                    ax1.grid(color='gray', linestyle=':')
            ax1.tick_params(axis='both', labelsize=fsizes['fsz_axlb'])
            ax1.axes.set_aspect('equal')
    # =============================================================================
    # Plot plot in rect coord_sys, but no cartopy
    # =============================================================================
    elif coord_sys == 'rect' and default_cartopy_cfg['enable_cartopy'] is False:
        xrdsvar_sorted = None
        az_grid=None
        r_grid_km=None
        if has_rect and rect_x is not None and rect_y is not None:
            mappable = ax1.pcolormesh(rect_x, rect_y, xrds[var2plot],
                                      shading="auto", cmap=pltprms.cmap,
                                      norm=pltprms.norm)
    # =============================================================================
    # Plot plot in proj coord_sys, using cartopy
    # =============================================================================
    elif default_cartopy_cfg["enable_cartopy"]:
        # 1. Set extent if provided
        if xlims and ylims:
            ax1.set_extent([*xlims, *ylims], crs=proj)
        # 2. Add tiles
        _add_cartopy_tiles(ax1, default_cartopy_cfg)
        # 3. Add features
        _add_cartopy_features(ax1, default_cartopy_cfg)
        # 4. Add gridlines
        # _add_cartopy_gridlines(ax1, default_cartopy_cfg)
        # 5. Print licensing information
        _print_cartopy_licenses(default_cartopy_cfg)
        # 6. Plot the radar field
        mappable = ax1.pcolormesh(proj_x, proj_y, xrds[var2plot],
                                  transform=default_cartopy_cfg['data_crs'],
                                  shading="auto", cmap=pltprms.cmap,
                                  norm=pltprms.norm,
                                  alpha=default_cartopy_cfg["alpha_rad"])
    if mappable is None:
        raise RuntimeError(
            "plot_ppi could not find the appropiate coords for the selected"
            " plotting params:\n"
            f"coord_sys='{coord_sys}', but has_polar={has_polar}, has_rect={has_rect}, "
            f"cartopy_enabled={default_cartopy_cfg['enable_cartopy']}.\n"
            "Dataset coords: " + ", ".join(list(xrds.coords)))

    # =============================================================================
    # Creates colorbar         
    # =============================================================================
    vunits = _safe_units(xrds[var2plot])
    if add_colorbar:
        _add_colorbar(fig, ax1, mappable, pltprms, fsizes, vunits,
                      coord_sys=coord_sys,
                      cartopy_enabled=default_cartopy_cfg["enable_cartopy"])
    # =============================================================================
    # Plot Artist objects     
    # =============================================================================
    mappable._pltprms = pltprms
    # Optional melting layer overlay
    if plot_mlyr:
        _addML2plot(ax1, coord_sys, xrds, mlyr_bnames=mlyr_bnames,
                    coord_names=coord_names, polarplot=polarplot,
                    cartopy_enabled=default_cartopy_cfg["enable_cartopy"],
                    data_crs=default_cartopy_cfg["data_crs"],
                    projcoord_names=projcoord_names)
    if range_rings is not None:
        _plot_range_rings(ax1, xrds, range_rings, coord_sys=coord_sys,
                          polarplot=polarplot, coord_names=coord_names)
    if plot_contourl:
        ctrprms = plot_params(plot_contourl, xrds, vars_bounds=vars_bounds,
                              unorm=unorm, cb_ext=cb_ext,
                              custom_rules=custom_rules,
                              ucmap=ucmap)
    
        ckw = {"levels": ctrprms.norm_boundaries, "norm": ctrprms.norm, 
               "cmap": ctrprms.cmap}
        if contour_kw:
            ckw.update(contour_kw)
        _plot_contours(ax1, xrds, plot_contourl, coord_sys=coord_sys,
                       polarplot=polarplot, coord_names=coord_names,
                       contour_kw=ckw, fs_label=fsizes["fsz_cbt"], 
                       xrdsvar_sorted=xrdsvar_sorted, az_grid=az_grid,
                       r_grid_km=r_grid_km)
    if points2plot is not None:
        _plot_points(ax1, points2plot, coord_names=coord_names,
                     ptsvar=ptsvar2plot if len(points2plot) >= 3 else None,
                     norm=pltprms.norm, cmap=pltprms.cmap, size=szpnts)
    if pixel_midp:
        _plot_pixel_midpoints(ax1, xrds, coord_names=coord_names,
                              proj_x=proj_x, proj_y=proj_y,
                              data_crs=default_cartopy_cfg["data_crs"])
    if rd_maxrange:
        _plot_max_range(ax1, xrds, coord_names=coord_names, proj_x=proj_x,
                        proj_y=proj_y, data_crs=default_cartopy_cfg["data_crs"])
    # =============================================================================
    # Set plot params
    # =============================================================================
    if plot_grid:
        ax1.grid(True)
    if coord_sys == "rect" and has_rect:
        if plot_axislabels:
            if default_cartopy_cfg["enable_cartopy"]:
                # da_x = xrds[coord_namex]
                # da_y = xrds[coord_namey]
                # x_std = da_x.attrs.get("standard_name", coord_namex)
                # y_std = da_y.attrs.get("standard_name", coord_namey)
                # x_unit = da_x.attrs.get("units", "")
                # y_unit = da_y.attrs.get("units", "")
                # x_label = f"{x_std} [{x_unit}]" if x_unit else x_std
                # y_label = f"{y_std} [{y_unit}]" if y_unit else y_std
                x_label = "longitude [degrees_east]"
                y_label = "latitude [degrees_north]"
                # Draw lon/lat ticks
                # Determine extent for tick generation
                if xlims is not None and ylims is not None:
                    xmin, xmax = xlims
                    ymin, ymax = ylims
                else:
                    # Get actual map extent in PlateCarree coordinates
                    xmin, xmax, ymin, ymax = ax1.get_extent(
                        crs=ccrs.PlateCarree())
                dx = default_cartopy_cfg["tick_spacing"]["dx"]
                dy = default_cartopy_cfg["tick_spacing"]["dy"]
                # xticks = np.arange(np.floor(xlims[0]),
                #                    np.ceil(xlims[1]) + dx, dx)
                # yticks = np.arange(np.floor(ylims[0]),
                #                    np.ceil(ylims[1]) + dy, dy)
                xticks = np.arange(np.floor(xmin), np.ceil(xmax) + dx, dx)
                yticks = np.arange(np.floor(ymin), np.ceil(ymax) + dy, dy)
                ax1.set_xticks(xticks, crs=ccrs.PlateCarree())
                ax1.set_yticks(yticks, crs=ccrs.PlateCarree())
                ax1.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
                ax1.yaxis.set_major_formatter(LATITUDE_FORMATTER)
            elif has_polar:
                # Only use range labels when NOT in Cartopy mode
                r_std = xrds[polarcoord_names['rng']].attrs.get("standard_name", "Range")
                r_unit_src = getcoordunits(xrds, polarcoord_names['rng'], "m")
                r_unit = "km" if r_unit_src.startswith("m") else r_unit_src
                x_label = f"{r_std} [{r_unit}]"
                y_label = f"{r_std} [{r_unit}]"
            else:
                x_label = coord_namex
                y_label = coord_namey
            ax1.set_xlabel(x_label, fontsize=fsizes['fsz_axlb'])
            ax1.set_ylabel(y_label, fontsize=fsizes['fsz_axlb'])
        ax1.axes.set_aspect('equal')
    elif coord_sys == "polar" and has_polar and not polarplot:
        if plot_axislabels:
            a_std = xrds[polarcoord_names.get('azi')].attrs.get("standard_name", "Azimuth")
            a_unit = "rad" if polarplot else getcoordunits(xrds, polarcoord_names.get('azi'), "deg")
            r_std = xrds[polarcoord_names.get('rng')].attrs.get("standard_name", "Range")
            r_unit_src = getcoordunits(xrds, polarcoord_names.get('rng'), "m")
            r_unit = "km" if r_unit_src.startswith("m") else r_unit_src
            x_label = f"{a_std} [{a_unit}]"
            y_label = f"{r_std} [{r_unit}]"
            ax1.set_xlabel(x_label, fontsize=fsizes['fsz_axlb'])
            ax1.set_ylabel(y_label, fontsize=fsizes['fsz_axlb'])
    # =============================================================================
    # Resolve title
    # =============================================================================
    if fig_title is not None:
        if coord_sys == "rect" and not default_cartopy_cfg["enable_cartopy"]:
            fig.suptitle(fig_title, fontsize=fsizes["fsz_pt"])
        else:
            ax1.set_title(fig_title, fontsize=fsizes["fsz_pt"])
    else:
        ptitle = f"{rname.title()} -- {dt_str}"
        auto_title = f"{ptitle}\n{elev_str} {swp_mode}\nPPI {var2plot} [{vunits}]"
        if coord_sys == "rect" and not default_cartopy_cfg["enable_cartopy"]:
            fig.suptitle(auto_title, fontsize=fsizes["fsz_pt"])
        else:
            ax1.set_title(auto_title, fontsize=fsizes["fsz_pt"])
    if default_cartopy_cfg['enable_cartopy'] is False:
        if xlims:
            ax1.set_xlim(xlims)
        if ylims:
            ax1.set_ylim(ylims)
        fig.tight_layout()
    return mappable, ax1


def plot_setppi_xr(xrds, varnames=None, coord_sys="polar", polarplot=False,
                ncols=None, nrows=None, fig_size=None, fig_title=None,
                # polarcoord_names = {"azi": "azimuth", "rng": "range"},
                # rectcoord_names = {"x": "grid_rectx", "y": "grid_recty"},
                add_colorbar=False, **ppi_kwargs):
    """
    Plot a set of PPIs from a single xarray.Dataset using plot_ppi().
    """
    # =============================================================================
    # Title metadata 
    # =============================================================================
    # Extract metadata safely
    meta = _safe_metadata(xrds)
    elev_str = meta["elev_str"]
    dt_str = meta["dt_str"]
    rname = meta["rname"]
    swp_mode = meta["swp_mode"]

    ptitle = fig_title or f"{rname.title()} [{swp_mode}] -- {dt_str}"

    # Variables to plot 
    if varnames is None:
        varnames = list(xrds.data_vars.keys())
    valid_vars = []
    for name in varnames:
        da = xrds[name]
        if da.ndim == 2 and set(da.dims) == {"azimuth", "range"}:
            valid_vars.append(name)
        # else:
        #     print(f"Skipping {name!r}: dims={da.dims} (expected ('azimuth', 'range')).")
        
    nvars = len(valid_vars)

    # Grid layout 
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(nvars)))
        nrows = int(np.ceil(nvars / ncols))
    elif nrows is None:
        nrows = int(np.ceil(nvars / ncols))
    elif ncols is None:
        ncols = int(np.ceil(nvars / nrows))
    if fig_size is None:
        fig_size = (6 * ncols, 6 * nrows)

    # Inti figure
    cartopy_cfg = ppi_kwargs.get("cartopy_cfg", None)
    if cartopy_cfg and cartopy_cfg.get("enable_cartopy", False):
        # proj = cartopy_cfg["projection"]
        proj = cartopy_cfg.get("projection", ccrs.PlateCarree())
        subplot_kw = {"projection": proj}
        use_constrained = True
        sharex, sharey = False, False
    elif coord_sys == "polar" and polarplot:
        subplot_kw = {"projection": "polar"}
        use_constrained = False
        sharex, sharey = False, False
    else:
        subplot_kw = {}
        use_constrained = False
        sharex, sharey = True, True
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size, sharex=sharex,
                             sharey=sharey, subplot_kw=subplot_kw,
                             constrained_layout=use_constrained)
    axes = np.atleast_1d(axes).flatten()

    # Plot each variable using plot_ppi
    for ax, vname in zip(axes, valid_vars):
        pcm, ax1 = plot_ppi_xr(
                xrds,
                var2plot=vname,
                coord_sys=coord_sys,
                polarplot=polarplot,
                # cartopy_cfg=cartopy_cfg,
                # coord_names=coord_names,
                # xlims=xlims,
                # ylims=ylims,
                fig=fig,
                ax1=ax,
                add_colorbar=False,
                fig_title=None,
                plot_axislabels=False,
                **ppi_kwargs
            )
        ax1.grid(True)
        vunits = _safe_units(xrds[vname])
        ax1.set_title(f'{vname} [{vunits}]', fontsize=12)
        if add_colorbar:
            pltprms = pcm._pltprms
            if 'rhohv' in vname.lower():
                clb = ax1.figure.colorbar(
                    pcm, ax=ax1, ticks=pltprms.norm_boundaries)
            else:
                clb = ax1.figure.colorbar(pcm, ax=ax1,)
            if pltprms.ticklabels is not None:
                clb.set_ticks(pltprms.norm_boundaries)
                clb.set_ticklabels(pltprms.ticklabels)
                clb.ax.tick_params(direction="in")

    # Label only first column and last row 
    if coord_sys == "rect":
        r_std = xrds["range"].attrs.get("standard_name", "Range")
        r_unit_src = getcoordunits(xrds, "range", "m")
        r_unit = "km" if r_unit_src.startswith("m") else r_unit_src
        x_label = f"{r_std} [{r_unit}]"
        y_label = f"{r_std} [{r_unit}]"
    elif coord_sys == "polar" and not polarplot:
        a_std = xrds["azimuth"].attrs.get("standard_name", "Azimuth")
        a_unit = "rad" if polarplot else getcoordunits(xrds, "azimuth", "deg")
        r_std = xrds["range"].attrs.get("standard_name", "Range")
        r_unit_src = getcoordunits(xrds, "range", "m")
        r_unit = "km" if r_unit_src.startswith("m") else r_unit_src
        x_label = f"{a_std} [{a_unit}]"
        y_label = f"{r_std} [{r_unit}]"
    elif coord_sys == "polar" and polarplot:
        x_label = ""
        y_label = ""
    else:
        x_label = "X"
        y_label = "Y"
    for i in range(nrows):
        axes[i*ncols].set_ylabel(y_label, fontsize=12)
    for j in range(ncols):
        axes[(nrows-1)*ncols + j].set_xlabel(x_label, fontsize=12)
    # Remove unused axes 
    for ax in axes[nvars:]:
        ax.remove()
    fig.suptitle(ptitle, fontsize=16)
    # fig.tight_layout()
    
    return fig, axes

# =============================================================================
# %%% Methodolgies
# =============================================================================

def _plot_rhohvmethod_single(snr_edges, rho_edges, hist, snr_db, rhohv_na,
                             snr_centers, theo_line, histmax, opt_noise,
                             fig_size=(8, 6)):
    fig, ax = plt.subplots(figsize=fig_size)
    pcm = ax.pcolormesh(snr_edges, rho_edges, hist.T,
                        norm=mpc.LogNorm(vmin=10**0, vmax=10**1),
                        cmap="tpylsc_useq_calm_r")

    ax.plot(snr_centers, theo_line, color="tab:purple", lw=3,
            label=r"theoretical $\rho_{HV}$")
    ax.plot(snr_centers, histmax, color="k", lw=2, ls=":",
            label="histogram maxima")
    ax.scatter(snr_db.values.flatten(), rhohv_na.values.flatten(),
               s=2, alpha=0.2, color="grey", label=r"raw $\rho_{HV}$")
    ax.axhline(1.0, ls="--", color="tab:red")
    ax.set_title(f"Noise level rc = {opt_noise:.2f} dB")
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel(r"$\rho_{HV}$ [-]")
    ax.set_xlim(5, 30)
    ax.set_ylim(0.8, 1.1)
    fig.colorbar(pcm, ax=ax, label="n points")
    ax.legend()
    plt.show()


def _plot_rhohvmethod_grid(Z, rng_km, rhohv_na, mode="linear", exp_curvet=20.0,
                           eps=0.005, rhohv_theo=(0.90, 1.0), opt_noise=None,
                           bins_snr=(5, 30, 0.1), bins_rho=(0.8, 1.1, 0.005),
                           fig_size=(16, 9)):
    """
    Plot calibration grid around the optimised noise level.
    Shows histograms for opt_noise ±5 dB in 1 dB steps,
    using corrected rhoHV (rhohv_corr).
    """
    from ..calib.calib_rhohv import _build_theo_line
    from ..eclass.snr import signal2noiseratio
    from ..utils.radutilities import xr_hist2d

    if opt_noise is None:
        raise ValueError("opt_noise must be provided (final noise_level_dB).")

    rc_values = np.arange(opt_noise - 4, opt_noise + 5, 1.0)

    snr_edges = np.arange(*bins_snr)
    rho_edges = np.arange(*bins_rho)
    rhohv_centers = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    snr_centers = 0.5 * (snr_edges[:-1] + snr_edges[1:])

    hists, histmax, theo_lines = [], [], []
    for rc in rc_values:
        snr_db = signal2noiseratio(Z, rng_km, rc, scale="db")
        snr_lin = signal2noiseratio(Z, rng_km, rc, scale="lin")
        rhohv_corr = (rhohv_na * (1 + 1 / snr_lin)).rename("rhohv_corr")
        hist = xr_hist2d(snr_db, rhohv_corr, snr_edges, rho_edges,
                         dim=list(snr_db.dims))
        hists.append((hist.values, snr_edges, rho_edges))

        rhohv_bin_dim = [d for d in hist.dims if d.endswith("_bin")][1]
        idx = hist.argmax(dim=rhohv_bin_dim)
        maxima = xr.apply_ufunc(lambda i: rhohv_centers[i], idx,
                                vectorize=True, dask="parallelized",
                                output_dtypes=[float]).values
        histmax.append(maxima)

        theo_line = _build_theo_line(snr_centers, rhohv_theo, mode=mode,
                                     exp_curvet=exp_curvet, eps=eps)
        theo_lines.append(theo_line)

    nc = min(3, len(hists))
    nr = len(hists) // nc + (len(hists) % nc > 0)
    fig, axes = plt.subplots(nrows=nr, ncols=nc,
                             sharex=True, sharey=True,
                             figsize=fig_size, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i < len(hists):
            H, snr_edges, rho_edges = hists[i]
            maxima = histmax[i]
            theo_line = theo_lines[i]
            rc = rc_values[i]

            if np.isclose(rc, opt_noise, atol=0.5):
                title_color, weight = "tab:purple", "bold"
            else:
                title_color, weight = "tab:grey", "normal"

            ax.set_title(f"{rc:.2f} dB", color=title_color, fontweight=weight)
            ax.plot(snr_edges[1:], maxima, color="k", lw=2, ls=":",
                    label="histogram maxima")
            ax.plot(snr_centers, theo_line, color="tab:purple", lw=2,
                    label="theoretical line")
            ax.axhline(1.0, ls="--", color="tab:red")
            pcm = ax.pcolormesh(snr_edges, rho_edges, H.T,
                                norm=mpc.LogNorm(vmin=10**0, vmax=10**1),
                                cmap="tpylsc_useq_calm_r", rasterized=True)
            ax.tick_params(axis="both", which="major", labelsize=9)

    clb = fig.colorbar(pcm, ax=axes, location="right", shrink=0.85)
    clb.ax.set_title("n points")
    fig.supxlabel("SNR [dB]")
    fig.supylabel(r"$\rho_{HV}$ [-]")
    plt.show()

# =============================================================================
# %%% Profiles
# =============================================================================

def _add_colored_profile(ax, x, y, cmap, norm, ticks=None, fmt="%.1f"):
    """
    Plot a coloured profile (value vs height) using LineCollection.
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(x)
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # Colorbar
    cbar = ax.get_figure().colorbar(lc, ax=ax, orientation="horizontal")
    if ticks is not None:
        cbar.set_ticks(ticks)
        cbar.ax.set_xticklabels([fmt % t for t in ticks])
    cbar.ax.tick_params(labelsize=10)

    # Set x-limits based on data
    if np.isfinite(np.nanmin(x)) and np.isfinite(np.nanmax(x)):
        ax.set_xlim(np.nanmin(x), np.nanmax(x))


def plot_profiles(ds, stats=None, colours=False, mlyr_top=None, mlyr_btm=None,
                  vars_bounds=None, ucmap=None, unorm=None,
                  cb_ext=None, fig_title=None, fig_size=None, ylims=None):
    """
    Plot radar profiles (QVPs, VPs, or RD-QVPs) contained in a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Profile dataset containing one or more variables defined on a 1D
        ``height`` coordinate.
    stats : {'std', 'sem'} or None, default None
        If not None, shade each profile using either the standard deviation
        (``'std'``) or standard error of the mean (``'sem'``), when the
        corresponding statistic variable (``f"{stats}_{var}"``) is present
        in ``ds``.
    colours : bool, default False
        If False, plot each profile as a black line. If True, plot each
        profile as a colour-mapped ``LineCollection`` and attach a colourbar
        using parameters from ``plot_params``.
    mlyr_top : float or None, default None
        Height (km) of the melting-layer top. If not None, a horizontal
        dashed line is drawn at this height.
    mlyr_btm : float or None, default None
        Height (km) of the melting-layer bottom. If not None, a horizontal
        dashed line is drawn at this height.
    vars_bounds : dict or None, default None
        Optional per-variable bounds overriding the defaults used by
        ``plot_params``. Keys are variable names, values are ``(vmin, vmax)``.
    ucmap : dict or None, default None
        Optional per-variable colormap overrides passed to ``plot_params``.
        Keys are variable names, values are Matplotlib colormap names or
        objects.
    unorm : dict or None, default None
        Optional per-variable normalisation objects for colour mapping,
        passed to ``plot_params``.
    cb_ext : dict or None, default None
        Optional per-variable colourbar extension settings, passed to
        ``plot_params``.
    fig_title : str or None, default None
        Custom figure title. If None, a title is constructed from dataset
        metadata (profile type, elevation, radar name, and scan time).
    fig_size : tuple of float or None, default None
        Figure size in inches. If None, defaults to ``(14, 10)``.
    ylims : tuple of float or None, default None
        Vertical limits for the height axis as ``(min_height, max_height)``.
        If None, defaults to ``(0, 10)`` km.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : list of matplotlib.axes.Axes
        List of axes corresponding to each plotted variable.
    mappables : list of matplotlib.artist.Artist
        The line or ``LineCollection`` objects associated with each panel,
        useful for further customisation or colourbar manipulation.

    Notes
    -----
    * Statistical variables with prefixes ``std_``, ``min_``, ``max_``,
      ``sem_`` and vertical-resolution diagnostics (``VRES1``, ``VRES2``,
      ``VRES``) are excluded from the main plotting loop.
    * Variable-specific plotting parameters (colormap, normalisation,
      bounds, colourbar settings) are obtained from
      ``towerpy.datavis.rad_display.plot_params``.
    """

    # 1. Resolve variables to plot (exclude stats variables)
    exclude = {"VRES1", "VRES2", "VRES"}
    vars_to_plot = [v for v in ds.data_vars
                    if not (v.startswith(("std_", "min_", "max_", "sem_"))
                            or v in exclude)]
    # 2. INit Figure and axes
    if fig_size is None:
        fig_size = (14, 10)
    fig, axes = plt.subplots(1, len(vars_to_plot), figsize=fig_size,
                             sharey=True)
    if len(vars_to_plot) == 1:
        axes = [axes]
    # 3. Title from metadata
    meta = _safe_metadata(ds)
    dt_str = meta["dt_str"]
    rname = meta["rname"]
    # swp_mode = meta["swp_mode"]
    elev_str = meta["elev_str"]
    # ptitle = f"{rname.title()} [{swp_mode}] -- {dt_str}"
    ptitle = (f"{ds.attrs.get('profs_type', 'Profiles')} [{elev_str}] \n"
              f"{rname.title()} -- {dt_str}")
    fig.suptitle(fig_title or ptitle, fontsize=20)
    # 4. Loop over variables and plot
    height = ds["height"].values
    mappables = []   # <-- collect mappables here
    for ax, var in zip(axes, vars_to_plot):
        da = ds[var]
        x = da.values
        units = _safe_units(da)
        # 4a. Get plotting parameters
        params = plot_params(var, ds, vars_bounds=vars_bounds, unorm=unorm,
                             cb_ext=cb_ext, ucmap=ucmap)
        # 4b. Black-line profile
        if not colours:
            line, = ax.plot(x, height, "k")
            mappables.append(line)
        # 4c. Coloured profile using LineCollection
        else:
            points = np.array([x, height]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=params.cmap, norm=params.norm)
            lc.set_array(x)
            lc.set_linewidth(2)
            ax.add_collection(lc)
            mappables.append(lc)
            # Colorbar
            _add_colorbar(fig, ax, lc, params, fsizes={"fsz_cb": 10},
                          vunits=units, coord_sys="rect", rotangle=90)
        # 5. Stats shading (std or sem)
        if stats in ("std", "sem"):
            stat_var = f"{stats}_{var}"
            if stat_var in ds:
                s = ds[stat_var].values
                ax.fill_betweenx(height, x - s, x + s, alpha=0.3, color="gray")
        # 6. Melting layer
        if mlyr_top is not None:
            ax.axhline(mlyr_top, c="tab:blue", ls="--", lw=3, alpha=0.5)
        if mlyr_btm is not None:
            ax.axhline(mlyr_btm, c="tab:purple", ls="--", lw=3, alpha=0.5)
        # 7. Labels, limits, grid
        ax.grid(True)
        ax.set_facecolor("none")
        if colours:
            bnd = params.norm_boundaries
            if bnd is not None and len(bnd) > 1:
                ax.set_xlim(bnd.min(), bnd.max())
        # Y-limits
        if ylims:
            ax.set_ylim(ylims)
        else:
            ax.set_ylim(0, 10)
        # X-label
        short = da.attrs.get("short_name", var)
        label = f"{short} [{units}]" if units else short
        ax.set_xlabel(label, fontsize=14)
    hunits = ds["height"].attrs.get('units', 'km')
    axes[0].set_ylabel(f"Height [{hunits}]", fontsize=14)
    plt.tight_layout()    
    return fig, axes, mappables


def plot_rdqvp(rdqvp, dss=None, all_desc=True, stats=None, mlyr_top=None,
               mlyr_btm=None, vars_bounds=None, unorm=None, cb_ext=None,
               ucmap=None, ylims=None, fig_size=None, spec_range=None,
               fig_title=None, elev_cmap="Spectral"):
    """
    Plot a Range‑Defined Quasi‑Vertical Profile (RD‑QVP) and, optionally,
    the contributing QVPs.

    Parameters
    ----------
    rdqvp : xarray.Dataset
        RD‑QVP dataset containing one or more variables defined on a 1D
        ``height`` coordinate. May include:
        - ``qvp_interp`` : interpolated QVPs with dims ``(elevation, variable, height)``
        - ``scan_datetime`` : optional coordinate for scan start/end times
    dss : list of xarray.Dataset or None, default None
        Optional list of individual QVP scan datasets used to draw the
        geometry panel (range–height curves).
    all_desc : bool, default True
        If True, plot individual QVPs (from ``qvp_interp``) coloured by
        elevation angle. If False, only the RD‑QVP is plotted.
    stats : {'std', 'sem'} or None, default None
        If not None, shade each RD‑QVP profile using the corresponding
        statistic (``'std'`` or ``'sem'``) when the variable
        ``f"{stats}_{var}"`` exists in ``rdqvp``.
    mlyr_top : float or None, default None
        Height (km) of the melting‑layer top. If provided, a horizontal
        dashed line is drawn.
    mlyr_btm : float or None, default None
        Height (km) of the melting‑layer bottom. If provided, a horizontal
        dashed line is drawn.
    vars_bounds : dict or None, default None
        Optional per‑variable bounds overriding defaults from ``plot_params``.
        Keys are variable names; values are ``(vmin, vmax)``.
    unorm : dict or None, default None
        Optional per‑variable normalisation objects for colour mapping,
        passed to ``plot_params``.
    cb_ext : dict or None, default None
        Optional per‑variable colourbar extension settings.
    ucmap : dict or None, default None
        Optional per‑variable colormap overrides passed to ``plot_params``.
    ylims : tuple of float or None, default None
        Vertical limits for the height axis as ``(min_height, max_height)``.
        If None, defaults to ``(0, max(height))``.
    fig_size : tuple of float or None, default None
        Figure size in inches. If None, defaults to ``(14, 10)``.
    spec_range : float or None, default None
        If provided, draw a vertical line in the geometry panel marking the
        RD‑QVP range (km).
    fig_title : str or None, default None
        Custom figure title. If None, a title is constructed from dataset
        metadata (profile type, radar name, and scan time).
    elev_cmap : str, default 'Spectral'
        Colormap used to colour individual QVPs by elevation angle.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : list of matplotlib.axes.Axes
        Axes corresponding to each plotted variable (and optional geometry panel).
    mappables : list of matplotlib.artist.Artist
        The line objects associated with each RD‑QVP panel, useful for
        further customisation or colourbar manipulation.

    Notes
    -----
    * Variables with prefixes ``std_``, ``min_``, ``max_``, ``sem_`` and
      variables in ``{"VRES1", "VRES2", "VRES", "qvp_interp"}`` are excluded
      from the main plotting loop.
    * If ``qvp_interp`` is present and ``all_desc=True``, individual QVPs are
      plotted using a colormap indexed by elevation angle.
    * When ``dss`` is provided, an additional geometry panel is added showing
      range–height curves for each contributing scan.
    * Variable‑specific plotting parameters (colormap, normalisation, bounds,
      colourbar settings) are obtained from
      ``towerpy.datavis.rad_display.plot_params``.
    """

    # 1. Resolve variables to plot (exclude stats, VRES, qvp_interp)
    exclude = {"VRES1", "VRES2", "VRES", "qvp_interp"}
    vars_to_plot = [v for v in rdqvp.data_vars
                    if not (v.startswith(("std_", "min_", "max_", "sem_"))
                            or v in exclude)]
    if fig_size is None:
        fig_size = (14, 10)
    # Geometry panel if dss is provided
    n_main = len(vars_to_plot)
    n_panels = n_main + (1 if dss is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=fig_size, sharey=True)
    if n_panels == 1:
        axes = [axes]
    # 2. Title from metadata / scan datetimes
    meta = _safe_metadata(rdqvp)
    rname = meta["rname"]
    profs_type = rdqvp.attrs.get("profs_type", "Profiles")
    if "scan_datetime" in rdqvp.coords:
        dts = [d for d in rdqvp["scan_datetime"].values if d is not None]
        try:
            dt1, dt2 = min(dts), max(dts)
            dt_str = f"{dt1:%Y-%m-%d %H:%M:%S} – {dt2:%H:%M:%S}"
        except Exception:
            dt_str = meta["dt_str"]
    else:
        dt_str = meta["dt_str"]
    ptitle = fig_title or f"{profs_type} (RD‑QVP) \n{rname.title()} -- {dt_str}"
    fig.suptitle(ptitle, fontsize=20)

    height = rdqvp["height"].values
    hunits = rdqvp["height"].attrs.get("units", "km")

    # Colormap for elevations (for individual QVPs)
    qvp_interp = rdqvp.data_vars.get("qvp_interp", None)
    if qvp_interp is not None and all_desc:
        elev_angles = rdqvp.coords.get("elevation_angle", None)
        n_elev = qvp_interp.sizes.get("elevation", 0)
        cmap_elev = mpl.colormaps[elev_cmap](np.linspace(0, 1, n_elev))
    else:
        elev_angles = None
        n_elev = 0
        cmap_elev = None
    # 3. Loop over variables and plot RD‑QVP (+ optional QVPs)
    mappables = []
    for i, var in enumerate(vars_to_plot):
        ax = axes[i]
        da = rdqvp[var]
        x = da.values
        units = _safe_units(da)
        # Plotting parameters for RD‑QVP
        # params = plot_params(var, rdqvp, vars_bounds=vars_bounds,
        #                      unorm=unorm, cb_ext=cb_ext,
        #                      ucmap=ucmap)
        # 3a. Optional individual QVPs (coloured by elevation)
        if qvp_interp is not None and all_desc:
            # qvp_interp: (elevation, variable, height)
            for j in range(n_elev):
                qvp_j = qvp_interp.isel(elevation=j).sel(variable=var)
                xj = qvp_j.values
                col = cmap_elev[j]
                label = None
                if elev_angles is not None:
                    elev_j = float(elev_angles.values[j])
                    label = f"{elev_j:.1f}" + r"$^{\circ}$"
                ax.plot(xj, height, ls="--", color=col, alpha=0.7,
                        label=label if i == 0 else None)
        # 3b. RD‑QVP profile (black)
        line, = ax.plot(x, height, "k", lw=2, label="RD‑QVP")
        mappables.append(line)
        # 3c. Stats shading (std or sem)
        if stats in ("std", "sem"):
            stat_var = f"{stats}_{var}"
            if stat_var in rdqvp:
                s = rdqvp[stat_var].values
                ax.fill_betweenx(height, x - s, x + s, alpha=0.3, color="gray")
        # 3d. Melting layer
        if mlyr_top is not None:
            ax.axhline(mlyr_top, c="tab:blue", ls="--", lw=3, alpha=0.5)
        if mlyr_btm is not None:
            ax.axhline(mlyr_btm, c="tab:purple", ls="--", lw=3, alpha=0.5)
        # 3e. Axes styling
        ax.grid(True)
        ax.set_facecolor("none")
        # X-limits from vars_bounds or plot_params
        # if vars_bounds and var in vars_bounds:
        #     ax.set_xlim(vars_bounds[var])
        # else:
        #     bnd = params.norm_boundaries
        #     if bnd is not None and len(bnd) > 1:
        #         ax.set_xlim(bnd.min(), bnd.max())
        # Y-limits
        if ylims:
            ax.set_ylim(ylims)
        else:
            ax.set_ylim(0, np.nanmax(height))
        # Labels
        short = da.attrs.get("short_name", var)
        label = f"{short} [{units}]" if units else short
        ax.set_xlabel(label, fontsize=14)
    axes[0].set_ylabel(f"Height [{hunits}]", fontsize=14)
    # 4. Optional geometry panel
    if dss is not None:
        axg = axes[-1]
        # Build elevation colormap consistent with QVPs
        if elev_angles is not None:
            cmap_geom = cmap_elev
        else:
            # Fallback: colour by index
            cmap_geom = mpl.colormaps[elev_cmap](np.linspace(0, 1, len(dss)))
        for j, ds in enumerate(dss):
            rng_km = convert(ds["range"], "km")
            if "beamc_height" in ds:
                bh = ds["beamc_height"].mean(dim="azimuth")
            else:
                # Fallback: use height from RD‑QVP if available
                bh = rdqvp["height"]
            axg.plot(rng_km.values, bh.values, ls="--", color=cmap_geom[j])
        if spec_range is not None:
            axg.axvline(spec_range, c="k", lw=2, label=f"RD={spec_range:g} km")
        axg.set_xlabel("Range [km]", fontsize=14)
        axg.grid(True)
        axg.set_facecolor("none")
    plt.tight_layout()
    return fig, axes, mappables

