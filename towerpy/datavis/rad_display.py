"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from ..utils import radutilities as rut
from ..base import TowerpyError

# warnings.filterwarnings("ignore", category=UserWarning)

# tpycm_ref = mpl.colormaps['tpylsc_ref']
# tpycm_plv = mpl.colormaps['tpylsc_pvars']
# tpycm_rnr = mpl.colormaps['tpylsc_rainrt']
# tpycm_2slope = mpl.colormaps['tpylsc_2slope']
# tpycm_dv = mpl.colormaps['tpylsc_dbu_rd']
# tpycm_3c = mpl.colormaps['tpylc_yw_gy_bu']


def plot_ppi(rad_georef, rad_params, rad_vars, var2plot=None, proj='rect',
             vars_bounds=None, xlims=None, ylims=None, data_proj=None,
             ucmap=None, unorm=None, ring=None, range_rings=None,
             cpy_feats=None, fig_size=None):
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
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    data_proj : Cartopy Coordinate Reference System object, optional
        Cartopy projection used to plot the data in a map e.g.,
        ccrs.OSGB(approx=False).
    ucmap : colormap, optional
        User-defined colormap.
    unorm : matplotlib.colors normalisation object, optional
        User-defined normalisation method to map colormaps onto radar data.
        The default is None.
    ring : int or float, optional
        Plot a circle in the given distance, in km.
    range_rings : int, float, list or tuple, optional
        If int or float, plot circles at a fixed range, in km.
        If list or tuple, plot circles at the given ranges, in km.
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
    """
    lpv = {'ZH [dBZ]': [-10, 60, 15], 'ZDR [dB]': [-2, 6, 17],
           'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
           'rhoHV [-]': [0.3, .9, 1], 'V [m/s]': [-5, 5, 11],
           'gradV [dV/dh]': [-1, 0, 11], 'LDR [dB]': [-35, 0, 11],
           'Rainfall [mm/hr]': [0.1, 64, 11]}
    if vars_bounds is not None:
        lpv.update(vars_bounds)
    bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
    if vars_bounds is None:
        bnd['bRainfall [mm/hr]'] = np.array((0.01, 0.5, 1, 2, 4, 8, 12, 20,
                                             28, 36, 48, 64, 80, 100))

    dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value,
                                               mpl.colormaps['tpylsc_pvars'].N,
                                               extend='both')
             for key, value in bnd.items()}
    if 'bZH [dBZ]' in bnd.keys():
        dnorm['nZH [dBZ]'] = mcolors.BoundaryNorm(bnd['bZH [dBZ]'],
                                                  mpl.colormaps['tpylsc_ref'].N,
                                                  extend='both')
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(bnd['brhoHV [-]'],
                                                   mpl.colormaps['tpylsc_pvars'].N,
                                                   extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        bnrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/hr]'],
                                    mpl.colormaps['tpylsc_rainrt'].N,
                                    extend='max')
        dnorm['nRainfall [mm/hr]'] = bnrr
    if 'bZDR [dB]' in bnd.keys():
        dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(bnd['bZDR [dB]'],
                                                  mpl.colormaps['tpylsc_2slope'].N,
                                                  extend='both')
        # dnorm['nZDR [dB]'] = mcolors.TwoSlopeNorm(vmin=lpv['ZDR [dB]'][0],
        #                                           vcenter=0,
        #                                           vmax=lpv['ZDR [dB]'][1],
    if 'bKDP [deg/km]' in bnd.keys():
        dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(bnd['bKDP [deg/km]'],
                                                      mpl.colormaps['tpylsc_2slope'].N,
                                                      extend='both')
    if 'bV [m/s]' in bnd.keys():
        dnorm['nV [m/s]'] = mcolors.BoundaryNorm(bnd['bV [m/s]'],
                                                 mpl.colormaps['tpylsc_dbu_rd'].N,
                                                 extend='both')
    # dtdes0 = f"[{rad_params['site_name']}]"
    dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # txtboxc = (0, -.09)
    # fc, ec = 'w', 'k'
    if unorm is not None:
        dnorm.update(unorm)
    cbtks_fmt = 0
    if var2plot is None or var2plot == 'ZH [dBZ]':
        if 'ZH [dBZ]' in rad_vars.keys():
            cmaph, normp = mpl.colormaps['tpylsc_ref'], dnorm['nZH [dBZ]']
            var2plot = 'ZH [dBZ]'
        else:
            var2plot = list(rad_vars.keys())[0]
            cmaph = mpl.colormaps['tpylsc_pvars']
            normp = dnorm.get('n'+var2plot)
            if '[-]' in var2plot:
                cbtks_fmt = 2
                cmaph = mpl.colormaps['tpylsc_pvars']
                tcks = bnd['brhoHV [-]']
            if '[dB]' in var2plot:
                cmaph = mpl.colormaps['tpylsc_2slope']
                cbtks_fmt = 1
            if '[deg/km]' in var2plot:
                cmaph = mpl.colormaps['tpylsc_2slope']
                # cbtks_fmt = 1
            if '[m/s]' in var2plot:
                cmaph = mpl.colormaps['tpylsc_dbu_rd']
            if '[mm/hr]' in var2plot:
                cmaph = mpl.colormaps['tpylsc_rainrt']
                # tpycm.set_under(color='#D2ECFA', alpha=0)
                # mpl.colormaps['tpylsc_rainrt'].set_bad(color='#D2ECFA', alpha=0)
                cbtks_fmt = 1
    else:
        cmaph = mpl.colormaps['tpylsc_pvars']
        normp = dnorm.get('n'+var2plot)
        if '[-]' in var2plot:
            cbtks_fmt = 2
            tcks = bnd['brhoHV [-]']
        if '[dB]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_2slope']
            cbtks_fmt = 1
        if '[dBZ]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_ref']
            cbtks_fmt = 1
        if '[deg/km]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_2slope']
        if '[m/s]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_dbu_rd']
        if '[mm/hr]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_rainrt']
            # tpycm.set_under(color='#D2ECFA', alpha=0)
            # mpl.colormaps['tpylsc_rainrt'].set_bad(color='#D2ECFA', alpha=0)
            cbtks_fmt = 1
    if ucmap is not None:
        cmaph = ucmap

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

    if proj == 'polar':
        ptitle = dtdes1 + dtdes2
        if fig_size is None:
            fig_size = (6, 6.15)
        fig, ax1 = plt.subplots(figsize=fig_size,
                                subplot_kw=dict(projection='polar'))
        f1 = ax1.pcolormesh(rad_georef['theta'], rad_georef['rho'],
                            np.flipud(rad_vars[var2plot]), shading='auto',
                            cmap=cmaph, norm=normp)
        ax1.set_title(f'{ptitle} \n' + f'PPI {var2plot}', fontsize=14)
        ax1.grid(color='gray', linestyle=':')
        ax1.set_theta_zero_location('N')
        ax1.tick_params(axis='both', labelsize=10)
        ax1.set_yticklabels([])
        ax1.set_thetagrids(np.arange(0, 360, 90))
        ax1.axes.set_aspect('equal')
        if var2plot == 'rhoHV [-]':
            cb1 = plt.colorbar(f1, ax=ax1, aspect=8, shrink=0.65,
                               pad=.1, norm=normp, ticks=tcks,
                               format=f'%.{cbtks_fmt}f')
            cb1.ax.tick_params(direction='in', axis='both', labelsize=10)
        else:
            cb1 = plt.colorbar(f1, ax=ax1, aspect=8, shrink=0.65,
                               pad=.1, norm=normp)
            cb1.ax.tick_params(direction='in', axis='both', labelsize=10)
        cb1.ax.set_title(f'{var2plot}', fontsize=10)
        # ax1.annotate('| Created using Towerpy |', xy=txtboxc,
        #              fontsize=8, xycoords='axes fraction',
        #              va='center', ha='center',
        #              bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
        plt.tight_layout()
        plt.show()

    elif proj == 'rect' and cpy_features['status'] is False:
        ptitle = dtdes1 + dtdes2
        if fig_size is None:
            fig_size = (6, 6.75)
        fig, ax1 = plt.subplots(figsize=fig_size)
        f1 = ax1.pcolormesh(rad_georef['xgrid'], rad_georef['ygrid'],
                            rad_vars[var2plot], shading='auto',
                            cmap=cmaph,
                            norm=normp)
        if range_rings is not None:
            if isinstance(range_rings, (int, float)):
                nrings = np.arange(range_rings*1000,
                                   rad_georef['range [m]'][-1],
                                   range_rings*1000)
            elif isinstance(range_rings, (list, tuple)):
                nrings = np.array(range_rings) * 1000
            idx_rs = [rut.find_nearest(rad_georef['range [m]'], r)
                      for r in nrings]
            dmmy_rsx = np.array([rad_georef['xgrid'][:, i] for i in idx_rs])
            dmmy_rsy = np.array([rad_georef['ygrid'][:, i] for i in idx_rs])
            dmmy_rsz = np.array([np.ones(i.shape) for i in dmmy_rsx])
            ax1.scatter(dmmy_rsx, dmmy_rsy, dmmy_rsz, c='grey', ls='--',
                        alpha=3/4)
            ax1.axhline(0, c='grey', ls='--', alpha=3/4)
            ax1.axvline(0, c='grey', ls='--', alpha=3/4)
            ax1.grid(True)

        if ring is not None:
            idx_rr = rut.find_nearest(rad_georef['range [m]'],
                                      ring*1000)
            dmmy_rx = rad_georef['xgrid'][:, idx_rr]
            dmmy_ry = rad_georef['ygrid'][:, idx_rr]
            dmmy_rz = np.ones(dmmy_rx.shape)
            ax1.scatter(dmmy_rx, dmmy_ry, dmmy_rz, c='k', ls='--', alpha=3/4)
        ax1_divider = make_axes_locatable(ax1)
        cax1 = ax1_divider.append_axes('top', size="7%", pad="2%")
        if var2plot == 'rhoHV [-]':
            cb1 = fig.colorbar(f1, cax=cax1, orientation='horizontal',
                               ticks=tcks, format=f'%.{cbtks_fmt}f')
            cb1.ax.tick_params(direction='in', labelsize=10)
        else:
            cb1 = fig.colorbar(f1, cax=cax1, orientation='horizontal')
            cb1.ax.tick_params(direction='in', labelsize=12)
        # cb1 = fig.colorbar(f1, cax=cax1, orientation='horizontal')
        cb1.ax.set_title(f'{ptitle} \n' + f'PPI {var2plot}', fontsize=14)
        # cb1.ax.minorticks_on()
        # cb1.ax.xaxis.set_ticks(bnd['brhoHV [-]'], minor=True)
        cax1.xaxis.set_ticks_position('top')
        if xlims is not None:
            ax1.set_xlim(xlims)
        if ylims is not None:
            ax1.set_ylim(ylims)
        ax1.set_xlabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax1.set_ylabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax1.tick_params(direction='in', axis='both', labelsize=10)
        ax1.axes.set_aspect('equal')
        # ax1.annotate('| Created using Towerpy |', xy=txtboxc,
        #              fontsize=8, xycoords='axes fraction',
        #              va='center', ha='center',
        #              bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
        # ax1.grid(True)
        plt.tight_layout()
        plt.show()

    elif proj == 'rect' and cpy_features['status']:
        ptitle = dtdes1 + dtdes2
        proj = ccrs.PlateCarree()
        if fig_size is None:
            fig_size = (9, 6)
        if data_proj:
            proj2 = data_proj
        else:
            raise TowerpyError('User must specify the projected coordinate'
                               ' system of the radar data e.g.'
                               ' ccrs.OSGB(approx=False) or ccrs.UTM(zone=32)')
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        plt.subplots_adjust(left=0.05, right=0.99, top=0.981, bottom=0.019,
                            wspace=0, hspace=1
                            )
        ax1 = fig.add_subplot(projection=proj)
        if xlims and ylims:
            extx = xlims
            exty = ylims
            ax1.set_extent(extx+exty, crs=proj)
        if cpy_features['tiles']:
            if cpy_features['tiles_source'] is None or cpy_features['tiles_source'] == 'OSM':
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
                    imtiles = cimgt.GoogleTiles(style=cpy_features['tiles_style'])
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
        gl.xlabel_style = {'size': 11}
        gl.ylabel_style = {'size': 11}
        ax1.set_title(f'{ptitle} \n' + f'PPI {var2plot}', fontsize=14)
        # lon_formatter = LongitudeFormatter(number_format='.4f',
        #                                 degree_symbol='',
        #                                dateline_direction_label=True)
        # lat_formatter = LatitudeFormatter(number_format='.0f',
        #                                    degree_symbol=''
        #                                   )
        mappable = ax1.pcolormesh(rad_georef['xgrid_proj'],
                                  rad_georef['ygrid_proj'],
                                  rad_vars[var2plot], transform=proj2,
                                  shading='auto', cmap=cmaph, norm=normp,
                                  alpha=cpy_features['alpha_rad'])
        # ax1.xaxis.set_major_formatter(lon_formatter)
        # ax1.yaxis.set_major_formatter(lat_formatter)
        plotunits = [i[i.find('['):]
                     for i in rad_vars.keys() if var2plot == i][0]

        def make_colorbar(ax1, mappable, **kwargs):
            ax1_divider = make_axes_locatable(ax1)
            orientation = kwargs.pop('orientation', 'vertical')
            if orientation == 'vertical':
                loc = 'right'
            elif orientation == 'horizontal':
                loc = 'top'
            cax = ax1_divider.append_axes(loc, '7%', pad='15%',
                                          axes_class=plt.Axes)
            ax1.get_figure().colorbar(mappable, cax=cax,
                                      orientation=orientation,
                                      ticks=bnd.get('b'+var2plot),
                                      format=f'%.{cbtks_fmt}f',
                                      )
            cax.tick_params(direction='in', labelsize=12)
            cax.xaxis.set_ticks_position('top')
            cax.set_title(plotunits, fontsize=14)
        make_colorbar(ax1, mappable, orientation='vertical')
        plt.show()


def plot_setppi(rad_georef, rad_params, rad_vars, xlims=None, ylims=None,
                vars_bounds=None, mlyr=None, fig_size=None):
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
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
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
    mlyr : MeltingLayer Class, optional
        Plots an isotropic melting layer. The default is None.
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
    dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value,
                                               mpl.colormaps['tpylsc_ref'].N,
                                               extend='both')
             for key, value in bnd.items()}
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(bnd['brhoHV [-]'],
                                                   mpl.colormaps['tpylsc_pvars'].N,
                                                   extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        bnrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/hr]'],
                                    mpl.colormaps['tpylsc_rainrt'].N,
                                    extend='max')
        dnorm['nRainfall [mm/hr]'] = bnrr
    if 'bZDR [dB]' in bnd.keys():
        dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(bnd['bZDR [dB]'],
                                                  mpl.colormaps['tpylsc_2slope'].N,
                                                  extend='both')
    if 'bKDP [deg/km]' in bnd.keys():
        dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(bnd['bKDP [deg/km]'],
                                                      mpl.colormaps['tpylsc_2slope'].N,
                                                      extend='both')
    if 'bV [m/s]' in bnd.keys():
        dnorm['nV [m/s]'] = mcolors.BoundaryNorm(bnd['bV [m/s]'],
                                                 mpl.colormaps['tpylsc_dbu_rd'].N,
                                                 extend='both')
    # if unorm is not None:
    #     dnorm.update(unorm)
    # cbtks_fmt = 0
    # if ucmap is not None:
    #     cmaph = ucmap
    if mlyr is not None:
        idx_bhb = rut.find_nearest(rad_georef['beam_height [km]'],
                                   mlyr.ml_bottom)
        idx_bht = rut.find_nearest(rad_georef['beam_height [km]'],
                                   mlyr.ml_top)
        dmmyx_mlb = rad_georef['xgrid'][:, idx_bhb]
        dmmyy_mlb = rad_georef['ygrid'][:, idx_bhb]
        dmmyz_mlb = np.ones(dmmyx_mlb.shape)
        dmmyx_mlt = rad_georef['xgrid'][:, idx_bht]
        dmmyy_mlt = rad_georef['ygrid'][:, idx_bht]
        dmmyz_mlt = np.ones(dmmyx_mlt.shape)

    dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # fc, ec = 'w', 'k'

    nr = 1
    nc = int(len(rad_vars))
    if len(rad_vars) > 3:
        nr = 2
        nc = int(len(rad_vars)/2)
    if len(rad_vars) > 3 and len(rad_vars) % 2:
        nc = int((len(rad_vars)//2)+1)
    if fig_size is None:
        fig_size = (16, 9)
    f, ax = plt.subplots(nr, nc, sharex=True, sharey=True, figsize=fig_size)
    f.suptitle(f'{ptitle}', fontsize=16)

    for a, (key, var2plot) in zip(ax.flatten(), rad_vars.items()):
        if key in lpv:
            norm = dnorm.get('n'+key)
        else:
            b1 = np.linspace(np.nanmin(var2plot), np.nanmax(var2plot), 11)
            norm = mcolors.BoundaryNorm(b1, mpl.colormaps['tpylsc_pvars'].N,
                                        extend='both')
        if '[dBZ]' in key:
            cmap = mpl.colormaps['tpylsc_ref']
        elif '[dB]' in key or '[deg/km]' in key:
            cmap = mpl.colormaps['tpylsc_2slope']
        elif '[mm/hr]' in key:
            cmap = mpl.colormaps['tpylsc_rainrt']
        elif '[m/s]' in key:
            cmap = mpl.colormaps['tpylsc_dbu_rd']
        else:
            cmap = mpl.colormaps['tpylsc_pvars']
        f1 = a.pcolormesh(rad_georef['xgrid'], rad_georef['ygrid'], var2plot,
                          shading='auto', cmap=cmap, norm=norm)
        if mlyr is not None:
            a.scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
            a.scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
        if xlims is not None:
            a.set_xlim(xlims)
        if ylims is not None:
            a.set_ylim(ylims)
        # a.set_title(f'{dtdes}' "\n" f'{key}')
        a.set_title(f'{key}', fontsize=12)
        a.set_xlabel('Distance from the radar [km]', fontsize=12)
        a.set_ylabel('Distance from the radar [km]', fontsize=12)
        a.grid(True)
        a.axes.set_aspect('equal')
        a.tick_params(axis='both', which='major', labelsize=10)
        if key.startswith('rhoHV'):
            plt.colorbar(f1, ax=a, ticks=bnd.get('b'+key),
                         format=f'%.{2}f',
                         )
        else:
            plt.colorbar(f1, ax=a)
    if len(rad_vars) == 5:
        f.delaxes(ax[1, 2])
    if len(rad_vars) == 7:
        f.delaxes(ax[1, 3])
    # txtboxc = (1.025, -.10)
    # txtboxc = (-3., -.10)
    # a.annotate('| Created using Towerpy |', xy=txtboxc, fontsize=8,
    #            xycoords='axes fraction', va='center', ha='center',
    #            bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    # plt.tight_layout()
    plt.show()


def plot_mgrid(rscans_georef, rscans_params, rscans_vars, var2plot=None,
               proj='rect', vars_bounds=None, xlims=None, ylims=None,
               data_proj=None, ucmap=None, unorm=None, cpy_feats=None,
               ncols=None, nrows=None, fig_size=None):
    """
    Graph multiple PPI scans into a grid.

    Parameters
    ----------
    rscans_georef : list
        List of eoreferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others, corresponding to each PPI scan.
    rscans_params : list
        List of radar technical details corresponding to each PPI scan.
    rscans_vars : list
        List of Dicts containing radar variables to plot corresponding to each
        PPI scan.
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
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    data_proj : Cartopy Coordinate Reference System object, optional
        Cartopy projection used to plot the data in a map e.g.,
        ccrs.OSGB(approx=False).
    ucmap : colormap, optional
        User-defined colormap.
    unorm : matplotlib.colors normalisation object, optional
        User-defined normalisation method to map colormaps onto radar data.
        The default is None.
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
    ncols : int, optional
        Set the number of columns used to build the grid. The default is None.
    nrows : int, optional
        Set the number of rows used to build the grid. The default is None.
    """
    from mpl_toolkits.axes_grid1 import ImageGrid
    from cartopy.mpl.geoaxes import GeoAxes

    lpv = {'ZH [dBZ]': [-10, 60, 15], 'ZDR [dB]': [-2, 6, 17],
           'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
           'rhoHV [-]': [0.3, .9, 1], 'V [m/s]': [-5, 5, 11],
           'gradV [dV/dh]': [-1, 0, 11], 'LDR [dB]': [-35, 0, 11],
           'Rainfall [mm/hr]': [0.1, 64, 11]}
    if vars_bounds is not None:
        lpv.update(vars_bounds)
    bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
    if vars_bounds is None:
        bnd['bRainfall [mm/hr]'] = np.array((0.01, 0.5, 1, 2, 4, 8, 12, 20,
                                             28, 36, 48, 64, 80, 100))

    dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value,
                                               mpl.colormaps['tpylsc_pvars'].N,
                                               extend='both')
             for key, value in bnd.items()}
    if 'bZH [dBZ]' in bnd.keys():
        dnorm['nZH [dBZ]'] = mcolors.BoundaryNorm(bnd['bZH [dBZ]'],
                                                  mpl.colormaps['tpylsc_ref'].N,
                                                  extend='both')
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(bnd['brhoHV [-]'],
                                                   mpl.colormaps['tpylsc_pvars'].N,
                                                   extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        bnrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/hr]'],
                                    mpl.colormaps['tpylsc_rainrt'].N,
                                    extend='max')
        dnorm['nRainfall [mm/hr]'] = bnrr
    if 'bZDR [dB]' in bnd.keys():
        dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(bnd['bZDR [dB]'],
                                                  mpl.colormaps['tpylsc_2slope'].N,
                                                  extend='both')
        # dnorm['nZDR [dB]'] = mcolors.TwoSlopeNorm(vmin=lpv['ZDR [dB]'][0],
        #                                           vcenter=0,
        #                                           vmax=lpv['ZDR [dB]'][1],
    if 'bKDP [deg/km]' in bnd.keys():
        dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(bnd['bKDP [deg/km]'],
                                                      mpl.colormaps['tpylsc_2slope'].N,
                                                      # tpycm_2slope.N,
                                                      extend='both')
    if 'bV [m/s]' in bnd.keys():
        dnorm['nV [m/s]'] = mcolors.BoundaryNorm(bnd['bV [m/s]'],
                                                 mpl.colormaps['tpylsc_dbu_rd'].N,
                                                 extend='both')

    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # txtboxc = (0, -.09)
    # fc, ec = 'w', 'k'
    if unorm is not None:
        dnorm.update(unorm)
    cbtks_fmt = 0
    if var2plot is None or var2plot == 'ZH [dBZ]':
        if all('ZH [dBZ]' in i.keys() for i in rscans_vars):
            cmaph, normp = mpl.colormaps['tpylsc_ref'], dnorm['nZH [dBZ]']
            var2plot = 'ZH [dBZ]'
        else:
            # var2plot = list(rad_vars.keys())[0]
            # var2plot = [sorted(list(i.keys()))[0] for i in rscans_vars]
            dskeys = [k for i in rscans_vars for k in i.keys()]
            var2plot = list(set([x for x in dskeys
                                 if dskeys.count(x) >= len(rscans_vars)]))[0]
            cmaph = mpl.colormaps['tpylsc_pvars']
            normp = dnorm.get('n'+var2plot)
            if 'rho' in var2plot:
                cbtks_fmt = 2
                cmaph = mpl.colormaps['tpylsc_pvars']
                tcks = bnd['brhoHV [-]']
            if 'ZDR' in var2plot:
                cmaph = mpl.colormaps['tpylsc_2slope']
                cbtks_fmt = 1
            if 'KDP' in var2plot:
                cmaph = mpl.colormaps['tpylsc_2slope']
                # cbtks_fmt = 1
            if var2plot == 'V [m/s]':
                cmaph = mpl.colormaps['tpylsc_dbu_rd']
            if 'Rainfall' in var2plot:
                cmaph = mpl.colormaps['tpylsc_rainrt']
                # tpycm.set_under(color='#D2ECFA', alpha=0)
                # tpycm_rnr.set_bad(color='#D2ECFA', alpha=0)
                cbtks_fmt = 1
    else:
        cmaph = mpl.colormaps['tpylsc_pvars']
        normp = dnorm.get('n'+var2plot)
        if 'rho' in var2plot:
            cbtks_fmt = 2
            tcks = bnd['brhoHV [-]']
        if 'ZDR' in var2plot:
            cmaph = mpl.colormaps['tpylsc_2slope']
            cbtks_fmt = 1
        if 'KDP' in var2plot:
            cmaph = mpl.colormaps['tpylsc_2slope']
        if var2plot == 'V [m/s]':
            cmaph = mpl.colormaps['tpylsc_dbu_rd']
        if 'Rainfall' in var2plot:
            cmaph = mpl.colormaps['tpylsc_rainrt']
            # tpycm.set_under(color='#D2ECFA', alpha=0)
            # mpl.colormaps['tpylsc_rainrt'].set_bad(color='#D2ECFA', alpha=0)
            cbtks_fmt = 1
    if ucmap is not None:
        cmaph = ucmap
    # plotunits = var2plot[var2plot .find('['):]
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

    # if proj == 'polar':

    # grvars = [i[var2plot] for i in rscans_vars]

    if var2plot.endswith('[dBZ]'):
        cmap = mpl.colormaps['tpylsc_ref']
    elif var2plot.endswith('[dB]') or var2plot.endswith('[deg/km]'):
        cmap = mpl.colormaps['tpylsc_2slope']
    elif var2plot.endswith('[mm/hr]'):
        cmap = mpl.colormaps['tpylsc_rainrt']
    elif var2plot.endswith('[m/s]'):
        cmap = mpl.colormaps['tpylsc_dbu_rd']
    else:
        cmap = mpl.colormaps['tpylsc_pvars']
    pttl = [f"{p['elev_ang [deg]']:{2}.{3}} Deg. "
            + f"{p['datetime']:%Y-%m-%d %H:%M:%S}"
            for p in rscans_params]
    if proj == 'rect' and cpy_features['status'] is False:
        if fig_size is None:
            fig_size = (15, 5)
        fig = plt.figure(figsize=fig_size)
        grgeor = [[i['xgrid'], i['ygrid']] for i in rscans_georef]
        if nrows is None and ncols is None:
            if len(rscans_vars) <= 3:
                nrw = 1
                ncl = len(rscans_vars)
            else:
                nrw = int(np.ceil(len(rscans_vars)/2))
                ncl = 3
        elif nrows is not None and ncols is None:
            if len(rscans_vars) <= 3:
                nrw = nrows
                ncl = len(rscans_vars)
            else:
                nrw = nrows
                ncl = 3
        elif ncols is not None and nrows is None:
            if len(rscans_vars) <= 3:
                nrw = 1
                ncl = ncols
            else:
                nrw = int(np.ceil(len(rscans_vars)/2))
                ncl = ncols
        else:
            nrw = nrows
            ncl = ncols
        grid2 = ImageGrid(fig, 111, nrows_ncols=(nrw, ncl),
                          axes_pad=0.05, label_mode="L", share_all=True,
                          cbar_location="right", cbar_mode="single",
                          cbar_size="10%", cbar_pad=0.25)
        for ax, z, g, pr, pt in zip(grid2, [i[var2plot] for i in rscans_vars],
                                    grgeor, rscans_params, pttl):
            f1 = ax.pcolormesh(g[0], g[1], z, shading='auto', cmap=cmap,
                               norm=normp)
            ax.set_title(f"{pt} \n {pr['site_name']} - PPI {var2plot}",
                         fontsize=12)
            ax.set_xlabel('Distance from the radar [km]', fontsize=12)
            ax.set_ylabel('Distance from the radar [km]', fontsize=12)
            ax.grid(True)
            ax.axes.set_aspect('equal')
            ax.tick_params(axis='both', which='major', labelsize=12)
        # With cbar_mode="single", cax attribute of all axes are identical.

        if var2plot == 'rhoHV [-]':
            ax.cax.colorbar(f1, ticks=tcks, format=f'%.{cbtks_fmt}f')
        else:
            ax.cax.colorbar(f1)
        ax.cax.tick_params(direction='in', which='both', labelsize=12)
        ax.cax.toggle_label(True)
        ax.cax.set_title(var2plot[var2plot .find('['):], fontsize=12)
        # for ax, im_title in zip(grid2, ["(a)", "(b)", "(c)"]):
        #     t = add_inner_title(ax, im_title, loc='upper left')
        #     t.patch.set_ec("none")
        #     t.patch.set_alpha(0.5)
        if len(rscans_vars) >= 3 and len(rscans_vars) % 2 == 0 and ncols is None and ncols is None:
            grid2[-1].remove()
        plt.tight_layout()
        # plt.show()
    elif proj == 'rect' and cpy_features['status']:
        if fig_size is None:
            fig_size = (16, 6)
        fig = plt.figure(figsize=fig_size, constrained_layout=True)
        projection = ccrs.PlateCarree()
        axes_class = (GeoAxes, dict(map_projection=projection))
        grgeor = [[i['xgrid_proj'], i['ygrid_proj']] for i in rscans_georef]
        if nrows is None and ncols is None:
            if len(rscans_vars) <= 2:
                nrw = 1
                ncl = len(rscans_vars)
            else:
                nrw = int(np.ceil(len(rscans_vars)/2))
                ncl = 2
        elif nrows is not None and ncols is None:
            if len(rscans_vars) <= 2:
                nrw = nrows
                ncl = len(rscans_vars)
            else:
                nrw = nrows
                ncl = 2
        elif ncols is not None and nrows is None:
            if len(rscans_vars) <= 2:
                nrw = 1
                ncl = ncols
            else:
                nrw = int(np.ceil(len(rscans_vars)/2))
                ncl = ncols
        else:
            nrw = nrows
            ncl = ncols
        grid2 = ImageGrid(fig, 111, nrows_ncols=(nrw, ncl),
                          axes_pad=1.5, label_mode="keep",
                          cbar_location="right", cbar_mode="single",
                          cbar_size="9%", cbar_pad=0.75,
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
                        imtiles = cimgt.GoogleTiles(style=cpy_features['tiles_style'])
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
            print('\N{COPYRIGHT SIGN}'
                  + f'{data_source}; license: {data_license}')
            if cpy_features['tiles_source'] == 'Stamen':
                print('\N{COPYRIGHT SIGN}' + 'Map tiles by Stamen Design, '
                      + 'under CC BY 3.0. Data by OpenStreetMap, under ODbL.')
            gl = ax1.gridlines(draw_labels=True, dms=False,
                               x_inline=False, y_inline=False)
            gl.xlabel_style = {'size': 11}
            gl.ylabel_style = {'size': 11}
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
                                      shading='auto', cmap=cmaph, norm=normp,
                                      alpha=cpy_features['alpha_rad'])
            grid2.cbar_axes[0].colorbar(mappable)
            ax1.cax.set_title(var2plot[var2plot .find('['):], fontsize=12)
            ax1.axes.set_aspect('equal')
            plt.show()
        if len(rscans_vars) > 2 and len(rscans_vars) % 2 != 0 and ncols is None and ncols is None:
            grid2[-1].remove()


def plot_cone_coverage(rad_georef, rad_params, rad_vars, var2plot=None,
                       vars_bounds=None, xlims=None, ylims=None, zlims=[0, 8],
                       limh=8, ucmap=None, unorm=None, fig_size=None):
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
        to be mapped. The default are:
            {'ZH [dBZ]': [-10, 60, 15],
             'ZDR [dB]': [-2, 6, 17],
             'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
             'rhoHV [-]': [0.3, .9, 1],
             'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
             'LDR [dB]': [-35, 0, 11],
             'Rainfall [mm/hr]': [0.1, 64, 11]}
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    zlims : 2-element tuple or list, optional
        Set the z-axis view limits [min, max]. The default is None.
    limh : int or float, optional
        Set a height limit to the plot. The default is None.
    ucmap : colormap, optional
        User-defined colormap.
    unorm : matplotlib.colors normalisation object, optional
        User-defined normalisation method to map colormaps onto radar data.
        The default is None.
    """
    from matplotlib.colors import LightSource

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
    dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value, mpl.colormaps['tpylsc_ref'].N,
                                               extend='both')
             for key, value in bnd.items()}
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(bnd['brhoHV [-]'],
                                                   mpl.colormaps['tpylsc_pvars'].N,
                                                   extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        bnrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/hr]'],
                                    mpl.colormaps['tpylsc_rainrt'].N,
                                    extend='max')
        dnorm['nRainfall [mm/hr]'] = bnrr
    if 'bZDR [dB]' in bnd.keys():
        dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(bnd['bZDR [dB]'],
                                                  mpl.colormaps['tpylsc_2slope'].N,
                                                  extend='both')
    if 'bKDP [deg/km]' in bnd.keys():
        dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(bnd['bKDP [deg/km]'],
                                                      mpl.colormaps['tpylsc_2slope'].N,
                                                      extend='both')
    if 'bV [m/s]' in bnd.keys():
        dnorm['nV [m/s]'] = mcolors.BoundaryNorm(bnd['bV [m/s]'],
                                                 mpl.colormaps['tpylsc_dbu_rd'].N,
                                                 extend='both')
    # dtdes0 = f"[{rad_params['site_name']}]"
    dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2
    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # txtboxc = (0, -.09)
    # fc, ec = 'w', 'k'
    if unorm is not None:
        dnorm.update(unorm)
    cbtks_fmt = 0
    if var2plot is None:
        if 'ZH [dBZ]' in rad_vars.keys():
            cmaph, normp = mpl.colormaps['tpylsc_ref'], dnorm['nZH [dBZ]']
            var2plot = 'ZH [dBZ]'
        else:
            var2plot = list(rad_vars.keys())[0]
            cmaph = mpl.colormaps['tpylsc_pvars']
            normp = dnorm.get('n'+var2plot)
            if '[-]' in var2plot:
                cbtks_fmt = 2
                cmaph = mpl.colormaps['tpylsc_pvars']
            if '[dB]' in var2plot:
                cmaph = mpl.colormaps['tpylsc_2slope']
                cbtks_fmt = 1
            if '[deg/km]' in var2plot:
                cmaph = mpl.colormaps['tpylsc_2slope']
                # cbtks_fmt = 1
            if '[m/s]' in var2plot:
                cmaph = mpl.colormaps['tpylsc_dbu_rd']
            if '[mm/hr]' in var2plot:
                cmaph = mpl.colormaps['tpylsc_rainrt']
                # tpycm.set_under(color='#D2ECFA', alpha=0)
                # mpl.colormaps['tpylsc_rainrt'].set_bad(color='#D2ECFA', alpha=0)
                cbtks_fmt = 1
    else:
        cmaph = mpl.colormaps['tpylsc_pvars']
        normp = dnorm.get('n'+var2plot)
        if '[-]' in var2plot:
            cbtks_fmt = 2
        if '[dB]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_2slope']
            cbtks_fmt = 1
        if '[deg/km]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_2slope']
        if '[m/s]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_dbu_rd']
        if '[mm/hr]' in var2plot:
            cmaph = mpl.colormaps['tpylsc_rainrt']
            # tpycm.set_under(color='#D2ECFA', alpha=0)
            # mpl.colormaps['tpylsc_rainrt'].set_bad(color='#D2ECFA', alpha=0)
            cbtks_fmt = 1
    tcks = bnd.get('b'+var2plot)
    if ucmap is not None:
        cmaph = ucmap
        tcks = bnd.get('b'+var2plot)

    limidx = [rut.find_nearest(row, limh)
              for row in rad_georef['beam_height [km]']]

    m = np.ma.masked_invalid(rad_vars[var2plot]).mask
    for n, rows in enumerate(m):
        rows[limidx[n]:] = 1
    R = rad_vars[var2plot]

    X, Y = rad_georef['xgrid'], rad_georef['ygrid']
    Z = np.resize(rad_georef['beam_height [km]'], R.shape)

    X = np.ma.array(X, mask=m)
    Y = np.ma.array(Y, mask=m)
    Z = np.ma.array(Z, mask=m)
    R = np.ma.array(R, mask=m)

    ls = LightSource(0, 0)

    rgb = ls.shade(R, cmap=cmaph, norm=normp, vert_exag=0.1, blend_mode='soft')
    if fig_size is None:
        fig_size = (11, 9)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=fig_size)

    # Plot the surface.
    ax.plot_surface(X, Y, Z, cmap=cmaph, norm=normp, facecolors=rgb,
                    linewidth=0, antialiased=True, shade=False,
                    rstride=1, cstride=1)

    ax.contourf(X, Y, R, zdir='z', offset=0, levels=tcks, cmap=cmaph,
                norm=normp)
    # Customize the axis.
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_zlim(zlims)
    ax.view_init(elev=10,)
    ax.tick_params(axis='both', labelsize=15)

    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    plt.colorbar(mcm.ScalarMappable(norm=normp, cmap=cmaph), shrink=0.4,
                 aspect=5).ax.tick_params(labelsize=14)
    ax.set_title(f'{ptitle} \n' + f'PPI {var2plot}', fontsize=14)
    ax.set_xlabel('Distance from the radar [km]', fontsize=20, labelpad=15)
    ax.set_ylabel('Distance from the radar [km]', fontsize=20, labelpad=15)
    ax.set_zlabel('Height [km]', fontsize=20, labelpad=15)

    plt.show()


def plot_snr(rad_georef, rad_params, snr_data, proj='rect', fig_size=None):
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
        Coordinates projection (polar or rectangular). The default is 'rect'.
    """
    dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2
    if fig_size is None:
        fig_size = (10.5, 6.15)
    if proj == 'polar':
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=fig_size,
                                       subplot_kw=dict(projection='polar'))

        f2 = ax2.pcolormesh(rad_georef['theta'], rad_georef['rho'],
                            np.flipud(snr_data['snr [dB]']), shading='auto',
                            cmap='tpylsc_ref')
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

        ax3.set_title('Signal (SNR>minSNR)', fontsize=14, y=-0.15)
        ax3.pcolormesh(rad_georef['theta'], rad_georef['rho'],
                       np.flipud(snr_data['snrclass']), shading='auto',
                       cmap=mpl.colormaps['tpylc_yw_gy_bu'])
        # ax3.axes.set_aspect('equal')
        ax3.grid(color='w', linestyle=':')
        ax3.set_theta_zero_location('N')
        ax3.set_thetagrids(np.arange(0, 360, 90))
        # ax3.set_yticklabels([])
        mpl.colormaps['tpylc_yw_gy_bu'].set_bad(color='#505050')
        plt.show()

    elif proj == 'rect':
        fig, (ax2, ax3) = plt.subplots(1, 2, figsize=fig_size,
                                       sharex=True, sharey=True)
        f2 = ax2.pcolormesh(rad_georef['xgrid'], rad_georef['ygrid'],
                            snr_data['snr [dB]'], shading='auto',
                            cmap='tpylsc_ref')
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

        mpl.colormaps['tpylc_yw_gy_bu'].set_bad(color='#505050')
        # ax3.set_title('Signal (SNR>minSNR)')
        ax3.set_title(f'{ptitle} \n' + 'SNR>minSNR \n' +
                      '[Signal = Blue; Noise = Gray]', fontsize=14)
        ax3.pcolormesh(rad_georef['xgrid'], rad_georef['ygrid'],
                       snr_data['snrclass'], shading='auto',
                       cmap=mpl.colormaps['tpylc_yw_gy_bu'])
        ax3_divider = make_axes_locatable(ax3)
        cax3 = ax3_divider.append_axes("top", size="7%", pad="2%")
        ax3.set_xlabel('Distance from the radar [km]', fontsize=12,
                       labelpad=10)
        ax3.tick_params(axis='both', which='major', labelsize=10)
        ax3.axes.set_aspect('equal')
        cax3.remove()
        plt.tight_layout()
        plt.show()


def plot_nmeclassif(rad_georef, rad_params, nme_classif, clutter_map=None,
                    xlims=None, ylims=None, fig_size=None):
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
    dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2

    # txtboxs = 'round, rounding_size=0.5, pad=0.5'
    # fc, ec = 'w', 'k'
    # =========================================================================
    #   Plot the Clutter classification
    # =========================================================================
    if fig_size is None:
        fig_size = (6, 6.15)
    fig, axs = plt.subplots(figsize=fig_size)
    ax = axs
    ax.set_title(f'{ptitle} \n' + 'Clutter classification \n' +
                 '[Precipitation = Blue; Clutter = Yellow; Noise = Gray]')
    ax.pcolormesh(rad_georef['xgrid'], rad_georef['ygrid'],
                  nme_classif, shading='auto',
                  cmap=mpl.colormaps['tpylc_yw_gy_bu'])
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(color='gray', linestyle=':')
    ax.set_xlabel('Distance from the radar [km]', labelpad=10)
    ax.set_ylabel('Distance from the radar [km]', labelpad=10)
    ax.axes.set_aspect('equal')
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    # txtboxc = (0, -.09)
    # ax.annotate('| Created using Towerpy |', xy=txtboxc, fontsize=8,
    #             xycoords='axes fraction', va='center', ha='center',
    #             bbox=dict(boxstyle=txtboxs, fc=fc, ec=ec))
    plt.tight_layout()

    # =========================================================================
    #   Plot the Clutter Map
    # =========================================================================
    if clutter_map is not None:
        norm = mcolors.BoundaryNorm(boundaries=np.linspace(0, 100, 11),
                                    ncolors=256)
        if fig_size is None:
            fig_size = (6, 6.5)
        fig, axs = plt.subplots(figsize=fig_size)
        ax = axs
        f1 = ax.pcolormesh(rad_georef['xgrid'], rad_georef['ygrid'],
                           clutter_map*100, shading='auto',
                           cmap='tpylsc_grad_bupkyw', norm=norm)
        ax1_divider = make_axes_locatable(ax)
        cax1 = ax1_divider.append_axes('top', size="7%", pad="2%")
        cb1 = fig.colorbar(f1, cax=cax1,
                           orientation='horizontal',
                           # ticks=tcks
                           )
        cb1.ax.tick_params(direction='in', labelsize=10)
        # cb1.ax.set_xticklabels(cb1.ax.get_xticklabels(), rotation=90)
        cb1.ax.set_title('Clutter probability  (%)')
        cax1.xaxis.set_ticks_position('top')
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(color='gray', linestyle=':')
        ax.set_xlabel('Distance from the radar [km]', labelpad=10)
        ax.set_ylabel('Distance from the radar [km]', labelpad=10)
        ax.axes.set_aspect('equal')
        if xlims is not None:
            ax.set_xlim(xlims)
        if ylims is not None:
            ax.set_ylim(ylims)
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
             'Rainfall [mm/hr]': [0.1, 64, 11]}
    mlyr : MeltingLayer Class, optional
        Plots an isotropic melting layer. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
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

    dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value,
                                               mpl.colormaps['tpylsc_ref'].N,
                                               extend='both')
             for key, value in bnd.items()}
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(bnd['brhoHV [-]'],
                                                   mpl.colormaps['tpylsc_pvars'].N,
                                                   extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        bnrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/hr]'],
                                    mpl.colormaps['tpylsc_rainrt'].N,
                                    extend='max')
        dnorm['nRainfall [mm/hr]'] = bnrr
    if 'bZDR [dB]' in bnd.keys():
        dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(bnd['bZDR [dB]'],
                                                  mpl.colormaps['tpylsc_2slope'].N,
                                                  extend='both')
        # dnorm['nZDR [dB]'] = mcolors.TwoSlopeNorm(vmin=lpv['ZDR [dB]'][0],
        #                                           vcenter=0,
        #                                           vmax=lpv['ZDR [dB]'][1],
    if 'bKDP [deg/km]' in bnd.keys():
        dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(bnd['bKDP [deg/km]'],
                                                      mpl.colormaps['tpylsc_2slope'].N,
                                                      extend='both')
    if 'bV [m/s]' in bnd.keys():
        dnorm['nV [m/s]'] = mcolors.BoundaryNorm(bnd['bV [m/s]'],
                                                 mpl.colormaps['tpylsc_dbu_rd'].N,
                                                 extend='both')
    if mlyr is not None:
        idx_bhb = rut.find_nearest(rad_georef['beam_height [km]'],
                                   mlyr.ml_bottom)
        idx_bht = rut.find_nearest(rad_georef['beam_height [km]'],
                                   mlyr.ml_top)
        dmmyx_mlb = rad_georef['xgrid'][:, idx_bhb]
        dmmyy_mlb = rad_georef['ygrid'][:, idx_bhb]
        dmmyz_mlb = np.ones(dmmyx_mlb.shape)
        dmmyx_mlt = rad_georef['xgrid'][:, idx_bht]
        dmmyy_mlt = rad_georef['ygrid'][:, idx_bht]
        dmmyz_mlt = np.ones(dmmyx_mlt.shape)

    dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
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
            cmap = mpl.colormaps['tpylsc_ref']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx['A'].pcolormesh(rad_georef['xgrid'],
                                           rad_georef['ygrid'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['A'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx['A'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx['A'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
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
            cmap = mpl.colormaps['tpylsc_ref']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx['B'].pcolormesh(rad_georef['xgrid'],
                                           rad_georef['ygrid'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['B'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx['B'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx['B'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
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
            cmap = mpl.colormaps['tpylsc_pvars']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx['C'].pcolormesh(rad_georef['xgrid'],
                                           rad_georef['ygrid'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['C'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['C'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx['C'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
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
            cmap = mpl.colormaps['tpylsc_pvars']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx2['A'].pcolormesh(rad_georef['xgrid'],
                                            rad_georef['ygrid'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['A'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx2['A'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx2['A'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
    plt.colorbar(fzhna, ax=ax_idx2['A']).ax.tick_params(labelsize=10)
    ax_idx2['A'].grid(True)
    ax_idx2['A'].axes.set_aspect('equal')
    ax_idx2['A'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if key == 'PhiDP [deg]':
            cmap = mpl.colormaps['tpylsc_pvars']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx2['B'].pcolormesh(rad_georef['xgrid'],
                                            rad_georef['ygrid'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['B'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx2['B'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx2['B'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
    plt.colorbar(fzhna, ax=ax_idx2['B']).ax.tick_params(labelsize=10)
    ax_idx2['B'].grid(True)
    ax_idx2['B'].axes.set_aspect('equal')
    ax_idx2['B'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if key == 'PhiDP* [deg]':
            cmap = mpl.colormaps['tpylsc_pvars']
            norm = dnorm.get('n'+key.replace('*', ''))
            fzhna = ax_idx2['C'].pcolormesh(rad_georef['xgrid'],
                                            rad_georef['ygrid'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['C'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx2['C'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx2['C'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
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
            cmap = 'tpylsc_grad_fiery'
            norm = dnorm.get('n'+key)
            fzhna = ax_idx3.pcolormesh(rad_georef['xgrid'],
                                       rad_georef['ygrid'], value,
                                       shading='auto', cmap=cmap,
                                       norm=norm)
            ax_idx3.set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx3.scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx3.scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                        c='tab:orange')
    plt.colorbar(fzhna, ax=ax_idx3).ax.tick_params(labelsize=10)
    ax_idx3.grid(True)
    ax_idx3.axes.set_aspect('equal')
    ax_idx3.tick_params(axis='both', labelsize=10)
    plt.tight_layout()

    fig_mos4, ax_idx4 = plt.subplots(figsize=fig_size2)
    for key, value in rad_vars_attcorr.items():
        if 'PIA' in key:
            cmap = 'tpylsc_grad_fiery'
            norm = dnorm.get('n'+key)
            fzhna = ax_idx4.pcolormesh(rad_georef['xgrid'],
                                       rad_georef['ygrid'], value,
                                       shading='auto', cmap=cmap,
                                       norm=norm)
            ax_idx4.set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx4.scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx4.scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                        c='tab:orange')
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
             'Rainfall [mm/hr]': [0.1, 64, 11]}
    mlyr : MeltingLayer Class, optional
        Plots an isotropic melting layer. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
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

    dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value,
                                               mpl.colormaps['tpylsc_ref'].N,
                                               extend='both')
             for key, value in bnd.items()}
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(bnd['brhoHV [-]'],
                                                   mpl.colormaps['tpylsc_pvars'].N,
                                                   extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        bnrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/hr]'],
                                    mpl.colormaps['tpylsc_rainrt'].N,
                                    extend='max')
        dnorm['nRainfall [mm/hr]'] = bnrr
    if 'bZDR [dB]' in bnd.keys():
        dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(bnd['bZDR [dB]'],
                                                  mpl.colormaps['tpylsc_2slope'].N,
                                                  extend='both')
        # dnorm['nZDR [dB]'] = mcolors.TwoSlopeNorm(vmin=lpv['ZDR [dB]'][0],
        #                                           vcenter=0,
        #                                           vmax=lpv['ZDR [dB]'][1],
    if 'bKDP [deg/km]' in bnd.keys():
        dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(bnd['bKDP [deg/km]'],
                                                      mpl.colormaps['tpylsc_2slope'].N,
                                                      extend='both')
    if 'bV [m/s]' in bnd.keys():
        dnorm['nV [m/s]'] = mcolors.BoundaryNorm(bnd['bV [m/s]'],
                                                 mpl.colormaps['tpylsc_dbu_rd'].N,
                                                 extend='both')
    if mlyr is not None:
        idx_bhb = rut.find_nearest(rad_georef['beam_height [km]'],
                                   mlyr.ml_bottom)
        idx_bht = rut.find_nearest(rad_georef['beam_height [km]'],
                                   mlyr.ml_top)
        dmmyx_mlb = rad_georef['xgrid'][:, idx_bhb]
        dmmyy_mlb = rad_georef['ygrid'][:, idx_bhb]
        dmmyz_mlb = np.ones(dmmyx_mlb.shape)
        dmmyx_mlt = rad_georef['xgrid'][:, idx_bht]
        dmmyy_mlt = rad_georef['ygrid'][:, idx_bht]
        dmmyz_mlt = np.ones(dmmyx_mlt.shape)

    dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
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
            cmap = mpl.colormaps['tpylsc_2slope']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx['D'].pcolormesh(rad_georef['xgrid'],
                                           rad_georef['ygrid'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['D'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx['D'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx['D'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                            c='tab:orange')
    plt.colorbar(fzhna, ax=ax_idx['D']).ax.tick_params(labelsize=10)
    ax_idx['D'].grid(True)
    ax_idx['D'].axes.set_aspect('equal')
    ax_idx['D'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if '[dB]' in key:
            cmap = mpl.colormaps['tpylsc_2slope']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx['E'].pcolormesh(rad_georef['xgrid'],
                                           rad_georef['ygrid'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['E'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx['E'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx['E'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                            c='tab:orange')
    plt.colorbar(fzhna, ax=ax_idx['E']).ax.tick_params(labelsize=10)
    ax_idx['E'].grid(True)
    ax_idx['E'].axes.set_aspect('equal')
    ax_idx['E'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if 'ADP' in key:
            cmap = mpl.colormaps['tpylsc_pvars']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx['F'].pcolormesh(rad_georef['xgrid'],
                                           rad_georef['ygrid'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['F'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['F'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx['F'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                            c='tab:orange')
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
            cmap = 'tpylsc_grad_fiery'
            norm = dnorm.get('n'+key)
            fzhna = ax_idx3.pcolormesh(rad_georef['xgrid'],
                                       rad_georef['ygrid'], value,
                                       shading='auto', cmap=cmap, norm=norm)
            ax_idx3.set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx3.scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx3.scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                        c='tab:orange')
    plt.colorbar(fzhna, ax=ax_idx3).ax.tick_params(labelsize=10)
    ax_idx3.grid(True)
    ax_idx3.axes.set_aspect('equal')
    ax_idx3.tick_params(axis='both', labelsize=10)
    plt.tight_layout()


def plot_attcorrection2(rad_georef, rad_params, rad_vars_att, rad_vars_attcorr,
                        vars_bounds=None, mlyr=None, xlims=None, ylims=None):
    """
    Plot the results of the attenuation correction method.

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
             'Rainfall [mm/hr]': [0.1, 64, 11]}
    mlyr : MeltingLayer Class, optional
        Plots an isotropic melting layer. The default is None.
    xlims : 2-element tuple or list, optional
        Set the x-axis view limits [min, max]. The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
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

    dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value,
                                               mpl.colormaps['tpylsc_ref'].N,
                                               extend='both')
             for key, value in bnd.items()}
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(bnd['brhoHV [-]'],
                                                   mpl.colormaps['tpylsc_pvars'].N,
                                                   extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        bnrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/hr]'],
                                    mpl.colormaps['tpylsc_rainrt'].N,
                                    extend='max')
        dnorm['nRainfall [mm/hr]'] = bnrr
    if 'bZDR [dB]' in bnd.keys():
        dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(bnd['bZDR [dB]'],
                                                  mpl.colormaps['tpylsc_2slope'].N,
                                                  extend='both')
        # dnorm['nZDR [dB]'] = mcolors.TwoSlopeNorm(vmin=lpv['ZDR [dB]'][0],
        #                                           vcenter=0,
        #                                           vmax=lpv['ZDR [dB]'][1],
    if 'bKDP [deg/km]' in bnd.keys():
        dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(bnd['bKDP [deg/km]'],
                                                      mpl.colormaps['tpylsc_2slope'].N,
                                                      extend='both')
    if 'bV [m/s]' in bnd.keys():
        dnorm['nV [m/s]'] = mcolors.BoundaryNorm(bnd['bV [m/s]'],
                                                 mpl.colormaps['tpylsc_dbu_rd'].N,
                                                 extend='both')
    if mlyr is not None:
        idx_bhb = rut.find_nearest(rad_georef['beam_height [km]'],
                                   mlyr.ml_bottom)
        idx_bht = rut.find_nearest(rad_georef['beam_height [km]'],
                                   mlyr.ml_top)
        dmmyx_mlb = rad_georef['xgrid'][:, idx_bhb]
        dmmyy_mlb = rad_georef['ygrid'][:, idx_bhb]
        dmmyz_mlb = np.ones(dmmyx_mlb.shape)
        dmmyx_mlt = rad_georef['xgrid'][:, idx_bht]
        dmmyy_mlt = rad_georef['ygrid'][:, idx_bht]
        dmmyz_mlt = np.ones(dmmyx_mlt.shape)

    dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2
    # if isinstance(rescorr4attzdr, dict):
    if any(key.lower().startswith('zdr') for key in rad_vars_attcorr):
        mosaic = 'ABC;DEF'
        fig_size = (16, 9)
    else:
        mosaic = 'ABC'
        fig_size = (16, 5)
    # =========================================================================
    # Creates plots for ZH and ZDR attenuation corrections.
    # =========================================================================
    fig_mos1 = plt.figure(figsize=fig_size, constrained_layout=True)
    ax_idx = fig_mos1.subplot_mosaic(mosaic, sharex=True, sharey=True)
    for key, value in rad_vars_att.items():
        if '[dBZ]' in key:
            cmap = mpl.colormaps['tpylsc_ref']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx['A'].pcolormesh(rad_georef['xgrid'],
                                           rad_georef['ygrid'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['A'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx['A'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx['A'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
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
            cmap = mpl.colormaps['tpylsc_ref']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx['B'].pcolormesh(rad_georef['xgrid'],
                                           rad_georef['ygrid'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['B'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx['B'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx['B'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
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
            cmap = mpl.colormaps['tpylsc_pvars']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx['C'].pcolormesh(rad_georef['xgrid'],
                                           rad_georef['ygrid'], value,
                                           shading='auto', cmap=cmap,
                                           norm=norm)
            ax_idx['C'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx['C'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx['C'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
    plt.colorbar(fzhna, ax=ax_idx['C']).ax.tick_params(labelsize=10)
    ax_idx['C'].grid(True)
    ax_idx['C'].axes.set_aspect('equal')
    ax_idx['C'].tick_params(axis='both', labelsize=10)
    # if isinstance(rescorr4attzdr, dict):
    if any(key.lower().startswith('zdr') for key in rad_vars_attcorr):
        for key, value in rad_vars_att.items():
            if '[dB]' in key:
                cmap = mpl.colormaps['tpylsc_2slope']
                norm = dnorm.get('n'+key)
                fzhna = ax_idx['D'].pcolormesh(rad_georef['xgrid'],
                                               rad_georef['ygrid'], value,
                                               shading='auto', cmap=cmap,
                                               norm=norm)
                ax_idx['D'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
        if mlyr is not None:
            ax_idx['D'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
            ax_idx['D'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                                c='tab:orange')
        plt.colorbar(fzhna, ax=ax_idx['D']).ax.tick_params(labelsize=10)
        ax_idx['D'].grid(True)
        ax_idx['D'].axes.set_aspect('equal')
        ax_idx['D'].tick_params(axis='both', labelsize=10)
        for key, value in rad_vars_attcorr.items():
            if '[dB]' in key:
                cmap = mpl.colormaps['tpylsc_2slope']
                norm = dnorm.get('n'+key)
                fzhna = ax_idx['E'].pcolormesh(rad_georef['xgrid'],
                                               rad_georef['ygrid'], value,
                                               shading='auto', cmap=cmap,
                                               norm=norm)
                ax_idx['E'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
        if mlyr is not None:
            ax_idx['E'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
            ax_idx['E'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                                c='tab:orange')
        plt.colorbar(fzhna, ax=ax_idx['E']).ax.tick_params(labelsize=10)
        ax_idx['E'].grid(True)
        ax_idx['E'].axes.set_aspect('equal')
        ax_idx['E'].tick_params(axis='both', labelsize=10)
        for key, value in rad_vars_attcorr.items():
            if 'ADP' in key:
                cmap = mpl.colormaps['tpylsc_pvars']
                norm = dnorm.get('n'+key)
                fzhna = ax_idx['F'].pcolormesh(rad_georef['xgrid'],
                                               rad_georef['ygrid'], value,
                                               shading='auto', cmap=cmap,
                                               norm=norm)
                ax_idx['F'].set_title(f"{ptitle}" "\n" f'{key}')
        if mlyr is not None:
            ax_idx['F'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
            ax_idx['F'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                                c='tab:orange')
        plt.colorbar(fzhna, ax=ax_idx['F']).ax.tick_params(labelsize=10)
        ax_idx['F'].grid(True)
        ax_idx['F'].axes.set_aspect('equal')
        ax_idx['F'].tick_params(axis='both', labelsize=10)
    # =========================================================================
    # Creates plots for PHIDP attenuation corrections.
    # =========================================================================
    fig_mos2 = plt.figure(figsize=fig_size, constrained_layout=True)
    ax_idx2 = fig_mos2.subplot_mosaic(mosaic, sharex=True, sharey=True)
    for key, value in rad_vars_att.items():
        if '[deg]' in key:
            cmap = mpl.colormaps['tpylsc_pvars']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx2['A'].pcolormesh(rad_georef['xgrid'],
                                            rad_georef['ygrid'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['A'].set_title(f"{ptitle}" "\n" f'Uncorrected {key}')
    if mlyr is not None:
        ax_idx2['A'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx2['A'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
    plt.colorbar(fzhna, ax=ax_idx2['A']).ax.tick_params(labelsize=10)
    ax_idx2['A'].grid(True)
    ax_idx2['A'].axes.set_aspect('equal')
    ax_idx2['A'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if key == 'PhiDP [deg]':
            cmap = mpl.colormaps['tpylsc_pvars']
            norm = dnorm.get('n'+key)
            fzhna = ax_idx2['B'].pcolormesh(rad_georef['xgrid'],
                                            rad_georef['ygrid'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['B'].set_title(f"{ptitle}" "\n" f'Corrected {key}')
    if mlyr is not None:
        ax_idx2['B'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx2['B'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
    plt.colorbar(fzhna, ax=ax_idx2['B']).ax.tick_params(labelsize=10)
    ax_idx2['B'].grid(True)
    ax_idx2['B'].axes.set_aspect('equal')
    ax_idx2['B'].tick_params(axis='both', labelsize=10)
    for key, value in rad_vars_attcorr.items():
        if key == 'PhiDP* [deg]':
            cmap = mpl.colormaps['tpylsc_pvars']
            norm = dnorm.get('n'+key.replace('*', ''))
            fzhna = ax_idx2['C'].pcolormesh(rad_georef['xgrid'],
                                            rad_georef['ygrid'], value,
                                            shading='auto', cmap=cmap,
                                            norm=norm)
            ax_idx2['C'].set_title(f"{ptitle}" "\n" f'{key}')
    if mlyr is not None:
        ax_idx2['C'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
        ax_idx2['C'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt, c='tab:orange')
    plt.colorbar(fzhna, ax=ax_idx2['C']).ax.tick_params(labelsize=10)
    ax_idx2['C'].grid(True)
    ax_idx2['C'].axes.set_aspect('equal')
    ax_idx2['C'].tick_params(axis='both', labelsize=10)
    # if isinstance(rescorr4attzdr, dict):
    if any(key.lower().startswith('zdr') for key in rad_vars_attcorr):
        for key, value in rad_vars_attcorr.items():
            if key == 'alpha [-]':
                cmap = 'tpylsc_grad_fiery'
                norm = dnorm.get('n'+key)
                fzhna = ax_idx2['D'].pcolormesh(rad_georef['xgrid'],
                                                rad_georef['ygrid'], value,
                                                shading='auto', cmap=cmap,
                                                norm=norm)
                ax_idx2['D'].set_title(f"{ptitle}" "\n" f'{key}')
        if mlyr is not None:
            ax_idx2['D'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
            ax_idx2['D'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                                 c='tab:orange')
        plt.colorbar(fzhna, ax=ax_idx2['D']).ax.tick_params(labelsize=10)
        ax_idx2['D'].grid(True)
        ax_idx2['D'].axes.set_aspect('equal')
        ax_idx2['D'].tick_params(axis='both', labelsize=10)
        for key, value in rad_vars_attcorr.items():
            if key == 'beta [-]':
                cmap = 'tpylsc_grad_fiery'
                norm = dnorm.get('n'+key)
                fzhna = ax_idx2['E'].pcolormesh(rad_georef['xgrid'],
                                                rad_georef['ygrid'], value,
                                                shading='auto', cmap=cmap,
                                                norm=norm)
                ax_idx2['E'].set_title(f"{ptitle}" "\n" f'{key}')
        if mlyr is not None:
            ax_idx2['E'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
            ax_idx2['E'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                                 c='tab:orange')
        plt.colorbar(fzhna, ax=ax_idx2['E']).ax.tick_params(labelsize=10)
        ax_idx2['E'].grid(True)
        ax_idx2['E'].axes.set_aspect('equal')
        ax_idx2['E'].tick_params(axis='both', labelsize=10)
        for key, value in rad_vars_attcorr.items():
            if 'PIA' in key:
                cmap = 'tpylsc_grad_fiery'
                norm = dnorm.get('n'+key)
                fzhna = ax_idx2['F'].pcolormesh(rad_georef['xgrid'],
                                                rad_georef['ygrid'], value,
                                                shading='auto', cmap=cmap,
                                                norm=norm)
                ax_idx2['F'].set_title(f"{ptitle}" "\n" f'{key}')
        if mlyr is not None:
            ax_idx2['F'].scatter(dmmyx_mlb, dmmyy_mlb, dmmyz_mlb, c='tab:blue')
            ax_idx2['F'].scatter(dmmyx_mlt, dmmyy_mlt, dmmyz_mlt,
                                 c='tab:orange')
        plt.colorbar(fzhna, ax=ax_idx2['F']).ax.tick_params(labelsize=10)
        ax_idx2['F'].grid(True)
        ax_idx2['F'].axes.set_aspect('equal')
        ax_idx2['F'].tick_params(axis='both', labelsize=10)


def plot_radprofiles(rad_params, beam_height, rad_profs, mlyr=None, ylims=None,
                     vars_bounds=None, stats=None, colours=False, unorm=None,
                     ucmap=None, fig_size=None):
    """
    Display a set of profiles of polarimetric variables.

    Parameters
    ----------
    rad_params : dict
        Radar technical details.
    beam_height : array
        The beam height.
    rad_profs : dict
        Profiles generated by the PolarimetricProfiles class.
    mlyr : MeltingLayer Class, optional
        Plots the melting layer within the polarimetric profiles.
        The default is None.
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    vars_bounds : dict containing key and 3-element tuple or list, optional
        Boundaries [min, max] between which radar variables are
        to be plotted.
    stats : dict, optional
        Statistics of the profiles generation computed by the
        PolarimetricProfiles class. The default is None.
    colours : Bool, optional
        Creates coloured profiles using norm to map colormaps.
    ucmap : colormap, optional
        User-defined colormap.
    unorm : matplotlib.colors normalisation object, optional
        User-defined normalisation method to map colormaps onto radar data.
        The default is None.
    """
    fontsizelabels = 20
    fontsizetitle = 25
    fontsizetick = 18
    lpv = {'ZH [dBZ]': [-10, 60, 15], 'ZDR [dB]': [-2, 6, 17],
           'PhiDP [deg]': [0, 180, 10], 'KDP [deg/km]': [-2, 6, 17],
           'rhoHV [-]': [0.3, .9, 1], 'V [m/s]': [-15, 15, 11],
           'gradV [dV/dh]': [-1, 0, 11], 'LDR [dB]': [-35, 0, 11],
           'Rainfall [mm/hr]': [0.1, 64, 11]}
    if vars_bounds is not None:
        lpv.update(vars_bounds)
    bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
    if vars_bounds is None:
        bnd['bRainfall [mm/hr]'] = np.array((0.01, 0.5, 1, 2, 4, 8, 12, 20,
                                             28, 36, 48, 64, 80, 100))

    dnorm = {'n'+key[1:]: mcolors.BoundaryNorm(value,
                                               mpl.colormaps['tpylsc_pvars'].N,
                                               extend='both')
             for key, value in bnd.items()}
    if 'bZH [dBZ]' in bnd.keys():
        dnorm['nZH [dBZ]'] = mcolors.BoundaryNorm(bnd['bZH [dBZ]'],
                                                  mpl.colormaps['tpylsc_ref'].N,
                                                  extend='both')
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(bnd['brhoHV [-]'],
                                                   mpl.colormaps['tpylsc_pvars'].N,
                                                   extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        bnrr = mcolors.BoundaryNorm(bnd['bRainfall [mm/hr]'],
                                    mpl.colormaps['tpylsc_rainrt'].N,
                                    extend='max')
        dnorm['nRainfall [mm/hr]'] = bnrr
    if 'bZDR [dB]' in bnd.keys():
        dnorm['nZDR [dB]'] = mcolors.BoundaryNorm(bnd['bZDR [dB]'],
                                                  mpl.colormaps['tpylsc_2slope'].N,
                                                  extend='both')
        # dnorm['nZDR [dB]'] = mcolors.TwoSlopeNorm(vmin=lpv['ZDR [dB]'][0],
        #                                           vcenter=0,
        #                                           vmax=lpv['ZDR [dB]'][1],
    if 'bKDP [deg/km]' in bnd.keys():
        dnorm['nKDP [deg/km]'] = mcolors.BoundaryNorm(bnd['bKDP [deg/km]'],
                                                      mpl.colormaps['tpylsc_2slope'].N,
                                                      extend='both')
    if 'bV [m/s]' in bnd.keys():
        dnorm['nV [m/s]'] = mcolors.BoundaryNorm(bnd['bV [m/s]'],
                                                 mpl.colormaps['tpylsc_dbu_rd'].N,
                                                 extend='both')
    if unorm is not None:
        dnorm.update(unorm)

    ttxt_elev = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    ttxt_dt = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ttxt = ttxt_elev+ttxt_dt
    if fig_size is None:
        fig_size = (14, 10)

    def make_colorbar(ax1, mappable, **kwargs):
        ax1_divider = make_axes_locatable(ax1)
        orientation = kwargs.pop('orientation', 'vertical')
        if orientation == 'vertical':
            loc = 'right'
        elif orientation == 'horizontal':
            loc = 'top'
        cax = ax1_divider.append_axes(loc, '7%', pad='5%',
                                      axes_class=plt.Axes)
        ax1.get_figure().colorbar(mappable, cax=cax,
                                  orientation=orientation,
                                  ticks=bnd.get('b'+key),
                                  format=f'%.{cbtks_fmt}f',
                                  )
        cax.tick_params(direction='in', labelsize=10, rotation=90)
        cax.xaxis.set_ticks_position('bottom')
    if rad_params['elev_ang [deg]'] > 89:
        fig, ax = plt.subplots(1, len(rad_profs), figsize=fig_size,
                               sharey=True)
        fig.suptitle(f'Vertical profiles of polarimetric variables'
                     '\n' f'{ttxt}',
                     fontsize=fontsizetitle)
    else:
        fig, ax = plt.subplots(1, len(rad_profs), figsize=fig_size,
                               sharey=True)
        fig.suptitle('Quasi-Vertical profiles of polarimetric variables \n'
                     f'{ttxt}',
                     fontsize=fontsizetitle)

    for n, (a, (key, value)) in enumerate(zip(ax.flatten(),
                                              rad_profs.items())):
        if colours is False:
            a.plot(value, beam_height, 'k')
        elif colours:
            cbtks_fmt = 0
            if '[dBZ]' in key:
                cmaph = mpl.colormaps['tpylsc_ref']
            if '[-]' in key:
                cmaph = mpl.colormaps['tpylsc_pvars']
                cbtks_fmt = 2
            if '[deg]' in key:
                cmaph = mpl.colormaps['tpylsc_pvars']
            if '[dB]' in key:
                cmaph = mpl.colormaps['tpylsc_2slope']
                cbtks_fmt = 1
            if '[dV/dh]' in key:
                cmaph = mpl.colormaps['tpylsc_dbu_rd_r']
                cbtks_fmt = 1
            if '[deg/km]' in key:
                cmaph = mpl.colormaps['tpylsc_2slope']
                cbtks_fmt = 1
            if '[m/s]' in key:
                cmaph = mpl.colormaps['tpylsc_dbu_rd']
            if '[mm/hr]' in key:
                cmaph = mpl.colormaps['tpylsc_rainrt']
            if ucmap is not None:
                cmaph = ucmap
            points = np.array([value, beam_height]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmaph,
                                norm=dnorm.get('n'+key))
            # Set the values used for colormapping
            lc.set_array(value)
            lc.set_linewidth(2)
            line = a.add_collection(lc)
            make_colorbar(a, lc, orientation='horizontal')
            a.set_xlim([np.nanmin(value), np.nanmax(value)])
        if stats:
            a.fill_betweenx(beam_height,
                            value + stats.get(key, value*np.nan),
                            value - stats.get(key, value*np.nan),
                            alpha=0.4, color='gray', label='std')
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
        elif key == 'gradV [dV/dh]' and rad_params['elev_ang [deg]'] > 89:
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


def plot_rdqvps(rscans_georef, rscans_params, tp_rdqvp, mlyr=None, ucmap=None,
                spec_range=None, vars_bounds=None, ylims=None, fig_size=None):
    """
    Display a set of RD-QVPS of polarimetric variables.

    Parameters
    ----------
    rscans_georef : List
        List of eoreferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others, corresponding to each QVP.
    rscans_params : List
        List of radar technical details corresponding to each QVP.
    tp_rdqvp : PolarimetricProfiles Class
        Outputs of the RD-QVPs function.
    mlyr : MeltingLayer Class, optional
        Plots the melting layer within the polarimetric profiles.
        The default is None.
    vars_bounds : dict containing key and 2-element tuple or list, optional
        Boundaries [min, max] between which radar variables are
        to be plotted. The default are:
            {'ZH [dBZ]': [-10, 60],
             'ZDR [dB]': [-2, 6],
             'PhiDP [deg]': [0, 180], 'KDP [deg/km]': [-2, 6],
             'rhoHV [-]': [0.6, 1],
             'LDR [dB]': [-35, 0],
             'V [m/s]': [-5, 5], 'gradV [dV/dh]': [-1, 0]}
    ylims : 2-element tuple or list, optional
        Set the y-axis view limits [min, max]. The default is None.
    ucmap : colormap, optional
        User-defined colormap.
    spec_range : int, optional
        Range from the radar within which the data was used.
    """
    tpcm = 'tpylsc_pvars_r'
    if fig_size is None:
        fig_size = (14, 10)
    cmaph = mpl.colormaps[tpcm](np.linspace(0., .8,
                                            len(tp_rdqvp.qvps_itp)))
    if ucmap is not None:
        cmaph = mpl.colormaps[ucmap](np.linspace(0, 1,
                                                 len(tp_rdqvp.qvps_itp)))

    fontsizelabels = 20
    fontsizetitle = 25
    fontsizetick = 18
    lpv = {'ZH [dBZ]': [-10, 60], 'ZDR [dB]': [-2, 2],
           'PhiDP [deg]': [0, 90], 'KDP [deg/km]': [-2, 6],
           'rhoHV [-]': [0.9, 1], 'LDR [dB]': [-35, 0],
           'V [m/s]': [-5, 5], 'gradV [dV/dh]': [-1, 0],
           }
    if vars_bounds:
        lpv.update(vars_bounds)

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

    for c, i in enumerate(tp_rdqvp.qvps_itp):
        for n, (a, (key, value)) in enumerate(zip(axd, i.items())):
            axd[a].plot(value, tp_rdqvp.georef['profiles_height [km]'],
                        color=cmaph[c], ls='--',
                        label=(f"{rscans_params[c]['elev_ang [deg]']:.1f}"
                               + r"$^{\circ}$"))
            axd[a].set_xlabel(f'{key}', fontsize=fontsizelabels)
            if n == 0:
                axd[a].set_ylabel('Height [km]', fontsize=fontsizelabels,
                                  labelpad=10)
            axd[a].tick_params(axis='both', labelsize=fontsizetick)
            axd[a].grid(True)
            # axd[a].legend(loc='upper right')
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

    scan_st = axd[mosaic[-1]]
    for c, i in enumerate(rscans_georef):
        scan_st.plot(i['range [m]']/1000, i['beam_height [km]'][0],
                     color=cmaph[c], ls='--',
                     label=(f"{rscans_params[c]['elev_ang [deg]']:.1f}"
                            + r"$^{\circ}$"))
        # scan_st.plot(i['range [m]']/-1000, i['beam_height [km]'][0],
        #               color=cmaph[c], ls='--')
    scan_st.set_xlabel('Range [km]', fontsize=fontsizelabels)
    scan_st.tick_params(axis='both', labelsize=fontsizetick)
    scan_st.grid(True)
    scan_st.legend(loc='upper right')
    if spec_range:
        scan_st.axvline(spec_range, c='k', lw=3)


def plot_rhocalibration(hists, histmax, idxminstd, rng_ite, fig_size=None):
    """
    S.

    Parameters
    ----------
    hists : TYPE
        DESCRIPTION.
    histmax : TYPE
        DESCRIPTION.
    idxminstd : TYPE
        DESCRIPTION.
    rng_ite : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    nr = 1
    nc = int(len(hists))
    if len(hists) > 3:
        nc = 3
        # nr = int(len(hists)/nc)
        nr = len(hists) // nc + (len(hists) % nc > 0)
    # if len(hists) > 3 and len(hists) % 2:
    #     nr = int((len(hists)//nc)+1)

    # f, ax = plt.subplots(nr, nc, sharex=True, sharey=True)
    # f.suptitle(f'{ptitle}', fontsize=16)
    # for a, (key, value) in zip(ax.flatten(), rad_vars.items()):
    if fig_size is None:
        fig_size = (16, 9)
    fig, axes = plt.subplots(sharex=True, sharey=True, nrows=nr, ncols=nc,
                             figsize=fig_size, constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        if i < len(hists):
            if i == idxminstd:
                ax.set_title(f'{rng_ite[i]:.3f}', c='tab:green')
            else:
                ax.set_title(f'{rng_ite[i]:.3f}', c='gray')
            ax.plot(hists[i][1][1:], histmax[i], color="k", zorder=10)
            pcm = ax.pcolormesh(hists[i][1], hists[i][2], hists[i][0].T,
                                # vmax=1e1,
                                norm=mcolors.LogNorm(vmax=1.5e1),
                                rasterized=True,
                                cmap='plasma')
            fig.colorbar(pcm, ax=ax, label="# points", pad=0)
            ax.set_xlim(5, 30)
            ax.set_ylim(0.8, 1.1)
            ax.tick_params(axis='both', which='major', labelsize=10)


def plot_offsetcorrection(rad_georef, rad_params, rad_var, fig_size=None,
                          var_name='PhiDP [deg]', cmap='tpylsc_dbu_w_rd'):
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
        User-defined colormap. The default is 'tpylsc_dbu_w_rd'.
    """
    var_mean = np.array([np.nanmean(i) for i in rad_var])
    if var_name == 'PhiDP [deg]':
        label1 = r'$\Phi_{DP}}$'
        labelm = r'$\overline{\Phi_{DP}}$'
        labelo = r'$\Phi_{DP}}$ offset'
        dval = 3
        dof = 10
    elif var_name == 'ZDR [dB]':
        label1 = '$Z_{DR}}$'
        labelm = r'$\overline{Z_{DR}}$'
        labelo = r'$Z_{DR}}$ offset'
        dval = 0.1
        dof = 1
    if fig_size is None:
        fig_size = (8, 8)

    fig, ax = plt.subplots(figsize=fig_size,
                           subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)
    dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
    dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
    ptitle = dtdes1 + dtdes2
    ax.set_title(ptitle, fontsize=16)
    ax.grid(color='gray', linestyle=':')
    ax.set_theta_zero_location('N', offset=0)
    # =========================================================================
    # Plot the radar variable values at each azimuth
    # =========================================================================
    ax.scatter((np.ones_like(rad_var.T) * [rad_georef['azim [rad]']]).T,
               rad_var, s=5, c=rad_var, cmap=cmap, label=label1,
               norm=mcolors.SymLogNorm(linthresh=.01, linscale=.01, base=2,
                                       vmin=var_mean.mean()-dval,
                                       vmax=var_mean.mean()+dval))
    # =========================================================================
    # Plot the radar variable mean value of each azimuth
    # =========================================================================
    ax.plot(rad_georef['azim [rad]'], var_mean, c='tab:green', linewidth=2,
            ls='', marker='s', markeredgecolor='g', alpha=0.4,
            label=labelm)
    # =========================================================================
    # Plot the radar variable offset
    # =========================================================================
    ax.plot(rad_georef['azim [rad]'], np.full(rad_georef['azim [rad]'].shape,
                                              var_mean.mean()),
            c='k', linewidth=2.5, label=labelo)

    ax.set_thetagrids(np.arange(0, 360, 90))
    ax.xaxis.grid(ls='-')
    ax.tick_params(axis='both', labelsize=14)
    ax.set_rlabel_position(-45)
    if var_name == 'PhiDP [deg]':
        ax.set_ylim([var_mean.mean()-dof, var_mean.mean()+dof])
        ax.set_yticks(np.arange(round(var_mean.mean()/dval)*dval-dof,
                                round(var_mean.mean()/dval)*dval+dof+1,
                                dval))
    else:
        ax.set_ylim([var_mean.mean()-dof, var_mean.mean()+dof])
        # ax.set_yticks(np.arange(round(var_mean.mean()/dval)*dval-dof,
        #                         round(var_mean.mean()/dval)*dval+dof+.1,
        #                         dval))
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

        at = AnchoredText(mfspk[key][1], loc=10,
                          prop=dict(size=18, color='white'), frameon=False)
        cax.add_artist(at)
        a.legend(fontsize=14)
    f.tight_layout()
