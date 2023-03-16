"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
from matplotlib.offsetbox import AnchoredText
import cartopy.io.shapereader as shpreader
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

tpycm_ref = mpl.colormaps['tpylsc_ref']
tpycm_plv = mpl.colormaps['tpylsc_pvars']
tpycm_rnr = mpl.colormaps['tpylsc_rainrt']
tpycm_2slope = mpl.colormaps['tpylsc_2slope']
tpycm_dv = mpl.colormaps['tpylsc_dbu_rd']
tpycm_3c = mpl.colormaps['tpylc_yw_gy_bu']


def anim_base(rad_georef, rad_params, rad_vars, gifdir=None, vars_bounds=None,
              var2plot=None, cpy_feats=None, single_site=True, shpfile=None,
              data_info=True, xlims=None, ylims=None, logo=False):
    """
    Create an animation using georeferenced radar data.

    Parameters
    ----------
    rad_georef : dict
        Georeferenced data containing descriptors of the azimuth, gates
        and beam height, amongst others.
    rad_params : dict
        Radar technical details.
    rad_vars : TYPE
        DESCRIPTION.
    gifdir : TYPE, optional
        DESCRIPTION. The default is None.
    vars_bounds : TYPE, optional
        DESCRIPTION. The default is None.
    var2plot : TYPE, optional
        DESCRIPTION. The default is None.
    cpy_feats : TYPE, optional
        DESCRIPTION. The default is None.
    single_site : TYPE, optional
        DESCRIPTION. The default is True.
    shpfile : TYPE, optional
        DESCRIPTION. The default is None.
    data_info : TYPE, optional
        DESCRIPTION. The default is True.
    xlims : TYPE, optional
        DESCRIPTION. The default is None.
    ylims : TYPE, optional
        DESCRIPTION. The default is None.
    logo : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    lpv = {'ZH [dBZ]': [-10, 60, 15], 'ZDR [dB]': [-2, 6, 17],
           'PhiDP [deg]': [0, 180, 11], 'KDP [deg/km]': [-2, 6, 17],
           'rhoHV [-]': [0.3, .9, 1],
           'V [m/s]': [-5, 5, 11], 'gradV [dV/dh]': [-1, 0, 11],
           'LDR [dB]': [-35, 0, 11],
           'Rainfall [mm/hr]': [0.1, 64, 11]}
    if vars_bounds is not None:
        lpv.update(vars_bounds)
    cpy_features = {'status': False,
                    # 'coastresolution': '10m',
                    'add_land': True,
                    'add_ocean': True,
                    'add_coastline': True,
                    'add_borders': False,
                    'borders_ls': ':',
                    'add_lakes': False,
                    'lakes_transparency': 0.5,
                    'add_rivers': False
                    }
    if cpy_feats is not None:
        cpy_features.update(cpy_feats)
    bnd = {'b'+key: np.linspace(value[0], value[1], value[2])
           if 'rhoHV' not in key
           else np.hstack((np.linspace(value[0], value[1], 4)[:-1],
                           np.linspace(value[1], value[2], 11)))
           for key, value in lpv.items()}
    bnd['bRainfall [mm/hr]'] = np.array([0.01, 0.5, 1, 2, 4, 8, 16, 32, 64])
    dnorm = {'n'+key[1:]: mpl.colors.BoundaryNorm(value, ncolors=256,
                                                  extend='both')
             if 'ZDR' not in key
             else mcolors.BoundaryNorm(value, ncolors=256, extend='both')
             for key, value in bnd.items()}
    if 'brhoHV [-]' in bnd.keys():
        dnorm['nrhoHV [-]'] = mcolors.BoundaryNorm(boundaries=bnd['brhoHV [-]'],
                                                   ncolors=256, extend='min')
    if 'bRainfall [mm/hr]' in bnd.keys():
        dnorm['nRainfall [mm/hr]'] = mcolors.BoundaryNorm(boundaries=bnd['bRainfall [mm/hr]'],
                                                          ncolors=256,
                                                          extend='max')
    txtboxs = 'round, rounding_size=0.5, pad=0.5'
    txtboxc = (0, -.09)
    fc, ec = 'w', 'k'
    widths = [1, 2, 1.5]
    heights = [6, 4, 2.5]
    # widths = [1.4, 1, 1]
    # heights = [1, 2, 1.5]

    if var2plot is None or var2plot == 'ZH [dBZ]':
        cmaph, normp = tpycm_ref, dnorm['nZH [dBZ]']
        var2plot = 'ZH [dBZ]'
        ltxvar = '$Z_H$'
    else:
        cmaph = tpycm_plv
        normp = dnorm.get('n'+var2plot)
        if 'ZDR' in var2plot:
            cmaph = tpycm_2slope
        if var2plot == 'V [m/s]':
            cmaph = tpycm_dv
        if 'Rainfall' in var2plot:
            cmaph = tpycm_rnr
        if 'ZDR' in var2plot:
            ltxvar = '$Z_{DR}$ [dB]'
        if 'rhoHV' in var2plot:
            ltxvar = r'$ \rho_{HV}$ [-]'
        if 'PhiDP' in var2plot:
            ltxvar = r'$ \Phi_{DP}$ [deg]'

    if shpfile is not None:
        de_geo = list(shpreader.Reader(shpfile).geometries())

    if logo:
        img = plt.imread('/home/dsanchez/codes/github/towerpy/towerpy/towerpy_logosd.png')
        # img = plt.imread('/home/enchiladaszen/Documents/github/enchilaDaSzen/towerpy/towerpy/towerpy_logosd.png')
    proj = ccrs.PlateCarree()
    if rad_params[0]['site_name'].endswith('xPol'):
        proj2 = ccrs.UTM(zone=32)
    else:
        proj2 = ccrs.OSGB(approx=False)
    # if 'OSGB' in proj2.proj4_init:
    #     unitm = 1000
    # else:
    #     unitm = 1
    if data_info:
        figanim = plt.figure(figsize=(15, 8), constrained_layout=True)
        plt.subplots_adjust(
            left=0.05, right=0.99, top=0.981, bottom=0.019,
            wspace=0, hspace=1
            )
        gs = figanim.add_gridspec(3, 3,
                                  width_ratios=widths,
                                  height_ratios=heights
                                  )
        ax2 = figanim.add_subplot(gs[:, -1])

        ax3 = figanim.add_subplot(gs[-1:, -1])
        if logo:
            ax3.imshow(img)
        ax3.axis('off')

    if cpy_features['status']:
        ax1 = figanim.add_subplot(gs[:-1], projection=proj)
    else:
        ax1 = figanim.add_subplot(gs[:-1])

    if cpy_features['status']:
        if xlims is None:
            if rad_params[0]['site_name'].endswith('xPol'):
                extx = [4.5, 16.5]  # BoxPol
            else:
                extx = [-10.5, 3.5]  # whole UK
        else:
            extx = xlims
        if ylims is None:
            if rad_params[0]['site_name'].endswith('xPol'):
                exty = [55.5, 46.5]  # BoxPol
            else:
                exty = [60, 48]  # whole UK
        else:
            exty = ylims

    gfname = ('anim' + rad_params[0]['datetime'].strftime("%Y%m%d%H%M%S_") +
              rad_params[-1]['datetime'].strftime("%Y%m%d%H%M%S"))

    # def init_run():
    #     if cpy_features['status']:
    #         ax1.set_extent(extx+exty, crs=proj)
    #         # ax1.coastlines(resolution='10m')
    #         if cpy_features['add_land']:
    #             ax1.add_feature(cfeature.LAND)
    #         if cpy_features['add_ocean']:
    #             ax1.add_feature(cfeature.OCEAN)
    #         if cpy_features['add_coastline']:
    #             ax1.add_feature(cfeature.COASTLINE)
    #         if cpy_features['add_borders']:
    #             ax1.add_feature(cfeature.BORDERS,
    #                             linestyle=cpy_features['borders_ls'])
    #         if cpy_features['add_lakes']:
    #             ax1.add_feature(cfeature.LAKES,
    #                             alpha=cpy_features['lakes_transparency'])
    #         if cpy_features['add_rivers']:
    #             ax1.add_feature(cfeature.RIVERS)
    #         # SOURCE = 'Natural Earth'
    #         # LICENSE = 'public domain'
    #         # Add a text annotation for the license information to the
    #         # the bottom right corner.
    #         # text = AnchoredText(r'$\copyright$ {}; license: {}'
    #         #                     ''.format(SOURCE, LICENSE),
    #         #                     loc=4, prop={'size': 12}, frameon=True)
    #         # ax1.add_artist(text)
    #         if shpfile is not None:
    #             ax1.add_geometries(de_geo, ccrs.PlateCarree(),
    #                                edgecolor='gray', facecolor='w', alpha=0.5,
    #                                linestyle='--', zorder=0)
    #         gl = ax1.gridlines(draw_labels=True, dms=False,
    #                        x_inline=False, y_inline=False)
    #         gl.xlabel_style = {'size': 11}
    #         gl.ylabel_style = {'size': 11}
    #     else:
    #         ax1.set(xlabel='Distance from the radar [km]',
    #                 ylabel='Distance from the radar [km]')
    #         ax1.grid(color='gray', linestyle=':')

    def animate(i):
        ax1.clear()
        if cpy_features['status']:
            ax1.set_extent(extx+exty, crs=proj)
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
            SOURCE = 'Natural Earth'
            LICENSE = 'public domain'
            # Add a text annotation for the license information to the
            # the bottom right corner.
            # text = AnchoredText(r'$\copyright$ {}; license: {}'
            #                     ''.format(SOURCE, LICENSE),
            #                     loc=4, prop={'size': 12}, frameon=True)
            # ax1.add_artist(text)
            if shpfile is not None:
                ax1.add_geometries(de_geo, ccrs.PlateCarree(),
                                   edgecolor='gray', facecolor='w', alpha=0.5,
                                   linestyle='--', zorder=0)
            gl = ax1.gridlines(draw_labels=True, dms=False,
                               x_inline=False, y_inline=False)
            gl.xlabel_style = {'size': 11}
            gl.ylabel_style = {'size': 11}
            ax1.pcolormesh(rad_georef[i]['xgrid_proj'],
                           rad_georef[i]['ygrid_proj'],
                           rad_vars[i][var2plot],
                           shading='auto', cmap=cmaph, norm=normp,
                           transform=proj2)
            anim_titl1 = f'{rad_params[i]["elev_ang [deg]"]:{2}.{3}} Deg.'
            anim_titl2 = f' {rad_params[i]["datetime"]:%Y-%m-%d %H:%M:%S}'
            anim_title = anim_titl1 + anim_titl2
            ax1.set_title(anim_title)
        else:
            ax1.pcolormesh(rad_georef[i]['xgrid'],
                           rad_georef[i]['ygrid'],
                           rad_vars[i][var2plot],
                           shading='auto', cmap=cmaph, norm=normp)
            ax1.set(xlabel='Distance from the radar [km]',
                    ylabel='Distance from the radar [km]')
            ax1.grid(color='gray', linestyle=':')

        ax2.clear()
        if single_site:
            titl3 = rad_params[i]['site_name']
            if rad_params[i]['site_name'].endswith('xPol'):
                rad_net = 'University of Bonn'
                ubgv = 20
            else:
                rad_net = 'UK Met Office Radar Network'
                ubgv = '?'
            ax2.text(0., .55,
                     f'Site: {titl3}'
                     '\n'f'Radar network: {rad_net}'
                     '\n'f"Lat: {rad_params[i]['latitude [dd]']:.2f}" r"$\degree$"
                     '\n'f"Lon: {rad_params[i]['longitude [dd]']:.2f}" r'$\degree$'
                     '\n'f"Alt: {rad_params[i]['altitude [m]']:.1f} m\n"+
                     '\n' + 'PPI Scan' +
                     '\n' f"Elevation angle: {rad_params[i]['elev_ang [deg]']:.1f}"  r'$\degree$'
                     '\n' f"Frequency: {rad_params[i]['frequency [GHz]']:.1f} GHz"
                     '\n' f"Wavelength: {rad_params[i]['wavelength [cm]']:.2f} cm"
                     '\n' f"Gate resolution: {rad_params[i]['gateres [m]']:.1f} m"
                     '\n' f"No. Gates: {rad_params[i]['ngates']}"
                     '\n' f"PRF: {rad_params[i]['prf [Hz]']:.0f} Hz"
                     '\n' f"Pulse width: {rad_params[i]['pulselength [ns]']:.1f} ns"
                     '\n' f"No. Samples: {rad_params[i]['avsamples']:.0f}"
                      # '\n' f"RPM: {rad_params[i]['rpm']:.2f}"
                     '\n' f"Beamwidth: {rad_params[i]['beamwidth [deg]']:.1f}" r'$\degree$'
                     '\n' rf'Unambiguous velocity: $\pm${ubgv} m/s'
                     '\n' f"Radar constant [dB]: {rad_params[i]['radar constant [dB]']:.1f}"
                     '\n'
                     '\n' 'Variable: ' f"{var2plot.split()[0]}"
                     '\n' 'Units: ' f"{var2plot.split()[1]}"
                     '\n',
                     size=14, ha='left', va='center'
                     )
        else:
            titl3 = 'Radar Mosaic'
            ax2.text(0., .65,
                     f"Site: {titl3}"
                     "\n" f"UK Met Office Radar Network"
                      "\n" f'Frequency: 5.66  GHz'
                     # "\n" f"Lat: {rqpe[i].params['latitude [dd]']:.2f}" r'$\degree$'
                     # "\n" f"Lon: {rqpe[i].params['longitude [dd]']:.2f}" r'$\degree$'
                     # "\n" f"Alt: {rqpe[i].params['altitude [m]']:.1f} m"
                     # "\n"
                     "\n" 'PPI Scan'
                     "\n" f"Gate resolution: 600 m"
                     "\n" f"No. Gates: 425"
                     "\n" f"Elevation/Azimuth: 0.5"  r'$\degree$'
                     "\n" f"PRF: 300 Hz"
                     "\n" f"Pulse width: 2000 ns"
                     "\n" f"No. Samples: 50"
                     # "\n" f"RPM: {rqpe[i].params['rpm']:.2f}"
                     "\n" f"Beamwidth: {1:.1f}" r'$\degree$'
                     "\n" 'Unambiguous velocity: ? m/s'
                     "\n"
                     "\n" 'Variable: ' f"{var2plot.split()[i]}"
                     "\n" 'Units: ' f"{var2plot.split()[1]}"
                     "\n",
                     size=14, ha='center', va='center'
                     )
        ax2.axis('off')

    cbv = figanim.colorbar(mpl.cm.ScalarMappable(norm=normp, cmap=cmaph),
                           ax=ax1, shrink=0.75, aspect=10, pad=0.11)
    cbv.ax.set_title(ltxvar, fontsize=15)
    cbv.ax.tick_params(labelsize=13)

    anim = animation.FuncAnimation(figanim,
                                   animate,
                                   # init_func=init_run,
                                   repeat_delay=10,
                                   frames=len(rad_params),
                                   # blit=True,
                                   # cache_frame_data=False
                                   )
    if gifdir is not None:
        anim.save(f'{gifdir}{gfname}.gif',
                  fps=3,
                  # dpi=50,
                  writer='imagemagick'
                  )
        # writer = animation.FFMpegWriter(fps=5,
        #                                 metadata=dict(artist='Me'),
        #                                 bitrate=1800)
        # anim.save(f'{gifdir}{gfname}.mp4',
        #           writer=writer)
