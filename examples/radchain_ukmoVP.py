#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:39:20 2021

@author: ds17589
"""

import numpy as np
import wradlib as wrl
import towerpy as tp
import cartopy.crs as ccrs

# =============================================================================
# Reads polar radar data
# =============================================================================
rsite = 'chenies'
# rsite = 'deanhill'
rsite = 'jersey'
# rsite = 'druima-starraig'
# fdir = '/home/dsanchez/radar_trials/metoffice/chenies/'
# fdir = f'/media/enchiladaszen/enchiladasz/safe/bristolphd/data4phd/radar_datasets/{rsite}/y2020/spel8/'
fdir = f'/run/media/dsanchez/enchiladasz/safe/bristolphd/data4phd/radar_datasets/{rsite}/y2020/spel8/'
rdata = tp.io.ukmo.Rad_scan(fdir+f'metoffice-c-band-rain-radar_{rsite}_'
                            '202010031836_raw-dual-polar-augzdr-sp-el8.dat',
                            rsite)
rdata.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])
rdata.vars['PhiDP [deg]'] *= -1

# =============================================================================
# Computes the Signal-to-Noise-Ratio
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
rsnr.signalnoiseratio(rdata.georef, rdata.params,
                      rdata.vars, 55,
                      data2correct=rdata.vars,
                      plot_method=(True)
                      )

# =============================================================================
# Classification of non-meteorological echoes
# =============================================================================
# fdircm = '/home/enchiladaszen/Documents/mygithub/enchilaDaSzen/towerpy/towerpy/eclass/ukmo_cmaps/chenies/chenies_cluttermap_el0.dat'
# # fdircm = '/home/dsanchez/codes/github/towerpy/towerpy/eclass/ukmo_cmaps/chenies/chenies_cluttermap_el0.dat'
# # # lp = 223 sp = 191, ldr = 113, noCM = 159
# rnme = tp.eclass.nme.NME_ID(rsnr)
# rnme.clutter_id(rdata.georef, rdata.params, rdata.vars, binary_class=0,
#                 min_snr=rsnr.min_snr,
#                 # clmap=np.loadtxt(fdircm),
#                 data2correct=rdata.vars,
#                 plot_method=True
#                 )

# =============================================================================
# Generates polarimetric profiles of polarimetric variables
# =============================================================================
rprofs = tp.profs.polprofs.PolarimetricProfiles(rdata)
rprofs.pol_vps(rdata.georef, rdata.params, rsnr.vars, stats=True)

tp.datavis.rad_display.plot_polarimetric_profiles(rdata.params, 
                                                  rprofs.georef['profiles_height [km]'],
                                                  rprofs.vps,
                                                   # mlyr=rmlyr
                                                  )

#%%
# =============================================================================
# ML detection
# =============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_detection(rprofs, min_h=1.1, comb_id=26,
                   plot_method=True
                   )
#%%
# =============================================================================
# ZDR calibration
# =============================================================================
rcalzdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rcalzdr.offsetdetection_vps(pol_profs=rprofs, mlyr=rmlyr, min_h=1.1,
                            # zhmin=5, zhmax=35, rhvmin=0.98, minbins=2
                            plot_method=True, rad_georef=rdata.georef,
                            rad_params=rdata.params, rad_vars=rsnr.vars)

#%%
# =============================================================================
# PhiDP offset detection
# =============================================================================
rcalpdp = tp.calib.calib_phidp.PhiDP_Calibration(rdata)
rcalpdp.offsetdetection_vps(pol_profs=rprofs, mlyr=rmlyr,
                            plot_method=True, rad_georef=rdata.georef,
                            rad_params=rdata.params, rad_vars=rsnr.vars)

# [i.phidp_offset for i in rphioffx]


#%%
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors

# def plot_PHiDP_offsetcorrection(rad_georef, rad_params, pdp, pdp_mean,
#                                 cmap='tpylsc_dbu_w_rd'):
#     fig, ax = plt.subplots(figsize=(8, 8),
#                            subplot_kw={'projection': 'polar', })
#     ax.set_theta_direction(-1)
#     dtdes1 = f"{rad_params['elev_ang [deg]']:{2}.{3}} Deg."
#     dtdes2 = f"{rad_params['datetime']:%Y-%m-%d %H:%M:%S}"
#     ptitle = dtdes1 + dtdes2
#     ax.set_title(ptitle, fontsize=16)
#     ax.grid(color='gray', linestyle=':')
#     ax.set_theta_zero_location('N', offset=0)
#     # =========================================================================
#     # Plot the PhiDP values at each azimuth
#     # =========================================================================
#     ax.scatter((np.ones_like(pdp.T) * [rad_georef['azim [rad]']]).T,
#                pdp, s=5, c=pdp, cmap=cmap,
#                norm=colors.SymLogNorm(linthresh=.01, linscale=.01, base=2,
#                                       vmin=pdp_mean.mean()-3,
#                                       vmax=pdp_mean.mean()+3,
#                                       # vmin=(round(pdp_mean.mean()/10)*10)-20,
#                                       # vmax=(round(pdp_mean.mean()/10)*10)+20,
#                                       ),
#                label=r'$\Phi_{DP}}$')
#     # =========================================================================
#     # Plot the PhiDP mean value of each azimuth
#     # =========================================================================
#     ax.plot(rad_georef['azim [rad]'], pdp_mean, c='tab:green', linewidth=2,
#             ls='', marker='s', markeredgecolor='g', alpha=0.4,
#             label=r'$\overline{\Phi_{DP}}$')
#     # =========================================================================
#     # Plot the PhiDP offset
#     # =========================================================================
#     ax.plot(rad_georef['azim [rad]'], np.full(rad_georef['azim [rad]'].shape,
#                                               pdp_mean.mean()),
#             c='k', linewidth=2.5, label=r'$\Phi_{DP}}$ offset')
#     # ax.set_rorigin(65)
#     # ax.set_rscale('symlog',
#     #               linthresh=(-80, -60),
#     #                 linscale=.01
#     #               )

#     of1 = 10
#     ax.set_ylim([pdp_mean.mean()-of1, pdp_mean.mean()+of1])
#     ax.set_thetagrids(np.arange(0, 360, 90))
#     ax.xaxis.grid(ls='-')
#     ax.tick_params(axis='both', labelsize=14)
#     ax.set_rlabel_position(-45)
#     # ax.set_yticks(np.arange(round(pdp_mean.mean()/10)*10-of1,
#     #                         round(pdp_mean.mean()/10)*10+of1+1,
#     #                         10))
#     angle = np.deg2rad(67.5)
#     ax.legend(fontsize=15, loc="lower left",
#               bbox_to_anchor=(.58 + np.cos(angle)/2, .4 + np.sin(angle)/2))
#     ax.axes.set_aspect('equal')
#     plt.tight_layout()

# # cmap = ''
# # cmap = 'tpylc_grad_tec_r'

# # rad_georef, rad_params, rad_vars = rdata.georef, rdata.params, rdata.vars
# var ='ZDR [dB]'
# var ='PhiDP [deg]'
# pdp = np.array([i[13:20] for i in rdata.vars[var]])
# pdpm = np.array([np.nanmean(i) for i in pdp])

# plot_PHiDP_offsetcorrection(rdata.georef, rdata.params, pdp, pdpm,
#                             # cmap=cmap
#                             )

#%%
tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                    rdata.vars,
                                    # rsnr.vars,
                                    # rnme.vars,
                                    )

#%%
tp.datavis.rad_display.plot_ppi(rdata.georef,
                                       rdata.params,
                                       rdata.vars,
                                        # var2plot='V [m/s]',
                                        # var2plot='ZDR [dB]',
                                        # var2plot='Rainfall [mm/hr]',
                                        # var2plot='PhiDP [deg]',
                                        var2plot='rhoHV [-]',
                                        # ucmap=cm[1]
                                       # var2plot='KDP [deg/km]',
                                        # var2plot='PhiDP [deg]',
                                        # vars_bounds={'KDP [deg/km]': (-0.5, 3, 15)}
                                        )
#%%
tp.datavis.rad_display.plot_ppi(rdata.georef,
                                        rdata.params,
                                        rqpe.r_z,
                                        # rqpe.r_ah,
                                        # rqpe.r_z_zdr,
                                        # rqpe.r_z_zdrl,
                                        # rqpe.r_kdp,
                                        # rqpe.r_kdp_zdr,
                                        var2plot='Rainfall [mm/hr]',
                                        # ucmap=cm[3]
                                        )
#%%
tp.datavis.rad_interactive.ppi_base(rdata.georef,
                                    rdata.params,
                                    rdata.vars,
                                    # rclassx.snr_class['vars'],
                                    # rnme.vars,
                                    # rattc.vars,
                                    # coord_sys='rect',
                                    # var2plot='ZDR [dB]',
                                    var2plot='rhoHV [-]',
                                    # ucmap=cm[0],
                                    # ylims={'ZH [dBZ]': (0, 50)},
                                    # vars_bounds={'rhoHV [-]': (0.3, .9,1)}
                                    # mlboundaries=rmlyrx
                                    )
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
