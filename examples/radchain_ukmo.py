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
rsite = 'cobbacombe'
fdir = f'/media/enchiladaszen/enchiladasz/safe/bristolphd/data4phd/radar_datasets/{rsite}/y2020/lpel0/'
fdir = f'/run/media/dsanchez/enchiladasz/safe/bristolphd/data4phd/radar_datasets/{rsite}/y2020/lpel0/'

rdata = tp.io.ukmo.Rad_scan(fdir+f'metoffice-c-band-rain-radar_{rsite}_'
                            '202010031735_raw-dual-polar-augzdr-lp-el0.dat',
                            rsite)
rdata.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])

# =============================================================================
# Computes the Signal-to-Noise-Ratio
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
rsnr.signalnoiseratio(rdata.georef, rdata.params,
                      rdata.vars, 35,
                      data2correct=rdata.vars,
                      # plot_method=(True)
                      )

# =============================================================================
# Classification of non-meteorological echoes
# =============================================================================
# fdircm = '/home/enchiladaszen/Documents/mygithub/enchilaDaSzen/towerpy/towerpy/eclass/ukmo_cmaps/chenies/chenies_cluttermap_el0.dat'

rnme = tp.eclass.nme.NME_ID(rsnr)
rnme.clutter_id(rdata.georef, rdata.params, rdata.vars, binary_class=159,
                min_snr=rsnr.min_snr,
                # clmap=np.loadtxt(fdircm),
                data2correct=rdata.vars,
                # plot_method=True
                )

# =============================================================================
# Melting layer allocation using polarimetric profiles
# =============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_bottom = 2

# =============================================================================
# ZDR calibration
# =============================================================================
rczdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rczdr.offset_correction(rnme.vars['ZDR [dB]'],
                        zdr_offset=-1, data2correct=rnme.vars)
#%%
# =============================================================================
# Attenuation correction
# =============================================================================
rattc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
rattc.zh_correction(rdata.georef, rdata.params, rczdr.vars,
                    rnme.nme_classif['classif'],
                    mlyr_b=rmlyr.ml_bottom,
                    attc_method='ABRI',
                    pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']),
                    pdp_pxavr_azm=1, pdp_dmin=10)

rattc.zdr_correction(rdata.georef, rdata.params, rczdr.vars,
                     rattc.vars,
                     rnme.nme_classif['classif'],
                     # mlyr_b=rmlyr.ml_bottom,
                     mlyr_b=rmlyr.ml_bottom,
                     rhv_thld=0.98, minbins=10, mov_avrgf_len=5, p2avrf=3,
                     beta_alpha_ratio=.2
                     )

# tp.datavis.rad_display.plot_attcorrection(rdata.georef, rdata.params,
#                                           rczdr.vars,
#                                           rattc.vars,
#                                           )
#%%
# =============================================================================
# KDP Derivation
# =============================================================================
rkdpv = {}

# KDP Vulpiani
rkdpv['PhiDP [deg]'], rkdpv['KDP [deg/km]'] = wrl.dp.process_raw_phidp_vulpiani(rattc.vars['PhiDP [deg]'],
                                                                                dr=rdata.params['gateres [m]']/1000,
                                                                                winlen=9)

rkdpv['PhiDP [deg]'][np.isnan(rattc.vars['ZH [dBZ]'])] = np.nan
rkdpv['KDP [deg/km]'][np.isnan(rattc.vars['ZH [dBZ]'])] = np.nan

# =============================================================================
# Rainfall estimation
# =============================================================================
rqpe = tp.qpe.qpe_algs.RadarQPE(rdata)

rqpe.z_to_r(rattc.vars['ZH [dBZ]'], a=200, b=1.6, mlyr_b=rmlyr.ml_bottom,
            bh_km=rdata.georef['beam_height [km]'])
rqpe.z_to_r(rnme.vars['ZH [dBZ]'], a=200, b=1.6, mlyr_b=rmlyr.ml_bottom,
            bh_km=rdata.georef['beam_height [km]'])
rqpe.ah_to_r(rattc.vars['AH [dB/km]'], mlyr_b=rmlyr.ml_bottom,
             bh_km=rdata.georef['beam_height [km]'])
rqpe.z_zdr_to_r1(rattc.vars['ZH [dBZ]'], rattc.vars['ZDR [dB]'],
                 mlyr_b=rmlyr.ml_bottom,
                 bh_km=rdata.georef['beam_height [km]'])
rqpe.z_zdr_to_r2(rattc.vars['ZH [dBZ]'], rattc.vars['ZDR [dB]'],
                 a=0.0121, b=0.822, c=-1.7486, mlyr_b=rmlyr.ml_bottom,
                 bh_km=rdata.georef['beam_height [km]'])
rqpe.kdp_to_r(rkdpv['KDP [deg/km]'], bh_km=rdata.georef['beam_height [km]'],
              mlyr_b=rmlyr.ml_bottom)
rqpe.kdp_zdr_to_r(rkdpv['KDP [deg/km]'], rattc.vars['ZDR [dB]'],
                  mlyr_b=rmlyr.ml_bottom,
                  bh_km=rdata.georef['beam_height [km]'])

#%%
# =============================================================================
# Plots
# =============================================================================
# Plot cone coverage
tp.datavis.rad_display.plot_cone_coverage(rdata.georef, rdata.params,
                                          rsnr.vars,
                                          var2plot=None,
                                          # zlims=[0, 8]
                                          )
# tp.datavis.rad_display.plot_snr(rdata.georef, rdata.params,
                                # rsnr.snr_class, proj='polar')
#%%
rdata.georef['xgrid_proj'] = rdata.georef['xgrid'] + rdata.params['easting [km]']
rdata.georef['ygrid_proj'] = rdata.georef['ygrid'] + rdata.params['northing [km]']

rdata.georef['xgrid_proj'] *= 1000
rdata.georef['ygrid_proj'] *= 1000

tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                        rqpe.r_z,
                                        # rqpe.r_ah,
                                        # rqpe.r_z_zdr,
                                        # rqpe.r_z_zdrl,
                                        # rqpe.r_kdp,
                                        # rqpe.r_kdp_zdr,
                                        # rdata.vars,
                                        # rnme.vars,
                                        # var2plot='Rainfall [mm/hr]',
                                        # var2plot='PhiDP [deg]',
                                        # var2plot='rhoHV [-]',
                                        # vars_bounds={'rhoHV [-]': (0.3, .9, 1)},
                                        data_proj=ccrs.OSGB(approx=False),
                                        cpy_feats={'status': True},
                                        # xlims=[-6, 1.5], ylims=[52, 46.5],
                                        )
#%%
tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params,
                                    rdata.vars,
                                    # rsnr.vars,
                                    # rnme.vars,
                                    )

tp.datavis.rad_display.plot_ppi(rdata.georef,
                                        rdata.params,
                                        # rdata.vars,
                                        # rattc.vars,
                                        rkdpv,
                                        # var2plot='rhoHV [-]',
                                        var2plot='PhiDP [deg]',
                                        # var2plot='KDP [deg/km]',
                                        # ucmap=cm[0],
                                        # var2plot='PhiDP [deg]',
                                        # vars_bounds={'KDP [deg/km]': (-.25, 0.75, 17)}
                                        )
#%%
# tp.datavis.rad_display.plot_ppi(rdata.georef,
#                                        rdata.params,
#                                        rdata.vars,
#                                         # var2plot='V [m/s]',
#                                         # var2plot='ZDR [dB]',
#                                         # var2plot='Rainfall [mm/hr]',
#                                         # var2plot='PhiDP [deg]',
#                                         var2plot='rhoHV [-]',
#                                         # ucmap=cm[1]
#                                        # var2plot='KDP [deg/km]',
#                                         # var2plot='PhiDP [deg]',
#                                         # vars_bounds={'KDP [deg/km]': (-0.5, 3, 15)}
#                                         )
#%%
# tp.datavis.rad_display.plot_ppi(rdata.georef,
#                                         rdata.params,
#                                         rqpe.r_z,
#                                         # rqpe.r_ah,
#                                         # rqpe.r_z_zdr,
#                                         # rqpe.r_z_zdrl,
#                                         # rqpe.r_kdp,
#                                         # rqpe.r_kdp_zdr,
#                                         var2plot='Rainfall [mm/hr]',
#                                         # ucmap=cm[3]
#                                         )
#%%
tp.datavis.rad_interactive.ppi_base(rdata.georef,
                                    rdata.params,
                                    # rdata.vars,
                                    # rclassx.snr_class['vars'],
                                    rnme.vars,
                                    # rattc.vars,
                                    # coord_sys='rect',
                                    # var2plot='alpha',
                                    # var2plot='PhiDP* [deg]',
                                    # var2plot='beta [-]',
                                    # var2plot='rhoHV [-]',
                                    # ucmap=cm[0],
                                    # ylims={'ZH [dBZ]': (0, 50)},
                                    # vars_bounds={'rhoHV [-]': (0.3, .9,1)}
                                    # mlboundaries=rmlyrx
                                    )
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
#%%
# ppiexplorer.savearray2binfile(file_name='metoffice-c-band-rain-radar_jersey_202010030735_raw-dual-polar-augzdr-lp-el0.dat',
#                               dir2save='/home/enchiladaszen/Documents/radar_trials/metoffice/')
#%%
# import pickle

# with open('/home/enchiladaszen/Documents/radar_trials/metoffice/'+'metoffice-c-band-rain-radar_jersey_202010030735_raw-dual-polar-augzdr-lp-el0.dat.tpy', 'rb') as handle:
#     coordobj = pickle.load(handle)

# b = coordobj.coord_lst
