#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:39:20 2021

@author: ds17589
"""

import numpy as np
import towerpy as tp
import cartopy.crs as ccrs
# from wradlib.dp import process_raw_phidp_vulpiani as kdpvpi

# =============================================================================
# Read polar radar data
# =============================================================================
rsite = 'chenies'
fdir = f'../datasets/{rsite}/y2020/lpel0/'
fname = f'metoffice-c-band-rain-radar_{rsite}_202010032105_raw-dual-polar-augzdr-lp-el0.dat'

rdata = tp.io.ukmo.Rad_scan(fdir+fname, rsite)
rdata.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])

# Plot the radar PPI
tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rdata.vars)

# %%
# =============================================================================
# Compute the Signal-to-Noise-Ratio
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
rsnr.signalnoiseratio(rdata.georef, rdata.params, rdata.vars, min_snr=35,
                      data2correct=rdata.vars, plot_method=(True))

# %%
# =============================================================================
# Classification of non-meteorological echoes
# =============================================================================
clmap = f'../towerpy/eclass/ukmo_cmaps/{rsite}/chenies_cluttermap_el0.dat'

rnme = tp.eclass.nme.NME_ID(rsnr)
rnme.clutter_id(rdata.georef, rdata.params, rdata.vars, binary_class=223,
                min_snr=rsnr.min_snr, clmap=np.loadtxt(clmap),
                data2correct=rdata.vars, plot_method=True)
# %%
# =============================================================================
# Melting layer allocation
# =============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_bottom = 2

# %%
# =============================================================================
# ZDR calibration
# =============================================================================
rczdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rczdr.offset_correction(rnme.vars['ZDR [dB]'],
                        zdr_offset=-0.28, data2correct=rnme.vars)
# %%
# =============================================================================
# Attenuation correction
# =============================================================================
rattc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
rattc.zh_correction(rdata.georef, rdata.params, rczdr.vars,
                    rnme.nme_classif['classif'], mlyr_b=rmlyr.ml_bottom,
                    attc_method='ABRI', pdp_pxavr_azm=1, pdp_dmin=10,
                    pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']))

rattc.zdr_correction(rdata.georef, rdata.params, rczdr.vars, rattc.vars,
                     rnme.nme_classif['classif'], mlyr_b=rmlyr.ml_bottom,
                     rhv_thld=0.98, minbins=10, mov_avrgf_len=5, p2avrf=3,
                     beta_alpha_ratio=.2)

tp.datavis.rad_display.plot_attcorrection(rdata.georef, rdata.params,
                                          rczdr.vars,
                                          rattc.vars)
# %%
# =============================================================================
# KDP Derivation
# =============================================================================
# rkdpv = {}

# # KDP Vulpiani
# kdp_vulp = kdpvpi(rattc.vars['PhiDP [deg]'], winlen=11,
#                   dr=rdata.params['gateres [m]']/1000)
# rkdpv['PhiDP [deg]'] = kdp_vulp[0]
# rkdpv['KDP [deg/km]'] = kdp_vulp[1]
# rkdpv['PhiDP [deg]'][np.isnan(rattc.vars['ZH [dBZ]'])] = np.nan
# rkdpv['KDP [deg/km]'][np.isnan(rattc.vars['ZH [dBZ]'])] = np.nan

# tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rattc.vars,
#                                 var2plot='KDP [deg/km]',
#                                 vars_bounds={'KDP [deg/km]': (-1, 3, 17)})

# tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rkdpv,
#                                 var2plot='KDP [deg/km]',
#                                 vars_bounds={'KDP [deg/km]': (-1, 3, 17)})

# %%

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
# rqpe.kdp_to_r(rkdpv['KDP [deg/km]'], bh_km=rdata.georef['beam_height [km]'],
#               mlyr_b=rmlyr.ml_bottom)
# rqpe.kdp_zdr_to_r(rkdpv['KDP [deg/km]'], rattc.vars['ZDR [dB]'],
#                   mlyr_b=rmlyr.ml_bottom,
#                   bh_km=rdata.georef['beam_height [km]'])

# %%
# =============================================================================
# Plots
# =============================================================================
# Plot cone coverage
tp.datavis.rad_display.plot_cone_coverage(rdata.georef, rdata.params,
                                          rsnr.vars,
                                          var2plot=None,
                                          # zlims=[0, 8]
                                          )
# Plot the radar data in a map
rdata.georef['xgrid_proj'] = rdata.georef['xgrid'] + rdata.params['easting [km]']
rdata.georef['ygrid_proj'] = rdata.georef['ygrid'] + rdata.params['northing [km]']

rdata.georef['xgrid_proj'] *= 1000
rdata.georef['ygrid_proj'] *= 1000

tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rqpe.r_z,
                                data_proj=ccrs.OSGB(approx=False),
                                cpy_feats={'status': True})
# Plot all the radar variables
tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rdata.vars)

# Plot an interactive PPI explorer
tp.datavis.rad_interactive.ppi_base(rdata.georef, rdata.params, rnme.vars,
                                    var2plot='rhoHV [-]')
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
# %%
# ppiexplorer.savearray2binfile(file_name='metoffice-c-band-rain-radar_jersey_202010030735_raw-dual-polar-augzdr-lp-el0.dat',
#                               dir2save='/home/enchiladaszen/Documents/radar_trials/metoffice/')
