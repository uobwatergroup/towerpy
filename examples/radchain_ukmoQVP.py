#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:39:20 2021

@author: ds17589
"""

import numpy as np
import towerpy as tp
import cartopy.crs as ccrs

# =============================================================================
# Read polar radar data
# =============================================================================
rsite = 'chenies'
fdir = f'../datasets/{rsite}/y2020/spel4/'
fname = f'metoffice-c-band-rain-radar_{rsite}_202010030730_raw-dual-polar-augzdr-sp-el4.dat'

rdata = tp.io.ukmo.Rad_scan(fdir+fname, rsite)
rdata.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])

# %%
# =============================================================================
# Compute the Signal-to-Noise-Ratio
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
rsnr.signalnoiseratio(rdata.georef, rdata.params, rdata.vars, min_snr=55,
                      data2correct=rdata.vars, plot_method=(True))

# %%
# =============================================================================
# Generate polarimetric profiles of polarimetric variables
# =============================================================================
rprofs = tp.profs.polprofs.PolarimetricProfiles(rdata)
rprofs.pol_qvps(rdata.georef, rdata.params, rsnr.vars, stats=True)

tp.datavis.rad_display.plot_radprofiles(rdata.params,
                                        rprofs.georef['profiles_height [km]'],
                                        rprofs.qvps)

# %%
# =============================================================================
# ML detection
# =============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_detection(rprofs, min_h=1.1, comb_id=14,
                   plot_method=True
                   )
# %%
# =============================================================================
# ZDR offset detection
# =============================================================================
rcalzdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rcalzdr.offsetdetection_qvps(pol_profs=rprofs, mlyr=rmlyr, min_h=0.25)

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

tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rsnr.vars,
                                data_proj=ccrs.OSGB(approx=False),
                                cpy_feats={'status': True})
# Plot all the radar variables
tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rdata.vars)

# Plot an interactive PPI explorer
tp.datavis.rad_interactive.ppi_base(rdata.georef, rdata.params, rsnr.vars,
                                    var2plot='rhoHV [-]')
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()