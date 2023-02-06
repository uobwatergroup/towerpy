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
fdir = f'/media/enchiladaszen/enchiladasz/safe/bristolphd/data4phd/radar_datasets/{rsite}/y2020/spel8/'
# fdir = f'/run/media/dsanchez/enchiladasz/safe/bristolphd/data4phd/radar_datasets/{rsite}/y2020/lpel0/'
fname = f'metoffice-c-band-rain-radar_{rsite}_202010032106_raw-dual-polar-augzdr-sp-el8.dat'

rdata = tp.io.ukmo.Rad_scan(fdir+fname, rsite)
rdata.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])
rdata.vars['PhiDP [deg]'] *= -1

tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rdata.vars,
                                # var2plot='rhoHV [-]'
                                )


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
# Generates polarimetric profiles of polarimetric variables
# =============================================================================
rprofs = tp.profs.polprofs.PolarimetricProfiles(rdata)
rprofs.pol_vps(rdata.georef, rdata.params, rsnr.vars, stats=True)

tp.datavis.rad_display.plot_radprofiles(rdata.params, 
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
