#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:58:25 2021

@author: enchiladaszen
"""

import numpy as np
import towerpy as tp
from pathlib import Path

# =============================================================================
# Reads polar radar data
# =============================================================================
rdata = tp.io.ukmo_wr.UKMO_Rad('dir/fname.dat')
rdata.read_rawrad(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])
tp.datavis.plots_polrad.plot_raw_ppi(rdata.vars, rdata.georef)

# =============================================================================
# Computes the Signal-to-Noise-Ratio 
# =============================================================================
rclass = tp.eclass.echoes_class.EchoesClass(rdata)
# #ldr = 28 lp = 35, sp = 55
rclass.signalnoiseratio(rdata, minsnr=35, rvars=rdata.vars)
tp.datavis.plots_polrad.plot_raw_snr_data(rdata.vars, rdata.georef,
                                          rclass.snr_class)

# =============================================================================
# Classification of non-meteorological echoes 
# =============================================================================
# lp = 223, vp = 191, ldr = 113, noCM = 159
rclass.clutterclass(rdata,
                    # path_mfs='pymor/eclass/mfs/',
                    binary_class=223,
                    clmap=np.loadtxt('clmap.dat'),
                    # classid={'meteorological_echoes':False, 'clutter':True,
                    #          'noise':np.nan},
                    datatocc=rdata.vars)
tp.datavis.plots_polrad.plot_ccpoldata(rdata.params, rdata.georef,
                                        rclass.ccdata['vars'],
                                        cclass_map=rclass.ccdata['cclass'],)
                                        # clutter_map=rclass.ccdata['clmap'])

# =============================================================================
# Melting layer definition
# =============================================================================
rmlyr = tp.ml.ml_detection.MeltingLayer(rdata)
rmlyr.melting_layer = {}
rmlyr.melting_layer['ml_top'] = 2.5
rmlyr.melting_layer['ml_bottom'] = 1.8

# =============================================================================
# ZDR offset correction
# =============================================================================
rcal = tp.calib.cal_zdr.ZDR_Calibration(rdata)
rcal.offset_correction(rclass.ccdata['vars']['Zdr [dB]'],
                       zdr_offset=-0.482)
rclass.ccdata['vars']['Zdr [dB](offsetcorr)'] = rcal.Zdr_dB_oc
tp.datavis.plots_polrad.plot_ccpoldata(rdata.params, rdata.georef,
                                        rclass.ccdata['vars'],
                                        mlboundaries=rmlyr.melting_layer
                                        )

# =============================================================================
# Attenuation correction
# =============================================================================
# ratc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
# ratc.zh_correction(rdata.georef, rclass.ccdata['vars'],
#                    rclass.ccdata['cclass'],
#                    mlvl=rmlyr.melting_layer['ml_bottom']+0.75,
#                    alg4atc='ABRI', p2apdpr= 7, p2apdpa= 1, minpdpt= 20)
# ratc.zdr_correction(rdata.georef, rclass.ccdata['vars'], 
#                     ratc.res_zhcorrection, rclass.ccdata['cclass'],
#                     mlvl=rmlyr.melting_layer['ml_bottom']+0.75,
#                     rhv_thld=0.98, minbins= 10, mov_avrgf_len=5, p2avrf=3)
# tp.datavis.plots_polrad.plot_attcorrection(rdata.params, rdata.georef,
#                                 rclass.ccdata['vars'],
#                                 ratc.res_zhcorrection | ratc.res_zdrcorrection)

tp.datavis.rtool.interactiveradarplot(rdata.params, rdata.georef,
                                  rclass.snr_class['vars'],
                                  coord='rect', )
radexpvis = tp.datavis.rtool.PointBrowser()
