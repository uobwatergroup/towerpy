#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:10:43 2023

@author: dsanchez
"""

import os
import numpy as np
# import pickle
from zoneinfo import ZoneInfo
import datetime as dt
import towerpy as tp
from tqdm import tqdm


wdir = '/mnt/disk/users/ds17589/'
wdir = '/media/enchiladaszen/enchiladasz/safe/bristolphd/data4phd/radar_datasets/'
# wdir = '/run/media/dsanchez/enchiladasz/safe/bristolphd/data4phd/radar_datasets/'


# minsnr -> ldr = 28 lp = 35, sp = 55
# bclass -> lp = 255 sp = 191, ldr = 113, noCM = 159
rparams = [
    # {'site_name': 'castor-bay', 'minsnr': 55, 'bclass': 159, 'minh90': 1.1},
    {'site_name': 'chenies', 'minsnr': 55, 'bclass': 159, 'minh90': 1.1},
    # {'site_name': 'clee-hill', 'minsnr': 54, 'bclass': 159, 'minh90': 1.45},
    # {'site_name': 'cobbacombe', 'minsnr': 55, 'bclass': 159, 'minh90': 1.25},
    # {'site_name': 'crug-y-gorrllwyn', 'minsnr': 55, 'bclass': 159, 'minh90': 1.25},
    # {'site_name': 'deanhill', 'minsnr': 54, 'bclass': 159, 'minh90': 1.1},
    # {'site_name': 'druima-starraig', 'minsnr': 55, 'bclass': 159, 'minh90': 1.1},
    # {'site_name': 'dudwick', 'minsnr': 56, 'bclass': 159, 'minh90': 1.1},
    # {'site_name': 'hameldon-hill', 'minsnr': 55, 'bclass': 159, 'minh90': 1.3},
    # {'site_name': 'high-moorsley', 'minsnr': 55, 'bclass': 159, 'minh90': 1.1},
    # {'site_name': 'holehead', 'minsnr': 55, 'bclass': 159, 'minh90': 1.4},
    # {'site_name': 'ingham', 'minsnr': 55, 'bclass': 159, 'minh90': 1.1},
    # {'site_name': 'jersey', 'minsnr': 55, 'bclass': 159, 'minh90': 1.1},
    # {'site_name': 'munduff-hill', 'minsnr': 55, 'bclass': 159, 'minh90': 1.3},
    # {'site_name': 'predannack', 'minsnr': 55, 'bclass': 159, 'minh90': 1.1},
    # {'site_name': 'thurnham', 'minsnr': 55, 'bclass': 159, 'minh90': 1.1},
    # {'site_name': 'wardon-hill', 'minsnr': 55, 'bclass': 159, 'minh90': 1.1},
    ]


rdirs = [f'{wdir}{rs["site_name"]}/y2020/spel8/'
         for rs in rparams]

fnames = [sorted(os.listdir(dirs)) for dirs in rdirs]

fnames = [rdirs[d]+j for d, i in enumerate(fnames) for j in i
          if '20201003' in j]  # spel8


#%%
# =============================================================================
# Reads radar data in polar format
# =============================================================================
rdata = [tp.io.ukmo.Rad_scan(i, rparams[0]['site_name']) for i in fnames]

[i.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]']) for i in rdata]

# for i, j in enumerate(rdata):
#     j.site_name = rdata[i].file_name[rdata[i].file_name.find('ds17589') +
#                                      8:rdata[i].file_name.find('y2020')-1]

#%%
# =============================================================================
# Noise and clutter suppression
# =============================================================================
rsnr = [tp.eclass.snr.SNR_Classif(robj) for robj in rdata]
[robj.signalnoiseratio(rdata[i].georef, rdata[i].params, rdata[i].vars,
                       min_snr=next(item['minsnr']
                                    for item in rparams
                                    if item['site_name'] == robj.site_name),
                       data2correct=rdata[i].vars)
 for i, robj in enumerate(tqdm(rsnr, desc='rsnr_towerpy'))]
# [robj.clutterclass(rdatax[i], binary_class=64, datatocc=rdatax[i].vars)
# for i, robj in enumerate(tqdm(rclassx))]
# rclass = [tp.eclass.echoes_class.EchoesClass(rd) for rd in rdata]

# for i, j in enumerate(rclass):
#     j.site_name = rdata[i].site_name

# [v.signalnoiseratio(rdata[i],
#                     minsnr=next(item['minsnr']
#                                 for item in rparams
#                                 if item['site_name'] == v.site_name),
#                     rvars=rdata[i].vars)
#  for i, v in enumerate(rclass)]

# [v.clutterclass(rdata[i],
#                 binary_class=next(item['bclass']
#                                   for item in rparams
#                                   if item['site_name'] == v.site_name),
#                 datatocc=rdata[i].vars)
#  for i, v in enumerate(rclass)]
#%%
# =============================================================================
# Generates polarimetric profiles of polarimetric variables
# =============================================================================
# rprofs = [tp.profs.polprofs.PolarimetricProfiles(rd) for rd in rdata]
# for i, j in enumerate(rprofs):
#     j.site_name = rdata[i].site_name
# [rprofs[i].radarvps(rdata[i], rclass[i].ccdata['vars'], stats=True)
 # for i, v in enumerate(rclass)]

rprofs = [tp.profs.polprofs.PolarimetricProfiles(rd) for rd in rdata]
[robj.pol_vps(rdata[i].georef, rdata[i].params, rsnr[i].vars, stats=True)
 for i, robj in enumerate(tqdm(rprofs, desc='rprofs_towerpy'))]

#%%
# =============================================================================
# ML detection
# =============================================================================
rmlyr = [tp.ml.mlyr.MeltingLayer(rd) for rd in rdata]
[rmlyr[i].ml_detection(v,
                       min_h=next(item['minh90']
                                  for item in rparams
                                  if item['site_name'] == v.site_name),
                       max_h=5,
                       comb_id=26,
                       # param_k=0.08, param_w=0.75,
                       # plot_method=True
                       ) for i, v in enumerate(rprofs)]
#%%
# =============================================================================
# ZDR calibration
# =============================================================================
# rcalzdr = [tp.calib.cal_zdr.ZDR_Calibration(rd) for rd in rdata]
# for i, j in enumerate(rcalzdr):
#     j.site_name = rdata[i].site_name
# [v.offsetdetection_vps(pol_profs=rprofs[i], mlyr=rmlyr[i],
#                        min_h=next(item['minh90']
#                                   for item in rparams
#                                   if item['site_name'] == v.site_name),
#                        zhmin=5, zhmax=35, rhvmin=0.98, minbins=2)
#  for i, v in enumerate(rcalzdr)]

rcalzdr = [tp.calib.calib_zdr.ZDR_Calibration(rd) for rd in rdata]
[robj.offsetdetection_vps(pol_profs=rprofs[i], mlyr=rmlyr[i],
                          min_h=next(item['minh90']
                                     for item in rparams
                                     if item['site_name'] == robj.site_name),
                          zhmin=5, zhmax=35, rhvmin=0.98, minbins=2)
 for i, robj in enumerate(tqdm(rcalzdr, desc='rcalzdr_towerpy'))]

[robj.zdr_offset for i, robj in enumerate(tqdm(rcalzdr))]

#%%
# =============================================================================
# Save data
# =============================================================================
# Create qperawdata directory and subsequent dirs /y2020/spel8/
# [os.mkdir(f'/home/ds17589/pyscripts/data4phd/qpe/qperawdata/{i["site_name"]}/y2020/spel8/')
#   for i in rparams]

# for i, j in enumerate(rprofs):
#     j.mlyr = rmlyr[i]
#     j.zdrc = rcalzdr[i]
#     j.file_name = j.file_name[j.file_name.find('spel8') +
#                               6:j.file_name.find('.dat')]
#     f2store = open(f'/home/ds17589/pyscripts/data4phd/qpe/qperawdata/{j.site_name}/y2020/spel8/{j.file_name}.pickle', 'wb')
#     pickle.dump(j, f2store)
#     f2store.close()

# =============================================================================
# QPE
# =============================================================================

# tp.qpe.qpe_algs.z_to_r(50, 200, 1.6)

# for i, j in enumerate(rclass):
#     j.ccdata['vars']['Rainfall [mm/hr]'] = tp.qpe.qpe_algs.z_to_r(j.ccdata['vars']['Zh [dBZ]'], 200, 1.6)
 
#%%

# profsvar = {k: np.array([nprof.vps[k] for nprof in rprofs]).T
#             for k in rprofs[0].vps.keys()}
# profshei = np.array([nprof.profs_height_km for nprof in rprofs]).T
# profsdtt = [nprof.scandatetime for nprof in rprofs]
# # profsdtt[:] = [nprof.scandatetime.replace(tzinfo=ZoneInfo('Europe/London')) for nprof in rprofs],
# profsmlt = [mlyr.ml_top if isinstance(mlyr.ml_top, float) else np.nan
#             for mlyr in rmlyr]
# profsmlb = [mlyr.ml_bottom if isinstance(mlyr.ml_bottom, float) else np.nan
#             for mlyr in rmlyr]
# profsstd = {k: np.array([nprof.vps_stats['std_dev'][k]
#                          for nprof in rprofs]).T
#             for k in rprofs[0].vps_stats['std_dev'].keys()}

n = 0
# tp.datavis.rad_display.plot_single_ppi(rdata[n].georef,
#                                        rdata[n].params,
#                                        rdata[n].vars,
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

# tp.datavis.rad_display.plot_polarimetric_profiles(rdata[n].params, 
#                                                   rprofs[n].profs_height_km,
#                                                   rprofs[n].vps,
#                                                   mlyr=rmlyr[n]
#                                                   )

#%%

radb = tp.datavis.rad_interactive.hti_base(rprofs, mlyrs=rmlyr,
                                            stats='std',
                                            # var2plot='ZDR [dB]',
                                            # var2plot='rhoHV [-]',
                                            # var2plot='gradV [dV/dh]',
                                            # var2plot='V [m/s]',
                                            # var2plot='PhiDP [deg]',
                                            # var2plot='V [m/s]',
                                            # mlyrtop=profsmlt,
                                            # mlyrbot=profsmlb,
                                            # profsstat=profsstd,
                                            htiylim=[0, 8],
                                            # ptype='contour',
                                            # contourl='rhoHV [-]',
                                            # ucmap='viridis_r',
                                            # htixlim=[dt.datetime(2018, 3, 9, 13),
                                                        # dt.datetime(2018, 3, 9, 23)],
                                            htixlim=[dt.datetime(2020, 10, 3, 0, 5, tzinfo=ZoneInfo('Europe/London')),
                                                     dt.datetime(2020, 10, 3, 23, 59, tzinfo=ZoneInfo('Europe/London'))],
                                            # vars_bounds={'ZDR [dB]':[-1, -0.5, 17]},
                                            # vars_bounds={'V [m/s]':[-8, 8, 13]},
                                            tz='Europe/London'
                                            )
radexpvis = tp.datavis.rad_interactive.HTI_Int()
radb.on_clicked(radexpvis.hzfunc)