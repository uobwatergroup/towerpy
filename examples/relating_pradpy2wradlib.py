#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:49:07 2021

@author: ds17589
"""

import os
import numpy as np
import wradlib as wrl
import towerpy as tp

# =============================================================================
# This example is part of the ωradlib documentation:
# https://docs.wradlib.org/en/stable/notebooks/classify/wradlib_fuzzy_echo_classify.html 
# =============================================================================

# Setting the file paths 
os.environ['WRADLIB_DATA'] = '/User/documents/wradlib-data/'
rhofile = wrl.util.get_wradlib_data_file('TAG-20120801-140046-02-R.nc')
phifile = wrl.util.get_wradlib_data_file('TAG-20120801-140046-02-P.nc')
reffile = wrl.util.get_wradlib_data_file('TAG-20120801-140046-02-Z.nc')
dopfile = wrl.util.get_wradlib_data_file('TAG-20120801-140046-02-V.nc')
zdrfile = wrl.util.get_wradlib_data_file('TAG-20120801-140046-02-D.nc')
mapfile = wrl.util.get_wradlib_data_file('TAG_cmap_sweeps_0204050607.hdf5')

# Using wradlib to read the data (radar moments and static clutter map)
dat = {}
dat['rho'], attrs_rho = wrl.io.read_edge_netcdf(rhofile)
dat['phi'], attrs_phi = wrl.io.read_edge_netcdf(phifile)
dat['ref'], attrs_ref = wrl.io.read_edge_netcdf(reffile)
dat['dop'], attrs_dop = wrl.io.read_edge_netcdf(dopfile)
dat['zdr'], attrs_zdr = wrl.io.read_edge_netcdf(zdrfile)
dat['map']            = wrl.io.from_hdf5(mapfile)[0][0]

#%%

# =============================================================================
# Using pradpy
# =============================================================================

# Creating an empty pradpy object to store the wradlib data
rdata = tp.io.empty.Rad_scan('test', site_name='UK')
rdata.empty_radobj(nrays=dat["ref"].shape[0], ngates=dat["ref"].shape[1])
rdata.params['altitude [m]'] = attrs_ref['Height']
rdata.params['beamwidth [deg]'] = 1.
rdata.params['elev_ang [deg]'] = 1.
rdata.params['radarconstant'] = 70. # Assumed

# As these data do not provide elev/azim info, we assume these params to
# define the relative coordinates
rdata.empty_radgeo(elev=np.deg2rad(np.full(dat["ref"].shape[0], 0.5)),
                     azim=np.deg2rad(np.arange(360)), gateres=1000)

rdata.vars['Zh [dBZ]'] = dat["ref"].astype(np.float64)
rdata.vars['Zdr [dB]'] = dat["zdr"].astype(np.float64)
rdata.vars['RhoHV [-]'] = dat["rho"].astype(np.float64)
rdata.vars['PhiDP [deg]'] = dat["phi"].astype(np.float64)
rdata.vars['V [m/s]'] = dat["dop"].astype(np.float64)

# Plotting the PPI
tp.datavis.plots_polrad.plot_raw_ppi(rdata.vars, rdata.georef,
                                     polvar='RhoHV [-]')

# Computing the SNR
rclass = tp.eclass.echoes_class.EchoesClass(rdata)
rclass.signalnoiseratio(rdata, minsnr=35, rvars=rdata.vars)
tp.datavis.plots_polrad.plot_raw_snr_data(rdata.vars, rdata.georef,
                                          rclass.snr_class)

# Clutter ID
rclass.clutterclass(rdata,
                    binary_class=159,
                    datatocc=rdata.vars
                    )
tp.datavis.plots_polrad.plot_ccpoldata(rdata.params, rdata.georef,
                                        rclass.ccdata['vars'],
                                        cclass_map=rclass.ccdata['cclass'],
                                        # clutter_map=rclass.ccdata['clmap']
                                        )

