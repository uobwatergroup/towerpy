"""Radar processing chain of UK Met Office radar data."""

import numpy as np
import towerpy as tp
import cartopy.crs as ccrs
# from wradlib.dp import process_raw_phidp_vulpiani as kdpvpi

rsite = 'chenies'
fdir = f'../datasets/{rsite}/y2020/lpel0/'
fname = (f'metoffice-c-band-rain-radar_{rsite}_202010032115_raw-dual-polar-'
         + 'augzdr-lp-el0.dat')

# =============================================================================
# Read polar radar data
# =============================================================================
rdata = tp.io.ukmo.Rad_scan(fdir+fname, rsite)
rdata.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])
rdata.ppi_ukmogeoref()

# Plot all the radar variables
tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rdata.vars)

# %%
# =============================================================================
# Compute the Signal-to-Noise-Ratio
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
rsnr.signalnoiseratio(rdata.georef, rdata.params, rdata.vars, min_snr=35,
                      data2correct=rdata.vars, plot_method=True)

# %%
# =============================================================================
# Classification of non-meteorological echoes
# =============================================================================
clmap = f'../towerpy/eclass/ukmo_cmaps/{rsite}/chenies_cluttermap_el0.dat'
tp.datavis.rad_display.plot_mfs(f'../towerpy/eclass/mfs_cband/')

rnme = tp.eclass.nme.NME_ID(rsnr)
rnme.clutter_id(rdata.georef, rdata.params, rdata.vars, binary_class=223,
                min_snr=rsnr.min_snr, clmap=np.loadtxt(clmap),
                data2correct=rdata.vars, plot_method=True)
# %%
# =============================================================================
# Melting layer allocation
# =============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_top = 1.8
rmlyr.ml_thickness = 0.85
rmlyr.ml_bottom = rmlyr.ml_top - rmlyr.ml_thickness

rmlyr.ml_ppidelimitation(rdata.georef, rdata.params, rsnr.vars)

# Plot rhoHV and the ML
tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rnme.vars,
                                var2plot='rhoHV [-]', mlyr=rmlyr)
# %%
# =============================================================================
# ZDR calibration
# =============================================================================
rczdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rczdr.offset_correction(rnme.vars['ZDR [dB]'],
                        zdr_offset=-0.27,
                        data2correct=rnme.vars)
# %%
# =============================================================================
# Attenuation correction
# =============================================================================
rattc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)
rattc.zh_correction(rdata.georef, rdata.params, rczdr.vars,
                    rnme.nme_classif['classif'], phidp_prepro=True,
                    attc_method='ABRI', mlyr=rmlyr, pdp_pxavr_azm=3,
                    pdp_dmin=5, plot_method=True,
                    pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']))

# %%
# =============================================================================
# ZDR Attenuation Correction
# =============================================================================
rattc.zdr_correction(rdata.georef, rdata.params, rczdr.vars, rattc.vars,
                     rnme.nme_classif['classif'], mlyr=rmlyr, rhv_thld=0.98,
                     minbins=10, mov_avrgf_len=5, p2avrf=3, plot_method=True)

# %%

# =============================================================================
# Rainfall estimation
# =============================================================================
rqpe = tp.qpe.qpe_algs.RadarQPE(rdata)

rqpe.z_to_r(rattc.vars['ZH [dBZ]'], a=200, b=1.6, mlyr=rmlyr,
            beam_height=rdata.georef['beam_height [km]'])
rqpe.ah_to_r(rattc.vars['AH [dB/km]'], mlyr=rmlyr,
             beam_height=rdata.georef['beam_height [km]'])
rqpe.z_zdr_to_r(rattc.vars['ZH [dBZ]'], rattc.vars['ZDR [dB]'], mlyr=rmlyr,
                beam_height=rdata.georef['beam_height [km]'])
rqpe.z_ah_to_r(rattc.vars['ZH [dBZ]'], rattc.vars['AH [dB/km]'], z_thld=40,
               mlyr=rmlyr, beam_height=rdata.georef['beam_height [km]'])
rqpe.kdp_to_r(rattc.vars['KDP [deg/km]'], mlyr=rmlyr,
              beam_height=rdata.georef['beam_height [km]'])
rqpe.kdp_zdr_to_r(rattc.vars['KDP [deg/km]'], rattc.vars['ZDR [dB]'],
                  mlyr=rmlyr,
                  beam_height=rdata.georef['beam_height [km]'])


# %%
# =============================================================================
# Plots
# =============================================================================
# Plot cone coverage
tp.datavis.rad_display.plot_cone_coverage(rdata.georef, rdata.params,
                                          rsnr.vars)
# Plot the radar data in a map
tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rqpe.r_z_ah,
                                data_proj=ccrs.OSGB(approx=False),
                                cpy_feats={'status': True})
#%%
# Plot an interactive PPI explorer
tp.datavis.rad_interactive.ppi_base(rdata.georef, rdata.params, rnme.vars,
                                    var2plot='rhoHV [-]')
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
# %%
# ppiexplorer.savearray2binfile(file_name='metoffice-c-band-rain-radar_jersey_202010030735_raw-dual-polar-augzdr-lp-el0.dat',
#                               dir2save='/home/enchiladaszen/Documents/radar_trials/metoffice/')

