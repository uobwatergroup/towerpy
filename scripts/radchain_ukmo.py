"""Radar processing chain of UK Met Office radar data."""

import datetime as dt
import numpy as np
import wradlib as wrl
import towerpy as tp
import cartopy.crs as ccrs
from wradlib.dp import phidp_kdp_vulpiani as kdpvpi

RSITE = 'jersey'
MODER = 'zdr'  # zdr or ldr
MODEP = 'lp'  # sp or lp
ELEV = 0
START_TIME = dt.datetime(2020, 10, 3, 3, 35)

EWDIR = ('../datasets/ukmo-nimrod/data/single-site/'
         f'{START_TIME.year}/{RSITE}/{MODEP}el{ELEV}/')

FRADNAME = (EWDIR + f'metoffice-c-band-rain-radar_{RSITE}_'
            f'{START_TIME.strftime("%Y%m%d%H%M")}_raw-dual-polar'
            + f'-aug{MODER}-{MODEP}-el{ELEV}.dat')

PLOT_METHODS = False

# =============================================================================
# Reads polar radar data
# =============================================================================
rdata = tp.io.ukmo.Rad_scan(FRADNAME, RSITE)
rdata.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])
rdata.ppi_ukmogeoref()

# Plot all the radar variables
tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rdata.vars)

# %%
# =============================================================================
# Compute the Signal-to-Noise-Ratio
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
rsnr.signalnoiseratio(rdata.georef, rdata.params, rdata.vars, min_snr=38,
                      data2correct=rdata.vars, plot_method=PLOT_METHODS)

# %%
# =============================================================================
# PhiDP offset correction and unfolding
# =============================================================================
ropdp = tp.calib.calib_phidp.PhiDP_Calibration(rdata)
ropdp.offsetdetection_ppi(rsnr.vars, preset=None)
print(f'Phi_DP(0) = {np.median(ropdp.phidp_offset):.2f}')

ropdp.offset_correction(rsnr.vars['PhiDP [deg]'],
                        phidp_offset=ropdp.phidp_offset,
                        data2correct=rsnr.vars)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rsnr.vars,
                                    var2plot='PhiDP [deg]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, ropdp.vars,
                                    var2plot='PhiDP [deg]')

ropdp.vars['PhiDP [deg]'] = np.ascontiguousarray(
    wrl.dp.unfold_phi(ropdp.vars['PhiDP [deg]'],
                      ropdp.vars['rhoHV [-]'],
                      width=3, copy=True).astype(np.float64))

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, ropdp.vars,
                                    var2plot='PhiDP [deg]')

# %%
# =============================================================================
# Classification of non-meteorological echoes
# =============================================================================
if RSITE == 'chenies':
    clmap = np.loadtxt(f'../towerpy/eclass/ukmo_cmaps/{RSITE}/'
                       + 'chenies_cluttermap_el0.dat')
    if PLOT_METHODS:
        tp.datavis.rad_display.plot_mfs('../towerpy/eclass/mfs_cband/')
    bin_class = 223
else:
    clmap = None
    bin_class = 223 - 64

rnme = tp.eclass.nme.NME_ID(rsnr)

rnme.lsinterference_filter(rdata.georef, rdata.params, ropdp.vars,
                           rhv_min=0.1, data2correct=ropdp.vars,
                           plot_method=PLOT_METHODS)

rnme.clutter_id(rdata.georef, rdata.params, rnme.vars, binary_class=bin_class,
                min_snr=rsnr.min_snr, clmap=clmap, data2correct=rnme.vars,
                plot_method=PLOT_METHODS)
# %%
# =============================================================================
# Melting layer allocation
# =============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rdata)
rmlyr.ml_top = 1.8
rmlyr.ml_thickness = 0.85
rmlyr.ml_bottom = rmlyr.ml_top - rmlyr.ml_thickness

rmlyr.ml_ppidelimitation(rdata.georef, rdata.params, rsnr.vars,
                         plot_method=PLOT_METHODS)

# Plot rhoHV and the ML
if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rnme.vars,
                                    mlyr=rmlyr, var2plot='rhoHV [-]')

# %%
# =============================================================================
# ZDR calibration
# =============================================================================
rozdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rozdr.offset_correction(rnme.vars['ZDR [dB]'], zdr_offset=-0.27,
                        data2correct=rnme.vars)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params,
                                    rnme.vars, var2plot='ZDR [dB]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rozdr.vars,
                                    var2plot='ZDR [dB]')
# %%
# =============================================================================
# Attenuation correction
# =============================================================================
rattc = tp.attc.attc_zhzdr.AttenuationCorrection(rdata)

rattc.attc_phidp_prepro(rdata.georef, rdata.params, rozdr.vars, rhohv_min=0.85,
                        phidp0_correction=False)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rozdr.vars,
                                    var2plot='PhiDP [deg]')
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rattc.vars,
                                    var2plot='PhiDP [deg]')

rattc.zh_correction(rdata.georef, rdata.params, rattc.vars,
                    rnme.nme_classif['classif'], mlyr=rmlyr, pdp_dmin=1,
                    pdp_pxavr_rng=round(4000/rdata.params['gateres [m]']),
                    attc_method='ABRI', pdp_pxavr_azm=3,
                    coeff_alpha=[0.08, 0.18, 0.11],
                    plot_method=PLOT_METHODS)

# %%
# =============================================================================
# PBBc and ZHAH
# =============================================================================
temp = 15

rzhah = tp.attc.r_att_refl.Attn_Refl_Relation(rdata)
rzhah.ah_zh(rattc.vars, zh_upper_lim=55, temp=temp, rband='C',
            copy_ofr=True, data2correct=rattc.vars)
rattc.vars['ZH* [dBZ]'] = rzhah.vars['ZH [dBZ]']

mov_avrgf_len = (1, 3)
zh_difnan = np.where(rzhah.vars['diff [dBZ]'] == 0, np.nan,
                     rzhah.vars['diff [dBZ]'])
zhpdiff = np.array([np.nanmedian(i) if ~np.isnan(np.nanmedian(i))
                    else 0 for cnt, i in enumerate(zh_difnan)])
zhpdiff_pad = np.pad(zhpdiff, mov_avrgf_len[1]//2, mode='wrap')
zhplus_maf = np.ma.convolve(
    zhpdiff_pad, np.ones(mov_avrgf_len[1])/mov_avrgf_len[1],
    mode='valid')
rattc.vars['ZH+ [dBZ]'] = np.array(
    [rattc.vars['ZH [dBZ]'][cnt] - i if i == 0
     else rattc.vars['ZH [dBZ]'][cnt] - zhplus_maf[cnt]
     for cnt, i in enumerate(zhpdiff)])

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(
        rdata.georef, rdata.params, rattc.vars, rattc.vars,
        var2plot1='ZH [dBZ]', var2plot2='ZH* [dBZ]')
# %%
# =============================================================================
# ZDR Attenuation Correction
# =============================================================================
rb_a = 0.39  # Continental
# rb_a = 0.14  # Tropical

rattc.zdr_correction(rdata.georef, rdata.params, rozdr.vars, rattc.vars,
                     rnme.nme_classif['classif'], mlyr=rmlyr, descr=True,
                     coeff_beta=[0.002, 0.07, 0.04], beta_alpha_ratio=rb_a,
                     rhv_thld=0.98, mov_avrgf_len=9, minbins=5, p2avrf=3,
                     attc_method='BRI', plot_method=PLOT_METHODS)

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(rdata.georef, rdata.params, rozdr.vars,
                                        rattc.vars, var2plot1='ZDR [dB]',
                                        var2plot2='ZDR [dB]',
                                        diff_lims=[-1, 1, .1])

# %%
# =============================================================================
# KDP Derivation
# =============================================================================
# KDP Vulpiani
zh_kdp = 'ZH+ [dBZ]'
rkdpv = {}
kdp_vulp = kdpvpi(rattc.vars['PhiDP [deg]'], winlen=3,
                  dr=rdata.params['gateres [m]']/1000, copy=True)
rkdpv['PhiDP [deg]'] = kdp_vulp[0]
rkdpv['KDP [deg/km]'] = kdp_vulp[1]
rkdpv['PhiDP [deg]'][np.isnan(rattc.vars['ZH [dBZ]'])] = np.nan
rkdpv['KDP [deg/km]'][np.isnan(rattc.vars['ZH [dBZ]'])] = np.nan

if PLOT_METHODS:
    tp.datavis.rad_display.plot_ppidiff(
        rdata.georef, rdata.params, rkdpv,
        rattc.vars, var2plot1='KDP [deg/km]', var2plot2='KDP [deg/km]',
        diff_lims=[-1, 1, .25],
        vars_bounds={'KDP [deg/km]': (-1, 3, 17)})

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
if PLOT_METHODS:
    # Plot cone coverage
    tp.datavis.rad_display.plot_cone_coverage(rdata.georef, rdata.params,
                                              rmlyr.mlyr_limits, limh=12,
                                              zlims=[0, 12],
                                              cbticks=rmlyr.regionID,
                                              ucmap='tpylc_div_yw_gy_bu')
    # Plot the radar data in a map
    tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rqpe.r_z_ah,
                                    data_proj=ccrs.OSGB(approx=False),
                                    cpy_feats={'status': True})
# %%
# Plot an interactive PPI explorer
tp.datavis.rad_interactive.ppi_base(rdata.georef, rdata.params, rattc.vars,
                                    mlyr=rmlyr)
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
# %%
# ppiexplorer.savearray2binfile(file_name='metoffice-c-band-rain-radar_jersey_202010030735_raw-dual-polar-augzdr-lp-el0.dat',
#                               dir2save='/home/enchiladaszen/Documents/radar_trials/metoffice/')
