"""Radar processing chain for generating VPs using UK Met Office data."""

import towerpy as tp

rsite = 'chenies'
fdir = f'../datasets/{rsite}/y2020/spel8/'
fname = (f'metoffice-c-band-rain-radar_{rsite}_202010030726_raw-dual-polar-'
         + 'augzdr-sp-el8.dat')

# =============================================================================
# Read polar radar data
# =============================================================================
rdata = tp.io.ukmo.Rad_scan(fdir+fname, rsite)
rdata.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])
rdata.ppi_ukmogeoref()

rdata.vars['PhiDP [deg]'] *= -1

tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rdata.vars)

# %%
# =============================================================================
# Compute the Signal-to-Noise-Ratio
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
rsnr.signalnoiseratio(rdata.georef, rdata.params, rdata.vars, min_snr=55,
                      data2correct=rdata.vars, plot_method=(True))

# %%
# =============================================================================
# Generates VPs of polarimetric variables
# =============================================================================
rprofs = tp.profs.polprofs.PolarimetricProfiles(rdata)
rprofs.pol_vps(rdata.georef, rdata.params, rsnr.vars, stats=True)

tp.datavis.rad_display.plot_radprofiles(rprofs,
                                        rprofs.georef['profiles_height [km]'],
                                        colours=True)

# %%
# =============================================================================
# ML detection
# =============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rprofs)
rmlyr.ml_detection(rprofs, min_h=1.1, comb_id=26, plot_method=True)

# %%
# =============================================================================
# ZDR offset detection
# =============================================================================
rcalzdr = tp.calib.calib_zdr.ZDR_Calibration(rdata)
rcalzdr.offsetdetection_vps(rprofs, mlyr=rmlyr, rad_georef=rdata.georef,
                            rad_params=rdata.params, rad_vars=rsnr.vars,
                            plot_method=True)

# %%
# =============================================================================
# PhiDP offset detection
# =============================================================================
rcalpdp = tp.calib.calib_phidp.PhiDP_Calibration(rdata)
rcalpdp.offsetdetection_vps(rprofs, mlyr=rmlyr, rad_vars=rsnr.vars,
                            rad_georef=rdata.georef, rad_params=rdata.params,
                            plot_method=True)

# %%
# =============================================================================
# Plots
# =============================================================================
# Plot all the radar variables
tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rdata.vars)
