"""Radar processing chain for generating QVPs using UK Met Office data."""

import towerpy as tp
import cartopy.crs as ccrs

rsite = 'chenies'
fdir = f'../datasets/{rsite}/y2020/spel4/'
fname = (f'metoffice-c-band-rain-radar_{rsite}_202010030730_raw-dual-polar-'
         + 'augzdr-sp-el4.dat')

# =============================================================================
# Read polar radar data
# =============================================================================
rdata = tp.io.ukmo.Rad_scan(fdir+fname, rsite)
rdata.ppi_ukmoraw(exclude_vars=['W [m/s]', 'SQI [-]', 'CI [dB]'])
rdata.ppi_ukmogeoref()


# %%
# =============================================================================
# Compute the Signal-to-Noise-Ratio
# =============================================================================
rsnr = tp.eclass.snr.SNR_Classif(rdata)
rsnr.signalnoiseratio(rdata.georef, rdata.params, rdata.vars, min_snr=55,
                      data2correct=rdata.vars, plot_method=(True))

# %%
# =============================================================================
# Generate QVPs of polarimetric variables
# =============================================================================
rprofs = tp.profs.polprofs.PolarimetricProfiles(rdata)
rprofs.pol_qvps(rdata.georef, rdata.params, rsnr.vars, stats=True)

tp.datavis.rad_display.plot_radprofiles(rprofs,
                                        rprofs.georef['profiles_height [km]'],
                                        colours=True)

# %%
# =============================================================================
# ML detection
# =============================================================================
rmlyr = tp.ml.mlyr.MeltingLayer(rprofs)
rmlyr.ml_detection(rprofs, min_h=1.1, comb_id=14, plot_method=True)

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
                                          rsnr.vars)

# Plot the radar data in a map
tp.datavis.rad_display.plot_ppi(rdata.georef, rdata.params, rsnr.vars,
                                data_proj=ccrs.OSGB(approx=False),
                                cpy_feats={'status': True})
# Plot all the radar variables
tp.datavis.rad_display.plot_setppi(rdata.georef, rdata.params, rdata.vars)

# Plot an interactive PPI explorer
tp.datavis.rad_interactive.ppi_base(rdata.georef, rdata.params, rsnr.vars,
                                    var2plot='rhoHV [-]')
ppiexplorer = tp.datavis.rad_interactive.PPI_Int()
