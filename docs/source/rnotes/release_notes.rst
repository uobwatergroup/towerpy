.. _rnotes:

What’s New In Towerpy
=====================

Summary – Release highlights
----------------------------

**Latest release: v2.0.0**
~~~~~~~~~~~~~~~~~~~~~~~~~~

Towerpy v2.0.0
~~~~~~~~~~~~~~
::
  **Towerpy v2.0.0**
  
    **Release Date:** 2026


    **New features**
      #. **xarray**: Towerpy now uses xarray for data handling, improving performance and enabling more efficient data manipulation.
      #. **Enhanced visualisation**: New plotting functions and improved interactivity for better data exploration.
      #. **Improved documentation**: Comprehensive updates to the documentation.

    **Deprecations**
      #. **Legacy data handling**: Support for older data formats will be removed in favor of xarray.
      #. **Obsolete functions**: Older functions based on python dictionaries and lists, or numpy arrays will be deprecated in favor of xarray-based functions.


Towerpy v1.1.0
~~~~~~~~~~~~~~
::
  **Towerpy v1.1.0**
  
    **Release Date:** 13 Jan 2026


    **New features**
      #. **rhoHV_Calibration**: Introduces a new function for correcting noise biases in rhoHV.
      #. **Attn_Refl_Relation**: Adds a rhoHV threshold to ensure that only rain echoes are used when computing the Z(A) relations.
      #. **calib_phidp: offsetdetection_ppi**: Adds a plotting option to illustrate the PhiDP offset computation.
      #. **rad_interactive**: Adds an interactive guide line showing the selected azimuth in PPI plots when using coord_sys='rect'.
      #. **georad**: Introduces ppi_georef, a new function for generating georeferenced grids for PPI radar scans.

    **Minor corrections**
      #. **ml_detection**: Improves the melting layer boundaries detection.
      #. **zh_correction & zdr_correction & PhiDP_offsetdetection_vps & ZDR_offsetdetection_vps & lsinterference_filter**: These functions no longer require rad_params during initialisation.
      #. **zdr_correction**: Fixes an issue in the computation of coefficient beta that could incorrectly treat np.nan as a valid value.
      #. **ml_detectionvis**: Resolves an error triggered when no melting layer is detected, ensuring stable plotting behaviour. 
      #. **ppi_ukmogeoref**: Now uses ppi_georef for generating georeference grids, improving consistency across modules.
      #. **datavis**: Sets 'polar' as the default coord_sys across all visualisation functions.

    **Deprecations**
      #. **ppi_emptygeoref**: Renamed to ppi_create_georef, where ppi_georef is used for generating georeference grids, improving consistency across modules.


Towerpy v1.0.9
~~~~~~~~~~~~~~
::
  **Towerpy v1.0.9**
  
    **Release Date:** 1 Sep 2025


    **New features**
      #. **attc_phidp_prepro**: The melting layer can be used as input for processing PhiDP.
      #. **plot_ppi**: Adds a feature to plot contour lines.

    **Minor corrections**
      #. **zh_correction**: Fix an issue when extending the PIA above the melting layer that may cause the use of np.nan as the last valid value.
      #. **rad_display**: Parameters required for some plots are now encapsulated in a single function to reduce repetition and shorten code.
      #. **plot_ppi, plot_setppi, plot_mgrid, plot_cone_coverage, plot_radprofiles, plot_ppidiff, ppi_base, hti_base**: Enables customisation of several plotting options.

    **Deprecations**
      #. **ALL MODULES**: radar objects are no longer strictly required to initialise the modules.

Towerpy v1.0.8
~~~~~~~~~~~~~~
::
  **Towerpy v1.0.8**
  
    **Release Date:** 21 Jun 2025


    **New features**
      #. **attc.attc_zhzdr**: The *zh_correction* function has a new argument (*phidp0*) that sets the PhiDP(0) value for the attenuation correction. This change allows users to use a fixed value for PhiDP(0) but also to estimate it from the data.
      #. **datavis.rad_display.plot_ppi**: The *font_sizes* argument was added to the *plot_ppi* function to allow users to set the font sizes of the plot.
  
    **Minor corrections**
      #. **datavis.rad_display and datavis.rad_interactive**: Fix minor bugs and improve the documentation.
      #. **eclass.nme.clutter_id**: Standardise the outputs of the function.
      #. **eclass.snr**: Standardise the outputs of the function.
      #. **ml.mlyr**: Fix some minor bugs in this module to improve the detection of the melting layer height.
      #. **polprofs.pol_qvps**: Fix a bug in the *pol_qvps* function that was not allowing the function to run properly.
      #. **radutilities.maf_radial**: Fix a bug in the *maf_radial* function that was not allowing the function to run properly.

Towerpy v1.0.7
~~~~~~~~~~~~~~
::
  **Towerpy v1.0.7**
  
    **Release Date:** 23 Mar 2025


    **New features**
      #. **attc.attc_zhzdr**: The tool to pre-process PhiDP for attenuation corrections is no longer an argument of the *zh_correction* function but a separate function (attc_phidp_prepro) instead. This change allows users to better visualise the steps of the attenuation correction.
      #. **datavis.rad_display and datavis.rad_interactive**: Add a new argument (cbticks) to set the colorbar ticks in different plots.
      #. **eclass.nme**: The _lsinterference_filter_ function was moved to this module and improved to generate similar outputs as other functions in this module.
      #. **qpe.qpe_algs**: There is a new rainfall estimator _ah_kdp_to_r_ that uses the KDP and AH variables to estimate rainfall rates.

    **Minor corrections**
      #. **attc.attc_zhzdr**: Fix an error in the *zdr_correction* argument,  _rparams,_ that was not being used properly.
      #. **calib.calib_phidp**: Improve the readability of the documentation.
      #. **datavis.tpy_colors**: The rad_model colormap was moved to the radar related colormaps.
      #. **ml.mlyr**: Fix some minor bugs in this module to improve the detection of the melting layer height.

    **Deprecations**
      #. attc.attc_zhzdr argument: **phidp_prepro** is now a separate function.
      #. **radutilities**: The **lsinterference_filter** function is now part of the eclass.nme module.

Towerpy v1.0.6
~~~~~~~~~~~~~~
::
  **Towerpy v1.0.6**
  
    **Release Date:** 5 Jan., 2025


    **New features**
      #. **PhiDP_Calibration**: This version adds new functions to estimate PhiDP(0).
      #. **plot_ppidiff**: Adds a function to plot the difference between a radar variable from different dicts.
      #. **radutilities**: Adds a function to filter linear signatures and speckles.

    **Minor corrections**
      #. Fix an error in the pre-process PhiDP (zh_correction) module when changing the window size.
      #. **rad_display** and **rad_interactive**: Adds location to legends.

    **Deprecations**
      #. None

Towerpy v1.0.5
~~~~~~~~~~~~~~
::
  **Towerpy v1.0.5**
  
    **Release Date:** 26 Aug., 2024


    **New features**
      #. **AttenuationCorrection**: This version adds a new function to pre-process PhiDP for the attenuation correction in the zh_correction module.
      #. **MeltingLayer**: A new function (ml_ppidelimitation) was added to delimit the melting layer within the PPI.
      #. **RadarQPE**: adp_to_r, ah_to_r, and z_ah_to_r now take temperature as input to interpolate the estimator coefficients.

    **Minor corrections**
      #. Fix minor errors in the datavis module.
      #. **Attn_Refl_Relation**: The default temperature is set to 20C

    **Deprecations**
      #. Printing the running time of the functions is now disabled.

Towerpy v1.0.4
~~~~~~~~~~~~~~
::
  **Towerpy v1.0.4**
  
    **Release Date:** May. 5, 2024


    **New features**
      #. Add [mm] as a unit in the datavis module so rainfall accumulations can be used.
      #. **rad_display**: the *plot_ppi* function can now plot a list of points and the melting layer.
      #. **Attn_Refl_Relation**: There is new module to compute the Z(A) relation.
      #. Users can define the ML as a list or array to use non-isotropic melting layer heights in different modules.

    **Minor corrections**
      #. The units of rainfall intensity are now **mm/h** instead of mm/hr.
      #. **attc_zhzdr module**: The PIA is used to propagate the ZDR attenuation correction beyond the ML.
      #. The notebooks were updated to use similar date and times for the 90, 9 and 0.5 deg scans to improve the understanding of the examples.

    **Deprecations**
      #. 

Towerpy v1.0.3
~~~~~~~~~~~~~~
::
  **Towerpy v1.0.3**
  
    **Release Date:** Dic. 4, 2023


    **New features**
      #. Adds this section! :D
      #. Adds the datavis/colormaps gallery.
      #. **rad_display**: the *plot_ppi* function can now plot a list of points. The *proj_suffix*, *rd_maxrange*, and *pixel_midp* are now possible arguments to be modified by users.
      #. **qpe_algs**: Adds the R(ADP) estimator.

    **Minor corrections**
      #. Corrects a bug in radchain_ukmo_QVP_hti.ipynb where the VPs and QVPs could not be read.
      #. **calib_zdr**: The *max_h* argument in the *offsetdetection_qvps* function is set to 3 as specified in the reference paper.
      #. **rad_display**: the *proj* argument is now named *coord_sys*.
      #. **rad_interactive**: The HTI plot and consequent RadioButtons function show the correct active button.

    **Deprecations**
      #. The *xgrid/ygrid* are now named *grid_rectx* and *grid_rectx*, respectively. This change enables setting different projections.
