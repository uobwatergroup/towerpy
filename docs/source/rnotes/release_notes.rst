.. _rnotes:

What’s New In Towerpy
=====================

Summary – Release highlights
----------------------------

**Latest release: v1.0.5**
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
