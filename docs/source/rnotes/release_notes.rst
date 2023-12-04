.. _rnotes:

What’s New In Towerpy
=====================

Summary – Release highlights
----------------------------

**Latest release: v1.0.3**
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
