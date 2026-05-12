# towerpy/xarray.py

"""
towerpy Xarray Accessors
------------------------

This module registers Towerpy-specific accessors on xarray objects.
"""

from __future__ import annotations

import xarray as xr
from .datavis.rad_display import plot_ppi_xr, plot_setppi_xr


#TODO: add accesors for the rest of the functions.
# =============================================================================
# DataArray accessor
# =============================================================================
if not hasattr(xr.DataArray, "tp"):
    @xr.register_dataarray_accessor("tp")
    class TowerpyDAAccessor:
        """
        Towerpy accessor for xarray.DataArray.

        Provides variable-level helpers.
        """

        def __init__(self, da: xr.DataArray):
            self._da = da

        # =============================================================================
        # Plotting
        # =============================================================================
        def plot_ppi_xr(self, **kwargs):
            """
            Plot a PPI using Towerpy.

            See :func:`towerpy.datavis.rad_display.plot_ppi_xr` for full
            documentation of parameters.
            """
            name = self._da.name or "var"
            ds = self._da.to_dataset(name=name)
            return plot_ppi_xr(ds, var2plot=name, **kwargs)

        def plot_cone_coverage_xr(self, **kwargs):
            """
            Plot a 3‑D cone‑coverage volume from a DataArray.
        
            Converts the DataArray to a Dataset and forwards the call.
        
            See :func:`towerpy.datavis.rad_display.plot_cone_coverage_xr`
            for full documentation of parameters.
            """
            from .datavis.rad_display import plot_cone_coverage_xr
            name = self._da.name or "var"
            ds = self._da.to_dataset(name=name)
            return plot_cone_coverage_xr(ds, var2plot=name, **kwargs)

        # =============================================================================
        # Profiles
        # =============================================================================
        #TODO: add accessor for profs plotting.


# =============================================================================
# Dataset accessor
# =============================================================================
if not hasattr(xr.Dataset, "tp"):
    @xr.register_dataset_accessor("tp")
    class TowerpyDSAccessor:
        """
        Towerpy accessor for xarray.Dataset.

        Provides dataset-level helpers.
        """

        def __init__(self, ds: xr.Dataset):
            self._ds = ds

        # =============================================================================
        # Plotting
        # =============================================================================        
        def plot_setppi_xr(self, **kwargs):
            """
            Plot a set of PPIs using Towerpy.

            See :func:`towerpy.datavis.rad_display.plot_setppi_xr` for full
            documentation of parameters.
            """
            return plot_setppi_xr(self._ds, **kwargs)

        def plot_cone_coverage_xr(self, **kwargs):
            """
            Plot a 3‑D cone‑coverage volume using Towerpy.
        
            See :func:`towerpy.datavis.rad_display.plot_cone_coverage_xr`
            for full documentation of parameters.
            """
            from .datavis.rad_display import plot_cone_coverage_xr
            return plot_cone_coverage_xr(self._ds, **kwargs)
