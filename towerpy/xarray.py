# towerpy/xarray.py

"""
towerpy Xarray Accessors
------------------------

This module registers Towerpy-specific accessors on xarray objects.
"""

from __future__ import annotations

import xarray as xr
from .datavis.rad_display import plot_ppi_xr, plot_setppi_xr
from .qpe.qpe_algs import compute_rqpe, _QPE_DOCS, _QPE_REFS

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

        # =============================================================================
        # Profiles
        # =============================================================================
        # def build_vp(self, **kwargs):
        #     # Convert to dataset
        #     ds = self._da.to_dataset(name=self._da.name)
        #     # Disable thresholding
        #     kwargs.setdefault("thresholds", None)
        #     return build_vp(ds, **kwargs)


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

        # ------------------------------------------------------------------
        # Plotting
        # ------------------------------------------------------------------
        def plot_setppi_xr(self, **kwargs):
            """
            Plot a set of PPIs using Towerpy.

            See :func:`towerpy.datavis.rad_display.plot_setppi_xr` for full
            documentation of parameters.
            """
            return plot_setppi_xr(self._ds, **kwargs)

        # ------------------------------------------------------------------
        # QPE
        # ------------------------------------------------------------------
        def compute_rqpe(self, *, qpe_based_on, out_name=None, qpe_amlb=False,
                         rband="C", temp=20.0, ml_mask=None,
                         append_to=None, ml_mask_name="ML_PCP_CLASS",
                         threshold_var_name=None, thld_var_value=40.0,
                         adp_a=None, adp_b=None, ah_a=None, ah_b=None,
                         kdp_a=None, kdp_b=None, zr_a=None, zr_b=None,
                         kdp_zdr_a=None, kdp_zdr_b=None, kdp_zdr_c=None,
                         z_zdr_a=None, z_zdr_b=None, z_zdr_c=None,
                         rz_a=None, rz_b=None, rah_a=None, rah_b=None,
                         rkdp_a=None, rkdp_b=None, zh=None, ah=None,
                         adp=None, kdp=None, zdr=None):
            """
            Compute a radar‑based QPE product using the dataset attached to this
            accessor.

            This is a thin wrapper around
            :func:`towerpy.qpe.qpe_algs.compute_rqpe`. See that function for full
            documentation of parameters and behaviour.
            """
            return compute_rqpe(
                self._ds, qpe_based_on=qpe_based_on, out_name=out_name,
                qpe_amlb=qpe_amlb, rband=rband, temp=temp, ml_mask=ml_mask,
                ml_mask_name=ml_mask_name, threshold_var_name=threshold_var_name,
                thld_var_value=thld_var_value, append_to=append_to,
                adp_a=adp_a, adp_b=adp_b, ah_a=ah_a, ah_b=ah_b, kdp_a=kdp_a,
                kdp_b=kdp_b, zr_a=zr_a, zr_b=zr_b, kdp_zdr_a=kdp_zdr_a,
                kdp_zdr_b=kdp_zdr_b, kdp_zdr_c=kdp_zdr_c, z_zdr_a=z_zdr_a,
                z_zdr_b=z_zdr_b, z_zdr_c=z_zdr_c, rz_a=rz_a, rz_b=rz_b,
                rah_a=rah_a, rah_b=rah_b, rkdp_a=rkdp_a, rkdp_b=rkdp_b,
                zh=zh, ah=ah, adp=adp, kdp=kdp, zdr=zdr)

        def add_qpe(self, *, qpe_based_on, **kwargs):
            """ Add a QPE product to this dataset."""
            return compute_rqpe(self._ds, qpe_based_on=qpe_based_on,
                                append_to=self._ds, **kwargs)

        def qpe_based_on_help(self, relation: str) -> str:
            """
            Return detailed documentation for a specific QPE estimator.

            Parameters
            ----------
            relation : str
                Name of the QPE relation (e.g. "adp", "kdp", "z", "z_kdp").

            Returns
            -------
            str
                Sphinx-formatted documentation block.
            """
            key = relation.lower()
            if key not in _QPE_DOCS:
                raise ValueError(
                    f"Unknown QPE relation: {relation}. "
                    f"Available: {', '.join(sorted(_QPE_DOCS))}"
                )

            doc = _QPE_DOCS[key]
            refs = "\n".join(_QPE_REFS[r] for r in doc["refs"])

            return (f"""{doc['title']}
                    {'-' * len(doc['title'])}
                    
                    Math
                    ----
                    {doc['math']}
                    
                    Units
                    -----
                    {doc['units']}
                    
                    Notes
                    -----
                    {doc['notes']}
                    
                    References
                    ----------
                    {refs}
                    """
                    )
