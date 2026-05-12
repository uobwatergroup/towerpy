"""Towerpy: an open-source toolbox for processing polarimetric radar data."""



def apply_offset(ds, var2correct, offset, *, replace_vars=False,
                 out_name=None, azimuth_dim="azimuth", suffix="_QC",
                 provenance_name="apply_offset"):
    """
    Apply an offset to a single radar variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    var2correct : str
        Name of the variable to correct.
    offset : float, array-like, or xarray.DataArray
        Offset to subtract. May be:
        - scalar (same for all rays)
        - 1D array/DataArray indexed by azimuth (per-ray offset)
    out_name : str, optional
        Name of the corrected variable. If None:
        - replace_vars=False -> var2correct + suffix
        - replace_vars=True -> overwrite var2correct
    replace_vars : bool
        If True, overwrite the parent variable.
        If False, write to ``<var>_QC`` unless ``out_name`` is provided.
    azimuth_dim : str, optional
        Name of azimuth dimension.
    suffix : str, optional
        Suffix for corrected variable when preserving original.
    provenance_name : str, optional
        Name recorded in provenance metadata.

    Returns
    -------
    xarray.Dataset
        Dataset with corrected variable and provenance metadata.
    """
    from ..io import modeltp as mdtp

    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f

    # Normalise inputs
    if var2correct not in ds.data_vars:
        raise KeyError(f"Variable '{var2correct}' not found in dataset.")
    if not isinstance(offset, xr.DataArray):
        offset = xr.DataArray(offset)
    var_da = ds[var2correct]
    # Broadcast offset if needed
    offset_b = offset
    if azimuth_dim in var_da.dims and offset_b.ndim == 1:
        offset_b = offset_b.broadcast_like(
            var_da.sel({azimuth_dim: var_da[azimuth_dim]}))
    # Determine output variable name
    if out_name is None:
        out_name = var2correct if replace_vars else f"{var2correct}{suffix}"
    # Apply correction
    corrected = (var_da - offset_b).rename(out_name)
    ds_out = ds.copy()
    # Build attrs: parent + canonical
    parent_attrs = ds[var2correct].attrs.copy()
    canonical_attrs = sweep_vars_attrs_f.get(out_name, {}).copy()
    # Remove provenance keys from canonical attrs
    # for key in ["correction", "parent", "mode", "provenance", "provenance_step",
    #             "provenance_base", "correction_chain"]:
    #     canonical_attrs.pop(key, None)
    merged_attrs = {**parent_attrs, **canonical_attrs}
    # Update correction_chain and provenance
    merged_attrs = add_correction_step(
        parent_attrs=merged_attrs, step=provenance_name, parent=var2correct,
        params={"offset": (offset.values.tolist() if hasattr(offset, "values")
                           else offset),
                "replace_vars": replace_vars,
                "suffix": suffix,
                "out_name": out_name}, outputs=[out_name],
        mode="overwrite" if replace_vars else "preserve")
    # Assign corrected variable
    if replace_vars:
        ds_out = safe_assign_variable(ds_out, var2correct, corrected)
    else:
        ds_out = safe_assign_variable(ds_out, out_name, corrected)
    # Dataset-level provenance
    params = {"var2correct": var2correct,
              "offset": (offset.values.tolist() if hasattr(offset, "values")
                         else offset),
              "out_name": out_name, "replace_vars": replace_vars,
              "azimuth_dim": azimuth_dim, "suffix": suffix}
    ds_out = record_provenance(ds_out, function=provenance_name,
                               inputs=[var2correct], outputs=[out_name],
                               parameters=params)
    return ds_out
