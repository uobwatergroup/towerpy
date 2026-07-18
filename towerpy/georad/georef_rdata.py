"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np
from ..io import modeltp as mdtp
from ..utils.unit_conversion import convert
from ..utils.radutilities import get_attrval

def cart2pol(x, y):
    """
    Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x, y : array_like
        Cartesian coordinates

    Returns
    -------
    rho : array_like
        Radial distance from the origin to each point in the x-y plane.
    theta : array_like
        Counter-clockwise angle, in radians, measured from the positive
        x-axis. Values are returned in the range [-pi, pi].

    """
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return rho, theta


def pol2cart(rho, theta):
    """
    Convert polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    rho : array_like
        Radial distance from the origin to each point in the x-y plane.
    theta : array_like
        Counter-clockwise angle, in radians, measured from the positive
        x-axis. Values are expected in the range [-pi, pi].

    Returns
    -------
    x, y : array_like
        Cartesian coordinates.
    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def height_beamc(elev_angle, rad_range, e_rad=6378, std_refr=4/3):
    r"""
    Calculate the height of the radar beam centre above Earth's surface.

    Parameters
    ----------
    elev_angle : float
        Radar elevation angle in degrees.
    rad_range : array
        Range from the radar in kilometres.
    e_rad : float, optional
        Effective Earth's radius. The default is 6378.
    std_refr : float, optional
        Standard refraction coefficient. The default is 4/3.

    Returns
    -------
    h : array
        Height of the radar beam centre above Earth's surface, in kilometres.

    Notes
    -----
    * Beam height is calculated for a standard atmosphere by adapting
      Eq. (2.28b) of Doviak and Zrnic (1993):

        :math:`h = \sqrt{r^2+(\frac{4}{3} E_r)^2+2r(\frac{4}{3} E_r)\sin\Theta}-\frac{4}{3} E_r`

        Where:

        h : height of the centre of the radar beam above Earth's surface

        r : radar range to the targets in kilometres.

        :math:`\Theta` : elevation angle of beam in degree

        :math:`E_r` : effective Earth's radius [approximately 6378 km]
        
        :math:`k_e` : effective-radius factor, typically :math:`4/3` under
        standard refraction.

    References
    ----------
    .. [1] Doviak, R., & Zrnic, D. S. (1993). Electromagnetic Waves and
        Propagation in *Doppler Radar and Weather Observations* (pp. 10-29).
        San Diego: Academic Press, Second Edition.
        https://doi.org/10.1016/B978-0-12-221422-6.50007-3
    """
    h = (np.sqrt((rad_range**2)+((std_refr*e_rad)**2) +
                 (2*rad_range*(std_refr*e_rad) *
                  np.sin(np.deg2rad(elev_angle)))) -
         (std_refr*e_rad))
    return h


def earth_arc_distance(elev_angle, rad_range, hbeam, e_rad=6378, std_refr=4/3):
    r"""
    Compute the distance (arc length) from the radar to the bins.

    Parameters
    ----------
    elev_angle : float
        Radar elevation angle in degrees.
    rad_range : array
        Range from the radar in kilometres.
    hbeam : float
        Height of the centre of the radar beam in kilometres.
    e_rad : float, optional
        Effective Earth's radius. The default is 6378.
    std_refr : float, optional
        Standard refraction coefficient. The default is 4/3.

    Returns
    -------
    s: array
        Ground‑range arc length (km) along the effective Earth sphere.

    Notes
    -----
    * The distance in Cartesian coordinates is calculated by adapting equations
      (2.28b and 2.28c) in [1]_:

      .. math::
          h = \sqrt{r^2+(\frac{4}{3} E_r)^2+2r(\frac{4}{3} E_r)\sin\Theta}-\frac{4}{3} E_r

          s = \frac{4}{3} E_r * arcsin(\frac{r*cos(\Theta)}{\frac{4}{3} E_r+h})

    References
    ----------
    .. [1] Doviak, R., & Zrnic, D. S. (1993). Electromagnetic Waves and
        Propagation in Doppler Radar and Weather Observations (pp. 10-29).
        San Diego: Academic Press, Second Edition.
        https://doi.org/10.1016/B978-0-12-221422-6.50007-3
    """
    s = (std_refr*e_rad) * np.arcsin(
        rad_range * np.cos(np.deg2rad(elev_angle)) / (std_refr*e_rad+hbeam))
    return s


def ppi_georef(rparams, georef=None, polarc_exist=True, elev=0.5,
               gate0=0, gateres=250, bh_geom=True):
    """
    Create a georeferenced grid for PPI radar scans.

    Parameters
    ----------
    rparams : dict
        Radar parameters dictionary containing:
            - 'nrays' : int
                number of rays
            - 'ngates' : int
                number of range gates
            - 'beamwidth [deg]' : float, optional
                Antenna beamwidth, in degrees. Required if ``bh_geom=True``.
    georef : dict, optional
        Existing georeference dictionary with keys 'azim [rad]',
        'elev [rad]', 'range [m]'. Required if `polarc_exist=True`.
    polarc_exist : bool, default True
        If True, polar coordinates (range, azimuth, elevation) are
        read directly from the georef attribute. If False,
        synthesise elevation, azimuth, and range.
        The default is True
    elev : float, default 0.5
        Elevation angle in degrees (used if `polarc_exist=False`).
    gate0 : float, default 0
        Starting range gate in metres (used if `polarc_exist=False`).
    gateres : float, default 250
        Gate resolution in metres (used if `polarc_exist=False`).
    bh_geom : bool, default True
        If True, compute beam-bottom and beam-top heights using the antenna
        beamwidth. The beam-centre height is always computed.

    Returns
    -------
    geogrid : dict
        Dictionary containing georeferenced arrays:
            - 'grid_rectx', 'grid_recty' : Cartesian grid coordinates in km,
            - 'beam_height [km]' : beam centre heights in km.
            - 'beambottom_height [km]' : beam bottom heights in km. Included
             only if ``bh_geom=True``.
            - 'beamtop_height [km]' : beam top heights in km. Included only
             if ``bh_geom=True``.
    base : dict
        Dictionary with base polar coordinates (always in radians/metres):
            - 'azim [rad]' : azimuth angles in radians
            - 'elev [rad]' : elevation angles in radians
            - 'range [m]'  : range values in metres

    Notes
    -----
    * The rectangular grid is returned in kilometres to match convention.
    """
    # Elevation, azimuth, range setup
    if polarc_exist and georef is not None:
        elev = georef['elev [rad]']
        azim = georef['azim [rad]']
        rng  = georef['range [m]']
    else:
        elev = np.deg2rad(np.full(rparams['nrays'], elev))
        azim = np.deg2rad(np.linspace(0, 360, rparams['nrays'],
                                      endpoint=False))
        rng  = gate0 + np.arange(rparams['ngates'], dtype=float) * gateres
    
    elev_deg = np.rad2deg(elev)
    if bh_geom:
        bw = rparams['beamwidth [deg]']
    rng_km = rng / 1000.0  # convert to km for height_beamc

    # Beam heights
    bhkm  = np.array([height_beamc(ray, rng_km) for ray in elev_deg])
    if bh_geom:
        bbhkm = np.array([height_beamc(ray - bw/2, rng_km)
                          for ray in elev_deg])
        bthkm = np.array([height_beamc(ray + bw/2, rng_km)
                          for ray in elev_deg])

    # Cartesian conversion
    s = np.array([earth_arc_distance(ray, rng_km, bhkm[i])
                  for i, ray in enumerate(elev_deg)])
    a = [pol2cart(arcl, azim) for arcl in s.T]
    xgrid = np.array([i[1] for i in a]).T
    ygrid = np.array([i[0] for i in a]).T

    # Build georef dict
    geogrid = {'grid_rectx': xgrid, 'grid_recty': ygrid,
               'beam_height [km]': bhkm}
    if bh_geom:
        geogrid['beambottom_height [km]'] = bbhkm
        geogrid['beamtop_height [km]'] = bthkm

    base = {'azim [rad]': azim, 'elev [rad]': elev, 'range [m]': rng}
    return geogrid, base


# =============================================================================
# %% xarray implementation
# =============================================================================

def ppi_rectgeoref(sweep, radar_altitude=None, bh_geom=True, beamwidth=None):
    r"""
    Add georeferenced Cartesian coordinates and beam‑height fields for a
    Plan Position Indicator (PPI) radar sweep.

    Parameters
    ----------
    sweep : xarray.Dataset
        Input dataset containing at least ``azimuth``, ``elevation`` and
        ``range`` coordinates defined on a ``(azimuth, range)`` grid.
    radar_altitude : float, optional
        Altitude of the radar above mean sea level (km). If provided, the
        beam‑centre, beam‑top, and beam‑bottom heights are offset by this
        value, yielding heights AMSL. If omitted, heights are returned
        relative to the radar (0 km at the radar location).
    bh_geom : bool, optional
        If ``True`` (default), compute full beam‑height geometry including
        beam‑top and beam‑bottom heights. If ``False``, only the beam‑centre
        height is computed.
    beamwidth : float, optional
        Antenna beamwidth, in degrees. If not provided, the function attempts
        to retrieve it from ``sweep`` using known metadata conventions.
        Required only if ``bh_geom=True``.

    Returns
    -------
    sweep : xarray.Dataset
        The input dataset with additional 2‑D georeferenced coordinates:

        - ``grid_rectx`` : Cartesian x‑coordinate (km)
        - ``grid_recty`` : Cartesian y‑coordinate (km)
        - ``beamc_height`` : beam‑centre height (km)
        - ``beamb_height`` : beam‑bottom height (km)
        - ``beamt_height`` : beam‑top height (km)

        All returned fields are aligned with the ``(azimuth, range)`` grid and
        include appropriate metadata (units, long_name, short_name).

    Notes
    -----
    * Internally, this function calls :func:`ppi_georef` to compute the
      georeferenced grid and beam‑height geometry.
    * Elevation, azimuth, and range are converted to radians/metres as needed.
    * Cartesian coordinates are returned in kilometres.
    """
    sweep_vars_attrs_f = mdtp.sweep_vars_attrs_f
    # Build georef dict from dataset coords
    georef = {"azim [rad]": convert(sweep.coords["azimuth"], "rad").values,
              "elev [rad]": convert(sweep.coords["elevation"], "rad").values,
              "range [m]": convert(sweep.coords["range"], "m").values}
    rparams = dict(sweep.attrs)  # copy
    if bh_geom:
        # Resolve beamwidth
        bw = get_attrval("beamwidth", sweep, default=beamwidth, required=False)
        rparams["beamwidth [deg]"] = float(bw)
    geogrid, _ = ppi_georef(rparams, georef=georef, bh_geom=bh_geom)
    # Attach as 2D coords aligned with (azimuth, range)
    if bh_geom:
        sweep = sweep.assign_coords({
            "grid_rectx": (("azimuth", "range"), geogrid["grid_rectx"]),
            "grid_recty": (("azimuth", "range"), geogrid["grid_recty"]),
            "beamc_height": (("azimuth", "range"),
                             geogrid["beam_height [km]"]),
            "beamb_height": (("azimuth", "range"),
                             geogrid["beambottom_height [km]"]),
            "beamt_height": (("azimuth", "range"),
                             geogrid["beamtop_height [km]"]),
            })
    else:
        sweep = sweep.assign_coords({
            "grid_rectx": (("azimuth", "range"), geogrid["grid_rectx"]),
            "grid_recty": (("azimuth", "range"), geogrid["grid_recty"]),
            "beamc_height": (("azimuth", "range"),
                             geogrid["beam_height [km]"]),
            })
    # Add metadata for coordinates from modeltp
    created = ["grid_rectx", "grid_recty", "beamc_height",
               "beamb_height", "beamt_height"]
    for vname in created:
        if vname in sweep and vname in sweep_vars_attrs_f:
            sweep[vname].attrs.update(sweep_vars_attrs_f[vname])
    # Apply altitude offset (AMSL)
    if radar_altitude is not None:
        sweep["beamc_height"] = sweep["beamc_height"] + radar_altitude
        if bh_geom:
            sweep["beamb_height"] = sweep["beamb_height"] + radar_altitude
            sweep["beamt_height"] = sweep["beamt_height"] + radar_altitude
        # Update metadata for AMSL reference
        for name in ["beamc_height", "beamb_height", "beamt_height"]:
            if name in sweep:
                sweep[name].attrs["reference_point"] = "mean_sea_level"
                sweep[name].attrs["height_reference"] = "amsl"
                sweep[name].attrs["description"] += (
                    " Height is referenced to mean sea level"
                    " (radar altitude added).")
    return sweep
