"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np
from ..utils.unit_conversion import convert
from ..utils.radutilities import get_attrval

def cart2pol(x, y):
    """
    Transform Cartesian coordinates to Polar.

    Parameters
    ----------
    x, y : array
        Cartesian coordinates

    Returns
    -------
    rho : array
        Radial coordinate, rho is the distance from the origin to a
        point in the x-y plane.
    theta : array
        Angular coordinates is the counterclockwise angle in the x-y
        plane measured in radians from the positive x-axis. The value
        of the angle is in the range [-pi, pi].

    """
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    return rho, theta


def pol2cart(rho, theta):
    """
    Transform polar coordinates to Cartesian.

    Parameters
    ----------
    rho : array
        Radial coordinate, rho is the distance from the origin to a
        point in the x-y plane.
    theta : array
        Angular coordinates is the counterclockwise angle in the x-y plane
        measured in radians from the positive x-axis. The value of the angle
        is in the range [-pi, pi].

    Returns
    -------
    x, y : array
        Cartesian coordinates.

    """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def height_beamc(elev_angle, rad_range, e_rad=6378, std_refr=4/3):
    r"""
    Calculate the height of the centre of the radar beam above Earth's surface.

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
        Height of the centre of the radar beam in kilometres.

    Notes
    -----
    The beam height above sea level or radar level for a standard atmosphere
    is calculated by adapting equation (2.28b) in [1]_:

        :math:`h = \sqrt{r^2+(\frac{4}{3} E_r)^2+2r(\frac{4}{3} E_r)\sin\Theta}-\frac{4}{3} E_r`

        Where:

        h : height of the centre of the radar beam above Earth's surface

        r : radar range to the targets in kilometres.

        :math:`\Theta` : elevation angle of beam in degree

        :math:`E_r` : effective Earth's radius [approximately 6378 km]

    References
    ----------
    .. [1] Doviak, R., & Zrnic, D. S. (1993). Electromagnetic Waves and
        Propagation in Doppler Radar and Weather Observations (pp. 10-29).
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
    Create georeferenced grid for PPI radar scans.

    Parameters
    ----------
    rparams : dict
        Radar parameters dictionary. Must contain:
            - 'nrays' : int, number of rays
            - 'ngates' : int, number of range gates
            - 'beamwidth [deg]' : float, beamwidth in degrees
    georef : dict, optional
        Existing georeference dictionary with keys 'azim [rad]',
        'elev [rad]', 'range [m]'. Required if `polarc_exist=True`.
    polarc_exist : bool
        If True, polar coordinates (range, azimuth, elevation) are
        read directly from the georef attribute. If False,
        synthesise elevation, azimuth, and range.
        The default is True
    elev : float
        Elevation angle in degrees (used if `polarc_exist=False`).
        The default is 0.5
    gate0 : float
        Starting range gate in metres (used if `polarc_exist=False`).
        The default is 0
    gateres : float
        Gate resolution in metres (used if `polarc_exist=False`).
        The default is 250

    Returns
    -------
    geogrid : dict
        Dictionary containing georeferenced arrays:
            - 'grid_rectx', 'grid_recty' : Cartesian grid coordinates in kilometres
            - 'beam_height [km]' : beam centre heights in kilometres
            - 'beambottom_height [km]' : beam bottom heights in kilometres
            - 'beamtop_height [km]' : beam top heights in kilometres
    base : dict
        Dictionary with base polar coordinates (always in radians/metres):
            - 'azim [rad]' : azimuth angles in radians
            - 'elev [rad]' : elevation angles in radians
            - 'range [m]'  : range values in metres

    Notes
    -----
    1. The rectangular grid is returned in kilometres to match convention.
    """
    # Elevation, azimuth, range setup
    if polarc_exist and georef is not None:
        elev = georef['elev [rad]']
        azim = georef['azim [rad]']
        rng  = georef['range [m]']
    else:
        elev = np.deg2rad(np.full(rparams['nrays'], elev))
        azim = np.deg2rad(np.linspace(0, 360, rparams['nrays'], endpoint=False))
        rng  = gate0 + np.arange(rparams['ngates'], dtype=float) * gateres
    
    elev_deg = np.rad2deg(elev)
    if bh_geom:
        bw = rparams['beamwidth [deg]']
    rng_km = rng / 1000.0  # convert to km for height_beamc

    # Beam heights
    bhkm  = np.array([height_beamc(ray, rng_km) for ray in elev_deg])
    if bh_geom:
        bbhkm = np.array([height_beamc(ray - bw/2, rng_km) for ray in elev_deg])
        bthkm = np.array([height_beamc(ray + bw/2, rng_km) for ray in elev_deg])

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

def ppi_rectgeoref(sweep, bh_geom=True, beamwidth=None):
    r"""
    Create georeferenced Cartesian coordinates and beam‑height fields for a
    Plan Position Indicator (PPI) radar sweep.
    
    Parameters
    ----------
    sweep : xarray.Dataset
        Dataset containing azimuth, elevation, and range.
    beamwidth : float, optional
        Beamwidth in degrees. If not provided, the function attempts to
        extract it from sweep.attrs using known conventions. The value is
        required to compute beam‑top and beam‑bottom heights.
    
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
    # Resolve beamwidth
    bw = get_attrval("beamwidth", sweep, default=beamwidth)

    # Build georef dict from dataset coords
    georef = {"azim [rad]": convert(sweep.coords["azimuth"], "rad").values,
              "elev [rad]": convert(sweep.coords["elevation"], "rad").values,
              "range [m]": convert(sweep.coords["range"], "m").values}
    rparams = dict(sweep.attrs)  # copy
    if bh_geom:
        rparams["beamwidth [deg]"] = float(bw)
    geogrid, _ = ppi_georef(rparams, georef=georef, bh_geom=bh_geom)
    # Attach as 2D coords aligned with (azimuth, range)
    if bh_geom:
        sweep = sweep.assign_coords({
            "grid_rectx": (("azimuth", "range"), geogrid["grid_rectx"]),
            "grid_recty": (("azimuth", "range"), geogrid["grid_recty"]),
            "beamc_height": (("azimuth", "range"), geogrid["beam_height [km]"]),
            "beamb_height": (("azimuth", "range"), geogrid["beambottom_height [km]"]),
            "beamt_height": (("azimuth", "range"), geogrid["beamtop_height [km]"]),
            })
    else:
        sweep = sweep.assign_coords({
            "grid_rectx": (("azimuth", "range"), geogrid["grid_rectx"]),
            "grid_recty": (("azimuth", "range"), geogrid["grid_recty"]),
            "beamc_height": (("azimuth", "range"), geogrid["beam_height [km]"]),
            })
    # Add metadata
    # sweep["grid_rectx"].attrs.update(
    #     {"units": "km", "long_name": "rectangular coordinates x"})
    # sweep["grid_recty"].attrs.update(
    #     {"units": "km", "long_name": "rectangular coordinates y"})
    # sweep["beamc_height"].attrs.update(
    #     {"units": "km", "long_name": "beam centre height",
    #      'short_name': 'BEAM_HEIGHT'})
    sweep["grid_rectx"].attrs.update({
        "units": "km",
        "long_name": "radar-centric Cartesian x-coordinate",
        "short_name": "XRECT",
        "standard_name": "radar_cartesian_x_coordinate",
        "description": ("Cartesian x-coordinate (km) derived from azimuth and"
                        " range, with origin at the radar location."),
        "coordinate_system": "radar_cartesian",
        "reference_point": "radar_location",
        "axis": "X"})
    sweep["grid_recty"].attrs.update({
        "units": "km",
        "long_name": "radar-centric Cartesian y-coordinate",
        "short_name": "YRECT",
        "standard_name": "radar_cartesian_y_coordinate",
        "description": ("Cartesian y-coordinate (km) derived from azimuth and"
                        " range, with origin at the radar location."),
        "coordinate_system": "radar_cartesian",
        "reference_point": "radar_location",
        "axis": "Y"})
    sweep["beamc_height"].attrs.update({
        "units": "km",
        "long_name": "beam centre height",
        "short_name": "BEAM_HEIGHT",
        "standard_name": "radar_beam_centre_height",
        "description": ("Height of the radar beam centre (km), computed from "
                        "elevation, range, and Earth curvature."),
        "axis": "Z",
        "coordinate_system": "radar_vertical",
        "reference_point": "radar_location",
        })
    if bh_geom:
        sweep["beamb_height"].attrs.update({
            "units": "km",
            "long_name": "beam bottom height",
            "short_name": "BEAM_BOTTOM_HEIGHT",
            "standard_name": "radar_beam_bottom_height",
            "description": ("Height of the lower edge of the radar beam (km),"
                            " computed from elevation, range, beamwidth, and"
                            " Earth curvature."),
            "axis": "Z",
            "coordinate_system": "radar_vertical",
            "reference_point": "radar_location",
        })
        sweep["beamt_height"].attrs.update({
            "units": "km",
            "long_name": "beam top height",
            "short_name": "BEAM_TOP_HEIGHT",
            "standard_name": "radar_beam_top_height",
            "description": ("Height of the upper edge of the radar beam (km),"
                            " computed from elevation, range, beamwidth, and"
                            " Earth curvature."),
            "axis": "Z",
            "coordinate_system": "radar_vertical",
            "reference_point": "radar_location",
        })

    return sweep