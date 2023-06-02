"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import numpy as np


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
        Radial coordinate, rho is the distance from the origin to a point in
        the x-y plane.
    theta : array
        Angular coordinates is the counterclockwise angle in the x-y plane
        measured in radians from the positive x-axis. The value of the angle
        is in the range [-pi, pi].

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
        Radial coordinate, rho is the distance from the origin to a point in
        the x-y plane.
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
        Radar elevation angle in deg.
    rad_range : array
        Range from the radar in km.
    e_rad : float, optional
        Effective Earth's radius. The default is 6378.
    std_refr : float, optional
        Standard refraction coefficient. The default is 4/3.

    Returns
    -------
    h : array
        Height of the centre of the radar beam in km.

    Notes
    -----
    The beam height above sea level or radar level for a standard atmosphere
    is calculated by adapting equation (2.28b) in [1]_:

        :math:`h = \sqrt{r^2+(\frac{4}{3} E_r)^2+2r(\frac{4}{3} E_r)\sin\Theta}-\frac{4}{3} E_r`

        Where:

        h : height of the centre of the radar beam above Earth's surface

        r : radar range to the targets in km.

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


def cartesian_distance(elev_angle, rad_range, hbeam, e_rad=6378, std_refr=4/3):
    r"""
    Compute the distance (arc length) from the radar to the bins.

    Parameters
    ----------
    elev_angle : float
        Radar elevation angle in deg.
    rad_range : array
        Range from the radar in km.
    hbeam : float
        Height of the centre of the radar beam in km.
    e_rad : float, optional
        Effective Earth's radius. The default is 6378.
    std_refr : float, optional
        Standard refraction coefficient. The default is 4/3.

    Returns
    -------
    s: array
        Distance (arc length) from the radar to the bins in km.

    Notes
    -----
    The distance in Cartesian coordinates is calculated by adapting equations
    (2.28b and 2.28c) in [1]_:

    :math:`h = \sqrt{r^2+(\frac{4}{3} E_r)^2+2r(\frac{4}{3} E_r)\sin\Theta}-\frac{4}{3} E_r`

    :math:`s = \frac{4}{3} E_r * arcsin(\frac{r*cos(\Theta)}{\frac{4}{3} E_r+h})`

    References
    ----------
    .. [1] Doviak, R., & Zrnic, D. S. (1993). Electromagnetic Waves and
        Propagation in Doppler Radar and Weather Observations (pp. 10-29).
        San Diego: Academic Press, Second Edition.
        https://doi.org/10.1016/B978-0-12-221422-6.50007-3
    """
    s = (std_refr*e_rad) * np.arcsin(rad_range
                                     * np.cos(np.deg2rad(elev_angle))
                                     / (std_refr*e_rad+hbeam))
    return s
