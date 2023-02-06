"""Towerpy: an open-source toolbox for processing polarimetric radar data."""

import math


class Coords_TT:
    """A class to work with the coordinate systems used in Great Britain."""

    def __init__(self, ellipsoid, projection):
        self.ellipsoid = ellipsoid
        self.projection = projection

        # Shape and size of biaxial ellipsoids used in the UK
        if ellipsoid == 'Airy_1830' or self.ellipsoid is None:
            ellipsoid_info = {'a [m]': 6377563.396, 'b [m]': 6356256.9091}
        if ellipsoid == 'Airy_1830_modified':
            ellipsoid_info = {'a [m]': 6377340.189, 'b [m]': 6356034.447}
        if ellipsoid == 'WGS84':
            ellipsoid_info = {'a [m]': 6378137, 'b [m]': 6356752.3141}

        # Transverse Mercator projections used in the UK
        if projection == 'OSGB' or projection is None:
            projection_info = {'N0': -100000, 'E0': 400000,
                               'F0': 0.9996012717, 'phi0': math.radians(49),
                               'lambda0': math.radians(-2)}

        self.ellipsoid_info = ellipsoid_info
        self.projection_info = projection_info

    def param_M(self, phi, a, b):
        """
        Return the parameter M for latitude phi using ellipsoid params a, b.

        Parameters
        ----------
        phi : TYPE
            DESCRIPTION.
        a : TYPE
            DESCRIPTION.
        b : TYPE
            DESCRIPTION.

        Returns
        -------
        M : TYPE
            DESCRIPTION.

        """
        n = (a-b)/(a+b)
        n2 = n**2
        n3 = n * n2
        dphi = phi - self.projection_info['phi0']
        sphi = phi + self.projection_info['phi0']
        M = b * self.projection_info['F0'] * (
                (1 + n + 5/4 * (n2+n3)) * dphi
                - (3*n + 3*n2 + 21/8 * n3) * math.sin(dphi) * math.cos(sphi)
                + (15/8 * (n2 + n3)) * math.sin(2*dphi) * math.cos(2*sphi)
                - (35/24 * n3 * math.sin(3*dphi) * math.cos(3*sphi))
                )
        return M

    def aux_params(phi, a, F0, e2):
        """
        Calculate and return the parameters rho, nu, and eta2.

        Parameters
        ----------
        phi : TYPE
            DESCRIPTION.
        a : TYPE
            DESCRIPTION.
        F0 : TYPE
            DESCRIPTION.
        e2 : TYPE
            DESCRIPTION.

        Returns
        -------
        rho : TYPE
            DESCRIPTION.
        nu : TYPE
            DESCRIPTION.
        eta2 : TYPE
            DESCRIPTION.

        """
        rho = a * F0 * (1-e2) * (1-e2*math.sin(phi)**2)**-1.5
        nu = a * F0 / math.sqrt(1-e2*math.sin(phi)**2)
        eta2 = nu/rho - 1
        return rho, nu, eta2

    def osgb2lalo(self, E, N):
        """
        Convert from OS grid reference to latitude and longitude.

        Parameters
        ----------
        E : TYPE
            DESCRIPTION.
        N : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        a, b = self.ellipsoid_info['a [m]'], self.ellipsoid_info['b [m]']
        e2 = (a**2 - b**2)/a**2
        M, phip = 0, self.projection_info['phi0']
        while abs(N-self.projection_info['N0']-M) >= 1.e-5:
            phip = (N - self.projection_info['N0']
                    - M)/(a*self.projection_info['F0']) + phip
            M = Coords_TT.param_M(self, phip, a, b)

        rho, nu, eta2 = Coords_TT.aux_params(phip, a,
                                             self.projection_info['F0'], e2)

        tan_phip = math.tan(phip)
        tan_phip2 = tan_phip**2
        nu3, nu5 = nu**3, nu**5
        sec_phip = 1./math.cos(phip)

        c1 = tan_phip/2/rho/nu
        c2 = tan_phip/24/rho/nu3 * (5 + 3*tan_phip2 + eta2 * (1 - 9*tan_phip2))
        c3 = tan_phip / 720/rho/nu5 * (61 + tan_phip2*(90 + 45 * tan_phip2))
        d1 = sec_phip / nu
        d2 = sec_phip / 6 / nu3 * (nu/rho + 2*tan_phip2)
        d3 = sec_phip / 120 / nu5 * (5 + tan_phip2*(28 + 24*tan_phip2))
        d4 = sec_phip / 5040 / nu**7 * (61 + tan_phip2*(662 + tan_phip2
                                                        * (1320
                                                           + tan_phip2*720)))
        EmE0 = E - self.projection_info['E0']
        EmE02 = EmE0**2
        phi = phip + EmE0**2 * (-c1 + EmE02*(c2 - c3*EmE02))
        lam = self.projection_info['lambda0'] + EmE0 * (d1 + EmE02*(-d2 + EmE02*(d3 - d4*EmE02)))
        self.latitude = math.degrees(phi)
        self.longitude = math.degrees(lam)
        self.easting = E
        self.northing = N

    def lalo2osgb(self, lat, long):
        """
        Convert from latitude and longitude to OS grid reference (E, N).

        Parameters
        ----------
        phi : TYPE
            DESCRIPTION.
        lam : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        phi = math.radians(lat)
        lam = math.radians(long)
        a, b = self.ellipsoid_info['a [m]'], self.ellipsoid_info['b [m]']
        e2 = (a**2 - b**2)/a**2
        rho, nu, eta2 = Coords_TT.aux_params(phi, a,
                                             self.projection_info['F0'], e2)

        M = Coords_TT.param_M(self, phi, a, b)

        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        cos_phi2 = cos_phi**2
        cos_phi3 = cos_phi2 * cos_phi
        cos_phi5 = cos_phi3 * cos_phi2
        tan_phi2 = math.tan(phi)**2
        tan_phi4 = tan_phi2 * tan_phi2

        a1 = M + self.projection_info['N0']
        a2 = nu/2 * sin_phi * cos_phi
        a3 = nu/24 * sin_phi * cos_phi3 * (5 - tan_phi2 + 9*eta2)
        a4 = nu/720 * sin_phi * cos_phi5 * (61 - 58*tan_phi2 + tan_phi4)
        b1 = nu * cos_phi
        b2 = nu/6 * cos_phi3 * (nu/rho - tan_phi2)
        b3 = nu/120 * cos_phi5 * (5 - 18*tan_phi2 + tan_phi4 + eta2*(14 -
                                  58*tan_phi2))
        lml0 = lam - self.projection_info['lambda0']
        lml02 = lml0**2
        N = a1 + lml02 * (a2 + lml02*(a3 + a4*lml02))
        E = self.projection_info['E0'] + lml0 * (b1 + lml02*(b2 + b3*lml02))
        self.latitude = lat
        self.longitude = long
        self.easting = E
        self.northing = N
