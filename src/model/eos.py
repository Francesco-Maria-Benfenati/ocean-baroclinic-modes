import numpy as np
from numpy.typing import NDArray


class Eos:
    """
    This class is for computing density through the Equation of State.
    """

    @staticmethod
    def compute_density(sal: float, temp: float, press: float) -> float:
        """
        Compute potential density from salinity and potential temperature.

        Arguments
        ---------
        press : <numpy.ndarray>
            pressure [bars]
        temp : <numpy.ndarray>
            sea water potential temperature [°C]
        sal : <numpy.ndarray>
            sea water salinity [PSU]
        NOTE: the three arguments must have same dimensions!

        Raises
        ------
        ValueError
            if input arrays have not same lengths

        Returns
        -------
        density : <numpy.ndarray>
            potential density [kg/(m^3)]


        The Eq. Of Seawater used is the one implemented in NEMO:

            rho(sal, temp, p) = rho(sal, temp, 0)/[1 - p/K(sal, temp, p)] ;

        where rho(sal, temp, 0) is a 15-term eq. in powers of sal and temp.
        K(sal, temp, p) is the secant bulk modulus of seawater: a 26-term eq.
        in powers of sal, temp and p.
        This is based on the Jackett and McDougall (1995) equation of state
        for calculating the in situ density basing on potential temperature
        and salinity. The polinomial coefficient may be found within
        Jackett's paper (Table A1).
        ====================================================================
        NOTE:
            While the original Jackett and McDougall equation is depending
            on pressure, here pressure is expressed in terms
            of depth. The pressure polynomial coefficients have been
            modified coherently in NEMO function by D. J. Lea.
        ====================================================================
        For reference material, see the UNESCO implementatio of Fortran
        function SVAN (Fofonoff and Millero, 1983), which may be found within
        'Algorithms for computation of fundamental properties of seawater'
        (UNESCO, 1983. Section 3, pp. 15-24).
        The following function is a later modification of the one found in
        NEMO by D. J. Lea, Dec 2006.
        """

        # ==================================================================
        # Compute reference density at atmospheric pressure
        #
        #   rho = rho(sal, temp, 0) = rho_0 + A*sal + B*sal^3/2 + C*sal^2 .
        #
        # Notation follows 'International one-atmosphere equation of state
        # of seawater' (Millero and Poisson, 1981).
        # ==================================================================

        rho = Eos.__compute_rho(temp, sal)

        # ==================================================================
        # Compute coefficients in the bulk modulus of seawater expression
        #
        #   K(sal, temp, p) = K_0 + Ap + Bp^2 , with K_0 = K(sal, temp, 0) .
        #
        # Each term is composed by a pure water term (subscript 'w') and
        # others involving pot. temperature and salinity:
        #   K_0 = Kw_0 + a*sal + b*sal^3/2
        #   A = Aw + c*sal + d*sal^3/2
        #   B = Bw + e*sal
        # Notation follows 'A new high pressure equation of state for
        # seawater' (Millero et al, 1980).
        # ==================================================================

        # Bulk modulus of seawater at atmospheric pressure.
        K_0 = Eos.__compute_K_0(temp, sal)
        # Compression term coefficients.
        A = Eos.__compute_A(temp, sal)
        B = Eos.__compute_B(temp, sal)

        # ==================================================================
        # Compute IN SITU POTENTIAL DENSITY IN TERMS OF DEPTH. The above
        # coeffients of terms in K(sal, temp, p) have been modified
        # consistently with Jackett and McDougall (1994).
        #
        #   density(sal, temp, press) = rho/[1 - press/K(sal, temp, press)]
        #                       = rho/[1 - press/(K_0 + (A*press + B*press^2))]
        # ==================================================================

        density = rho / (1.0 - press / (K_0 + press * (A + press * B)))

        # Return density array.
        return density

    @staticmethod
    def __compute_rho(temp: float, sal: float) -> float:
        """
        Compute reference density at atmospheric pressure

        rho = rho(sal, temp, 0) = rho_0 + A*sal + B*sal^3/2 + C*sal^2 .

        Notation follows 'International one-atmosphere equation of state
        of seawater' (Millero and Poisson, 1981).

        Arguments
        ---------
        temp : <class 'numpy.ndarray'>
            sea water potential temperature [°C]
        sal : <class 'numpy.ndarray'>
            sea water salinity [PSU]

        Returns
        -------
        rho: <class 'numpy.ndarray'>
            reference density of sea water at atmospheric pressure
        """

        # Square root of salinity.
        SR = np.sqrt(sal)
        # Density of pure water.
        rho_0 = (
            (
                ((6.536336e-9 * temp - 1.120083e-6) * temp + 1.001685e-4) * temp
                - 9.095290e-3
            )
            * temp
            + 6.793952e-2
        ) * temp + 999.842594
        # Coefficients involving salinity and pot. temperature.
        A = (
            ((5.3875e-9 * temp - 8.2467e-7) * temp + 7.6438e-5) * temp - 4.0899e-3
        ) * temp + 0.824493
        B = (-1.6546e-6 * temp + 1.0227e-4) * temp - 5.72466e-3
        C = 4.8314e-4
        # International one-atmosphere Eq. of State of seawater.
        rho = rho_0 + (A + B * SR + C * sal) * sal

        return rho

    @staticmethod
    def __compute_K_0(temp: float, sal: float) -> float:
        """
        Compute bulk modulus of seawater at atmospheric pressure term

        K_0 = Kw_0 + a*sal + b*sal^3/2

        Notation follows 'A new high pressure equation of state for
        seawater' (Millero et al, 1980).

        Arguments
        ---------
        temp : <class 'numpy.ndarray'>
            sea water potential temperature [°C]
        sal : <class 'numpy.ndarray'>
            sea water salinity [PSU]

        Returns
        -------
        K_0: <class 'numpy.ndarray'>
        bulk modulus of seawater at atmospheric pressure
        """

        # Square root of salinity.
        SR = np.sqrt(sal)
        # Bulk modulus of seawater at atmospheric pressure: pure water term
        Kw_0 = (
            ((-4.190253e-05 * temp + 9.648704e-03) * temp - 1.706103) * temp
            + 1.444304e02
        ) * temp + 1.965933e04
        # Coefficients involving salinity and pot. temperature.
        a = (
            (-5.084188e-05 * temp + 6.283263e-03) * temp - 3.101089e-01
        ) * temp + 5.284855e01
        b = (-4.619924e-04 * temp + 9.085835e-03) * temp + 3.886640e-01
        # Bulk modulus of seawater at atmospheric pressure.
        K_0 = Kw_0 + (a + b * SR) * sal

        return K_0

    @staticmethod
    def __compute_A(temp: float, sal: float) -> float:
        """
        Compute compression term coefficient A in bulk modulus of seawater

        A = Aw + c*sal + d*sal^3/2

        Notation follows 'A new high pressure equation of state for
        seawater' (Millero et al, 1980).

        Arguments
        ---------
        temp : <class 'numpy.ndarray'>
            sea water potential temperature [°C]
        sal : <class 'numpy.ndarray'>
            sea water salinity [PSU]

        Returns
        -------
        A: <class 'numpy.ndarray'>
        compression term coefficient A
        """

        # Square root of salinity.
        SR = np.sqrt(sal)
        # Compression term.
        Aw = (
            (1.956415e-06 * temp - 2.984642e-04) * temp + 2.212276e-02
        ) * temp + 3.186519
        c = (2.059331e-07 * temp - 1.847318e-04) * temp + 6.704388e-03
        d = 1.480266e-04
        A = Aw + (c + d * SR) * sal

        return A

    @staticmethod
    def __compute_B(temp: float, sal: float) -> float:
        """
        Compute compression term coefficient A in bulk modulus of seawater

        B = Bw + e*sal

        Notation follows 'A new high pressure equation of state for
        seawater' (Millero et al, 1980).

        Arguments
        ---------
        temp : <class 'numpy.ndarray'>
            sea water potential temperature [°C]
        sal : <class 'numpy.ndarray'>
            sea water salinity [PSU]

        Returns
        -------
        B: <class 'numpy.ndarray'>
        compression term coefficient B
        """

        # Compression term.
        Bw = (1.394680e-07 * temp - 1.202016e-05) * temp + 2.102898e-04
        e = (6.207323e-10 * temp + 6.128773e-08) * temp - 2.040237e-06
        B = Bw + e * sal

        return B

    @staticmethod
    def adiabtempgrad(sal: float, temp: float, press: float) -> float:
        """
        Compute adiabatic lapse rate (adiabatic temperature gradient), from UNESCO (1983).

        :params sal, temp, press : salinity, temperature, pressure
        """

        DS = sal - 35.0
        ATG = (
            (
                ((-2.1687e-16 * temp + 1.8676e-14) * temp - 4.6206e-13) * press
                + (
                    (2.7759e-12 * temp - 1.1351e-10) * DS
                    + ((-5.4481e-14 * temp + 8.733e-12) * temp - 6.7795e-10) * temp
                    + 1.8741e-8
                )
            )
            * press
            + (-4.2393e-8 * temp + 1.8932e-6) * DS
            + ((6.6228e-10 * temp - 6.836e-8) * temp + 8.5258e-6) * temp
            + 3.5803e-5
        )
        return ATG

    @staticmethod
    def potential_temperature(
        sal: float, temp: float, press: float, ref_press: float = 0
    ) -> float:
        """
        Compute potential temperature, from UNESCO (1983).

        :params sal, temp, press, ref_press : local salinity, local temperature, local pressure, reference pressure
        """

        H = ref_press - press
        XK = H * Eos.adiabtempgrad(sal, temp, press)
        temp = temp + 0.5 * XK
        Q = XK
        press = press + 0.5 * H
        XK = H * Eos.adiabtempgrad(sal, temp, press)
        temp = temp + 0.29289322 * (XK - Q)
        Q = 0.58578644 * XK + 0.121320344 * Q
        XK = H * Eos.adiabtempgrad(sal, temp, press)
        temp = temp + 1.707106781 * (XK - Q)
        Q = 3.414213562 * XK - 4.121320344 * Q
        press = press + 0.5 * H
        XK = H * Eos.adiabtempgrad(sal, temp, press)
        THETA = temp + (XK - 2.0 * Q) / 6.0
        return THETA


if __name__ == "__main__":

    def test_compute_rho_pure_water():
        """
        Test if _compute_rho() gives correct ref density at atm press. for
        pure water (Sal = 0 PSU) at Temperature = 5, 25 °C.
        Reference values may be found within
        'Algorithms for computation of fundamental properties of seawater'
        (UNESCO, 1983. Section 3, p.19)
        """
        ref_Sal = 0  # PSU
        ref_Temp = [5, 25]  # °C
        ref_rho = [999.96675, 997.04796]  # kg/m^3
        out_rho = []
        out_rho.append(Eos._Eos__compute_rho(ref_Temp[0], ref_Sal))
        out_rho.append(Eos._Eos__compute_rho(ref_Temp[1], ref_Sal))
        error = 1e-06  # kg/m^3
        assert np.allclose(ref_rho, out_rho, atol=error)

    def test_compute_rho_standard_seawater():
        """
        Test if _compute_rho() gives correct ref density at atm press. for
        standard seawater (Sal = 35 PSU) at Temperature = 5, 25 °C.
        Reference values may be found within
        'Algorithms for computation of fundamental properties of seawater'
        (UNESCO, 1983. Section 3, p.19)
        """
        ref_Sal = 35  # PSU
        ref_Temp = [5, 25]  # °C
        ref_rho = [1027.67547, 1023.34306]  # kg/m^3
        out_rho = []
        out_rho.append(Eos._Eos__compute_rho(ref_Temp[0], ref_Sal))
        out_rho.append(Eos._Eos__compute_rho(ref_Temp[1], ref_Sal))
        error = 1e-06  # kg/m^3
        assert np.allclose(ref_rho, out_rho, atol=error)

    # Test compute density at atmospheric pressure, from UNESCO documentation.
    test_compute_rho_pure_water()
    test_compute_rho_standard_seawater()

    # Test adiabatic lapse rate and potential temperature
    # From UNESCO documentation.
    assert np.isclose(Eos.adiabtempgrad(40, 40, 10000), 3.255976e-4)
    assert np.isclose(Eos.potential_temperature(40, 40, 10000), 36.89073)

    # Test Compute density from Jackett and Mcdougall, 1995;
    assert np.isclose(Eos.compute_density(35.5, 3.0, 300), 1041.83267)
