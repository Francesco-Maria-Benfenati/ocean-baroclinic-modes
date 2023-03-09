# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:25:22 2022

@author: Francesco Maria
"""
# ======================================================================
# This files includes functions related to the
# *Equation of Seawater(EOS)*
#
# In particular:
#   A) a function implemented for computing density from Potential
#      Temperature and Salinity;
#   B) a function for computing Brunt-Vaisala frequency squared from
#      Potential Density.
# ======================================================================
import numpy as np


def compute_density(z, temp, S):
    """
    Compute potential density from salinity and potential temperature.

    Arguments
    ---------
    z : <numpy.ndarray>
        depth [m]
    temp : <numpy.ndarray>
           sea water potential temperature [°C]
    S : <numpy.ndarray>
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

        rho(S, temp, p) = rho(S, temp, 0)/[1 - p/K(S, temp, p)] ;

    where rho(S, temp, 0) is a 15-term eq. in powers of S and temp.
    K(S, temp, p) is the secant bulk modulus of seawater: a 26-term eq.
    in powers of S, temp and p.
    This is based on the Jackett and McDougall (1994) equation of state
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
    #   rho = rho(S, temp, 0) = rho_0 + A*S + B*S^3/2 + C*S^2 .
    #
    # Notation follows 'International one-atmosphere equation of state
    # of seawater' (Millero and Poisson, 1981).
    # ==================================================================
    
    rho = _compute_rho(temp, S)

    # ==================================================================
    # Compute coefficients in the bulk modulus of seawater expression
    #
    #   K(S, temp, p) = K_0 + Ap + Bp^2 , with K_0 = K(S, temp, 0) .
    #
    # Each term is composed by a pure water term (subscript 'w') and
    # others involving pot. temperature and salinity:
    #   K_0 = Kw_0 + a*S + b*S^3/2
    #   A = Aw + c*S + d*S^3/2
    #   B = Bw + e*S
    # Notation follows 'A new high pressure equation of state for
    # seawater' (Millero et al, 1980).
    # ==================================================================
    
    # Bulk modulus of seawater at atmospheric pressure.
    K_0 = _compute_K_0(temp, S)
    # Compression term coefficients.
    A = _compute_A(temp, S)
    B = _compute_B(temp, S)
    
    # ==================================================================
    # Compute IN SITU POTENTIAL DENSITY IN TERMS OF DEPTH. The above
    # coeffients of terms in K(S, temp, p) have been modified
    # consistently with Jackett and McDougall (1994).
    #
    #   density(S, temp, z) = rho/[1 - z/K(S, temp, z)]
    #                       = rho/[1 - z/(K_0 + (Az + Bz^2))]
    # ==================================================================

    density = rho / (1.0 - z/(K_0 + z*(A + z*B)) )
   
    # Return density array.
    return density


def _compute_rho(temp, S):
    """
    Compute reference density at atmospheric pressure
    
      rho = rho(S, temp, 0) = rho_0 + A*S + B*S^3/2 + C*S^2 .
    
    Notation follows 'International one-atmosphere equation of state
    of seawater' (Millero and Poisson, 1981).
    
    Arguments
    ---------
    temp : <class 'numpy.ndarray'>
           sea water potential temperature [°C]
    S : <class 'numpy.ndarray'>
        sea water salinity [PSU]
    
    Returns
    -------
    rho: <class 'numpy.ndarray'>
         reference density of sea water at atmospheric pressure
    """
    
    # Square root of salinity.
    SR = np.sqrt(S)
    # Density of pure water.
    rho_0 = ( ( ( (6.536336e-9*temp - 1.120083e-6)*temp + 1.001685e-4)
               *temp - 9.095290e-3)*temp + 6.793952e-2)*temp + 999.842594
    # Coefficients involving salinity and pot. temperature.
    A = ( ( (5.3875e-9*temp - 8.2467e-7)*temp + 7.6438e-5)
           *temp - 4.0899e-3)*temp + 0.824493
    B = (-1.6546e-6*temp + 1.0227e-4)*temp - 5.72466e-3
    C = 4.8314e-4
    # International one-atmosphere Eq. of State of seawater.
    rho = rho_0 + (A + B*SR + C*S)*S
    
    return rho


def _compute_K_0(temp, S):
    """
    Compute bulk modulus of seawater at atmospheric pressure term
    
    K_0 = Kw_0 + a*S + b*S^3/2

    Notation follows 'A new high pressure equation of state for
    seawater' (Millero et al, 1980).
    
    Arguments
    ---------
    temp : <class 'numpy.ndarray'>
           sea water potential temperature [°C]
    S : <class 'numpy.ndarray'>
        sea water salinity [PSU]
    
    Returns
    -------
    K_0: <class 'numpy.ndarray'>
       bulk modulus of seawater at atmospheric pressure
    """
    
    # Square root of salinity.
    SR = np.sqrt(S)
    # Bulk modulus of seawater at atmospheric pressure: pure water term
    Kw_0 = ( ( ( (- 4.190253e-05*temp + 9.648704e-03)*temp - 1.706103)
              *temp + 1.444304e+02)*temp + 1.965933e+04)
    # Coefficients involving salinity and pot. temperature.
    a = ( ( (- 5.084188e-05*temp + 6.283263e-03)*temp - 3.101089e-01)
           *temp + 5.284855e+01)
    b = (- 4.619924e-04*temp + 9.085835e-03)*temp + 3.886640e-01
    # Bulk modulus of seawater at atmospheric pressure.
    K_0 = Kw_0 + (a + b*SR)*S

    return K_0


def _compute_A(temp, S):
    """
    Compute compression term coefficient A in bulk modulus of seawater
    
    A = Aw + c*S + d*S^3/2

    Notation follows 'A new high pressure equation of state for
    seawater' (Millero et al, 1980).
    
    Arguments
    ---------
    temp : <class 'numpy.ndarray'>
           sea water potential temperature [°C]
    S : <class 'numpy.ndarray'>
        sea water salinity [PSU]
    
    Returns
    -------
    A: <class 'numpy.ndarray'>
       compression term coefficient A
    """
    
    # Square root of salinity.
    SR = np.sqrt(S)
    # Compression term.
    Aw = ( ( (1.956415e-06*temp - 2.984642e-04)*temp + 2.212276e-02)
          *temp + 3.186519)
    c = (2.059331e-07*temp - 1.847318e-04)*temp + 6.704388e-03
    d = 1.480266e-04
    A = Aw + (c + d*SR)*S

    return A


def _compute_B(temp, S):
    """
    Compute compression term coefficient A in bulk modulus of seawater
    
    B = Bw + e*S

    Notation follows 'A new high pressure equation of state for
    seawater' (Millero et al, 1980).
    
    Arguments
    ---------
    temp : <class 'numpy.ndarray'>
           sea water potential temperature [°C]
    S : <class 'numpy.ndarray'>
        sea water salinity [PSU]
    
    Returns
    -------
    B: <class 'numpy.ndarray'>
       compression term coefficient B
    """
    
    # Compression term.
    Bw = (1.394680e-07*temp - 1.202016e-05)*temp + 2.102898e-04
    e = (6.207323e-10*temp + 6.128773e-08)*temp - 2.040237e-06
    B = Bw + e*S 

    return B


def compute_BruntVaisala_freq_sq(z, mean_density):
    """
    Compute Brunt-Vaisala frequency squared from depth and density.

    Arguments
    ---------
    z : <class 'numpy.ndarray'>
        depth  [m]
    mean_density : <class 'numpy.ndarray'>
        mean potential density vertical profile [kg/(m^3)], 
        defined on z grid

    Raises
    ------
    IndexError
        if input density has not same length as z along depth direction
                        - or -
        if input arrays are empty

    Returns
    -------
    N2 : <class 'numpy.ndarray'>
        Brunt-Vaisala frequency squared [(cycles/s)^2]

    The Brunt-Vaisala frequency N is computed as in Grilli, Pinardi
    (1999) 'Le cause dinamiche della stratificazione verticale nel
    mediterraneo'

    N = (- g/rho_0 * ∂rho_s/∂z)^1/2  with g = 9.806 m/s^2,

    where rho_0 is the reference density.
    --------------------------------------------------------------------
    NOTE:
           Here, the SQUARE OF BRUNT-VAISALA FREQUENCY is computed
           N2 = N^2 ,
           in order to have greater precision when computing the
           baroclinic rossby radius, which needs the N^2 value.
           Furthermore, N2 absolute value is returned for avoiding
           troubles with depth sign convenction, which may change from
           one dataset to another one.
    --------------------------------------------------------------------
    """
        
    # Take absolute value of depth, so that to avoid trouble with depth
    # sign convention.  
    z = abs(z)

    len_z = len(z)
    
    # Defining value of gravitational acceleration.
    g = 9.806 # (m/s^2)
    # Defining value of reference density rho_0.
    rho_0 = 1025 #(kg/m^3)
  
    # Create empty array for squared Brunt-Vaisala frequency, N^2.
    N2 = np.empty(len_z)
    
    # Compute  N^2 for the surface level (forward finite difference).
    dz_0 = z[1] - z[0]
    N2[0] = ( - (g/rho_0)
              *(mean_density[0] - mean_density[1])/dz_0)
    # Compute  N^2 for the surface level (backward finite difference).
    dz_n = z[-2] - z[-1]
    N2[-1] = ( - (g/rho_0)
              *(mean_density[-1] - mean_density[-2])/dz_n)

    if len_z > 2:
        # Compute  N^2 for the surface level (centered finite difference).
        # Do it only if len_z>2.
        for i in range(1, len_z-1):
            dz = z[i+1] - z[i-1]
            N2[i] = ( - (g/rho_0)
                      *(mean_density[i-1] - mean_density[i+1])/(dz))

    # Return N2 absolute value, for avoiding depth sign conventions.
    return N2
