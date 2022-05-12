# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 10:25:22 2022

@author: Francesco Maria
"""
# ======================================================================
# This files includes functions related to the *Equation of State (EOS)*
#
# In particular:
#   A) a function implemented for computing density from Potential 
#      Temperature and Salinity;
# ======================================================================
import numpy as np


def compute_density(z, temp, S):
    """
    Compute potential density from salinity and potential temperature.
    
    Arguments
    ---------
    z : <class 'xarray.core.variable.Variable'>
        depth [m]
    temp : <class 'xarray.core.variable.Variable'>
        sea water potential temperature [°C]
    S : <class 'xarray.core.variable.Variable'>
        sea water salinity [PSU]
    NOTE: the three arguments must have same dimensions!
    
    Raises
    ------
    ValueError
        if input arrays have not same lengths or number of dimensions
    AttributeError
        if input arrays are not of type 'xarray.core.variable.Variable'
        
    Returns
    -------
    density : <class 'xarray.core.variable.Variable'>
        potential density [kg/(m^3)]
    
    
    The Eq. of state used is the one implemented in NEMO:
                     
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
    
    # Check if input arrays have same number of dimensions.
    if (len(z.dims) == len(temp.dims) and len(temp.dims) == len(S.dims)):
        pass
    else:
        raise ValueError('dimension mismatch')
        
    SR = np.sqrt(S) 
    # ==================================================================
    # Compute reference density at atmospheric pressure
    #
    #   rho = rho(S, temp, 0) = rho_0 + A*S + B*S^3/2 + C*S^2 .
    #
    # Notation follows 'International one-atmosphere equation of state
    # of seawater' (Millero and Poisson, 1981).
    # ==================================================================
    
    # Density of pure water.
    rho_0 = ( ( ( (6.536332e-9*temp - 1.120083e-6)*temp + 1.001685e-4)
             *temp - 9.095290e-3)*temp + 6.793952e-2)*temp + 999.842594
    # Coefficients involving salinity and pot. temperature.
    A = ( ( (5.3875e-9*temp - 8.2467e-7)*temp + 7.6438e-5)
           *temp - 4.0899e-3)*temp + 0.824493
    B = (-1.6546e-6*temp + 1.0227e-4)*temp - 5.72466e-3
    C = 4.8314e-4
    # International one-atmosphere Eq. of State of seawater.
    rho = rho_0 + (A + B*SR + C*S)*S
    del(rho_0, A, B, C)
    
    # ==================================================================
    # Compute bulk modulus of seawater
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
    
    # Bulk modulus of seawater at atmospheric pressure: pure water term
    Kw_0 = ( ( ( (- 1.361629e-4*temp - 1.852732e-2)*temp - 30.41638) 
              *temp + 2098.925)*temp + 190925.6)
    # Coefficients involving salinity and pot. temperature.
    a = ( ( (2.326469e-3*temp + 1.553190)*temp - 65.00517) 
           *temp + 1044.077)
    b = (- 0.1909078*temp + 7.390729)*temp - 55.87545
    # Bulk modulus of seawater at atmospheric pressure.
    K_0 = Kw_0 + (a + b*SR)*S 
    del(Kw_0, a, b)
    # Compression terms.
    Aw = ( ( (-5.939910e-6*temp - 2.512549e-3)*temp + 0.1028859)
          *temp + 4.721788)
    c = (7.267926e-5*temp - 2.598241e-3)*temp - 0.1571896
    d = 2.042967e-2
    A = Aw + (c + d*SR)*S
    del(Aw, c, d)
    Bw = (-1.296821e-6*temp + 5.782165e-9)*temp - 1.045941e-4
    e = (3.508914e-8*temp + 1.248266e-8)*temp + 2.595994e-6
    B = Bw + e*S
    del(Bw, e) 
    
    # ==================================================================
    # Compute IN SITU POTENTIAL DENSITY IN TERMS OF DEPTH. The above 
    # coeffients of terms in K(S, temp, p) have been modified 
    # consistently with Jackett and McDougall (1994). 
    #
    #   density(S, temp, z) = rho/[1 - z/K(S, temp, z)]
    #                        = rho/[1 - z/(K_0 + (Az + Bz^2))]
    # ==================================================================
    
    density = rho / (1.0 - z/(K_0 + z*(A + z*B)) )
    # Associate attributes to density xarray object.
    dens_attrs = {'long_name': 'Density', 
                 'standard_name': 'sea_water_potential_density', 
                 'units': 'kg/m^3', 'unit_long':'kilograms per meter cube'}
    density.attrs = dens_attrs
    
    # Return density xarray.
    return density
