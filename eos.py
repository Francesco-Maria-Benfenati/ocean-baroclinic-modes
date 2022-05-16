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
#   B) a function for computing Brunt-Vaisala frequency squared from
#      Potential Density.
# ======================================================================
import xarray
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


def compute_BruntVaisala_freq_sq(z, density):
    """
    Compute Brunt-Vaisala frequency squared from depth and density.

    Arguments
    ---------
    z : <class 'numpy.ndarray'>
        domain depth coordinate [m]
    density : <class 'xarray.core.variable.Variable'>
        potential density [kg/(m^3)]
    NOTE:
        input arrays must have length >= 2 along z for computing 
        Brunt-Vaisala frequency through the finite difference algorithm.
        
    Raises
    ------
    ValueError
        if input density has not same length as z along depth direction
    AttributeError
        if input density is not of type 'xarray.core.variable.Variable'
        - or - if z is not of type 'numpy.ndarray'
    IndexError
        if input density has length smaller than z along depth direction
    
    Returns
    -------
    N2_xarray : <class 'xarray.core.variable.Variable'>
        Brunt-Vaisala frequency squared [(cycles/s)^2]
    mean_dens_xarr : <class 'xarray.core.variable.Variable'>
        mean density vertical profile [kg/(m^3)]
    
    The Brunt-Vaisala frequency N is computed as in Grilli, Pinardi 
    (1999) 'Le cause dinamiche della stratificazione verticale nel
    mediterraneo'
    
    N = (- g/rho_s * ∂rho_s/∂z)^1/2  with g = 9.81 m/s^2,
    
    where rho_s is the density mean vertical profile, 
    depending only on z.
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
    
    # Check if depth is a 1D array.
    if z.ndim > 1:
        raise ValueError('depth must be 1D')
        
    len_z = len(z)
    # Defining value of gravitational acceleration.
    g = 9.806
    # Create empty array for squared Brunt-Vaisala frequency, N^2.
    N2 = np.empty(len_z)
    
    # Compute density mean vertical profile, ignoring NaNs.
    # Mean axis is set depending on input density number of dimensions.
    if len(density.dims) == 1: # 1D density
                                mean_dens = density.values
    elif len(density.dims) == 2: # 2D density
                                mean_dens = np.nanmean(density, axis=0)
    else:
        mean_dens = np.nanmean(density, axis=(1,2))# 3D density
    
    # Check if density has same length as z along depth direction.
    if len(mean_dens) > len_z:
        raise ValueError('legth mismatch along depth direction.')
    
    # Compute  N^2 for the surface level (forward finite difference).
    N2[0] = ( (- g/mean_dens[0])
             *(mean_dens[1] - mean_dens[0])/(z[1] - z[0]) )
    # Compute  N^2 for the surface level (backward finite difference).
    N2[-1] = ( (- g/mean_dens[-1])
              *(mean_dens[-2] - mean_dens[-1])/(z[-2] - z[-1]) )
    
    if len_z > 2:
        # Compute  N^2 for the surface level (centered finite difference).
        # Do it only if len
        for i in range(1, len_z-1):
             N2[i] = ( (- g/mean_dens[i])
                      *(mean_dens[i+1] - mean_dens[i-1])
                      /(z[i+1] - z[i-1]) )
    
    #-------------------------------------------------------------------
    # Return Brunt-Vaisala frequency & mean density profiles.
    #-------------------------------------------------------------------
    # Make N2 of 'xarray.core.variable.Variable' type.
    N2_xarray =  xarray.Variable(dims = 'depth', data = N2)
    # Associate attributes to N2 xarray object.
    N2_attrs = {'long_name': 'Brunt-Vaisala frequency squared.', 
                 'standard_name': 'N_squared', 
                 'units': '1/s^2', 'unit_long':'cycles per second squared.'}
    N2_xarray.attrs = N2_attrs
    
    # Make mean density of 'xarray.core.variable.Variable' type.
    mean_dens_xarr =  xarray.Variable(dims = 'depth', data = mean_dens)
    # Associate attributes to mean density xarray object.
    mean_dens_xarr_attrs = {'long_name': 
                                    'mean potential density', 
                 'standard_name': 'mean_pot_density', 
                 'units': 'kg/m^3', 'unit_long':'kilograms per meter cube'}
    mean_dens_xarr.attrs = mean_dens_xarr_attrs
    
    # Return N2 absolute value, for avoiding depth sign conventions.
    return abs(N2_xarray), mean_dens_xarr
