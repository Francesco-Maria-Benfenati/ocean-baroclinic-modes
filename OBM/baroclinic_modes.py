# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 12:18:27 2022

@author: Francesco Maria
"""
# ======================================================================
# This file includes one function for computing the 
# *vertical structure function of motion* & the 
# *Rossby radius vertical profile* for each mode of motion
# (both barotropic and baroclinic).
# ======================================================================
import numpy as np
import scipy as sp
from scipy import interpolate, integrate


def compute_barocl_modes(depth, N2, n_modes):
    """
    Computes Rossby radius vertical profile and vert. structure func.

    Arguments
    ---------
    depth : <class 'numpy.ndarray'>
        depth variable (1D)
    N2 : <class 'numpy.ndarray'>
        Brunt-Vaisala frequency squared (along depth, 1D)
    n_modes : 'int'
        number of modes of motion to be considered

    Raises
    ------
    ValueError
        if N2 and depth have different lengths (& if N2 is empty or NaN)
    ArpackError
        if N2 is null
    
    Returns
    -------
    rossby_rad : <class 'numpy.ndarray'>
        baroclinic Rossby radius vertical profile, for each mode
        of motion considered 
    phi : <class 'numpy.ndarray'>
        vertical structure function, for each mode of motion considered 
     
        ---------------------------------------------------------------
                            About the algorithm
        ---------------------------------------------------------------
    0) The scaling parameters L, H, f_0 are defined.                    
    1) N2 is linearly interpolated on a new equally spaced depth grid
       with grid step = 1 m.
    2) N2 is scaled, so that it goes from 0 to 1. This way, the 
       the algorithm for finding the problem eigenvalues works best.
       The problem parameter S is then computed as in 
       Grilli, Pinardi (1999)
    
            S = (N2 * H^2)/(f_0^2 * L^2)
            
       where L and H are respectively the horizontal and vertical 
       scales of motion; f_0 is the Coriolis parameter.
    3) The finite difference matrix 'A' and the S matrix 'B' are
       computed and the eigenvalues problem is solved:
           
           A * w = lambda * B * w  (lambda eigenvalues, w eigenvectors)
       with    
                        | 0      0    0     0   . . . . . 0  |
                        | 12    -24   12     0   . . . . . 0 |
                        | -1    16   -30    16  -1  0 . . 0  |
       A = (1/12dz^2) * | .     -1    16  -30   16  -1  . 0 |    ,
                        | .        .      .   .     .    .  |
                        | .       0   -1    16  -30   16  -1| 
                        | .            0    0    12  -24  12|
                        | 0      0    0     0   . . . . . 0 |
       and         
                         | -S_0    0    0     . . .   0 |
                         | 0     -S_1   0     0   . . 0 |
                         | 0      0   -S_2    0 . . . 0 |
        B =              | .      0     0     .       . |    ;
                         | .        .      .      .   . |
                         | .          0     0      -S_n |
                         
        where dz is scaled as considering depth between 0 and 1.
        Boundary Conditions are w = 0 at z=0,H : they are implemented
        setting the first and last lines of A = 0.
        Eigenvalues are computed through scipy algorithm.
    4) The eigenvectors are computed integrating the equation 
    
        (d^2/dz^2) * w = - lambda * S * w
        
       through Numerov's numerical method (with BCs  w = 0 at z=0,H).
       The first value is computed as
           w(0 + dz)= (- eigenvalues*phi_0)*dz/(1+lambda*S[1]*(dz**2)/6)
       where phi_0 = phi(z=0) set = 1 as BC.
    5) The baroclinic Rossby radius is obtained as 
           
            R_n = 1/lambda_n
       
       for each mode of motion 'n'.
                             - & - 
       The vertical structure function Phi(z) is computed integrating
       S*w between 0 and z (for each mode of motion), considering
       depth between 0 and 1.
    """
    
    # Check if depth and BV freq. squared (N2) have the same length.
    if len(N2) == len(depth):
        pass
    else:
        raise ValueError('length mismatch between BVfreq and depth.')
    # Take absolute value of depth, so that to avoid trouble with depth
    # sign convention.
    depth = abs(depth)

    # ==================================================================
    # 0) Define scaling values for the adimensional QG equation.
    # ==================================================================
    f_0 = 1e-04 # coriolis parameter (1/s)
    H = int(max(depth)) # vertical scale (m)
    L = 100e+03 # horizontal scale (m)

    # Delete NaN elements from N2 (and corresponding dept values).
    where_nan_N2 = np.where(np.isnan(N2))
    N2_nan_excl = np.delete(N2, where_nan_N2, None)
    depth_nan_excl = np.delete(depth, where_nan_N2, None)

    # ==================================================================
    # 1) Interpolate values on a new equally spaced depth grid.
    # ==================================================================
    # Create new equally spaced depth array (grid step = 1 m)
    n = H + 1
    z = np.linspace(0, H, n)

    # Create new (linearly) interpolated array for N2.
    f = interpolate.interp1d(depth_nan_excl, N2_nan_excl, 
                              fill_value = 'extrapolate', kind = 'linear')
    interp_N2 = f(z)

    # ==================================================================
    # 2) Scale N2 values (so that scipy algorithm works best) and 
    #    compute problem parameter 
    #       S = (N2 * H^2)/(f_0^2 * L^2)  .
    # ==================================================================
    # Scale N2 values (so that they go from 0 to 1), if N2 not constant.
    if max(interp_N2) != min(interp_N2): 
        interp_N2 = (interp_N2 - 
                      min(interp_N2))/(max(interp_N2)-min(interp_N2))
    # Compute S(z) parameter.
    S = (interp_N2 * (H**2))/((f_0**2) * (L**2))

    # ==================================================================
    # 3) Compute matrices of the eigenvalues/eigenvectors problem:
    #               A * v = (lambda * B) * v   .
    # ==================================================================
    # Store new scaled z grid step (z considered from 0 to 1).
    dz = (z[1] - z[0])/H
   
    # Create matrices (only null values).
    A = np.zeros([n,n]) # finite difference matrix.
    B = np.zeros([n,n]) # right side S-depending matrix.
 
    # Fill matrices with values (centered finite difference & S).
    for i in range(3, n-1):
          A[i-1,i-3] = -1/(12*dz**2)
          A[i-1,i-2] = 16/(12*dz**2)
          A[i-1,i-1] = -30/(12*dz**2)
          A[i-1,i] = 16/(12*dz**2)
          A[i-1,i+1] = -1/(12*dz**2)
          B[i-1,i-1] = - S[i-1] 

    # Set Boundary Conditions.
    A[1,0] = 1/(dz**2)
    A[1,1] = -2/(dz**2)
    A[1,2] = 1/(dz**2)
    A[n-2,n-1] = 1/(dz**2)
    A[n-2,n-2] = -2/(dz**2) 
    A[n-2,n-3] = 1/(dz**2)
    B[0,0] = - S[0] # first row
    B[1,1] = - S[1]
    B[n-1,n-1] = - S[n-1] # last row
    B[n-2,n-2] = - S[n-2] # last row
    
    # Delete BCs rows which may be not considered.
    A = np.delete(A, n-1, axis = 0)
    A = np.delete(A, 0, axis = 0)
    A = np.delete(A, n-1, axis = 1)
    A = np.delete(A, 0, axis = 1)
    B = np.delete(B, n-1, axis = 0)
    B = np.delete(B, 0, axis = 0)
    B = np.delete(B, n-1, axis = 1)
    B = np.delete(B, 0, axis = 1)
    
    # ==================================================================
    # Compute eigenvalues.
    # ==================================================================
    # Change sign to matrices (for consistency with scipy algorithm).
    A *= -1
    B *= -1
    # Compute smallest Eigenvalues.
    val = sp.sparse.linalg.eigsh(A, k = n_modes, M=B, sigma=0, which='LM', 
                                return_eigenvectors=False)
    
    # Take real part of eigenvalues and sort them in ascending order.
    eigenvalues = np.real(val)
    sort_index = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_index]
    
    # ==================================================================
    # 4) Compute eigenvectors through Numerov's Algortihm.
    # ==================================================================      
    # Store array for eigenvectors ( w(0) = w(n-1) = 0 for BCs).
    w = np.zeros([n,n_modes])
    
    # Define integration constant phi_0 = phi(z = 0) as BC.
    phi_0 = np.ones([n_modes])
   
    # Define constant k
    k = eigenvalues * ((dz**2) / 12)
    # First value computed from Phi_0 value
    w[1,:] = (- eigenvalues*phi_0)*dz/(1+S[1]*k*2)
    # Numerov's algorithm.
    for j in range(2,n-1):
        w[j,:] = ((2 - 10*k*S[j-1])/(1 + k*S[j])
                  ) * w[j-1,:] - ((1 + k*S[j-2])/(1 + k*S[j])) * w[j-2,:] 
        
    # ==================================================================
    # 5) Compute baroclinic Rossby radius and Phi.
    # ==================================================================
    # Store rossby radius array (only n° of modes desired).
    rossby_rad = np.zeros([n, n_modes])
    # Store function phi describing vertical modes of motion
    # (only n° of modes desired).
    phi = np.zeros([n, n_modes])

    # Fill rossby radius and phi.
    for i in range(n_modes):
        # Obtain Rossby radius from eigenvalues.
        rossby_rad[:,i] = 1/np.sqrt(eigenvalues[i])
        # Obtain Phi integrating eigenvectors * S.
        integral_argument = S * w[:,i]
        for j in range(n):
            phi[j,i] = integrate.trapezoid(integral_argument[:j], 
                                           dx=dz)                + phi_0[i]
      
    # Insert barotropic mode in the number of modes to be considered
    # (mode 0 = barotropic mode). 
    barotr_rossby_rad = np.nan
    barotr_phi = 1
    rossby_rad = np.insert(rossby_rad, 0, barotr_rossby_rad, axis=1)
    phi = np.insert(phi, 0, barotr_phi, axis=1)
    
    # Return newly interpolated rossby radius and phi.
    return rossby_rad, phi
