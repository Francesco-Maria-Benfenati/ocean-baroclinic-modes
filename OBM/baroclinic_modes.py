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


def compute_barocl_modes(depth, mean_depth, mean_lat, N2, n_modes):
    """
    Computes baroclinic Rossby radius & vertical structure function.

    Arguments
    ---------
    depth : <class 'numpy.ndarray'>
            depth variable (1D)
    mean_depth : <class 'numpy.ndarray'> or 'float'
        mean depth of the considered region (m)
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
        baroclinic Rossby radius [m], for each mode of motion considered 
    phi : <class 'numpy.ndarray'>
        vertical structure function, for each mode of motion considered 
     
        ---------------------------------------------------------------
                            About the algorithm
        ---------------------------------------------------------------
    Coriolis parameter (used for scaling) and region mean depth are
    defined.
    
    1) N2 is linearly interpolated on a new equally spaced 
       nondimensional depth grid with grid step dz = 1 m.
    2) The problem parameter S is then computed as in 
       Grilli, Pinardi (1999)
    
            S = (N2)/(f^2)
            
       where f is the Coriolis parameter.
    3) The finite difference matrix 'A' and the S matrix 'B' are
       computed and the eigenvalues problem is solved:
           
           A * w = lambda * B * w  (lambda eigenvalues, w eigenvectors)
           
       with BCs: w = 0 at z=0,1.
     
    4) The eigenvectors are computed through numerical integration
       of the equation 
    
        (d^2/dz^2) * w = - lambda * S * w    (with BCs  w = 0 at z=0,1).
       
    5) The baroclinic Rossby radius is obtained as 
           
            R_n = 1/sqrt(lambda_n)
       
       for each mode of motion 'n'.
                             - & - 
       The vertical structure function Phi(z) is computed integrating
       S*w between 0 and z (for each mode of motion).
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
    # Define parameters in the QG equation.
    # ==================================================================
    earth_angvel = 7.29 * 1e-05 # earth angular velocity (1/s)
    # coriolis parameter (1/s)
    coriolis_param = 2 * earth_angvel * np.sin(mean_lat * np.pi /180) 
    
    # ==================================================================
    # 1) Interpolate values on a new equally spaced depth grid 'z' 
    #    (1 m grid step).
    # ==================================================================
    interp_N2 = _interpolate_N2(depth, N2)
    # Take only interp_N2 part from 0 to mean depth.
    mean_depth = int(mean_depth)
    n =  mean_depth + 1 # number of vertical levels
    interp_N2 = interp_N2[:n]
    
    # ==================================================================
    # 2) Compute problem parameter S:
    #       S = (N2 * H^2)/(f_0^2 * L^2)   .
    # ==================================================================
    
    # Compute S(z) parameter.
    S = interp_N2 / coriolis_param**2

    # ==================================================================
    # 3) Compute matrices of the eigenvalues/eigenvectors problem:
    #               A * v = (lambda * B) * v   .
    # ==================================================================
    # Store new z grid step (= 1m).
    dz = 1 # (m)
    
    A = _compute_matrix_A(n, dz)
    B = _compute_matrix_B(n, S)
    
    # ==================================================================
    # Compute eigenvalues.
    # ==================================================================
    eigenvalues = _compute_eigenvals(A, B, n_modes)

    # ==================================================================
    # 4) Compute eigenvectors through numerical integration.
    #     d^2/dz^2 w(z) = f(z) * w(z)
    # ================================================================== 
    # Define integration constant phi_0 = phi(z = 0) as BC.
    phi_0 = np.ones(n_modes)
    # Define f_n(z).
    f_n = np.empty([n, n_modes])
    for i in range(n_modes):
        f_n[:,i] = - eigenvalues[i] * S[:]
    # Define dw_0 = dw/dz at z=0.
    dw_0 = - eigenvalues*phi_0
    # Define BCs: w_0 = w(z=0) = 0 ; w_N = w(z=1) = 0.
    w_0 = 0
    w_N = 0
    
    # Compute eigenvectors through Numerov's Algortihm.
    w = np.empty([n, n_modes])
    for i in range(n_modes):
        w[:,i] = _Numerov_method(dz, f_n[:,i], dw_0[i], w_0, w_N)
        
    # ==================================================================
    # 5) Compute baroclinic Rossby radius and Phi.
    # ==================================================================
    # Store rossby radius array (only n° of modes desired).
    rossby_rad = np.empty([n_modes])
    # Store function phi describing vertical modes of motion.
    phi = np.empty([n, n_modes])

    # Fill rossby radius and phi.
    for i in range(n_modes):
        # Obtain Rossby radius from eigenvalues.
        rossby_rad[i] = 1/np.sqrt(eigenvalues[i])
        # Obtain Phi integrating eigenvectors * S.
        integral_argument = S * w[:,i]
        for j in range(n):
            phi[j,i] = integrate.trapezoid(integral_argument[:j], 
                                           dx=dz)                + phi_0[i]
      
    # Insert barotropic mode in the number of modes to be considered
    # (mode 0 = barotropic mode). 
    barotr_phi = 1
    phi = np.insert(phi, 0, barotr_phi, axis=1)
    barotr_rad = np.nan # For a barotropic ocean it would be sqrt(g*H)/f
    rossby_rad = np.insert(rossby_rad, 0, barotr_rad)
    
    # Return newly interpolated rossby radius and phi.
    return rossby_rad, phi


def _interpolate_N2(depth, N2):
    """
    Interpolate B-V freq. squared on a new equally spaced depth grid.

    Arguments
    ---------
    depth : <class 'numpy.ndarray'>
            depth variable (1D)
    N2 : <class 'numpy.ndarray'>
         Brunt-Vaisala frequency squared (along depth, 1D)

    Returns
    -------
    interp_N2 : <class 'numpy.ndarray'>
                interpolated Brunt-Vaisala frequency squared.   
    """
    
    # Delete NaN elements from N2 (and corresponding dept values).
    where_nan_N2 = np.where(np.isnan(N2))
    N2_nan_excl = np.delete(N2, where_nan_N2, None)
    depth_nan_excl = np.delete(depth, where_nan_N2, None)
    
    # Create new equally spaced depth array (grid step = 1 m)
    H = int(max(depth))
    n = H + 1 # number of vertical levels
    z = np.linspace(0, H, n)

    # Create new (linearly) interpolated array for N2.
    f = interpolate.interp1d(depth_nan_excl, N2_nan_excl, 
                              fill_value = 'extrapolate', kind = 'linear')
    interp_N2 = f(z)
    
    # Return grid and
    return interp_N2


def _compute_matrix_A(n, dz):
    """
    Compute L.H.S. matrix in the eigenvalues/eigenvectors problem.
    
    Arguments
    ---------
    n : 'int'
        number of vertical levels
    dz : 'float'
         vertical grid step
    
    Returns
    -------
    A : <class 'numpy.ndarray'>
        L.H.S. finite difference matrix
       
                     | 0      0    0     0   . . . . .  0 |
                     | 12    -24   12     0   . . . . . 0 |
                     | -1    16   -30    16  -1  0 . .  0 |
    A = (1/12dz^2) * | .     -1    16  -30   16  -1  .  0 |    
                     | .        .      .   .     .    .   |
                     | .       0   -1    16  -30   16  -1 | 
                     | .            0    0    12  -24  12 |
                     | 0      0    0     0   . . . . .  0 |
                   
    where dz is the scaled grid step (= 1m/H).
    Boundary Conditions are implemented setting the first and 
    last lines of A = 0.
    """
    
    # Create matrix (only null values).
    A = np.zeros([n,n]) # finite difference matrix.
 
    # Fill matrix with values (centered finite difference).
    for i in range(3, n-1):
          A[i-1,i-3] = -1/(12*dz**2)
          A[i-1,i-2] = 16/(12*dz**2)
          A[i-1,i-1] = -30/(12*dz**2)
          A[i-1,i] = 16/(12*dz**2)
          A[i-1,i+1] = -1/(12*dz**2)

    # Set Boundary Conditions.
    A[1,0] = 1/(dz**2)
    A[1,1] = -2/(dz**2)
    A[1,2] = 1/(dz**2)
    A[n-2,n-1] = 1/(dz**2)
    A[n-2,n-2] = -2/(dz**2) 
    A[n-2,n-3] = 1/(dz**2)
    
    # Delete BCs rows which may be not considered.
    A = np.delete(A, n-1, axis = 0)
    A = np.delete(A, 0, axis = 0)
    A = np.delete(A, n-1, axis = 1)
    A = np.delete(A, 0, axis = 1)
    
    return A


def _compute_matrix_B(n, S):
    """
    Comput R.H.S. matrix in the eigenvalues/eigenvectors problem.

    Arguments
    ---------
    n : 'int'
        number of vertical levels
    dz : 'float'
         vertical grid step
    S: <class 'numpy.ndarray'>
       problem parameter S = N2/f_0^2 
       
    Returns
    -------
    B : <class 'numpy.ndarray'>
        R.H.S. S-depending matrix
  
                     | -S_0    0    0     . . .   0 |
                     | 0     -S_1   0     0   . . 0 |
                     | 0      0   -S_2    0 . . . 0 |
    B =              | .      0     0     .       . |    
                     | .        .      .      .   . |
                     | .          0     0      -S_n |
                    
    where dz is the scaled grid step (= 1m/H).
    """
    
    # Create matrix (only null values).
    B = np.zeros([n,n]) # right side S-depending matrix.
 
    # Fill matrix with values (S).
    for i in range(3, n-1):
          B[i-1,i-1] = - S[i-1] 

    # Set Boundary Conditions.
    B[0,0] = - S[0] # first row
    B[1,1] = - S[1]
    B[n-1,n-1] = - S[n-1] # last row
    B[n-2,n-2] = - S[n-2] # last row
    
    # Delete BCs rows which may be not considered.
    B = np.delete(B, n-1, axis = 0)
    B = np.delete(B, 0, axis = 0)
    B = np.delete(B, n-1, axis = 1)
    B = np.delete(B, 0, axis = 1)

    return B


def _compute_eigenvals(A, B, n_modes):
    """
    Compute eigenvalues solving the eigenvalues/eigenvectors problem.
        
    Parameters
    ----------
    A : <class 'numpy.ndarray'>
        L.H.S. finite difference matrix
    B : <class 'numpy.ndarray'>
        R.H.S. S-depending matrix
    n_modes : 'int'
              number of modes of motion to be considered

    Returns
    -------
    eigenvalues : <class 'numpy.ndarray'>
                  problem eigenvalues 'lambda'
    
    The eigenvalues/eigenvectors problem is 
    
        A * w = lambda * B * w  (lambda eigenvalues, w eigenvectors)

        with BCs: w = 0 at z=0,1 .
        
    Here, a scipy algorithm is used.
    """
    
    # Change sign to matrices (for consistency with scipy algorithm).
    A *= -1
    B *= -1
    # Compute smallest Eigenvalues.
    val = sp.sparse.linalg.eigs(A, k = n_modes, M=B, sigma=0, which='LM', 
                                return_eigenvectors=False)
    
    # Take real part of eigenvalues and sort them in ascending order.
    eigenvalues = np.real(val)
    sort_index = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_index]
    
    return eigenvalues


def _Numerov_method(dz, f, dw_0, w_0, w_N):
    """
    Compute eigenvectors through Numerov's method.
    
    Solves the generic ODE:
       -------------------------------
      | d^2/dz^2 w(z) = f(z) * w(z)  | .
      --------------------------------

    Parameters
    ----------
    dz : 'float'
         vertical grid step
    f : <class 'numpy.ndarray'>
        problem parameter, depending on z
    dw_0 : 'float'
           first derivative of w: dw/dz computed at z = 0. 
    w_0, w_N : 'float'
               BCs, respectively at z = 0,1

    Returns
    -------
    w : <class 'numpy.ndarray'>
        problem eigenvectors
        
    The problem eigenvectors are computed through Numerov's numerical 
    method, integrating the equation 
    
        (d^2/dz^2) * w = - lambda * S * w    (with BCs  w = 0 at z=0,1).
        
    The first value is computed as
        w(0 + dz)= (- eigenvalues*phi_0)*dz/(1+lambda*S[1]*(dz**2)/6)
    where phi_0 = phi(z=0) set = 1 as BC.
    """
    
    # Number of vertical levels.
    n = len(f)
    # Store array for eigenvectors ( w(0) = w(n-1) = 0 for BCs).
    w = np.empty([n]) 
    w[0] = w_0
    w[n-1] = w_N
    # Define constant k
    k = (dz**2) / 12
    # First value computed from BCs and dw/dz|z=0 .
    w[1] = (w[0] + dz*dw_0 + (1/3)*(dz**2)*f[0]*w[0])/(1-k*2*f[1])
    # Numerov's algorithm.
    for j in range(2,n-1):  
        w[j] = (1/(1 - k*f[j])) * ((2 + 10*k*f[j-1])*w[j-1] 
                                    - (1 - k*f[j-2])* w[j-2])
    
    return w
