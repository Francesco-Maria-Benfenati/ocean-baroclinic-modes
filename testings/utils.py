# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:36:30 2022

@author: Francesco Maria
"""

# ======================================================================
# This file contains UTILS for testings.
# ======================================================================
import numpy as np
from scipy import special 


def baroclModes_constN2(N2_0, H, n_modes):
    """
    Computes vertical structure func. (and barocl. rad) for N^2(z)=const.

    Arguments
    ---------
    N2_0 : int or float
        BV frequency squared, constant value 
    H : int
        mean depth of the region considered
    n_modes : int
        number of modes of motion considered

    Returns
    -------
    theor_Phi : numpy.ndarray
        Theoretical Vertical Structure Function for constant N^2(z)
    theor_R : numpy.ndarray
        Theoretical baroclinic deformation radius for const. N^2(z)
    """
    
    integers = np.arange(1, n_modes+1)# "n" integers
    new_z = np.linspace(0, H, H+1)# new equally spaced depth grid
    # Compute problem parameter S
    f_0 = 1e-04
    L = 100e+03
    S = (N2_0 * H**2)/(f_0**2 * L**2)
    # Eigenvalues
    eigenvals = (integers**2)*(np.pi**2)/S # See Pedlosky, GFD book.
    # Theoretical solution.
    theor_Phi = np.empty([H+1, n_modes])
    theor_R = L/np.sqrt(S*eigenvals)
    for i in range(n_modes):
        theor_Phi[:,i] = np.cos(integers[i] * np.pi * new_z/H)
  
    return theor_Phi, theor_R


def baroclModes_expN2(gamma, alpha, H):
    """
    Computes vertical structure function for exponential N^2(z).

    Arguments
    ---------
    gamma : list or numpy.ndarray
        gamma values
    alpha : int or float
        alpha coefficient
    H : int
        mean depth of the region considered

    Returns
    -------
    theor_Phi : numpy.ndarray
        Theoretical Vertical Structure Function for exponential N^2(z)
    
    # ---------------------------- NOTE --------------------------------
        N2 is exponential of type N2 = N0 * exp(alpha*z) ; z < 0  
        
        alpha = c/H, where c is a real or integer constant
        
        gamma_n = (N_0 * lambda_n) /(alpha * f_0), where lambda_n is 
        the eigenvalue corresponding to mode of motion "n".
        
        (see LaCasce, 2012). 
    # ------------------------------------------------------------------
    """
    
    A = 1 # amplitude
    n_modes = len(gamma) # n° modes of motion
    new_z = - np.linspace(0, H, H+1)# new equally spaced depth grid
    # Theoretical Vertical Structure Function Phi(z)
    theor_Phi = np.empty([H+1, n_modes])
    for i in range(n_modes):
        theor_Phi[:,i] = (A * np.exp(alpha*new_z/2)
                         * (special.yn(0 , 2*gamma[i])
                         *  special.jv(1, 2*gamma[i]*np.exp(alpha*new_z/2)) 
                         -  special.jv(0, 2*gamma[i]) 
                         *  special.yn(1, 2*gamma[i]*np.exp(alpha*new_z/2))
                            ) )
        theor_Phi[:,i] /= max(theor_Phi[:,i]) # Norm. theor sol.
    
    return theor_Phi
