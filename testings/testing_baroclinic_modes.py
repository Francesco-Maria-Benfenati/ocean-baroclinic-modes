# -*- coding: utf-8 -*-
"""
Created on Sun May  8 18:55:44 2022

@author: Francesco Maria
"""
# ======================================================================
# THIS FILE INCLUDES PART OF THE TESTS FOR FUNCTIONS IMPLEMENTED
# IN COMPUTING THE BAROCLINIC ROSSBY RADIUS ...
# ======================================================================
import numpy as np

from hypothesis import given, settings
import hypothesis.strategies as st
import utils

#=======================================================================
# Testing the functions for computing the baroclinic Rossby radius 
# vertical profile and the vertical modes of motion.
#=======================================================================
import sys 
sys.path.append('..')
import OBM.baroclinic_modes as modes

# ----------------------------------------------------------------------
#                   Testing compute_barocl_modes()
# ----------------------------------------------------------------------


@settings(deadline = None, max_examples = 50)
@given(n=st.integers(50,100), H = st.integers(2000, 3000))
def test_compute_modes_N2_const(n, H):
    """
    Test if compute_barocl_modes() gives correct output when N2 is
    constant (see Pedlosky, GFD book, 1987).
    """
    
    # Problem parameters.
    mean_depth = int(H/2)
    n_modes = 3
    N2_0 = 1.0 * 1e-04
    # Theoretical Solution.
    theor_Phi, theor_R = utils.baroclModes_constN2(N2_0, mean_depth, n_modes)
    # Numerical solution. 
    z = np.linspace(0.5, 2*H, n)
    N2 = np.full(n, N2_0)
    num_R, num_Phi = modes.compute_barocl_modes(z, mean_depth, N2, n_modes)
    # Comparison.
    Phi_coherence = np.allclose(num_Phi[:,1:], theor_Phi, atol = 1e-02)
    R_coherence = np.allclose(num_R[1:], theor_R, atol = 1) # error = 1m
        
    assert np.logical_and(Phi_coherence, R_coherence)


@settings(deadline = None, max_examples = 50)
@given(n=st.integers(50,100), H = st.integers(2000, 5000))
def test_compute_modes_N2_exp(n, H):
    """
    Test if compute_barocl_modes() computes correct modes of motion
    when N2 is exponential of type N2 = N0 * exp(alpha*z) 
    with alpha = 2/H, z < 0 ; (see LaCasce, 2012).
    Here, H is the region mean_depth.
    """

    # Problem parameters.
    mean_depth = int(H/2)
    alpha = 2/mean_depth
    n_modes = 4
    z = - np.linspace(0.5, H, n)
    N2_0 = 1.0 
    N2 = N2_0 * np.exp(alpha*z)
    
    # Theoretical solution (see LaCasce, 2012).
    gamma = np.array([4.9107, 9.9072, 14.8875, 19.8628])/2 
    theor_Phi = utils.baroclModes_expN2(gamma, alpha, mean_depth)
    # Numerical solution.   
    num_R, num_Phi = modes.compute_barocl_modes(z, mean_depth, N2, n_modes)
    # Comparison.
    Phi_coherence = np.allclose(num_Phi[:,1:], theor_Phi, atol= 2e-02)

    assert Phi_coherence


@settings(deadline = None, max_examples = 50)
@given(n=st.integers(50,100), H = st.integers(2000, 5000))
def test_compute_modes_N2_exp_strong_stratification(n, H):
    """
    Test if compute_barocl_modes() computes correct modes of motion
    when N2 is exponential of type N2 = N0 * exp(alpha*z) ; z < 0  
    with alpha = 10/H, i.e. strong stratification (see LaCasce, 2012).
    """
    
    # Problem parameters.
    mean_depth = int(H/2)
    alpha = 10/mean_depth
    n_modes = 4
    z = - np.linspace(0.5, H, n)
    N2_0 = 1.0
    N2 = N2_0 * np.exp(alpha*z)
    # Theoretical solution (see LaCasce, 2012).
    gamma = np.array([2.7565, 5.9590, 9.1492, 12.334])/2 
    theor_Phi = utils.baroclModes_expN2(gamma, alpha, mean_depth)
    # Numerical solution.   
    num_R, num_Phi = modes.compute_barocl_modes(z, mean_depth, N2, n_modes)
    # Comparison.
    Phi_coherence = np.allclose(num_Phi[:,1:], theor_Phi, atol= 5.2e-02)
   
    assert Phi_coherence
