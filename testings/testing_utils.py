# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:36:46 2022

@author: Francesco Maria
"""

# ======================================================================
# This file is for testing functions implemented ad utils in "utils.py"
# ======================================================================
import numpy as np
import utils


def test_constN2_case():
    """
    test utils. baroclModes_constN2() function, for N2(z) = const case.
    """
    
    # Parameters
    N2_0 = 1
    H = 1000
    n_modes = 3
    L = 100e+03 # [km]
    # Exact Sol.
    x = np.linspace(0,1,1001)
    Phi_1 = np.cos(1 * np.pi *x)
    Phi_2 = np.cos(2 * np.pi *x)
    Phi_3 = np.cos(3 * np.pi *x)
    R_1 = L/(np.pi*1)**2
    R_2 = L/(np.pi*2)**2
    R_3 = L/(np.pi*3)**2
    expected_Phi = np.stack((Phi_1, Phi_2, Phi_3), axis=1)
    expected_R = np.array([R_1, R_2, R_3])
    # UTILS sol.
    utils_Phi, utils_R = utils.baroclModes_constN2(N2_0, H, n_modes)
    # Comparison.
    Phi_coherence = np.allclose(utils_Phi, expected_Phi, rtol = 1e-03)
    R_coherence = np.allclose(utils_R, expected_R, rtol = 1e-03)
    
    assert np.logical_and(Phi_coherence, R_coherence)


def test_expN2_alphaNull():
    """
    test utils.baroclModes_expN2() function leads back to Const. case
    when alpha = 1.
    """
    
    # Parameters
    gamma = [0.6, 7, 8, 200]
    alpha = 0
    H = 1000
    # Expected Sol.
    expected_Phi = np.ones([H+1,4])
    # UTILS sol.
    utils_Phi = utils.baroclModes_expN2(gamma, alpha, H)
    
    assert np.allclose(utils_Phi, expected_Phi, rtol = 1e-03)
