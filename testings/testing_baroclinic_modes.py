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
from  scipy.sparse.linalg import ArpackError

#=======================================================================
# Testing the functions for computing the baroclinic Rossby radius 
# vertical profile and the vertical modes of motion.
#=======================================================================

# ----------------------------------------------------------------------
#                   Testing compute_barocl_modes()
# ----------------------------------------------------------------------
from OBM.baroclinic_modes import compute_barocl_modes as modes


# Test compute_barocl_modes() works with N2(z)=const.
def test_compute_modes_when_N2_const():
    n_modes = 5
    z = np.linspace(0.5, 5000, 50)
    N2 = np.full(len(z), 2.0)
    try:
        modes(z, N2, n_modes)
    except ValueError:
        assert False
    else:
        assert True
        

# Test compute_barocl_modes() gives ValueError when input N2 array
# is empty.
def test_compute_modes_when_empty_input():
    n_modes = 5
    z = np.linspace(0.5, 5000, 50)
    N2 = []
    try:
        modes(z, N2, n_modes)
    except ValueError:
        assert True
    else:
        assert False

    
# Test compute_barocl_modes() gives error when all NaN values 
# are given as N2 input and interpolation can not be done.
def test_compute_modes_when_all_NaNs_as_input():
    n_modes = 5
    len_z = 50
    z = np.linspace(0.5, 5000, len_z)
    N2 = np.full(len_z, np.nan)
    try:
        modes(z, N2, n_modes)
    except ValueError:
        assert True
    else:
        assert False
    

# Test compute_barocl_modes() gives ArpackError when input N2 array
# is null.
def test_compute_modes_when_null_input():
    n_modes = 5
    len_z = 50
    z = np.linspace(0.5, 5000, len_z)
    N2 = np.full(len_z, 0)
    try:
        modes(z, N2, n_modes)
    except ArpackError:
        assert True
    else:
        assert False

        
# Test compute_barocl_modes() output arrays are of type 'numpy.ndarray'
# when the same type is given as input.
def test_compute_modes_out_type_when_input_ndarray():
    n_modes = 5
    z = np.linspace(0.5, 5000, 50)
    N2 = np.full(len(z), 2.0)
    R, Phi = modes(z, N2, n_modes)
    assert np.logical_and(type(R)==type(z), type(Phi)==type(z))


# Test compute_barocl_modes() gives ValueError when input arrays
# have different lengths.
def test_compute_modes_when_input_different_lengths():
    n_modes = 5
    z = np.linspace(0.5, 5000, 50)
    N2 = np.full(len(z) + 1, 2.0)
    try:
        modes(z, N2, n_modes)
    except ValueError:
        assert True
    else:
        assert False


# Test compute_barocl_modes() returns structure functions Phi(z) 
# included between -1 and 1 for constant N2.
def test_compute_modes_output_in_right_range_when_const_input():
    n_modes = 5
    len_z = 50
    z = np.linspace(0, 5000, len_z)
    N2 = np.full(len_z, 2.0)
    R, Phi = modes(z, N2, n_modes)
    for i in range(n_modes):
        max_val = max(Phi[:,i])
        min_val = min(Phi[:,i])
        assert np.logical_and(max_val <= 1.01, min_val >= - 1.01)


# Test compute_barocl_modes() returns output arrays of length 
# equal to max(depth) expressed in m.
def test_compute_modes_output_length():
    n_modes = 5
    len_z = 50
    H = 5e+03
    z = np.linspace(0, H, len_z)
    N2 = np.full(len_z, 2.0)
    R, Phi = modes(z, N2, n_modes)
    assert np.logical_and(len(R)==H+1, len(Phi)==H+1)  


# Test if compute_barocl_modes() works whell when depth is taken with
# negative sign convention.
def test_compute_modes_when_neg_z():
    n_modes = 5
    len_z = 50
    z_neg = - np.linspace(0.5, 5000, len_z)
    z_pos = np.linspace(0.5, 5000, len_z)
    N2 = np.full(len_z, 2.0)
    R_pos, Phi_pos = modes(z_pos, N2, n_modes)
    R_neg, Phi_neg = modes(z_neg, N2, n_modes) 
    assert np.logical_and(np.allclose(R_pos, R_neg, rtol=1e-05, atol=1e-08, 
                                      equal_nan=True),
                          np.allclose(Phi_pos, Phi_neg, rtol=1e-05, atol=1e-08, 
                                      equal_nan=True))


# Test if compute_barocl_modes() gives all positive Rossby radii values
# as output.
def test_compute_modes_positive_Rossby_Rad():
    n_modes = 5
    len_z = 50
    z = np.linspace(0.5, 5000, len_z)
    N2 = np.full(len_z, 2.0)
    R, Phi = modes(z, N2, n_modes)
    assert R.all()>0
    

# Test if compute_barocl_modes gives the same output length, always
# approximating max depth down when converted to int.
def test_compute_modes_H_approximation_as_int():
    n_modes = 5
    len_z = 50
    H_1 = 5000.01
    H_2 = 5000.99
    z_1 = np.linspace(0, H_1, len_z)
    z_2 = np.linspace(0, H_2, len_z)
    N2 = np.full(len_z, 2.0)
    R_1, Phi_1 = modes(z_1, N2, n_modes)
    R_2, Phi_2 = modes(z_2, N2, n_modes)
    assert np.logical_and(len(R_1)==len(R_2), len(Phi_1)==len(Phi_2))  
