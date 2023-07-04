# -*- coding: utf-8 -*-
"""
Created on Thu May  5 17:49:21 2022

@author: Francesco Maria
"""
# ======================================================================
# THIS FILE INCLUDES PART OF THE TESTS FOR FUNCTIONS IMPLEMENTED
# IN COMPUTING THE BAROCLINIC ROSSBY RADIUS ...
# ======================================================================
import numpy as np
from hypothesis import given
import hypothesis.strategies as st

#=======================================================================
# Testing the functions for computing Potential Density & Brunt-Vaisala
# frequency. (See file 'eos.py')
#=======================================================================
import sys 
sys.path.append('..')
import OBM.eos as eos


#-----------------------------------------------------------------------
#                    Testing compute_BruntVaisala_freq_sq()
#-----------------------------------------------------------------------
from OBM.eos import compute_BruntVaisala_freq_sq as compute_BVsq
    

def test_compute_BV_const_dens():
    """
    Test if compute_BVsq() computes the BV freq. squared correctly 
    in a known constant case 
    rho(z)=const --> N^2 = 0 .
    """
    
    # Theoretical case.
    H = 1000
    depth = - np.linspace(0, H, 50)
    len_z = len(depth)
    rho_0 = 1025 #kg/m^3
    mean_density = np.full([len_z], rho_0)
    theor_BV2 = np.full([len_z], 0.0)
    # Numerical solution.
    out_BV2 = compute_BVsq(depth, mean_density)
    # Error
    dz = 1/50
    N2_0 = theor_BV2[0]
    error = (dz**2) * N2_0 #error related to finite differences (dz**2)
    assert np.allclose(out_BV2, theor_BV2, atol=error)


@given(a = st.floats(0, 55))
def test_compute_BV_linear_dens(a):
    """
    Test if compute_BVsq() computes the BV freq. squared correctly 
    in a known linear case
    rho(z) = a * z + rho_0, z < 0 --> N^2 = -g .
    """
    
    # Theoretical case.
    H = 1000
    depth = np.linspace(0, H, 50)
    len_z = len(depth)
    rho_0 = 1025 #(kg/m^3) ref. density
    a/=H
    mean_density = a*depth + rho_0
    g = 9.806 # (m/s^2)
    theor_BV2 = (g*a/rho_0)* np.ones(len_z)
    # Output product.
    out_BV2 = compute_BVsq(depth, mean_density)
    # Error
    dz = 1/50
    N2_0 = theor_BV2[0]
    error = (dz**2) * N2_0 #error related to finite differences (dz**2)
    assert np.allclose(out_BV2, theor_BV2, atol=error)
   

@given(a = st.floats(0, 0.05))
def test_compute_BV_expon_dens(a):
    """
    Test if compute_BVsq() computes the BV freq. squared correctly 
    in a known exponential case 
    rho(z) = rho_0 * exp(a*z), z < 0 --> N^2 = g*a*exp(a*z)
    """
    
    # Theoretical case.
    H = 1000
    depth = np.linspace(0, H, 50)
    rho_0 = 1025 #(kg/m^3) ref. density
    a /= H
    mean_density = rho_0*np.exp(a*depth) 
    g = 9.806 # (m/s^2)
    theor_BV2 = g*a*np.exp(a*depth)
    # Output product.
    out_BV2 = compute_BVsq(depth, mean_density)
    # Error (boundaries and interior)
    dz = 1/50
    N2_0 = theor_BV2[0]
    error_bnd = dz *  N2_0 # boundaries error: fwd/bwd fin. diff. O(dz)
    error = (dz**2) * N2_0 #error related to centered fin. diff. O(dz^2)
    # Boolean conditions
    surface = np.allclose(out_BV2[0], theor_BV2[0], atol= error_bnd)
    bottom = np.allclose(out_BV2[-1], theor_BV2[-1], atol= error_bnd)
    boundaries = np.logical_and(surface, bottom) 
    interior = np.allclose(out_BV2[1:-1], theor_BV2[1:-1], atol= error)
    
    assert np.logical_and(boundaries, interior)
  

def test_compute_BVsq_NaNs_behaviour():
    """
    Test if in compute_BVsq() behaviour is as expected when NaNs values. 
    """
    
    depth = np.arange(1, 7)
    mean_density = [1, np.nan, 3, 4, 5, np.nan]
    output_N2 = compute_BVsq(depth, mean_density)
    where_NaNs = np.where(np.isnan(output_N2))[0]
    expected_indeces = np.array([0,2,4,5])

    assert np.array_equal(where_NaNs, expected_indeces)
  

@given(arr_end = st.integers(3,100))
def test_compute_BVsq_when_lenDepth_is_greater_than_lenDens(arr_end):
    """
    Test if compute_BVsq() gives error when input depth has length
    greater than density one.
    """
    
    depth = np.arange(1, arr_end + 1)
    density = np.arange(1, arr_end)
    try:
        compute_BVsq(depth, density)
    except IndexError:
        assert True
    else:
        assert False
