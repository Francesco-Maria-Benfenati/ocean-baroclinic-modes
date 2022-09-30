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
#                   Testing _interp_N2()
# ----------------------------------------------------------------------


def test_interp_when_NaNs_as_input():
    """
    Test _interp_N2() gives correct output when all elements except two
    are NaN, using linear interpolation.
    """
    
    len_z = 50
    H = 1000
    z = np.linspace(1, H, len_z)
    N2 = np.full(len_z, np.nan)
    a = 3.5e-05
    c = 2.3e-05
    new_z = np.linspace(0, H, H+1)
    expected_N2 = a * new_z/H + c # linear BV freq. sq.
    N2[0] = expected_N2[1]
    N2[-1] = expected_N2[-1]
    out_N2 = modes._interpolate_N2(z, N2)
    
    assert np.allclose(out_N2, expected_N2, atol = 1e-08)
    
    
@given(a = st.floats(1, 5), c = st.floats(1,5))
def test_interp_when_different_grid_step(a, c):
    """
    Test _interp_N2() gives the same results for two different grid
    steps, whit dz_2 = 1/2 dz_1. 
    BV freq. is taken as exponential.
    """
    
    len_z_1 = 50
    len_z_2 = 100
    H = 1000
    z_1 = np.linspace(0, H, len_z_1+1)
    z_2 = np.linspace(0, H, len_z_2+1)
    
    N = (a * np.exp( - z_2/H ) + c) * 1e-03
    N2 = N**2
 
    out_N2_1 = modes._interpolate_N2(z_1, N2[0::2])
    out_N2_2 = modes._interpolate_N2(z_2, N2)
    
    assert np.allclose(out_N2_1, out_N2_2, atol = 1e-08)


def test_H_approximation_as_int():
    """
    Test if interpolate() gives the same output length, always
    approximating max depth down when converted to int.
    """
    
    len_z = 50
    H_1 = 5000.01
    H_2 = 5000.99
    z_1 = np.linspace(0, H_1, len_z)
    z_2 = np.linspace(0, H_2, len_z)
    N2 = np.full(len_z, 2.0)
    
    out_N2_1 = modes._interpolate_N2(z_1, N2)
    out_N2_2 = modes._interpolate_N2(z_2, N2)
    
    assert len(out_N2_1) == len(out_N2_2)
 

# ----------------------------------------------------------------------
#                   Testing _compute_matrix_A()
# ----------------------------------------------------------------------


@given(dz = st.floats(0.1,1))
def test_correct_matrix_A(dz):
    """
    Test if Matrix A is computed correctly for a general dz.
    """

    A = (1/(12*dz**2))* np.array([[-24,  12,  0,  0,  0,  0],
                                  [ 16, -30, 16, -1,  0,  0],
                                  [ -1,  16,-30, 16, -1,  0],
                                  [  0,  -1, 16,-30, 16, -1],
                                  [  0,   0, -1, 16,-30, 16],
                                  [  0,   0,  0,  0, 12,-24] ])
    computed_A = modes._compute_matrix_A(8, dz)
 
    assert np.allclose(A, computed_A)


# ----------------------------------------------------------------------
#                   Testing _compute_matrix_B()
# ----------------------------------------------------------------------


@given(n = st.integers(5, 100))
def test_correct_matrix_B(n):
    """
    Test if Matrix B is computed correctly for a generic S.
    """

    S = np.arange(n)
    B = np.diag(- S[1:-1])
    computed_B = modes._compute_matrix_B(n, S)

    assert np.allclose(B, computed_B)


# ----------------------------------------------------------------------
#                   Testing _compute_eigenvals()
# ----------------------------------------------------------------------


def test_compute_eigenvals_simple_problem():
    """
    Test if eigenvalues are computed correctly for a simple problem
    taken from a linear algebra book.
    """
    
    n_modes = 3
    A = np.array([[8.0,-18.0,9.0],[3.0,-7.0,3.0],[0.0,0.0,-1.0]])
    B = np.diag(np.ones(3))
    expected_eigenvals = np.array([-1, -1, 2])
    out_eigenvals = modes._compute_eigenvals(A, B, n_modes)
    
    assert np.allclose(out_eigenvals, expected_eigenvals)


@settings(deadline = None, max_examples = 50)
@given(n = st.integers(100, 1000))
def test_compute_eigenvals_stationary_wave(n):
    """
    Test if eigenvalues are computed correctly for a simple problem
    with well-know resolution: the stationary wave (non-dimensional).
    """
    
    n_modes = 5 # take first 5 eigenvalues (the most interesting ones).
    expected_integers = np.arange(1,n_modes+1) 
    dz = 1/n 
    A = modes._compute_matrix_A(n, dz)
    B = np.diag(- np.ones(n-2))
    
    out_eigenvals = modes._compute_eigenvals(A, B, n_modes)
    out_integers = (np.sqrt(out_eigenvals)/(np.pi))

    assert np.allclose(out_integers, expected_integers, atol=1e-01)


# ----------------------------------------------------------------------
#                   Testing _Numerov_Method()
# ----------------------------------------------------------------------


def test_Numerov_method_harm_oscillator():
    """
    Test if _Numerov_method() gives correct eigenvectors for the
    harmonic oscillator problem.
    """
    
    k = 6.25 #N/m
    m = 1 #kg
    omega = np.sqrt(k/m)
    A = 1
    n = 100
    L = 10
    x = np.linspace(0,L,n)
    dx = abs(x[1]-x[0])
    theor_sol = A*np.cos(omega * (x/L) * 2 * np.pi)
    dw_0 = 0
    f = -omega*np.ones(n)
    w_0 = 1 * A
    w_N = -1 * A
    num_sol = modes._Numerov_method(dx, f, dw_0, w_0, w_N)
   
    assert np.allclose(theor_sol, num_sol ,atol=1e-01)


def test_Numerov_method_stationary_wave():
    """
    Test if _Numerov_method() gives correct eigenvectors for the
    stationary wave problem, 3rd mode of motion "n=3" .
    """
    
    # Problem parameters
    L = 9.5
    N = 1000
    n = 3 # integer corresponding to mode of motion
    eigenvals = np.sqrt(n * np.pi / L)
    x = np.linspace(0,L,N)
    dx = abs(x[1]-x[0])
    # Theoretical solution
    theor_sol_3rdmode = np.sin(n *np.pi * x/L)
    # Numerical solution
    w_0 = 0
    w_N = 0
    dw_0 = n*np.pi/L 
    f_val = - (eigenvals**2)
    f = np.full(N, f_val)
    num_sol_3rdmode = modes._Numerov_method(dx, f, dw_0, w_0, w_N )
    
    assert np.allclose(theor_sol_3rdmode, num_sol_3rdmode, atol=1e-01) 


# ----------------------------------------------------------------------
#                   Testing compute_barocl_modes()
# ----------------------------------------------------------------------


@settings(deadline = None, max_examples = 50)
@given(n=st.integers(30,100), H = st.integers(1000, 3000))
def test_compute_modes_N2_const(n, H):
    """
    Test if compute_barocl_modes() gives correct output when N2 is
    constant (see Pedlosky, GFD book, 1987).
    """
    
    # Problem parameters.
    n_modes = 3
    N2_0 = 1.0
    # Theoretical Solution.
    theor_Phi, theor_R = utils.baroclModes_constN2(N2_0, H, n_modes)
    # Numerical solution. 
    z = np.linspace(0.5, 2*H, n)
    N2 = np.full(n, N2_0)
    num_R, num_Phi = modes.compute_barocl_modes(z, H, N2, n_modes)
    # Comparison.
    Phi_coherence = np.allclose(num_Phi[:,1:], theor_Phi, atol = 1e-02)
    R_coherence = np.allclose(num_R[1:], theor_R, atol = 1e-03)
    
    assert np.logical_and(Phi_coherence, R_coherence)


@settings(deadline = None, max_examples = 50)
@given(n=st.integers(30,100), H = st.integers(1000, 3000))
def test_compute_modes_N2_exp(n, H):
    """
    Test if compute_barocl_modes() computes correct modes of motion
    when N2 is exponential of type N2 = N0 * exp(alpha*z) 
    with alpha = 2/H, z < 0 ; (see LaCasce, 2012).
    """
    
    # Problem parameters.
    alpha = 2/H
    n_modes = 3
    z = - np.linspace(0.5, 2*H, n)
    N2_0 = 1.0
    N2 = N2_0 * np.exp(alpha*z)
    # Theoretical solution (see LaCasce, 2012).
    gamma = np.array([4.9107, 9.9072, 14.8875])/2 
    theor_Phi = utils.baroclModes_expN2(gamma, alpha, H)
    # Numerical solution.   
    num_R, num_Phi = modes.compute_barocl_modes(z, H, N2, n_modes)
    # Comparison.
    Phi_coherence = np.allclose(num_Phi[:,1:], theor_Phi, 
                                                      atol = 2e-02)
    
    assert Phi_coherence


@settings(deadline = None, max_examples = 50)
@given(n=st.integers(30,100), H = st.integers(1000, 3000))
def test_compute_modes_N2_exp_strong_stratification(n, H):
    """
    Test if compute_barocl_modes() computes correct modes of motion
    when N2 is exponential of type N2 = N0 * exp(alpha*z) ; z < 0  
    with alpha = 10/H, i.e. strong stratification (see LaCasce, 2012).
    """
    
    # Problem parameters.
    alpha = 10/H
    n_modes = 3
    z = - np.linspace(0.5, 2*H, n)
    N2_0 = 1.0
    N2 = N2_0 * np.exp(alpha*z)
    # Theoretical solution (see LaCasce, 2012).
    gamma = np.array([2.7565, 5.9590, 9.1492])/2 
    theor_Phi = utils.baroclModes_expN2(gamma, alpha, H)
    # Numerical solution.   
    num_R, num_Phi = modes.compute_barocl_modes(z, H, N2, n_modes)
    # Comparison.
    Phi_coherence = np.allclose(num_Phi[:,1:], theor_Phi, atol = 5e-02)

    assert Phi_coherence
  

def test_compute_modes_when_input_different_lengths():
    """
    Test compute_barocl_modes() gives ValueError when input arrays
    have different lengths.
    """
    
    n_modes = 3
    H = 5000
    z = np.linspace(0.5, H, 50)
    N2 = np.full(len(z) + 1, 2.0)
    try:
        modes.compute_barocl_modes(z, H, N2, n_modes)
    except ValueError:
        assert True
    else:
        assert False


def test_compute_modes_when_neg_z():
    """
    Test if compute_barocl_modes() works whell when depth is taken with
    negative sign convention.
    """
    
    n_modes = 3
    len_z = 50
    H = 5000
    z_neg = - np.linspace(0.5, H, len_z)
    z_pos = np.linspace(0.5, H, len_z)
    N2 = np.full(len_z, 2.0)
    R_pos, Phi_pos = modes.compute_barocl_modes(z_pos, H, N2, n_modes)
    R_neg, Phi_neg = modes.compute_barocl_modes(z_neg, H, N2, n_modes) 
    assert np.logical_and(np.allclose(R_pos, R_neg, rtol=1e-05, atol=1e-08, 
                                      equal_nan=True),
                          np.allclose(Phi_pos, Phi_neg, rtol=1e-05, atol=1e-08, 
                                      equal_nan=True))
