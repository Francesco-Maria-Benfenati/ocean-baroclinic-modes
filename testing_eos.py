# -*- coding: utf-8 -*-
"""
Created on Thu May  5 17:49:21 2022

@author: Francesco Maria
"""
# ======================================================================
# THIS FILE INCLUDES PART OF THE TESTS FOR FUNCTIONS IMPLEMENTED
# IN COMPUTING THE BAROCLINIC ROSSBY RADIUS ...
# ======================================================================
import xarray
import numpy as np

from hypothesis import given
import hypothesis.strategies as st

#=======================================================================
# Testing the functions for computing Potential Density & Brunt-Vaisala
# frequency. (See file 'eos.py')
#=======================================================================

#-----------------------------------------------------------------------
#                     Testing compute_density()
#
# NOTE:
#       'arr_end' is the variable defining the end value
#        of trial arrays.
#-----------------------------------------------------------------------
from eos import compute_density


# Test if compute_density() gives 'xarray.Variable' type as output when
# same type is given as input.
@given(arr_end = st.integers(0,100))
def test_compute_dens_output_type(arr_end):
    new_arr = np.arange(arr_end)
    trial_dims = 'depth'
    trial_array = xarray.Variable(trial_dims, new_arr)
    density = compute_density(trial_array, trial_array, trial_array)
    assert type(density) == type(trial_array)
  

# Test if compute_density() gives correct output dimensions.
@given(arr_end = st.integers(0,100))
def test_compute_dens_output_dims(arr_end):
     new_arr = np.arange(arr_end)
     trial_dims = ('lat', 'lon')
     trial_array = xarray.Variable(trial_dims, [new_arr, new_arr])
     density = compute_density(trial_array, trial_array, trial_array)
     assert density.dims == trial_dims


# Test if compute_density() gives correct output attributes.
@given(arr_end = st.integers(0,100))
def test_compute_dens_output_attrs(arr_end):
     new_arr = np.arange(arr_end)
     trial_dims = ('lat', 'lon')
     trial_array = xarray.Variable(trial_dims, [new_arr, new_arr])
     dens_attrs = {'long_name': 'Density', 
                  'standard_name': 'sea_water_potential_density', 
                  'units': 'kg/m^3', 'unit_long':'kilograms per meter cube'}
     density = compute_density(trial_array, trial_array, trial_array)
     assert density.attrs == dens_attrs
     
     
# Test if compute_density() gives error when 'numpy.ndarray' is given
# as input type instead of 'xarray.Variable'.
@given(arr_end = st.integers(0,100))
def test_compute_dens_incorrect_input_type(arr_end):
     ndarray = np.arange(arr_end)
     try:
         compute_density(ndarray, ndarray, ndarray)
     except AttributeError:
         assert True
     else: 
         assert False
    
    
# Test if compute_density() gives error when input arrays have 
# different lengths.
@given(arr_end = st.integers(0,100))
def test_compute_dens_when_different_input_lengths(arr_end):
     new_arr1 = np.arange(arr_end)
     new_arr2 = np.arange(arr_end + 1)
     trial_dims = ('depth')
     trial_array1 = xarray.Variable(trial_dims, new_arr1)
     trial_array2 = xarray.Variable(trial_dims, new_arr2)
     try:
         compute_density(trial_array1, trial_array2, trial_array2)
     except ValueError:
         assert True
     else:
         assert False


# Test if compute_density() gives error when input arrays have 
# different dimensions.
@given(arr_end = st.integers(0,100))
def test_compute_dens_when_different_input_dims(arr_end):
     new_arr = np.arange(arr_end)
     trial_dim1 = ('depth')
     trial_dim2 = ('lat', 'lon')
     trial_array1 = xarray.Variable(trial_dim1, new_arr)
     trial_array2 = xarray.Variable(trial_dim2, [new_arr, new_arr])
     try:
         compute_density(trial_array1, trial_array2, trial_array2)
     except ValueError:
         assert True
     else:
         assert False
