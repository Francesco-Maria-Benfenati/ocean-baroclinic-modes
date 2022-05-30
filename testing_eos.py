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
import warnings
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


#-----------------------------------------------------------------------
#                    Testing compute_BruntVaisala_freq_sq()
#
# * NOTE * : 
#            'arr_end' is the variable defining the end value
#            of trial arrays. It starts from 3 in order to be 
#            sure the algorithm will work well (it works for 
#            arrays with length >= 2).
#            Through the next functions, the algorithm is 
#            tested when input arrays has length == 2, too.
#            arrays are = numpy.arange(1, arr_end) where
#            number 1 is set in order not to have problems 
#            dividing numbers by 0.
#-----------------------------------------------------------------------
from eos import compute_BruntVaisala_freq_sq as compute_BVsq


# Test if compute_BVsq() gives error when input length is < 0 and
# the algorithm can not be applied.
@given(arr_end = st.integers(3, 100))
def test_compute_BVsq_when_input_length_less_than_2(arr_end):
      depth = np.arange(arr_end - 1, arr_end)
      print(len(depth))
      density = np.arange(arr_end - 1, arr_end)
      trial_dim = ('depth')
      trial_dens = xarray.Variable(trial_dim, density)
      try:
          compute_BVsq(depth, trial_dens)
      except IndexError:
          assert True
      else:
          assert False
 
      
# Test if compute_BVsq() gives warning when depth values are zero.
@given(arr_end = st.integers(3, 100))
def test_compute_BVsq_when_null_depth_values(arr_end):
      depth = np.zeros(arr_end)
      density = np.arange(arr_end)
      trial_dim = ('depth')
      trial_dens = xarray.Variable(trial_dim, density)
      warnings.simplefilter('error', RuntimeWarning)
      try:
          compute_BVsq(depth, trial_dens)
      except Exception:
          assert True
      else:
          assert False
      warnings.simplefilter('default')    
      
      
# Test if compute_BVsq() behaves correctly when 1D input arrays.
@given(arr_end = st.integers(3,100))
def test_compute_BVsq_1D_input(arr_end):
    depth = np.arange(1, arr_end) 
    arr_1D =  np.arange(1, arr_end) 
    dims_1D = 'depth'
    trial_1D_array = xarray.Variable(dims_1D, arr_1D)
    try:
        compute_BVsq(depth, trial_1D_array)
    except Exception: 
        assert False
    else:
        assert True


# Test if compute_BVsq() behaves correctly when 2D input arrays.
@given(arr_end = st.integers(3,100))
def test_compute_BVsq_2D_input(arr_end):
    depth = np.arange(1, arr_end) 
    arr =  np.arange(1, arr_end) 
    len_lat = 5
    arr_2D = np.tile(arr, (len_lat,1))
    dims_2D = ('depth', 'lat')
    trial_2D_array = xarray.Variable(dims_2D, arr_2D)
    try:
        compute_BVsq(depth, trial_2D_array)
    except Exception: 
        assert False
    else:
        assert True
      
       
# Test if compute_BVsq() behaves correctly when 3D input arrays.
@given(arr_end = st.integers(3,100))
def test_compute_BVsq_3D_input(arr_end):
    depth = np.arange(1, arr_end) 
    arr =  np.arange(1, arr_end) 
    len_lat = 5
    len_lon = 3
    dims_3D = ('depth', 'lat', 'lon')
    arr_3D = np.empty((len(depth), len_lat, len_lon))
    for i in range(len(depth)):
        arr_3D[i,:,:] = np.tile(arr[i], (len_lat, len_lon)) 
    trial_3D_array = xarray.Variable(dims_3D, arr_3D)
    try:
        compute_BVsq(depth, trial_3D_array)
    except Exception: 
        assert False
    else:
        assert True


# Test if compute_BVsq() gives 'xarray.Variable' type as output.
@given(arr_end = st.integers(3,100))
def test_compute_BVsq_output_type(arr_end):
    depth = np.arange(1, arr_end) 
    arr = np.arange(1, arr_end)
    trial_dim = 'depth'
    trial_array = xarray.Variable(trial_dim, arr)
    print(trial_array.dims)
    N2 = compute_BVsq(depth, trial_array)[0]
    assert type(N2) == type(trial_array)


# Test if compute_BVsq() gives correct output dimensions.
@given(arr_end = st.integers(3,100))
def test_compute_BVsq_output_dims(arr_end):
     depth = np.arange(1, arr_end)
     len_lat = 5
     arr= np.tile(np.arange(1, arr_end), (len_lat,1))
     trial_dims = ('depth', 'lat')
     trial_array = xarray.Variable(trial_dims, arr)
     N2 = compute_BVsq(depth, trial_array)[0]
     assert N2.dims[:] == ('depth',)


# Test if compute_BVsq() gives correct output attributes.
@given(arr_end = st.integers(3,100))
def test_compute_BVsq_output_attrs(arr_end):
     depth = np.arange(1, arr_end)
     arr = np.arange(1, arr_end)
     trial_dim = 'depth'
     trial_array = xarray.Variable(trial_dim, arr)
     N2_attrs = {'long_name': 'Brunt-Vaisala frequency squared.', 
                  'standard_name': 'N_squared', 
                  'units': '1/s^2', 'unit_long':'cycles per second squared.'}
     N2 = compute_BVsq(depth, trial_array)[0]
     assert N2.attrs == N2_attrs


# Test if compute_BVsq() gives error when 'numpy.ndarray' is given
# as input type instead of 'xarray.Variable'.
@given(arr_end = st.integers(3,100))
def test_compute_BVsq_incorrect_input_type(arr_end):
     depth = np.arange(1, arr_end)
     ndarray = np.arange(1, arr_end)
     try:
         compute_BVsq(depth, ndarray)
     except AttributeError:
         assert True
     else: 
         assert False
    
   
# Test if compute_BVsq() gives error when input depth has length
# greater than density one.
@given(arr_end = st.integers(3,100))
def test_compute_BVsq_when_lenDepth_is_greater_than_lenDens(arr_end):
      depth = np.arange(1, arr_end + 1)
      density = np.arange(1, arr_end)
      trial_dim = ('depth')
      trial_dens = xarray.Variable(trial_dim, density)
      try:
          compute_BVsq(depth, trial_dens)
      except ValueError:
          assert True
      else:
          assert False


# Test if compute_BVsq() gives error when input density has length
# greater than depth one.
@given(arr_end = st.integers(3,100))
def test_compute_BVsq_when_lenDens_is_greater_than_lenDepth(arr_end):
      depth = np.arange(1, arr_end)
      density = np.arange(1, arr_end + 1)
      trial_dim = ('depth')
      trial_dens = xarray.Variable(trial_dim, density)
      try:
          compute_BVsq(depth, trial_dens)
      except ValueError:
          assert True
      else:
          assert False


# Test if compute_BVsq() gives error when input depth is not 1D.
@given(arr_end = st.integers(3,100))
def test_compute_BVsq_when_depth_not_1D(arr_end):
      depth = np.arange(1, arr_end)
      depth_2D = np.tile(depth, (4,1))
      density = np.arange(1, arr_end)
      trial_dim = ('depth')
      trial_dens = xarray.Variable(trial_dim, density)
      try:
          compute_BVsq(depth_2D, trial_dens)
      except ValueError:
          assert True
      else:
          assert False
          
          
# Test if compute_BVsq() gives err when input depth is not 'np.ndarray'.
def test_compute_BVsq_when_depth_not_ndarray():
      depth = [1, 2, 3, 4]
      density = np.arange(4)
      trial_dim = ('depth')
      trial_dens = xarray.Variable(trial_dim, density)
      try:
          compute_BVsq(depth, trial_dens)
      except AttributeError:
          assert True
      else:
          assert False
   
   
# Test if compute_BVsq() gives error when depth and dens are both empty.
def test_compute_BVsq_when_empty_depth():
      depth = np.array([])
      density = []
      trial_dim = ('depth')
      trial_dens = xarray.Variable(trial_dim, density)
      try:
          compute_BVsq(depth, trial_dens)
      except IndexError:
          assert True
      else:
          assert False  
    
      
# Test if in compute_BVsq() values which are NaNs becomes values due
# to interpolation.
def test_compute_BVsq_values_NOT_adjacent_to_NaNs():
      depth = np.arange(1, 7)
      density = [1, np.nan, 3, 4, 5, np.nan]
      trial_dim = ('depth')
      trial_dens = xarray.Variable(trial_dim, density)
      output_N2 = compute_BVsq(depth, trial_dens)[0]
      print(output_N2)
      where_NaNs = np.where(np.isnan(output_N2.values))[0]
      expected_indeces = np.array([])
      assert np.array_equal(where_NaNs, expected_indeces)
