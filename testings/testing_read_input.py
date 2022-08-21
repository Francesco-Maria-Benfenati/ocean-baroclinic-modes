# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:13:54 2022

@author: Francesco Maria
"""
# ======================================================================
# THIS FILE INCLUDES PART OF THE TESTS FOR FUNCTIONS IMPLEMENTED
# IN COMPUTING THE BAROCLINIC ROSSBY RADIUS ...
# ======================================================================
import numpy as np
import xarray

#=======================================================================
# Testing functions for reading input config and data files.
# (See file 'read_input.py')
#=======================================================================
import sys 
sys.path.append('..')

import OBM.read_input as read

#-----------------------------------------------------------------------
#                   Testing read_JSON_config_file()
#-----------------------------------------------------------------------


def test_read_JSON_output_dictionary():
    """
    Test if read_JSON_config_file() output is correct, as expected.
    *** NOTE: This test exploits the "test case" configuration file. ***
    """
    
    config_path = '../test_case/'
    config_file = config_path + 'config_test_case.json'
    # Store configuration parameters from JSON config file.
    config_parameters = read.read_JSON_config_file(config_file,
                                                     section_key = 'sections',
                                                     title_key = 'title',
                                                     items_key = 'items',
                                                     name_key = 'name',
                                                     type_key = 'type', 
                                                     value_key = 'value'     )
    expected_config_param = {
        'set_paths': 
            {'experiment_name': 'Azores_JAN21', 
             'region_name': 'Azores', 
             'input_file_name': 'azores_Jan2021.nc', 
             'bathy_file_name': 'mask_bathy_test_case_azores.nc', 
             'indata_path': 'test_case/dataset_azores/'}, 
        'set_variables': 
            {'temperature_name': 'thetao', 
             'salinity_name': 'so', 
             'lat_var_name': 'latitude', 
             'lon_var_name': 'longitude', 
             'depth_var_name': 'depth', 
             'time_var_name': 'time', 
             'bathy_var_name': 'deptho'}, 
        'set_dimensions': 
            {'lat_name': 'latitude', 
             'lon_name': 'longitude', 
             'depth_name': 'depth', 
             'time_name': 'time'}, 
        'set_domain': 
            {'lat_min': 33.0, 
             'lat_max': 35.0,
             'lat_step': 0.03,
             'lon_min': -30.0, 
             'lon_max': -28.0, 
             'lon_step': 0.03 }, 
        'set_time': 
            {'period_name': 'January 2021', 
             'starting_time': '2021-01-01 12:00:00', 
             'ending_time': '2021-01-31 12:00:00'}, 
        'set_modes':{'n_modes': 4}                                        }
        
    assert config_parameters == expected_config_param


#-----------------------------------------------------------------------
#                       Testing _find_time_step()
#-----------------------------------------------------------------------


def test_find_time_when_same_format():
    """ 
    Test if function _find_time_step() gives correct output index
    when the user time format is the same as the array one.
    """
    
    time = np.array(['2021-01-05 12:30:00', '2021-01-04 12:00:00', 
                     '2021-01-05 09:00:00', '2021-01-05 12:00:30',
                     '2021-01-05 12:00:00'], dtype='datetime64'   )
    user_time = '2021-01-05 12:00:00'

    time_index = read._find_time_step(time, user_time)
    expected_index = 4
    
    assert time_index == expected_index

    
def test_find_time_when_various_user_formats():
    """ 
    Test if function _find_time_step() gives correct output index
    with various time formats given by the user.
    """
    time = np.array(['2021-01-05 12:30:00', '2021-01-04 12:00:00', 
                     '2021-01-05 09:00:00', '2021-01-05 12:00:30',
                     '2021-01-05 12:00:00'], dtype='datetime64'   )
    user_time_1 = '2021 jan 05 12:30:00'
    user_time_2 = 'jan 04 2021 12:00:00'
    user_time_3 = '09:00 jan 5 21'
    user_time_4 = 'January 05 21, 12:00:30'
    user_time = [user_time_1, user_time_2, user_time_3, user_time_4]
    
    time_indeces = []
    for val in user_time:
        time_index = read._find_time_step(time, val)
        time_indeces.append(time_index)
   
    expected_indeces = [0, 1, 2, 3]
    assert time_indeces == expected_indeces


def test_find_time_no_value_found():
    """
    Test if function _find_time_step() gives err when no value is found.
    """
    
    time = np.array(['2025-12-31 12:00:00'], dtype='datetime64')
    user_time = '2021-01-05 12:00:00'
    try: 
        read._find_time_step(time, user_time)
    except ValueError:
        assert True
    else:
        assert False
    
 
def test_find_time_with_repeated_values():
    """
    Test if function _find_time_step() gives error when the same time
    value is found more than once.
    """
    
    time = np.array(['2021-01-05 12:00:00', 
                     '2021-01-05 12:00:00' ], dtype='datetime64')
    user_time = '2021-01-05 12:00:00'
    try: 
        read._find_time_step(time, user_time)
    except ValueError:
        assert True
    else: assert False
    
    
#-----------------------------------------------------------------------
#                       Testing _find_nearest()
#-----------------------------------------------------------------------   


def test_find_nearest_correct_output():
    """
    Test if function _find_nearest() gives correct output.
    """
    
    array = np.linspace(30.0, 50.0, 20)
    values = [38, 45, 42.53]
    
    out_idx = read._find_nearest(array, values)
    expected_indeces = [8, 14, 12]
    
    assert out_idx == expected_indeces


def test_find_nearest_when_range_exceeded():
    """
    Test if function _find_nearest() gives the extremant values when
    the input values exceed the array range.
    """
    
    array = np.linspace(30.0, 50.0, 20)
    exceeding_values = [20,80]
    
    out_idx = read._find_nearest(array, exceeding_values)
    extremants_indeces = [0, 19]
    
    assert out_idx == extremants_indeces
    

#-----------------------------------------------------------------------
#                       Testing _compute_mean_var()
#-----------------------------------------------------------------------


def test_compute_mean_var_correct_average():
    """
    Test if function _compute_mean_var() computes the mean correctly
    in the desired time range, when dims do not need transposition.
    """
    
    # Input Variable.
    dims = 't', 'z', 'y', 'x' # time, depth, lat, lon.
    data = np.arange(15,30)
    temp = np.empty([15, 10, 7, 5])    
    for i in range(len(data)):
        temp[i,:,:,:] = np.full([10,7,5], data[i])
    temperature = xarray.Variable(dims, temp)
    # Expected Output.
    mean_value = 20.0
    mean_temp = np.full([10,7,5], mean_value)
    expected_output = xarray.Variable(['z', 'y', 'x'], mean_temp)
    # Compute Output.
    t_start, t_end = 0, 10
    lat_min, lat_max = 0, 6
    lon_min, lon_max = 0, 5
    indeces = t_start, t_end, lat_min, lat_max, lon_min, lon_max
    out_var = read._compute_mean_var(temperature, dims, indeces)

    assert np.array_equal(out_var.values, expected_output.values)
  

def test_compute_mean_var_with_different_dims_order():
    """
    Test if function _compute_mean_var() gives correct output when
    dimensions are ordered differently.
    """
    
    # Input Variable.
    dims = 'y', 'z', 'x', 't' # lat, depth, lon, time.
    data = np.arange(15,26)
    temp = np.empty([5, 10, 7, 11]) 
    for i in range(len(data)):
        temp[:,:,:,i] = np.full([5,10,7], data[i])
    temperature = xarray.Variable(dims, temp)
    # Expected Output.
    mean_value = 20.0
    mean_temp = np.full([10,5,7], mean_value)
    expected_output = xarray.Variable(['z', 'y', 'x'], mean_temp)
    right_dims = 't', 'z', 'y', 'x' # time, depth, lat, lon.
    # Compute Output
    t_start, t_end = 0, 10
    lat_min, lat_max = 0, 10 # consider all values along lat.
    lon_min, lon_max = 0, 10 # consider all values along lon.
    indeces = t_start, t_end, lat_min, lat_max, lon_min, lon_max
    out_var = read._compute_mean_var(temperature, right_dims, indeces)

    assert np.array_equal(out_var.values, expected_output.values)
  

def test_compute_mean_var_correct_transposition():
    """
    Test if function _compute_mean_var() transposes dims correctly in
    the desired lat-lon range.
    """
    
    # Input Variable.
    dims = 'y', 'z', 'x', 't' # lat, depth, lon, time.
    data = np.arange(15,26)
    temp = np.empty([7, 10, 5, 11]) 
    for i in range(len(data)):
        temp[:,:,:,i] = np.full([7,10,5], data[i])
    # Expected Output.
    temperature = xarray.Variable(dims, temp)
    lat_min, lat_max = 0, 3
    lon_min, lon_max = 0, 2
    right_dims = 't', 'z', 'y', 'x' # time, depth, lat, lon.
    expected_sizes = {'z': 10, 'y': lat_max+1, 'x': lon_max+1}
    # Compute Output
    t_start, t_end = 0, 10

    indeces = t_start, t_end, lat_min, lat_max, lon_min, lon_max
    out_var = read._compute_mean_var(temperature, right_dims, indeces)
    
    assert out_var.sizes == expected_sizes


#-----------------------------------------------------------------------
#                       Testing _compute_mean_bathy()
#-----------------------------------------------------------------------


def test_compute_mean_bathy_correct_average():
    """
    Test if function _compute_mean_bathy() compute the mean correctly
    when dims do not need transposition.
    """
    
    # Input Variable.
    dims = ['y', 'x'] # lon, lat.
    bathy =  np.random.random((5,7))*1000
    bathymetry = xarray.Variable(dims, bathy)
    # Expected Output.
    lat_min, lat_max = 0, 10 # take all values
    lon_min, lon_max = 0, 10 # take all values
    expected_mean_value = int(np.mean(bathy, axis=None))
    # Compute Output.
    indeces = [lat_min, lat_max, lon_min, lon_max]
    out_mean_bathy = read._compute_mean_bathy(bathymetry, dims, indeces)
    
    assert out_mean_bathy == expected_mean_value
    
 
def test_compute_mean_bathy_inverted_dims():
    """
    Test if function _compute_mean_bathy() computes the mean correctly
    when dimensions are inverted.
    """
    
    # Input Variable.
    dims = ['x', 'y'] # lon, lat.
    bathy =  np.random.random((5,7))*1000
    bathymetry = xarray.Variable(dims, bathy)
    # Expected Output.
    right_dims = 'y', 'x' # lat, lon.
    expected_mean_value = int(np.mean(bathy, axis=None))
    # Compute Output.
    lat_min, lat_max = 0, 10 # take all values
    lon_min, lon_max = 0, 10 # take all values
    indeces = [lat_min, lat_max, lon_min, lon_max]
    out_mean_bathy = read._compute_mean_bathy(bathymetry, right_dims, indeces)
    
    assert out_mean_bathy == expected_mean_value

    
def test_compute_mean_bathy_correct_range():
    """
    Test if function _compute_mean_bathy() compute the mean correctly
    within the desired lat-lon range.
    """
    
    # Input Variable.
    dims = ['x', 'y'] # lon, lat.
    bathy =  np.random.random((5,7))*1000
    bathymetry = xarray.Variable(dims, bathy)
    # Expected Output.
    right_dims = 'y', 'x' # lat, lon.
    lat_min, lat_max = 0, 4 # take few values
    lon_min, lon_max = 0, 2 # take few values
    expected_mean_value = int(np.mean(bathy[0:3,0:5], axis=None))
    # Compute Output.
    indeces = [lat_min, lat_max, lon_min, lon_max]
    out_mean_bathy = read._compute_mean_bathy(bathymetry, right_dims, indeces)
    
    assert out_mean_bathy == expected_mean_value


#-----------------------------------------------------------------------
#             Testing extract_data_from_NetCDF_input_file()
#-----------------------------------------------------------------------


def test_extract_data_correct_output_temp():
    """
    Test if extract_data_from_NetCDF_input_file() gives correct output
    temperature, after having tested all subfunctions inside it.
    *** NOTE: This test exploits the "dataset_for_testing.nc" file. ***
    """
    
    # Output results.
    config_param = {
        'set_paths': 
            {'input_file_name': 'dataset_for_testing.nc', 
             'bathy_file_name': 'dataset_for_testing.nc', 
             'indata_path': './'}, 
        'set_variables': 
            {'temperature_name': 'theta', 
             'salinity_name': 'salinity', 
             'lat_var_name': 'latitude', 
             'lon_var_name': 'longitude', 
             'depth_var_name': 'depth', 
             'time_var_name': 'time', 
             'bathy_var_name': 'bathy'}, 
        'set_dimensions': 
            {'lat_name': 'latitude', 
             'lon_name': 'longitude', 
             'depth_name': 'depth', 
             'time_name': 'time'}, 
        'set_domain': 
            {'lat_min': 2, 
             'lat_max': 3, 
             'lon_min': 4, 
             'lon_max': 6}, 
        'set_time': 
            {'starting_time': '2021-01-05 12:00:00', 
             'ending_time': '2021-01-15 12:00:00'},                          }    
    [depth_xarray, mean_temperature, mean_salinity, 
     mean_bathy] = read.extract_data_from_NetCDF_input_file(config_param)
    # Expected value.
    expected_mean_temp = np.full([10,2,3], np.sum(np.arange(4,15))/11)
    
    assert np.array_equal(mean_temperature, expected_mean_temp)


def test_extract_data_correct_output_sal():
    """
    Test if extract_data_from_NetCDF_input_file() gives correct output
    salinity, after having tested all subfunctions inside it.
    *** NOTE: This test exploits the "dataset_for_testing.nc" file. ***
    """
    
    # Output results.
    config_param = {
        'set_paths': 
            {'input_file_name': 'dataset_for_testing.nc', 
             'bathy_file_name': 'dataset_for_testing.nc', 
             'indata_path': './'}, 
        'set_variables': 
            {'temperature_name': 'theta', 
             'salinity_name': 'salinity', 
             'lat_var_name': 'latitude', 
             'lon_var_name': 'longitude', 
             'depth_var_name': 'depth', 
             'time_var_name': 'time', 
             'bathy_var_name': 'bathy'}, 
        'set_dimensions': 
            {'lat_name': 'latitude', 
             'lon_name': 'longitude', 
             'depth_name': 'depth', 
             'time_name': 'time'}, 
        'set_domain': 
            {'lat_min': 2, 
             'lat_max': 3, 
             'lon_min': 4, 
             'lon_max': 6}, 
        'set_time': 
            {'starting_time': '2021-01-05 12:00:00', 
             'ending_time': '2021-01-15 12:00:00'},                          }    
    [depth_xarray, mean_temperature, mean_salinity, 
     mean_bathy] = read.extract_data_from_NetCDF_input_file(config_param)
    # Expected value.
    expected_mean_sal = np.full([10,2,3], np.sum(np.arange(1005,1016))/11)
    
    assert np.array_equal(mean_salinity, expected_mean_sal)
    

def test_extract_data_correct_output_bathy():
    """
    Test if extract_data_from_NetCDF_input_file() gives correct output
    mean bathymetry, after having tested all subfunctions inside it.
    *** NOTE: This test exploits the "dataset_for_testing.nc" file. ***
    """
    
    # Output results.
    config_param = {
        'set_paths': 
            {'input_file_name': 'dataset_for_testing.nc', 
             'bathy_file_name': 'dataset_for_testing.nc', 
             'indata_path': './'}, 
        'set_variables': 
            {'temperature_name': 'theta', 
             'salinity_name': 'salinity', 
             'lat_var_name': 'latitude', 
             'lon_var_name': 'longitude', 
             'depth_var_name': 'depth', 
             'time_var_name': 'time', 
             'bathy_var_name': 'bathy'}, 
        'set_dimensions': 
            {'lat_name': 'latitude', 
             'lon_name': 'longitude', 
             'depth_name': 'depth', 
             'time_name': 'time'}, 
        'set_domain': 
            {'lat_min': 2, 
             'lat_max': 3, 
             'lon_min': 4, 
             'lon_max': 6}, 
        'set_time': 
            {'starting_time': '2021-01-05 12:00:00', 
             'ending_time': '2021-01-15 12:00:00'},                          }    
    [depth_xarray, mean_temperature, mean_salinity, 
     mean_bathy] = read.extract_data_from_NetCDF_input_file(config_param)
    # Expected value.
    expected_mean_bathy = int((500+750)/2)
    
    assert np.array_equal(expected_mean_bathy, mean_bathy)
    
    
def test_extract_data_correct_output_depth():
    """
    Test if extract_data_from_NetCDF_input_file() gives correct output
    depth xarray, after having tested all subfunctions inside it.
    *** NOTE: This test exploits the "dataset_for_testing.nc" file. ***
    """
    
    # Output results.
    config_param = {
        'set_paths': 
            {'input_file_name': 'dataset_for_testing.nc', 
             'bathy_file_name': 'dataset_for_testing.nc', 
             'indata_path': './'}, 
        'set_variables': 
            {'temperature_name': 'theta', 
             'salinity_name': 'salinity', 
             'lat_var_name': 'latitude', 
             'lon_var_name': 'longitude', 
             'depth_var_name': 'depth', 
             'time_var_name': 'time', 
             'bathy_var_name': 'bathy'}, 
        'set_dimensions': 
            {'lat_name': 'latitude', 
             'lon_name': 'longitude', 
             'depth_name': 'depth', 
             'time_name': 'time'}, 
        'set_domain': 
            {'lat_min': 2, 
             'lat_max': 3, 
             'lon_min': 4, 
             'lon_max': 6}, 
        'set_time': 
            {'starting_time': '2021-01-05 12:00:00', 
             'ending_time': '2021-01-15 12:00:00'},                          }    
    [depth_xarray, mean_temperature, mean_salinity, 
     mean_bathy] = read.extract_data_from_NetCDF_input_file(config_param)
    # Expected value.
    expected_depth = np.empty([10,2,3])
    for i in range(10):
        expected_depth[i,:,:] = np.full([2,3], i)

    assert np.array_equal(depth_xarray.values, expected_depth)
