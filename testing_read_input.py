# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:13:54 2022

@author: Francesco Maria
"""
# ======================================================================
# THIS FILE INCLUDES PART OF THE TESTS FOR FUNCTIONS IMPLEMENTED
# IN COMPUTING THE BAROCLINIC ROSSBY RADIUS ...
# ======================================================================
import xarray
import numpy as np
import pandas as pd

#=======================================================================
# Testing functions for reading input config and data files.
# (See file 'read_input.py')
#=======================================================================
import read_input as read

#-----------------------------------------------------------------------
#                   Testing read_JSON_config_file()
#-----------------------------------------------------------------------


# Test if output from read_JSON_config_file() is 'dict' type.
def test_JSON_output_type():
    file_name = 'config.json'
    output_dict = read.read_JSON_config_file(file_name,  
                                             section_key = 'sections',
                                             title_key = 'title',
                                             items_key = 'items',
                                             name_key = 'name',
                                             type_key = 'type', 
                                             value_key = 'value'      )
    trial_dict = {}
    assert type(output_dict) == type(trial_dict)


# Test if outputs 'lat_min', 'lat_max', 'lon_min', 'lon_max' from 
# read_JSON_config_file() are 'float' type.
def test_JSON_set_domain_with_float_values():
    file_name = 'config.json'
    output_dict = read.read_JSON_config_file(file_name,  
                                            section_key = 'sections',
                                            title_key = 'title',
                                            items_key = 'items',
                                            name_key = 'name',
                                            type_key = 'type', 
                                            value_key = 'value'      )

    extremants = [val for val in output_dict['set_domain'].values()]
    assert [type(extr) == float for extr in extremants]
   

# Test if read_JSON_config_file() gives error when values which may be
# floats are not of the right type.
def test_JSON_set_domain_with_nonfloat_values():
    file_name = 'config_modified_for_testing_DO-NOT-USE-IT.json'
    try:
        read.read_JSON_config_file(file_name,  
                                  section_key = 'MODIFIED_sections',
                                  title_key = 'title',
                                  items_key = 'items',
                                  name_key = 'name',
                                  type_key = 'type', 
                                  value_key = 'value'      )
    except ValueError:
        assert True
    else: 
        assert False


# Test if read_JSON_config_file() gives error when keys given do not 
# correspond to the one in JSON file.
def test_JSON_incoherent_keys():
    file_name = 'config_modified_for_testing_DO-NOT-USE-IT.json'
    try:
        read.read_JSON_config_file(file_name,  
                                  section_key = 'sections', # INCORRECT KEY
                                  title_key = 'title',
                                  items_key = 'items',
                                  name_key = 'name',
                                  type_key = 'type', 
                                  value_key = 'value'      )
    except KeyError:
        assert True
    else: 
        assert False
        
        
#-----------------------------------------------------------------------
#                       Testing find_time_step()
#-----------------------------------------------------------------------
# Test if function find_time_step() gives err when time array is empty.
def test_find_time_when_empty_array():
    time = pd.to_datetime([])
    user_time = '2021-01-05 12:00:00'
    try: 
        read.find_time_step(time, user_time)
    except ValueError:
        assert True
    else:
        assert False
        
        
# Test if function find_time_step() gives err when no value is found.
def test_find_time_no_value_found():
    time = np.array(['2025-12-31 12:00:00'], dtype='datetime64')
    user_time = '2021-01-05 12:00:00'
    try: 
        read.find_time_step(time, user_time)
    except ValueError:
        assert True
    else:
        assert False
    
 
# Test if function find_time_step() gives error when the same time
# value is found more than once.
def test_find_time_with_repeated_values():
    time = np.array(['2021-01-05 12:00:00', 
                     '2021-01-05 12:00:00' ], dtype='datetime64')
    user_time = '2021-01-05 12:00:00'
    try: 
        read.find_time_step(time, user_time)
    except ValueError:
        assert True
    else: assert False
    
    
#-----------------------------------------------------------------------
#                       Testing find_boundaries()
#-----------------------------------------------------------------------


# Test if find_boundaries() gives error when arrays are empty.
def test_find_boundaries_when_empty_array():
    lat = np.array([])
    lon = np.array([])
    set_domain = {'lat_min' : 30.0, 'lat_max' : 50.,
                  'lon_min' : -38,  'lon_max' : -18,
                  'lat_step': 0.5,  'lon_step': 0.5 }
    try:
        read.find_boundaries(lat, lon, set_domain)
    except ValueError:
        assert True
    else: assert False
    
    
# Test if find_boundaries() gives error when no value is found.
def test_find_boundaries_no_value_found():
    lat = np.array([20.0, 100.0])
    lon = np.array([-57.0, -122])
    set_domain = {'lat_min' : 30.0, 'lat_max' : 50.,
                  'lon_min' : -38,  'lon_max' : -18,
                  'lat_step': 0.5,  'lon_step': 0.5 }
    try:
        read.find_boundaries(lat, lon, set_domain)
    except ValueError:
        assert True
    else: assert False


# Test if find_boundaries() gives error if values are not 
# exactly the same.
def test_find_boundaries_precision():
    lat = np.array([30.1, 50.2], dtype = float)
    lon = np.array([-18.01, -38.005,], dtype = float)
    set_domain = {'lat_min' : 30.0, 'lat_max' : 50.,
                  'lon_min' : -38,  'lon_max' : -18,
                  'lat_step': 0.0,  'lon_step': 0.0 }
    try:
        read.find_boundaries(lat, lon, set_domain)
    except ValueError:
        assert True
    else: assert False
    
    
#-----------------------------------------------------------------------
#             Testing extract_data_from_NetCDF_input_file()
#-----------------------------------------------------------------------


# Test if output variables from extract_data_from_NetCDF_input_file()
# are of type 'xarray.core.variable.Variable'.
def test_NetCDF_output_type():
    config_dict = {'set_paths': 
                       {'input_file_name': 'azores_Jan2021.nc', 
'indata_path':
         '/mnt/d/Physics/SubMesoscale_Ocean/SOFTWARE_Rossby_Radius/Datasets/'},
                   'set_variables': 
                       {'temperature_name': 'thetao', 
                        'salinity_name': 'so',
                        'lat_var_name': 'latitude', 
                        'lon_var_name': 'longitude', 
                        'depth_var_name': 'depth', 
                        'time_var_name': 'time'}, 
                   'set_dimensions': 
                       {'lat_name': 'latitude', 
                        'lon_name': 'longitude', 
                        'depth_name': 'depth', 
                        'time_name': 'time'}, 
                   'set_domain': 
                       {'lat_min': 35.0, 
                        'lat_max': 40.0, 
                        'lon_min': -30.0, 
                        'lon_max': -25.0,
                        'lat_step': 0.0, 
                        'lon_step': 0.0}, 
                   'set_time': 
                       {'starting_time': '2021-01-01 12:00:00',
                        'ending_time': '2021-01-31 12:00:00'}                   
                   }
    output_vars = read.extract_data_from_NetCDF_input_file(config_dict)
    trial_data = np.arange(0,10)
    trial_dims = 'depth'
    trial_var = xarray.Variable(trial_dims, trial_data)
    assert( [type(output_vars[i]) == type(trial_var)
             for i in range(len(output_vars))])
    

# Test if output variables from extract_data_from_NetCDF_input_file()
# are of of correct dimensions.
def test_NetCDF_output_dims():
    config_dict = {'set_paths': 
                       {'input_file_name': 'azores_Jan2021.nc', 
'indata_path':
         '/mnt/d/Physics/SubMesoscale_Ocean/SOFTWARE_Rossby_Radius/Datasets/'},
                   'set_variables': 
                       {'temperature_name': 'thetao', 
                        'salinity_name': 'so',
                        'lat_var_name': 'latitude', 
                        'lon_var_name': 'longitude', 
                        'depth_var_name': 'depth', 
                        'time_var_name': 'time'}, 
                   'set_dimensions': 
                       {'lat_name': 'latitude', 
                        'lon_name': 'longitude', 
                        'depth_name': 'depth', 
                        'time_name': 'time'},
                   'set_domain': 
                       {'lat_min': 35.0, 
                        'lat_max': 40.0, 
                        'lon_min': -30.0, 
                        'lon_max': -25.0,
                        'lat_step': 0.0, 
                        'lon_step': 0.0}, 
                   'set_time': 
                       {'starting_time': '2021-01-01 12:00:00',
                        'ending_time': '2021-01-31 12:00:00'}                    
                   }
    output_vars = read.extract_data_from_NetCDF_input_file(config_dict)
    assert [var.dims == ('depth', 'latitude', 
                         'longitude'         ) for var in output_vars]
    
    
# Test if extract_data_from_NetCDF_input_file() gives error when keys
# within the function do not correspond to the ones in JSON file.
def test_NetCDF_incoherent_keys():
    config_dict = {'set_paths': 
                      {'input_file_name': 'azores_Jan2021.nc', 
'indata_path':
         '/mnt/d/Physics/SubMesoscale_Ocean/SOFTWARE_Rossby_Radius/Datasets/'},
                   'set_variables': 
                       {'MODIFIED_temperature_name': 'thetao', 
                        'salinity_name': 'so',
                        'lat_var_name': 'latitude', 
                        'lon_var_name': 'longitude', 
                        'depth_var_name': 'depth', 
                        'time_var_name': 'time'}, 
                   'set_dimensions': 
                       {'lat_name': 'latitude', 
                        'lon_name': 'longitude', 
                        'depth_name': 'depth', 
                        'time_name': 'time'},
                   'set_domain': 
                      {'lat_min': 35.0, 
                       'lat_max': 40.0, 
                       'lon_min': -30.0, 
                       'lon_max': -25.0,
                       'lat_step': 0.0, 
                       'lon_step': 0.0}, 
                   'set_time': 
                      {'starting_time': '2021-01-01 12:00:00',
                       'ending_time': '2021-01-31 12:00:00'}                      
                   }
    try:
        read.extract_data_from_NetCDF_input_file(config_dict)
    except KeyError:
        assert True
    else: 
        assert False

    
# Test if extract_data_from_NetCDF_input_file() gives error when 
# variables name within NetCDF file does not correspond to the ones 
# in JSON file.
def test_NetCDF_wrong_variables_name():
    config_dict = {'set_paths': 
                      {'input_file_name': 'azores_Jan2021.nc', 
'indata_path':
         '/mnt/d/Physics/SubMesoscale_Ocean/SOFTWARE_Rossby_Radius/Datasets/'},
                  'set_variables': 
                      {'temperature_name': 'WRONG_VARIABLE_NAME', 
                       'salinity_name': 'so',
                       'lat_var_name': 'latitude', 
                       'lon_var_name': 'longitude', 
                       'depth_var_name': 'depth', 
                       'time_var_name': 'time'}, 
                  'set_dimensions': 
                      {'lat_name': 'latitude', 
                       'lon_name': 'longitude', 
                       'depth_name': 'depth', 
                       'time_name': 'time'}, 
                  'set_domain': 
                      {'lat_min': 35.0, 
                       'lat_max': 40.0, 
                       'lon_min': -30.0, 
                       'lon_max': -25.0,
                       'lat_step': 0.0, 
                       'lon_step': 0.0}, 
                  'set_time': 
                      {'starting_time': '2021-01-01 12:00:00',
                       'ending_time': '2021-01-31 12:00:00'}                     
                  }
    try:
        read.extract_data_from_NetCDF_input_file(config_dict)
    except KeyError:
        assert True
    else: 
        assert False
        
        
# Test if extract_data_from_NetCDF_input_file() gives error when 
# dimensions name within NetCDF file does not correspond to the ones 
# in JSON file.
def test_NetCDF_wrong_depth_name():
    config_dict = {'set_paths': 
                      {'input_file_name': 'azores_Jan2021.nc', 
'indata_path':
         '/mnt/d/Physics/SubMesoscale_Ocean/SOFTWARE_Rossby_Radius/Datasets/'},
                   'set_variables': 
                       {'temperature_name': 'thetao', 
                        'salinity_name': 'so',
                        'lat_var_name': 'latitude', 
                        'lon_var_name': 'longitude', 
                        'depth_var_name': 'depth', 
                        'time_var_name': 'time'}, 
                   'set_dimensions': 
                       {'lat_name': 'latitude', 
                        'lon_name': 'longitude', 
                        'depth_name': 'WRONG_DIMENSION_NAME', 
                        'time_name': 'time'},
                   'set_domain': 
                      {'lat_min': 35.0, 
                       'lat_max': 40.0, 
                       'lon_min': -30.0, 
                       'lon_max': -25.0,
                       'lat_step': 0.0, 
                       'lon_step': 0.0}, 
                   'set_time': 
                      {'starting_time': '2021-01-01 12:00:00',
                       'ending_time': '2021-01-31 12:00:00'}                      
                   }
    try:
        read.extract_data_from_NetCDF_input_file(config_dict)
    except ValueError:
        assert True
    else: 
        assert False


# Test if extract_data_from_NetCDF_input_file() works well when starting
# and ending time are the same (i.e. one time step given).
def test_NetCDF_one_tstep_given():
    config_dict = {'set_paths': 
                      {'input_file_name': '20210131_AZORES.nc', 
'indata_path':
         '/mnt/d/Physics/SubMesoscale_Ocean/SOFTWARE_Rossby_Radius/Datasets/'},
                   'set_variables': 
                       {'temperature_name': 'thetao', 
                        'salinity_name': 'so',
                        'lat_var_name': 'latitude', 
                        'lon_var_name': 'longitude', 
                        'depth_var_name': 'depth', 
                        'time_var_name': 'time'}, 
                   'set_dimensions': 
                       {'lat_name': 'latitude', 
                        'lon_name': 'longitude', 
                        'depth_name': 'WRONG_DIMENSION_NAME', 
                        'time_name': 'time'},
                   'set_domain': 
                      {'lat_min': 35.0, 
                       'lat_max': 40.0, 
                       'lon_min': -30.0, 
                       'lon_max': -25.0,
                       'lat_step': 0.0, 
                       'lon_step': 0.0}, 
                   'set_time': 
                      {'starting_time': '2021-01-31 12:00:00',
                       'ending_time': '2021-01-31 12:00:00'}                      
                   }
    try:
        read.extract_data_from_NetCDF_input_file(config_dict)
    except Exception:
        assert True
    else: 
        assert False
        

# Test if extract_data_from_NetCDF_input_file() output type is correct
# when handling different datasets (one time step given).
def test_NetCDF_correct_output_type_when_different_dataset():
    config_dict = {'set_paths': 
                      {'input_file_name':'SURF_1h_20200702_20200706_grid_T.nc', 
'indata_path':
         '/mnt/d/Physics/SubMesoscale_Ocean/SOFTWARE_Rossby_Radius/Datasets/'},
                  'set_variables': 
                      {'temperature_name': 'votemper', 
                       'salinity_name': 'vosaline',
                       'lat_var_name': 'nav_lat', 
                       'lon_var_name': 'nav_lon', 
                       'depth_var_name': 'deptht', 
                       'time_var_name': 'time_counter'}, 
    # Input file imensions are (in order): 
    # x, y, deptht, time_counter, tbnds.
                  'set_dimensions': 
                      {'lat_name': 'y', 
                       'lon_name': 'x', 
                       'depth_name': 'deptht', 
                       'time_name': 'time_counter'}, 
                  'set_domain': 
                      {'lat_min': -26, 
                       'lat_max': -23.5, 
                       'lon_min': -45, 
                       'lon_max': -43.5,
                       'lat_step': 0.01,
                       'lon_step': 0.01},
                  'set_time': 
                      {'starting_time': '2020-07-02T18:30:00.000000000',
                       'ending_time': '2020-07-02T18:30:00.000000000'}                   
                  }
    output_vars = read.extract_data_from_NetCDF_input_file(config_dict)
    trial_data = np.arange(0,10)
    trial_dims = 'depth'
    trial_var = xarray.Variable(trial_dims, trial_data)
    assert( [type(var) == type(trial_var)
                    for var in output_vars])
    
    
# Test if dimensions of extract_data_from_NetCDF_input_file() outputs
# are correct when:
# 1) additional dimensions are given (more than time, depth, lat, lon)
# 2) input dimension are in a different order.
# 3) latitude and longitude are 2D arrays.
# 4) one time step given.
# This may happen with different input datasets.
# NOTE: 
#      Here more than one property is tested at the same time. 
#      This is related to the difficulties of handling different
#      datasets for testing the function correct behavior.      
def test_NetCDF_correct_output_dims_when_different_dataset():
    config_dict = {'set_paths': 
                      {'input_file_name':'SURF_1h_20200702_20200706_grid_T.nc', 
'indata_path':
         '/mnt/d/Physics/SubMesoscale_Ocean/SOFTWARE_Rossby_Radius/Datasets/'},
                  'set_variables': 
                      {'temperature_name': 'votemper', 
                       'salinity_name': 'vosaline',
                       'lat_var_name': 'nav_lat', 
                       'lon_var_name': 'nav_lon', 
                       'depth_var_name': 'deptht', 
                       'time_var_name': 'time_counter'}, 
    # Input file imensions are (in order): 
    # x, y, deptht, time_counter, tbnds.
                  'set_dimensions': 
                      {'lat_name': 'y', 
                       'lon_name': 'x', 
                       'depth_name': 'deptht', 
                       'time_name': 'time_counter'}, 
                  'set_domain': 
                      {'lat_min': -26, 
                       'lat_max': -23.5, 
                       'lon_min': -45, 
                       'lon_max': -43.5,
                       'lat_step': 0.01,
                       'lon_step': 0.01},
                  'set_time': 
                      {'starting_time': '2020-07-02T18:30:00.000000000',
                       'ending_time': '2020-07-02T18:30:00.000000000'}            
                  }
    output_vars = read.extract_data_from_NetCDF_input_file(config_dict)
    assert [var.dims == ('deptht', 'y', 'x') for var in output_vars]
    