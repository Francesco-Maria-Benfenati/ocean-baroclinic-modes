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