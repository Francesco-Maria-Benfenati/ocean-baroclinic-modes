# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:06:25 2022

@author: Francesco M. Benfenati
"""
# ======================================================================
# This file includes the functions implemented for reading the 
# user's configuration file and extract data from NetCDF input file
# containing Potential Temperature & Salinity fields.
# ======================================================================       
import json
import numpy as np
import pandas as pd


def read_JSON_config_file( file_name,
                           section_key, title_key, items_key, 
                           name_key, type_key, value_key      ):
    """
    Reads JSON configuration file, returns configuration parameters. 

    Arguments
    ---------
    file_name : 'str'
        Name of (or path to) the JSON configuration file
    section_key, title_key, items_key, 
    name_key, type_key, value_key      : 'str'
        keys to dictionary values within JSON configuration file
        
        NOTE: config. file must have at least structure of type
        {'section_key': 
         [
          {'title_key': -section title-,
           'items_key': [
                     {'name_key': -item name-,
                      'type_key': -'float', 'string', 'int' or 'bool'-,
                      'value_key': -item value-}
                     ... 
                     ]
           }
           ...
          ]
         }
          
     Raises
     ------
     FileNotFoundError
         if JSON file name or path given is not found.
     JSONDecodeError
         if syntax is not correct in JSON file.
     KeyError
         if dictionary keys are not the expected ones.
         
    Returns
    -------
    config_param : 'dict'
        A dictionary containing the configuration parameters as found
        within JSON configuration file.
        Following the config. file, parameters are grouped together 
        in sections through subdictionaries. Within each subdictionary, 
        keys correspond to the names of items in each section.
        Values correspond to the value of each item within the section. 
    """
    
    in_file = open(file_name, 'r')
    # Load user's dictionary in JSON configuration file.
    user_dict = json.loads(in_file.read())
    
    # Print project title from JSON file to log file.
    project_title = user_dict['project_name']
    print('===========================================================')
    print(' * ', project_title, ' * ')
    print('===========================================================')
    
    # Create empty dictionary for configuration parameters.
    config_param = {}
    # Fill dictionary with keys and values from user's one.
    for section in user_dict[section_key]:  
        section_name = section[title_key]
        config_param[section_name] = {}
        for item in section[items_key]:  
            item_name = item[name_key]
            item_type = item[type_key]
            item_value = item[value_key]
            
            if item_type == ('float'): 
                config_param[section_name][item_name] = float(item_value)
            elif item_type == ('int'): 
                config_param[section_name][item_name] = int(item_value)
            elif item_type == ('bool'): 
                config_param[section_name][item_name] = bool(item_value)
            else: 
                config_param[section_name][item_name] = str(item_value)
    
    # Close input file.
    in_file.close()
    
    # Return dictionary containing configuration parameters.          
    return config_param


def find_time_step(time, user_time):
    """
    Find the index corresponding to the time step wanted by the user.

    Arguments
    ---------
    time : <class 'numpy.ndarray'>
        time array
    user_time : 'str'
        user time instant expressed as a string

    Raises
    ------
    ValueError
        if the wanted time step has not been found within the array.
       - or -
        if more than one index are found corresponding to the 
        wanted time date.

    Returns
    -------
    time_step : 'int'
        index corresponding to the time date looked for.
    """

    # Convert user time into pandas datetime.
    time_date = pd.to_datetime(user_time, errors='raise')
    # # Find index corresponding to user time date.
    time_index = np.where(time == time_date)[0]
    # Check if index has been found or if more than one has been found.
    if time_index.size == 0 : 
        raise ValueError(
            'Time step wanted has not been found within the dataset.')
    elif time_index.size > 1:
        raise ValueError(
             'There are more time steps with the same values:\
                 indeces array:', time_index)
    # Store index as int and return it.
    time_step = int(time_index)
    return time_step